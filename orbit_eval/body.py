# orbit-eval — honest statistics for robot-policy evaluation.
# Copyright 2026 ORBIT Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which robot can physically do this: arithmetic with units, never a judgement.

Three questions get confused. Which robot can physically do a task is reach,
payload, degrees of freedom and price against a spec sheet, and that is
arithmetic. Which body anybody has got working on a job shape is the map, and
that is counting. What new body should be designed is morphology co-design, a
thirty-year field this package has no data in and does not touch.

So this module says "the SO-101 is rated 0.5 kg and you need 3 kg, short by
2.5", and never "this arm is not right for your task". Every number comes
from the robot database that ships with the package, and every field in that
database names the page it was read from.

The same arithmetic runs the other way for a downloaded skill. A frozen
manifest carries the joint ranges the checkpoint was trained on; `--skill`
compares them with the ranges in your own recording, joint by joint, before
anybody runs it. LeRobot normalises commands to each arm's own calibration,
so two SO-101s are not the same body until their ranges are shown to overlap.
"""

import json
import os

from . import dataset, robomap

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robots", "robots.json")

# (database field, requirement key, comparison, unit)
CHECKS = (
    ("reach_mm", "reach_mm", ">=", "mm"),
    ("payload_kg", "payload_kg", ">=", "kg"),
    ("dof", "dof", ">=", "dof"),
    ("price_usd", "price_usd", "<=", "USD"),
)
CLASSES = ("fixed_arm", "bimanual", "mobile_manipulator", "wheeled", "legged", "humanoid", "hand")

# A recording that reaches past the skill's trained range by more than this
# fraction of the skill's span is outside it on that joint.
OUTSIDE_FRAC = 0.05


def load_db(path=DB_FILE):
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return {"generated": None, "robots": []}
    if not isinstance(d, dict) or not isinstance(d.get("robots"), list):
        return {"generated": None, "robots": []}
    return d


def find(db, ident):
    ident = (ident or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    for r in db.get("robots", []):
        for key in (r.get("id"), r.get("name")):
            if key and key.lower().replace("-", "").replace("_", "").replace(" ", "") == ident:
                return r
    return None


# --------------------------------------------------------------------- fit

def fit(db, need):
    """Every robot against the requirement, with the margin per field.

    `need` is a dict with any of reach_mm, payload_kg, dof, price_usd and
    class. A robot passes when every stated requirement is met by a published
    figure, fails when a published figure falls short, and is unknown when a
    requirement cannot be checked because the figure was never published.
    """
    rows = []
    want_class = need.get("class")
    for r in db.get("robots", []):
        if want_class and r.get("class") != want_class:
            continue
        checks, failed, unknown = [], False, False
        for field, key, op, unit in CHECKS:
            want = need.get(key)
            if want is None:
                continue
            have = r.get(field)
            if have is None:
                checks.append({"field": field, "need": want, "have": None, "ok": None,
                               "margin": None, "unit": unit})
                unknown = True
                continue
            ok = (have >= want) if op == ">=" else (have <= want)
            margin = (have - want) if op == ">=" else (want - have)
            checks.append({"field": field, "need": want, "have": have, "ok": ok,
                           "margin": margin, "unit": unit})
            if not ok:
                failed = True
        verdict = "fails" if failed else ("unknown" if unknown else "passes")
        rows.append({"id": r.get("id"), "name": r.get("name"), "class": r.get("class"),
                     "verdict": verdict, "checks": checks,
                     "lerobot_driver": r.get("lerobot_driver"),
                     "mujoco_menagerie": r.get("mujoco_menagerie"),
                     "price_usd": r.get("price_usd")})
    order = {"passes": 0, "unknown": 1, "fails": 2}
    rows.sort(key=lambda x: (order[x["verdict"]], x.get("price_usd") or 1e12, x["id"] or ""))
    return rows


def format_fit(rows, need, m=None, job_shape=None):
    L = [""]
    L.append("  THE TASK NEEDS")
    bits = []
    for field, key, op, unit in CHECKS:
        if need.get(key) is not None:
            v = need[key]
            bits.append("%s %s %s" % (field.replace("_", " ").replace(" mm", "").replace(" kg", "")
                                       .replace(" usd", ""),
                                       "at least" if op == ">=" else "at most",
                                       _num(v, unit)))
    if need.get("class"):
        bits.append("class %s" % need["class"])
    if job_shape:
        bits.append("job shape %s" % job_shape)
    L.append("    " + ("; ".join(bits) if bits else "nothing was specified; every robot passes"))

    for verdict, title in (("passes", "PASSES"), ("unknown", "CANNOT BE CHECKED"),
                           ("fails", "FAILS")):
        group = [r for r in rows if r["verdict"] == verdict]
        if not group:
            continue
        L.append("")
        L.append("  %s  (%d)" % (title, len(group)))
        for r in group:
            why = []
            for c in r["checks"]:
                if c["ok"] is None:
                    why.append("%s not published" % c["field"].split("_")[0])
                elif verdict == "fails" and not c["ok"]:
                    why.append("%s %s, short by %s" % (
                        c["field"].split("_")[0], _num(c["have"], c["unit"]),
                        _num(abs(c["margin"]), c["unit"])))
                elif verdict == "passes":
                    why.append("%s %s" % (c["field"].split("_")[0], _num(c["have"], c["unit"])))
            extra = []
            if r.get("lerobot_driver"):
                extra.append("lerobot driver")
            if r.get("mujoco_menagerie"):
                extra.append("mujoco model")
            L.append("    %-14s %s%s" % ((r["name"] or r["id"])[:14], "; ".join(why),
                                          ("  [" + ", ".join(extra) + "]") if extra else ""))

    if m is not None and job_shape:
        L.append("")
        L.append("  MEASURED ON THIS JOB SHAPE  (%s)" % job_shape)
        any_cell = False
        for r in rows:
            if r["verdict"] == "fails":
                continue
            cells = robomap.cell(m, job_shape=job_shape, embodiment=r["id"])
            if cells:
                any_cell = True
                L.append("    %s" % (r["name"] or r["id"]))
                for line in robomap.format_cells(cells):
                    L.append("      " + line)
        if not any_cell:
            L.append("    nobody has measured this job shape on any body that passes. That is")
            L.append("    the honest answer and the reason to run the measurement.")
    L.append("")
    L.append("  Spec arithmetic, not a judgement: a robot that passes can still fail the task,")
    L.append("  and one that fails a figure may work with a different fixture. Sources per")
    L.append("  field: `orbit body <robot>`.")
    L.append("")
    return "\n".join(line.rstrip() for line in L)


# ------------------------------------------------------------------- sheet

def format_sheet(r, m=None):
    L = [""]
    L.append("  %s  (%s)" % (r.get("name") or r.get("id"), r.get("maker") or "maker not recorded"))
    src = r.get("source") or {}
    for field, label, unit in (("class", "class", ""), ("dof", "dof", ""), ("gripper", "gripper", ""),
                               ("reach_mm", "reach", "mm"), ("payload_kg", "payload", "kg"),
                               ("repeatability_mm", "repeatability", "mm"),
                               ("weight_kg", "weight", "kg"), ("price_usd", "price", "USD"),
                               ("lerobot_driver", "lerobot driver", ""),
                               ("mujoco_menagerie", "mujoco model", "")):
        v = r.get(field)
        shown = "not published" if v is None else (_num(v, unit) if unit else str(v))
        L.append("    %-14s %-24s %s" % (label, shown, src.get(field, "") if v is not None else ""))
    if r.get("notes"):
        L.append("    %-14s %s" % ("notes", r["notes"]))
    if m is not None:
        cells = robomap.cell(m, embodiment=r.get("id"))
        skills = robomap.skills_for(m, embodiment=r.get("id"))
        L.append("")
        L.append("  MEASURED ON THIS BODY")
        if cells:
            L.append("    %-11s %s" % ("job shape", ""))
            for c in cells:
                for line in robomap.format_cells([c]):
                    if line.startswith("status"):
                        continue
                    L.append("    %-11s %s" % (c["job_shape"], line))
        else:
            L.append("    nobody has measured anything on this body yet")
        if skills:
            L.append("")
            L.append("  PUBLIC SKILLS FOR THIS BODY  (%d, none with a measured number)" % len(skills)
                     if not any(s.get("measured") for s in skills) else
                     "  PUBLIC SKILLS FOR THIS BODY  (%d)" % len(skills))
            for s in skills[:12]:
                L.append("    %-10s %s" % (s.get("model") or "", s.get("url") or s.get("hub_id")))
            if len(skills) > 12:
                L.append("    ... and %d more" % (len(skills) - 12))
    L.append("")
    return "\n".join(line.rstrip() for line in L)


def format_list(db, m=None):
    L = [""]
    L.append("  %d robots, every figure from a page named in `orbit body <robot>`" % len(db.get("robots", [])))
    L.append("")
    L.append("    %-14s %-18s %-4s %-8s %-9s %-9s %-7s %s" % (
        "id", "class", "dof", "reach", "payload", "price", "driver", "on the map"))
    for r in db.get("robots", []):
        n_cells = len(robomap.cell(m, embodiment=r.get("id"))) if m is not None else 0
        n_skills = len(robomap.skills_for(m, embodiment=r.get("id"))) if m is not None else 0
        L.append("    %-14s %-18s %-4s %-8s %-9s %-9s %-7s %s" % (
            (r.get("id") or "")[:14], (r.get("class") or "")[:18],
            "" if r.get("dof") is None else r["dof"],
            "" if r.get("reach_mm") is None else "%d mm" % r["reach_mm"],
            "" if r.get("payload_kg") is None else "%g kg" % r["payload_kg"],
            "" if r.get("price_usd") is None else "$%s" % format(int(r["price_usd"]), ","),
            "yes" if r.get("lerobot_driver") else "",
            ("%s cell%s" % (n_cells, "" if n_cells == 1 else "s") if n_cells else "none") +
            (", %d public skill%s" % (n_skills, "" if n_skills == 1 else "s") if n_skills else "")))
    L.append("")
    return "\n".join(line.rstrip() for line in L)


# -------------------------------------------------------------------- skill

def skill_fit(manifest, meta):
    """Joint by joint, does this recording's range sit inside the skill's?"""
    d = (manifest or {}).get("dataset") or {}
    sk_names = d.get("joints") or []
    sk = (d.get("ranges") or {}).get(dataset.STATE) or {}
    block = meta.block(dataset.STATE) or {}
    my_lo, my_hi = dataset._vec(block.get("min")), dataset._vec(block.get("max"))
    my_names = meta.joint_names
    rows = []
    if not sk or my_lo is None or my_hi is None:
        return {"rows": rows, "comparable": False,
                "why": "the manifest or the recording carries no joint ranges"}
    for i, name in enumerate(my_names):
        if name in sk_names:
            j = sk_names.index(name)
        elif i < len(sk_names) and len(sk_names) == len(my_names):
            j = i
        else:
            rows.append({"joint": name, "verdict": "no match", "skill": None, "yours": None})
            continue
        s_lo, s_hi = sk["min"][j], sk["max"][j]
        y_lo, y_hi = my_lo[i], my_hi[i]
        span = max(s_hi - s_lo, 1e-9)
        low_over = max(0.0, s_lo - y_lo)
        high_over = max(0.0, y_hi - s_hi)
        outside = low_over > OUTSIDE_FRAC * span or high_over > OUTSIDE_FRAC * span
        rows.append({"joint": name, "skill": [s_lo, s_hi], "yours": [y_lo, y_hi],
                     "outside_low": low_over, "outside_high": high_over,
                     "verdict": "outside" if outside else "inside"})
    n_out = sum(1 for r in rows if r["verdict"] == "outside")
    n_miss = sum(1 for r in rows if r["verdict"] == "no match")
    return {"rows": rows, "comparable": True, "n_outside": n_out, "n_unmatched": n_miss,
            "robot_type": {"skill": d.get("robot_type"), "yours": meta.robot_type}}


def format_skill_fit(f, manifest_path):
    L = [""]
    L.append("  SKILL FIT  %s" % manifest_path)
    if not f.get("comparable"):
        L.append("    cannot compare: %s" % f.get("why"))
        L.append("")
        return "\n".join(L)
    rt = f.get("robot_type") or {}
    if rt.get("skill") or rt.get("yours"):
        L.append("    %-18s skill %s, yours %s" % ("robot type", rt.get("skill"), rt.get("yours")))
    L.append("    %-18s %-22s %-22s %s" % ("joint", "skill trained on", "your recording", ""))
    for r in f["rows"]:
        if r["verdict"] == "no match":
            L.append("    %-18s %-22s %-22s no joint of that name in the skill" % (r["joint"][:18], "", ""))
            continue
        s, y = r["skill"], r["yours"]
        note = "inside"
        if r["verdict"] == "outside":
            parts = []
            if r["outside_low"] > 0:
                parts.append("low end by %.0f" % r["outside_low"])
            if r["outside_high"] > 0:
                parts.append("high end by %.0f" % r["outside_high"])
            note = "OUTSIDE on the " + " and the ".join(parts)
        L.append("    %-18s %-22s %-22s %s" % (r["joint"][:18], "%.0f to %.0f" % tuple(s),
                                              "%.0f to %.0f" % tuple(y), note))
    L.append("")
    if f["n_outside"]:
        L.append("  %d joint%s in your recording reach%s past the range this skill was trained on."
                 % (f["n_outside"], "" if f["n_outside"] == 1 else "s",
                    "es" if f["n_outside"] == 1 else ""))
        L.append("  LeRobot normalises commands to each arm's own calibration, so the same")
        L.append("  number means a different position on your arm. Re-calibrate to match, or")
        L.append("  expect the skill to act outside what it ever saw on those joints.")
    else:
        L.append("  Every joint of your recording sits inside the range this skill was trained on.")
        L.append("  That is a calibration match, not a promise the skill works here.")
    L.append("")
    return "\n".join(line.rstrip() for line in L)


def _num(v, unit):
    if unit == "USD":
        return "$%s" % format(int(round(v)), ",")
    if unit == "dof":
        return "%g dof" % v
    if isinstance(v, float) and v != int(v):
        return "%g %s" % (v, unit)
    return "%d %s" % (int(v), unit)
