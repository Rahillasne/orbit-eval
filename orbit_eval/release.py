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

"""orbit-eval release — which of your skills did the new model break?

One command, one question. Point it at evaluation logs you have already run and
it answers, per named skill, whether the release you are about to ship (or just
shipped) is worse than the one it replaces, separating what is real from what is
inside your own measurement noise.

Why this and not the suite average, which is what every harness reports: on a
50-task suite, shipping the best-validation model damaged at least one task in
100% of measured draws while the suite mean got THREE TIMES steadier
(research/cleanrel50/). The certifying number stabilises exactly as it stops
certifying anything. A fleet average that moves +2 pp is arithmetically
consistent with one skill dropping twenty points, and one broken skill is what a
customer notices.

Two rules this module will not break, both bought with measured evidence:

  1. DAMAGE IS PRICED. A skill counts as regressed only when the drop clears BOTH
     an absolute floor and the measurement noise at that skill's episode count.
     The unpriced version is noise-saturated: on one banked battery the unpriced
     ">=5 pp worse" call fires on 81% of releases and the priced call on 2.5%
     (research/damage_jcurve/DAMAGEJ_RESULTS_2026-09-08.md, reproducing
     CLEANREL-50's RAW5 leg in a second policy class).
  2. NO SILENT MULTIPLICITY CORRECTION. Testing T skills at alpha means false
     flags; the report states the expected count rather than quietly shrinking
     the alpha, because on the damage objective a Holm correction measurably
     removed the very defections that were preventing damage.

Input is a CSV/TSV you already have — one row per episode:

    version,skill,episode,success
    v1.4,pick_from_bin,0,1
    ...

Column names are auto-detected (version|model|checkpoint|policy|run|release,
skill|task|sku|task_id, episode|ep|trial, success|ok|outcome|pass). Two versions
give a release check. More than two are read as release history, in file order
unless --order is given.
"""

import hashlib
import json
import math
import os
import time

from . import stats
from .route import _load_csv, _sha256

# A skill is REGRESSED when it drops by at least DROP_PP AND by at least Z_DMG
# standard errors. Both are inherited from research/cleanrel50/law_score.py,
# where the noise-only null of this exact rule was pre-checked at 0.024.
DROP_PP = 5.0
Z_DMG = 2.0
ALPHA = 0.05
# Below this many episodes a skill cannot be judged at all: the interval degenerates
# (a 1-episode bootstrap returns a zero-width 95% interval) and no drop is separable
# from noise. Such skills are reported as UNDERPOWERED rather than silently passed,
# because a false all-clear is the worst output this tool could produce. The field's
# own defaults sit near this line: LeRobot documents 10 episodes/task and published
# real-robot protocols run 10-25 per condition.
MIN_EPISODES = 10


def _rate(bits):
    return 100.0 * sum(1 for b in bits if b) / len(bits) if bits else float("nan")


def _se_two(p1, n1, p2, n2):
    """SE of a difference of two proportions, in pp."""
    p1, p2 = p1 / 100.0, p2 / 100.0
    return 100.0 * math.sqrt(max(p1 * (1 - p1), 1e-12) / max(n1, 1)
                             + max(p2 * (1 - p2), 1e-12) / max(n2, 1))


def _paired(a, b, ids_a=None, ids_b=None):
    """Discordant counts (n10, n01) if the two runs really are episode-aligned.

    Equal LENGTH is not alignment. Two versions evaluated on completely different
    initial states have equal length and nothing else, and treating them as matched
    pairs would understate every interval. When episode ids are available they must
    match exactly; without ids we refuse to pair.

    n10 = incumbent solved it and the candidate did not (a loss for the candidate);
    n01 = the candidate solved it and the incumbent did not.
    """
    if len(a) != len(b):
        return None
    if ids_a is None or ids_b is None or list(ids_a) != list(ids_b):
        return None
    _n11, n10, n01, _n00 = stats.discordant_counts([bool(x) for x in a],
                                                   [bool(x) for x in b])
    return n10, n01


def check(data, incumbent, candidate, drop_pp=DROP_PP, z_dmg=Z_DMG, alpha=ALPHA,
          paired=True, min_episodes=MIN_EPISODES, episode_ids=None):
    """Per-skill regression check of `candidate` against `incumbent`.

    `data` maps version -> skill -> [bool, ...]. Returns a dict with one row per
    skill, the suite delta, and the honest resolution statement.
    """
    skills = sorted(set(data[incumbent]) & set(data[candidate]))
    missing = sorted((set(data[incumbent]) | set(data[candidate])) - set(skills))
    rows = []
    for s in skills:
        A, B = data[incumbent][s], data[candidate][s]
        pa, pb = _rate(A), _rate(B)
        d = pb - pa
        ids = episode_ids or {}
        pair = (_paired(A, B, ids.get(incumbent, {}).get(s), ids.get(candidate, {}).get(s))
                if paired else None)
        if pair is not None:
            n10, n01 = pair
            # SE of the paired difference (n01 - n10)/n, in pp
            se = 100.0 * math.sqrt(max(n10 + n01, 1)) / len(A)
            p_one = stats.mcnemar_one_sided(n10, n01) if (n10 + n01) else 1.0
            # paired_boot_ci returns (first - second); we want candidate - incumbent
            lo, hi = stats.paired_boot_ci([bool(x) for x in B], [bool(x) for x in A])
            method = "paired"
        else:
            se = _se_two(pa, len(A), pb, len(B))
            z = d / se if se > 0 else 0.0
            p_one = stats.Phi(z)
            lo, hi = stats.newcombe_ci(sum(1 for x in B if x), len(B),
                                       sum(1 for x in A if x), len(A))
            method = "unpaired"
        # PRICED, and decided on the INTERVAL rather than a Wald z. A skill is
        # REGRESSED when we are 95% confident it lost at least drop_pp: the whole
        # interval sits at or below -drop_pp. Deciding on d >= z*SE instead looks
        # equivalent and is not: the Wald SE collapses to zero when either version
        # sits at exactly 0% or 100% (the most common state for a skill that works),
        # which flagged rows whose own printed interval contained zero. This is
        # deliberately stricter than research/cleanrel50/law_score.py's ">=5pp AND
        # >=2*SE", which was applied where both arms had equal n and no boundary
        # rates; the product cannot contradict the interval it prints.
        n_eff = min(len(A), len(B))
        underpowered = n_eff < min_episodes
        real = (not underpowered) and (hi <= -drop_pp)
        suspect = (not underpowered) and (-d >= drop_pp) and not real
        rows.append({
            "skill": s, "n_incumbent": len(A), "n_candidate": len(B),
            "incumbent_pct": pa, "candidate_pct": pb, "delta_pp": d,
            "ci95": [lo, hi], "se_pp": se, "p_one_sided": p_one,
            "method": method,
            # Improvement is priced exactly like damage: it is asymmetric and
            # self-serving to demand evidence for bad news and take good news on the
            # point estimate.
            "verdict": ("UNDERPOWERED" if underpowered else
                        "REGRESSED" if real else
                        "SUSPECT" if suspect else
                        "IMPROVED" if lo >= drop_pp else "held"),
            "resolvable_drop_pp": max(drop_pp, z_dmg * se),
            "underpowered": underpowered,
        })
    rows.sort(key=lambda r: r["delta_pp"])
    # Unweighted mean over skills, which is what every harness means by "suite
    # average" and, critically, uses ONE weight vector for both versions. Weighting
    # each side by its own episode counts makes the headline a pure allocation
    # artifact: with unequal episode counts it can report +47 pp while every single
    # skill fell 10 pp. With a common weight, suite_delta is exactly the mean of the
    # per-skill deltas, so the headline and the table can never disagree in sign.
    suite_a = (sum(r["incumbent_pct"] for r in rows) / len(rows)) if rows else float("nan")
    suite_b = (sum(r["candidate_pct"] for r in rows) / len(rows)) if rows else float("nan")
    return {
        "incumbent": incumbent, "candidate": candidate,
        "n_skills": len(rows), "skills_missing_from_one_side": missing,
        "skills_dropped": sorted(set(data[incumbent]) - set(data[candidate])),
        "skills_added": sorted(set(data[candidate]) - set(data[incumbent])),
        "rows": rows,
        "n_regressed": sum(1 for r in rows if r["verdict"] == "REGRESSED"),
        "n_suspect": sum(1 for r in rows if r["verdict"] == "SUSPECT"),
        "n_improved": sum(1 for r in rows if r["verdict"] == "IMPROVED"),
        "n_underpowered": sum(1 for r in rows if r["verdict"] == "UNDERPOWERED"),
        "all_underpowered": bool(rows) and all(r["verdict"] == "UNDERPOWERED" for r in rows),
        "min_episodes": min_episodes,
        "suite_incumbent_pct": suite_a, "suite_candidate_pct": suite_b,
        "suite_delta_pp": suite_b - suite_a,
        "expected_false_flags": alpha * len(rows),
        "paired": all(r["method"] == "paired" for r in rows) if rows else False,
        "median_resolvable_drop_pp": (sorted(r["resolvable_drop_pp"] for r in rows)
                                      [len(rows) // 2] if rows else float("nan")),
    }


def history(data, order, **kw):
    """Across consecutive releases: how often did at least one skill regress?"""
    steps = []
    for a, b in zip(order, order[1:]):
        c = check(data, a, b, **kw)
        steps.append({"from": a, "to": b, "n_skills": c["n_skills"],
                      "n_regressed": c["n_regressed"],
                      "n_suspect": c["n_suspect"], "suite_delta_pp": c["suite_delta_pp"],
                      "worst_skill": c["rows"][0]["skill"] if c["rows"] else None,
                      "worst_delta_pp": c["rows"][0]["delta_pp"] if c["rows"] else 0.0,
                      "regressed_skills": [r["skill"] for r in c["rows"]
                                           if r["verdict"] == "REGRESSED"]})
    hit = sum(1 for s in steps if s["n_regressed"] >= 1)
    return {"releases": len(steps), "with_a_regression": hit,
            "rate": hit / len(steps) if steps else float("nan"), "steps": steps}


def _name(s, w=24):
    return s if len(s) <= w else s[:w - 1] + "\u2026"


def _bar(pct, width=18):
    k = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "#" * k + "." * (width - k)


def format_check(c, meta=None):
    L = []
    nr, ns, nu = c["n_regressed"], c["n_suspect"], c["n_underpowered"]
    if c["all_underpowered"]:
        head = "This file cannot answer the question"
    elif nr:
        head = ("%d of your %d skills regressed" % (nr, c["n_skills"])
                if c["n_skills"] != 1 else "Your one skill regressed")
    elif nu:
        # Never an all-clear while a skill went unjudged: one of them can have
        # collapsed to zero and this file cannot see it. Exit code follows (3).
        judged = c["n_skills"] - nu
        head = ("No regression among the %d skill%s this file could judge"
                % (judged, "" if judged == 1 else "s"))
    else:
        head = "No skill regressed beyond your measurement noise"
    L.append("=" * 74)
    L.append("  %s" % head.upper())
    L.append("  %s  ->  %s" % (c["incumbent"], c["candidate"]))
    L.append("=" * 74)
    L.append("")
    if c["all_underpowered"]:
        L.append("  %s fewer than %d episodes on at least one"
                 % (("Your one skill has" if c["n_skills"] == 1 else
                     "Every one of your %d skills has" % c["n_skills"]), c["min_episodes"]))
        L.append("  side, so NOTHING here is a verdict. This is not an all-clear. A skill")
        L.append("  could have gone to zero and this file could not tell you.")
        L.append("")
        L.append("  What it would take, per skill, to detect a drop of:")
        for tgt in (10.0, 20.0, 30.0):
            n_need = int(math.ceil(2 * 0.25 * (100.0 * Z_DMG / tgt) ** 2))
            L.append("      %4.0f pp   about %4d episodes per skill per version" % (tgt, n_need))
        L.append("  (at a 50% success rate, the worst case; fewer if your rates are extreme)")
        L.append("")
    elif nu:
        L.append("  %d of your %d skills have fewer than %d episodes and were NOT judged;"
                 % (nu, c["n_skills"], c["min_episodes"]))
        L.append("  they are listed as UNDERPOWERED below, not as passing. This is not an")
        L.append("  all-clear: any one of them could have collapsed to zero without this")
        L.append("  file being able to tell you, so this run exits 3 rather than 0.")
        L.append("")
    if nr:
        L.append("  These are real: the whole 95%% interval sits at or below -%.0f pp, so at"
                 % DROP_PP)
        L.append("  your episode count the drop is not explainable by luck. Investigate")
        L.append("  before this ships.")
        L.append("")
    for r in c["rows"]:
        if r["verdict"] not in ("REGRESSED", "SUSPECT", "UNDERPOWERED"):
            continue
        tag = {"REGRESSED": "BROKE  ", "SUSPECT": "suspect",
               "UNDERPOWERED": "no data"}[r["verdict"]]
        if r["verdict"] == "UNDERPOWERED":
            L.append("  %s  %-24s %5.1f%% -> %5.1f%%  %+6.1f pp  interval not reportable  n=%d"
                     % (tag, _name(r["skill"]), r["incumbent_pct"], r["candidate_pct"],
                        r["delta_pp"], r["n_candidate"]))
        else:
            nlab = ("n=%d" % r["n_candidate"] if r["n_incumbent"] == r["n_candidate"]
                    else "n=%d vs %d" % (r["n_incumbent"], r["n_candidate"]))
            L.append("  %s  %-24s %5.1f%% -> %5.1f%%  %+6.1f pp  95%% [%+.1f, %+.1f]  %s"
                     % (tag, _name(r["skill"]), r["incumbent_pct"], r["candidate_pct"],
                        r["delta_pp"], r["ci95"][0], r["ci95"][1], nlab))
        if r["verdict"] == "SUSPECT":
            L.append("           %s at n=%d this size of drop is not distinguishable from noise"
                     % (" " * 24, r["n_candidate"]))
        elif r["verdict"] == "UNDERPOWERED":
            L.append("           %s only %d episode(s) — not judged, and not a pass"
                     % (" " * 24, r["n_candidate"]))
    if c["skills_dropped"]:
        L.append("")
        L.append("  GONE FROM THE NEW MODEL — %d skill(s) the incumbent had and the candidate"
                 % len(c["skills_dropped"]))
        L.append("  does not: %s" % ", ".join(c["skills_dropped"][:8]))
        L.append("  These are not compared, and removing a hard skill RAISES the average of")
        L.append("  what is left. If that was not deliberate, it is the first thing to check.")
    if c["skills_added"]:
        L.append("")
        L.append("  NEW IN THE CANDIDATE, no baseline to compare against: %s"
                 % ", ".join(c["skills_added"][:8]))
    L.append("")
    L.append("  SUITE AVERAGE  %5.1f%% -> %5.1f%%   %+.2f pp"
             % (c["suite_incumbent_pct"], c["suite_candidate_pct"], c["suite_delta_pp"]))
    if nr:
        worst = c["rows"][0]
        if abs(c["suite_delta_pp"]) < abs(worst["delta_pp"]) / 3.0:
            L.append("  The average moved %.2f pp while %s lost %.1f. That is the whole"
                     % (c["suite_delta_pp"], worst["skill"], -worst["delta_pp"]))
            L.append("  problem: an average over %d skills absorbs one collapse, and it gets"
                     % c["n_skills"])
            L.append("  STEADIER as you add skills, not more sensitive. It is the number")
            L.append("  release decisions are usually made on.")
    L.append("")
    L.append("  Per skill:")
    for r in c["rows"]:
        L.append("    %-24s %s %5.1f%% %+6.1f pp   %s"
                 % (_name(r["skill"]), _bar(r["candidate_pct"]), r["candidate_pct"],
                    r["delta_pp"], r["verdict"]))
    L.append("")
    L.append("  WHAT THIS RUN COULD NOT SEE")
    L.append("    A drop is called real only when the whole 95%% interval clears -%.0f pp,"
             % DROP_PP)
    L.append("    so the smallest drop this file can call real is about %.1f pp (median"
             % c["median_resolvable_drop_pp"])
    L.append("    across skills). Anything smaller is invisible here, however real it is.")
    L.append("    %s"
             % ("Episodes are aligned, so skills are compared as matched pairs."
                if c["paired"] else
                "Episodes are NOT aligned between versions, so this is an unpaired"))
    if not c["paired"]:
        L.append("    comparison and it is less sensitive than it could be. Evaluating both")
        L.append("    versions on the same initial states would tighten every interval here.")
    L.append("    Across %d skill%s, each tested independently, some flags are expected"
             % (c["n_skills"], "" if c["n_skills"] == 1 else "s"))
    L.append("    even when nothing changed. The per-skill rule is a one-sided 95% interval")
    L.append("    ANDed with a %.0f pp floor, so the per-skill false-flag rate is at most" % DROP_PP)
    L.append("    2.5%% and in practice far lower — the floor does most of the work, and it")
    L.append("    falls, not rises, as you add episodes. This tool states that rather than")
    L.append("    silently correcting for it, because on the damage objective a multiplicity")
    L.append("    correction measurably removed the flags that were doing the work.")

    L.append("")
    L.append("  ONE THING THIS DOES NOT TELL YOU")
    L.append("    Whether a skill is down because the new model is worse, or because")
    L.append("    training is a lottery and you drew a different ticket. Re-evaluating one")
    L.append("    UNCHANGED checkpoint ten times gave a suite standard deviation of 2.45 pp")
    L.append("    and a spread of 9.0 pp between the best and worst of those ten runs, in")
    L.append("    the one cell where that was measured (research/FLOOR0_RESULTS_2026-08-29).")
    L.append("    To separate the two you need the same recipe trained more than once, and")
    L.append("    then `orbit-eval route build` decides per skill over that pool.")
    if meta:
        L.append("")
        L.append("  record: %s  inputs: %s" % (meta.get("record_sha256", "")[:16],
                                               meta.get("inputs_sha256", "")[:16]))
    return "\n".join(L)


def format_history(h):
    L = ["=" * 74]
    L.append("  RELEASE HISTORY — %d of your last %d releases regressed at least one skill"
             % (h["with_a_regression"], h["releases"]))
    L.append("=" * 74)
    L.append("")
    for s in h["steps"]:
        flag = "REGRESSED" if s["n_regressed"] else ("suspect" if s["n_suspect"] else "clean")
        L.append("  %-14s -> %-14s  suite %+6.2f pp   %-10s %s"
                 % (s["from"][:14], s["to"][:14], s["suite_delta_pp"], flag,
                    (", ".join(s["regressed_skills"][:3]) if s["regressed_skills"] else "")))
    L.append("")
    hid = [s for s in h["steps"] if s["n_regressed"] and s["suite_delta_pp"] >= -1.0]
    if hid:
        L.append("  %d of the %d releases that broke a skill had a suite average that held or"
                 % (len(hid), h["with_a_regression"]))
        L.append("  improved (%s). The suite average is the number release decisions are"
                 % ", ".join("%s %+.2f pp" % (s["to"], s["suite_delta_pp"]) for s in hid[:3]))
        L.append("  usually made on, and it did not object.")
    elif h["with_a_regression"]:
        L.append("  Here the suite average moved with the breakage, so a suite-level gate")
        L.append("  would have caught these. That is not guaranteed: an average over many")
        L.append("  skills gets steadier as you add skills, so the more skills you ship the")
        L.append("  less a suite gate can see. Re-run this as your suite grows.")
    else:
        L.append("  No release in this history broke a skill beyond your measurement noise.")
    L.append("")
    L.append("  Two cautions before quoting the rate above. It counts flags, not causes: a")
    L.append("  skill can drop because the model got worse OR because training is a lottery")
    L.append("  and you drew a different ticket. And at %d skills you should expect about"
             % (h["steps"][0]["n_skills"] if h["steps"] and "n_skills" in h["steps"][0] else 0))
    L.append("  %.1f false flags per release even when nothing changed."
             % (ALPHA * (h["steps"][0].get("n_skills", 0) if h["steps"] else 0)))
    return "\n".join(L)


def load(path, order=None):
    """Read an episode-level CSV into version -> skill -> [bool]."""
    raw, _blocks = _load_csv(path)
    versions = list(raw.keys())
    if order:
        want = [v.strip() for v in order.split(",")]
        unknown = [v for v in want if v not in raw]
        if unknown:
            raise ValueError("--order names versions not in the file: %s (present: %s)"
                             % (", ".join(unknown), ", ".join(sorted(versions))))
        versions = want
    return raw, versions


def record(payload, argv, inputs=()):
    body = dict(payload)
    files = []
    for f in inputs:
        try:
            files.append({"path": os.path.relpath(f), "sha256": _sha256(f)})
        except OSError:
            files.append({"path": str(f), "sha256": None})
    body["meta"] = {"tool": "orbit-eval release",
                    "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "argv": list(argv),
                    "inputs": files,
                    "rule": {"drop_pp": payload.get("thresholds", {}).get("drop_pp", DROP_PP),
                             "min_episodes": payload.get("thresholds", {}).get(
                                 "min_episodes", MIN_EPISODES),
                             "decision": "REGRESSED iff the whole 95% interval is at or "
                                         "below -drop_pp"}}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["record_sha256"] = hashlib.sha256(blob).hexdigest()
    return body
