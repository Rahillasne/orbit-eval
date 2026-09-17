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

"""What the recording is missing: factor coverage, from design of experiments.

Treat the task as a set of factors and the recording as a set of episodes, and
report the combinations that were never recorded. That is combinatorial
testing applied to demonstrations, and it is counting rather than prediction.
The instruction text gives the first factors: "put the {A} on the {B}" has two,
and a cell like bowl + stove either holds episodes or it does not. The
per-episode statistics give the rest: how long each episode ran, which joints
moved in it, and where each joint sat on average, split into thirds of the
range the whole recording covered.

Every number here traces to a file or an arithmetic step. The expected count of
a cell is the episode count divided by the number of cells, which is what even
coverage would put there, and the largest gap is the empty cell with the
largest expected count. Nothing here says a gap caused a failure and nothing
here scores a demonstration. This company retired a package for claiming to
predict training outcomes from data, and coverage prints counts and empty
cells because that is the whole of what the metadata can support.

The v2.1 layout keeps the per-episode table as JSON lines and is read with no
dependency. The v3.0 layout keeps it as parquet; that is read through
`pyarrow` when it is installed (`pip install orbit-eval[coverage]`) and the
report says what it needs when it is not. The base install stays
dependency-free either way.
"""

import glob
import json
import os

from . import dataset

STATE = dataset.STATE

# A template with more slots than this is noise, not a design.
SLOT_MAX = 3
# A template needs at least this many shared words, and at least this fraction
# of the shortest instruction, before its slots are read as factors.
ANCHOR_MIN = 2
ANCHOR_FRAC = 0.5
# A joint that moved less than this fraction of its recorded range in an
# episode was idle in that episode.
IDLE_FRAC = 0.05
# Episode-length outliers, as multiples of the median.
LONG_X = 2.0
SHORT_X = 0.5
THIRDS = ("lower", "middle", "upper")
MAX_LIST = 6

CLOSING = ("This counts what was recorded. It does not say a gap caused a "
           "failure, and it never scores a demonstration.")


class Episode(object):
    """One recorded episode: its instruction, its length, and its joint statistics."""

    __slots__ = ("index", "tasks", "length", "state")

    def __init__(self, index, tasks, length, state=None):
        self.index = index
        self.tasks = list(tasks or [])
        self.length = length
        # {"min": [...], "max": [...], "mean": [...], "std": [...]} or None
        self.state = state


class Template(object):
    """The shared shape of a set of instructions, and each instruction's slot values."""

    def __init__(self, template, slots, levels):
        self.template = template      # "put the {A} on the {B}", or None
        self.slots = slots            # ["A", "B"], or ["instruction"], or []
        self.levels = levels          # {instruction: {slot: value}}


# ------------------------------------------------------------------ reading

def _pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return pq


def _read_jsonl(path):
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def load_tasks(meta):
    """Every instruction the dataset declares, in task-index order, or None.

    The tasks file can list an instruction no episode carries, and that is a
    coverage fact worth reporting: a cell with nothing in it.
    """
    root = meta.root
    p = os.path.join(root, "meta", "tasks.jsonl")
    if os.path.isfile(p):
        rows = []
        for r in _read_jsonl(p):
            if isinstance(r, dict) and isinstance(r.get("task"), str):
                rows.append((_int(r.get("task_index")), r["task"]))
        return [t for _i, t in sorted(rows, key=lambda x: (x[0] is None, x[0] or 0))]
    p = os.path.join(root, "meta", "tasks.parquet")
    if os.path.isfile(p):
        pq = _pyarrow()
        if pq is None:
            return None
        try:
            rows = pq.read_table(p).to_pylist()
        except Exception:
            return None
        out = []
        for r in rows:
            text = r.get("task")
            if not isinstance(text, str):
                text = r.get("__index_level_0__")
            if not isinstance(text, str):
                cands = [v for k, v in r.items() if isinstance(v, str)]
                text = cands[0] if cands else None
            if isinstance(text, str):
                out.append((_int(r.get("task_index")), text))
        return [t for _i, t in sorted(out, key=lambda x: (x[0] is None, x[0] or 0))]
    return None


def load_episodes(meta):
    """(episodes, what was read, why not) for one dataset.

    `why not` is None on success, "pyarrow" when the table is parquet and
    pyarrow is not installed, and "missing" when there is no per-episode table.
    """
    root = meta.root
    jl = os.path.join(root, "meta", "episodes.jsonl")
    if os.path.isfile(jl):
        eps = []
        for r in _read_jsonl(jl):
            if not isinstance(r, dict):
                continue
            idx = _int(r.get("episode_index"))
            eps.append(Episode(idx if idx is not None else len(eps),
                               r.get("tasks") or [], _int(r.get("length"))))
        how = "meta/episodes.jsonl"
        st = os.path.join(root, "meta", "episodes_stats.jsonl")
        if os.path.isfile(st):
            by_index = {e.index: e for e in eps}
            for r in _read_jsonl(st):
                if not isinstance(r, dict):
                    continue
                e = by_index.get(_int(r.get("episode_index")))
                blocks = r.get("stats") if isinstance(r.get("stats"), dict) else r
                block = blocks.get(STATE) if isinstance(blocks, dict) else None
                if e is not None and isinstance(block, dict):
                    e.state = _state_block(block.get("min"), block.get("max"),
                                           block.get("mean"), block.get("std"))
            how += " and meta/episodes_stats.jsonl"
        eps.sort(key=lambda e: e.index)
        return eps, how, None

    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"),
                             recursive=True))
    if files:
        how = "meta/episodes/*.parquet"
        pq = _pyarrow()
        if pq is None:
            return None, how, "pyarrow"
        eps = []
        for f in files:
            try:
                rows = pq.read_table(f).to_pylist()
            except Exception:
                continue
            for r in rows:
                if not isinstance(r, dict):
                    continue
                idx = _int(r.get("episode_index"))
                state = _state_block(r.get("stats/%s/min" % STATE), r.get("stats/%s/max" % STATE),
                                     r.get("stats/%s/mean" % STATE), r.get("stats/%s/std" % STATE))
                eps.append(Episode(idx if idx is not None else len(eps),
                                   r.get("tasks") or [], _int(r.get("length")), state))
        eps.sort(key=lambda e: e.index)
        return eps, how, None
    return None, "", "missing"


def _state_block(lo, hi, mean, std):
    lo, hi, mean = dataset._vec(lo), dataset._vec(hi), dataset._vec(mean)
    if lo is None or hi is None or mean is None:
        return None
    return {"min": [_f(x) for x in lo], "max": [_f(x) for x in hi],
            "mean": [_f(x) for x in mean],
            "std": [_f(x) for x in dataset._vec(std)] if dataset._vec(std) else None}


# ----------------------------------------------------------------- template

def induce_template(instructions):
    """The words every instruction shares, in order, with the rest as slots.

    "pick up the milk and place it in the basket" and "pick up the bbq sauce
    and place it in the basket" share nine words in order, so the template is
    "pick up the {A} and place it in the basket" and A is a factor with two
    levels. Instructions that share too little are one factor, the instruction
    itself, because reading slots into them would invent a design nobody used.
    """
    seen, instrs = set(), []
    for s in instructions:
        if isinstance(s, str) and s not in seen:
            seen.add(s)
            instrs.append(s)
    if not instrs:
        return Template(None, [], {})
    if len(instrs) == 1:
        return Template(instrs[0], [], {instrs[0]: {}})
    toks = [s.split() for s in instrs]
    low = [[t.lower() for t in ts] for ts in toks]
    ref = min(range(len(low)), key=lambda i: len(low[i]))
    anchors, cursors = [], [0] * len(low)
    for tok in low[ref]:
        hits = []
        for ts, c in zip(low, cursors):
            try:
                hits.append(ts.index(tok, c))
            except ValueError:
                hits = None
                break
        if hits is None:
            continue
        anchors.append(tok)
        cursors = [h + 1 for h in hits]
    if len(anchors) < ANCHOR_MIN or len(anchors) < ANCHOR_FRAC * len(low[ref]):
        return _single_factor(instrs)

    per, ref_anchor_words = [], []
    for i, (ts, orig) in enumerate(zip(low, toks)):
        c, segs = 0, []
        for a in anchors:
            h = ts.index(a, c)
            segs.append(" ".join(orig[c:h]))
            if i == ref:
                ref_anchor_words.append(orig[h])
            c = h + 1
        segs.append(" ".join(orig[c:]))
        per.append(segs)
    keep = [k for k in range(len(anchors) + 1) if any(p[k] for p in per)]
    if not keep:
        return Template(" ".join(toks[ref]), [], {s: {} for s in instrs})
    if len(keep) > SLOT_MAX:
        return _single_factor(instrs)
    names = [chr(ord("A") + i) for i in range(len(keep))]
    parts = []
    for k in range(len(anchors) + 1):
        if k in keep:
            parts.append("{%s}" % names[keep.index(k)])
        if k < len(anchors):
            parts.append(ref_anchor_words[k])
    levels = {}
    for s, segs in zip(instrs, per):
        levels[s] = {names[j]: segs[k] for j, k in enumerate(keep)}
    return Template(" ".join(parts), names, levels)


def _single_factor(instrs):
    return Template(None, ["instruction"], {s: {"instruction": s} for s in instrs})


# ----------------------------------------------------------------- counting

def scan(path="."):
    """The coverage report for the dataset at or below `path`."""
    meta = dataset.load(path)
    eps, how, reason = load_episodes(meta)
    tasks = load_tasks(meta)
    return report(meta, eps, tasks, how, reason)


def report(meta, eps, tasks, how="", reason=None):
    n_total = meta.n_episodes if meta.n_episodes is not None else len(eps or [])
    rep = {
        "root": meta.root,
        "name": os.path.basename(meta.root.rstrip(os.sep)) or meta.root,
        "version": meta.version,
        "robot_type": meta.robot_type,
        "n_episodes": n_total,
        "n_tasks": len(tasks) if tasks else _int(meta.info.get("total_tasks")),
        "read": how,
        "needs": "pyarrow" if reason == "pyarrow" else None,
        "instructions": None,
        "lengths": None,
        "joints": [],
        "regions": None,
        "largest_gap": None,
        "notes": [],
    }
    if eps is None:
        if reason == "pyarrow":
            rep["notes"].append(
                "the per-episode table is parquet in the v3.0 layout and pyarrow is "
                "not installed, so per-task counts and per-episode factors could not "
                "be read. `pip install 'orbit-eval[coverage]'` and run this again.")
        else:
            rep["notes"].append("no per-episode table under meta/, so only the totals "
                                "in info.json are known.")
        return rep

    # -- instructions: the factors that come from the task text
    first = [e.tasks[0] if e.tasks else "" for e in eps]
    multi = sum(1 for e in eps if len(e.tasks) > 1)
    if multi:
        rep["notes"].append("%d episode%s carr%s more than one instruction; each is "
                            "counted under its first." % (multi, "" if multi == 1 else "s",
                                                          "ies" if multi == 1 else "y"))
    ordered = []
    for t in list(tasks or []) + first:
        if t and t not in ordered:
            ordered.append(t)
    counts = {}
    for t in first:
        counts[t] = counts.get(t, 0) + 1
    n = len(eps)
    tmpl = induce_template(ordered)
    per_task = [{"task": t, "episodes": counts.get(t, 0)} for t in ordered]
    empty_tasks = [t for t in ordered if counts.get(t, 0) == 0]
    slots = []
    for s in tmpl.slots:
        lv = {}
        for t in ordered:
            key = tmpl.levels.get(t, {}).get(s, "")
            lv[key] = lv.get(key, 0) + counts.get(t, 0)
        slots.append({"name": s, "levels": lv})
    pairs = []
    if len(tmpl.slots) >= 2:
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots[i], slots[j]
                cell = {}
                for t in ordered:
                    lv = tmpl.levels.get(t, {})
                    k = (lv.get(a["name"], ""), lv.get(b["name"], ""))
                    cell[k] = cell.get(k, 0) + counts.get(t, 0)
                empty = [[x, y] for x in a["levels"] for y in b["levels"]
                         if cell.get((x, y), 0) == 0]
                n_cells = len(a["levels"]) * len(b["levels"])
                pairs.append({"slots": [a["name"], b["name"]], "cells": n_cells,
                              "empty": empty,
                              "expected": (float(n) / n_cells) if n_cells else 0.0})
    rep["instructions"] = {"template": tmpl.template, "slots": slots,
                           "per_task": per_task, "empty_tasks": empty_tasks,
                           "pairs": pairs}
    rep["n_tasks"] = len(ordered)

    # -- lengths
    lengths = [(e.index, e.length) for e in eps if e.length]
    if lengths:
        med = _median([l for _i, l in lengths])
        rep["lengths"] = {
            "median": med,
            "min": min(l for _i, l in lengths), "max": max(l for _i, l in lengths),
            "long": [i for i, l in lengths if l > LONG_X * med],
            "short": [i for i, l in lengths if l < SHORT_X * med],
        }

    # -- joints: which moved, and where each sat, per episode
    with_state = [e for e in eps if e.state]
    if with_state:
        names = meta.joint_names
        width = min(len(e.state["min"]) for e in with_state)
        width = min(width, min(len(e.state["max"]) for e in with_state),
                    min(len(e.state["mean"]) for e in with_state))
        joints, empty_regions = [], []
        tasks_with = [t for t in ordered if counts.get(t, 0)]
        n_regions = 0
        for j in range(width):
            name = dataset._name(names, j)
            lo = min(e.state["min"][j] for e in with_state if e.state["min"][j] is not None)
            hi = max(e.state["max"][j] for e in with_state if e.state["max"][j] is not None)
            span = hi - lo
            idle_eps, thirds = [], [0, 0, 0]
            by_task_third = {}
            for e in with_state:
                travel = e.state["max"][j] - e.state["min"][j]
                if span > 0 and travel < IDLE_FRAC * span:
                    idle_eps.append(e)
                if span > 0:
                    third = int(3 * (e.state["mean"][j] - lo) / span)
                    third = max(0, min(2, third))
                    thirds[third] += 1
                    t = e.tasks[0] if e.tasks else ""
                    by_task_third.setdefault(t, [0, 0, 0])[third] += 1
            idle_tasks = []
            for t in tasks_with:
                mine = [e for e in with_state if (e.tasks[0] if e.tasks else "") == t]
                if mine and all(e in idle_eps for e in mine):
                    idle_tasks.append(t)
            if span > 0:
                for t in tasks_with:
                    row = by_task_third.get(t, [0, 0, 0])
                    n_t = sum(row)
                    if not n_t:
                        continue
                    n_regions += 3
                    for k, c in enumerate(row):
                        if c == 0:
                            empty_regions.append({
                                "task": t, "joint": name, "third": THIRDS[k],
                                "others": [row[m] for m in range(3) if m != k],
                                "expected": n_t / 3.0})
            joints.append({"name": name, "range": [lo, hi],
                           "idle_episodes": len(idle_eps),
                           "idle_index": [e.index for e in idle_eps],
                           "idle_tasks": idle_tasks,
                           "thirds": thirds if span > 0 else None})
        empty_regions.sort(key=lambda c: (-c["expected"], c["task"], c["joint"]))
        rep["joints"] = joints
        rep["regions"] = {"cells": n_regions, "empty": empty_regions}
    else:
        rep["notes"].append("no per-episode joint statistics, so which joints moved "
                            "and where the arm sat could not be counted.")

    rep["largest_gap"] = _largest_gap(rep, n)
    return rep


def _largest_gap(rep, n):
    """The empty cell that even coverage would have filled the most."""
    cands = []
    ins = rep.get("instructions") or {}
    n_tasks = len(ins.get("per_task") or [])
    for t in ins.get("empty_tasks") or []:
        e = float(n) / n_tasks if n_tasks else 0.0
        cands.append({"kind": "task", "expected": e, "cell": t,
                      "sentence": ("'%s' is listed in meta/tasks and has no episode; "
                                   "at even coverage it would hold about %d of the %d."
                                   % (t, round(e), n))})
    for p in ins.get("pairs") or []:
        for x, y in p["empty"]:
            cands.append({"kind": "pair", "expected": p["expected"],
                          "cell": "%s=%s, %s=%s" % (p["slots"][0], x, p["slots"][1], y),
                          "sentence": ("no episode has %s = %s together with %s = %s; "
                                       "at even coverage that cell would hold about %d "
                                       "of the %d episodes."
                                       % (p["slots"][0], x or "(none)", p["slots"][1],
                                          y or "(none)", round(p["expected"]), n))})
    for c in ((rep.get("regions") or {}).get("empty") or []):
        cands.append({"kind": "region", "expected": c["expected"],
                      "cell": "%s / %s / %s" % (c["task"], c["joint"], c["third"]),
                      "sentence": ("in '%s', %s never averaged in the %s third of its "
                                   "recorded range; the other two thirds hold %d and %d "
                                   "episodes." % (c["task"], c["joint"], c["third"],
                                                  c["others"][0], c["others"][1]))})
    if not cands:
        return None
    return max(cands, key=lambda c: c["expected"])


# ----------------------------------------------------------------- printing

def format_cover(rep):
    L = [""]
    head = "%d episodes" % rep["n_episodes"] if rep["n_episodes"] is not None else "episodes"
    if rep.get("n_tasks"):
        head += ", %d instruction%s" % (rep["n_tasks"], "" if rep["n_tasks"] == 1 else "s")
    if rep.get("version"):
        head += ", %s" % rep["version"]
    L.append("  COVERAGE  %s  (%s)" % (rep["name"], head))
    if rep.get("read"):
        L.append("  read from %s" % rep["read"])

    if rep.get("needs") == "pyarrow":
        L.append("")
        L.append("  NEEDS")
        for note in rep["notes"]:
            for line in _wrap(note, 72):
                L.append("    " + line)
        L.append("")
        L.append("  " + CLOSING)
        L.append("")
        return "\n".join(line.rstrip() for line in L)

    ins = rep.get("instructions")
    if ins:
        L.append("")
        L.append("  INSTRUCTIONS")
        if ins["template"] and ins["slots"]:
            L.append("    template     %s" % ins["template"])
            for s in ins["slots"]:
                items = ["%s %d" % (k or "(none)", v) for k, v in s["levels"].items()]
                for i, line in enumerate(_wrap(", ".join(items), 60)):
                    L.append("    %-12s %s" % (s["name"] if i == 0 else "", line))
        elif ins["template"] and not ins["slots"]:
            L.append("    one instruction: %s" % ins["template"])
        else:
            rows = ins["per_task"]
            for r in rows[:MAX_LIST]:
                L.append("    %4d  %s" % (r["episodes"], r["task"]))
            if len(rows) > MAX_LIST:
                L.append("          ... and %d more" % (len(rows) - MAX_LIST))
        if ins["empty_tasks"]:
            L.append("    never recorded   %s" % ins["empty_tasks"][0])
            for t in ins["empty_tasks"][1:MAX_LIST]:
                L.append("                     %s" % t)
            if len(ins["empty_tasks"]) > MAX_LIST:
                L.append("                     ... and %d more"
                         % (len(ins["empty_tasks"]) - MAX_LIST))
        for p in ins["pairs"]:
            label = "%s x %s" % tuple(p["slots"])
            if p["empty"]:
                cells = ", ".join("%s + %s" % (x or "(none)", y or "(none)")
                                  for x, y in p["empty"][:MAX_LIST])
                more = ("" if len(p["empty"]) <= MAX_LIST
                        else " and %d more" % (len(p["empty"]) - MAX_LIST))
                L.append("    %-12s %d cells, %d never recorded: %s%s"
                         % (label, p["cells"], len(p["empty"]), cells, more))
            else:
                L.append("    %-12s %d cells, every one recorded" % (label, p["cells"]))

    ln = rep.get("lengths")
    if ln or rep.get("joints"):
        L.append("")
        L.append("  EPISODES")
    if ln:
        s = "%d to %d frames, median %d" % (ln["min"], ln["max"], ln["median"])
        bits = []
        if ln["long"]:
            bits.append("%d over %gx the median (index %s)"
                        % (len(ln["long"]), LONG_X, _idx(ln["long"])))
        if ln["short"]:
            bits.append("%d under half the median (index %s)"
                        % (len(ln["short"]), _idx(ln["short"])))
        s += "; " + ", ".join(bits) if bits else "; none beyond %gx or under half the median" % LONG_X
        for i, line in enumerate(_wrap(s, 60)):
            L.append("    %-12s %s" % ("length" if i == 0 else "", line))
    idle = [j for j in rep.get("joints") or [] if j["idle_episodes"]]
    if idle:
        n_state = rep["n_episodes"]
        n_joints = len(rep.get("joints") or [])
        groups = []                      # joints idle in exactly the same episodes
        for j in idle:
            key = tuple(j.get("idle_index") or [])
            for g in groups:
                if g[0] == key:
                    g[1].append(j)
                    break
            else:
                groups.append((key, [j]))
        first = True
        for key, js in groups:
            who = ("all %d joints" % n_joints if len(js) == n_joints
                   else ", ".join(j["name"] for j in js))
            s = "%s moved under %d%% of %s range in %d of %d episodes" % (
                who, int(IDLE_FRAC * 100), "their" if len(js) > 1 else "its",
                len(key), n_state)
            if key and len(key) <= MAX_LIST:
                s += " (index %s)" % _idx(list(key))
            tasks = js[0]["idle_tasks"]
            if tasks:
                s += ", every episode of '%s'%s" % (
                    tasks[0], "" if len(tasks) == 1 else " and %d more" % (len(tasks) - 1))
            for k, line in enumerate(_wrap(s, 60)):
                L.append("    %-12s %s" % ("idle joints" if first and k == 0 else "", line))
            first = False
    elif rep.get("joints"):
        L.append("    %-12s every joint moved in every episode" % "idle joints")

    joints = [j for j in rep.get("joints") or [] if j["thirds"]]
    if joints:
        L.append("")
        L.append("  WHERE THE ARM SAT   mean joint position per episode, by thirds of its "
                 "recorded range")
        L.append("    %-20s %6s %6s %6s" % ("", "lower", "middle", "upper"))
        for j in joints:
            L.append("    %-20s %6d %6d %6d" % ((j["name"][:20],) + tuple(j["thirds"])))
        reg = rep.get("regions") or {}
        if reg.get("cells"):
            L.append("    %-12s %d of %d task x joint x third combinations hold no episode"
                     % ("empty cells", len(reg["empty"]), reg["cells"]))

    L.append("")
    L.append("  THE LARGEST GAP")
    gap = rep.get("largest_gap")
    if gap:
        for line in _wrap(gap["sentence"], 72):
            L.append("    " + line)
    else:
        L.append("    no empty cell: every instruction, every instruction pair and every")
        L.append("    third of every joint's range holds at least one episode.")

    notes = [n for n in rep.get("notes") or []]
    if notes:
        L.append("")
        L.append("  NOTE")
        for note in notes:
            for line in _wrap(note, 72):
                L.append("    " + line)

    L.append("")
    for line in _wrap(CLOSING, 74):
        L.append("  " + line)
    L.append("")
    return "\n".join(line.rstrip() for line in L)


# ------------------------------------------------------------------ helpers

def _idx(xs):
    s = ", ".join(str(x) for x in xs[:MAX_LIST])
    return s + (", ..." if len(xs) > MAX_LIST else "")


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    m = 0.5 * (xs[mid - 1] + xs[mid])
    return int(m) if m == int(m) else m


def _int(v):
    try:
        if isinstance(v, bool):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w) if line else w
    if line:
        out.append(line)
    return out or [""]
