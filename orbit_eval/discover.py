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

"""orbit-eval discover — find the evaluations already sitting on this machine.

`release` and `route` ask you to hand them a tidy matrix. Nobody has a tidy matrix.
What people have is whatever `lerobot-eval` last wrote, a spreadsheet someone kept
while standing at the robot, and three directories named `outputs/`. This module
walks a tree and works out which of those are evaluations, which rows belong to the
same trained policy, and which policy is the one already running.

Three shapes are recognised, all of which the loaders elsewhere in this package
already read:

  * LeRobot     any `eval_info.json` carrying `per_task[*].metrics.successes`,
                or the single-task shape, which carries `per_episode[*].success`
                or only `aggregated.pc_success` and `aggregated.n_episodes`
  * episode CSV  a CSV/TSV with columns for version, skill, episode and success
  * matrix JSON  `{"candidates": {name: {task: [0/1, ...]}}}`

Nothing here computes a statistic. It decides what the inputs ARE; every number is
still produced by `release`, `route` and `power`. Keeping that boundary is the point:
detection is heuristic and revisable, the decision is not.
"""

import csv
import json
import os
import re

GENERIC_DIRS = {
    # directory names that describe the artifact rather than the policy that made it
    "eval", "evals", "eval_final", "eval_last", "eval_best", "evaluation", "evaluations",
    "checkpoints", "checkpoint", "ckpt", "last", "best", "final", "pretrained_model",
    "outputs", "output", "out", "results", "result", "logs", "log", "runs", "artifacts",
    "videos", "video", "wandb", "train", ".",
}
SKIP_DIRS = {".git", ".hg", "node_modules", "__pycache__", ".venv", "venv", ".tox",
             ".mypy_cache", ".pytest_cache", "site-packages", ".cache"}
MAX_FILES = 20000

VERSION_COLS = ("version", "candidate", "model", "policy", "run", "checkpoint",
                "experiment", "arm", "variant", "method", "system")
SKILL_COLS = ("skill", "task", "job", "task_name", "skill_name", "task_suite",
              "suite", "benchmark", "axis", "scenario", "split")
EPISODE_COLS = ("episode", "ep", "trial", "rollout", "episode_index", "i")
SUCCESS_COLS = ("success", "succeeded", "ok", "pass", "passed", "result", "outcome", "reward")
# Why a trial failed, when somebody recorded it. `orbit log` writes this column.
REASON_COLS = ("reason", "failure_mode", "failure", "why", "error", "cause", "tag")

# How many trials a row summarises. Deliberately a SHORT, exact list: a column called
# n_demos is a training budget, not an episode count, and reading it as one would put
# a confident interval on a number that never had the trials behind it.
COUNT_COLS = ("n", "n_eval", "n_episodes", "episodes", "num_episodes", "n_trials",
              "trials", "total_episodes", "n_rollouts", "rollouts")
COUNT_PREFIXES = ("n_per_arm",)
# Counts of successes in an aggregated row.
KCOUNT_COLS = ("successes", "n_success", "num_success", "successful", "n_successes",
               "hits", "passes")
# A rate, which with a count recovers the same two numbers.
RATE_HINTS = ("sr", "success_rate", "success_rate_percent", "rate", "pc_success",
              "successrate")
RATE_SUFFIXES = ("_sr", "_success_rate", "_rate", "_succ")

# The job name for an eval_info.json that names no task. A single-task
# `lerobot-eval` writes `per_episode` and `aggregated` and nothing else; it is
# the commonest evaluation artifact in public and it has no task in it.
SINGLE_TASK = "task/0"

TRUE = {"1", "true", "t", "yes", "y", "success", "succeeded", "pass", "passed", "ok"}
FALSE = {"0", "false", "f", "no", "n", "failure", "failed", "fail", "miss", ""}


class Source(object):
    """One place evaluations were found, already decoded to {candidate: {skill: [bool]}}."""

    def __init__(self, kind, path, data, mtimes, note="", paired_ok=False,
                 reasons=None, media=None, seeds=None):
        self.kind = kind          # "lerobot" | "csv" | "matrix"
        self.path = path          # the directory or file a human would name
        self.data = data          # {candidate: {skill: [bool, ...]}}
        self.mtimes = mtimes      # {candidate: float} newest contributing file
        self.note = note
        # Whether the k-th trial means the same thing for every policy. False for
        # anything reconstructed from counts, where the individual trials are gone
        # and pairing them would invent an agreement that was never measured.
        self.paired_ok = paired_ok
        # {candidate: {skill: {reason: count}}} for the failures, when recorded.
        self.reasons = reasons or {}
        # {candidate: {skill: [path or None, ...]}} aligned with the episode lists.
        # LeRobot writes one video per episode and nobody ever watches them, because
        # a directory of 1,000 mp4s with no index is not evidence. Carrying them here
        # is what lets a regression name the two files worth opening.
        self.media = media or {}
        # {candidate: {skill: [seed, ...]}} when the file recorded one per
        # episode. Two runs with the same seed at every index ARE common random
        # numbers, and `check` verifies that instead of asking for --crn.
        self.seeds = seeds or {}

    @property
    def candidates(self):
        return sorted(self.data)

    @property
    def n_skills(self):
        return len(set().union(*(set(v) for v in self.data.values()))) if self.data else 0

    def episodes_per_skill(self):
        ns = [len(b) for v in self.data.values() for b in v.values()]
        return (min(ns), max(ns)) if ns else (0, 0)

    def summary(self):
        lo, hi = self.episodes_per_skill()
        eps = "%d" % lo if lo == hi else "%d-%d" % (lo, hi)
        return "%d %s x %d %s, %s episodes each (%s)" % (
            len(self.data), "policy" if len(self.data) == 1 else "policies",
            self.n_skills, "job" if self.n_skills == 1 else "jobs", eps, self.kind)


def _norm(s):
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")


def _truthy(v):
    s = _norm(str(v))
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    try:                      # a reward column: anything above zero counts as a success
        return float(s) > 0.0
    except ValueError:
        raise ValueError("cannot read %r as a success value" % (v,))


def _pick(fieldnames, wanted):
    low = {_norm(f): f for f in (fieldnames or [])}
    for w in wanted:
        if w in low:
            return low[w]
    return None


# File stems that describe the artifact rather than the policy that produced it.
GENERIC_STEMS = {"eval_info", "eval_info-checkpoint", "eval", "evaluation", "results",
                 "result", "summary", "metrics", "output", "out", "info", "log",
                 "eval_results", "final", "smoke_eval_info"}


def candidate_name(eval_path, root):
    """Name the trained policy that produced `eval_path`.

    Two layouts exist in the wild and both have to work. Either one directory per
    policy holding an `eval_info.json`, where the directory is the name; or one
    directory holding `eval_act_50k.json` and `eval_smolvla_20k.json` side by side,
    where the FILE is the name. Reading only the first layout silently collapsed the
    second into a single policy, which is the quietest way to lose a comparison.
    """
    base = os.path.basename(eval_path)
    stem = os.path.splitext(base)[0]
    if _norm(stem) not in GENERIC_STEMS and not stem.isdigit():
        # trim the part that says "this is an evaluation" and keep what names the run
        name = re.sub(r"^(eval|evals|evaluation|results?)[-_]+", "", stem, flags=re.I)
        name = re.sub(r"[-_]*(eval_info|eval|results?)$", "", name, flags=re.I)
        if name and _norm(name) not in GENERIC_STEMS:
            return name
    d = os.path.dirname(os.path.abspath(eval_path))
    root = os.path.abspath(root)
    parts = []
    while d and d != root and d != os.path.dirname(d):
        base = os.path.basename(d)
        parts.append(base)
        if _norm(base) not in GENERIC_DIRS and not base.isdigit():
            return base
        d = os.path.dirname(d)
    return "/".join(reversed(parts)) if parts else os.path.basename(root) or "candidate"


def _walk(root):
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames)
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in sorted(filenames):
            n += 1
            if n > MAX_FILES:
                return
            yield os.path.join(dirpath, fn)


def _read_lerobot(path, media_out=None, flags_out=None, seeds_out=None):
    """{skill: [bool]} from one eval_info.json, or None if it is not one.

    Three shapes, in order of how much they carry. `per_task` names every task
    and lists every trial. The single-task shape lists every trial under
    `per_episode` but names no task, so the one job is called `SINGLE_TASK`.
    The thinnest shape, which is what the Hub serves for most published
    policies, carries only `aggregated.pc_success` and `n_episodes`; the trials
    are reconstructed from the counts and `flags_out["reconstructed"]` is set,
    because a list rebuilt from a rate must never be paired episode by episode.
    """
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    out = {}
    for pt in d.get("per_task") or []:
        if not isinstance(pt, dict):
            continue
        m = pt.get("metrics") or {}
        s = m.get("successes")
        if not isinstance(s, list) or not s:
            continue
        key = "%s/%s" % (pt.get("task_group", "task"), pt.get("task_id", len(out)))
        out[key] = [bool(x) for x in s]
        if media_out is not None:
            _align_media(media_out, key, m.get("video_paths"), len(s))
        if seeds_out is not None:
            sd = m.get("seeds")
            if isinstance(sd, list) and len(sd) == len(s) and all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in sd):
                seeds_out[key] = [int(x) for x in sd]
    if out:
        return out

    eps = d.get("per_episode")
    if isinstance(eps, list) and eps:
        rows = [e for e in eps if isinstance(e, dict) and e.get("success") is not None]
        flags = [bool(e["success"]) for e in rows]
        if flags:
            out[SINGLE_TASK] = flags
            if media_out is not None:
                _align_media(media_out, SINGLE_TASK, d.get("video_paths"), len(flags))
            if seeds_out is not None:
                sd = [e.get("seed") for e in rows]
                if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in sd):
                    seeds_out[SINGLE_TASK] = [int(x) for x in sd]
            return out

    for block in (d.get("aggregated"), d.get("overall")):
        if not isinstance(block, dict):
            continue
        n = block.get("n_episodes")
        if not isinstance(n, (int, float)) or isinstance(n, bool) or n <= 0:
            continue
        n = int(n)
        k = block.get("n_success")
        if not isinstance(k, (int, float)) or isinstance(k, bool):
            pct = block.get("pc_success")
            if not isinstance(pct, (int, float)) or isinstance(pct, bool):
                continue
            k = n * float(pct) / 100.0
        out[SINGLE_TASK] = _bits(n, k)
        if flags_out is not None:
            flags_out["reconstructed"] = True
        return out
    return None


def _align_media(media_out, key, vids, n):
    if not isinstance(vids, list):
        return
    # Real files carry an off-by-one: one extra aggregate video is common.
    # Truncate rather than guess, so an index never names the wrong clip.
    media_out[key] = [str(v) for v in vids[:n]]
    if len(media_out[key]) < n:
        media_out[key] += [None] * (n - len(media_out[key]))


def _looks_like_rate(name):
    n = _norm(name)
    return n in RATE_HINTS or n.endswith(RATE_SUFFIXES)


def _pick_count(fieldnames):
    c = _pick(fieldnames, COUNT_COLS)
    if c:
        return c
    for f in fieldnames or []:
        if _norm(f).startswith(COUNT_PREFIXES):
            return f
    return None


def _num(v):
    try:
        return float(str(v).strip().replace("%", ""))
    except (TypeError, ValueError):
        return None


def _bits(n, k):
    n = int(round(n))
    k = max(0, min(n, int(round(k))))
    return [True] * k + [False] * (n - k)


def _rate_to_k(rate, n):
    """Rates in the wild are 0-1 or 0-100. Above 1 can only be a percentage."""
    r = rate / 100.0 if rate > 1.0 else rate
    return r * n


def _data_lines(path):
    """The file with leading blank and # comment lines removed.

    Real evaluation CSVs are written by researchers and very often open with a note
    about how the numbers were produced. Treating that note as the header is the
    difference between reading someone's file and telling them there is nothing here.
    """
    out, started = [], False
    with open(path, newline="") as fh:
        for line in fh:
            if not started:
                st = line.strip()
                if not st or st.startswith("#") or st.startswith("//"):
                    continue
                started = True
            out.append(line)
    return out


def _read_csv(path):
    """({candidate: {skill: [bool]}}, note, paired_ok) or (None, reason, False)."""
    try:
        lines = _data_lines(path)
        if not lines:
            return None, "empty", False
        sample = "".join(lines[:40])
        delim = "\t" if (path.lower().endswith(".tsv") or
                          sample.count("\t") > sample.count(",")) else ","
        rdr = csv.DictReader(lines, delimiter=delim)
        fn = rdr.fieldnames or []
        vc0 = _pick(fn, VERSION_COLS)
        sc0 = _pick(fn, SKILL_COLS)
        ec0 = _pick(fn, EPISODE_COLS)
        uc0 = _pick(fn, SUCCESS_COLS)
        cc0 = _pick_count(fn)
        kc0 = _pick(fn, KCOUNT_COLS)
        rate_cols = [f for f in fn if _looks_like_rate(f) and f not in (cc0, kc0)]
        # AGGREGATE: one row is many trials, given as a count plus successes or a rate.
        if cc0 and (kc0 or len(rate_cols) == 1) and not (ec0 and uc0):
            return _read_aggregate(lines, delim, vc0, sc0, cc0, kc0,
                                   rate_cols[0] if rate_cols else None)
        # WIDE: one row is a task, one column per policy, plus how many trials each ran.
        if cc0 and len(rate_cols) >= 2 and sc0:
            return _read_wide(lines, delim, sc0, cc0, rate_cols)
        if cc0 and len(rate_cols) >= 2 and not sc0:
            return None, "one column per policy but no task column to key them by", False
        if rate_cols and not cc0:
            return None, ("rates but no trial-count column, so no interval can be "
                          "computed from it"), False
        if not (sc0 or vc0):
            return None, "no policy or task column", False
        if not uc0:
            return None, "no success column", False
        return _read_episodes(lines, delim, vc0, sc0, ec0, uc0,
                              _pick(fn, REASON_COLS))
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return None, str(e), False


def _read_aggregate(lines, delim, vc, sc, cc, kc, rc):
    rdr = csv.DictReader(lines, delimiter=delim)
    data = {}
    for row in rdr:
        n = _num(row.get(cc))
        if not n or n <= 0:
            continue
        k = _num(row.get(kc)) if kc else None
        if k is None and rc is not None:
            r = _num(row.get(rc))
            if r is None:
                continue
            k = _rate_to_k(r, n)
        if k is None:
            continue
        cand = (row.get(vc) or "").strip() if vc else "candidate"
        skill = (row.get(sc) or "").strip() if sc else "all"
        if not cand:
            cand = "candidate"
        if not skill:
            skill = "all"
        if skill in data.get(cand, {}):          # same cell twice: keep the first
            continue
        data.setdefault(cand, {})[skill] = _bits(n, k)
    if not data:
        return None, "no rows with a usable trial count", False
    return data, "aggregated rows expanded from counts", False


def _read_wide(lines, delim, sc, cc, rate_cols):
    rdr = csv.DictReader(lines, delimiter=delim)
    data = {}
    for row in rdr:
        n = _num(row.get(cc))
        if not n or n <= 0:
            continue
        skill = (row.get(sc) or "").strip() or "all"
        for col in rate_cols:
            r = _num(row.get(col))
            if r is None:
                continue
            cand = col.strip()
            if skill in data.get(cand, {}):
                continue
            data.setdefault(cand, {})[skill] = _bits(n, _rate_to_k(r, n))
    if len(data) < 2:
        return None, "wide layout but fewer than two policy columns had numbers", False
    return data, "wide layout, one column per policy", False


def _read_episodes(lines, delim, vc, sc, ec, uc, rc=None):
    """One row per trial, which is the shape `orbit log` writes and the only one
    that can be paired: the k-th trial of a job means the same thing for every
    policy, so a later comparison may use common random numbers."""
    rdr = csv.DictReader(lines, delimiter=delim)
    data, order, reasons = {}, {}, {}
    for i, row in enumerate(rdr):
        cand = (row.get(vc) or "candidate").strip() if vc else "candidate"
        skill = (row.get(sc) or "").strip() if sc else "all"
        if not skill:
            continue
        if not cand:
            cand = "candidate"
        try:
            ok = _truthy(row.get(uc))
        except ValueError:
            return None, "unreadable success value on row %d" % (i + 2), False
        try:
            ep = int(float(row.get(ec))) if ec and row.get(ec) not in (None, "") else None
        except (TypeError, ValueError):
            ep = None
        data.setdefault(cand, {}).setdefault(skill, [])
        order.setdefault(cand, {}).setdefault(skill, [])
        data[cand][skill].append(ok)
        order[cand][skill].append(ep if ep is not None else len(order[cand][skill]))
        if not ok and rc:
            why = (row.get(rc) or "").strip() or "unrecorded"
            bucket = reasons.setdefault(cand, {}).setdefault(skill, {})
            bucket[why] = bucket.get(why, 0) + 1
    if not data:
        return None, "no rows", False
    for cand in data:                       # sort each skill by its episode index
        for skill in data[cand]:
            pairs = sorted(zip(order[cand][skill], data[cand][skill]), key=lambda p: p[0])
            data[cand][skill] = [ok for _ep, ok in pairs]
    return (data, "%d row group(s)" % sum(len(v) for v in data.values()), True,
            reasons)


NOT_EVALS = {"info.json", "stats.json", "config.json", "train_config.json",
             "episodes_stats.json", "release.json", "tasks.json"}


def _maybe_eval_json(path):
    """Cheap sniff: does this JSON mention evaluation at all? Only used to explain."""
    if os.path.basename(path).lower() in NOT_EVALS:
        return False
    try:
        with open(path) as fh:
            head = fh.read(4096).lower()
    except (OSError, UnicodeDecodeError):
        return False
    return any(w in head for w in ("success", "per_task", "eval", "rollout", "episode"))


def _read_matrix(path):
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("candidates"), dict):
        return None
    out = {}
    for name, tasks in obj["candidates"].items():
        if not isinstance(tasks, dict):
            return None
        out[str(name)] = {str(t): [bool(x) for x in v] for t, v in tasks.items()}
    return out or None


def scan(root=".", with_rejects=False):
    """Walk `root` and return the Sources found, richest first.

    With `with_rejects`, also return the files that looked like evaluations and were
    not usable, each with the reason. Telling somebody their file was skipped and why
    is worth more than a clean empty result: a near miss they cannot see is the way a
    tool gets abandoned.
    """
    root = os.path.abspath(root)
    lerobot, lero_mtime, lero_files = {}, {}, []
    lero_media, lero_seeds = {}, {}
    lero_from_counts = 0
    sources, rejected = [], []
    for path in _walk(root):
        base = os.path.basename(path)
        low = base.lower()
        # Content, not filename. In the wild this file is called eval_act_50k.json,
        # smoke_eval_info.json, eval_100K.json or eval_info-checkpoint.json at least as
        # often as eval_info.json, and matching the name threw away most of them.
        if low.endswith(".json"):
            vids, flags, sds = {}, {}, {}
            got = _read_lerobot(path, media_out=vids, flags_out=flags, seeds_out=sds)
            if got and flags.get("reconstructed"):
                lero_from_counts += 1
            if got:
                name = candidate_name(path, root)
                slot = lerobot.setdefault(name, {})
                mslot = lero_media.setdefault(name, {})
                for k, v in vids.items():
                    mslot.setdefault(k, v)
                sslot = lero_seeds.setdefault(name, {})
                for k, v in sds.items():
                    sslot.setdefault(k, v)
                for k, v in got.items():
                    if k not in slot:          # first file wins; duplicates are noted
                        slot[k] = v
                try:
                    lero_mtime[name] = max(lero_mtime.get(name, 0.0), os.path.getmtime(path))
                except OSError:
                    pass
                lero_files.append(path)
            if got:
                continue
        if low.endswith((".csv", ".tsv")):
            res = _read_csv(path)
            got, note, paired_ok = res[0], res[1], res[2]
            reasons = res[3] if len(res) > 3 else None
            if got:
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    mt = 0.0
                sources.append(Source("csv", path, got, {c: mt for c in got}, note,
                                      paired_ok=paired_ok, reasons=reasons))
            elif note:
                rejected.append((path, note))
            continue
        if low.endswith(".json"):
            got = _read_matrix(path)
            if got is None and low != "eval_info.json" and _maybe_eval_json(path):
                rejected.append((path, "JSON without per_task successes or a "
                                       "candidates matrix"))
            if got:
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    mt = 0.0
                # a hand-built matrix is an explicit claim that column k of every
                # row is the same episode, so it is pairable by construction
                sources.append(Source("matrix", path, got, {c: mt for c in got},
                                      paired_ok=True))
    if lerobot:
        note = "%d eval_info.json" % len(lero_files)
        if lero_from_counts:
            # Trials rebuilt from a rate look perfectly aligned and are not: the
            # individual episodes are gone, so no file here may be paired.
            note += ", %d rebuilt from counts (pc_success and n_episodes only)" % lero_from_counts
        sources.insert(0, Source("lerobot", root, lerobot, lero_mtime, note,
                                 paired_ok=not lero_from_counts, media=lero_media,
                                 seeds=lero_seeds))
    sources.sort(key=lambda s: (-len(s.data), -s.n_skills))
    return (sources, rejected) if with_rejects else sources


def common_skills(data):
    """Skills every candidate shares."""
    sets = [set(v) for v in data.values()]
    return sorted(set.intersection(*sets)) if sets else []


def comparable_group(data):
    """The largest group of policies that actually ran the same jobs.

    Requiring EVERY policy to share EVERY job is how a real directory becomes
    "nothing found": one run on a different benchmark, or one smoke test on a single
    task, empties the intersection for everybody else. A directory with eight
    policies where six ran LIBERO-Goal has an obvious comparison in it and the tool
    should find it rather than shrug.

    Returns (policies, skills, dropped) with dropped naming what was left out, so the
    choice is visible instead of silent.
    """
    if not data:
        return [], [], []
    best = None
    for probe in data.values():
        S = set(probe)
        if not S:
            continue
        members = sorted(c for c, v in data.items() if S.issubset(set(v)))
        if len(members) < 2:
            continue
        score = (len(members) * len(S), len(members), len(S))
        if best is None or score > best[0]:
            best = (score, members, sorted(S))
    if best is None:                       # nothing shares anything: fall back to all
        return sorted(data), common_skills(data), []
    _score, members, skills = best
    dropped = sorted(set(data) - set(members))
    return members, skills, dropped


def trim_to_common(data, skills=None):
    """Restrict to shared skills and to the shortest episode count per skill."""
    skills = skills or common_skills(data)
    n = {s: min(len(data[c][s]) for c in data) for s in skills}
    return {c: {s: data[c][s][:n[s]] for s in skills} for c in data}


def order_by_time(source):
    """Candidates oldest first, which is the order a release history is read in."""
    return sorted(source.data, key=lambda c: (source.mtimes.get(c, 0.0), c))
