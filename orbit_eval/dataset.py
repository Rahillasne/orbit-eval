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

"""Defects a recording has, found in its metadata alone.

A LeRobot dataset keeps per-feature minimum, maximum, mean and standard
deviation in `meta/stats.json`, and the joint names in `meta/info.json`. Both
are plain JSON in the v2.x and v3.0 layouts, and together they come to a few
kilobytes. That is enough to find the faults that get read as policy failures:
a servo that never moved, a command clipped at the limit of its range, an arm
that was told to go somewhere it never reached.

Nothing here opens a parquet file, decodes a video, or touches the network, so
the checks run in the time it takes to read one small file and they add no
dependency to a package that has none.

What this module will not do is guess. Every finding names the joint, says what
was measured, and says what to do about it. A dataset that passes is not a
dataset that will train well, and the report says so in those words.
"""

import json
import os


# LeRobot's motor bus clamps every raw encoder reading to the range recorded at
# calibration time and then maps that range to -100..100 for the arm joints and
# 0..100 for the gripper (motors_bus.py, MotorsBus._normalize; verified against
# 0.5.0 and main at 30074f7, see validation/NORMALISATION_BOUND_2026-09-17.md).
# The bound is the calibration, not a per-dataset rescale, so a recorded command
# sitting exactly on it was clipped on the way out. The gripper is exempt: its
# two bounds are the calibrated closed and open positions, and reaching them is
# what a gripper does.
LIMIT = 100.0
LIMIT_EPS = 0.5

# Below this fraction of the typical joint's travel, a joint did not move.
DEAD_RANGE_FRAC = 0.005
DEAD_STD_ABS = 1e-9

# A command that overshoots what the arm reached by more than half of that
# joint's own travel is a tracking failure, not a demonstration.
FOLLOW_FRAC = 0.5

# LeRobot's own agent guide starts users at fifty episodes.
FEW_EPISODES = 50

# Only interpret the -100..100 bound when the numbers are plainly on that
# scale. Radians, metres and raw encoder counts are all left alone.
NORMALISED_MIN_SPAN = 20.0

STATE = "observation.state"
ACTION = "action"

# Two public datasets' metadata ship with the package (demo/SOURCES.md) so the
# tool can be tried on a machine that has no data on it yet.
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
DEMOS = {
    "status": ("so101_pickplace", "lerobot/svla_so101_pickplace",
               "the official SO-101 example, 50 episodes, metadata only"),
    "cover": ("so101_table_cleanup", "youliangtan/so101-table-cleanup",
              "a community SO-101 recording, 80 episodes over four "
              "instructions, metadata only"),
    "check": ("revolve_loop", "REVOLVE, arXiv 2609.14633, Table II",
              "a published self-improving loop: four real tasks on a ViperX arm, "
              "five iterations, 100 rollouts per task per iteration"),
}


def demo_path(command):
    """The bundled dataset `orbit <command> --demo` reads, and its provenance."""
    folder, hub_id, blurb = DEMOS[command]
    return os.path.join(DEMO_DIR, folder), hub_id, blurb

SKIP_DIRS = {".git", ".hg", "node_modules", "__pycache__", ".venv", "venv",
             ".tox", ".mypy_cache", ".pytest_cache", "wandb", "outputs"}
MAX_DEPTH = 6


class Finding(object):
    """One defect, named, measured, and paired with the thing to do about it."""

    __slots__ = ("code", "severity", "joint", "message", "detail", "fix", "short")

    def __init__(self, code, severity, message, fix, joint=None, detail=None,
                 short=None):
        self.code = code
        self.severity = severity      # "blocking" or "advisory"
        self.joint = joint
        self.message = message
        # The same fact with the joint name taken out, so a sentence that has
        # already named the joint does not name it twice.
        self.short = short or message
        self.detail = detail or ""
        self.fix = fix

    def as_dict(self):
        return {"code": self.code, "severity": self.severity, "joint": self.joint,
                "message": self.message, "short": self.short,
                "detail": self.detail, "fix": self.fix}

    def __repr__(self):
        return "Finding(%s, %s, %r)" % (self.code, self.severity, self.message)


class DatasetMeta(object):
    """What `meta/info.json` and `meta/stats.json` say about one recording."""

    def __init__(self, root, info, stats):
        self.root = root
        self.info = info or {}
        self.stats = stats or {}

    @property
    def n_episodes(self):
        return _int(self.info.get("total_episodes"))

    @property
    def n_frames(self):
        return _int(self.info.get("total_frames"))

    @property
    def fps(self):
        return _int(self.info.get("fps"))

    @property
    def robot_type(self):
        return self.info.get("robot_type")

    @property
    def version(self):
        return self.info.get("codebase_version")

    @property
    def joint_names(self):
        feats = self.info.get("features") or {}
        for key in (STATE, ACTION):
            names = (feats.get(key) or {}).get("names")
            if isinstance(names, list) and names and all(
                    isinstance(n, str) for n in names):
                return list(names)
        n = self._width()
        return ["joint%d" % i for i in range(n)] if n else []

    @property
    def seconds(self):
        if self.n_frames and self.fps:
            return self.n_frames / float(self.fps)
        return None

    def block(self, key):
        b = self.stats.get(key)
        return b if isinstance(b, dict) else None

    def _width(self):
        for key in (STATE, ACTION):
            b = self.block(key)
            if b and isinstance(b.get("min"), list):
                return len(b["min"])
        return 0


# ------------------------------------------------------------------ loading

def find(path, max_depth=MAX_DEPTH):
    """The nearest directory at or below `path` that holds `meta/info.json`."""
    path = os.path.abspath(path)
    if _is_dataset(path):
        return path
    base_depth = path.rstrip(os.sep).count(os.sep)
    best = None
    for dirpath, dirnames, _files in os.walk(path):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        if dirpath.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
            continue
        if _is_dataset(dirpath):
            if best is None or len(dirpath) < len(best):
                best = dirpath
            dirnames[:] = []
    return best


def load(path):
    """Read one dataset's metadata. `path` may be the root or its `meta/` dir."""
    root = _root_of(path)
    if root is None:
        raise ValueError("no LeRobot dataset metadata under %s "
                         "(looked for meta/info.json)" % os.path.abspath(path))
    info = _read_json(os.path.join(root, "meta", "info.json")) or {}
    stats = _read_json(os.path.join(root, "meta", "stats.json"))
    if stats is None:
        stats = _read_stats_jsonl(os.path.join(root, "meta", "episodes_stats.jsonl"))
    return DatasetMeta(root, info, stats or {})


def _is_dataset(d):
    return os.path.isfile(os.path.join(d, "meta", "info.json"))


def _root_of(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if _is_dataset(path):
        return path
    parent = os.path.dirname(path)
    if os.path.basename(path) == "meta" and _is_dataset(parent):
        return parent
    return find(path)


def _read_json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _read_stats_jsonl(p):
    """v2.1 keeps per-episode stats; pool them into one dataset-level block."""
    rows = []
    try:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        return None
    if not rows:
        return None
    pooled = {}
    for row in rows:
        blocks = row.get("stats") if isinstance(row.get("stats"), dict) else row
        if not isinstance(blocks, dict):
            continue
        for key, block in blocks.items():
            if not isinstance(block, dict) or "min" not in block:
                continue
            acc = pooled.setdefault(key, {"min": None, "max": None,
                                          "mean": None, "std": None})
            acc["min"] = _elementwise(acc["min"], block.get("min"), min)
            acc["max"] = _elementwise(acc["max"], block.get("max"), max)
            acc["mean"] = _elementwise(acc["mean"], block.get("mean"), _keep_first)
            acc["std"] = _elementwise(acc["std"], block.get("std"), max)
    return pooled or None


def _keep_first(a, _b):
    return a


def _elementwise(acc, new, fn):
    new = _vec(new)
    if new is None:
        return acc
    if acc is None:
        return list(new)
    out = list(acc)
    for i in range(min(len(out), len(new))):
        a, b = out[i], new[i]
        out[i] = b if a is None else (a if b is None else fn(a, b))
    return out


# ------------------------------------------------------------------- checks

def checks(meta):
    """Every defect the metadata can support, worst first."""
    found = []
    found.extend(_nan_checks(meta))
    found.extend(_joint_checks(meta))
    found.extend(_size_checks(meta))
    order = {"blocking": 0, "advisory": 1}
    return sorted(found, key=lambda f: (order.get(f.severity, 2), f.code,
                                        f.joint or ""))


def _nan_checks(meta):
    bad = []
    for key in (STATE, ACTION):
        block = meta.block(key)
        if not block:
            continue
        for field in ("min", "max", "mean", "std"):
            vec = _vec(block.get(field))
            if vec is None:
                continue
            for i, v in enumerate(vec):
                if v is None or v != v:
                    bad.append((key, field, i))
    if not bad:
        return []
    names = meta.joint_names
    where = ", ".join("%s %s[%s]" % (k, f, _name(names, i)) for k, f, i in bad[:4])
    return [Finding(
        "nan_stats", "blocking",
        "the recorded statistics contain a value that is not a number (%s)" % where,
        "re-record the affected episodes, or delete them and rebuild the "
        "dataset; a NaN here usually means a dropped servo read",
        detail="%d value%s affected" % (len(bad), "" if len(bad) == 1 else "s"))]


def _joint_checks(meta):
    state = meta.block(STATE)
    action = meta.block(ACTION)
    if not state:
        return []
    names = meta.joint_names
    s_min, s_max = _vec(state.get("min")), _vec(state.get("max"))
    s_std = _vec(state.get("std"))
    if s_min is None or s_max is None:
        return []
    n = min(len(s_min), len(s_max))
    ranges = [_span(s_min[i], s_max[i]) for i in range(n)]
    typical = _median([r for r in ranges if r is not None and r > 0])
    found = []

    for i in range(n):
        span = ranges[i]
        if span is None:
            continue
        std = s_std[i] if s_std and i < len(s_std) else None
        dead = False
        if typical and span <= DEAD_RANGE_FRAC * typical:
            dead = True
        elif std is not None and std == std and std <= DEAD_STD_ABS:
            dead = True
        if dead:
            found.append(Finding(
                "dead_joint", "blocking",
                "%s never moved in this recording" % _name(names, i),
                "check the servo, the cable and the calibration for that joint, "
                "then re-record; a policy trained on this will never move it either",
                joint=_name(names, i), short="it never moved in this recording",
                detail="range %.3f over the whole dataset" % span))

    if action:
        found.extend(_action_checks(meta, state, action, names, ranges))
    return found


def _action_checks(meta, state, action, names, s_ranges):
    a_min, a_max = _vec(action.get("min")), _vec(action.get("max"))
    s_min, s_max = _vec(state.get("min")), _vec(state.get("max"))
    if a_min is None or a_max is None or s_min is None or s_max is None:
        return []
    n = min(len(a_min), len(a_max), len(s_min), len(s_max))
    found = []

    if _looks_normalised(a_min, a_max, s_min, s_max):
        for i in range(n):
            lo, hi = a_min[i], a_max[i]
            if lo is None or hi is None or lo != lo or hi != hi:
                continue
            name = _name(names, i)
            if "gripper" in name.lower():
                continue
            hit = None
            if hi >= LIMIT - LIMIT_EPS:
                hit = "+%g" % LIMIT
            elif lo <= -LIMIT + LIMIT_EPS:
                hit = "-%g" % LIMIT
            if hit:
                found.append(Finding(
                    "saturated_action", "blocking",
                    "the command to %s was clipped at the end of its range" % name,
                    "re-calibrate that joint, or move the workspace so the task "
                    "fits inside the arm's reach; every clipped frame teaches "
                    "the policy a command the arm cannot execute",
                    joint=name,
                    short="the command was clipped at the end of its range",
                    detail="commanded %s, the limit of the normalised range" % hit))

    for i in range(n):
        span = s_ranges[i] if i < len(s_ranges) else None
        if not span or span <= 0:
            continue
        over = _overshoot(a_min[i], a_max[i], s_min[i], s_max[i])
        if over is not None and over > FOLLOW_FRAC * span:
            name = _name(names, i)
            if any(f.joint == name and f.code == "saturated_action" for f in found):
                continue
            found.append(Finding(
                "not_following", "blocking",
                "%s was commanded well past where it actually went" % name,
                "the joint is stalling, mis-calibrated, or carrying more load "
                "than it can hold; fix the arm before you read anything into "
                "a success rate from this data",
                joint=name,
                short="it was commanded well past where it actually went",
                detail="commanded %.1f beyond the reached range of %.1f"
                       % (over, span)))
    return found


def _size_checks(meta):
    found = []
    n = meta.n_episodes
    if n is not None and 0 < n < FEW_EPISODES:
        found.append(Finding(
            "few_episodes", "advisory",
            "%d episodes recorded" % n,
            "LeRobot's own guide starts at %d for a single task; record %d more "
            "before reading much into a result" % (FEW_EPISODES, FEW_EPISODES - n),
            detail="%d of the %d this kind of task usually needs"
                   % (n, FEW_EPISODES)))
    return found


# ------------------------------------------------------------------ helpers

def _looks_normalised(*vecs):
    """True when the numbers are plainly on the -100..100 servo scale."""
    span = 0.0
    for vec in vecs:
        for v in vec or []:
            if v is None or v != v:
                continue
            if abs(v) > LIMIT + LIMIT_EPS:
                return False
            span = max(span, abs(v))
    return span >= NORMALISED_MIN_SPAN


def _overshoot(a_lo, a_hi, s_lo, s_hi):
    vals = [a_lo, a_hi, s_lo, s_hi]
    if any(v is None or v != v for v in vals):
        return None
    return max(a_hi - s_hi, s_lo - a_lo, 0.0)


def _span(lo, hi):
    if lo is None or hi is None or lo != lo or hi != hi:
        return None
    return abs(hi - lo)


def _vec(v):
    if isinstance(v, list):
        if v and isinstance(v[0], list):      # image stats arrive nested
            return [x[0] if isinstance(x, list) and x else None for x in v]
        return v
    return None


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _name(names, i):
    if names and 0 <= i < len(names):
        return names[i]
    return "joint%d" % i
