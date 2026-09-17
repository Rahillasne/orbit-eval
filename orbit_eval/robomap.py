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

"""The map: what has been measured, on which body, with how many seeds.

Three axes. A job shape rather than a task name, because names do not
transfer and shapes do. An embodiment. A model family. Each cell holds the
pooled success rate, the trials behind it, the interval, how many independent
sources measured it, and how many independent retrains stand behind the
number, which is the column no other board carries: a cell measured on one
training seed says so, because one seed is one draw of a 20-point lottery.

It is a map, not a leaderboard. It reports measurement status, never rank, so
it cannot be gamed and it has nothing to say about which model is best. Most
of it is empty, and an empty cell is the honest answer and the reason to run
the measurement. Every cell names its files.

The data ships with the package as `map/map.json`, built from the banked
corpus by `research/map/build_map.py` in the research repository. Nothing in
this module computes a statistic beyond the pooled rate and its interval.
"""

import json
import math
import os

MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map")
MAP_FILE = os.path.join(MAP_DIR, "map.json")

JOB_SHAPES = ("pick_place", "insertion", "sort", "push", "articulated",
              "deformable", "long_horizon", "pour", "unknown")
STATUSES = ("UNMEASURED", "ONE_SEED", "ONE_SOURCE", "MEASURED")

Z975 = 1.959964


def load(path=MAP_FILE):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"generated": None, "cells": [], "bodies": {}, "models": {}, "skills": []}


def status_of(n_trials, n_seeds, n_sources):
    """The one rule. UNMEASURED beats ONE_SEED beats ONE_SOURCE beats MEASURED."""
    if not n_trials:
        return "UNMEASURED"
    if (n_seeds or 0) <= 1:
        return "ONE_SEED"
    if (n_sources or 0) <= 1:
        return "ONE_SOURCE"
    return "MEASURED"


def wilson(k, n, z=Z975):
    if not n:
        return None
    p = k / float(n)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [round(100.0 * max(0.0, centre - half), 1), round(100.0 * min(1.0, centre + half), 1)]


def bodies_for(m, robot_id):
    """The map body ids that stand for one database robot, itself included."""
    ids = {robot_id}
    for bid, info in (m.get("bodies") or {}).items():
        if isinstance(info, dict) and info.get("db") == robot_id:
            ids.add(bid)
    return ids


def cell(m, job_shape=None, embodiment=None, model=None):
    """Every cell matching the given axes (None matches anything).

    `embodiment` may be a database robot id; simulated bodies that stand in
    for that robot (the LIBERO Panda for the FR3) are included and labelled.
    """
    out = []
    bodies = bodies_for(m, embodiment) if embodiment else None
    for c in m.get("cells", []):
        if job_shape and c.get("job_shape") != job_shape:
            continue
        if bodies and c.get("embodiment") not in bodies:
            continue
        if model and c.get("model") != model:
            continue
        out.append(c)
    order = {s: i for i, s in enumerate(STATUSES)}
    out.sort(key=lambda c: (-order.get(c.get("status"), 0), -(c.get("n_trials") or 0)))
    return out


def skills_for(m, embodiment=None, job_shape=None):
    out = []
    for s in m.get("skills", []):
        if embodiment and s.get("embodiment") != embodiment:
            continue
        if job_shape and s.get("job_shape") != job_shape:
            continue
        out.append(s)
    return out


def body_name(m, bid):
    info = (m.get("bodies") or {}).get(bid)
    if isinstance(info, dict):
        return info.get("name") or bid
    return info or bid


def summary(m):
    cells = m.get("cells", [])
    by = {s: 0 for s in STATUSES}
    for c in cells:
        by[c.get("status", "UNMEASURED")] = by.get(c.get("status", "UNMEASURED"), 0) + 1
    bodies = sorted(set(c.get("embodiment") for c in cells) | set(m.get("bodies", {})))
    models = sorted(set(c.get("model") for c in cells) | set(m.get("models", {})))
    shapes = [s for s in JOB_SHAPES if s != "unknown"]
    possible = len(bodies) * len(models) * len(shapes)
    return {"cells": len(cells), "by_status": by, "bodies": bodies, "models": models,
            "job_shapes": shapes, "possible": possible,
            "empty_fraction": (1.0 - len(cells) / possible) if possible else None,
            "skills": len(m.get("skills", []))}


def format_cells(cells, width=72):
    """A short table: status, model, trials, seeds, sources, rate with interval."""
    if not cells:
        return ["nobody has measured this yet"]
    L = ["%-11s %-10s %7s %6s %8s  %s" % ("status", "model", "trials", "seeds", "sources", "rate")]
    for c in cells:
        rate = ("%.0f%% [%.0f, %.0f]" % (c["sr"], c["ci95"][0], c["ci95"][1])
                if c.get("sr") is not None and c.get("ci95") else "")
        if not rate and c.get("sigma_run") is not None:
            rate = "variance only: sigma_run %.1f pts over %d retrains" % (
                c["sigma_run"], (c.get("sigma_df") or 0) + 1)
        L.append("%-11s %-10s %7s %6s %8s  %s%s" % (
            c["status"], (c.get("model") or "")[:10], c.get("n_trials") or 0,
            c.get("n_seeds") or 0, c.get("n_sources") or 0, rate,
            "  sim" if c.get("sim_or_real") == "sim" else ""))
    return L
