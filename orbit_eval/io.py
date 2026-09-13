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

"""Parsers for the two eval-result formats found in the ORBIT repo.

(1) LeRobot ``eval_info.json``::

      {"per_task":  [{"task_group", "task_id",
                      "metrics": {"sum_rewards": [...], "max_rewards": [...],
                                  "successes": [bool, ...], ...}}, ...],
       "per_group": {...},
       "overall":   {"pc_success": float, "n_episodes": int, ...},
       "seed": int  # only if the harness recorded it; stock LeRobot does NOT}

    NOTE: stock LeRobot does not write the eval seed into eval_info.json, and
    its silent default is seed=1000 — every default eval scores initial states
    1000..1000+n-1 (common random numbers whether you asked for them or not).

(2) ORBIT ``model_result.json`` (one per run; waves also concatenate them into
    ``*_results.json`` lists)::

      {"mask_id", "episodes", "seed", "set_hash", "arm", "sr",
       "final_step", "design_steps", "n_eval", "budget_ok"}
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from . import stats


@dataclass
class EvalRun:
    run_id: str
    fmt: str                       # "lerobot" | "orbit"
    path: Optional[str] = None
    sr: Optional[float] = None     # pp, 0..100
    n_eval: Optional[int] = None
    seed: Optional[int] = None
    successes: Optional[List[bool]] = None
    sum_rewards: Optional[List[float]] = None
    max_rewards: Optional[List[float]] = None
    set_hash: Optional[str] = None
    arm: Optional[str] = None
    final_step: Optional[int] = None
    design_steps: Optional[int] = None
    budget_ok: Optional[bool] = None
    episodes: Optional[list] = None
    task_groups: List[str] = field(default_factory=list)
    # set when a LeRobot file's per-episode successes vector disagreed with
    # its own overall header (n_episodes / pc_success): the vector was DROPPED
    # so no paired statistic can run on episodes that do not correspond to the
    # reported n_eval/SR; the message says what disagreed. audit surfaces it
    # as SR_VECTOR_MISMATCH.
    sr_vector_mismatch: Optional[str] = None

    @property
    def is_libero(self):
        return any("libero" in (g or "").lower() for g in self.task_groups)


def parse_eval_info(data, path=None):
    """LeRobot eval_info.json dict -> EvalRun."""
    overall = data.get("overall", {}) or {}
    successes, sums, maxs, groups = [], [], [], []
    for pt in data.get("per_task", []) or []:
        m = pt.get("metrics", {}) or {}
        if m.get("successes") is not None:
            successes.extend(bool(s) for s in m["successes"])
        if m.get("sum_rewards") is not None:
            sums.extend(m["sum_rewards"])
        if m.get("max_rewards") is not None:
            maxs.extend(m["max_rewards"])
        if pt.get("task_group"):
            groups.append(str(pt["task_group"]))
    seed = data.get("seed", data.get("eval_seed"))
    if seed is None and isinstance(data.get("args"), dict):
        seed = data["args"].get("seed")
    n_hdr = overall.get("n_episodes")
    sr_hdr = overall.get("pc_success")
    # Header/vector consistency (2026-08-21). A file where some per_task
    # blocks lack 'successes' (or whose header disagrees with its own
    # episodes) would otherwise yield an EvalRun whose sr/n come from the
    # header while the successes vector is shorter/different — and every
    # paired statistic downstream would run McNemar on episodes that do not
    # correspond to the reported n_eval or SR. Engine pre-check 11b already
    # recomputes SR from the vector citing exactly this hazard; here the
    # mismatched vector is DROPPED at parse time (header kept, reason
    # recorded) so no consumer can trust it by accident.
    mismatch = None
    if successes:
        if n_hdr is not None and len(successes) != int(n_hdr):
            mismatch = (
                "per-episode successes vector has %d entries but "
                "overall.n_episodes says %d — the vector is partial or the "
                "header is wrong; vector dropped, header kept"
                % (len(successes), int(n_hdr)))
        elif sr_hdr is not None:
            sr_vec = 100.0 * sum(successes) / len(successes)
            if abs(sr_vec - float(sr_hdr)) > 0.1:
                mismatch = (
                    "per-episode successes give SR %.2f pp but "
                    "overall.pc_success says %.2f pp — the file is "
                    "internally inconsistent; vector dropped, header kept"
                    % (sr_vec, float(sr_hdr)))
    if mismatch:
        successes = []
    n = n_hdr
    if n is None and successes:
        n = len(successes)
    sr = sr_hdr
    if sr is None and successes:
        sr = 100.0 * sum(successes) / len(successes)
    run_id = _run_id_from_path(path) if path else "eval_info"
    return EvalRun(
        run_id=run_id, fmt="lerobot", path=path,
        sr=float(sr) if sr is not None else None,
        n_eval=int(n) if n is not None else None,
        seed=int(seed) if seed is not None else None,
        successes=successes or None,
        sum_rewards=sums or None, max_rewards=maxs or None,
        task_groups=groups,
        sr_vector_mismatch=mismatch,
    )


def parse_model_result(rec, path=None):
    """ORBIT model_result.json record -> EvalRun."""
    rid = str(rec.get("mask_id") or rec.get("run_id")
              or (_run_id_from_path(path) if path else "model_result"))
    rid = rid.split("/")[-1]
    eps = rec.get("episodes")
    sh = rec.get("set_hash")
    if sh is None and isinstance(eps, list) and eps:
        sh = stats.set_hash(eps)   # build_atlas.py convention
    fs = rec.get("final_step")
    ds = rec.get("design_steps")
    bok = rec.get("budget_ok")
    if bok is None and fs is not None and ds is not None:
        bok = int(fs) == int(ds)
    return EvalRun(
        run_id=rid, fmt="orbit", path=path,
        sr=float(rec["sr"]) if rec.get("sr") is not None else None,
        n_eval=int(rec["n_eval"]) if rec.get("n_eval") is not None else None,
        seed=int(rec["seed"]) if rec.get("seed") is not None else None,
        set_hash=sh, arm=rec.get("arm"),
        final_step=int(fs) if fs is not None else None,
        design_steps=int(ds) if ds is not None else None,
        budget_ok=bool(bok) if bok is not None else None,
        episodes=eps,
    )


def parse_file(path):
    """Parse one JSON file -> list[EvalRun] (a *_results.json holds many)."""
    with open(path) as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError("%s: invalid JSON: %s" % (path, e))
    if isinstance(data, list):
        return [parse_model_result(r, path) for r in data
                if isinstance(r, dict) and ("sr" in r or "mask_id" in r)]
    if isinstance(data, dict):
        if "per_task" in data or "overall" in data:
            return [parse_eval_info(data, path)]
        if "sr" in data or "mask_id" in data:
            return [parse_model_result(data, path)]
    raise ValueError("%s: not a recognisable eval_info.json / model_result.json"
                     % path)


RESULT_SUFFIXES = ("_results.json",)


def _is_result_file(name):
    # matches eval_info.json AND LeRobot-style <run>_eval_info.json,
    # model_result.json, corpus-style <wave>__model_result.json, our
    # model_result_<id>.json, *_results.json
    return (name.endswith("eval_info.json") or "model_result" in name
            or any(name.endswith(s) for s in RESULT_SUFFIXES))


def discover(root):
    """Recursively yield (path, [EvalRun...] or None, error) for every
    recognisable result file under root."""
    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        for name in sorted(filenames):
            if not _is_result_file(name):
                continue
            p = os.path.join(dirpath, name)
            try:
                yield p, parse_file(p), None
            except (ValueError, json.JSONDecodeError, OSError) as e:
                yield p, None, str(e)


def load_single(path):
    """Load exactly one run from a file or directory (for compare/regress)."""
    if os.path.isdir(path):
        found = [(p, runs) for p, runs, err in discover(path) if runs]
        runs = [r for _p, rs in found for r in rs]
        if not runs:
            raise ValueError("%s: no eval_info.json / model_result.json found"
                             % path)
        if len(runs) > 1:
            raise ValueError(
                "%s: %d runs found, need exactly one — point at the file "
                "itself (%s...)" % (path, len(runs),
                                    ", ".join(r.run_id for r in runs[:4])))
        return runs[0]
    runs = parse_file(path)
    if len(runs) != 1:
        raise ValueError("%s: holds %d runs, need exactly one"
                         % (path, len(runs)))
    return runs[0]


def _run_id_from_path(path):
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    parts = [p for p in parts if p]
    tail = parts[-3:] if len(parts) >= 3 else parts
    if tail and tail[-1].endswith(".json"):
        tail[-1] = tail[-1][:-len(".json")]
    return "/".join(tail)
