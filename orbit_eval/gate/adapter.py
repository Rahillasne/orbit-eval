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

"""LeRobot training-output directory adapter for the shipgate CLI.

``lerobot-train`` writes one output tree per run::

    <out_dir>/
      train_config.json
      checkpoints/<step>/...              # periodic checkpoints; a harness
                                          #   that evals at save time leaves
                                          #   an eval_info.json in each
      eval_<step>/eval_info.json          # periodic eval snapshots
      eval_final/eval_info.json           # the final eval, when the run ran
                                          #   to completion

The v1 product contract: ``shipgate check`` positional args may be such
OUTPUT DIRS directly — the adapter locates the authoritative eval so the
user never hand-digs the tree:

  1. ``eval_final/eval_info.json`` when present (the run's declared final
     eval beats any snapshot);
  2. else the NEWEST ``eval*/eval_info.json`` — newest = the largest
     integer embedded in the eval directory name (``eval_020000`` beats
     ``eval_010000``; a digitless name sorts oldest; ties break on the
     directory name, then the path — fully deterministic, no mtime, which
     a copy/rsync would scramble).

``--history`` may likewise be a run dir whose ``checkpoints/<step>/`` tree
holds one eval_info.json per checkpoint: every one is loaded (ascending
step) as the selection pool for the history-bootstrap winner's-curse
measurement. The pool contract is the engine's (engine.gate): it must be
the ACTUAL selection pool including the candidate's own eval — a recorded
warning fires when no pool vector matches the candidate's.

``shipgate capture`` uses the HARVEST form (:func:`find_run_evals`): EVERY
``eval*/eval_info.json`` under a (possibly multi-run) tree, grouped by run
dir — where :func:`find_output_eval` picks the one authoritative eval for
gating, capture wants the whole checkpoint-eval history of every run.

Discovery reuses :func:`orbit_eval.io.discover` — one file-walking
convention, not two. Parse failures on a located file RAISE (a silently
dropped checkpoint would shrink the selection pool and underestimate the
curse; a silently skipped final eval would gate the wrong snapshot); the
harvest form instead RETURNS the error verbatim — a capture sweep must
disclose a corrupt snapshot, not sink the whole wave on it.
"""

import os
import re
from collections import defaultdict

from .. import io

_DIGITS_RE = re.compile(r"(\d+)")


def _step_key(name):
    """'Newest' ordering key for an eval/checkpoint directory name: the
    LARGEST integer embedded in it (``eval_020000`` -> 20000, ``0005000``
    -> 5000); a digitless name -> -1, sorting oldest. The last digit group
    wins so ``eval2_010000``-style prefixes cannot shadow the step."""
    nums = _DIGITS_RE.findall(name)
    return int(nums[-1]) if nums else -1


def _eval_candidates(root):
    """Every discovered ``eval_info.json`` sitting DIRECTLY in an ``eval*``
    directory under root -> list of (path, parent_dir_name, runs, error).
    io.discover does the walking and the parsing; this only filters."""
    out = []
    for p, runs, err in io.discover(root):
        if not os.path.basename(p).endswith("eval_info.json"):
            continue
        parent = os.path.basename(os.path.dirname(p))
        if parent.startswith("eval"):
            out.append((p, parent, runs, err))
    return out


def find_output_eval(out_dir):
    """LeRobot training OUTPUT DIR -> the :class:`orbit_eval.io.EvalRun` of
    its authoritative eval, or None when the tree holds no
    ``eval*/eval_info.json`` at all — the caller then falls back to the
    plain single-file loaders, so a directory holding exactly one result
    file keeps its pre-adapter behaviour byte for byte.

    Selection order (see the module docstring): ``eval_final`` beats the
    newest ``eval*`` snapshot. More than one ``eval_final/eval_info.json``
    under the tree means the path spans SEVERAL runs' output dirs — that is
    ambiguous and raises ValueError (silently picking one would gate a run
    the caller never named). A located file that fails to parse, or that
    holds anything but exactly one run, also raises."""
    cands = _eval_candidates(out_dir)
    if not cands:
        return None
    finals = [c for c in cands if c[1] == "eval_final"]
    if len(finals) > 1:
        raise ValueError(
            "%s: %d eval_final/eval_info.json files under this path (%s) — "
            "the directory spans several runs' output trees; point at ONE "
            "run's output dir"
            % (out_dir, len(finals),
               ", ".join(sorted(c[0] for c in finals)[:4])))
    if finals:
        path, _parent, runs, err = finals[0]
    else:
        path, _parent, runs, err = max(
            cands, key=lambda c: (_step_key(c[1]), c[1], c[0]))
    if err:
        raise ValueError("%s: chose %s but it does not parse: %s"
                         % (out_dir, path, err))
    if not runs or len(runs) != 1:
        raise ValueError("%s: chose %s but it holds %d runs, need exactly "
                         "one" % (out_dir, path, len(runs or [])))
    return runs[0]


def find_run_evals(root):
    """EVERY eval snapshot under a (possibly multi-run) LeRobot output tree,
    grouped by run dir — the harvest form of the discovery, for
    ``shipgate capture``.

    Returns ``[(run_dir, [(eval_name, step, path, runs, err), ...]), ...]``:
    run dirs sorted; within a run the evals in ascending step order with
    ``eval_final`` LAST (chronologically the run's last eval; its digitless
    name carries no embedded step, so ``step`` is None there — and for any
    other digitless eval dir). ``runs``/``err`` come from io.discover
    verbatim, one entry per located file whether it parsed or not: the
    CALLER decides what a failure means (capture counts it unreadable and
    keeps going; gating via :func:`find_output_eval` still raises). An
    empty list means the tree holds no ``eval*/eval_info.json`` at all."""
    by_run = defaultdict(list)
    for p, parent, runs, err in _eval_candidates(root):
        by_run[os.path.dirname(os.path.dirname(p))].append(
            (parent, p, runs, err))
    out = []
    for run_dir in sorted(by_run):
        evals = []
        for name, p, runs, err in by_run[run_dir]:
            k = _step_key(name)
            evals.append((name, k if k >= 0 else None, p, runs, err))
        evals.sort(key=lambda e: (e[0] == "eval_final",
                                  e[1] if e[1] is not None else -1,
                                  e[0], e[2]))
        out.append((run_dir, evals))
    return out


def load_history_dir(run_dir):
    """Run dir with a ``checkpoints/`` subtree -> the ordered checkpoint
    eval pool for the history-bootstrap curse: every
    ``checkpoints/<step>/**/eval_info.json``, one EvalRun each, sorted by
    ascending step (``_step_key`` of the first path component under
    checkpoints/, ties by name then path).

    Returns None when ``run_dir`` has no ``checkpoints/`` subdirectory —
    the caller falls back to loading the path as a single eval file/dir.
    Raises ValueError when checkpoints/ exists but holds no parseable
    eval_info.json, or when ANY located checkpoint eval fails to parse: a
    silently dropped checkpoint shrinks the selection pool and
    underestimates the winner's-curse correction (engine.curse_from_history
    measures the actual selection event over the FULL pool)."""
    ck = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ck):
        return None
    entries = []
    for p, runs, err in io.discover(ck):
        if not os.path.basename(p).endswith("eval_info.json"):
            continue
        if err:
            raise ValueError("%s: checkpoint eval does not parse: %s"
                             % (run_dir, err))
        if not runs or len(runs) != 1:
            raise ValueError("%s: checkpoint eval %s holds %d runs, need "
                             "exactly one" % (run_dir, p, len(runs or [])))
        step_dir = os.path.relpath(p, ck).split(os.sep)[0]
        entries.append((_step_key(step_dir), step_dir, p, runs[0]))
    if not entries:
        raise ValueError(
            "%s: checkpoints/ holds no eval_info.json — the --history "
            "run-dir form needs checkpoints/<step>/eval_info.json per "
            "checkpoint" % run_dir)
    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    return [e[3] for e in entries]
