# orbit-eval — honest statistics for robot-policy evaluation.
# Copyright (C) 2026 ORBIT Research
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the license along with this program. If not,
# see <https://www.gnu.org/licenses/>.
#
# A commercial license, exempting you from the AGPL's source-disclosure terms, is
# available from ORBIT Research.

"""orbit-eval route — task-conditioned release selection over candidate checkpoints.

The release step between "training finished" and "this is the policy we ship" is
usually `cp checkpoint_final prod/`. This module makes it a measured decision over the
candidates the training budget already paid for (retrains, mid-training checkpoints):

  * per task, ship the candidate that is reliably better than the incumbent, and
    ABSTAIN to the incumbent otherwise (one-sided z >= Z_ABSTAIN on the selection
    episodes) — routing must never itself be a lottery;
  * report the held-out gain honestly: a stratified split-half over episodes (select on
    one half, score on the disjoint other half), the in-sample plug-in gain being
    optimistic by construction;
  * price damage above measurement noise (a task is "damaged" when its held-out SR is
    >= 5 pp + z*SE below the candidate-pool mean);
  * say what the evaluation budget can and cannot resolve, and what more would cost.

Inputs: a directory of <candidate>/**/eval_info.json (LeRobot), a JSON matrix, or a
CSV export (candidate,task,episode,success[,block]). The incumbent may be one candidate
or a per-task mapping (a fleet that already runs different models per SKU). Episodes
may carry a BLOCK label (robot, day, cell): halves are then stratified within block,
and every defection reports whether its advantage holds in every block.

Sources of truth (ported faithfully, pure stdlib):
  research/ROUTE1_RESULTS_2026-08-29.md         split-half arms, damage count, budget curves
  research/cleanrel50/law_score.py              abstention rule (Z_ABST = 1.645), PRICED damage
  research/kr1_metaworld/KR1_RESULTS_2026-08-29.md   J-curve / budget curve, second cell

Everything here assumes the candidates were evaluated under common random numbers:
the k-th episode of task t used the SAME initial state (and block) for every candidate.
The loader refuses unequal episode counts and unequal block layouts; it cannot verify
the states themselves — say so in the record (`crn_asserted_by_user`).

Success rates are percentage points (0..100).
"""

import csv
import hashlib
import json
import math
import os
import random

from . import stats

Z_ABSTAIN = 1.645     # one-sided alpha 0.05, law_score.py Z_ABST
Z_DMG = 2.0           # law_score.py Z_DMG (priced damage margin)
DMG_PP = 5.0          # law_score.py DMG_PP
N_DRAWS = 2000
RNG_SEED = 20260906
MIN_EPISODES = 10
BUDGET_TARGETS_PP = (5.0, 10.0, 15.0, 20.0)
ARMS = ("RANDOM", "INCUMBENT", "ROUTE", "ROUTE+ABSTAIN", "ORACLE")

# Measured sizing rows (ROUTE minus best-validation on held-out episodes, pp). Cell-local:
# the no-transferable-sigma law forbids quoting them for an unmeasured cell — `plan`
# prints them WITH their stamps.
MEASURED_CELLS = {
    "pi05-ft / libero_object / k=88 / sim (ROUTE-1, 2026-08-29)": {
        "budget_eps_per_task": {5: 4.24, 10: 4.58, 20: 5.00, 35: 5.47, 50: 5.78},   # J=8
        "J": {2: 2.96, 3: 4.02, 4: 4.29, 6: 4.69, 8: 5.72},                          # 50 eps/task
        "damaged_suite": 1.82, "damaged_route": 0.22,
    },
    "MLP / Meta-World MT10 / scripted demos / sim (KR-1, 2026-08-29; median of 3 replicates)": {
        "budget_eps_per_task": {5: 4.43, 10: 4.99, 25: 5.94, 50: 6.54, 100: 6.81},  # J=8
        "J": {2: 2.46, 3: 3.81, 4: 4.63, 6: 5.93, 8: 6.89},                          # 50 eps/task
        "damaged_suite": 1.40, "damaged_route": 0.00,
    },
}

# Measured REGRESSION RISK by pool size: P(>=1 truly damaged task) under the priced rule
# (>=5pp below the full retrain pool AND >=2*SE on held-out episodes), median over each
# family's replicates. DAMAGE-J, declaration sha fb1c760f, scored 2026-09-08.
# Cell-local like everything else here, but note what the SHAPE says across cells:
# best-validation selection does not reduce this risk at any J, and the pool size that
# does reduce it grows with the number of tasks. `plan` prints the rows nearest your T.
DAMAGE_J_CELLS = {
    "MLP / Meta-World MT10 / sim (DAMAGE-J, median of 3 replicates)": {
        "T": 10, "n": 200, "ship_one": 0.578,
        "best_val": {2: 0.416, 3: 0.321, 4: 0.239, 6: 0.107, 8: 0.080},
        "route": {2: 0.170, 3: 0.040, 4: 0.012, 6: 0.002, 8: 0.000},
        "route_abstain": {2: 0.212, 3: 0.075, 4: 0.039, 6: 0.009, 8: 0.000},
    },
    "MLP / Meta-World MT50 / sim (DAMAGE-J, median of 3 replicates)": {
        "T": 50, "n": 200, "ship_one": 1.000,
        "best_val": {2: 1.000, 3: 1.000, 4: 1.000, 6: 1.000, 8: 1.000},
        "route": {2: 0.840, 3: 0.405, 4: 0.082, 6: 0.003, 8: 0.000},
        "route_abstain": {2: 0.915, 3: 0.548, 4: 0.183, 6: 0.031, 8: 0.001},
    },
    "SmolVLA-450M / libero_object / sim (DAMAGE-J, 2026-09-08)": {
        "T": 10, "n": 100, "ship_one": 0.206,
        "best_val": {2: 0.195, 3: 0.188, 4: 0.167, 6: 0.141, 8: 0.129},
        "route": {2: 0.015, 3: 0.004, 4: 0.002, 6: 0.002, 8: 0.001},
        "route_abstain": {2: 0.077, 3: 0.024, 4: 0.011, 6: 0.005, 8: 0.004},
    },
    "SmolVLA-450M / libero_spatial / sim (DAMAGE-J, 2026-09-08)": {
        "T": 10, "n": 200, "ship_one": 0.109,
        "best_val": {2: 0.025, 3: 0.029, 4: 0.026, 6: 0.026, 8: 0.028},
        "route": {2: 0.022, 3: 0.015, 4: 0.012, 6: 0.011, 8: 0.011},
        "route_abstain": {2: 0.023, 3: 0.025, 4: 0.022, 6: 0.019, 8: 0.017},
    },
}

_TRUTHY = {"1", "true", "t", "yes", "y", "success", "pass", "1.0"}
_FALSY = {"0", "false", "f", "no", "n", "fail", "failure", "0.0"}


# ---------------------------------------------------------------- loading

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bool(v, where):
    s = str(v).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    raise ValueError("%s: success value %r is not 0/1/true/false" % (where, v))


def _load_csv(path):
    """CSV/TSV with columns candidate,task,episode,success and an optional block column.
    Header aliases: candidate|checkpoint|model|policy|run, task|task_id|sku|skill,
    episode|episode_id|ep|trial, success|ok|outcome|succeeded, block|robot|robot_id|day|
    date|cell|site|batch. Returns (data, blocks_by_cand)."""
    delim = "\t" if path.lower().endswith(".tsv") else ","
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=delim))
    if not rows:
        raise ValueError("%s: empty CSV" % path)
    cols = {c.lower().strip(): c for c in rows[0].keys() if c is not None}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_cand = pick("candidate", "checkpoint", "model", "policy", "run")
    c_task = pick("task", "task_id", "sku", "skill")
    c_ep = pick("episode", "episode_id", "ep", "trial")
    c_ok = pick("success", "ok", "outcome", "succeeded")
    c_blk = pick("block", "robot", "robot_id", "day", "date", "cell", "site", "batch")
    if None in (c_cand, c_task, c_ep, c_ok):
        raise ValueError("%s: need columns candidate,task,episode,success (got %s)"
                         % (path, ", ".join(rows[0].keys())))
    tmp, blk = {}, {}
    for r in rows:
        ok = _bool(r[c_ok], path)
        try:
            ep = int(float(r[c_ep]))
        except (TypeError, ValueError):
            raise ValueError("%s: episode %r is not an integer" % (path, r[c_ep]))
        cand, task = str(r[c_cand]).strip(), str(r[c_task]).strip()
        tmp.setdefault(cand, {}).setdefault(task, {})[ep] = ok
        if c_blk is not None:
            blk.setdefault(cand, {}).setdefault(task, {})[ep] = str(r[c_blk]).strip()
    data, blocks = {}, {}
    for cand, tasks in tmp.items():
        data[cand] = {}
        for t, eps in tasks.items():
            order = sorted(eps)
            data[cand][t] = [eps[k] for k in order]          # episode index order = CRN position
            if c_blk is not None:
                blocks.setdefault(cand, {})[t] = [blk[cand][t][k] for k in order]
    return data, (blocks if c_blk is not None else None)


def load_candidates(path):
    """Return (data, files, blocks_by_cand).

    data = {candidate: {task: [bool, ...]}}; blocks_by_cand = {candidate: {task: [label,
    ...]}} or None. Accepts a directory (each immediate subdirectory is one candidate;
    every eval_info.json below it contributes its per_task blocks, keyed
    "<task_group>/<task_id>"), a JSON file {"candidates": {name: {task: [0/1, ...]}},
    "blocks": {task: [label, ...]}} (blocks optional, shared by all candidates), or a
    CSV/TSV export.
    """
    if os.path.isfile(path) and path.lower().endswith((".csv", ".tsv")):
        data, blocks = _load_csv(path)
        return data, [path], blocks
    if os.path.isfile(path):
        with open(path) as fh:
            try:
                obj = json.load(fh)
            except json.JSONDecodeError as e:
                raise ValueError("%s: not JSON (%s). Accepted inputs: a directory of "
                                 "<candidate>/**/eval_info.json, a JSON "
                                 "{\"candidates\": {name: {task: [0/1,...]}}}, or a CSV with "
                                 "columns candidate,task,episode,success[,block]" % (path, e))
        cands = obj.get("candidates") if isinstance(obj, dict) else None
        if not isinstance(cands, dict) or not cands:
            raise ValueError("%s: expected {\"candidates\": {name: {task: [0/1,...]}}}" % path)
        data = {}
        for name, tasks in cands.items():
            if not isinstance(tasks, dict) or not tasks:
                raise ValueError("candidate %r: expected {task: [0/1,...]}" % name)
            data[str(name)] = {str(t): [_bool(x, "candidate %s task %s" % (name, t)) for x in v]
                               for t, v in tasks.items()}
        blocks = None
        if isinstance(obj.get("blocks"), dict):
            shared = {str(t): [str(x) for x in v] for t, v in obj["blocks"].items()}
            blocks = {name: dict(shared) for name in data}
        return data, [path], blocks
    if not os.path.isdir(path):
        raise ValueError("%s: not a file or directory" % path)
    data, files = {}, []
    for cand in sorted(os.listdir(path)):
        cdir = os.path.join(path, cand)
        if not os.path.isdir(cdir) or cand.startswith("."):
            continue
        tasks = {}
        for root, _dirs, fnames in os.walk(cdir):
            for fn in sorted(fnames):
                if not fn.endswith("eval_info.json"):
                    continue
                fp = os.path.join(root, fn)
                with open(fp) as fh:
                    try:
                        d = json.load(fh)
                    except json.JSONDecodeError as e:
                        raise ValueError("%s: %s" % (fp, e))
                for pt in d.get("per_task", []) or []:
                    m = pt.get("metrics", {}) or {}
                    if m.get("successes") is None:
                        continue
                    key = "%s/%s" % (pt.get("task_group", "task"), pt.get("task_id", 0))
                    if key in tasks:
                        raise ValueError("candidate %s: task %s appears in two files (%s)"
                                         % (cand, key, fp))
                    tasks[key] = [bool(s) for s in m["successes"]]
                files.append(fp)
        if tasks:
            data[cand] = tasks
    if not data:
        raise ValueError("%s: no <candidate>/**/eval_info.json with per_task successes" % path)
    return data, files, None


def validate(data, blocks=None):
    """Return (invalid, warnings): invalid non-empty = unusable."""
    invalid, warn = [], []
    names = sorted(data)
    if len(names) < 2:
        invalid.append("need >= 2 candidates, got %d" % len(names))
        return invalid, warn
    task_sets = {n: set(data[n]) for n in names}
    common = set.intersection(*task_sets.values())
    union = set.union(*task_sets.values())
    if union - common:
        invalid.append("task sets differ across candidates; missing somewhere: %s"
                       % ", ".join(sorted(union - common)))
    for t in sorted(common):
        ns = {n: len(data[n][t]) for n in names}
        if len(set(ns.values())) != 1:
            invalid.append("task %s: unequal episode counts %s — CRN pairing requires the "
                           "same episode list for every candidate" % (t, ns))
        elif min(ns.values()) < MIN_EPISODES:
            invalid.append("task %s: only %d episodes (< %d)" % (t, min(ns.values()), MIN_EPISODES))
    if blocks:
        for n in names:
            if n not in blocks:
                invalid.append("candidate %s has no block labels while others do" % n)
        for t in sorted(common):
            labs = [tuple(blocks.get(n, {}).get(t, [])) for n in names]
            if any(len(l) != len(data[n][t]) for l, n in zip(labs, names)):
                invalid.append("task %s: block labels do not cover every episode" % t)
                continue
            if len(set(labs)) != 1:
                if len({tuple(sorted(l)) for l in labs}) == 1:
                    invalid.append("task %s: same block composition but episode k sits in "
                                   "different blocks across candidates — re-index episodes so "
                                   "that block(k) is the same for every candidate" % t)
                else:
                    invalid.append("task %s: block composition differs across candidates "
                                   "(a candidate evaluated on a different robot/day mix is "
                                   "confounded with drift)" % t)
                continue
            counts = {}
            for b in labs[0]:
                counts[b] = counts.get(b, 0) + 1
            small = [b for b, c in counts.items() if c < 4]
            if small:
                warn.append("task %s: block(s) %s have < 4 episodes; per-block advantages "
                            "are indicative only" % (t, ", ".join(sorted(small))))
    return invalid, warn


def shared_blocks(data, blocks):
    """{task: [label per episode]} once validate() passed, else None."""
    if not blocks:
        return None
    first = sorted(data)[0]
    return {t: list(blocks[first][t]) for t in sorted(data[first])}


def parse_incumbent_spec(spec, candidates, tasks):
    """--incumbent: one candidate name, 'task=cand,task=cand', or a JSON/CSV file mapping
    task -> candidate. Returns None (best suite mean) or {task: candidate} covering every
    task (missing tasks fall back to a default only if the spec names one with '*')."""
    if spec is None:
        return None
    names = set(candidates)
    mapping = None
    if os.path.isfile(spec):
        if spec.lower().endswith((".csv", ".tsv")):
            delim = "\t" if spec.lower().endswith(".tsv") else ","
            with open(spec, newline="") as fh:
                rows = list(csv.DictReader(fh, delimiter=delim))
            cols = {c.lower().strip(): c for c in (rows[0].keys() if rows else [])}
            ct = cols.get("task") or cols.get("task_id") or cols.get("sku")
            cc = cols.get("candidate") or cols.get("incumbent") or cols.get("checkpoint") or cols.get("model")
            if not (ct and cc):
                raise ValueError("%s: need columns task,candidate" % spec)
            mapping = {str(r[ct]).strip(): str(r[cc]).strip() for r in rows}
        else:
            with open(spec) as fh:
                obj = json.load(fh)
            if isinstance(obj, dict) and isinstance(obj.get("incumbent"), dict):
                obj = obj["incumbent"]
            if not isinstance(obj, dict):
                raise ValueError("%s: expected {task: candidate}" % spec)
            mapping = {str(k): str(v) for k, v in obj.items()}
    elif "=" in spec:
        mapping = {}
        for part in spec.split(","):
            if "=" not in part:
                raise ValueError("bad --incumbent entry %r (want task=candidate)" % part)
            k, v = part.split("=", 1)
            mapping[k.strip()] = v.strip()
    else:
        if spec not in names:
            raise ValueError("incumbent %r is not a candidate (%s)" % (spec, ", ".join(sorted(names))))
        return {t: spec for t in tasks}
    default = mapping.pop("*", None)
    out = {}
    for t in tasks:
        c = mapping.get(t, default)
        if c is None:
            raise ValueError("incumbent mapping has no entry for task %r (add it or a '*' default)" % t)
        if c not in names:
            raise ValueError("incumbent %r for task %r is not a candidate (%s)" % (c, t, ", ".join(sorted(names))))
        out[t] = c
    unknown = sorted(set(mapping) - set(tasks))
    if unknown:
        raise ValueError("incumbent mapping names tasks not in the data: %s" % ", ".join(unknown))
    return out


# ---------------------------------------------------------------- helpers

def _sr(bits, idx=None):
    if idx is None:
        return 100.0 * sum(bits) / len(bits)
    return 100.0 * sum(bits[i] for i in idx) / len(idx)


def _se_diff(p1, p2, n):
    """SE (pp) of a difference of two binomial rates on n episodes each (unpaired)."""
    a, b = p1 / 100.0, p2 / 100.0
    return 100.0 * math.sqrt(max(a * (1 - a) + b * (1 - b), 1e-12) / n)


def wilson(k, n, z=stats.Z975):
    """Wilson interval in pp."""
    if n == 0:
        return (0.0, 100.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h))


def _argmax(keys, score):
    return sorted(keys, key=lambda k: (-score[k], str(k)))[0]


def _split_indices(n, labels, rng):
    """Half/half split of range(n), stratified within block labels when given."""
    if not labels:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx[: n // 2], idx[n // 2:]
    groups = {}
    for i, b in enumerate(labels):
        groups.setdefault(b, []).append(i)
    A, B = [], []
    flip = 0
    for b in sorted(groups):
        g = groups[b]
        rng.shuffle(g)
        h = len(g) // 2
        if len(g) % 2 == 1:           # alternate where the odd episode goes, block by block
            h += flip
            flip = 1 - flip
        A += g[:h]
        B += g[h:]
    if not A or not B:                # degenerate tiny task: fall back to a plain split
        idx = list(range(n))
        rng.shuffle(idx)
        return idx[: n // 2], idx[n // 2:]
    return A, B


# ---------------------------------------------------------------- budget

def budget(n, J, T, p=0.5, z_abstain=Z_ABSTAIN, targets=BUDGET_TARGETS_PP):
    """What n selection episodes per task can resolve, and what more would cost.

    threshold: the per-task advantage at which the rule defects (z*SE at rate p);
    power80: the true advantage detected with 80% power; per target advantage Y:
    the n needed for 80% power and the extra episodes (J*T*(n_needed-n)) it implies.
    Rate p=0.5 is the worst case; a task at 80% or 20% needs ~36% fewer episodes.
    """
    se = 100.0 * math.sqrt(2 * p * (1 - p) / n)
    k = z_abstain + stats.Z80
    rows = []
    for y in targets:
        need = int(math.ceil(2 * p * (1 - p) * (100.0 * k / y) ** 2))
        rows.append({"advantage_pp": y, "n_needed_per_task": need,
                     "extra_episodes_total": max(0, need - n) * J * T,
                     "already_resolvable": need <= n})
    return {"n": n, "J": J, "T": T, "rate_assumed": p, "z_abstain": z_abstain,
            "defection_threshold_pp": z_abstain * se, "advantage_80pct_power_pp": k * se,
            "selection_episodes": J * T * n,
            "expected_false_defections_under_null": (1 - stats.Phi(z_abstain)) * T,
            "targets": rows}


# ---------------------------------------------------------------- estimators

def split_half(data, draws=N_DRAWS, seed=RNG_SEED, z_abstain=Z_ABSTAIN,
               incumbent=None, z_dmg=Z_DMG, dmg_pp=DMG_PP, blocks=None):
    """Stratified split-half estimate of every arm on disjoint held-out episodes.

    Per draw and task: a random half of the episodes SELECTS, the other half SCORES
    (stratified within block when blocks are given). Arms: RANDOM (ship one candidate
    at random), INCUMBENT (the named per-task incumbent, or the best-suite-mean
    candidate on the SELECT half when none is named), ROUTE (per-task argmax on
    SELECT), ROUTE+ABSTAIN (defect from the incumbent only at one-sided z >= z_abstain
    on SELECT), ORACLE (per-task argmax on the SCORE half — an upper bound, never
    deployable). Damage is PRICED: held-out task SR at least dmg_pp + z_dmg*SE below
    the candidate-pool mean on that task.
    """
    keys = sorted(data)
    tasks = sorted(data[keys[0]])
    J, T = len(keys), len(tasks)
    rng = random.Random(seed)
    acc = {a: [] for a in ARMS}
    dmg = {a: [] for a in ARMS}
    worst = {a: [] for a in ARMS}
    defects = []
    pool_full = {t: sum(_sr(data[k][t]) for k in keys) / J for t in tasks}
    labels = blocks or {}
    for _ in range(draws):
        sel = {k: {} for k in keys}
        ev = {k: {} for k in keys}
        h = {}
        for t in tasks:
            n = len(data[keys[0]][t])
            A, B = _split_indices(n, labels.get(t), rng)
            h[t] = len(A)
            for k in keys:
                sel[k][t] = _sr(data[k][t], A)
                ev[k][t] = _sr(data[k][t], B)
        if incumbent is None:
            suite_sel = {k: sum(sel[k].values()) / T for k in keys}
            wbest = _argmax(keys, suite_sel)
            w = {t: wbest for t in tasks}
        else:
            w = incumbent
        pool_ev = {t: sum(ev[k][t] for k in keys) / J for t in tasks}
        chosen = {"INCUMBENT": dict(w), "ROUTE": {}, "ROUTE+ABSTAIN": {}, "ORACLE": {}, "RANDOM": {}}
        jr = keys[rng.randrange(J)]
        nd = 0
        for t in tasks:
            best = _argmax(keys, {k: sel[k][t] for k in keys})
            chosen["ROUTE"][t] = best
            se = _se_diff(sel[best][t], sel[w[t]][t], h[t])
            defect = best != w[t] and (sel[best][t] - sel[w[t]][t]) >= z_abstain * max(se, 1e-9)
            chosen["ROUTE+ABSTAIN"][t] = best if defect else w[t]
            nd += int(defect)
            chosen["ORACLE"][t] = _argmax(keys, {k: ev[k][t] for k in keys})
            chosen["RANDOM"][t] = jr
        defects.append(nd)
        for a in ARMS:
            per = [ev[chosen[a][t]][t] for t in tasks]
            acc[a].append(sum(per) / T)
            dl = []
            for t in tasks:
                p = pool_full[t] / 100.0
                se_t = 100.0 * math.sqrt(max(p * (1 - p), 1e-12) / h[t]) * math.sqrt(1 - 1.0 / J)
                d = ev[chosen[a][t]][t] - pool_ev[t]
                dl.append((d, d <= -(dmg_pp + z_dmg * se_t)))
            dmg[a].append(sum(1 for _d, flag in dl if flag))
            worst[a].append(min(d for d, _f in dl))

    def ms(xs):
        m = sum(xs) / len(xs)
        s = (stats.var(xs) ** 0.5 / len(xs) ** 0.5) if len(xs) > 1 else 0.0
        return m, s

    out = {"draws": draws, "seed": seed, "J": J, "T": T, "arms": {},
           "stratified_by_block": bool(blocks)}
    for a in ARMS:
        m, s = ms(acc[a])
        out["arms"][a] = {"heldout_sr": m, "sem": s, "damaged_tasks": ms(dmg[a])[0],
                          "worst_task_vs_pool": ms(worst[a])[0]}
    out["defections_per_draw"] = ms(defects)[0]
    out["abstain_rate"] = 1.0 - ms(defects)[0] / T
    out["gain_route_abstain_vs_incumbent"] = (out["arms"]["ROUTE+ABSTAIN"]["heldout_sr"]
                                              - out["arms"]["INCUMBENT"]["heldout_sr"])
    out["gain_route_vs_incumbent"] = out["arms"]["ROUTE"]["heldout_sr"] - out["arms"]["INCUMBENT"]["heldout_sr"]
    return out


def release_table(data, z_abstain=Z_ABSTAIN, incumbent=None, blocks=None):
    """The deployable routing table, decided on ALL episodes with the abstention rule.

    Per task: the incumbent (per task), its SR, the best candidate, the one-sided z of
    the advantage, the decision (DEFECT / ABSTAIN), the chosen candidate's Wilson 95% CI,
    the one-sided 95% upper bound on (incumbent - chosen) — <= 0 for every defection by
    construction — and, when blocks are given, the advantage inside every block and
    whether all blocks agree in sign.
    """
    keys = sorted(data)
    tasks = sorted(data[keys[0]])
    J, T = len(keys), len(tasks)
    SR = {k: {t: _sr(data[k][t]) for t in tasks} for k in keys}
    suite = {k: sum(SR[k].values()) / T for k in keys}
    if incumbent is None:
        wbest = _argmax(keys, suite)
        w = {t: wbest for t in tasks}
        how = "best suite mean over all episodes (no --incumbent given)"
    else:
        for t in tasks:
            if incumbent.get(t) not in data:
                raise ValueError("incumbent for task %r (%r) is not a candidate" % (t, incumbent.get(t)))
        w = {t: incumbent[t] for t in tasks}
        how = "named per task (--incumbent)" if len(set(w.values())) > 1 else "named by --incumbent"
    rows = []
    for t in tasks:
        n = len(data[w[t]][t])
        best = _argmax(keys, {k: SR[k][t] for k in keys})
        adv = SR[best][t] - SR[w[t]][t]
        se = _se_diff(SR[best][t], SR[w[t]][t], n)
        z = adv / se if se > 0 else 0.0
        defect = best != w[t] and z >= z_abstain
        chosen = best if defect else w[t]
        lo, hi = wilson(sum(data[chosen][t]), n)
        reg = max(0.0, -adv + stats.norm_q(0.95) * se) if defect else 0.0
        row = {"task": t, "n": n, "incumbent": w[t], "incumbent_sr": SR[w[t]][t], "best": best,
               "best_sr": SR[best][t], "advantage_pp": adv, "se_pp": se, "z": z,
               "decision": "DEFECT" if defect else "ABSTAIN", "chosen": chosen,
               "chosen_sr": SR[chosen][t], "ci95": [lo, hi], "regression_bound_pp": reg}
        if blocks and t in blocks:
            per = {}
            for i, b in enumerate(blocks[t]):
                per.setdefault(b, [0, 0, 0])
                per[b][0] += 1
                per[b][1] += int(data[chosen][t][i])
                per[b][2] += int(data[w[t]][t][i])
            adv_b = {b: 100.0 * (c[1] - c[2]) / c[0] for b, c in per.items()}
            row["block_advantage_pp"] = adv_b
            row["blocks_agree"] = (all(v >= 0 for v in adv_b.values()) if defect else None)
            row["min_block_advantage_pp"] = (min(adv_b.values()) if defect else None)
        rows.append(row)
    n_def = sum(1 for r in rows if r["decision"] == "DEFECT")
    inc_suite = sum(SR[w[t]][t] for t in tasks) / T
    plug_in = sum(r["chosen_sr"] for r in rows) / T - inc_suite
    disagree = [r["task"] for r in rows if r.get("blocks_agree") is False]
    return {"incumbent": w, "incumbent_how": how, "incumbent_suite_sr": inc_suite,
            "candidates": keys, "suite_sr": suite, "tasks": tasks, "J": J, "T": T,
            "rows": rows, "n_defections": n_def,
            "candidates_used": sorted({r["chosen"] for r in rows}),
            "plug_in_gain_pp": plug_in,
            "regression_bound_pp": max(r["regression_bound_pp"] for r in rows),
            "expected_false_defections_under_null": (1 - stats.Phi(z_abstain)) * T,
            "z_abstain": z_abstain, "min_n": min(r["n"] for r in rows),
            "blocks": sorted({b for t in (blocks or {}) for b in blocks[t]}) if blocks else [],
            "defections_with_block_disagreement": disagree}


def plan(J, T, n_select, p=0.5, z_abstain=Z_ABSTAIN):
    """What a selection budget can resolve, before spending it."""
    out = budget(n_select, J, T, p=p, z_abstain=z_abstain)
    out["measured_cells"] = {}
    for cell, rows in MEASURED_CELLS.items():
        bj = min(rows["J"], key=lambda j: abs(j - J))
        bn = min(rows["budget_eps_per_task"], key=lambda n: abs(n - n_select))
        out["measured_cells"][cell] = {
            "nearest_J": bj, "route_minus_suite_at_nearest_J_pp": rows["J"][bj],
            "nearest_budget": bn, "route_minus_suite_at_nearest_budget_pp": rows["budget_eps_per_task"][bn],
            "damaged_tasks_suite": rows["damaged_suite"], "damaged_tasks_route": rows["damaged_route"]}
    out["regression_risk"] = damage_rows(J, T)
    return out


def damage_rows(J, T):
    """Measured P(>=1 damaged task) by pool size, from the cells nearest this task count.

    Returns the two nearest-T cells (by |log T ratio|, so 10 and 50 both surface for a
    caller at T=25) with their curves and the pool size each needed to reach 0.05.
    """
    def _need(curve, target=0.05):
        for j in sorted(curve):
            if curve[j] <= target:
                return j
        return None

    scored = []
    for name, r in DAMAGE_J_CELLS.items():
        gap = abs(math.log(max(T, 1) / float(r["T"])))
        scored.append((gap, name, r))
    scored.sort(key=lambda x: (x[0], x[1]))
    rows = []
    for gap, name, r in scored[:2]:
        near = min(r["route"], key=lambda j: abs(j - J))
        rows.append({
            "cell": name, "cell_T": r["T"], "cell_n": r["n"],
            "ship_one": r["ship_one"],
            "nearest_J": near,
            "best_val_at_nearest_J": r["best_val"][near],
            "route_at_nearest_J": r["route"][near],
            "route_abstain_at_nearest_J": r["route_abstain"][near],
            "J_for_5pct_route": _need(r["route"]),
            "J_for_5pct_best_val": _need(r["best_val"]),
        })
    return rows


# ---------------------------------------------------------------- reports

def _budget_lines(bud, prefix="  "):
    L = ["%sat %d selection episodes/task (rate %.0f%% assumed): the rule defects at a per-task "
         "advantage >= %.1f pp; a true advantage of %.1f pp is detected with 80%% power."
         % (prefix, bud["n"], 100 * bud["rate_assumed"], bud["defection_threshold_pp"], bud["advantage_80pct_power_pp"])]
    L.append("%sunder a no-difference null, %.2f of %d tasks would defect falsely (per-task alpha 0.05)."
             % (prefix, bud["expected_false_defections_under_null"], bud["T"]))
    L.append("%sto resolve a per-task advantage of ... with 80%% power:" % prefix)
    for r in bud["targets"]:
        if r["already_resolvable"]:
            L.append("%s  %5.1f pp -> n = %4d per task per candidate  (you have it)" % (prefix, r["advantage_pp"], r["n_needed_per_task"]))
        else:
            L.append("%s  %5.1f pp -> n = %4d per task per candidate  (+%d episodes in total for J=%d, T=%d)"
                     % (prefix, r["advantage_pp"], r["n_needed_per_task"], r["extra_episodes_total"], bud["J"], bud["T"]))
    return L


def format_build_report(rel, sh, meta, bud=None):
    L = []
    L.append("# orbit-eval route build — release report")
    L.append("")
    ns = sorted({r["n"] for r in rel["rows"]})
    L.append("candidates: %d (%s)   tasks: %d   episodes/task: %s   orbit-eval %s"
             % (rel["J"], ", ".join(rel["candidates"]), rel["T"],
                ("%d" % ns[0]) if len(ns) == 1 else ("%d..%d" % (ns[0], ns[-1])), meta["version"]))
    incs = sorted(set(rel["incumbent"].values()))
    L.append("incumbent: %s (%s), incumbent suite SR %.2f"
             % (", ".join(incs) if len(incs) <= 4 else "%d per-task incumbents" % len(incs),
                rel["incumbent_how"], rel["incumbent_suite_sr"]))
    L.append("abstention: defect only at one-sided z >= %.3f on the selection episodes" % rel["z_abstain"])
    if rel["blocks"]:
        L.append("blocks: %s (halves stratified within block; every defection reports per-block agreement)"
                 % ", ".join(rel["blocks"]))
    L.append("CRN: %s" % ("asserted by user (--assume-crn)" if meta.get("crn_asserted_by_user") else
                          "NOT asserted — episode counts match, initial states unverified; pass --assume-crn "
                          "only if the k-th episode of every task used the same initial state for every candidate"))
    L.append("")
    L.append("## Held-out estimate (stratified split-half, %d draws, seed %d)" % (sh["draws"], sh["seed"]))
    L.append("")
    L.append("| arm | held-out SR | vs INCUMBENT | damaged tasks | worst task vs pool |")
    L.append("|---|---|---|---|---|")
    s0 = sh["arms"]["INCUMBENT"]["heldout_sr"]
    for a, r in sh["arms"].items():
        tag = "  (upper bound, not deployable)" if a == "ORACLE" else ""
        L.append("| %s | %.2f ± %.2f | %+.2f | %.2f | %+.2f |%s"
                 % (a, r["heldout_sr"], r["sem"], r["heldout_sr"] - s0, r["damaged_tasks"], r["worst_task_vs_pool"], tag))
    L.append("")
    L.append("ROUTE+ABSTAIN defects on %.2f of %d tasks per draw (abstain rate %.0f%%)."
             % (sh["defections_per_draw"], sh["T"], 100 * sh["abstain_rate"]))
    L.append("")
    L.append("## Release table (decided on all episodes)")
    L.append("")
    hasb = bool(rel["blocks"])
    L.append("| task | n | incumbent | inc SR | best | best SR | adv (pp) | z | decision | ships | 95%% CI | regression bound |%s"
             % (" blocks agree | min block adv |" if hasb else ""))
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|%s" % ("---|---|" if hasb else ""))
    for r in rel["rows"]:
        extra = ""
        if hasb:
            ba = r.get("blocks_agree")
            extra = " %s | %s |" % ("—" if ba is None else ("yes" if ba else "NO"),
                                    "—" if r.get("min_block_advantage_pp") is None else "%+.1f" % r["min_block_advantage_pp"])
        L.append("| %s | %d | %s | %.1f | %s | %.1f | %+.1f | %.2f | %s | %s | [%.1f, %.1f] | %.1f |%s"
                 % (r["task"], r["n"], r["incumbent"], r["incumbent_sr"], r["best"], r["best_sr"], r["advantage_pp"], r["z"],
                    r["decision"], r["chosen"], r["ci95"][0], r["ci95"][1], r["regression_bound_pp"], extra))
    L.append("")
    verdict = "ROUTE" if rel["n_defections"] else "INCUMBENT-STANDS"
    L.append("**Verdict: %s** — %d defection(s), %d candidate(s) in the release: %s."
             % (verdict, rel["n_defections"], len(rel["candidates_used"]), ", ".join(rel["candidates_used"])))
    if rel["defections_with_block_disagreement"]:
        L.append("**Block warning:** the advantage reverses sign in at least one block on: %s — "
                 "treat these defections as drift-sensitive and re-evaluate on the disagreeing block before shipping."
                 % ", ".join(rel["defections_with_block_disagreement"]))
    L.append("In-sample plug-in gain over the incumbent: %+.2f pp (optimistic; the held-out row above is the "
             "number to quote: %+.2f pp)." % (rel["plug_in_gain_pp"], sh["gain_route_abstain_vs_incumbent"]))
    L.append("Regression bound: at one-sided 95%% per task, no defected task ships more than %.1f pp below its "
             "incumbent; abstained tasks keep the incumbent exactly. Per-task alpha is not family-wise: under a "
             "no-difference null, %.2f of %d defections would be false on average."
             % (rel["regression_bound_pp"], rel["expected_false_defections_under_null"], rel["T"]))
    if bud is not None:
        L.append("")
        L.append("## Budget — what this evaluation can and cannot see")
        L.append("")
        L.extend(_budget_lines(bud, prefix=""))
        if bud["defection_threshold_pp"] > 15:
            L.append("")
            L.append("At this budget only very large per-task differences can defect; an INCUMBENT-STANDS verdict "
                     "here means 'unresolved', not 'no difference'. The lines above price the episodes that would resolve it.")
    L.append("")
    L.append("## Scope")
    L.append("One cell, the candidates you evaluated, this episode list. Gains are cell-local (no "
             "transferable-sigma law); task identity must be known at inference to serve a routed release; "
             "J candidates are held and served (or their shared backbone + per-candidate modules).")
    L.append("")
    L.append("## Provenance")
    L.append("inputs: %d file(s), sha256 of the sorted (path, sha256) list: %s" % (meta["n_files"], meta["inputs_sha256"]))
    L.append("record sha256 covers this table and the held-out estimate; `--out` writes it as JSON "
             "(load it with orbit_eval.routed_policy.RoutedPolicy.from_release).")
    return "\n".join(L)


def format_plan(pl):
    L = ["orbit-eval route plan — what %d selection episodes/task can resolve" % pl["n"], ""]
    L.append("  candidates J=%d, tasks T=%d -> %d selection episodes total" % (pl["J"], pl["T"], pl["selection_episodes"]))
    L.extend(_budget_lines(pl))
    L.append("")
    L.append("  Measured cells (route minus best-validation, held-out, pp) — cell-local, NOT a forecast for yours:")
    for cell, r in pl["measured_cells"].items():
        L.append("    %s" % cell)
        L.append("      nearest J=%d: %+.2f    nearest budget %d eps/task: %+.2f    damaged tasks suite %.2f -> route %.2f"
                 % (r["nearest_J"], r["route_minus_suite_at_nearest_J_pp"], r["nearest_budget"],
                    r["route_minus_suite_at_nearest_budget_pp"], r["damaged_tasks_suite"], r["damaged_tasks_route"]))
    L.append("")
    L.append("  Below ~20 selection episodes/task the advantage was unreliable in every measured cell (ROUTE-1 B').")
    L.append("  The gain is a property of YOUR cell, not of the method: on libero_spatial it was")
    L.append("  +1.4 pp with zero inside the interval at 100 selection episodes/task.")
    L.append("")
    L.append("  REGRESSION RISK — P(>=1 task damaged vs your own retrain pool), measured cells nearest T=%d:" % pl["T"])
    for r in pl["regression_risk"]:
        L.append("    %s  [T=%d, %d eps/task]" % (r["cell"], r["cell_T"], r["cell_n"]))
        L.append("      ship one retrain: %.3f    at J=%d: best-validation %.3f | route %.3f | route+abstain %.3f"
                 % (r["ship_one"], r["nearest_J"], r["best_val_at_nearest_J"],
                    r["route_at_nearest_J"], r["route_abstain_at_nearest_J"]))
        nv, nb = r["J_for_5pct_route"], r["J_for_5pct_best_val"]
        L.append("      pool size that reached 5%% risk: routing J=%s; best-validation %s"
                 % (nv if nv else ">8", ("J=%d" % nb) if nb else "NEVER, at any J tested"))
    L.append("")
    L.append("  Read those two rows together: training more candidates and shipping the")
    L.append("  best-validation one did not reduce regression risk at any pool size, and the")
    L.append("  pool size that did grew with the task count. Constants are cell-local; the")
    L.append("  shape is what is claimed to transfer (DAMAGE-J, declaration fb1c760f).")
    return "\n".join(L)


def inputs_digest(files):
    items = sorted((os.path.relpath(f), _sha256(f)) for f in files)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest(), len(items)
