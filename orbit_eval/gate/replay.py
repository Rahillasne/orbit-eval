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

"""Historical replay: run the founder's own eval corpus back through the
Ship-Gate and report which past "wins" were inside measured noise.

Input is the ORBIT per-episode corpus JSONL
(``experiments/forecast/*_per_episode.jsonl``): one JSON object per run with
``run_id, wave, policy_class, suite, task, k_episodes, train_seed, set_hash,
n_eval, sr_atlas, eval_seed`` and ``successes`` as a '0'/'1' character string
— per-episode outcomes, CRN-ordered (same episode index = same initial state
across runs with equal ``n_eval`` and ``eval_seed``; the corpus records
``eval_seed_provenance = lerobot_default_1000_implicit`` throughout).
NOTE: the harvester stores ``k_episodes``, ``train_seed`` and ``steps`` as
STRINGS — :func:`parse_corpus_line` coerces them.

Three replay modes, every number computed, none asserted:

  NULL-PAIR  (the headline validation) — pairs of runs with IDENTICAL
             set_hash / n_eval / eval_seed but DIFFERENT train_seed: seed
             replicates trained on identical data, so any measured
             difference is pure retraining noise (sigma_run; atlas
             provenance PHASE2C / Phase 1c). Every SHIP verdict on a null
             pair is a FALSE SHIP; the false-ship rate is reported with a
             Wilson CI against the configured alpha PLUS the disclosed
             dependence structure (gates are mirrored direction-pairs
             sharing runs within replicate groups), a cluster bootstrap CI
             over replicate groups (the correct unit of independence), the
             retraining noise the null gaps imply (sigma_from_gaps, printed
             beside the atlas sigma_run), and a synthetic exact-null
             calibration control of the paired McNemar path.
             v0.1: the primary pass gates at comparison_level='retrain'
             with the atlas sigma_run prior auto-resolved per pair
             (retrain_prior_regime); a second pass at 'checkpoint' (the
             v0 eval-only semantics) feeds the report's BEFORE/AFTER
             false-ship table.
  CURSE      cells with >= 3 runs (e.g. the pi05 wave): pick the max-SR run
             out of a pool, gate it with and without winner's-curse
             correction — max-over-checkpoints selection bias made visible
             (PI05_CELL1_RESULTS: +4.00 pp mean inflation at J=4, measured
             prospectively).
  EFFECT     same-cell pairs with DIFFERENT set_hash (different training
             sets): how often the gate resolves real effects vs
             COLLECT-MORE, and at what implied episode budgets.
             v0.1: gated at comparison_level='retrain' with the per-pair
             auto-resolved sigma_run prior — a different-set candidate
             arrives via an independent retrain.

STACK EPOCHS: gym_pusht was REBUILT mid-programme and the atlas measured the
damage (STACK_SHIFT_PUSHT_PP = -18.2 pp mean, non-uniform; CAVEATS
['cross-stack']: never pool or compare SRs across a rebuild). Null and
effect pair keys therefore include a wave->stack-epoch mapping so
cross-rebuild pairs are NEVER formed (excluded pairings are counted and
disclosed in the report); cross-WAVE pairs within one epoch are kept but
carry a recorded ASSUMPTION that the stack is identical across those waves.

v0 is fixed-sample throughout (anytime-valid sequential replay is v1).
This module layers on engine/records/io/stats and reimplements no
statistics.
"""

import dataclasses
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

from .. import atlas, io, stats
from .engine import curse_from_prior, gate
from .records import GateConfig, append_record

_ALL_N_RE = re.compile(r"^all(\d+)$")

# resolution order for aliased corpus fields (frozen normalization)
_SR_KEYS = ("sr_atlas", "sr_recomputed", "sr")


# ---------------------------------------------------------------- reader

@dataclass
class CorpusRun:
    """One corpus line, normalized: the :class:`orbit_eval.io.EvalRun` the
    gate consumes plus the corpus metadata replay groups by. ``raw`` keeps
    the original parsed dict verbatim (nothing silently dropped)."""
    run: io.EvalRun
    wave: str
    policy_class: str
    suite: str
    task: str
    k_episodes: Optional[int]
    train_seed: Optional[int]
    steps: Optional[int]
    n_tasks_hint: Optional[int]
    source: Optional[str]
    raw: dict


def _coerce_int(d, keys, ctx):
    """First present-and-non-None value among aliased keys -> int.
    The corpus stores k_episodes/train_seed/steps as STRINGS (measured on
    the real files, not assumed) — coerce, and treat garbage as a bad line."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                raise ValueError("%s: %s=%r is not an integer" % (ctx, k, v))
    return None


def _coerce_successes(v, ctx):
    """'0'/'1' character string OR list of bool/0/1 -> List[bool]."""
    if v is None:
        return None
    if isinstance(v, str):
        bad = sorted(set(v) - {"0", "1"})
        if bad:
            raise ValueError("%s: successes string holds non-'0'/'1' chars %s"
                             % (ctx, bad))
        return [ch == "1" for ch in v]
    if isinstance(v, list):
        out = []
        for i, s in enumerate(v):
            if isinstance(s, bool):
                out.append(s)
            elif s in (0, 1):
                out.append(bool(s))
            else:
                raise ValueError("%s: successes[%d]=%r is not bool/0/1"
                                 % (ctx, i, s))
        return out
    raise ValueError("%s: successes must be a '0'/'1' string or a list, got %s"
                     % (ctx, type(v).__name__))


def parse_corpus_line(line, path=None, lineno=None):
    """One corpus JSONL line -> :class:`CorpusRun`; raises ValueError on a
    bad line (malformed JSON, missing run_id, successes/n_eval mismatch,
    non-integer string ints).

    Frozen normalization (field aliases seen across corpus generations):
    suite <- suite|task_group, eval_seed <- eval_seed|seed,
    sr <- sr_atlas|sr_recomputed|sr, k_episodes <- k_episodes|k,
    steps <- steps|final_step, policy_class <- policy_class|policy
    (default 'unknown'). ``design_steps`` is left None: the corpus never
    records the designed budget, so the budget gate passes with its
    'unverifiable' caveat rather than a fabricated pass.
    """
    ctx = "%s:%s" % (path or "<line>", lineno if lineno is not None else "?")
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError("%s: invalid JSON: %s" % (ctx, e))
    if not isinstance(d, dict):
        raise ValueError("%s: line is not a JSON object" % ctx)

    run_id = d.get("run_id")
    if not run_id:
        raise ValueError("%s: run_id required" % ctx)
    run_id = str(run_id)
    ctx = "%s (%s)" % (ctx, run_id)

    suite = d.get("suite", d.get("task_group"))
    suite = str(suite) if suite is not None else "unknown"
    task = str(d.get("task")) if d.get("task") is not None else "unknown"
    wave = str(d.get("wave")) if d.get("wave") is not None else "unknown"
    policy_class = d.get("policy_class", d.get("policy"))
    policy_class = str(policy_class) if policy_class is not None else "unknown"

    sr = None
    for k in _SR_KEYS:
        if d.get(k) is not None:
            sr = float(d[k])
            break

    k_episodes = _coerce_int(d, ("k_episodes", "k"), ctx)
    train_seed = _coerce_int(d, ("train_seed",), ctx)
    steps = _coerce_int(d, ("steps", "final_step"), ctx)
    eval_seed = _coerce_int(d, ("eval_seed", "seed"), ctx)

    successes = _coerce_successes(d.get("successes"), ctx)
    n_eval = _coerce_int(d, ("n_eval",), ctx)
    if n_eval is None and successes is not None:
        n_eval = len(successes)
    if successes is not None and n_eval is not None and len(successes) != n_eval:
        raise ValueError("%s: len(successes)=%d != n_eval=%d"
                         % (ctx, len(successes), n_eval))

    set_hash = d.get("set_hash")
    set_hash = str(set_hash) if set_hash is not None else None

    m = _ALL_N_RE.match(task)
    n_tasks_hint = int(m.group(1)) if m else None

    run = io.EvalRun(
        run_id=run_id, fmt="corpus", path=path, sr=sr, n_eval=n_eval,
        seed=eval_seed, successes=successes, set_hash=set_hash,
        final_step=steps, design_steps=None, task_groups=[suite],
    )
    # engine reads getattr(run, 'policy_class', None) — recorded, not typed
    # into EvalRun, so the base io schema stays untouched.
    run.policy_class = policy_class

    return CorpusRun(
        run=run, wave=wave, policy_class=policy_class, suite=suite, task=task,
        k_episodes=k_episodes, train_seed=train_seed, steps=steps,
        n_tasks_hint=n_tasks_hint,
        source=str(d["source"]) if d.get("source") is not None else None,
        raw=d,
    )


def load_corpus(paths):
    """Parse corpus JSONL files -> (runs, errors). Blank lines are skipped;
    bad lines are COLLECTED as ``{'path', 'line', 'error'}`` dicts (1-based
    line numbers), never silently dropped and never fatal — one corrupt line
    must not sink a replay. Raises OSError only if a file is unreadable."""
    cruns, errors = [], []
    for path in paths:
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    cruns.append(parse_corpus_line(line, path=path,
                                                   lineno=lineno))
                except ValueError as e:
                    errors.append({"path": path, "line": lineno,
                                   "error": str(e)})
    return cruns, errors


# ---------------------------------------------------------------- stack epochs

# Wave -> environment-stack epoch for PushT, confirmed against the run
# history (not assumed): FORECAST_RESULTS.md — "All 12 PushT TEST runs come
# off a rebuilt gym_pusht stack" = phase1d_pusht (8 runs) + phase4_pusht
# (4 runs); PHASE1C_REPAIR_RESULTS.md R1 measured the rebuild at -18.2 pp
# mean (33.0 -> 14.8, range -10.6..-24.4, non-uniform) — the corpus means
# match (datamodel/phase1c/phase1c_wave2 35.0/35.0/34.2 vs phase1d/phase4
# 17.0/17.6). atlas.CAVEATS['cross-stack']: never pool across the rebuild.
_PUSHT_STACK_EPOCHS = {
    "datamodel_pusht": "pusht-stack-old",
    "phase1c_pusht": "pusht-stack-old",
    "phase1c_wave2": "pusht-stack-old",
    "phase1d_pusht": "pusht-stack-rebuilt",
    "phase4_pusht": "pusht-stack-rebuilt",
}


def stack_epoch(cr):
    """Environment-stack epoch a run was evaluated on, for pair keys.

    PushT suites: mapped per wave from the run history (see
    _PUSHT_STACK_EPOCHS); an UNKNOWN pusht wave gets its own epoch (= the
    wave name) so it can never silently pool across a possible rebuild.
    Non-pusht suites: one epoch — no rebuild was measured there — so
    cross-wave pairing is allowed WITH the recorded assumption
    (_cross_wave_assumptions)."""
    if "pusht" in (cr.suite or "").lower():
        return _PUSHT_STACK_EPOCHS.get(cr.wave, "pusht-wave:%s" % cr.wave)
    return "no-rebuild-measured"


def _cross_wave_assumptions(a, b):
    """extra_assumptions for a cross-wave pair (same stack epoch by
    construction — cross-EPOCH pairs are never formed), else None."""
    if a.wave == b.wave:
        return None
    return ["ASSUMPTION: cross-wave pair (wave %r vs %r, stack epoch %r): "
            "CRN pairing assumes the environment stack is IDENTICAL across "
            "these waves. The measured gym_pusht rebuild (SR shift %.1f pp "
            "mean, NON-uniform — OPS_RUNBOOK sec 9) is keyed out of pairing "
            "entirely via the wave->epoch map; no rebuild is recorded "
            "between these waves [atlas CAVEATS['cross-stack']]"
            % (a.wave, b.wave, stack_epoch(a), atlas.STACK_SHIFT_PUSHT_PP)]


# ---------------------------------------------------------------- pair finders

def _null_key(cr):
    """Replicate-group key; stack_epoch LAST so key[:-1] is the epoch-blind
    key the cross-epoch exclusion counters group on."""
    return (cr.policy_class, cr.suite, cr.task, cr.run.set_hash,
            cr.run.n_eval, cr.run.seed, cr.steps, stack_epoch(cr))


def find_null_pairs(cruns):
    """Seed replicates: runs grouped on (policy_class, suite, task, set_hash,
    n_eval, eval_seed, steps, stack_epoch); within a group every unordered
    pair whose train_seeds are BOTH RECORDED and DIFFERENT is a null pair —
    trained on the identical episode set (set_hash is sha1(sorted
    episodes)[:16], the atlas convention), so the true SR difference is zero
    and any measured gap is pure retraining noise. Runs without successes or
    set_hash cannot be gated/certified and are excluded; an unrecorded
    train_seed cannot certify a replicate (measured, not assumed) and is
    likewise excluded. The stack_epoch key component keeps replicate pairs
    from straddling the measured gym_pusht rebuild (excluded pairings are
    counted by replay_null_pairs and disclosed in the report).

    Pairs are deterministic: each tuple ordered by run_id, the list sorted
    by (run_id_a, run_id_b)."""
    groups = defaultdict(list)
    for cr in cruns:
        if cr.run.successes and cr.run.set_hash and cr.train_seed is not None:
            groups[_null_key(cr)].append(cr)
    pairs = []
    for key in sorted(groups, key=lambda k: tuple(map(str, k))):
        rs = sorted(groups[key], key=lambda cr: cr.run.run_id)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if rs[i].train_seed != rs[j].train_seed:
                    a, b = sorted((rs[i], rs[j]), key=lambda cr: cr.run.run_id)
                    pairs.append((a, b))
    pairs.sort(key=lambda p: (p[0].run.run_id, p[1].run.run_id))
    return pairs


def _config_for(config, cr):
    """Per-gate config: inject the run's all<N> task-count hint when the
    caller declared no task structure. Never mutates the caller's config
    (dataclasses.replace)."""
    if config.n_tasks is None and config.task_blocks is None \
            and cr.n_tasks_hint:
        return dataclasses.replace(config, n_tasks=cr.n_tasks_hint)
    return config


def _delta(rec):
    return rec["statistics"].get("delta_hat_pp")


def _abs_or_zero(x):
    return abs(x) if x is not None else 0.0


def _median(xs):
    """Median of a non-empty list (mean of the middle two for even counts);
    stdlib-only, deterministic."""
    ys = sorted(xs)
    m = len(ys)
    if m % 2:
        return ys[m // 2]
    return (ys[m // 2 - 1] + ys[m // 2]) / 2.0


# ---------------------------------------------------------------- null mode

# policy_class (corpus spelling) -> atlas regime 'policy' field
_POLICY_TO_ATLAS = {"dp": "diffusion", "diffusion": "diffusion",
                    "smolvla_ft": "smolvla-ft", "smolvla-ft": "smolvla-ft",
                    "pi05": "pi05-ft", "pi05_ft": "pi05-ft",
                    "pi05-ft": "pi05-ft"}

# LIBERO / DiffusionPolicy sigma_run was measured at three k values; the
# retrain prior picks the NEAREST k regime (frozen v0.1 replay rule).
_LIBERO_DP_K_REGIMES = ("libero-dp-k22", "libero-dp-k31", "libero-dp-k44")


def retrain_prior_regime(cr):
    """Atlas regime whose measured sigma_run prices THIS run's retraining
    lottery — the v0.1 replay auto-resolution, per pair:

      pusht + DP, old stack epoch      -> 'pusht-dp'          (2.09 pp)
      pusht + DP, rebuilt stack epoch  -> 'pusht-dp-rebuilt'  (1.292 pp,
                                          df=4, resolved 2026-08-20)
      libero + DP                      -> nearest-k of libero-dp-k22/k31/k44
                                          by the run's k_episodes
      smolvla_ft (any spelling)        -> 'smolvla-ft'        (3.31 pp)
      pi05 (any spelling)              -> 'pi05-ft'  (9.74 pp, df 10;
                                          PI05-PAIRS overturn of 2.02)

    pusht-dp-rebuilt is FLOOR-DOMINATED (sigma_run 1.292 sits below its own
    1.682 pp binomial floor; the atlas row says "DO NOT price from this
    row"). It still auto-resolves here — by the 2026-08-20 design, engine
    pre-check 11c stamps every record priced from a floor-dominated regime
    with the ANTI-CONSERVATIVE PRIOR warning rather than silently swapping
    the value; operators who want the corpus's floor-respecting pricing
    policy (PRIOR_CALIBRATION column (f), max(atlas, EB-shrunk) = 1.80) pass
    it as an explicit sigma_run_pp, which beats the regime by the frozen
    resolution order.

    Returns None when no measured prior applies — an UNKNOWN pusht stack
    epoch (absolute SRs shifted across the rebuild, so neither epoch's prior
    is honest for an unmapped wave), a libero DP run without k_episodes, or
    an unmapped policy: the gate then runs at retrain level with the recorded
    no-prior disclosure instead of a fabricated prior."""
    pol = (cr.policy_class or "").lower().replace("-", "_")
    s = (cr.suite or "").lower()
    if pol in ("pi05", "pi05_ft"):
        return "pi05-ft"
    if pol in ("smolvla_ft",):
        return "smolvla-ft"
    if pol in ("dp", "diffusion"):
        if "pusht" in s:
            epoch = stack_epoch(cr)
            if epoch == "pusht-stack-old":
                return "pusht-dp"
            if epoch == "pusht-stack-rebuilt":
                return "pusht-dp-rebuilt"
            return None
        if "libero" in s:
            if cr.k_episodes is None:
                return None
            return min(_LIBERO_DP_K_REGIMES,
                       key=lambda key: (abs(atlas.REGIMES[key]["k"]
                                            - cr.k_episodes),
                                        atlas.REGIMES[key]["k"]))
    return None


def _regime_key(cr):
    """'<env>/<policy_class>' grouping key for the retraining-noise
    decomposition (env collapsed to pusht/libero where recognisable)."""
    s = (cr.suite or "").lower()
    env = "pusht" if "pusht" in s else ("libero" if "libero" in s else s)
    return "%s/%s" % (env, cr.policy_class)


def _atlas_sigma_refs(regime_key):
    """Atlas sigma_run reference values for a '<env>/<policy>' key —
    measured atlas data quoted next to the replay's own measurement; []
    when the atlas has no regime for that policy (e.g. pi05)."""
    env, _, policy_class = regime_key.partition("/")
    pol = _POLICY_TO_ATLAS.get(policy_class.lower())
    if pol is None:
        return []
    return [{"regime": key, "sigma_run_pp": atlas.REGIMES[key]["sigma_run"]}
            for key in sorted(atlas.REGIMES)
            if atlas.REGIMES[key].get("policy") == pol
            and atlas.REGIMES[key].get("env") == env]


def _count_cross_epoch_null_exclusions(cruns):
    """Replicate pairings the stack-epoch key REFUSES to form: same
    epoch-blind null key, different recorded train_seeds, DIFFERENT stack
    epoch — i.e. seed 'replicates' evaluated across the measured gym_pusht
    rebuild. Counted for disclosure; never gated."""
    groups = defaultdict(list)
    for cr in cruns:
        if cr.run.successes and cr.run.set_hash and cr.train_seed is not None:
            groups[_null_key(cr)[:-1]].append(cr)
    n = 0
    for rs in groups.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if rs[i].train_seed != rs[j].train_seed \
                        and stack_epoch(rs[i]) != stack_epoch(rs[j]):
                    n += 1
    return n


def _null_pair_config(base, a, prior, level):
    """Per-gate config for one null pair at the given comparison_level.
    The retrain pass records the auto-resolved prior regime (the engine
    resolves sigma_run from it; an explicit base sigma_run_pp still beats
    it, per the frozen resolution order). The checkpoint pass strips any
    explicit prior — sigma_run_pp at checkpoint level is a config error by
    design ('prior would be silently unused')."""
    cfg = _config_for(base, a)
    if level == "retrain":
        return dataclasses.replace(cfg, comparison_level="retrain",
                                   regime=prior)
    return dataclasses.replace(cfg, comparison_level="checkpoint",
                               regime=prior, sigma_run_pp=None,
                               sigma_run_df=None)


def _gate_null_pass(pairs, base, now, level):
    """One gating pass over the null pairs at comparison_level=level: every
    pair BOTH directions (2 calls — the verdict must be noise-proof
    whichever run history happened to call 'incumbent'). Returns the raw
    pass artifacts; replay_null_pairs assembles the summaries."""
    records, gaps, counts = [], [], Counter()
    group_gates = {}                 # null key -> [n_gates, n_ships]
    regime_gaps = defaultdict(list)  # env/policy -> one delta per pair
    paired_profiles = []             # (n, n_discordant) per CRN-paired pair
    prior_gates = {}                 # resolved prior label -> [gates, ships]
    n_pairs_shipped = 0
    n_cross_wave = 0
    for a, b in pairs:
        key = _null_key(a)
        prior = retrain_prior_regime(a)
        cfg = _null_pair_config(base, a, prior, level)
        extra = _cross_wave_assumptions(a, b)
        if extra:
            n_cross_wave += 1
        gg = group_gates.setdefault(key, [0, 0])
        pg = prior_gates.setdefault(prior or "(no prior)", [0, 0])
        pair_shipped = False
        first_rec = None
        for inc, cand in ((a, b), (b, a)):
            dec = gate(inc.run, cand.run, cfg, now=now,
                       extra_assumptions=extra)
            records.append(dec.record)
            counts[dec.verdict] += 1
            gg[0] += 1
            pg[0] += 1
            if dec.verdict == "SHIP":
                gg[1] += 1
                pg[1] += 1
                pair_shipped = True
            if first_rec is None:
                first_rec = dec.record
            gaps.append({"a": inc.run.run_id, "b": cand.run.run_id,
                         "delta_hat_pp": _delta(dec.record),
                         "verdict": dec.verdict})
        if pair_shipped:
            n_pairs_shipped += 1
        st = first_rec["statistics"]
        if st.get("delta_hat_pp") is not None:
            regime_gaps[_regime_key(a)].append(st["delta_hat_pp"])
        if st.get("paired") and st.get("n_discordant") is not None:
            paired_profiles.append((a.run.n_eval, st["n_discordant"]))
    return {"records": records, "gaps": gaps, "counts": counts,
            "group_gates": group_gates, "regime_gaps": regime_gaps,
            "paired_profiles": paired_profiles, "prior_gates": prior_gates,
            "n_pairs_shipped": n_pairs_shipped,
            "n_cross_wave": n_cross_wave}


def _cluster_ci(group_gates, n_boot, seed):
    """Percentile bootstrap CI of the per-gate false-ship rate, resampling
    REPLICATE GROUPS with replacement — the correct unit of independence
    (seeded, deterministic)."""
    group_list = [group_gates[k]
                  for k in sorted(group_gates,
                                  key=lambda k: tuple(map(str, k)))]
    if not (group_list and n_boot):
        return None
    rng = random.Random(seed)
    g_n = len(group_list)
    rates = []
    for _ in range(n_boot):
        tg = ts = 0
        for _ in range(g_n):
            g, s = group_list[rng.randrange(g_n)]
            tg += g
            ts += s
        rates.append(ts / tg)        # tg > 0: every group holds >= 2 gates
    rates.sort()
    return [rates[int(0.025 * n_boot)],
            rates[min(n_boot - 1, int(0.975 * n_boot))]]


def _null_pass_summary(p, n_pairs, config):
    """Rates + Wilson CIs + cluster bootstrap for one gating pass."""
    n_gates = len(p["records"])
    n_false = p["counts"].get("SHIP", 0)
    by_prior = {}
    for label in sorted(p["prior_gates"]):
        g, s = p["prior_gates"][label]
        by_prior[label] = {
            "n_gates": g, "n_false_ship": s,
            "false_ship_rate": (s / g) if g else None,
            "wilson_ci": list(stats.wilson(s, g)) if g else None}
    return {
        "n_gates": n_gates,
        "verdict_counts": dict(p["counts"]),
        "n_false_ship": n_false,
        "false_ship_rate": (n_false / n_gates) if n_gates else None,
        "wilson_ci": list(stats.wilson(n_false, n_gates)) if n_gates
        else None,
        "n_pairs_shipped": p["n_pairs_shipped"],
        "pair_wilson_ci": list(stats.wilson(p["n_pairs_shipped"], n_pairs))
        if n_pairs else None,
        "cluster_ci": _cluster_ci(p["group_gates"], config.n_boot,
                                  config.boot_seed) if n_gates else None,
        "by_prior": by_prior,
    }


def replay_null_pairs(cruns, config, now=None):
    """Gate every null pair BOTH directions (2 calls per pair — the gate's
    verdict must be noise-proof whichever run history happened to call
    'incumbent'). Selection is forced to 'final': a seed replicate was not
    picked as a max over checkpoints, and the null test must show the
    UNCORRECTED gate's false-ship rate. The ground truth here is that the
    true difference is 0, so every SHIP is a FALSE SHIP.

    v0.1: the PRIMARY pass gates at comparison_level='retrain' with the
    measured atlas sigma_run prior auto-resolved per pair
    (retrain_prior_regime); the SAME pairs are then re-gated at
    comparison_level='checkpoint' (the v0 eval-only semantics) and that
    pass's summary lands in 'eval_only', so the report can render the
    BEFORE/AFTER false-ship table (overall and per resolved prior, Wilson
    CIs and cluster bootstrap both sides). Only the primary pass's records
    are returned/ledgered.

    Besides the spec-mandated per-gate Wilson CI, the result DISCLOSES the
    dependence structure the Wilson CI ignores (gates are mirrored
    direction-pairs sharing runs within replicate groups) and computes:

      pair_wilson_ci   Wilson CI over unordered pairs (a pair false-ships
                       if EITHER direction ships)
      cluster_ci       percentile bootstrap CI of the per-gate rate,
                       resampling REPLICATE GROUPS with replacement — the
                       correct unit of independence (seeded, n_boot reps)
      retrain_noise_pp sigma_run implied by the null gaps per env/policy
                       regime: stats.sigma_from_gaps (= sqrt(mean(g^2)/2)),
                       one gap per unordered pair, quoted beside the atlas
                       sigma_run references
      calibration      synthetic exact-null control of the PAIRED McNemar
                       path: c* ~ Binomial(n_d, 1/2) redrawn at the
                       observed (n, n_d) profiles, realised rate of
                       p_ship <= alpha and of the full ship rule (p AND
                       min-effect; single-block, no curse) — regenerated
                       with every report, never asserted. Level-blind by
                       construction: it exercises the eval-level exact
                       McNemar path (recorded as p_eval at retrain level).
      n_cross_wave_pairs / n_cross_epoch_excluded  stack-epoch bookkeeping
    """
    base = dataclasses.replace(config, selection="final", n_checkpoints=None)
    pairs = find_null_pairs(cruns)
    after = _gate_null_pass(pairs, base, now, "retrain")
    before = _gate_null_pass(pairs, base, now, "checkpoint")
    after_sum = _null_pass_summary(after, len(pairs), config)
    before_sum = _null_pass_summary(before, len(pairs), config)

    retrain_noise = {}
    regime_gaps = after["regime_gaps"]   # delta_hat is level-independent
    for rk in sorted(regime_gaps):
        gs = regime_gaps[rk]
        retrain_noise[rk] = {"n_pairs": len(gs),
                             "sigma_run_pp": stats.sigma_from_gaps(gs),
                             "atlas_refs": _atlas_sigma_refs(rk)}

    # exact-null calibration control (paired path); boot_seed+1 keeps it
    # independent of the cluster bootstrap stream, still deterministic
    calibration = None
    paired_profiles = after["paired_profiles"]
    if paired_profiles and config.n_boot:
        rng = random.Random(config.boot_seed + 1)
        n_sig = n_ship_rule = 0
        for i in range(config.n_boot):
            n, nd = paired_profiles[i % len(paired_profiles)]
            c_star = sum(1 for _ in range(nd) if rng.random() < 0.5)
            b_star = nd - c_star
            if stats.mcnemar_one_sided(c_star, b_star) <= config.alpha:
                n_sig += 1
                if 100.0 * (c_star - b_star) / n >= config.min_effect_pp:
                    n_ship_rule += 1
        calibration = {"n_profiles": len(paired_profiles),
                       "n_sims": config.n_boot,
                       "n_sig": n_sig,
                       "sig_rate": n_sig / config.n_boot,
                       "n_ship_rule": n_ship_rule,
                       "ship_rule_rate": n_ship_rule / config.n_boot}

    gaps = after["gaps"]
    gaps.sort(key=lambda g: (-_abs_or_zero(g["delta_hat_pp"]),
                             g["a"], g["b"]))
    return {"n_pairs": len(pairs), "n_gates": after_sum["n_gates"],
            "comparison_level": "retrain",
            "n_groups": len(after["group_gates"]),
            "verdict_counts": after_sum["verdict_counts"],
            "n_false_ship": after_sum["n_false_ship"],
            "false_ship_rate": after_sum["false_ship_rate"],
            "wilson_ci": after_sum["wilson_ci"], "alpha": config.alpha,
            "n_pairs_shipped": after_sum["n_pairs_shipped"],
            "pair_wilson_ci": after_sum["pair_wilson_ci"],
            "cluster_ci": after_sum["cluster_ci"],
            "by_prior": after_sum["by_prior"],
            "eval_only": before_sum,
            "retrain_noise_pp": retrain_noise,
            "calibration": calibration,
            "n_cross_wave_pairs": after["n_cross_wave"],
            "n_cross_epoch_excluded": _count_cross_epoch_null_exclusions(cruns),
            "top_null_gaps": gaps, "records": after["records"]}


# ---------------------------------------------------------------- curse mode

def _cell_key(cr):
    return (cr.wave, cr.policy_class, cr.suite, cr.task, cr.run.n_eval,
            cr.run.seed, cr.steps)


def _cell_str(key):
    wave, pol, suite, task, n, seed, steps = key
    return "%s/%s/%s/%s/n%s/seed%s/steps%s" % (wave, pol, suite, task, n,
                                               seed, steps)


def replay_curse(cruns, config, now=None):
    """Max-over-checkpoints selection bias, demonstrated on cells (wave,
    policy_class, suite, task, n_eval, eval_seed, steps) with >= 3 runs
    carrying per-episode successes (runs without a recorded sr cannot be
    ranked and are excluded — selection is by measured SR). The median-SR
    run plays incumbent (even count -> lower median; ties broken by
    run_id); the candidate is the argmax-SR run of the remaining pool of
    J runs — exactly the 'pick the best-looking checkpoint' move that
    PI05_CELL1_RESULTS measured at +4.00 pp mean inflation for J=4. Each
    cell is gated twice: uncorrected (selection='final') and corrected
    (selection='max-over-checkpoints' with the full pool as history ->
    history-bootstrap correction, no scaling assumption needed)."""
    groups = defaultdict(list)
    for cr in cruns:
        if cr.run.successes and cr.run.sr is not None:
            groups[_cell_key(cr)].append(cr)
    cells = []
    for key in sorted(groups, key=lambda k: tuple(map(str, k))):
        rs = groups[key]
        if len(rs) < 3:
            continue
        by_sr = sorted(rs, key=lambda cr: (cr.run.sr, cr.run.run_id))
        incumbent = by_sr[(len(by_sr) - 1) // 2]   # lower median
        pool = [cr for cr in by_sr if cr is not incumbent]
        best_sr = max(cr.run.sr for cr in pool)
        candidate = min((cr for cr in pool if cr.run.sr == best_sr),
                        key=lambda cr: cr.run.run_id)
        j = len(pool)
        cfg = _config_for(config, incumbent)
        cfg_unc = dataclasses.replace(cfg, selection="final",
                                      n_checkpoints=None)
        cfg_cor = dataclasses.replace(cfg, selection="max-over-checkpoints",
                                      n_checkpoints=j)
        dec_unc = gate(incumbent.run, candidate.run, cfg_unc, now=now)
        dec_cor = gate(incumbent.run, candidate.run, cfg_cor,
                       history=[p.run for p in pool], now=now)
        cells.append({
            "cell": _cell_str(key), "j": j,
            "incumbent": incumbent.run.run_id,
            "candidate": candidate.run.run_id,
            "sr_incumbent_pp": incumbent.run.sr,
            "sr_candidate_pp": candidate.run.sr,
            "delta_hat_pp": _delta(dec_unc.record),
            "correction_prior_pp": curse_from_prior(j, config.curse_prior_pp,
                                                    config.curse_prior_j),
            "correction_history_pp":
                dec_cor.record["statistics"].get("curse_correction_pp"),
            "verdict_uncorrected": dec_unc.verdict,
            "verdict_corrected": dec_cor.verdict,
            "records": [dec_unc.record, dec_cor.record],
        })
    return {"n_cells": len(cells), "cells": cells}


# ---------------------------------------------------------------- effect mode

def _effect_cell_key(cr):
    """Effect-cell key; stack_epoch LAST so key[:-1] is the epoch-blind key
    the cross-epoch exclusion counter groups on."""
    return (cr.policy_class, cr.suite, cr.task, cr.run.n_eval, cr.run.seed,
            cr.steps, stack_epoch(cr))


def replay_effects(cruns, config, now=None):
    """Real-effect pairs: same cell (policy_class, suite, task, n_eval,
    eval_seed, steps, stack_epoch), DIFFERENT set_hash — different training
    sets, so the gate faces a possibly-real difference. One representative
    per set_hash (min recorded train_seed, unrecorded last; ties by run_id)
    so replicate retrains do not multiply-count an effect; all unordered
    representative pairs gated both directions with selection='final'.
    Reports how often the gate RESOLVES (SHIP or HOLD) vs
    COLLECT-MORE/UNRESOLVABLE, and the implied additional episode budgets.

    v0.1: every gate runs at comparison_level='retrain' with the measured
    atlas sigma_run prior auto-resolved from the first representative
    (retrain_prior_regime; same cell = same policy/suite/epoch) — a
    different-training-set candidate arrives via an independent retrain, so
    the retraining lottery is priced into these verdicts. Pairs with no
    measured prior gate with the recorded no-prior disclosure. On the
    retrains-needed sizing path episodes_needed is None (excluded from the
    episode-budget summary).

    The stack_epoch key component keeps effect pairs from straddling the
    measured gym_pusht rebuild — a cross-rebuild 'delta' mixes policy
    quality with harness drift (STACK_SHIFT_PUSHT_PP = -18.2 pp mean,
    non-uniform) and would count harness artifacts as training-set effects.
    Refused pairings are counted (n_cross_epoch_excluded) and disclosed;
    cross-wave pairs within one epoch carry a recorded ASSUMPTION."""
    base = dataclasses.replace(config, selection="final", n_checkpoints=None)
    groups = defaultdict(lambda: defaultdict(list))
    for cr in cruns:
        if cr.run.successes and cr.run.set_hash:
            groups[_effect_cell_key(cr)][cr.run.set_hash].append(cr)
    reps_by_key = {}
    for key in sorted(groups, key=lambda k: tuple(map(str, k))):
        by_hash = groups[key]
        reps = []
        for h in sorted(by_hash):
            rs = sorted(by_hash[h],
                        key=lambda cr: (cr.train_seed is None, cr.train_seed,
                                        cr.run.run_id))
            reps.append(rs[0])
        reps.sort(key=lambda cr: cr.run.run_id)
        reps_by_key[key] = reps
    records, entries, counts, needed = [], [], Counter(), []
    n_pairs = 0
    n_cross_wave = 0
    for key in sorted(reps_by_key, key=lambda k: tuple(map(str, k))):
        reps = reps_by_key[key]
        if len(reps) < 2:
            continue
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                n_pairs += 1
                cfg = dataclasses.replace(
                    _config_for(base, reps[i]), comparison_level="retrain",
                    regime=retrain_prior_regime(reps[i]))
                extra = _cross_wave_assumptions(reps[i], reps[j])
                if extra:
                    n_cross_wave += 1
                for inc, cand in ((reps[i], reps[j]), (reps[j], reps[i])):
                    dec = gate(inc.run, cand.run, cfg, now=now,
                               extra_assumptions=extra)
                    records.append(dec.record)
                    counts[dec.verdict] += 1
                    ep = dec.record["statistics"].get("episodes_needed")
                    if dec.verdict in ("COLLECT-MORE", "UNRESOLVABLE"):
                        if ep is not None:
                            needed.append(ep)
                    entries.append({"a": inc.run.run_id, "b": cand.run.run_id,
                                    "delta_hat_pp": _delta(dec.record),
                                    "verdict": dec.verdict,
                                    "episodes_needed": ep})
    # cross-epoch representative pairings the epoch key refused to form
    # (different set_hash across different stack epochs of one epoch-blind
    # cell) — counted for disclosure, never gated
    no_epoch = defaultdict(list)
    for key, reps in reps_by_key.items():
        no_epoch[key[:-1]].append(reps)
    n_cross_epoch_excluded = 0
    for rep_lists in no_epoch.values():
        for x in range(len(rep_lists)):
            for y in range(x + 1, len(rep_lists)):
                for ra in rep_lists[x]:
                    for rb in rep_lists[y]:
                        if ra.run.set_hash != rb.run.set_hash:
                            n_cross_epoch_excluded += 1
    n_gates = len(records)
    resolved = counts.get("SHIP", 0) + counts.get("HOLD", 0)
    entries.sort(key=lambda g: (-_abs_or_zero(g["delta_hat_pp"]),
                                g["a"], g["b"]))
    return {
        "n_pairs": n_pairs, "n_gates": n_gates,
        "comparison_level": "retrain",
        "verdict_counts": dict(counts),
        "resolved_rate": (resolved / n_gates) if n_gates else None,
        "episodes_needed": {
            "min": min(needed) if needed else None,
            "median": _median(needed) if needed else None,
            "max": max(needed) if needed else None,
        },
        "n_cross_wave_pairs": n_cross_wave,
        "n_cross_epoch_excluded": n_cross_epoch_excluded,
        "pairs": entries, "records": records,
    }


# ---------------------------------------------------------------- report

def _fmt(x, spec="%.2f"):
    return spec % x if x is not None else "-"


def _verdict_line(counts):
    order = ["SHIP", "HOLD", "INVALID", "COLLECT-MORE", "UNRESOLVABLE"]
    parts = ["%s=%d" % (v, counts.get(v, 0)) for v in order if v in counts]
    return ", ".join(parts) if parts else "none"


def _all_records(results):
    for mode in ("null", "curse", "effect"):
        res = results.get(mode)
        if not res:
            continue
        if mode == "curse":
            for cell in res["cells"]:
                for rec in cell["records"]:
                    yield rec
        else:
            for rec in res["records"]:
                yield rec


def render_report(results, config):
    """REPLAY_REPORT.md from a :func:`run_replay` results dict. Every number
    is formatted from ``results`` — none hard-coded, none asserted."""
    meta = results.get("meta", {})
    L = []
    p = L.append
    p("# Ship-Gate REPLAY report")
    p("")
    p("Historical eval corpus replayed through the Ship-Gate v0.1 "
      "(fixed-sample; retrain-aware primary at null/effect, checkpoint "
      "BEFORE pass for the v0 comparison; sequential is v1). Every number "
      "below is computed from the input files; none asserted.")
    p("")
    p("- files: %s" % (", ".join("`%s`" % pt for pt in meta.get("paths", []))
                       or "-"))
    p("- runs parsed: %s (parse errors: %s)"
      % (meta.get("n_runs", 0), meta.get("n_errors", 0)))
    p("- engine_version: %s" % meta.get("engine_version"))
    p("- gate_config sha256: `%s`" % config.sha256())
    p("- alpha: %g (one-sided) | min_effect: %g pp (corpus-measured "
      "resolvability floor: below ~2 pp no rollout budget separates "
      "candidates — ADAPTIVE_EVAL_PROGRAMME)"
      % (config.alpha, config.min_effect_pp))
    p("- created: %s | modes: %s"
      % (meta.get("created"), ", ".join(meta.get("modes", ())) or "-"))
    if config.sequential_eval:
        p("- pairing: sequential (non-vectorized) eval declared — CRN "
          "pairing holds on LIBERO suites.")
    else:
        degraded = any(
            rec["statistics"].get("paired") is False and any(
                "autoreset" in c for c in rec.get("caveats", []))
            for rec in _all_records(results))
        if degraded:
            p("- pairing: LIBERO suites present WITHOUT sequential_eval "
              "declared — CRN pairing DEGRADED to UNPAIRED on those gates "
              "(LIBERO vectorized autoreset makes init states "
              "outcome-dependent under early termination; LeRobot issue "
              "#4152).")
        else:
            p("- pairing: CRN-paired wherever eval seeds verify; no LIBERO "
              "pairing degrade triggered.")
    errors = meta.get("errors") or []
    if errors:
        p("")
        p("Parse errors:")
        for e in errors:
            p("- `%s:%s`: %s" % (e["path"], e["line"], e["error"]))
    p("")

    # ---------------- NULL
    p("## NULL-PAIR mode — seed replicates, true difference = 0")
    p("")
    null = results.get("null")
    if not null:
        p("_mode not run_")
        p("")
    else:
        p("Runs trained on the IDENTICAL episode set (same set_hash) with a "
          "different train_seed: any measured gap is pure retraining noise, "
          "so every SHIP verdict here is a FALSE SHIP.")
        p("")
        p("- null pairs: %d, gate calls (both directions): %d, replicate "
          "groups: %d" % (null["n_pairs"], null["n_gates"],
                          null.get("n_groups", 0)))
        if null.get("comparison_level") == "retrain":
            p("- comparison_level: retrain (v0.1 primary) — the "
              "retraining-lottery prior sigma_run is auto-resolved per "
              "pair from the measured atlas; the SAME pairs re-gated at "
              "checkpoint level (the v0 eval-only semantics) render the "
              "BEFORE/AFTER table below. Pairs with no measured prior "
              "gate with the recorded no-prior disclosure.")
        p("- stack epochs: cross-epoch replicate pairings excluded by the "
          "wave->epoch key (measured gym_pusht rebuild, OPS_RUNBOOK sec 9): "
          "%d; cross-wave pairs within an epoch: %d (each record carries "
          "the identical-stack ASSUMPTION)"
          % (null.get("n_cross_epoch_excluded", 0),
             null.get("n_cross_wave_pairs", 0)))
        p("- verdicts: %s" % _verdict_line(null["verdict_counts"]))
        if null["false_ship_rate"] is not None:
            lo, hi = null["wilson_ci"]
            rate = null["false_ship_rate"]
            p("- **FALSE-SHIP RATE (retrain-aware): %.4f** (%d/%d), Wilson "
              "95%% CI [%.4f, %.4f] — configured alpha %g (%s alpha)"
              % (rate, null["n_false_ship"], null["n_gates"], lo, hi,
                 null["alpha"],
                 "rate <=" if rate <= null["alpha"] else "rate ABOVE"))
            p("- dependence structure (DISCLOSED — the Wilson CI above "
              "treats gate calls as independent Bernoulli trials, which "
              "they are not): %d gates = %d mirrored direction-pairs (at "
              "most one direction of a pair can SHIP) drawn from %d "
              "replicate groups whose pairs share runs, so the per-gate "
              "Wilson interval is too NARROW."
              % (null["n_gates"], null["n_pairs"], null.get("n_groups", 0)))
            if null.get("pair_wilson_ci"):
                plo, phi = null["pair_wilson_ci"]
                p("- pair-level false ships (a pair false-ships if EITHER "
                  "direction ships): %d/%d, Wilson 95%% CI [%.4f, %.4f]"
                  % (null.get("n_pairs_shipped", 0), null["n_pairs"],
                     plo, phi))
            if null.get("cluster_ci"):
                clo, chi = null["cluster_ci"]
                p("- cluster bootstrap over the %d replicate groups (the "
                  "correct unit of independence): per-gate false-ship rate "
                  "95%% CI [%.4f, %.4f] (%s alpha at the lower bound)"
                  % (null.get("n_groups", 0), clo, chi,
                     "still ABOVE" if clo > null["alpha"]
                     else "compatible with"))
            eo = null.get("eval_only")
            if eo and eo.get("false_ship_rate") is not None:
                p("")
                p("### BEFORE/AFTER — eval-only (v0 checkpoint semantics) "
                  "vs retrain-aware (v0.1)")
                p("")
                p("The same null pairs gated twice: BEFORE at "
                  "comparison_level='checkpoint' (the v0 semantics — alpha "
                  "bounds eval-sampling error only) and AFTER at "
                  "comparison_level='retrain' with the measured atlas "
                  "sigma_run prior auto-resolved per pair. Every SHIP is a "
                  "false ship; the per-gate Wilson CIs below inherit the "
                  "dependence caveat disclosed above.")
                p("")
                p("| regime (sigma_run prior) | gates | false ships BEFORE "
                  "(eval-only) | BEFORE rate [Wilson 95% CI] | false ships "
                  "AFTER (retrain-aware) | AFTER rate [Wilson 95% CI] |")
                p("|---|---:|---:|---|---:|---|")
                bp_b = eo.get("by_prior") or {}
                bp_a = null.get("by_prior") or {}

                def _rate_cell(row):
                    if not row or not row.get("n_gates"):
                        return "-"
                    wlo, whi = row["wilson_ci"]
                    return "%.4f [%.4f, %.4f]" % (row["false_ship_rate"],
                                                  wlo, whi)

                for label in sorted(set(bp_b) | set(bp_a)):
                    rb, ra = bp_b.get(label), bp_a.get(label)
                    sig = atlas.REGIMES.get(label, {}).get("sigma_run")
                    p("| %s | %d | %d | %s | %d | %s |"
                      % (label if sig is None
                         else "%s (%.2f pp)" % (label, sig),
                         (ra or rb)["n_gates"],
                         rb["n_false_ship"] if rb else 0, _rate_cell(rb),
                         ra["n_false_ship"] if ra else 0, _rate_cell(ra)))
                p("| **overall** | %d | %d | %s | %d | %s |"
                  % (null["n_gates"], eo["n_false_ship"],
                     _rate_cell(eo), null["n_false_ship"],
                     _rate_cell(null)))
                if eo.get("cluster_ci") and null.get("cluster_ci"):
                    p("")
                    p("- cluster bootstrap over the %d replicate groups, "
                      "both semantics: BEFORE [%.4f, %.4f], AFTER "
                      "[%.4f, %.4f]"
                      % (null.get("n_groups", 0), eo["cluster_ci"][0],
                         eo["cluster_ci"][1], null["cluster_ci"][0],
                         null["cluster_ci"][1]))
                if eo.get("pair_wilson_ci"):
                    plo, phi = eo["pair_wilson_ci"]
                    p("- pair-level false ships BEFORE (eval-only): %d/%d, "
                      "Wilson 95%% CI [%.4f, %.4f]"
                      % (eo.get("n_pairs_shipped", 0), null["n_pairs"],
                         plo, phi))
                # Regimes whose AFTER (retrain-aware) rate still sits above
                # alpha are called out EXPLICITLY — not left derivable from
                # the table plus the noise decomposition below (data-driven,
                # never asserted).
                rn_after = null.get("retrain_noise_pp") or {}
                for label in sorted(bp_a):
                    ra = bp_a[label]
                    if not ra.get("n_gates") \
                            or ra.get("false_ship_rate") is None \
                            or ra["false_ship_rate"] <= null["alpha"]:
                        continue
                    wlo, whi = ra["wilson_ci"]
                    sig_atlas = atlas.REGIMES.get(label, {}).get("sigma_run")
                    measured = None
                    for rk in sorted(rn_after):
                        refs = rn_after[rk].get("atlas_refs") or []
                        if any(r.get("regime") == label for r in refs):
                            measured = (rk, rn_after[rk].get("sigma_run_pp"))
                            break
                    line = ("- ABOVE-ALPHA AFTER rate — %s: %d/%d = %.4f "
                            "[Wilson %.4f, %.4f] still exceeds alpha %g "
                            "under retrain-aware semantics"
                            % (label, ra["n_false_ship"], ra["n_gates"],
                               ra["false_ship_rate"], wlo, whi,
                               null["alpha"]))
                    if measured and measured[1] is not None \
                            and sig_atlas is not None:
                        line += (" — measured-vs-atlas sigma_run gap (noise "
                                 "table below): these null pairs measure "
                                 "sigma_run %.2f pp (%s) against the %.2f "
                                 "pp atlas prior the gate priced%s"
                                 % (measured[1], measured[0], sig_atlas,
                                    ", so the prior under-prices the "
                                    "retraining lottery here."
                                    if measured[1] > sig_atlas else "."))
                    else:
                        line += ("; no measured sigma_run decomposition "
                                 "matches this regime.")
                    p(line)
        else:
            p("- no gateable null pairs found")
        if null["false_ship_rate"] is not None:
            p("")
            p("What a rate above alpha MEANS here (computed, not asserted): "
              "the fixed-sample gate's alpha bounds EVAL-SAMPLING error "
              "only — 'could this gap be an accident of the shared eval "
              "draw?'. Seed replicates are genuinely DIFFERENT policies, "
              "separated by the retraining lottery (sigma_run), which no "
              "CRN eval can cancel; a false-ship rate above alpha on null "
              "pairs therefore QUANTIFIES retraining noise rather than "
              "indicating a mis-calibrated test (see the exact-null control "
              "below). min_effect_pp = %g pp is the EVAL resolvability "
              "floor (ADAPTIVE_EVAL_PROGRAMME), NOT a retraining-noise "
              "floor: bounding promotion error against the retrain lottery "
              "needs min_effect_pp to clear ~sqrt(2)*sigma_run for the "
              "regime, or multiple retrains per side — a disclosed v0 "
              "semantics gap (sigma_run lives in training; CRN pairing "
              "cancels eval-draw noise only)." % config.min_effect_pp)
            rn = null.get("retrain_noise_pp") or {}
            if rn:
                p("")
                p("Retraining noise measured from THESE null pairs "
                  "(sigma_run = sqrt(mean(gap^2)/2), stats.sigma_from_gaps, "
                  "one gap per unordered pair), beside the atlas "
                  "measurements:")
                p("")
                p("| regime (env/policy) | null pairs | sigma_run measured "
                  "(pp) | atlas sigma_run (pp) |")
                p("|---|---:|---:|---|")
                for rk in sorted(rn):
                    row = rn[rk]
                    refs = row.get("atlas_refs") or []
                    ref_txt = "; ".join(
                        "%s %.2f" % (r["regime"], r["sigma_run_pp"])
                        for r in refs if r.get("sigma_run_pp") is not None) \
                        or "no atlas regime measured"
                    p("| %s | %d | %s | %s |"
                      % (rk, row["n_pairs"],
                         _fmt(row["sigma_run_pp"]), ref_txt))
            cal = null.get("calibration")
            if cal:
                p("")
                p("- exact-null calibration control (regenerated with this "
                  "report, seeded): redrawing the discordant split c* ~ "
                  "Binomial(n_d, 1/2) at the %d observed CRN-paired (n, "
                  "n_d) profiles, %d sims: p_ship <= alpha in %.4f of "
                  "sims; full ship rule (p AND min-effect, single-block, "
                  "no curse) in %.4f — %s the configured alpha %g, so the "
                  "paired McNemar path itself is %s. The elevated rate on "
                  "the REAL null pairs is a property of the data "
                  "(sigma_run), not of the test."
                  % (cal["n_profiles"], cal["n_sims"], cal["sig_rate"],
                     cal["ship_rule_rate"],
                     "at or below" if cal["sig_rate"] <= null["alpha"]
                     else "ABOVE",
                     null["alpha"],
                     "calibrated (discreteness makes it conservative)"
                     if cal["sig_rate"] <= null["alpha"]
                     else "MIS-CALIBRATED — investigate before trusting "
                          "any verdict"))
            else:
                p("")
                p("- exact-null calibration control: no CRN-paired null "
                  "pairs in this corpus (all degraded to unpaired), so the "
                  "paired-path control has nothing to resample here.")
        gaps = null["top_null_gaps"][:10]
        if gaps:
            p("")
            p("Top null gaps — pure retraining noise, measured in pp:")
            p("")
            p("| incumbent | candidate | delta_hat (pp) | verdict |")
            p("|---|---|---:|---|")
            for g in gaps:
                p("| %s | %s | %s | %s |"
                  % (g["a"], g["b"], _fmt(g["delta_hat_pp"], "%+.2f"),
                     g["verdict"]))
        if null["records"]:
            ships = [r for r in null["records"] if r["verdict"] == "SHIP"]
            example = ships[0] if ships else null["records"][0]
            p("")
            p("Example decision record (%s):"
              % ("a FALSE SHIP" if ships else "first gate call"))
            p("")
            p("```json")
            p(json.dumps(example, indent=2, sort_keys=True))
            p("```")
        p("")

    # ---------------- CURSE
    p("## CURSE mode — max-over-checkpoints selection bias")
    p("")
    curse = results.get("curse")
    if not curse:
        p("_mode not run_")
        p("")
    else:
        p("Candidate = argmax-SR run of a >= 3-run cell pool: the 'best "
          "looking checkpoint' move. Corrected gates subtract the "
          "history-bootstrap optimism of that max (PI05_CELL1_RESULTS "
          "measured +4.00 pp mean at J=4, prospectively).")
        p("")
        p("- cells: %d" % curse["n_cells"])
        if curse["cells"]:
            p("")
            p("| cell | J | incumbent | candidate | sr_inc | sr_cand | "
              "delta (pp) | prior corr (pp) | history corr (pp) | "
              "uncorrected | corrected |")
            p("|---|---:|---|---|---:|---:|---:|---:|---:|---|---|")
            for c in curse["cells"]:
                p("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                  % (c["cell"], c["j"], c["incumbent"], c["candidate"],
                     _fmt(c["sr_incumbent_pp"], "%.1f"),
                     _fmt(c["sr_candidate_pp"], "%.1f"),
                     _fmt(c["delta_hat_pp"], "%+.2f"),
                     _fmt(c["correction_prior_pp"]),
                     _fmt(c["correction_history_pp"]),
                     c["verdict_uncorrected"], c["verdict_corrected"]))
        p("")

    # ---------------- EFFECT
    p("## EFFECT mode — different training sets, same cell")
    p("")
    effect = results.get("effect")
    if not effect:
        p("_mode not run_")
        p("")
    else:
        p("- representative pairs: %d, gate calls: %d"
          % (effect["n_pairs"], effect["n_gates"]))
        p("- stack epochs: cross-epoch pairings excluded by the wave->epoch "
          "key: %d — a cross-rebuild 'delta' would mix policy quality with "
          "the measured %.1f pp gym_pusht harness shift and count harness "
          "artifacts as training-set effects [atlas CAVEATS['cross-stack'], "
          "OPS_RUNBOOK sec 9]. Cross-wave pairs within an epoch: %d (each "
          "record carries the identical-stack ASSUMPTION)."
          % (effect.get("n_cross_epoch_excluded", 0),
             atlas.STACK_SHIFT_PUSHT_PP,
             effect.get("n_cross_wave_pairs", 0)))
        p("- verdicts: %s" % _verdict_line(effect["verdict_counts"]))
        if effect["resolved_rate"] is not None:
            p("- resolved rate (SHIP or HOLD): %.4f"
              % effect["resolved_rate"])
        en = effect["episodes_needed"]
        p("- additional episodes needed on COLLECT-MORE/UNRESOLVABLE gates: "
          "min %s / median %s / max %s"
          % (_fmt(en["min"], "%d"),
             _fmt(en["median"], "%g"),
             _fmt(en["max"], "%d")))
        if effect["pairs"]:
            p("")
            p("Wins inside noise — every pair, |delta| descending; a verdict "
              "other than SHIP/HOLD means the observed gap was NOT outside "
              "measured noise at this budget:")
            p("")
            p("| incumbent | candidate | delta_hat (pp) | verdict | "
              "episodes needed |")
            p("|---|---|---:|---|---:|")
            for g in effect["pairs"]:
                p("| %s | %s | %s | %s | %s |"
                  % (g["a"], g["b"], _fmt(g["delta_hat_pp"], "%+.2f"),
                     g["verdict"], _fmt(g["episodes_needed"], "%d")))
        p("")

    return "\n".join(L)


# ---------------------------------------------------------------- driver

def run_replay(paths, config=None, modes=("null", "curse", "effect"),
               out_dir=None, ledger_path=None, now=None):
    """Load the corpus, run the requested modes, render REPLAY_REPORT.md.

    One ``created`` timestamp is stamped for ALL records (``now`` verbatim,
    or current UTC computed exactly once — records.py itself never reads
    the clock). Per-gate configs are derived with dataclasses.replace; the
    caller's config is never mutated. If ``out_dir`` is given the report is
    written to ``<out_dir>/REPLAY_REPORT.md``; if ``ledger_path`` is given
    every produced decision record is appended (append-only JSONL,
    re-verifiable with records.verify_ledger)."""
    paths = list(paths)
    if config is None:
        config = GateConfig()
    created = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cruns, errors = load_corpus(paths)
    results = {
        "meta": {"paths": paths, "n_runs": len(cruns),
                 "n_errors": len(errors), "errors": errors,
                 "engine_version": _engine_version(),
                 "config_sha256": config.sha256(), "created": created,
                 "modes": list(modes)},
        "null": None, "curse": None, "effect": None,
    }
    if "null" in modes:
        results["null"] = replay_null_pairs(cruns, config, now=created)
    if "curse" in modes:
        results["curse"] = replay_curse(cruns, config, now=created)
    if "effect" in modes:
        results["effect"] = replay_effects(cruns, config, now=created)

    if ledger_path:
        for rec in _all_records(results):
            append_record(rec, ledger_path)

    report = render_report(results, config)
    report_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "REPLAY_REPORT.md")
        with open(report_path, "w") as fh:
            fh.write(report)

    results["report"] = report
    results["report_path"] = report_path
    return results


def _engine_version():
    from .. import __version__
    return __version__
