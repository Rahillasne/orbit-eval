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

"""The Ship-Gate verdict engine — v0, FIXED-SAMPLE.

Given an incumbent and a candidate policy eval (io.EvalRun), run the CRN
pre-check ladder, the paired primary test, winner's-curse handling, per-task
regression flags, and COLLECT-MORE sizing; emit a signed decision record
(records.build_record) with every statistical choice DISCLOSED — caveats,
warnings, and 'ASSUMPTION: ' strings are part of the record, not log noise.

v0 is fixed-sample: the alpha is honest for ONE look at a pre-sized eval.
Anytime-valid SEQUENTIAL testing (peek at the eval as it grows) is explicitly
v1 — re-running this gate on a growing eval inflates the false-ship rate.

Nothing statistical is reimplemented; the validated orbit_eval core is reused:
  stats.budget_gate          OPS_RUNBOOK sec 3/4 truncation gate (directional!)
  stats.discordant_counts    paired (b, c) from aligned per-episode successes
  stats.mcnemar_one_sided    exact one-sided McNemar on discordant pairs
  stats.paired_boot_ci       episode bootstrap CI, within-episode pairing kept
  stats.two_prop_test        the unpaired degrade path
  stats.fisher_exact_one_sided  exact per-task test on the degrade path
                             (blocks are ~20 episodes: the pooled-z Gaussian
                             is uncontrolled there, the exact tail is not)
  stats.newcombe_ci          unpaired CI (Wilson-based)
  stats.norm_q / stats.Phi   normal quantile / CDF for the sizing formulas
  atlas.CAVEATS              measured runtime caveats, quoted VERBATIM

Delta convention (frozen): delta = candidate - incumbent, pp.
b = incumbent-only successes, c = candidate-only successes;
p_ship = mcnemar_one_sided(c, b)   evidence the candidate is BETTER
p_worse = mcnemar_one_sided(b, c)  evidence the candidate is WORSE

Winner's curse: if the candidate was picked as max over J checkpoints, its
measured delta is optimistic. With the checkpoint eval history we MEASURE the
optimism (bootstrap, curse_from_history); without it we fall back to the
atlas prior (+4.0 pp mean at J=4, SmolVLA-ft, measured prospectively —
PI05_CELL1_RESULTS) scaled as 4.0*sqrt(ln(J)/ln(4)) pp, a DISCLOSED
assumption. The correction reduces the delta the SHIP rule sees; it never
manufactures a HOLD.

v0.1 "retrain-aware" mode (comparison_level='retrain'): the v0 alpha bounds
EVAL-sampling error only — on the 193 real ORBIT seed-replicate null pairs
it false-shipped 62/386 (16%), while the exact-null control sat at 4.0% <=
alpha (validation/REPLAY_REPORT.md): the gap is SEMANTICS (the retraining
lottery sigma_run, which CRN evals cannot cancel), not calibration. With a
sigma_run prior (explicit or atlas-resolved via config.regime) the governing
p-values become the normal approximation (method 'z-retrain-aware') on
se_total = sqrt(se_eval^2 + 2*sigma_run^2) — one independent retrain per
side; the eval-level exact p is still computed and recorded as p_eval. The
paired se_eval uses the H0 discordance variance p_d/n (conservative under
H1). Sizing gains the 1-retrain/side MDE floor (z_a+z_p)*sqrt(2)*sigma_run;
below it, episodes cannot resolve and COLLECT-MORE is priced in RETRAINS
per side. Without a prior the gate still runs, with v0 statistics plus a
recorded disclosure. The ship rule is unchanged in FORM.
"""

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from .. import atlas, stats
from .records import build_record

# Verdict -> process exit code (frozen).
_EXIT_CODES = {"SHIP": 0, "HOLD": 1, "INVALID": 2,
               "COLLECT-MORE": 3, "UNRESOLVABLE": 4}

# Measured resolvability floor [ADAPTIVE_EVAL_PROGRAMME]. Attached whenever
# |delta_effective_pp| < 2.0 and the verdict is COLLECT-MORE / UNRESOLVABLE:
# below the floor, no budget the corpus ever tested tells the runs apart.
CAVEAT_FLOOR_2PP = (
    "delta < 2 pp resolves at 42-63% accuracy at every budget tested "
    "(m<=100) — a coin flip; no rollout budget separates candidates inside "
    "~2 pp [ADAPTIVE_EVAL_PROGRAMME]"
)


class GateInputError(ValueError):
    """Malformed CONFIG or caller misuse (config.validate() fails; history
    passed with selection='final'; selection='max-over-checkpoints' with
    neither n_checkpoints nor history). Bad run DATA never raises — it yields
    a verdict-'INVALID' record so the refusal itself is ledger-auditable."""


@dataclass
class GateDecision:
    """The gate's answer: verdict + the signed record + human reasons."""
    verdict: str
    record: dict
    reasons: List[str]

    @property
    def exit_code(self):
        """SHIP=0 HOLD=1 INVALID=2 COLLECT-MORE=3 UNRESOLVABLE=4."""
        return _EXIT_CODES[self.verdict]


# ---------------------------------------------------------------- curse

def curse_from_prior(j, prior_pp=4.0, prior_j=4):
    """Winner's-curse correction (pp) from the atlas-measured prior.

    PI05_CELL1_RESULTS: max-over-J=4-checkpoints inflated the measured SR by
    +4.00 pp mean (max +12.5), measured PROSPECTIVELY. For other J we scale
    as prior_pp * sqrt(ln(j)/ln(prior_j)) — the sqrt-log growth of the
    expected max of correlated draws. The scaling beyond J=prior_j is an
    ASSUMPTION, not a measurement; gate() discloses it in the record.
    j <= 1 means no selection happened: correction 0."""
    if j <= 1:
        return 0.0
    return prior_pp * math.sqrt(math.log(j) / math.log(prior_j))


def curse_from_history(history_successes, n_boot=4000, seed=7):
    """MEASURED winner's-curse correction (pp): bootstrap optimism of
    max-over-checkpoints selection, given every checkpoint's per-episode
    successes on the SAME CRN eval.

    Each rep resamples episode indices with replacement — the SAME indices
    across all checkpoints, so CRN pairing is preserved inside the rep —
    picks j* = argmax resampled SR (tie -> lowest index), and scores the
    optimism: resampled SR of j* minus j*'s full-sample SR. The mean over
    reps estimates how much 'picking the max' inflates the winner's measured
    SR. Clamped at 0 (a negative correction would reward selection)."""
    runs = [[1 if s else 0 for s in h] for h in history_successes]
    if not runs or not runs[0]:
        return 0.0
    n = len(runs[0])
    full = [100.0 * sum(r) / n for r in runs]
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        best_j, best_sr = 0, -1.0
        for j, r in enumerate(runs):
            s = 0
            for i in idxs:
                s += r[i]
            sr = 100.0 * s / n
            if sr > best_sr:              # strict >: ties keep lowest index
                best_sr, best_j = sr, j
        total += best_sr - full[best_j]
    return max(0.0, total / n_boot)


# ---------------------------------------------------------------- sizing

def episodes_for_min_effect(p_d, min_effect_pp, alpha=0.05, power_target=0.8):
    """Total PAIRED episodes for the McNemar test to reach power_target at a
    true effect of min_effect_pp, given discordance rate p_d.

    Derivation: delta_hat = (c-b)/n with var(delta_hat) ~ p_d/n (the
    concordant pairs cancel; only discordant episodes carry information), so
    n = p_d * ((z_{1-alpha} + z_{power}) / delta)^2. The 1e-9 ceil guard is
    the draws_needed FP convention (see power.draws_needed). Caller floors
    p_d at 1/n when b+c == 0 — a measured rate of exactly 0 would claim
    infinite power."""
    z = stats.norm_q(1 - alpha) + stats.norm_q(power_target)
    return math.ceil(p_d * (z / (min_effect_pp / 100.0)) ** 2 - 1e-9)


def power_at_min_effect_paired(n, p_d, min_effect_pp, alpha=0.05):
    """Power of the one-sided paired test at a true effect of min_effect_pp,
    n paired episodes, discordance rate p_d (var(delta_hat) ~ p_d/n)."""
    return stats.Phi(math.sqrt(n) * (min_effect_pp / 100.0) / math.sqrt(p_d)
                     - stats.norm_q(1 - alpha))


def episodes_for_min_effect_unpaired(p_bar, min_effect_pp, alpha=0.05,
                                     power_target=0.8):
    """Episodes PER RUN for the unpaired two-proportion test (pooled rate
    p_bar) to reach power_target at min_effect_pp — the degrade-path price:
    var(delta_hat) ~ 2*p_bar*(1-p_bar)/n with no CRN cancellation."""
    z = stats.norm_q(1 - alpha) + stats.norm_q(power_target)
    return math.ceil(2 * p_bar * (1 - p_bar)
                     * (z / (min_effect_pp / 100.0)) ** 2 - 1e-9)


def power_at_min_effect_unpaired(n, p_bar, min_effect_pp, alpha=0.05):
    """Power of the one-sided unpaired test at min_effect_pp, n per run."""
    return stats.Phi((min_effect_pp / 100.0)
                     / math.sqrt(2 * p_bar * (1 - p_bar) / n)
                     - stats.norm_q(1 - alpha))


# ---------------------------------------------------------------- retrain prior

def resolve_sigma_run(config):
    """Resolve the retrain-noise prior for comparison_level='retrain'.

    Frozen resolution order (v0.1 spec): explicit config.sigma_run_pp beats
    config.regime (atlas get_regime(regime)['sigma_run']) beats None. Returns
    (sigma_run_pp, source) with source 'explicit' | 'atlas:<regime>' | None;
    (None, None) means the gate runs with v0 statistics plus the no-prior
    disclosure. An unknown regime raises GateInputError — a typo'd regime at
    retrain level must never silently fall back to unpriced retrain noise
    (the GateConfig.from_dict ethos)."""
    if config.comparison_level != "retrain":
        return None, None
    if config.sigma_run_pp is not None:
        return float(config.sigma_run_pp), "explicit"
    if config.regime is not None:
        try:
            reg = atlas.get_regime(config.regime)
        except KeyError as e:
            raise GateInputError(
                "comparison_level='retrain' with unresolvable regime: %s"
                % e)
        return float(reg["sigma_run"]), "atlas:%s" % reg["key"]
    return None, None


def resolve_sigma_pertask(config):
    """Resolve the PER-TASK retrain-noise prior (HOLDINJ_PREREG 2026-08-15).

    Mirrors resolve_sigma_run: explicit config.sigma_pertask_pp beats
    config.regime (atlas regime['sigma_pertask'], which may be absent or
    None for rows whose per-task sigma is unmeasured or df-starved and
    therefore NOT quotable) beats None. With None the per-task flags stay
    EVAL-level v0 behavior, byte-identical, with the existing unpriced
    disclosure."""
    if config.comparison_level != "retrain":
        return None, None
    if config.sigma_pertask_pp is not None:
        return float(config.sigma_pertask_pp), "explicit"
    if config.regime is not None:
        try:
            reg = atlas.get_regime(config.regime)
        except KeyError as e:
            raise GateInputError(
                "comparison_level='retrain' with unresolvable regime: %s"
                % e)
        sp = reg.get("sigma_pertask")
        if sp is not None:
            return float(sp), "atlas:%s" % reg["key"]
    return None, None


# ---------------------------------------------------------------- gate

def gate(incumbent, candidate, config, history=None, now=None,
         cross_stack=False, extra_assumptions=None):
    """Run the fixed-sample Ship-Gate. Returns a GateDecision.

    incumbent/candidate: io.EvalRun. config: records.GateConfig.
    history: optional list of io.EvalRun — the candidate's checkpoint evals
    (each must match the candidate's n_eval and eval seed, else INVALID);
    with selection='max-over-checkpoints' it upgrades the curse correction
    from the atlas prior to a direct bootstrap measurement. The pool must be
    the ACTUAL selection pool, i.e. include the candidate's own checkpoint
    eval — a pool without the winner underestimates the optimism, and a
    recorded warning fires when no history vector matches the candidate's.
    now: ISO-8601 UTC 'YYYY-MM-DDTHH:MM:SSZ' string, or None to stamp
    datetime.now(timezone.utc) here (records.py never reads the clock; pass
    now explicitly for byte-reproducible records).
    cross_stack: caller declares the two runs were evaluated on DIFFERENT
    environment/simulator builds. The atlas MEASURED a gym_pusht rebuild
    shifting SR -18.2 pp mean, non-uniformly (OPS_RUNBOOK sec 9): such a
    comparison cannot separate policy quality from harness drift, so the
    verdict is INVALID with atlas.CAVEATS['cross-stack'] on the record.
    extra_assumptions: caller-declared disclosure strings recorded VERBATIM
    in the record's assumptions list (e.g. replay's cross-wave same-epoch
    pairing assumption); they never alter the statistics.

    Config misuse raises GateInputError; bad run DATA never raises — it
    yields a verdict-'INVALID' record listing every cause."""
    errs = config.validate()
    if errs:
        raise GateInputError("invalid GateConfig: " + "; ".join(errs))
    hist = list(history) if history else None
    if hist and config.selection == "final":
        raise GateInputError(
            "history passed with selection='final' — if the candidate was "
            "picked as a max over its checkpoints, declare "
            "selection='max-over-checkpoints'; if not, drop the history")
    if (config.selection == "max-over-checkpoints" and not hist
            and config.n_checkpoints is None):
        raise GateInputError(
            "selection='max-over-checkpoints' needs either n_checkpoints (J) "
            "for the atlas-prior correction or the checkpoint eval history "
            "for a measured one")
    created = now if now is not None else \
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    warnings, caveats, assumptions, invalid = [], [], [], []
    if extra_assumptions:
        assumptions.extend(str(a) for a in extra_assumptions)

    # ---- comparison-level semantics (v0.1; UNCONDITIONAL disclosure) ----
    sigma_run_pp, sigma_run_source = resolve_sigma_run(config)
    retrain_aware = (config.comparison_level == "retrain"
                     and sigma_run_pp is not None)
    sigma_pt_pp, sigma_pt_source = resolve_sigma_pertask(config)
    pertask_priced = (config.comparison_level == "retrain"
                      and sigma_pt_pp is not None)
    if config.comparison_level == "checkpoint":
        assumptions.append(
            "ASSUMPTION: checkpoint-level comparison: alpha bounds "
            "eval-sampling error only; NOT valid for independent retrains "
            "(measured 16% false-ship on ORBIT retrain null pairs)")
    elif not retrain_aware:
        assumptions.append(
            "ASSUMPTION: retrain comparison WITHOUT a sigma_run prior — "
            "alpha bounds eval-sampling error only; false-ship rate vs the "
            "retraining lottery is unbounded (measured 16% on the ORBIT "
            "null corpus)")

    # ---- pre-check ladder (order frozen; collects EVERY cause) ----
    # 0. declared cross-build comparison: forbidden by measurement, not taste
    if cross_stack:
        caveats.append(atlas.CAVEATS["cross-stack"])
        invalid.append(
            "caller declared the runs come from different environment/"
            "simulator builds (cross_stack): any delta mixes policy quality "
            "with harness drift — the gym_pusht rebuild shifted SR %.1f pp "
            "mean, NON-uniformly (range -10.6..-24.4); re-anchor with known "
            "sets before comparing [OPS_RUNBOOK sec 9]"
            % atlas.STACK_SHIFT_PUSHT_PP)
    # 1. budget gate — truncation is directional, not mean-zero
    for tag, r in (("incumbent", incumbent), ("candidate", candidate)):
        ok, why = stats.budget_gate(r.final_step, r.design_steps)
        if not ok:
            invalid.append(
                "%s (%s): %s — truncation is DIRECTIONAL (an undertrained "
                "model scores LOW, impersonating whichever hypothesis "
                "predicts a lower score), not mean-zero [OPS_RUNBOOK sec 3/4]"
                % (tag, r.run_id, why))
        elif why:
            warnings.append("%s (%s): %s" % (tag, r.run_id, why))
    # 2. sr present
    for tag, r in (("incumbent", incumbent), ("candidate", candidate)):
        if r.sr is None:
            invalid.append("%s (%s): no sr recorded" % (tag, r.run_id))
    # 3. per-episode successes present
    for tag, r in (("incumbent", incumbent), ("candidate", candidate)):
        if not r.successes:
            invalid.append(
                "%s (%s): per-episode successes missing or empty — the gate "
                "is a paired per-episode instrument; a bare SR cannot be "
                "CRN-paired" % (tag, r.run_id))
    inc_s = [bool(s) for s in incumbent.successes] if incumbent.successes else None
    cand_s = [bool(s) for s in candidate.successes] if candidate.successes else None
    # 4. equal episode counts
    n = None
    if inc_s and cand_s:
        if len(inc_s) != len(cand_s):
            invalid.append(
                "episode counts differ (%d vs %d): CRN pairing impossible — "
                "the same episode index must mean the same initial state"
                % (len(inc_s), len(cand_s)))
        else:
            n = len(inc_s)
    # 5. task sets must match
    if (incumbent.task_groups and candidate.task_groups
            and list(incumbent.task_groups) != list(candidate.task_groups)):
        invalid.append(
            "task_groups differ (%s vs %s): not the same eval set — any "
            "delta would mix task identity with policy quality"
            % (incumbent.task_groups, candidate.task_groups))
    # 6./7. eval seed ladder
    si, sc = incumbent.seed, candidate.seed
    if si is not None and sc is not None and si != sc:
        invalid.append(
            "eval seeds differ (%s vs %s): CRN pairing impossible — re-eval "
            "one side on the other's seed" % (si, sc))
    mixed = (si is None) != (sc is None)
    recorded_seed = si if si is not None else sc
    if mixed and recorded_seed != 1000:
        invalid.append(
            "one eval seed recorded (%s), the other unrecorded (LeRobot's "
            "silent default is 1000): seeds almost certainly differ — CRN "
            "pairing impossible" % recorded_seed)
    # 8. history must be on the candidate's CRN eval
    if hist:
        for h in hist:
            if not h.successes:
                invalid.append("history run %s: per-episode successes "
                               "missing or empty" % h.run_id)
                continue
            if n is not None and len(h.successes) != n:
                invalid.append(
                    "history run %s: %d episodes != candidate's %d — the "
                    "curse bootstrap needs every checkpoint on the SAME "
                    "CRN eval" % (h.run_id, len(h.successes), n))
            if (h.seed is not None and sc is not None and h.seed != sc) or \
                    (((h.seed is None) != (sc is None))
                     and (h.seed if h.seed is not None else sc) != 1000):
                invalid.append(
                    "history run %s: eval seed %s does not match candidate "
                    "seed %s — not the same CRN eval" % (h.run_id, h.seed, sc))
        # 8b. the pool must contain the winner: the bootstrap measures the
        # optimism of the ACTUAL selection event, and the candidate is part
        # of its own selection pool. A pool of only the other J-1
        # checkpoints bootstraps the wrong event (typically underestimating
        # the correction) — warn on the record, never silently.
        if cand_s is not None and not any(
                h.successes is not None
                and [bool(x) for x in h.successes] == cand_s
                for h in hist):
            warnings.append(
                "history pool does not include the candidate's own "
                "checkpoint eval (no history successes vector matches the "
                "candidate's): the history-bootstrap curse correction is "
                "only correct when the pool is the actual selection pool "
                "INCLUDING the winner — a pool of the other J-1 checkpoints "
                "underestimates the optimism")
    # 9./10. task-block structure must tile the eval
    if n is not None and config.task_blocks is not None \
            and sum(config.task_blocks) != n:
        invalid.append("task_blocks sum to %d but the eval has %d episodes"
                       % (sum(config.task_blocks), n))
    if n is not None and config.task_blocks is None \
            and config.n_tasks is not None and n % config.n_tasks != 0:
        invalid.append(
            "n_tasks=%d does not divide n=%d: contiguous equal task blocks "
            "impossible — pass task_blocks explicitly"
            % (config.n_tasks, n))
    # 11. regime scope (2026-08-16): a --regime whose measured cell
    # verifiably contradicts THIS comparison's data must never price it —
    # mispricing with the wrong cell's sigma is the exact failure the
    # product exists to prevent. Fires ONLY on verifiable contradictions
    # (input metadata present and inconsistent); when the inputs carry no
    # metadata the regime scope remains an operator assertion, disclosed as
    # ever via sigma_run_source='atlas:<key>'. An unknown regime key keeps
    # its existing error path in resolve_sigma_run (not duplicated here).
    # 11c. FLOOR-DOMINATED PRIOR (2026-08-20). A regime whose sigma_run sits at
    # or below its own binomial floor carries no detectable retrain variance;
    # pricing from it UNDER-prices the lottery, which is the one direction this
    # gate exists to avoid. Warn loudly on the record rather than fail, because
    # the honest measurement may legitimately be floor-dominated (see
    # pusht-dp-rebuilt: 1.292 at df=4, floor 1.682 at its own SR).
    if config.comparison_level == "retrain" and config.regime is not None \
            and config.sigma_run_pp is None:
        try:
            _fr = atlas.get_regime(config.regime)
        except KeyError:
            _fr = None
        if _fr and _fr.get("sigma_run") and _fr.get("sr_mean") is not None \
                and _fr.get("n_eval"):
            _p = _fr["sr_mean"] / 100.0
            _fl = 100.0 * math.sqrt(max(0.0, _p * (1 - _p)) / _fr["n_eval"])
            if _fr["sigma_run"] <= _fl:
                warnings.append(
                    "ANTI-CONSERVATIVE PRIOR: regime '%s' has sigma_run %.3f pp "
                    "at or BELOW its own binomial floor %.3f pp (SR %.1f%%, "
                    "n=%d) — the cell has no retrain variance detectable above "
                    "sampling, so pricing from it UNDERSTATES the lottery. Use "
                    "a floor-respecting prior (e.g. the corpus max(atlas, "
                    "EB-shrunk)) or measure this cell properly."
                    % (config.regime, _fr["sigma_run"], _fl,
                       _fr["sr_mean"], _fr["n_eval"]))

    if config.comparison_level == "retrain" and config.regime is not None:
        try:
            _reg = atlas.get_regime(config.regime)
        except KeyError:
            _reg = None
        # the check guards PRICING: it applies only when the regime would
        # actually resolve a sigma (explicit values beat the regime, leaving
        # it as recorded context — a contradiction there prices nothing)
        _prices = _reg is not None and (
            (config.sigma_run_pp is None
             and _reg.get("sigma_run") is not None)
            or (config.sigma_pertask_pp is None
                and _reg.get("sigma_pertask") is not None))
        if _prices:
            _env = (_reg.get("env") or "").lower()
            _known = ("libero", "pusht", "aloha")
            for tag, r in (("incumbent", incumbent), ("candidate", candidate)):
                tg = [g for g in (r.task_groups or []) if g]
                # env fires only when the group names carry a RECOGNIZED env
                # vocabulary that excludes the regime's env — an absent or
                # unrecognized name proves nothing and stays silent
                seen = {e for e in _known
                        if any(e in g.lower() for g in tg)}
                if _env and seen and _env not in seen:
                    invalid.append(
                        "%s (%s): regime '%s' is measured on env '%s' but "
                        "the eval's task groups name %s — wrong-cell sigma "
                        "would misprice the lottery" % (tag, r.run_id,
                        config.regime, _env, sorted(seen)))
                if tg and config.n_tasks is not None \
                        and len(tg) != config.n_tasks:
                    invalid.append(
                        "%s (%s): eval declares %d task group(s) but config "
                        "asserts n_tasks=%d — per-task pricing on a "
                        "mismatched battery is wrong-cell pricing"
                        % (tag, r.run_id, len(tg), config.n_tasks))
                k_reg = _reg.get("k")
                if k_reg is not None and r.episodes:
                    if len(r.episodes) != k_reg:
                        invalid.append(
                            "%s (%s): trained on k=%d episodes but regime "
                            "'%s' is the k=%d cell — sigma does not "
                            "transfer across k (KCURVE)" % (tag, r.run_id,
                            len(r.episodes), config.regime, k_reg))
                # 11b. SR-regime scope (2026-08-18, PERTASK-3 audit): sigma in
                # pp carries a binomial component that scales as
                # sqrt(p(1-p)), so a sigma measured on a near-floor cell
                # under-prices a healthy one (and vice versa) even when env,
                # task count and k all match. Fires only for rows that declare
                # the SR band their sigma was measured in — absent sr_range
                # keeps the previous operator-assertion behaviour.
                sr_band = _reg.get("sr_range")
                # Compute the SR from the successes vector, not the self-reported
                # r.sr field: parse_eval_info takes overall.pc_success verbatim,
                # so a JSON whose header disagrees with its own episodes would
                # otherwise steer the guard (audit 2026-08-20). Fall back to r.sr
                # only when no vector exists.
                _succ = getattr(r, "successes", None)
                _sr = (100.0 * sum(bool(x) for x in _succ) / len(_succ)) \
                    if _succ else r.sr
                _n = len(_succ) if _succ else r.n_eval
                if sr_band and _sr is not None:
                    mism = atlas.sr_floor_mismatch(sr_band, _sr, _n)
                    if mism is not None \
                            and mism > atlas.SR_SCOPE_FLOOR_RATIO:
                        invalid.append(
                            "%s (%s): SR %.2f%% sits outside the SR band "
                            "regime '%s' was measured in (%.1f-%.1f%%) — the "
                            "binomial floor mismatch is %s vs the %.1fx "
                            "tolerance, so this cell's sigma does not price "
                            "this eval (out-of-regime pricing)"
                            % (tag, r.run_id, _sr, config.regime,
                               min(sr_band), max(sr_band),
                               ("opposite side of the 50% axis"
                                if mism == float("inf") else "%.2fx" % mism),
                               atlas.SR_SCOPE_FLOOR_RATIO))

    inputs = {"incumbent": _run_inputs(incumbent),
              "candidate": _run_inputs(candidate),
              "n_paired": n}

    if invalid:
        record = build_record("INVALID", config, inputs, _null_statistics(
            incumbent.sr, candidate.sr), caveats, warnings, assumptions,
            invalid, created)
        return GateDecision(verdict="INVALID", record=record, reasons=invalid)

    # ---- pairing decision (honest, disclosed) ----
    seeds_equal = si is not None and sc is not None and si == sc
    seeds_none = si is None and sc is None
    mixed_default = mixed and recorded_seed == 1000
    paired = False
    if seeds_equal:
        paired = True
    elif seeds_none or mixed_default:
        if config.assume_crn:
            paired = True
            warnings.append(
                "eval seeds unrecorded on both runs" if seeds_none else
                "one eval seed unrecorded, the other recorded as 1000 "
                "(LeRobot's silent default)")
            warnings[-1] += ("; CRN pairing ASSUMED via assume_crn — "
                            "LeRobot's silent default is seed 1000 for "
                            "every eval")
        else:
            warnings.append(
                ("eval seeds unrecorded on both runs" if seeds_none else
                 "one eval seed unrecorded, the other recorded as 1000 "
                 "(LeRobot's silent default)")
                + ": cannot VERIFY common random numbers — degrading to "
                  "UNPAIRED comparison (set assume_crn if both runs used "
                  "the same eval seed)")

    if (incumbent.is_libero or candidate.is_libero) \
            and config.sequential_eval is not True:
        caveats.append(atlas.CAVEATS["libero-autoreset"])
        if paired:
            paired = False
            warnings.append(
                "LIBERO eval not declared sequential (sequential_eval is "
                "False): vectorized autoreset makes later initial states "
                "outcome-dependent under early termination — CRN pairing "
                "degraded to UNPAIRED (LeRobot issue #4152)")
    if paired:
        caveats.append(atlas.CAVEATS["pairing-validated"])
        caveats.append(atlas.CAVEATS["async-envs"])

    # ---- primary statistics ----
    k_inc, k_cand = sum(inc_s), sum(cand_s)
    sr_inc, sr_cand = 100.0 * k_inc / n, 100.0 * k_cand / n
    if paired:
        _n11, b, c, _n00 = stats.discordant_counts(inc_s, cand_s)
        n_discordant = b + c
        discordance_rate = n_discordant / n
        delta_hat = 100.0 * (c - b) / n
        p_ship = stats.mcnemar_one_sided(c, b)
        p_worse = stats.mcnemar_one_sided(b, c)
        ci95 = list(stats.paired_boot_ci(cand_s, inc_s, n_boot=config.n_boot,
                                         seed=config.boot_seed))
    else:
        b = c = n_discordant = discordance_rate = None
        delta_hat, z, _p2 = stats.two_prop_test(k_cand, n, k_inc, n)
        p_ship = min(1.0, max(0.0, 1 - stats.Phi(z)))
        p_worse = min(1.0, max(0.0, stats.Phi(z)))
        ci95 = list(stats.newcombe_ci(k_cand, n, k_inc, n))

    # ---- retrain-aware overlay (v0.1) ----
    # p_eval keeps the eval-level one-sided p (exact McNemar on the paired
    # path, two-proportion z unpaired) in EVERY record; at retrain level
    # with a resolved prior the GOVERNING p_ship/p_worse switch to the
    # normal approximation on se_total = sqrt(se_eval^2 + 2*sigma_run^2) —
    # one independent retrain per side (method 'z-retrain-aware', DISCLOSED).
    p_eval = p_ship
    method = "mcnemar-exact" if paired else "two-prop-z"
    se_eval_pp = se_total_pp = None
    if retrain_aware:
        method = "z-retrain-aware"
        if paired:
            # H0 variance of delta_hat: p_d/n (the concordant pairs cancel);
            # conservative under H1, where the true variance is
            # (p_d - delta^2)/n <= p_d/n.
            se_eval_pp = 100.0 * math.sqrt((n_discordant / n) / n)
        else:
            p_inc_hat, p_cand_hat = k_inc / n, k_cand / n
            se_eval_pp = 100.0 * math.sqrt(
                p_inc_hat * (1 - p_inc_hat) / n
                + p_cand_hat * (1 - p_cand_hat) / n)
        se_total_pp = math.sqrt(se_eval_pp ** 2 + 2 * sigma_run_pp ** 2)
        z_total = delta_hat / se_total_pp if se_total_pp > 0 else 0.0
        p_ship = min(1.0, max(0.0, 1 - stats.Phi(z_total)))
        p_worse = min(1.0, max(0.0, stats.Phi(z_total)))
        assumptions.append(
            "ASSUMPTION: retrain-level comparison: governing p-values use "
            "the normal approximation (method 'z-retrain-aware') on "
            "se_total = sqrt(se_eval^2 + 2*sigma_run^2) — one independent "
            "retrain per side, sigma_run %.2f pp (%s); the eval-level "
            "%s p is recorded as p_eval; se_eval %s"
            % (sigma_run_pp, sigma_run_source,
               "exact McNemar" if paired else "two-proportion",
               "uses the H0 discordance variance p_d/n (conservative "
               "under H1)" if paired
               else "pays the full independent-draw price "
                    "pa*(1-pa)/na + pb*(1-pb)/nb"))

    # ---- winner's-curse correction ----
    curse_prior_ref = None
    if config.selection == "final":
        correction, curse_method = 0.0, "none"
    elif hist:
        correction = curse_from_history([h.successes for h in hist],
                                        config.n_boot, config.boot_seed)
        curse_method = "history-bootstrap"
        j_hist = len(hist)
        curse_prior_ref = curse_from_prior(j_hist, config.curse_prior_pp,
                                           config.curse_prior_j)
        # Estimand disclosure: the bootstrap and the prior measure DIFFERENT
        # quantities. On the corpus the bootstrap runs 0.0-3.8 pp while the
        # prior scales 3.6-6.8 pp — a record quoted alone must say why.
        assumptions.append(
            "ASSUMPTION: history-bootstrap curse correction measures the "
            "EVAL-DRAW selection optimism of max-over-J=%d checkpoints on "
            "this CRN eval only; it does NOT include retrain-lottery "
            "optimism, which the prospectively measured J=%d prior "
            "(+%.1f pp mean, PI05_CELL1_RESULTS) does include — the "
            "prior-scaled reference for J=%d is %.2f pp "
            "(curse_prior_reference_pp)"
            % (j_hist, config.curse_prior_j, config.curse_prior_pp,
               j_hist, curse_prior_ref))
    else:
        correction = curse_from_prior(config.n_checkpoints,
                                      config.curse_prior_pp,
                                      config.curse_prior_j)
        curse_method = "atlas-prior-scaled"
        assumptions.append(
            "ASSUMPTION: winner's-curse correction scaled from the measured "
            "J=%d prior as %.1f*sqrt(ln(J)/ln(%d)) pp (PI05_CELL1_RESULTS: "
            "+4.00 pp mean inflation, max +12.5, measured prospectively; the "
            "sqrt-log scaling beyond J=%d is assumed, not measured)"
            % (config.curse_prior_j, config.curse_prior_pp,
               config.curse_prior_j, config.curse_prior_j))
    delta_effective = delta_hat - correction

    # ---- per-task regression ----
    blocks = None
    if config.task_blocks is not None:
        blocks = list(config.task_blocks)
    elif config.n_tasks is not None:
        npt = n // config.n_tasks
        blocks = [npt] * config.n_tasks
        assumptions.append(
            "ASSUMPTION: task blocks contiguous equal, n_per_task = %d/%d = "
            "%d (LeRobot orders eval episodes by task_id)"
            % (n, config.n_tasks, npt))
    else:
        assumptions.append(
            "ASSUMPTION: no task structure declared (n_tasks/task_blocks "
            "unset) — treating all %d episodes as a single block; per-task "
            "regression NOT checked" % n)
    per_task, any_regression = [], False
    if blocks:
        start = 0
        for i, bn in enumerate(blocks):
            inc_b, cand_b = inc_s[start:start + bn], cand_s[start:start + bn]
            start += bn
            label = (config.task_labels[i]
                     if config.task_labels and i < len(config.task_labels)
                     else "task%d" % i)
            if paired:
                _, tb, tc, _ = stats.discordant_counts(inc_b, cand_b)
                dpp = 100.0 * (tc - tb) / bn
                pw = stats.mcnemar_one_sided(tb, tc)
            else:
                # exact test on the degrade path too: blocks are ~20
                # episodes, where the pooled-z Gaussian's realised level
                # drifts above nominal — Fisher's exact tail is valid at
                # any block size (see stats.fisher_exact_one_sided).
                tb = tc = None
                kc, ki = sum(cand_b), sum(inc_b)
                dpp = 100.0 * (kc - ki) / bn
                pw = stats.fisher_exact_one_sided(kc, bn, ki, bn)
            flag_eval = (dpp <= -config.regression_min_pp
                         and pw <= config.regression_alpha)
            row = {"task": label, "n": bn, "b": tb, "c": tc,
                   "delta_pp": dpp, "p_worse": pw}
            if pertask_priced:
                # Retrain-priced per-task test (HOLDINJ_PREREG 2026-08-15,
                # sha256 938a3d46...): the operative null for an independent
                # retrain is the per-task lottery (measured sigma 11.00 pp at
                # production k, PERTASK2_RESULTS), not eval sampling. Same
                # overlay convention as the suite-level z-retrain-aware path.
                if paired:
                    se_t_eval = 100.0 * math.sqrt(((tb + tc) / bn) / bn) \
                        if (tb + tc) > 0 else 100.0 * math.sqrt((1.0 / bn) / bn)
                else:
                    pi_b, pc_b = sum(inc_b) / bn, sum(cand_b) / bn
                    se_t_eval = 100.0 * math.sqrt(
                        pi_b * (1 - pi_b) / bn + pc_b * (1 - pc_b) / bn)
                se_t_total = math.sqrt(se_t_eval ** 2 + 2 * sigma_pt_pp ** 2)
                p_retrain = min(1.0, max(0.0, stats.Phi(dpp / se_t_total)))
                flag = (dpp <= -config.regression_min_pp
                        and p_retrain <= config.regression_alpha)
                row["p_retrain"] = p_retrain
                row["regression"] = flag
                # Two-tier classification: eval-level flags that the lottery
                # explains are REAL differences but not gate-actionable.
                row["tier"] = ("regression-beyond-lottery" if flag
                               else ("lottery-expected" if flag_eval
                                     else None))
            else:
                flag = flag_eval
                row["regression"] = flag
            any_regression = any_regression or flag
            per_task.append(row)
        if len(blocks) > 1:
            # Multiplicity is a THRESHOLD choice, disclosed, not silently
            # multiplied: regression_alpha is per task (frozen v0 rule,
            # GateConfig docstring), so T tasks give up to 1-(1-alpha)^T
            # family-wise false-flag probability under per-task nulls.
            t = len(blocks)
            fw = 1.0 - (1.0 - config.regression_alpha) ** t
            assumptions.append(
                "ASSUMPTION: regression_alpha %g is applied PER TASK with "
                "NO family-wise correction (frozen v0 rule): across T=%d "
                "tasks the family-wise false-flag probability is up to "
                "1-(1-alpha)^T = %.3f under per-task nulls (the exact "
                "per-task tests are discrete-conservative, so the realised "
                "rate is lower)"
                % (config.regression_alpha, t, fw))
        if pertask_priced:
            # Retrain-draw budget: what a 5-20 pp per-task regression costs
            # to resolve when a single pair cannot (HOLDINJ_PREREG band
            # consequence). m(delta) = ceil(((z95+z80)*sqrt(2)*sigma)/d)^2.
            _zsum = 1.6448536269514722 + 0.8416212335729143
            def _m(d):
                return int(math.ceil(
                    (_zsum * math.sqrt(2.0) * sigma_pt_pp / d) ** 2))
            assumptions.append(
                "ASSUMPTION: per-task regression flags are RETRAIN-PRICED "
                "(HOLDRATE_DIAG 2026-08-15: the unpriced rule HOLDs 96.4%% "
                "of identical-data retrain pairs at production k): governing "
                "per-task p uses se = sqrt(se_eval^2 + 2*sigma_pertask^2), "
                "sigma_pertask %.2f pp (%s); eval-level flags the lottery "
                "explains are recorded tier='lottery-expected', not "
                "gate-actionable. Single-pair resolution at this sigma: a "
                "%.0f pp per-task drop; smaller regressions need retrain "
                "draws — ~%d retrains/side at 10 pp, ~%d at 20 pp (80%% "
                "power, one-sided alpha %g)"
                % (sigma_pt_pp, sigma_pt_source,
                   (1.6448536269514722 + 0.8416212335729143)
                   * math.sqrt(2.0) * sigma_pt_pp,
                   _m(10.0), _m(20.0), config.regression_alpha))
        elif retrain_aware:
            # Per-task sigma unresolved for this regime (only suite-level
            # priors exist), so retrain noise is NOT priced into per-task
            # flags even at retrain level — disclosed.
            assumptions.append(
                "ASSUMPTION: per-task regression flags remain EVAL-level "
                "exact tests even at comparison_level='retrain' — per-task "
                "sigma_run is unmeasured for this regime, so the retraining "
                "lottery is NOT priced into per-task flags (measured "
                "consequence: 96.4% of identical-data retrain pairs HOLD at "
                "production k, HOLDRATE_DIAG 2026-08-15)")

    # ---- COLLECT-MORE sizing (computed on EVERY non-INVALID record) ----
    if paired:
        p_d_used = max(n_discordant / n, 1.0 / n)
        p_bar_used = None
    else:
        p_d_used = None
        p_bar = (k_inc + k_cand) / (2.0 * n)
        p_bar_used = min(max(p_bar, 0.5 / n), 1.0 - 0.5 / n)
        if p_bar_used != p_bar:
            assumptions.append(
                "ASSUMPTION: pooled success rate %.6g is degenerate — "
                "unpaired sizing uses the clamped p_bar = %.6g (a 0%%/100%% "
                "pool has zero variance, which would claim infinite power)"
                % (p_bar, p_bar_used))
    mde_floor_pp = None
    collect_more_unit = "episodes"
    retrains_needed = None
    if retrain_aware:
        # One-sided sizing constants (stats.norm_q; NOT power.py's frozen
        # two-sided 2.80 — that constant belongs to the two-sided method
        # comparison, not this one-sided gate).
        z_a = stats.norm_q(1 - config.alpha)
        z_p = stats.norm_q(config.power_target)
        # MDE floor at 1 retrain per side and INFINITE eval episodes:
        # se_total -> sqrt(2)*sigma_run, so no episode budget resolves an
        # effect below (z_a+z_p)*sqrt(2)*sigma_run. Recorded always.
        mde_floor_pp = (z_a + z_p) * math.sqrt(2.0) * sigma_run_pp
        if config.min_effect_pp <= mde_floor_pp:
            # Episodes cannot resolve: price the comparison in RETRAINS per
            # side instead. The eval term is IGNORED here (infinite episodes
            # per retrain assumed) — disclosed; the true requirement is >=.
            collect_more_unit = "retrains"
            retrains_needed = math.ceil(
                2 * ((z_a + z_p) * sigma_run_pp / config.min_effect_pp) ** 2)
            n_required = None
            episodes_needed = None
            assumptions.append(
                "ASSUMPTION: retrain sizing ignores the eval-sampling term "
                "(retrains_needed = ceil(2*((z_a+z_p)*sigma_run/"
                "min_effect)^2) per side assumes infinite eval episodes per "
                "retrain) — the true requirement is >= retrains_needed")
        else:
            # Episodes such that se_total <= min_effect/(z_a+z_p) at 1
            # retrain/side; the min_effect > mde_floor guard above keeps the
            # denominator > 0. Units: pp^2 (p_d/n and p_bar variance are
            # fraction-scale, hence the 10000).
            denom = ((config.min_effect_pp / (z_a + z_p)) ** 2
                     - 2 * sigma_run_pp ** 2)
            if paired:
                n_required = math.ceil(p_d_used * 10000.0 / denom)
            else:
                n_required = math.ceil(
                    2 * p_bar_used * (1 - p_bar_used) * 10000.0 / denom)
            episodes_needed = max(0, n_required - n)
        # Power against the TOTAL noise at the current eval size.
        power = stats.Phi(config.min_effect_pp / se_total_pp - z_a) \
            if se_total_pp > 0 else 1.0
    elif paired:
        n_required = episodes_for_min_effect(
            p_d_used, config.min_effect_pp, config.alpha, config.power_target)
        power = power_at_min_effect_paired(
            n, p_d_used, config.min_effect_pp, config.alpha)
        episodes_needed = max(0, n_required - n)
    else:
        n_required = episodes_for_min_effect_unpaired(
            p_bar_used, config.min_effect_pp, config.alpha,
            config.power_target)
        power = power_at_min_effect_unpaired(
            n, p_bar_used, config.min_effect_pp, config.alpha)
        episodes_needed = max(0, n_required - n)
    if paired and n_discordant == 0 and collect_more_unit == "episodes":
        # Disclosed only when the EPISODES sizing path actually consumed the
        # floored p_d (checkpoint level, or retrain with min_effect above the
        # MDE floor) — the retrains path prices retrains, not episodes, and
        # never reads p_d.
        assumptions.append(
            "ASSUMPTION: zero discordant episodes observed (b+c=0) — "
            "sizing uses the floor discordance rate p_d = 1/n = %.6g; a "
            "measured rate of exactly 0 would claim infinite power"
            % (1.0 / n))

    # ---- decision (order frozen; curse affects SHIP only, never HOLD) ----
    reasons = []
    if any_regression:
        flagged = [row["task"] for row in per_task if row["regression"]]
        verdict = "HOLD"
        reasons.append(
            "per-task regression flag on %s: drop <= -%.1f pp at one-sided "
            "p <= %g — an overall win must not ship a task it broke"
            % (", ".join(flagged), config.regression_min_pp,
               config.regression_alpha))
    elif p_worse <= config.alpha:
        verdict = "HOLD"
        reasons.append("candidate significantly WORSE: p_worse %.3g <= "
                       "alpha %g" % (p_worse, config.alpha))
    elif p_ship <= config.alpha and delta_effective >= config.min_effect_pp:
        verdict = "SHIP"
        reasons.append("p_ship %.3g <= alpha %g" % (p_ship, config.alpha))
        reasons.append("delta_effective %.1f pp >= min_effect %.1f pp"
                       % (delta_effective, config.min_effect_pp))
        reasons.append("no per-task regression flag")
    else:
        verdict = "COLLECT-MORE"
        if p_ship > config.alpha:
            reasons.append("p_ship %.3g > alpha %g: not significant"
                           % (p_ship, config.alpha))
        if delta_effective < config.min_effect_pp:
            reasons.append(
                "delta_effective %.1f pp < min_effect %.1f pp%s"
                % (delta_effective, config.min_effect_pp,
                   " (delta_hat %.1f pp minus winner's-curse correction "
                   "%.1f pp)" % (delta_hat, correction)
                   if correction else ""))
        if collect_more_unit == "retrains":
            # Retrain-aware sizing said episodes cannot resolve: min_effect
            # sits at/below the 1-retrain/side MDE floor, so the price is
            # independent RETRAINS per side, not eval episodes. max_budget
            # (an EPISODE budget) does not apply on this path.
            reasons.append(
                "collect %d independent retrains per side: min_effect "
                "%.1f pp <= mde_floor %.2f pp (the 1-retrain/side floor "
                "(z_a+z_p)*sqrt(2)*sigma_run at sigma_run %.2f pp) — no "
                "eval episode budget resolves this comparison; the retrain "
                "sizing ignores the eval term (disclosed) "
                "[reason-code: retrains-needed]"
                % (retrains_needed, config.min_effect_pp, mde_floor_pp,
                   sigma_run_pp))
        elif config.max_budget is not None and n_required > config.max_budget:
            verdict = "UNRESOLVABLE"
            reasons.append(
                "n_required %d exceeds max_budget %d: this comparison cannot "
                "be resolved at any affordable eval size"
                % (n_required, config.max_budget))
        elif episodes_needed == 0:
            # Well-powered already: n >= n_required, yet the effect (after
            # any curse correction) sits below the shippable floor. "Collect
            # 0 more episodes" would be incoherent — more data cannot lift
            # delta_effective past min_effect here. Distinct reason-code so
            # machine consumers can tell this from 'underpowered'.
            reasons.append(
                "already powered at min_effect (n=%d >= n_required %d for "
                "%.0f%% power at %.1f pp): the measured effect is genuinely "
                "below the shippable floor — collecting more episodes will "
                "not change this verdict [reason-code: effect-below-floor]"
                % (n, n_required, 100 * config.power_target,
                   config.min_effect_pp))
        else:
            reasons.append(
                "collect %d more episodes (n_required %d total for %.0f%% "
                "power at %.1f pp, observed %s rate %.4g)"
                % (episodes_needed, n_required, 100 * config.power_target,
                   config.min_effect_pp,
                   "discordance" if paired else "pooled-success",
                   p_d_used if paired else p_bar_used))
    if abs(delta_effective) < 2.0 and verdict in ("COLLECT-MORE",
                                                  "UNRESOLVABLE"):
        caveats.append(CAVEAT_FLOOR_2PP)

    statistics = {
        "paired": paired,
        "method": method,
        "b": b, "c": c,
        "n_discordant": n_discordant,
        "discordance_rate": discordance_rate,
        "sr_incumbent_pp": sr_inc,
        "sr_candidate_pp": sr_cand,
        "delta_hat_pp": delta_hat,
        "ci95_pp": ci95,
        "p_eval": p_eval,
        "p_ship": p_ship,
        "p_worse": p_worse,
        "sigma_run_pp_used": sigma_run_pp if retrain_aware else None,
        "sigma_run_source": sigma_run_source if retrain_aware else None,
        "se_eval_pp": se_eval_pp,
        "se_total_pp": se_total_pp,
        "mde_floor_pp": mde_floor_pp,
        "curse_correction_pp": correction,
        "curse_method": curse_method,
        "curse_prior_reference_pp": curse_prior_ref,
        "delta_effective_pp": delta_effective,
        "per_task": per_task,
        "power_at_min_effect": power,
        "n_required": n_required,
        "episodes_needed": episodes_needed,
        "collect_more_unit": collect_more_unit,
        "retrains_needed": retrains_needed,
    }
    record = build_record(verdict, config, inputs, statistics, caveats,
                          warnings, assumptions, reasons, created)
    return GateDecision(verdict=verdict, record=record, reasons=reasons)


# ---------------------------------------------------------------- helpers

def _run_inputs(r):
    """The per-run inputs sub-dict of the decision record."""
    return {"run_id": r.run_id, "path": r.path, "fmt": r.fmt,
            "sr_pp": r.sr, "n_eval": r.n_eval, "eval_seed": r.seed,
            "set_hash": r.set_hash,
            "policy_class": getattr(r, "policy_class", None),
            "task_groups": list(r.task_groups or []),
            "final_step": r.final_step, "design_steps": r.design_steps}


def _null_statistics(sr_inc, sr_cand):
    """Schema-stable statistics dict for INVALID records: honest nulls."""
    return {"paired": None, "method": None, "b": None, "c": None,
            "n_discordant": None,
            "discordance_rate": None, "sr_incumbent_pp": sr_inc,
            "sr_candidate_pp": sr_cand, "delta_hat_pp": None,
            "ci95_pp": None, "p_eval": None, "p_ship": None, "p_worse": None,
            "sigma_run_pp_used": None, "sigma_run_source": None,
            "se_eval_pp": None, "se_total_pp": None, "mde_floor_pp": None,
            "curse_correction_pp": None, "curse_method": None,
            "curse_prior_reference_pp": None,
            "delta_effective_pp": None, "per_task": [],
            "power_at_min_effect": None, "n_required": None,
            "episodes_needed": None, "collect_more_unit": None,
            "retrains_needed": None}
