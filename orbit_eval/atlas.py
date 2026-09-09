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

"""The measured ORBIT noise atlas, shipped as data.

Every number here was MEASURED on the ORBIT research program (2026-06..08), not
assumed. Provenance is given per entry. Units are percentage points (pp) of
success rate throughout.

Two quantities, two regimes — this distinction is the product's soul:

  sigma_run  run-to-run SD of a *retrained* checkpoint on a FIXED episode set
             (fixed data, fixed budget, new training seed). Governs question (b):
             "is selection method A better than selection method B?" — every arm
             must be independently retrained, so sigma_run (and sigma_set, when
             each retrain also redraws its episode set) sets the price.

  sigma_0    the harness floor: SD between two evals of the IDENTICAL checkpoint
             with the IDENTICAL eval seed. Governs question (a): "did my
             checkpoint / dataset change move the metric?" — common random
             numbers (CRN) pairing cancels everything except this floor, which
             is why question (a) is powered at ~2 seeds while question (b)
             needs tens of retrains per arm.

CRN pairing CANNOT rescue question (b): pairing eval episodes across arms
cancels eval-draw noise only; sigma_run and sigma_set live in training, not in
the eval draw.
"""

import math

# z-constants frozen in power_analysis_phase2.py (alpha=.05 two-sided, power=.80)
Z975 = 1.959964
Z80 = 0.841621

# OPS_RUNBOOK sec 9 — measured, not assumed: a gym_pusht rebuild shifted PushT
# absolute SR by -18.2 pp (33.0 -> 14.8 on re-run sets, range -10.6..-24.4, so
# NOT a uniform offset). Never pool SRs across an environment rebuild.
STACK_SHIFT_PUSHT_PP = -18.2

# SR-REGIME SCOPE (2026-08-18, PERTASK-3 audit). A sigma measured on a
# near-floor cell does not price a healthy one and vice versa: the binomial
# component of sigma in pp scales as sqrt(p(1-p)), so the SAME behavioural
# spread reads as a very different pp-sigma at SR 1% and at SR 60%. A regime
# row MAY declare `sr_range` = (lo_pp, hi_pp), the span of success rates the
# runs behind its sigma actually landed in. When it does, the gate refuses to
# price an eval whose binomial floor differs from EVERY point of that band by
# more than SR_SCOPE_FLOOR_RATIO (pre-check 11b). Rows without `sr_range` are
# unguarded — the scope stays an operator assertion, exactly as before.
SR_SCOPE_FLOOR_RATIO = 2.0

# --- CROSS-CELL sigma_run COMPARISON (2026-08-20) --------------------------
# Two fields exist so that a sigma_run from one cell can be honestly set beside
# another's. Both are REQUIRED before any multiple ("ACT is Nx DP") is quoted.
#
#   n_eval                   episodes per run behind THIS row's sigma_run.
#                            SOURCING, corrected 2026-08-20 after audit: the
#                            forecast CSVs cover only DP / smolvla_ft / pi05 on
#                            pusht+libero. The TEN such rows are verified from
#                            their per-run n_eval column. The SIX ACT/ALOHA rows
#                            and pusht-dp-rebuilt have NO runs in those CSVs;
#                            their n=500 comes from PERTASK3_RESULTS.md,
#                            ACTHORIZON_PREREG.md, score_acthorizon.py and
#                            PHASE1C_REPAIR_RESULTS.md. An earlier revision of
#                            this note claimed all 16 were CSV-verified, which
#                            was false. Needed because the
#                            RAW sigma_run carries a battery-sampling term that
#                            scales as 1/sqrt(n): 1.80 at n=500 and 1.80 at
#                            n=2000 are not the same quantity.
#
#   sigma_run_convention     how the spread was taken. RESOLVED 2026-08-20 for
#                            15 of 16 rows.
#     'eval-inclusive' - the spread INCLUDES the eval draw, so the binomial
#         floor at the row's own p_bar is the right correction to reach a
#         population sigma, and two such rows are on the same footing.
#         Evidence: the 2026-08 waves state 'common-seed-list' in their own
#         provenance; power_analysis_phase2.py says in terms "sigma_run includes
#         the 200-ep eval draw" for the Phase-2 family (pusht-dp, libero-dp-*,
#         smolvla-ft); PI05_SCALE is df=7 across 8 seeds on ONE episode set and
#         PI05_PAIRS is a within-mask ANOVA over 16 runs — both fixed-battery.
#     'unknown' - not established. Only smolvla-ft-pooled (2.82, "9
#         budget-matched pairs") remains here.
#
# CORRECTION 2026-08-20: an earlier revision of this note asserted the DP rows
# were PAIRED/variance-reduced, reasoning that pusht-dp-rebuilt's sigma_run 1.80
# sits below the unpaired binomial floor. That floor was computed at SR ~60%,
# taken from a claim in PERTASK3_RESULTS.md that is itself WRONG. DP on PushT
# actually runs at mean SR ~34% pre-rebuild and ~17.6% post-rebuild (measured,
# 251 runs in experiments/forecast/*.csv). At 17.6% and n=500 the floor is 1.70,
# and 1.80 sits ABOVE it. The paired inference is withdrawn; these rows are
# eval-inclusive like the rest.
#
# Consequence, enforced by compare_sigma_run(): a cross-cell multiple is
# quotable only when both rows record n_eval AND share a KNOWN convention.
# Resolving the pre-2026-08 convention is unfinished work, not a formality.


def binom_floor_shape(p):
    """sqrt(p(1-p)) — the SR-dependent shape factor of the binomial SD."""
    return math.sqrt(max(0.0, p * (1.0 - p)))


def sr_floor_mismatch(sr_range, sr_pp, n_eval=None):
    """How badly an observed SR's binomial floor mismatches a regime's band.

    Returns the SMALLEST floor ratio achievable anywhere in `sr_range` (1.0
    when the observed SR's floor is inside the band's floor range), or None
    when the comparison is not decidable (degenerate inputs). Ratio > 1 means
    the observed cell's sampling floor is that many times bigger/smaller than
    anything the regime measured.
    """
    if not sr_range or sr_pp is None:
        return None
    lo, hi = min(sr_range), max(sr_range)
    if not (0.0 <= lo <= hi <= 100.0):
        return None
    if not (0.0 <= sr_pp <= 100.0):
        return None                  # garbage SR must not read as an in-scope cell
    # The mirror rule is decided by SIDE, not magnitude, so it must be answered
    # BEFORE any degenerate-floor bail-out. (Audit 2026-08-20: it sat after, so a
    # 0%/100% eval with no recorded n returned None and the guard silently
    # switched off — the most extreme mismatch was the one that passed.)
    if hi <= 50.0 and sr_pp > 50.0:
        return float("inf")
    if lo >= 50.0 and sr_pp < 50.0:
        return float("inf")
    edge = 0.5 / n_eval if n_eval else 1e-4
    p = min(max(sr_pp / 100.0, edge), 1.0 - edge)
    f_obs = binom_floor_shape(p)
    if f_obs <= 0.0:
        return None
    # f(p)=sqrt(p(1-p)) is continuous, so its image over the band is exactly
    # [min at an endpoint, max at an endpoint or at p=0.5 if the band spans it]
    pts = [min(max(lo / 100.0, edge), 1.0 - edge),
           min(max(hi / 100.0, edge), 1.0 - edge)]
    if lo < 50.0 < hi:
        pts.append(0.5)
    fs = [binom_floor_shape(x) for x in pts]
    f_min, f_max = min(fs), max(fs)
    if f_min <= 0.0:
        return None
    if f_obs > f_max:
        return f_obs / f_max
    if f_obs < f_min:
        return f_min / f_obs
    return 1.0

# sigma_set: SD of the true SR across independently drawn k-episode training
# sets. SCOPE (K3, KCURVE_RESULTS.md 2026-08-05): 19.01 pp was measured at
# k=22 SINGLE-TASK (paired leg 2 / Addendum A; frozen in PAIRED_PREREG.md and
# power_paired.py). It is a SMALL-DATA number and must not be quoted for
# production-scale sets: on a multi-task pool (P=454), sigma_set(pure) falls
# 12.47 -> 3.86 pp (K4B-pooled sigma_run; 2.43 under the original df=2
# reading) from k=22 to k=176 (faster than 1/sqrt(k); see
# SIGMA_SET_PURE_BY_K). At small k, set-level draws are additionally BIMODAL
# (a failure mode, not a Gaussian spread), so Gaussian MDE propagation of any
# small-k sigma_set overstates precision.
SIGMA_SET_PRODUCT_PP = 19.01

# sigma_set(pure) vs k — SmolVLA-ft, LIBERO-object MULTI-TASK pool P=454,
# 6 draws + 2 replicate pairs per k (KCURVE_RESULTS.md, 32 runs, 2026-08-05).
# "pure" = sqrt(sigma_set_total^2 - sigma_run^2); df=5 per k. The k=176 entry
# uses the K4B-pooled sigma_run = 3.36 (df=6, CI [2.17, 7.40]; K4B_RESULTS.md,
# decision bands frozen pre-launch, outcome band B3 indeterminate). Under the
# original df=2 sigma_run (4.51) the entry was 2.43 — the K3 verdict FLIPPED on
# re-measurement and both readings are kept wherever quoted. sigma_set(total,
# k=176) = 5.12 either way, so draws/arm prices are unchanged; the k<=88 shape
# is unaffected.
SIGMA_SET_PURE_BY_K = {
    22: 12.47,   # draws/arm for a 10 pp effect: ~24.8
    44: 4.67,    # ~5.2
    88: 2.75,    # ~2.9
    176: 3.86,   # ~4.1 draws/arm unchanged (prices use sigma_set total 5.12);
                 # 2.43 under the original df=2 sigma_run — see K4B_RESULTS.md
}

REGIMES = {
    # -- PushT / Diffusion Policy ------------------------------------------
    "pusht-dp": dict(
        label="PushT / DiffusionPolicy k=103 (old gym_pusht stack)",
        sigma_run=2.09,   # pp; 8 noise pairs, Phase 1c (power_analysis_phase2.py)
        df=7,  # J-1: analyze_phase2 uses sd(deltas, ddof=1)/sqrt(2)
        sigma_0=1.91,     # pp; harness floor, paired leg 1 arm z (identical ckpt+seed)
        sigma_set=None,
        sr_mean=34.19,
        n_eval=500,
        sigma_run_convention='eval-inclusive',
        k=103, env="pusht", policy="diffusion",
        provenance="Phase 1c, 8 seed-replicate pairs (old gym_pusht stack, "
                   "pre-rebuild). sigma_0 1.91 (df=8, paired leg 1 replicate "
                   "arm z + AMENDMENT_FLOOR) was measured on the REBUILT "
                   "stack and is carried over here as an assumed-transferable "
                   "variance; the harness floor itself was never measured "
                   "pre-rebuild.",
    ),
    "pusht-dp-rebuilt": dict(
        label="PushT / DiffusionPolicy k=103 (rebuilt gym_pusht stack) "
              "-- FLOOR-DOMINATED, not usable as a comparator",
        # RESOLVED 2026-08-20. Two values were in circulation for this ONE
        # regime key, each computed from HALF the evidence:
        #   atlas 1.80  = phase1c_repair pairs only (covmaxR 20.6/20.6 -- a gap
        #                 of exactly 0.00 -- and covminR 14.2/10.6), df=2
        #   corpus 0.32 = phase4_pusht pairs only (mibot 17.6/17.0,
        #                 mitop 17.8/18.0), df=2, PRIOR_CALIBRATION.md:48
        # Pooling all FOUR post-rebuild replicate pairs gives 1.292 at df=4 --
        # which PRIOR_CALIBRATION's own EB-q75 column independently lists as
        # 1.29. ADOPTED 2026-08-20 by founder decision: 1.292 at df=4 is the
        # full-evidence estimate and supersedes both half-evidence values.
        #
        # ANTI-CONSERVATIVE PRICING WARNING. At the pooled mean SR 17.05% and
        # n=500 the binomial floor is 1.682 pp, so sigma_run 1.292 sits BELOW
        # its own eval floor: floor share 169%, floor-subtracted 0.00. There is
        # no retrain variance detectable above sampling in this cell. Pricing a
        # gate from 1.292 therefore UNDER-prices the lottery -- the one
        # direction this programme exists to avoid. For PRICING use the
        # corpus's own EB policy (PRIOR_CALIBRATION column (f) max(atlas,
        # EB-shrunk) = 1.80); 1.292 is the honest MEASUREMENT, not a safe
        # prior. compare_sigma_run() refuses the row as a comparator on the
        # floor-share rule either way.
        sigma_run=1.292, df=4,
        sigma_0=1.91,
        sigma_set=None,
        sr_mean=17.05,
        n_eval=500,
        sigma_run_convention='eval-inclusive',
        k=103, env="pusht", policy="diffusion",
        provenance="RESOLVED 2026-08-20, founder-adopted: sigma_run 1.292 at df=4, pooling all FOUR post-rebuild replicate pairs. Supersedes TWO half-evidence values that were both in circulation for this one key: the atlas's 1.80 (phase1c_repair pairs only, df=2, one gap exactly 0.00; REPAIR_JUDGE_OUTPUT.txt: \"REPORTABLE BUG: replicate spread under the eval floor\") and PRIOR_CALIBRATION.md:48's 0.32 (phase4_pusht pairs only, df=2). THE CELL IS FLOOR-DOMINATED: at the pooled mean SR 17.05%, n=500, the binomial floor is 1.682 pp, so sigma_run sits BELOW it -- floor share 169%, floor-subtracted 0.00. There is no retrain variance detectable above sampling here. DO NOT price from this row and do not use it as a comparator; compare_sigma_run() refuses it on floor share. Absolute SRs shifted vs the old stack (OPS_RUNBOOK sec 9): never pool across the rebuild. NOTE: PERTASK-3 B1's T3 transplant used 1.80 as an EXPLICIT config value, so nothing already scored moves.",
    ),
    # -- LIBERO / Diffusion Policy (from scratch) --------------------------
    "libero-dp-k22": dict(
        label="LIBERO / DiffusionPolicy k=22 (Phase 2)",
        sigma_run=11.43, df=7,  # J-1: analyze_phase2 uses sd(deltas, ddof=1)/sqrt(2) sigma_0=None, sigma_set=None,
        n_eval=200,
        sigma_run_convention='eval-inclusive',
        k=22, env="libero", policy="diffusion",
        provenance="Phase 2, 8 noise pairs (phase2_summary.json noise.sigma_run).",
    ),
    "libero-dp-k31": dict(
        label="LIBERO / DiffusionPolicy k=31 (Tier 1)",
        sigma_run=7.48, df=5,  # J-1: analyze_phase2 uses sd(deltas, ddof=1)/sqrt(2) sigma_0=None, sigma_set=None,
        n_eval=200,
        sigma_run_convention='eval-inclusive',
        k=31, env="libero", policy="diffusion",
        provenance="PHASE2B, 6 noise pairs.",
    ),
    "libero-dp-k44": dict(
        label="LIBERO / DiffusionPolicy k=44 full (Tier 1)",
        sigma_run=6.24, df=7, sigma_0=None, sigma_set=None,
        n_eval=200,
        sigma_run_convention='eval-inclusive',
        k=44, env="libero", policy="diffusion",
        provenance="PHASE2B, 8 seeds of full-44.",
    ),
    # -- LIBERO / SmolVLA fine-tune ----------------------------------------
    "smolvla-ft": dict(
        label="LIBERO / SmolVLA fine-tune k=22 (Tier 2)",
        sigma_run=3.31, df=6,
        sigma_0=0.00,   # 20K-step SmolVLA-ft eval is bit-deterministic: measured 0.00
        sigma_set=SIGMA_SET_PRODUCT_PP,
        n_eval=200,
        sigma_run_convention='eval-inclusive',
        k=22, env="libero", policy="smolvla-ft",
        schedule_stamp=("truncated (20k steps / 30k cosine decay, the "
                        "ecosystem default); ANNEAL-1 (2026-08-21) measured "
                        "the completion effect for this class as NULL at k=88 "
                        "libero_object, NOT at this cell — transferring that "
                        "null to this k=22 single-task cell is an ASSUMPTION, "
                        "not a measurement"),
        provenance="PHASE2C, 6 noise pairs, fine-tune. sigma_0 = 0.00 measured "
                   "(bit-deterministic replicate evals at 20K steps). sigma_set "
                   "19.01 pp from the paired leg 2 design — SCOPED to k=22 "
                   "single-task; at k>=88 on a multi-task pool sigma_set(pure) "
                   "is ~2.4-2.8 pp (SIGMA_SET_PURE_BY_K, KCURVE_RESULTS.md).",
    ),
    # -- LIBERO / pi0.5 fine-tune ------------------------------------------
    "pi05-ft": dict(
        label="LIBERO / pi0.5 fine-tune k=22",
        sigma_run=9.74, df=10, sigma_0=None, sigma_set=None,
        n_eval=200,
        sigma_run_convention='eval-inclusive',
        k=22, env="libero", policy="pi05-ft",
        schedule_stamp=("truncated (20k steps / 30k decay); completion effect UNTESTED for this class -- ANNEAL-1 measured SmolVLA only, K-D licenses no further anneal wave"),
        provenance="PI05_PAIRS_RESULTS (2026-08-09): within-mask ANOVA over "
                   "16 runs, 6 masks, df=10, chi-square CI [6.81, 17.10]; "
                   "excl. the seed-attributable null04 collapse: 5.23 (df=9, "
                   "CI [3.60, 9.55]). OVERTURNS the cell1 df=2 value of 2.02 "
                   "(pairs -0.5/-4.0 — an optimistic low-df draw, same "
                   "failure mode as pusht-dp 2.09-vs-3.58). The pretraining "
                   "ladder DP 11.43 -> SmolVLA-ft 3.31 -> pi05-ft 9.74 is "
                   "NOT monotone; the collapse mode (null04: 50.0 vs 87.5 "
                   "across seeds on one set) is RETRAIN-level. "
                   "CORROBORATED by PI05-CHAR (2026-08-10): null04 taken to "
                   "10 seeds gives sigma_run 12.64 (df=9, CI [8.69, 23.08]) "
                   "on that mask ALONE -- larger than the pooled 9.74, so the "
                   "50.0 was not an unrepeatable draw. Its prereg H1 (>=1 of 8 "
                   "new seeds < 60.0) was FALSIFIED at min 60.5 -- a 0.5 pp "
                   "margin, inside the corpus resolvability floor -- and H1's "
                   "stated consequence (adopt the 5.23 null04-excluded value) "
                   "is refuted by that same wave's H4. 9.74 STANDS.",
    ),
    "pi05-ft-k88-multitask": dict(
        label="LIBERO / pi0.5 fine-tune k=88 multi-task (production k)",
        sigma_run=4.05, df=7, sigma_0=None, sigma_set=None,
        sigma_pertask=11.00,        # pp; floor-subtracted BINDING reading
        sigma_pertask_ci=[6.86, 15.13],
        sigma_pertask_raw=11.82,    # fixed-battery reading; both-readings rule:
                                    # never quote 11.00 without 11.82 beside it
        sigma_pertask_df=7,
        curse_naive_pp=0.94,        # measured naive winner's curse, J=4, n=200
        n_eval=200,
        sigma_run_convention='eval-inclusive',
        k=88, env="libero", policy="pi05-ft",
        schedule_stamp=("truncated (20k steps / 30k decay); completion effect UNTESTED for this class -- ANNEAL-1 measured SmolVLA only, K-D licenses no further anneal wave"),
        provenance="PI05_SCALE W1 (sigma_run 4.05, df=7, CI [2.68, 8.24], "
                   "n=200; 4.23 re-measured at n=1000 in PERTASK-2 — suite "
                   "noise confirmed real, floor 1.53). sigma_pertask from "
                   "PERTASK2_RESULTS (2026-08-14): 8 seeds x 10 tasks x 100 "
                   "eps, set 2d4a570b7025a761, prereg-frozen bands -> "
                   "WEDGE-LIVES; mechanism task-idiosyncratic (cross-task "
                   "corr 0.011 vs ~0.46 under a global effect). Worst task "
                   "SD 20.4 pp across identical-data retrains. curse_naive "
                   "0.94 from PROMOTION_STREAM k=88 cell (n=200); at n=1000 "
                   "on the 2-ckpt PERTASK-2 series naive +0.27, held-out "
                   "promo -0.31 (PERTASK2_CURVE_NOTE, exploratory).",
    ),
    "pi05-ft-k22-multitask": dict(
        label="LIBERO / pi0.5 fine-tune k=22 multi-task",
        sigma_run=6.02, df=7, sigma_0=None, sigma_set=None,
        sigma_pertask=12.01,
        sigma_pertask_ci=[8.70, 15.31],
        sigma_pertask_raw=15.33,
        sigma_pertask_df=7,
        n_eval=200,
        sigma_run_convention='eval-inclusive',
        k=22, env="libero", policy="pi05-ft",
        schedule_stamp=("truncated (20k steps / 30k decay); completion effect UNTESTED for this class -- ANNEAL-1 measured SmolVLA only, K-D licenses no further anneal wave"),
        provenance="PI05_SCALE W2 (sigma_run 6.02, df=7, CI [3.98, 12.25]); "
                   "sigma_pertask from PERTASK_RESULTS (2026-08-13, n=20/task "
                   "-- resolvable at this cell because two tasks are "
                   "degenerate-stable; the k=88 n=20 cell was NOT and was "
                   "superseded by PERTASK-2). Per-task instability barely "
                   "moves with k (12.01 -> 11.00) while the suite average "
                   "falls (6.02 -> 4.05): suite-level reporting hides it.",
    ),
    "smolvla-ft-k88-multitask": dict(
        label="LIBERO / SmolVLA fine-tune k=88 multi-task",
        sigma_run=3.35, df=2, sigma_0=0.00, sigma_set=None,
        sigma_run_n2000=2.11,       # same-wave suite sigma_run, df=7, at
                                    # n=2000/model (Leg A' fresh finals) —
                                    # T1's comparator; the df=2 3.35 kcurve
                                    # value stays beside it, direction-only
        sigma_pertask=6.69,         # pp; floor-subtracted BINDING reading
        sigma_pertask_ci=[5.60, 7.77],
        sigma_pertask_raw=7.33,     # fixed-battery reading; both-readings rule:
                                    # never quote 6.69 without 7.33 beside it
        sigma_pertask_df=7,
        sigma_pertask_suite_stamp=("PER-SUITE (PERTASK-4 T4-2 DIVERGENT, "
                                    "2026-08-21): 6.69/7.33 is libero_object; "
                                    "libero_spatial measured 3.29/4.52 "
                                    "CI [1.18,5.41] same class/k -- never "
                                    "quote sigma_pertask without its suite"),
        schedule_stamp=("truncated (20k steps / 30k cosine decay, the "
                        "ecosystem default); ANNEAL-1 (2026-08-21, prereg "
                        "7b22044e) measured the completion effect at this "
                        "cell as NULL: sigma_pertask 6.94 [5.56,8.31] vs "
                        "7.28 [6.08,8.47] at n=100, mean SR -0.9pp -- "
                        "LOTTERY-SURVIVES, CONCORDANT"),
        curse_naive_pp=2.31,        # PROMOTION_CONFIRM (2026-08-15), 8 runs,
                                    # J=4, n=200; held-out promo +1.19+/-1.38
        n_eval=200,                     # the PRIMARY sigma_run (3.35, df=2, kcurve)
        n_eval_n2000=2000,              # the sigma_run_n2000=2.11 re-measurement
        sigma_run_convention='eval-inclusive',
        k=88, env="libero", policy="smolvla-ft",
        provenance="sigma_run 3.35 (df=2, kcurve k=88 replicate pairs — "
                   "direction only); 2.11 re-measured at df=7, n=2000/model "
                   "on the Leg A' fresh finals (suite SRs 52.5-58.2). "
                   "sigma_pertask from PERTASK-3 Leg A' (2026-08-18, "
                   "Amendment A sha256 8088ab5b..., frozen before the "
                   "training wave; the ORIGINAL Leg A finals were destroyed "
                   "pre-scoring -> NO CLAIM, re-wave founder-authorized): "
                   "8 FRESH SmolVLA-ft retrains, masks verbatim (set "
                   "2d4a570b7025a761, seeds 100-107, 20K steps), 10-task "
                   "libero_object x 200 eps, eval seed 4242, jackknife "
                   "df=7 -> T1 WEDGE-TRANSFERS: the per-task retrain "
                   "lottery is measured in BOTH classes (scope: 2 classes, "
                   "LIBERO, k=88). Cross-task corr -0.036 "
                   "(task-idiosyncratic); worst task SD 11.6 pp. T2-1 "
                   "priced flag rate 22/560 = 3.9% (PASS <= 0.100); T2-2 "
                   "priced pair-HOLD 19/56 = 33.9% inside the pre-derived "
                   "band [0.253, 0.537] (PASS). curse_naive from "
                   "PROMOTION_CONFIRM_RESULTS: prereg-frozen confirmatory "
                   "wave, PC1/PC2/PC3 all PASS (PC3 margin 0.16 pp, "
                   "disclosed); naive +2.31 (sd 2.43, max +7.0), honest "
                   "+1.50, promo +1.19+/-1.38 ~= 0. Cross-class "
                   "matched-set (same episodes as pi05-ft-k88-multitask): "
                   "the curse feeds on checkpoint-curve wobble, not "
                   "final-sigma alone (exploratory, never headline). PER-SUITE STAMP (warranted practice, PERTASK-4 2026-08-20): this "
                   "sigma_pertask is the libero_OBJECT battery. On "
                   "libero_SPATIAL the same class at the same k, same "
                   "config and same eval protocol measured 3.29 pp "
                   "[1.18, 5.41] -- the CIs DO NOT OVERLAP (T4-2 "
                   "DIVERGENT), so sigma_pertask is suite-dependent in "
                   "magnitude and must never be quoted without naming "
                   "the suite. The SHAPE differs too: on libero_object "
                   "no task fell below its floor and the largest carried "
                   "27.61%; on libero_spatial ONE task carries 67.4% of the "
                   "TRUNCATED excess (76.1% of the binding SIGNED total) and "
                   "four tasks sit at/below their floor, contributing "
                   "NEGATIVELY (-14.03 pp^2) rather than zero. T4-1 was INDETERMINATE "
                   "(underpowered at df=7 against a ~3.3 pp effect), so "
                   "nothing is claimed about the wedge holding or "
                   "failing there -- PERTASK4_RESULTS.md.",
    ),
    "smolvla-ft-pooled": dict(
        label="LIBERO / SmolVLA fine-tune, pooled across suites",
        sigma_run=2.82, df=8, sigma_0=0.00, sigma_set=SIGMA_SET_PRODUCT_PP,
        n_eval=200,
        sigma_run_convention='unknown',
        # RESIDUE 3, closed 2026-08-20 as STRUCTURALLY UNRESOLVABLE rather than
        # merely unrecorded. 2.82 is pooled over 9 budget-matched pairs spanning
        # THREE suites (goal 1.08, spatial 1.84, long10 2.94) at SRs from ~22.5
        # to ~95.5 (TIER3_RESULTS.md). It therefore has no single p_bar and so
        # no single binomial floor -- even if the pairing convention were
        # established, the row could not take part in any floor-based
        # comparison, because the floor is not defined for it. Do not spend time
        # resolving the convention: use the per-suite values instead.
        # compare_sigma_run() refuses it on the 'unknown' convention, which is
        # the right outcome for the right reason.
        schedule_stamp=("truncated (20k steps / 30k cosine decay, the "
                        "ecosystem default); ANNEAL-1 (2026-08-21) measured "
                        "the completion effect for this class as NULL at k=88 "
                        "libero_object, NOT at these pooled Tier-3 suites — "
                        "transferring that null here is an ASSUMPTION, not a "
                        "measurement"),
        k=None, env="libero", policy="smolvla-ft",
        provenance="Pooled sigma_ft 2.82 pp over 9 budget-matched pairs "
                   "(df 8); per suite: goal 1.08, spatial 1.84, long10 2.94 "
                   "(Tier-3 repair, TIER3_RESULTS.md resolved block "
                   "2026-07-30).",
        per_suite={"goal": 1.08, "spatial": 1.84, "long10": 2.94},
    ),
    "pusht-act": dict(
        label="PushT / ACT from-scratch (lerobot defaults) — LOW-SR regime",
        sigma_run=0.784, df=7, sigma_0=None, sigma_set=None,
        sr_mean=1.35,
        n_eval=500,
        sigma_run_convention='eval-inclusive',
        k=206, env="pusht", policy="act",
        # the SR span the 8 retrains behind this sigma actually landed in;
        # pre-check 11b refuses to price evals whose binomial floor is
        # >SR_SCOPE_FLOOR_RATIO x away from every point of it (the LOW-SR
        # stamp below, made machine-enforceable — 2026-08-18 audit)
        sr_range=(0.2, 3.0),
        provenance="PERTASK-3 Leg B1 (2026-08-16, prereg sha256 22ff5c67..., "
                   "frozen before launch): J=8 byte-identical-data retrains "
                   "(full 206-ep set, seeds 0-7, dataset sha 77c68ca9...), "
                   "100k steps, PURE lerobot ACT defaults (no crop; the "
                   "prereg's crop parenthetical was a DP-only field, "
                   "deviation recorded pre-scoring), n=500 common-seed eval "
                   "(seed 1000, sync envs). sigma_from_gaps over 28 gaps, "
                   "chi2 CI [0.52, 1.60]. SCOPE: SRs 0.2-3.0 (mean 1.35) -- "
                   "the class is near-floor on PushT at defaults; binomial "
                   "floor 0.52 pp is ~2/3 of the raw sigma "
                   "(floor-subtracted ~0.59 pp); do NOT read this cell as "
                   "'ACT is low-noise' outside the low-SR regime, and no "
                   "per-task decomposition exists (single task: "
                   "sigma_pertask == sigma_run by identity). Dual-scored 56 "
                   "null pairs: eval-only 2/56 false SHIPs vs priced 0/56 "
                   "(transplant 1.80/3.31), T3 TRANSPLANT-CALIBRATED with "
                   "the flag-side VOID disclosure (a >=5 pp drop is "
                   "unrealizable at these SRs). SELF-PRICING SENSITIVITY "
                   "(2026-08-18 audit, quote beside the 0/56): priced with "
                   "THIS cell's own sigma the gate false-ships 1/56 "
                   "(1/56 at 0.784 and at the floor-subtracted 0.59; 0/56 "
                   "only from 1.60 = its own chi2 CI upper, upward) -- the "
                   "0/56 is a property of the 2.3x-larger TRANSPLANT, not "
                   "evidence that self-measured pricing suppresses the "
                   "pathology here; a floor-dominated cell cannot validate "
                   "its own pricing -- PERTASK3_RESULTS.md.",
    ),
    # -- PushT / ACT at CORRECTED eval horizons (ACT-HORIZON, 2026-08-19) --
    # Same eight B1 checkpoints as pusht-act, observed through a shorter
    # open-loop action horizon. Eval-side only; no retraining. pusht-act is NOT
    # superseded - it remains the right row for lerobot ACT defaults.
    "pusht-act-h8": dict(
        label="PushT / ACT from-scratch, eval n_action_steps=8 (chunk 100)",
        sigma_run=4.849, df=7, sigma_0=None, sigma_set=None,
        sr_mean=16.35,
        n_eval=500,
        sigma_run_convention='eval-inclusive',
        k=206, env="pusht", policy="act",
        sr_range=(9.2, 21.4),
        provenance="ACT-HORIZON (2026-08-19, prereg sha256 a4744452..., frozen "
                   "to gs://.../acthorizon/PREREG.sha256 before any episode was "
                   "scored). The EIGHT PUSHT-ACT CHECKPOINTS RE-EVALUATED, not "
                   "retrained: same weights (sha256 manifest in "
                   "gs://.../acthorizon/CKPT_MANIFEST.json), same n=500 sync "
                   "eval at seed 1000, only --policy.n_action_steps changed "
                   "from the ALOHA-inherited default of 100 to 8. chunk_size "
                   "stays 100 and temporal_ensemble_coeff stays null. "
                   "SRs 9.2-21.4 (mean 16.35, spread 12.2 pp). "
                   "sigma_from_gaps over 28 gaps, chi2 CI [3.21, 9.87]; "
                   "binomial floor 1.654 pp = 12% of variance; "
                   "floor-subtracted 4.559 pp (both readings). "
                   "EVAL-HORIZON STAMP: this row prices retrains EVALUATED at "
                   "n_action_steps=8. It is not comparable to pusht-act (h=100) "
                   "as a same-cell measurement - the operating point differs - "
                   "and NO sigma_run multiple against pusht-act is quotable: "
                   "compare_sigma_run() refuses the pair because the sampling "
                   "floors are not on the same scale (1.65 pp at SR 16.4% vs "
                   "0.52 pp at SR 1.4%, ratio 3.20x > the 2.0x tolerance), so "
                   "a ratio would partly be a floor artifact. Qualitatively "
                   "the spread here is far larger than the near-floor "
                   "pusht-act row's. h=8 was the "
                   "PRE-COMMITTED NON-ARGMAX level of the 08-18 sweep, so its "
                   "SR level carries no selection bias, unlike h=16.",
    ),
    "pusht-act-h16": dict(
        label="PushT / ACT from-scratch, eval n_action_steps=16 (chunk 100)",
        sigma_run=4.839, df=7, sigma_0=None, sigma_set=None,
        sr_mean=22.1,
        n_eval=500,
        sigma_run_convention='eval-inclusive',
        k=206, env="pusht", policy="act",
        sr_range=(14.6, 30.0),
        provenance="ACT-HORIZON (2026-08-19, prereg sha256 a4744452..., frozen "
                   "before any episode was scored). Same eight pusht-act "
                   "checkpoints re-evaluated at --policy.n_action_steps=16; no "
                   "retraining. SRs 14.6-30.0 (mean 22.10, spread 15.4 pp). "
                   "sigma_from_gaps over 28 gaps, chi2 CI [3.20, 9.85]; "
                   "binomial floor 1.856 pp = 15% of variance; "
                   "floor-subtracted 4.469 pp (both readings). "
                   "SELECTION DISCLOSURE (prereg sec 2, mandatory): h=16 was "
                   "the ARGMAX of the 2026-08-18 out-of-band sweep over "
                   "{1,8,16,100} on ONE checkpoint, so THE SR LEVEL HERE IS "
                   "OPTIMISTICALLY BIASED and must never be quoted as 'ACT "
                   "achieves 22% on PushT'. The sigma_run/spread estimand is "
                   "far less affected - the sweep selected on one checkpoint's "
                   "mean and carried no information about across-seed spread - "
                   "and h=8 (non-argmax) gives sigma_run 4.849, essentially "
                   "identical. EVAL-HORIZON STAMP as for pusht-act-h8.",
    ),
    # -- ALOHA / ACT (PERTASK-3 Leg B2, the class's NATIVE cell) -----------
    "aloha-transfer-cube-act": dict(
        label="ALOHA sim_transfer_cube / ACT from-scratch (lerobot defaults)",
        sigma_run=6.515, df=7, sigma_0=None, sigma_set=None,
        sr_mean=77.875,
        n_eval=500,
        sigma_run_convention='eval-inclusive',
        k=50, env="aloha", policy="act",
        sr_range=(69.6, 88.4),
        provenance="PERTASK-3 Leg B2 (2026-08-18, prereg sha256 22ff5c67..., "
                   "frozen before launch): J=8 byte-identical-data retrains on "
                   "the full 50-episode lerobot/aloha_sim_transfer_cube_human "
                   "set, seeds 0-7, 100k steps, PURE lerobot ACT defaults, "
                   "n=500 common-seed-list eval (seed 1000, SYNC envs -- async "
                   "vec envs crash at construction on gym-aloha). SRs 69.6-88.4 "
                   "(mean 77.88, spread 18.8 pp). sigma_from_gaps over 28 gaps, "
                   "chi2 CI [4.31, 13.26]. BOTH READINGS: binomial floor at "
                   "n=500 is 1.856 pp = only 8% of the variance, "
                   "floor-subtracted 6.245 pp. "
                   "PAIRING UNLICENSED (UNLICENSED-G2FAIL-20260818): the G2 "
                   "obs-hash alignment check scored 0/6 on gym-aloha (EGL "
                   "pixel-level render nondeterminism across processes; the "
                   "STATE mapping was separately proven deterministic), so the "
                   "frozen demotion applies -- this is a common-seed-list "
                   "UNPAIRED row, no paired statistics and no null-pair gate "
                   "scoring were run against it. Environment provenance in "
                   "gs://.../b2/env_manifest.json (FFmpeg major is part of the "
                   "data path; verify it matches before any cross-wave ACT "
                   "comparison). This is the HEALTHY-regime ACT measurement "
                   "pusht-act could not provide. NO CROSS-CELL MULTIPLE IS "
                   "QUOTABLE AGAINST pusht-dp-rebuilt (settled "
                   "2026-08-20 after three revisions): that row is a "
                   "df=2 estimate from TWO gaps, one exactly 0.00, at "
                   "85% sampling floor, and the ratio's own 95% F "
                   "interval [0.58, 9.26] CONTAINS 1 -- the cells are "
                   "not distinguishable. compare_sigma_run() now "
                   "refuses on df, floor share and ratio CI. What "
                   "stands WITHOUT a comparator: 18.8 pp between two "
                   "byte-identical-data retrains at a full 50-episode "
                   "set, only 8% of that variance sampling "
                   "-- PERTASK3_RESULTS.md sec 7.",
    ),
    "aloha-insertion-act": dict(
        label="ALOHA sim_insertion / ACT from-scratch (lerobot defaults) "
              "-- LOW-SR regime",
        sigma_run=2.261, df=7, sigma_0=None, sigma_set=None,
        sr_mean=11.975,
        n_eval=500,
        sigma_run_convention='eval-inclusive',
        k=50, env="aloha", policy="act",
        sr_range=(9.0, 15.4),
        provenance="PERTASK-3 Leg B2 (2026-08-18, prereg sha256 22ff5c67..., "
                   "frozen before launch): J=8 byte-identical-data retrains on "
                   "the full 50-episode lerobot/aloha_sim_insertion_human set, "
                   "seeds 0-7, 100k steps, PURE lerobot ACT defaults, n=500 "
                   "common-seed-list eval (seed 1000, SYNC envs). SRs 9.0-15.4 "
                   "(mean 11.98, spread 6.4 pp). sigma_from_gaps over 28 gaps, "
                   "chi2 CI [1.50, 4.60]. "
                   "LOW-SR SCOPE STAMP (mandatory on every quote): the "
                   "binomial floor at n=500 is 1.452 pp = 41% OF THE VARIANCE "
                   "-- close to pusht-act's 43% -- so this sigma is "
                   "substantially sampling, not retrain lottery; "
                   "floor-subtracted 1.734 pp, both readings carried. Do NOT "
                   "read the pp gap against aloha-transfer-cube-act (6.515) as "
                   "ACT being 'more stable' on insertion: in RELATIVE terms the "
                   "ordering reverses (18.9% of mean here vs 8.4% there). Same "
                   "architecture, same protocol, same k -- the 2.9x pp "
                   "difference is SR regime, which is why both rows declare "
                   "sr_range and pre-check 11b refuses to cross-price them. "
                   "PAIRING UNLICENSED (UNLICENSED-G2FAIL-20260818), as for the "
                   "transfer_cube row -- PERTASK3_RESULTS.md sec 7.",
    ),
}

DEFAULT_REGIME = "smolvla-ft"

# Runtime caveats. Each of these cost the research program real GPU-days.
CAVEATS = {
    "libero-task-ids": (
        "LIBERO env task_ids differ from dataset episode order (dataset ep 0 is "
        "env task 9 on libero_object). Match the env's task *language string*, "
        "not the episode index; a mismatch trains one task and evals another "
        "(SR ~ 0). [OPS_RUNBOOK sec 8]"
    ),
    "async-envs": (
        "use_async_envs must be false for reproducible seeding: async eval envs "
        "break the seed->initial-state mapping that CRN pairing relies on."
    ),
    "libero-autoreset": (
        "LIBERO vectorized autoreset makes the init-state sequence "
        "outcome-dependent under early termination (LeRobot issue #4152): a "
        "policy that finishes episodes early sees DIFFERENT initial states than "
        "one that does not. CRN pairing on LIBERO therefore requires "
        "sequential (non-vectorized) eval."
    ),
    "cross-stack": (
        "Cross-stack comparisons are invalid: a gym_pusht rebuild shifted SR by "
        f"{STACK_SHIFT_PUSHT_PP} pp (measured, non-uniform). Never pool or compare "
        "SRs across an environment/simulator rebuild; re-anchor with known sets "
        "first. [OPS_RUNBOOK sec 9]"
    ),
    "pairing-validated": (
        "Episode-index-to-initial-state alignment VALIDATED 2026-08-02 (6/6 "
        "comparisons, lerobot 0.6.0): reset-time state at episode index i is "
        "identical across fresh processes, sync/async vector envs, and batch "
        "sizes, for gym_pusht and LIBERO (libero_object task 1). Evidence: "
        "experiments/paired/alignment_check/. Scope: reset-time mapping; on "
        "LIBERO, full closed-loop evals with early termination still shift "
        "later init states (issue #4152) - see the 'libero-autoreset' caveat."
    ),
}

HONESTY_NOTE = (
    "Two different questions, two different prices:\n"
    "  (a) FIXED-CHECKPOINT / fixed-dataset comparison under common random\n"
    "      numbers (CRN): eval noise cancels down to the harness floor sigma_0;\n"
    "      powered at ~2 eval seeds. This is the cheap question.\n"
    "  (b) SELECTION-METHOD comparison across independent retrains: dominated\n"
    "      by sigma_run (and sigma_set when each arm redraws its episode set).\n"
    "      The price is k-DEPENDENT: at k=22 single-task (sigma_set 19.01,\n"
    "      sigma_run 3.31) a 10 pp effect needs ~59 set draws PER ARM; on a\n"
    "      multi-task pool sigma_set(pure) falls 12.47 -> 3.86 pp by k=176\n"
    "      (2.43 under the pre-K4B sigma_run reading; both kept),\n"
    "      and random-draw comparisons at k>=88 measured at ~3-4 draws/arm\n"
    "      (KCURVE_RESULTS.md + K4B_RESULTS.md, bands frozen pre-launch).\n"
    "      At small k, set draws are bimodal (a failure mode, not a Gaussian\n"
    "      spread) — Gaussian MDE propagation overstates precision there.\n"
    "      CRN pairing cannot rescue (b) at any k: sigma_run and sigma_set\n"
    "      live in training, not in the eval draw.\n"
    "\n"
    "PROVENANCE: every atlas sigma was MEASURED on PushT and LIBERO with\n"
    "  DiffusionPolicy / SmolVLA-class policies (ORBIT, 2026-06..08). It may NOT\n"
    "  transfer to other envs, policies, or harnesses — measure your own sigma_run\n"
    "  (a handful of replicate retrains) before trusting these MDEs on a\n"
    "  different stack."
)


def sigma_run_readings(regime, sr_pp=None, n_eval=None, sigma_run=None):
    """Both readings of a row's sigma_run, with its sampling floor separated.

    Returns a dict: raw, n_eval, floor (binomial at the row's own SR and n),
    floor_subtracted, floor_share_of_variance, convention. sr_pp defaults to the
    midpoint of the row's sr_range when it declares one; without an SR the floor
    cannot be computed and floor/floor_subtracted come back None.
    """
    r = get_regime(regime) if isinstance(regime, str) else regime
    sig = sigma_run if sigma_run is not None else r.get("sigma_run")
    n = n_eval if n_eval is not None else r.get("n_eval")
    if sr_pp is None:
        # prefer the row's MEASURED mean SR; the midpoint of a min-max band is
        # not p_bar and does not reproduce the row's own published floor
        # (audit 2026-08-20). Fall back to the midpoint only if no mean exists.
        if r.get("sr_mean") is not None:
            sr_pp = r["sr_mean"]
        elif r.get("sr_range"):
            lo, hi = min(r["sr_range"]), max(r["sr_range"])
            sr_pp = (lo + hi) / 2.0
    out = {"regime": r.get("key"), "raw": sig, "n_eval": n,
           "convention": r.get("sigma_run_convention"),
           "floor": None, "floor_subtracted": None,
           "floor_share_of_variance": None, "sr_pp": sr_pp}
    if sig is None or not n or sr_pp is None:
        return out
    if not (0.0 <= sr_pp <= 100.0):
        return out
    edge = 0.5 / n                   # a 0/n or n/n eval is not literally p=0/1
    p = min(max(sr_pp / 100.0, edge), 1.0 - edge)
    floor = 100.0 * math.sqrt(max(0.0, p * (1 - p)) / n)
    out["floor"] = floor
    out["floor_subtracted"] = math.sqrt(max(0.0, sig * sig - floor * floor))
    out["floor_share_of_variance"] = (floor * floor) / (sig * sig) if sig else None
    return out


def compare_sigma_run(a, b, sr_a=None, sr_b=None):
    """Is a cross-cell sigma_run multiple quotable, and if so what is it?

    Refuses rather than returning a number when the comparison is not sound.
    Returns {'quotable': bool, 'reasons': [...], 'ratio_raw': float|None,
    'ratio_floor_subtracted': float|None, 'readings': {...}}. The refusal
    reasons are the point: a multiple between rows measured at different n, or
    under different/unknown spread conventions, is not a fact about the
    policies (see the module note above compare_sigma_run's fields).
    """
    ra, rb = get_regime(a), get_regime(b)
    A = sigma_run_readings(ra, sr_pp=sr_a)
    B = sigma_run_readings(rb, sr_pp=sr_b)
    reasons = []
    for tag, R, reg in (("a", A, ra), ("b", B, rb)):
        if not R["n_eval"]:
            reasons.append("%s (%s) records no n_eval" % (tag, reg.get("key")))
        if R["convention"] != "eval-inclusive":
            reasons.append(
                "%s (%s) sigma_run convention is %r — an unresolved convention "
                "cannot be set beside an eval-inclusive one"
                % (tag, reg.get("key"), R["convention"]))
        if reg.get("df") is None:
            reasons.append(
                "%s (%s) records no df — a ratio without both dfs has no "
                "uncertainty and must not be quoted" % (tag, reg.get("key")))
        if R["raw"] in (None, 0):
            reasons.append("%s (%s) has no usable sigma_run" % (tag, reg.get("key")))
        # a sigma that is mostly its own sampling floor cannot carry a ratio
        fs = R["floor_share_of_variance"]
        if fs is not None and fs >= 0.75:
            reasons.append(
                "%s (%s) is %.0f%% sampling floor at SR %.1f%% (sigma %.3f vs "
                "floor %.3f) — its floor-subtracted reading is noise, not a "
                "measurement" % (tag, reg.get("key"), 100 * fs, R["sr_pp"],
                                 R["raw"], R["floor"]))
    if A["n_eval"] and B["n_eval"] and A["n_eval"] != B["n_eval"]:
        reasons.append(
            "n_eval differs (%d vs %d): the RAW readings carry different "
            "battery-sampling terms, so compare floor-subtracted or not at all"
            % (A["n_eval"], B["n_eval"]))
    if A["floor"] is not None and B["floor"] is not None:
        fr = max(A["floor"] / B["floor"], B["floor"] / A["floor"])
        if fr > SR_SCOPE_FLOOR_RATIO:
            reasons.append(
                "sampling floors are not on the same scale (%.2f pp at SR "
                "%.1f%% vs %.2f pp at SR %.1f%%, ratio %.2fx > %.1fx): a "
                "sigma ratio here would partly be a floor artifact"
                % (A["floor"], A["sr_pp"], B["floor"], B["sr_pp"], fr,
                   SR_SCOPE_FLOOR_RATIO))
    else:
        reasons.append(
            "an SR is missing for at least one row, so the sampling floors "
            "cannot be checked (pass sr_a / sr_b, or give the row an sr_range)")

    # --- ratio uncertainty. A point multiple between two variance estimates is
    # meaningless without it: at df 7 vs 2 the 95% F interval spans more than an
    # order of magnitude. If the interval covers 1, the cells are not
    # distinguishable and no multiple may be quoted. (Audit 2026-08-20.)
    from . import stats as _st          # local: avoids an import cycle
    ratio_raw = ratio_sub = ratio_ci = None
    dfa, dfb = ra.get("df"), rb.get("df")
    if A.get("raw") and B.get("raw") and dfa and dfb:
        r = A["raw"] / B["raw"]
        lo = r / math.sqrt(_st.f_ppf_975(dfa, dfb))
        hi = r * math.sqrt(_st.f_ppf_975(dfb, dfa))
        ratio_ci = [lo, hi]
        if lo <= 1.0 <= hi:
            reasons.append(
                "the ratio's own 95%% CI [%.2f, %.2f] (F, df %d vs %d) CONTAINS "
                "1 — these two cells are not distinguishable and no multiple is "
                "quotable" % (lo, hi, dfa, dfb))
    quotable = not reasons
    if quotable:
        ratio_raw = A["raw"] / B["raw"]
        fa, fb = A["floor_subtracted"], B["floor_subtracted"]
        ratio_sub = (fa / fb) if (fa is not None and fb) else None
    return {"quotable": quotable, "reasons": reasons, "ratio_raw": ratio_raw,
            "ratio_floor_subtracted": ratio_sub, "ratio_ci_95": ratio_ci,
            "readings": {"a": A, "b": B}}


def get_regime(name):
    """Look up a regime by key; raises KeyError with the available names."""
    key = name.replace("_", "-").lower()
    if key not in REGIMES:
        raise KeyError(
            "unknown regime %r; available: %s" % (name, ", ".join(sorted(REGIMES)))
        )
    return dict(REGIMES[key], key=key)
