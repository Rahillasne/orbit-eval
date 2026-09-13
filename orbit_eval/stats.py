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

"""Statistics core, ported from the ORBIT research repo (pure stdlib).

Sources of truth (ported faithfully, not reinvented):
  experiments/forecast/fit_forecaster.py   phi/Phi/norm_q/binom_* helpers
  experiments/forecast/build_atlas.py      set_hash convention
  experiments/paired/analyze_paired.py     acceptance (budget) gate, sigma from
                                           replicate gaps, bootstrap-CI pattern
  experiments/influence/power_analysis_phase2.py  z constants
All success rates are percentage points (pp, 0..100) unless stated otherwise.
"""

import hashlib
import math
import random

Z975 = 1.959964  # two-sided alpha = 0.05
Z80 = 0.841621   # power = 0.80
ALPHA = 0.05
N_BOOT = 4000    # analyze_paired.py N_BOOT


# ---------------------------------------------------------------- basics

def mean(xs):
    return sum(xs) / len(xs)


def var(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def phi(z):
    """Standard normal pdf. [fit_forecaster.py]"""
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def Phi(z):
    """Standard normal CDF. [fit_forecaster.py]"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_q(p):
    """Acklam-style inverse normal CDF (abs err < 1.15e-9). [fit_forecaster.py]"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------- binomial

def binom_pmf(n, p):
    """List of Binomial(n, p) pmf values k=0..n via lgamma. [fit_forecaster.py]"""
    if p <= 0.0:
        return [1.0] + [0.0] * n
    if p >= 1.0:
        return [0.0] * n + [1.0]
    out = []
    for k in range(n + 1):
        lc = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
              + k * math.log(p) + (n - k) * math.log(1 - p))
        out.append(math.exp(lc))
    return out


def binom_interval(n, p, mass=0.95):
    """Central acceptance region [lo, hi] holding >= mass of Binomial(n,p).
    [fit_forecaster.py]"""
    pmf = binom_pmf(n, p)
    lo, hi, tail = 0, n, (1 - mass) / 2
    c = 0.0
    for k in range(n + 1):
        c += pmf[k]
        if c > tail:
            lo = k
            break
    c = 0.0
    for k in range(n, -1, -1):
        c += pmf[k]
        if c > tail:
            hi = k
            break
    return lo, hi


def binom_lower_floor(n, p, alpha=0.05):
    """Largest k with P(X < k) <= alpha (one-sided 95% floor). [fit_forecaster.py]"""
    pmf = binom_pmf(n, p)
    c, floor = 0.0, 0
    for k in range(n + 1):
        if c <= alpha:
            floor = k
        c += pmf[k]
    return floor


def wilson(k, n, z=Z975):
    """Wilson score interval for a binomial proportion -> (lo, hi) as fractions."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_width_pp(k, n, z=Z975):
    lo, hi = wilson(k, n, z)
    return (hi - lo) * 100.0


# ---------------------------------------------------------------- set hash

def set_hash(episodes):
    """SHA1 of the sorted training-episode list, first 16 hex chars — the atlas
    convention, so replicate groups are detected mechanically. [build_atlas.py]"""
    return hashlib.sha1(
        ",".join(str(int(e)) for e in sorted(episodes)).encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------- budget gate

def budget_gate(final_step, design_steps):
    """OPS_RUNBOOK sec 3/4 via analyze_paired.acceptance_gate: a crashed run
    emits a complete-looking result with a plausible sr. A run is VALID only if
    its realised budget equals the designed budget. Truncation is directional,
    not mean-zero: an undertrained model scores LOW, impersonating whichever
    hypothesis predicts a lower score.

    Returns (ok, reason). Unknown budgets pass with a caveat reason."""
    if design_steps is None or final_step is None:
        return True, "budget unverifiable (final_step/design_steps missing)"
    if int(final_step) != int(design_steps):
        return False, "final_step %s != design %s" % (final_step, design_steps)
    return True, None


def sigma_from_gaps(gaps):
    """sigma from replicate gaps: each gap is a difference of two iid draws, so
    Var(gap) = 2 sigma^2 -> sigma = sqrt(mean(g^2)/2). [analyze_paired.py V4]"""
    if not gaps:
        return None
    return math.sqrt(sum(g * g for g in gaps) / len(gaps) / 2)


# ---------------------------------------------------------------- paired tests

def mcnemar_exact(b, c):
    """Exact McNemar test on discordant pairs: b = A-only successes,
    c = B-only successes. Two-sided p from Binomial(b+c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    pmf = binom_pmf(n, 0.5)
    lo = min(b, c)
    p = 2.0 * sum(pmf[: lo + 1])
    return min(1.0, p)


def mcnemar_one_sided(b, c):
    """One-sided p for 'B is worse than A' (i.e. c small, b large):
    P(X <= c | n_disc, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    pmf = binom_pmf(n, 0.5)
    return min(1.0, sum(pmf[: c + 1]))


def discordant_counts(succ_a, succ_b):
    """(n11, n10, n01, n00) from two aligned per-episode success lists."""
    n11 = n10 = n01 = n00 = 0
    for a, b in zip(succ_a, succ_b):
        if a and b:
            n11 += 1
        elif a and not b:
            n10 += 1
        elif b and not a:
            n01 += 1
        else:
            n00 += 1
    return n11, n10, n01, n00


def paired_boot_ci(succ_a, succ_b, n_boot=N_BOOT, seed=7, level=0.95):
    """Percentile bootstrap CI on the paired SR difference (A - B) in pp,
    resampling EPISODES with replacement so the within-episode pairing is
    preserved (the analyze_paired bootstrap pattern, over episodes)."""
    n = len(succ_a)
    if n == 0 or n != len(succ_b):
        return None
    rng = random.Random(seed)
    diffs = [(1 if a else 0) - (1 if b else 0) for a, b in zip(succ_a, succ_b)]
    reps = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        reps.append(100.0 * s / n)
    reps.sort()
    tail = (1 - level) / 2
    lo = reps[int(tail * n_boot)]
    hi = reps[min(n_boot - 1, int((1 - tail) * n_boot))]
    return lo, hi


# ---------------------------------------------------------------- unpaired

def two_prop_test(k_a, n_a, k_b, n_b):
    """Two-proportion z-test (pooled SE). Returns (diff_pp, z, p_two_sided).
    diff = A - B in pp."""
    p_a, p_b = k_a / n_a, k_b / n_b
    pool = (k_a + k_b) / (n_a + n_b)
    se = math.sqrt(max(pool * (1 - pool), 1e-12) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se if se > 0 else 0.0
    p = 2 * (1 - Phi(abs(z)))
    return (p_a - p_b) * 100.0, z, min(1.0, p)


def fisher_exact_one_sided(k_a, n_a, k_b, n_b):
    """One-sided Fisher exact test: P(A's success count <= k_a) with both
    table margins fixed — the lower tail of Hypergeom(N=n_a+n_b, K=k_a+k_b,
    draws=n_a). Small p is evidence A's true rate is LOWER than B's.

    Exact at any block size: the gate's per-task regression check on the
    UNPAIRED degrade path feeds ~20-episode blocks, where the pooled-z
    Gaussian's realised level drifts above nominal (measured 0.051-0.052 at
    alpha=0.05 for per-task SR 0.70-0.90) and is uncontrolled at extreme
    SRs. Discreteness makes this tail conservative (realised level <=
    alpha). lgamma-based, stdlib only."""
    n_tot, k_tot = n_a + n_b, k_a + k_b
    if k_tot == 0 or k_tot == n_tot:
        return 1.0
    lden = (math.lgamma(n_tot + 1) - math.lgamma(k_tot + 1)
            - math.lgamma(n_tot - k_tot + 1))
    p = 0.0
    for x in range(max(0, k_tot - n_b), k_a + 1):
        p += math.exp(math.lgamma(n_a + 1) - math.lgamma(x + 1)
                      - math.lgamma(n_a - x + 1)
                      + math.lgamma(n_b + 1) - math.lgamma(k_tot - x + 1)
                      - math.lgamma(n_b - (k_tot - x) + 1) - lden)
    return min(1.0, p)


def newcombe_ci(k_a, n_a, k_b, n_b, z=Z975):
    """Newcombe (Wilson-based) CI on the difference of proportions A-B, in pp."""
    p_a, p_b = k_a / n_a, k_b / n_b
    l1, u1 = wilson(k_a, n_a, z)
    l2, u2 = wilson(k_b, n_b, z)
    d = p_a - p_b
    lo = d - math.sqrt((p_a - l1) ** 2 + (u2 - p_b) ** 2)
    hi = d + math.sqrt((u1 - p_a) ** 2 + (p_b - l2) ** 2)
    return lo * 100.0, hi * 100.0


# ---------------------------------------------------------------- comparison

def compare_runs(run_a, run_b, assume_crn=False, n_boot=N_BOOT, seed=7):
    """Compare two runs. Returns a result dict; never prints.

    Pairing decision (honest, in this order):
      - both runs invalid under the budget gate      -> invalid, no stats
      - per-episode successes on both, equal length, and seeds recorded and
        equal — or, under assume_crn, NEITHER seed recorded (or exactly one
        missing while the other is 1000, the silent default)  -> CRN-PAIRED
      - otherwise (incl. one seed recorded != 1000, other
        missing: they almost certainly differ)       -> UNPAIRED + warning
    """
    out = {"run_a": run_a.run_id, "run_b": run_b.run_id,
           "invalid": [], "warnings": [], "paired": False}

    for tag, r in (("A", run_a), ("B", run_b)):
        ok, why = budget_gate(r.final_step, r.design_steps)
        if not ok:
            out["invalid"].append("%s (%s): %s" % (tag, r.run_id, why))
        elif why:
            out["warnings"].append("%s (%s): %s" % (tag, r.run_id, why))
        if r.sr is None:
            out["invalid"].append("%s (%s): no sr" % (tag, r.run_id))
    if out["invalid"]:
        return out

    a_s, b_s = run_a.successes, run_b.successes
    can_pair = (a_s is not None and b_s is not None and len(a_s) == len(b_s)
                and len(a_s) > 0)
    seeds_known = run_a.seed is not None and run_b.seed is not None
    seeds_equal = seeds_known and run_a.seed == run_b.seed
    seeds_none = run_a.seed is None and run_b.seed is None
    # mixed: exactly one seed recorded. Pairable under --assume-crn ONLY when
    # the recorded seed is 1000 (LeRobot's silent default, so the unrecorded
    # run almost certainly used it too); any other recorded seed almost
    # certainly differs from the other run's silent default 1000.
    mixed = (run_a.seed is None) != (run_b.seed is None)
    recorded_seed = run_a.seed if run_a.seed is not None else run_b.seed
    mixed_default = mixed and recorded_seed == 1000

    if can_pair and (seeds_equal
                     or (assume_crn and (seeds_none or mixed_default))):
        out["paired"] = True
        if seeds_none:
            out["warnings"].append(
                "seeds unrecorded on BOTH runs; pairing ASSUMED via "
                "--assume-crn (LeRobot's silent default is seed 1000 for "
                "every eval)")
        elif mixed_default:
            out["warnings"].append(
                "one seed unrecorded, the other recorded as 1000 (LeRobot's "
                "silent default); pairing ASSUMED via --assume-crn")
        n11, n10, n01, n00 = discordant_counts(a_s, b_s)
        out.update(n=len(a_s), n11=n11, n10=n10, n01=n01, n00=n00,
                   sr_a=100.0 * sum(map(bool, a_s)) / len(a_s),
                   sr_b=100.0 * sum(map(bool, b_s)) / len(b_s))
        out["diff_pp"] = out["sr_a"] - out["sr_b"]
        out["p_mcnemar"] = mcnemar_exact(n10, n01)
        out["p_one_sided_b_worse"] = mcnemar_one_sided(n10, n01)
        out["ci"] = paired_boot_ci(a_s, b_s, n_boot=n_boot, seed=seed)
        return out

    # unpaired fallback
    if can_pair and seeds_known and not seeds_equal:
        out["warnings"].append(
            "per-episode successes exist but eval seeds differ "
            "(%s vs %s): CRN pairing unavailable, falling back to UNPAIRED"
            % (run_a.seed, run_b.seed))
    elif can_pair and mixed and not mixed_default:
        out["warnings"].append(
            "one eval seed recorded (%s), the other unrecorded (LeRobot's "
            "silent default is 1000): seeds almost certainly differ — CRN "
            "pairing unavailable, falling back to UNPAIRED"
            % recorded_seed)
    elif can_pair and mixed and mixed_default:
        out["warnings"].append(
            "one seed unrecorded, the other recorded as 1000 (LeRobot's "
            "silent default): cannot verify common random numbers; treating "
            "as UNPAIRED (pass --assume-crn if both used the default seed)")
    elif can_pair and seeds_none:
        out["warnings"].append(
            "per-episode successes exist but seeds are unrecorded: cannot "
            "verify common random numbers; treating as UNPAIRED "
            "(pass --assume-crn if both used the same eval seed)")
    elif (a_s is not None and b_s is not None and len(a_s) != len(b_s)):
        out["warnings"].append(
            "episode counts differ (%d vs %d): CRN pairing impossible, "
            "using unpaired two-proportion comparison"
            % (len(a_s), len(b_s)))
    else:
        out["warnings"].append(
            "per-episode successes unavailable: CRN pairing impossible, "
            "using unpaired two-proportion comparison")

    ka, na = _counts(run_a)
    kb, nb = _counts(run_b)
    if na is None or nb is None:
        out["invalid"].append("n_eval unknown on at least one run: cannot "
                              "even form the unpaired comparison")
        return out
    out.update(n_a=na, n_b=nb, sr_a=100.0 * ka / na, sr_b=100.0 * kb / nb)
    d, z, p = two_prop_test(ka, na, kb, nb)
    out["diff_pp"], out["z"], out["p_two_prop"] = d, z, p
    # one-sided p for "B worse than A": diff = A-B, so evidence is large
    # positive z; p = P(Z >= z_obs) under H0.
    out["p_one_sided_b_worse"] = min(1.0, max(0.0, 1 - Phi(z)))
    out["ci"] = newcombe_ci(ka, na, kb, nb)
    return out


def _counts(run):
    """(successes, n) for a run, from per-episode data or sr*n_eval."""
    if run.successes is not None and len(run.successes) > 0:
        return sum(1 for s in run.successes if s), len(run.successes)
    if run.n_eval and run.sr is not None:
        return int(round(run.sr / 100.0 * run.n_eval)), int(run.n_eval)
    return None, None


# F upper-quantile, needed for a CI on the ratio of two sigma estimates.
# A point multiple between two variance estimates without this is meaningless:
# at df 7 vs 2 the 95% interval spans more than an order of magnitude.
#
# COMPLETE grid over the dfs the atlas ships (2, 4, 5, 6, 7, 8, 10 — df=4
# entered with pusht-dp-rebuilt, adopted 2026-08-20; the earlier table had no
# df=4 entries and its nearest-neighbour fallback mapped (4,7) to (5,7)=5.29 <
# the true 5.52, an ANTI-conservative CI, the opposite of the documented
# behaviour). Values recomputed 2026-08-21 from the regularized incomplete
# beta (stdlib, 200-step bisection); the recomputation also corrected the old
# (7,10) entry 4.20 -> 3.95 (4.20 is F_{0.975}(7,9)).
_F975 = {   # (df1, df2) -> F_{0.975}(df1, df2)
    (2, 2): 39.00, (2, 4): 10.65, (2, 5): 8.43, (2, 6): 7.26, (2, 7): 6.54,
    (2, 8): 6.06, (2, 10): 5.46,
    (4, 2): 39.25, (4, 4): 9.60, (4, 5): 7.39, (4, 6): 6.23, (4, 7): 5.52,
    (4, 8): 5.05, (4, 10): 4.47,
    (5, 2): 39.30, (5, 4): 9.36, (5, 5): 7.15, (5, 6): 5.99, (5, 7): 5.29,
    (5, 8): 4.82, (5, 10): 4.24,
    (6, 2): 39.33, (6, 4): 9.20, (6, 5): 6.98, (6, 6): 5.82, (6, 7): 5.12,
    (6, 8): 4.65, (6, 10): 4.07,
    (7, 2): 39.36, (7, 4): 9.07, (7, 5): 6.85, (7, 6): 5.70, (7, 7): 4.99,
    (7, 8): 4.53, (7, 10): 3.95,
    (8, 2): 39.37, (8, 4): 8.98, (8, 5): 6.76, (8, 6): 5.60, (8, 7): 4.90,
    (8, 8): 4.43, (8, 10): 3.85,
    (10, 2): 39.40, (10, 4): 8.84, (10, 5): 6.62, (10, 6): 5.46, (10, 7): 4.76,
    (10, 8): 4.30, (10, 10): 3.72,
}

_F975_DFS = sorted({a for a, _ in _F975})     # 2, 4, 5, 6, 7, 8, 10


def _df_round_down(df):
    """Largest tabulated df <= df (the smallest tabulated df when df sits
    below the grid). F_{0.975} is decreasing in BOTH dfs, so rounding each df
    DOWN yields an F at least as large as the true one — a genuinely WIDER
    CI. (dfs below 2 do not occur in the atlas; they clamp to 2, the one case
    that would not err wide.)"""
    lower = [d for d in _F975_DFS if d <= df]
    return lower[-1] if lower else _F975_DFS[0]


def f_ppf_975(df1, df2):
    """F_{0.975}(df1, df2). Table lookup for the dfs the atlas actually uses;
    off-grid dfs round each df DOWN to the nearest tabulated value, which
    errs LARGE (wider CI) because F_{0.975} decreases in both dfs."""
    key = (int(df1), int(df2))
    if key in _F975:
        return _F975[key]
    return _F975[(_df_round_down(int(df1)), _df_round_down(int(df2)))]
