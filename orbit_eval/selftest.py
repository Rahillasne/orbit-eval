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

"""``orbit-eval selftest`` — the analyze_paired.py --selftest pattern, ported:
the tool must verify ITSELF on synthetic data before you trust it on yours.

  A  SIZE   the false-positive rate of the tests compare uses matches alpha
            (exact McNemar is conservative by construction: rate <= alpha).
  B  POWER  the simulated rejection rate matches the analytically predicted
            power, for both the paired and the unpaired test.
  C  BIAS   the SR-difference estimator and the sigma-from-replicate-gaps
            estimator are unbiased.
  D  FROZEN the design-rule numbers shipped in the atlas reproduce:
            MDE/draws round-trip, and the product-regime ~59 draws/arm @10pp.

Prints a pass/fail table; exit 0 iff every row is OK.
"""

import math
import random

from . import power
from .stats import (ALPHA, Phi, Z975, binom_pmf, mcnemar_exact,
                    sigma_from_gaps, two_prop_test)
from .atlas import SIGMA_SET_PRODUCT_PP, REGIMES

N_REP = 600


# ------------------------------------------------------------ paired machinery

def _mcnemar_predicted(n, p10, p01, alpha=ALPHA):
    """Exact P(reject) for the McNemar test under cell probs (p10, p01):
    n_disc ~ Bin(n, pd); b | n_disc ~ Bin(n_disc, p10/pd); reject from the
    exact two-sided binomial region."""
    pd = p10 + p01
    pb = p10 / pd
    pmf_nd = binom_pmf(n, pd)
    total = 0.0
    for k in range(1, n + 1):
        if pmf_nd[k] < 1e-12:
            continue
        half = binom_pmf(k, 0.5)
        # largest t with 2*P(X <= t) <= alpha
        c, t = 0.0, -1
        for i in range(k + 1):
            c += half[i]
            if 2 * c <= alpha:
                t = i
            else:
                break
        if t < 0:
            continue
        pk = binom_pmf(k, pb)
        p_rej = sum(pk[: t + 1]) + sum(pk[k - t:])
        total += pmf_nd[k] * p_rej
    return total


def _sim_paired(n, p11, p10, p01, rng):
    """One synthetic CRN-paired eval -> (b, c, diff_pp)."""
    b = c = n11 = 0
    for _ in range(n):
        u = rng.random()
        if u < p11:
            n11 += 1
        elif u < p11 + p10:
            b += 1
        elif u < p11 + p10 + p01:
            c += 1
    diff = 100.0 * (b - c) / n
    return b, c, diff


def _mcnemar_rate(n, p11, p10, p01, n_rep, rng, alpha=ALPHA):
    hits, diffs = 0, []
    for _ in range(n_rep):
        b, c, d = _sim_paired(n, p11, p10, p01, rng)
        diffs.append(d)
        if mcnemar_exact(b, c) <= alpha:
            hits += 1
    return hits / n_rep, diffs


# ------------------------------------------------------------ unpaired machinery

def _two_prop_rate(n, pa, pb_, n_rep, rng, alpha=ALPHA):
    hits = 0
    for _ in range(n_rep):
        ka = sum(1 for _ in range(n) if rng.random() < pa)
        kb = sum(1 for _ in range(n) if rng.random() < pb_)
        _d, _z, p = two_prop_test(ka, n, kb, n)
        if p <= alpha:
            hits += 1
    return hits / n_rep


def _two_prop_predicted(n, pa, pb_, alpha=ALPHA):
    delta = abs(pa - pb_)
    se = math.sqrt(pa * (1 - pa) / n + pb_ * (1 - pb_) / n)
    return Phi(delta / se - Z975) + Phi(-delta / se - Z975)


# ------------------------------------------------------------ the table

def run_selftest(quiet=False, n_rep=N_REP, seed=20260802):
    rows, ok_all = [], True

    def row(section, label, got, want_lo, want_hi):
        nonlocal ok_all
        good = want_lo <= got <= want_hi
        ok_all &= good
        rows.append((section, label, got, want_lo, want_hi, good))

    rng = random.Random(seed)

    # A — size
    pred = _mcnemar_predicted(200, 0.06, 0.06)
    rate, _ = _mcnemar_rate(200, 0.30, 0.06, 0.06, n_rep, rng)
    band = 3 * math.sqrt(max(pred * (1 - pred), 1e-9) / n_rep)
    row("A size ", "McNemar FPR, n=200 p10=p01=.06 (pred %.3f)" % pred,
        rate, max(0.0, pred - band), min(ALPHA + 0.005, pred + band))
    rate = _two_prop_rate(500, 0.35, 0.35, n_rep, rng)
    row("A size ", "two-prop FPR, n=500/arm p=.35", rate, 0.02, 0.08)

    # B — power vs prediction
    pred = _mcnemar_predicted(200, 0.11, 0.03)
    rate, diffs = _mcnemar_rate(200, 0.30, 0.11, 0.03, n_rep, rng)
    band = 3 * math.sqrt(pred * (1 - pred) / n_rep)
    row("B power", "McNemar, true diff +8pp (pred %.2f)" % pred,
        rate, pred - band, pred + band)
    pred = _two_prop_predicted(200, 0.45, 0.35)
    rate = _two_prop_rate(200, 0.45, 0.35, n_rep, rng)
    row("B power", "two-prop, 10pp @ n=200/arm (pred %.2f)" % pred,
        rate, pred - 0.07, pred + 0.07)

    # C — unbiasedness
    bias = sum(diffs) / len(diffs) - 8.0
    row("C bias ", "paired SR-diff estimator bias (pp)", bias, -0.5, 0.5)
    sig, reps = 3.0, 400
    m2 = sum(sigma_from_gaps([rng.gauss(0, sig * math.sqrt(2))
                              for _ in range(8)]) ** 2
             for _ in range(reps)) / reps
    row("C bias ", "sigma_run^2 from replicate gaps (true 9.0)", m2, 8.2, 9.8)

    # D — frozen design-rule numbers
    rt = all(power.draws_needed(2.09, power.mde(2.09, m)) == m
             for m in (2, 3, 5, 8, 20))
    row("D rules", "MDE <-> draws-needed round-trip", 1.0 if rt else 0.0, 1.0, 1.0)
    n59 = power.draws_needed(
        power.method_sigma(SIGMA_SET_PRODUCT_PP, REGIMES["smolvla-ft"]["sigma_run"]),
        10.0)
    row("D rules", "product regime draws/arm @10pp (want 59)", float(n59), 59, 59)

    lines = []
    p = lines.append
    p("=" * 74)
    p("orbit-eval selftest — the tool judging itself on synthetic data")
    p("(%d simulated evals per row; the pattern of analyze_paired.py --selftest)"
      % n_rep)
    p("=" * 74)
    sec = None
    for s, label, got, lo, hi, good in rows:
        if s != sec:
            p("")
            sec = s
        p("  [%s] %-46s %8.3f  want [%.3f, %.3f]  %s"
          % (s, label, got, lo, hi, "OK " if good else "BAD"))
    p("")
    p("self-test %s" % ("PASSED" if ok_all else "FAILED"))
    text = "\n".join(lines)
    if not quiet:
        print(text)
    return (0 if ok_all else 1), rows, text
