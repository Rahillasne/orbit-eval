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

"""Power / MDE design rules, ported from power_analysis_phase2.py, plus the
paired-wave generative model (power_paired.py) used by the self-test.

Model [power_analysis_phase2.py]: a "run" of a fixed subset draws
SR ~ N(mu_subset, sigma_run^2); two-arm comparisons with m seeds per arm use
the standard normal two-sample formulas with sigma treated as known.

    MDE = (z_{1-a/2} + z_{power}) * sigma * sqrt(2/m) = 2.80 * sigma * sqrt(2/m)

For a selection-METHOD comparison in which each arm redraws its episode set,
the per-draw sigma is sqrt(sigma_set^2 + sigma_run^2). SCOPE (K3,
KCURVE_RESULTS.md): with the k=22 SINGLE-TASK measurement (sigma_set 19.01,
sigma_run 3.31) that is 19.30 pp, giving ~59 set draws per arm for a 10 pp
effect — a small-data price. sigma_set is k-dependent: on a multi-task pool
sigma_set(pure) falls 12.47 -> 3.86 pp from k=22 to k=176 (K4B-pooled; atlas
SIGMA_SET_PURE_BY_K), and at small k set draws are bimodal, so the Gaussian
MDE here overstates precision in exactly the regime where sigma_set is
largest. CRN pairing cannot reduce (b) at any k: sigma_set and sigma_run
live in training, not in the eval draw.
"""

import math

from .stats import Phi, Z975, Z80

# chi-square (0.05, 0.95) quantiles by df, for CIs on sigma_run
# [power_analysis_phase2.py]
CHI2_05_95 = {
    4: (0.711, 9.488), 5: (1.145, 11.070), 6: (1.635, 12.592),
    7: (2.167, 14.067), 8: (2.733, 15.507),
}


def sigma_ci(sigma, df):
    """90% CI on sigma given a chi-square-distributed variance estimate with df.
    [power_analysis_phase2.py]"""
    if df is None or df not in CHI2_05_95:
        return None
    lo_q, hi_q = CHI2_05_95[df]
    return sigma * math.sqrt(df / hi_q), sigma * math.sqrt(df / lo_q)


def mde(sigma, m, z_a=Z975, z_b=Z80):
    """Minimal detectable effect (pp), two-arm, m draws/arm, alpha=.05
    power=.80. [power_analysis_phase2.py]"""
    return (z_a + z_b) * sigma * math.sqrt(2.0 / m)


def draws_needed(sigma, delta, z_a=Z975, z_b=Z80):
    """Draws per arm to detect a true gap of delta pp (alpha=.05, power=.80).
    [power_analysis_phase2.py seeds_needed — matches the source EXCEPT a 1e-9
    ceil guard, a deliberate port deviation: the source's bare ceil returns
    N+1 when FP noise pushes the argument fractionally above an integer (e.g.
    delta = mde(m) exactly); the guard keeps the MDE <-> draws round-trip
    exact and changes no published table value (checked against
    POWER_ANALYSIS.md tables 1-2 and the 59-draw product-regime number).]"""
    return math.ceil(2.0 * ((z_a + z_b) * sigma / delta) ** 2 - 1e-9)


def two_sided_power(delta, sigma, m):
    """Power of the two-arm test at true gap delta, m draws/arm.
    [power_analysis_phase2.py]"""
    se = sigma * math.sqrt(2.0 / m)
    return Phi(delta / se - Z975) + Phi(-delta / se - Z975)


def method_sigma(sigma_set, sigma_run):
    """Per-draw sigma for a method comparison where each arm's episode set is
    an independent draw: variance components add."""
    return math.sqrt(sigma_set ** 2 + sigma_run ** 2)


def vrf_true(w, m, d_bar, sigma_set, sigma_0):
    """True variance-reduction factor of pairing at overlap distance m under
    the paired-wave generative model. [power_paired.py]"""
    num = 2 * sigma_set ** 2 + 2 * sigma_0 ** 2
    den = 2 * sigma_set ** 2 * ((1 - w) * m / d_bar + w) + 2 * sigma_0 ** 2
    return num / den
