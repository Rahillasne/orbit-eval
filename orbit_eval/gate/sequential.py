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

"""Sequential / anytime-valid Ship-Gate — the FROZEN v1 math spec (exact).

v0/v0.1 (engine.py) are fixed-sample: their alpha is honest for exactly ONE
look at a pre-sized eval, and re-running that gate on a growing eval inflates
the false-ship rate. This module is the v1 sequential layer: two e-processes
whose running value may be inspected after EVERY observation and acted on at
ANY data-dependent stopping time, with the false-crossing rate bounded by
Ville's inequality — P(sup_t E_t >= 1/alpha) <= alpha under H0 for any
nonnegative supermartingale with E_0 = 1 (Ville 1939; Ramdas, Grunwald, Vovk
& Shafer 2023, "Game-theoretic statistics and safe anytime-valid inference",
Statistical Science 38(4) — the e-process framing used throughout).
Determinism: NO RNG anywhere in this module — identical streams give
identical states, and identical streams plus an explicit `now` give
byte-identical signed records.

EPISODE STREAM — EpisodeSequential, record method "e-process-episodes".
  A paired CRN eval consumed one episode at a time; concordant pairs cancel
  exactly as in McNemar, so the informative observations are the discordant
  pairs: X_i = 1 iff the candidate won discordant pair i; H0_ship: p <= 1/2.
    E_t = (1/9) * sum_{lambda in {0.1,...,0.9}}
                  prod_{i<=t} (1 + lambda*(2*X_i - 1))
  the uniform betting mixture over the frozen lambda grid (Robbins' mixture
  method; Waudby-Smith & Ramdas 2023, "Estimating means of bounded random
  variables by betting", JRSS-B, betting supermartingales): under any
  p <= 1/2 each factor has mean 1 + lambda*(2p-1) <= 1, so every per-lambda
  product is a nonnegative supermartingale and the uniform mixture preserves
  that. Per-lambda LOG products are tracked for numerical stability; the
  mirror e-process for the worse direction substitutes X -> 1-X.
  Always-valid p-value: p_t = min(1, 1/max_{s<=t} E_s) — monotone
  nonincreasing by construction (Ville).
  Confidence sequence on p by Beta(1,1)-mixture inversion: with
  S_t = sum X_i,
    E_t(p0) = exp( lgamma(S_t+1) + lgamma(t-S_t+1) - lgamma(t+2)
                   - [S_t*ln(p0) + (t-S_t)*ln(1-p0)] )
  and CS_t = { p0 : max_{s<=t} E_s(p0) < 1/alpha }, implemented on the
  frozen p0 grid 0.001..0.999 step 0.001 with a running max per grid point
  (O(grid) per discordant update); reported as [min, max] of the surviving
  grid points. Delta mapping (DISCLOSED approximate; the exact joint CS is
  v2): delta_pp ~= 100 * p_d_hat * (2p - 1) evaluated at the CS endpoints,
  p_d_hat = (b+c)/n treated as known, its Wilson band recorded alongside.

RETRAIN STREAM — RetrainSequential, record method "e-process-retrains".
  Observations are per-retrain-pair deltas d_j (pp) with KNOWN variance
  sigma_j^2 = 2*sigma_run^2 + se_eval_j^2 (sigma_run resolved exactly as the
  fixed engine resolves it — explicit > regime > None-with-disclosure;
  se_eval_j from that pair's own eval). Sufficient statistics
  S_t = sum d_j/sigma_j^2, V_t = sum 1/sigma_j^2. One-sided half-normal
  mixture e-process for H0: delta <= 0, prior scale tau
  (config.sequential_tau, default 5.0 pp, DISCLOSED in every record):
    A_t = V_t + 1/tau^2
    E_t = (2/(tau*sqrt(A_t))) * exp(S_t^2/(2*A_t)) * Phi(S_t/sqrt(A_t))
  — the likelihood-ratio martingale exp(delta*S_t - delta^2*V_t/2)
  integrated against the half-N(0, tau^2) prior on delta > 0 (Robbins 1970,
  "Statistical methods related to the law of the iterated logarithm", Ann.
  Math. Statist. 41 — normal-mixture supermartingales). Sanity anchor (spec,
  asserted in tests): at t=0, S=0, V=0 -> A=1/tau^2 ->
  E = (2/(tau*(1/tau)))*(1/2) = 1 exactly; E is a nonnegative martingale
  under delta = 0. Mirror for the worse direction: S -> -S. Two-sided
  Robbins normal-mixture confidence sequence for delta, valid simultaneously
  over ALL t at level 1-alpha (coverage verified by simulation in the test
  suite):
    S_t/V_t +/- (1/V_t)*sqrt( (V_t + 1/tau^2)
                              * (ln(1 + tau^2*V_t) + 2*ln(1/alpha)) )

DECISIONS (both streams; min_effect and winner's-curse semantics identical
to the fixed-sample engine, frozen order — worse direction first):
  HOLD-EVIDENCE  when max_{s<=t} E_worse >= 1/alpha
  SHIP-EVIDENCE  when max_{s<=t} E_ship >= 1/alpha AND
                 (delta_hat_pp - curse_correction_pp) >= min_effect_pp
  CONTINUE       otherwise (the state carries the CS and both always-valid p)
PRECEDENCE IS PART OF THE FROZEN SPEC (v1 addendum, resolving the listing
ambiguity in the spec text): the WORSE direction is checked first, exactly
as the fixed engine tests p_worse before p_ship — because crossings are
sup-latched (Ville), a stream whose ship and hold e-processes have BOTH
crossed 1/alpha decides HOLD-EVIDENCE permanently, and a reimplementation
must not invert this order (pinned by
test_hold_takes_precedence_once_both_crossed).
Records map SHIP-EVIDENCE / HOLD-EVIDENCE / CONTINUE onto the frozen verdict
vocabulary SHIP / HOLD / COLLECT-MORE so ledgers, exit codes and downstream
consumers are unchanged; the sequential decision string itself is preserved
in statistics['decision']. Records are schema-v2 (records.build_record);
e-values are reported as exp(min(log_e, 700)) so a signed record can never
contain an inf (canonical_json is allow_nan=False), with the EXACT log_e
values recorded alongside.
"""

import math
from datetime import datetime, timezone

from .. import stats
from .engine import (CAVEAT_FLOOR_2PP, GateInputError, curse_from_prior,
                     resolve_sigma_run)
from .records import build_record

# Frozen v1 spec constants.
LAMBDA_GRID = tuple(k / 10.0 for k in range(1, 10))     # 0.1 .. 0.9
_LOG_UP = tuple(math.log1p(l) for l in LAMBDA_GRID)     # log(1 + lambda)
_LOG_DOWN = tuple(math.log1p(-l) for l in LAMBDA_GRID)  # log(1 - lambda)
_N_LAMBDA = len(LAMBDA_GRID)
_LOG_N_LAMBDA = math.log(float(_N_LAMBDA))
_LOG2 = math.log(2.0)

P0_GRID = tuple(i / 1000.0 for i in range(1, 1000))     # 0.001 .. 0.999
_LOG_P0 = tuple(math.log(p) for p in P0_GRID)
_LOG_Q0 = tuple(math.log(1.0 - p) for p in P0_GRID)

# exp() overflow guard for JSON-safe records (math.exp overflows past
# ~709.78; canonical_json rejects inf). log_e values are always exact.
_E_LOG_CAP = 700.0

# Sequential decision -> frozen fixed-sample verdict vocabulary, so ledger
# consumers and exit-code semantics are unchanged.
DECISION_TO_VERDICT = {"SHIP-EVIDENCE": "SHIP",
                       "HOLD-EVIDENCE": "HOLD",
                       "CONTINUE": "COLLECT-MORE"}


# ------------------------------------------------------------- math helpers

def _lse_mix(logs):
    """log of the uniform mixture (1/9)*sum(exp(l)) of the per-lambda log
    products — max-shifted logsumexp, exact at t=0 (all l=0 -> 0.0)."""
    m = max(logs)
    s = 0.0
    for v in logs:
        s += math.exp(v - m)
    return m + math.log(s) - _LOG_N_LAMBDA


def _cap_exp(log_x):
    """exp(log_x) capped at exp(700) so records stay JSON-finite (the exact
    log value is recorded alongside every capped e-value)."""
    return math.exp(log_x if log_x < _E_LOG_CAP else _E_LOG_CAP)


def _av_p(max_log_e):
    """Always-valid p from the running max log e-value: min(1, 1/max E)."""
    return min(1.0, math.exp(-max_log_e))


def episode_p0_log_e(t, s, p0):
    """log E_t(p0) of the Beta(1,1)-mixture e-value for a hypothesised
    discordant-win probability p0, given t discordant pairs with s candidate
    wins (frozen spec formula, lgamma-based):

      log E = lgamma(s+1) + lgamma(t-s+1) - lgamma(t+2)
              - [s*ln(p0) + (t-s)*ln(1-p0)]

    The integrand is the likelihood ratio Beta-mixture marginal /
    p0-likelihood, an exact nonnegative martingale under Bernoulli(p0), so
    max_{s<=t} E_s(p0) >= 1/alpha excludes p0 from the level-(1-alpha)
    confidence sequence (Ville). At t=0 the value is 0.0 exactly (E=1).
    The subtraction order matches the class's grid loop bit-for-bit, so
    grid values and direct evaluations are exactly comparable."""
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1), got %r" % (p0,))
    return (math.lgamma(s + 1) + math.lgamma(t - s + 1) - math.lgamma(t + 2)
            - s * math.log(p0) - (t - s) * math.log(1.0 - p0))


def log_phi_cdf(x):
    """log Phi(x), stable in the deep lower tail.

    For x >= -6 the direct log(Phi(x)) is accurate (the erf cancellation in
    Phi keeps >= ~8 significant digits down to -6, so the exact boundary
    point takes the more accurate direct branch). Below that, the Mills
    asymptotic Phi(x) = phi(x)/(-x) * (1 - 1/x^2 + 3/x^4 - 15/x^6 + ...)
    (Abramowitz & Stegun 26.2.12) with terms through x^-6: relative error
    ~105/x^8 (< 6e-5 just below x=-6, vanishing beyond), far inside what an
    e-value threshold comparison can see."""
    if x >= -6.0:
        return math.log(stats.Phi(x))
    x2 = x * x
    corr = 1.0 - 1.0 / x2 + 3.0 / (x2 * x2) - 15.0 / (x2 * x2 * x2)
    return (-0.5 * x2 - math.log(-x) - 0.5 * math.log(2.0 * math.pi)
            + math.log(corr))


def _stamp(now):
    """ISO-8601 UTC timestamp: caller-supplied verbatim, else the clock
    (records.py itself never reads the clock — engine.gate convention)."""
    if now is not None:
        return now
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------- shared base

class _SequentialBase(object):
    """Config validation, curse resolution and the frozen decision rule
    shared by both streams. Curse semantics identical to the fixed engine's
    PRIOR path (a stream has no checkpoint eval history to bootstrap):
    selection='final' -> 0; 'max-over-checkpoints' -> the atlas prior scaled
    to J=n_checkpoints (engine.curse_from_prior, PI05_CELL1_RESULTS), with
    n_checkpoints REQUIRED (the engine's no-J misuse rule)."""

    def __init__(self, config):
        errs = config.validate()
        if errs:
            raise GateInputError("invalid GateConfig: " + "; ".join(errs))
        if (config.selection == "max-over-checkpoints"
                and config.n_checkpoints is None):
            raise GateInputError(
                "selection='max-over-checkpoints' needs n_checkpoints (J) "
                "for the atlas-prior curse correction — a sequential stream "
                "has no checkpoint eval history to bootstrap a measured one")
        self.config = config
        self._alpha = float(config.alpha)
        self._log_thresh = -math.log(self._alpha)   # log(1/alpha)
        if config.selection == "final":
            self._curse = 0.0
            self._curse_method = "none"
        else:
            self._curse = curse_from_prior(config.n_checkpoints,
                                           config.curse_prior_pp,
                                           config.curse_prior_j)
            self._curse_method = "atlas-prior-scaled"
        # Running maxima of log E per direction; E_0 = 1 -> log 0.0, so the
        # always-valid p starts at exactly 1 and the sup includes t=0.
        self._mx_ship = 0.0
        self._mx_hold = 0.0

    # cheap read-only monitors (used by the calibration simulations; state()
    # builds a full dict and is the surface the CLI consumes)
    @property
    def max_log_e_ship(self):
        """Running max of log E_ship over all times seen (includes E_0=1)."""
        return self._mx_ship

    @property
    def max_log_e_hold(self):
        """Running max of log E_hold over all times seen (includes E_0=1)."""
        return self._mx_hold

    def _decision(self, delta_effective_pp):
        """Frozen decision order (worse direction first, mirroring the fixed
        engine's p_worse-before-p_ship order): HOLD-EVIDENCE on the mirror
        crossing; SHIP-EVIDENCE on the ship crossing AND the curse-corrected
        delta clearing min_effect (the ship rule unchanged in FORM); else
        CONTINUE. Crossings are sup-based (Ville), so evidence never
        un-crosses; the min_effect leg is evaluated at the CURRENT delta."""
        if self._mx_hold >= self._log_thresh:
            return "HOLD-EVIDENCE"
        if (self._mx_ship >= self._log_thresh
                and delta_effective_pp is not None
                and delta_effective_pp >= self.config.min_effect_pp):
            return "SHIP-EVIDENCE"
        return "CONTINUE"

    def _curse_assumption(self):
        """The engine's atlas-prior curse disclosure, or None."""
        if self._curse_method != "atlas-prior-scaled":
            return None
        return ("ASSUMPTION: winner's-curse correction scaled from the "
                "measured J=%d prior as %.1f*sqrt(ln(J)/ln(%d)) pp "
                "(PI05_CELL1_RESULTS: +4.00 pp mean inflation, max +12.5, "
                "measured prospectively; the sqrt-log scaling beyond J=%d "
                "is assumed, not measured)"
                % (self.config.curse_prior_j, self.config.curse_prior_pp,
                   self.config.curse_prior_j, self.config.curse_prior_j))


# ------------------------------------------------------------ episode gate

class EpisodeSequential(_SequentialBase):
    """Anytime-valid episode-stream gate on a paired CRN eval (frozen v1
    spec, part A). Feed update(inc_success, cand_success) per CRN episode;
    discordant pairs drive the betting-mixture e-process (ship) and its
    X -> 1-X mirror (hold); concordant pairs only grow n. state() is the
    machine surface (see keys below); record(now=...) signs a schema-v2
    decision record with method "e-process-episodes".

    track_cs=False disables the O(999)-per-discordant-update confidence-
    sequence grid (cs_p / cs_delta_pp_approx then report None); e-processes,
    always-valid p and decisions are unaffected. The heavy seeded
    calibration tests use it; production streams keep the default True.

    state() keys: n, b, c (spec: b = incumbent-only, c = candidate-only
    wins, the engine's frozen convention), n_discordant, e_ship, e_hold
    (capped at exp(700) for JSON safety), log_e_ship, log_e_hold (exact,
    current), max_log_e_ship, max_log_e_hold (running sup), p_ship_av,
    p_hold_av (always-valid), cs_p ([lo, hi] surviving-grid envelope or
    None), cs_delta_pp_approx (the DISCLOSED approximate mapping), p_d_hat,
    p_d_wilson (fractions), sr_incumbent_pp, sr_candidate_pp, delta_hat_pp,
    curse_correction_pp, delta_effective_pp, decision, mode."""

    def __init__(self, config, track_cs=True):
        _SequentialBase.__init__(self, config)
        self._track_cs = bool(track_cs)
        self._n = 0          # all CRN episodes seen
        self._b = 0          # incumbent-only wins  (X = 0)
        self._c = 0          # candidate-only wins  (X = 1)
        self._k_inc = 0
        self._k_cand = 0
        self._lp_ship = [0.0] * _N_LAMBDA   # per-lambda log products
        self._lp_hold = [0.0] * _N_LAMBDA
        # running max log E_s(p0) per grid point; E_0(p0) = 1 -> 0.0
        self._cs_max = [0.0] * len(P0_GRID) if self._track_cs else None

    def update(self, inc_success, cand_success):
        """Consume one CRN episode (truthy success flags). Concordant pairs
        carry no discordance information and leave both e-processes exactly
        unchanged; a discordant pair multiplies every per-lambda product by
        (1 + lambda) on a win for its direction, (1 - lambda) on a loss —
        tracked as log products (frozen spec) so 2000-step streams cannot
        overflow. The running maxima are refreshed via the exact mixture
        logsumexp, gated by the free bound log E <= max_lambda log-product
        (mean of exps <= max exp), which skips the exp calls whenever the
        bound cannot beat the stored sup. Returns None (call state())."""
        inc = bool(inc_success)
        cand = bool(cand_success)
        self._n += 1
        if inc:
            self._k_inc += 1
        if cand:
            self._k_cand += 1
        if inc == cand:
            return
        if cand:
            self._c += 1
            up, down = _LOG_UP, _LOG_DOWN
        else:
            self._b += 1
            up, down = _LOG_DOWN, _LOG_UP
        lp = self._lp_ship
        m = -1e308
        for k in range(_N_LAMBDA):
            v = lp[k] + up[k]
            lp[k] = v
            if v > m:
                m = v
        if m > self._mx_ship:
            le = _lse_mix(lp)
            if le > self._mx_ship:
                self._mx_ship = le
        lp = self._lp_hold
        m = -1e308
        for k in range(_N_LAMBDA):
            v = lp[k] + down[k]
            lp[k] = v
            if v > m:
                m = v
        if m > self._mx_hold:
            le = _lse_mix(lp)
            if le > self._mx_hold:
                self._mx_hold = le
        if self._track_cs:
            t = self._b + self._c
            s = self._c
            ts = t - s
            base = (math.lgamma(s + 1) + math.lgamma(ts + 1)
                    - math.lgamma(t + 2))
            cs = self._cs_max
            lp0, lq0 = _LOG_P0, _LOG_Q0
            for i in range(len(cs)):
                v = base - s * lp0[i] - ts * lq0[i]
                if v > cs[i]:
                    cs[i] = v

    def _cs_endpoints(self):
        """[min, max] of the grid points still inside the CS (the survived
        set is an interval: each E_s(p0) is quasi-convex in p0 with its
        trough at the running empirical rate, so the sublevel sets are
        intervals and their intersection over s stays one). None when the
        grid is disabled or (degenerately) every point is excluded."""
        if not self._track_cs:
            return None
        cs, th = self._cs_max, self._log_thresh
        lo = hi = None
        for i in range(len(cs)):
            if cs[i] < th:
                lo = P0_GRID[i]
                break
        if lo is None:
            return None
        for i in range(len(cs) - 1, -1, -1):
            if cs[i] < th:
                hi = P0_GRID[i]
                break
        return [lo, hi]

    def state(self):
        """Current machine-readable state (keys documented on the class)."""
        n, b, c = self._n, self._b, self._c
        t = b + c
        log_e_ship = _lse_mix(self._lp_ship)
        log_e_hold = _lse_mix(self._lp_hold)
        if n:
            delta_hat = 100.0 * (c - b) / n
            delta_eff = delta_hat - self._curse
            p_d_hat = t / n
            wl, wh = stats.wilson(t, n)
            p_d_wilson = [wl, wh]
            sr_inc = 100.0 * self._k_inc / n
            sr_cand = 100.0 * self._k_cand / n
        else:
            delta_hat = delta_eff = p_d_hat = None
            p_d_wilson = sr_inc = sr_cand = None
        cs_p = self._cs_endpoints()
        if cs_p is not None and n:
            cs_delta = [100.0 * p_d_hat * (2.0 * cs_p[0] - 1.0),
                        100.0 * p_d_hat * (2.0 * cs_p[1] - 1.0)]
        else:
            cs_delta = None
        return {
            "mode": "episodes",
            "n": n, "b": b, "c": c, "n_discordant": t,
            "e_ship": _cap_exp(log_e_ship),
            "e_hold": _cap_exp(log_e_hold),
            "log_e_ship": log_e_ship, "log_e_hold": log_e_hold,
            "max_log_e_ship": self._mx_ship,
            "max_log_e_hold": self._mx_hold,
            "p_ship_av": _av_p(self._mx_ship),
            "p_hold_av": _av_p(self._mx_hold),
            "cs_p": cs_p,
            "cs_delta_pp_approx": cs_delta,
            "p_d_hat": p_d_hat, "p_d_wilson": p_d_wilson,
            "sr_incumbent_pp": sr_inc, "sr_candidate_pp": sr_cand,
            "delta_hat_pp": delta_hat,
            "curse_correction_pp": self._curse,
            "delta_effective_pp": delta_eff,
            "decision": self._decision(delta_eff),
        }

    def record(self, now=None, source=None, extra_assumptions=None):
        """Signed schema-v2 decision record for the current state, method
        "e-process-episodes". `now` verbatim (None stamps UTC here — pass it
        explicitly for byte-reproducible records); `source` is recorded in
        inputs.stream_source; extra_assumptions recorded verbatim."""
        st = self.state()
        created = _stamp(now)
        verdict = DECISION_TO_VERDICT[st["decision"]]
        inv_alpha = 1.0 / self._alpha
        reasons, caveats, warnings, assumptions = [], [], [], []
        if extra_assumptions:
            assumptions.extend(str(a) for a in extra_assumptions)
        assumptions.append(
            "ASSUMPTION: anytime-valid episode-stream gate (method "
            "'e-process-episodes'): uniform betting-mixture e-process over "
            "lambda grid 0.1..0.9 on discordant pairs; Ville's inequality "
            "bounds the false-crossing rate at alpha under any "
            "p(candidate wins discordant) <= 1/2, at ANY data-dependent "
            "stopping time — peeking after every episode is valid")
        assumptions.append(
            "ASSUMPTION: cs_delta_pp_approx uses the DISCLOSED approximate "
            "mapping delta_pp ~= 100*p_d_hat*(2p-1) at the CS-on-p "
            "endpoints, with p_d_hat = (b+c)/n treated as known (its Wilson "
            "band is recorded as p_d_wilson); the exact joint CS on "
            "(p_d, p) is v2")
        if self.config.comparison_level == "checkpoint":
            assumptions.append(
                "ASSUMPTION: checkpoint-level comparison: alpha bounds "
                "eval-sampling error only; NOT valid for independent "
                "retrains (measured 16% false-ship on ORBIT retrain null "
                "pairs)")
        else:
            assumptions.append(
                "ASSUMPTION: comparison_level='retrain' declared, but the "
                "episode-stream e-process bounds EVAL-sampling error only — "
                "the retraining lottery (sigma_run) is NOT priced by this "
                "stream (measured 16% false-ship on the ORBIT null corpus); "
                "use RetrainSequential for retrain-level sequential gating")
        ca = self._curse_assumption()
        if ca:
            assumptions.append(ca)
        self._decision_reasons(st, reasons, inv_alpha)
        if (verdict == "COLLECT-MORE" and st["delta_effective_pp"] is not None
                and abs(st["delta_effective_pp"]) < 2.0):
            caveats.append(CAVEAT_FLOOR_2PP)
        inputs = {"mode": "episodes", "stream_source": source,
                  "n_observations": st["n"], "n_paired": st["n"]}
        statistics = {
            "paired": True,
            "method": "e-process-episodes",
            "b": st["b"], "c": st["c"],
            "n_discordant": st["n_discordant"],
            "discordance_rate": st["p_d_hat"],
            "sr_incumbent_pp": st["sr_incumbent_pp"],
            "sr_candidate_pp": st["sr_candidate_pp"],
            "delta_hat_pp": st["delta_hat_pp"],
            "e_ship": st["e_ship"], "e_hold": st["e_hold"],
            "log_e_ship": st["log_e_ship"], "log_e_hold": st["log_e_hold"],
            "max_log_e_ship": st["max_log_e_ship"],
            "max_log_e_hold": st["max_log_e_hold"],
            "p_ship_av": st["p_ship_av"], "p_hold_av": st["p_hold_av"],
            "cs_p": st["cs_p"],
            "cs_delta_pp_approx": st["cs_delta_pp_approx"],
            "p_d_hat": st["p_d_hat"], "p_d_wilson": st["p_d_wilson"],
            "curse_correction_pp": st["curse_correction_pp"],
            "curse_method": self._curse_method,
            "delta_effective_pp": st["delta_effective_pp"],
            "n": st["n"],
            "decision": st["decision"],
        }
        return build_record(verdict, self.config, inputs, statistics,
                            caveats, warnings, assumptions, reasons, created)

    def _decision_reasons(self, st, reasons, inv_alpha):
        """Human reasons matching the frozen decision order."""
        if st["decision"] == "HOLD-EVIDENCE":
            reasons.append(
                "sequential evidence candidate WORSE: max_s E_hold %.6g >= "
                "1/alpha %.6g (mirror e-process, X -> 1-X; Ville)"
                % (_cap_exp(st["max_log_e_hold"]), inv_alpha))
        elif st["decision"] == "SHIP-EVIDENCE":
            reasons.append(
                "sequential evidence-to-ship: max_s E_ship %.6g >= 1/alpha "
                "%.6g (Ville: P(sup E >= 1/alpha) <= alpha under any "
                "p <= 1/2)" % (_cap_exp(st["max_log_e_ship"]), inv_alpha))
            reasons.append(
                "delta_effective %.2f pp >= min_effect %.2f pp%s"
                % (st["delta_effective_pp"], self.config.min_effect_pp,
                   " (delta_hat %.2f pp minus winner's-curse correction "
                   "%.2f pp)" % (st["delta_hat_pp"], self._curse)
                   if self._curse else ""))
        else:
            if (self._mx_ship >= self._log_thresh
                    and st["delta_effective_pp"] is not None):
                reasons.append(
                    "e_ship crossed 1/alpha but delta_effective %.2f pp < "
                    "min_effect %.2f pp%s — evidence of direction, not of a "
                    "shippable margin"
                    % (st["delta_effective_pp"], self.config.min_effect_pp,
                       " (delta_hat %.2f pp minus winner's-curse correction "
                       "%.2f pp)" % (st["delta_hat_pp"], self._curse)
                       if self._curse else ""))
            else:
                reasons.append(
                    "no e-process crossed 1/alpha %.6g: max E_ship %.4g, "
                    "max E_hold %.4g — CONTINUE (always-valid p_ship %.4g, "
                    "p_hold %.4g)"
                    % (inv_alpha, _cap_exp(st["max_log_e_ship"]),
                       _cap_exp(st["max_log_e_hold"]), st["p_ship_av"],
                       st["p_hold_av"]))


# ------------------------------------------------------------ retrain gate

class RetrainSequential(_SequentialBase):
    """Anytime-valid retrain-stream gate (frozen v1 spec, part B). Feed
    add_pair(delta_pp, se_eval_pp) per independent retrain pair; the
    one-sided half-normal mixture e-process (prior scale tau =
    config.sequential_tau, DISCLOSED) and its S -> -S mirror accumulate on
    the precision-weighted sufficient statistics. Requires
    comparison_level='retrain' (the class IS the retrain-level instrument;
    a 'checkpoint' config here would be a category error and raises).
    sigma_run resolution is exactly the fixed engine's
    (engine.resolve_sigma_run): explicit > regime > None — with None the
    stream still runs on se_eval alone, with the unpriced-lottery
    disclosure recorded (v0.1 no-prior semantics).

    state() keys mirror EpisodeSequential where meaningful: n (retrain
    pairs), b = c = None, e_ship, e_hold (capped), log_e_ship, log_e_hold,
    max_log_e_ship, max_log_e_hold, p_ship_av, p_hold_av, cs_p (None — the
    p-CS is an episode-stream object), cs_delta_pp (the EXACT two-sided
    Robbins CS, not an approximation — hence not named *_approx),
    delta_hat_pp (= S_t/V_t), curse_correction_pp, delta_effective_pp,
    sigma_run_pp_used, sigma_run_source, tau_pp, decision, mode."""

    def __init__(self, config):
        _SequentialBase.__init__(self, config)
        if config.comparison_level != "retrain":
            raise GateInputError(
                "RetrainSequential requires comparison_level='retrain' — "
                "the retrain stream prices the retraining lottery by "
                "construction; a 'checkpoint' declaration here would "
                "silently mislabel the estimand")
        self._sigma_run, self._sigma_src = resolve_sigma_run(config)
        self._sr = self._sigma_run if self._sigma_run is not None else 0.0
        tau = float(config.sequential_tau)
        self._tau = tau
        self._inv_tau2 = 1.0 / (tau * tau)
        self._S = 0.0
        self._V = 0.0
        self._t = 0
        # E_0 computed THROUGH the formula (not pinned): A=1/tau^2 makes
        # inv_tau2/A == 1.0 bit-exactly and log2 + log(Phi(0)) == 0.0, so
        # the spec's t=0 anchor E=1 holds exactly.
        self._le_ship = self._log_e(0.0)
        self._le_hold = self._le_ship

    def _log_e(self, s_signed):
        """log E for the direction whose signed sufficient stat is s_signed
        (+S ship, -S mirror): log 2 + 0.5*log((1/tau^2)/A) + s^2/(2A)
        + log Phi(s/sqrt(A)), A = V + 1/tau^2 — algebraically identical to
        the spec's (2/(tau*sqrt(A))) * exp(S^2/(2A)) * Phi(S/sqrt(A)),
        arranged so t=0 evaluates to exactly 0.0 (E=1)."""
        A = self._V + self._inv_tau2
        return (_LOG2 + 0.5 * math.log(self._inv_tau2 / A)
                + s_signed * s_signed / (2.0 * A)
                + log_phi_cdf(s_signed / math.sqrt(A)))

    def add_pair(self, delta_pp, se_eval_pp):
        """Consume one independent retrain pair: measured delta (pp) and its
        eval-sampling SE (pp). Per-pair variance sigma_j^2 = 2*sigma_run^2
        + se_eval_j^2 (frozen spec; one independent retrain per side); a
        zero total variance is caller misuse (an exactly-noise-free pair
        does not exist) and raises, as do non-finite inputs (a NaN delta
        would silently poison S_t forever; an infinite se would be a
        zero-weight no-op that still counts as an observation). Returns
        None (call state())."""
        d = float(delta_pp)
        se = float(se_eval_pp)
        if not (math.isfinite(d) and math.isfinite(se)):
            raise GateInputError(
                "delta_pp and se_eval_pp must be finite (got delta_pp %r, "
                "se_eval_pp %r): a non-finite observation would corrupt or "
                "zero-weight the e-process while still counting toward n"
                % (d, se))
        var = 2.0 * self._sr * self._sr + se * se
        if not var > 0.0:
            raise GateInputError(
                "per-pair variance 2*sigma_run^2 + se_eval^2 must be > 0 "
                "(sigma_run %r, se_eval %r): a zero-variance retrain pair "
                "would carry infinite weight" % (self._sigma_run, se))
        w = 1.0 / var
        self._S += d * w
        self._V += w
        self._t += 1
        ls = self._log_e(self._S)
        lh = self._log_e(-self._S)
        self._le_ship = ls
        self._le_hold = lh
        if ls > self._mx_ship:
            self._mx_ship = ls
        if lh > self._mx_hold:
            self._mx_hold = lh

    def _cs(self):
        """Two-sided Robbins normal-mixture confidence sequence for delta,
        valid simultaneously over all t at level 1-alpha (frozen spec
        formula); None before the first pair (V=0: the CS is the line)."""
        V = self._V
        if V <= 0.0:
            return None
        centre = self._S / V
        rad = (1.0 / V) * math.sqrt(
            (V + self._inv_tau2)
            * (math.log1p(self._tau * self._tau * V)
               + 2.0 * self._log_thresh))
        return [centre - rad, centre + rad]

    def state(self):
        """Current machine-readable state (keys documented on the class)."""
        V = self._V
        if V > 0.0:
            delta_hat = self._S / V
            delta_eff = delta_hat - self._curse
        else:
            delta_hat = delta_eff = None
        return {
            "mode": "retrains",
            "n": self._t, "b": None, "c": None,
            "e_ship": _cap_exp(self._le_ship),
            "e_hold": _cap_exp(self._le_hold),
            "log_e_ship": self._le_ship, "log_e_hold": self._le_hold,
            "max_log_e_ship": self._mx_ship,
            "max_log_e_hold": self._mx_hold,
            "p_ship_av": _av_p(self._mx_ship),
            "p_hold_av": _av_p(self._mx_hold),
            "cs_p": None,
            "cs_delta_pp": self._cs(),
            "delta_hat_pp": delta_hat,
            "curse_correction_pp": self._curse,
            "delta_effective_pp": delta_eff,
            "sigma_run_pp_used": self._sigma_run,
            "sigma_run_source": self._sigma_src,
            "tau_pp": self._tau,
            "decision": self._decision(delta_eff),
        }

    def record(self, now=None, source=None, extra_assumptions=None):
        """Signed schema-v2 decision record for the current state, method
        "e-process-retrains". Same conventions as EpisodeSequential.record."""
        st = self.state()
        created = _stamp(now)
        verdict = DECISION_TO_VERDICT[st["decision"]]
        inv_alpha = 1.0 / self._alpha
        reasons, caveats, warnings, assumptions = [], [], [], []
        if extra_assumptions:
            assumptions.extend(str(a) for a in extra_assumptions)
        assumptions.append(
            "ASSUMPTION: anytime-valid retrain-stream gate (method "
            "'e-process-retrains'): one-sided half-normal mixture e-process "
            "(Robbins normal-mixture supermartingale; Ramdas et al. safe "
            "anytime-valid inference) on per-retrain-pair deltas with known "
            "variance 2*sigma_run^2 + se_eval^2, prior scale tau = %.4g pp "
            "(config.sequential_tau, DISCLOSED); Ville bounds the "
            "false-crossing rate at alpha at ANY stopping time, and the "
            "delta CS is valid simultaneously over all t" % self._tau)
        if self._sigma_run is None:
            assumptions.append(
                "ASSUMPTION: no sigma_run prior resolved (explicit and "
                "regime both unset) — per-pair variance uses se_eval only, "
                "so the retraining lottery is UNPRICED (measured 16% "
                "false-ship on the ORBIT null corpus at eval-only alpha); "
                "deltas here are between independent retrains, interpret "
                "accordingly")
        else:
            assumptions.append(
                "ASSUMPTION: retrain-level stream: per-pair variance prices "
                "one independent retrain per side, sigma_run %.2f pp (%s)"
                % (self._sigma_run, self._sigma_src))
        ca = self._curse_assumption()
        if ca:
            assumptions.append(ca)
        self._decision_reasons(st, reasons, inv_alpha)
        if (verdict == "COLLECT-MORE" and st["delta_effective_pp"] is not None
                and abs(st["delta_effective_pp"]) < 2.0):
            caveats.append(CAVEAT_FLOOR_2PP)
        inputs = {"mode": "retrains", "stream_source": source,
                  "n_retrain_pairs": st["n"]}
        statistics = {
            "paired": None,
            "method": "e-process-retrains",
            "b": None, "c": None,
            "n_retrain_pairs": st["n"],
            "delta_hat_pp": st["delta_hat_pp"],
            "e_ship": st["e_ship"], "e_hold": st["e_hold"],
            "log_e_ship": st["log_e_ship"], "log_e_hold": st["log_e_hold"],
            "max_log_e_ship": st["max_log_e_ship"],
            "max_log_e_hold": st["max_log_e_hold"],
            "p_ship_av": st["p_ship_av"], "p_hold_av": st["p_hold_av"],
            "cs_delta_pp": st["cs_delta_pp"],
            "sigma_run_pp_used": st["sigma_run_pp_used"],
            "sigma_run_source": st["sigma_run_source"],
            "tau_pp": st["tau_pp"],
            "curse_correction_pp": st["curse_correction_pp"],
            "curse_method": self._curse_method,
            "delta_effective_pp": st["delta_effective_pp"],
            "decision": st["decision"],
        }
        return build_record(verdict, self.config, inputs, statistics,
                            caveats, warnings, assumptions, reasons, created)

    def _decision_reasons(self, st, reasons, inv_alpha):
        """Human reasons matching the frozen decision order."""
        if st["decision"] == "HOLD-EVIDENCE":
            reasons.append(
                "sequential evidence candidate WORSE: max_s E_hold %.6g >= "
                "1/alpha %.6g (mirror e-process, S -> -S; Ville)"
                % (_cap_exp(st["max_log_e_hold"]), inv_alpha))
        elif st["decision"] == "SHIP-EVIDENCE":
            reasons.append(
                "sequential evidence-to-ship: max_s E_ship %.6g >= 1/alpha "
                "%.6g (Ville: P(sup E >= 1/alpha) <= alpha under any "
                "delta <= 0)" % (_cap_exp(st["max_log_e_ship"]), inv_alpha))
            reasons.append(
                "delta_effective %.2f pp >= min_effect %.2f pp%s"
                % (st["delta_effective_pp"], self.config.min_effect_pp,
                   " (delta_hat %.2f pp minus winner's-curse correction "
                   "%.2f pp)" % (st["delta_hat_pp"], self._curse)
                   if self._curse else ""))
        else:
            if (self._mx_ship >= self._log_thresh
                    and st["delta_effective_pp"] is not None):
                reasons.append(
                    "e_ship crossed 1/alpha but delta_effective %.2f pp < "
                    "min_effect %.2f pp%s — evidence of direction, not of "
                    "a shippable margin"
                    % (st["delta_effective_pp"], self.config.min_effect_pp,
                       " (delta_hat %.2f pp minus winner's-curse correction "
                       "%.2f pp)" % (st["delta_hat_pp"], self._curse)
                       if self._curse else ""))
            else:
                cs = st["cs_delta_pp"]
                reasons.append(
                    "no e-process crossed 1/alpha %.6g: max E_ship %.4g, "
                    "max E_hold %.4g — CONTINUE (always-valid p_ship %.4g, "
                    "p_hold %.4g%s)"
                    % (inv_alpha, _cap_exp(st["max_log_e_ship"]),
                       _cap_exp(st["max_log_e_hold"]), st["p_ship_av"],
                       st["p_hold_av"],
                       "; delta CS [%.2f, %.2f] pp" % (cs[0], cs[1])
                       if cs else ""))
