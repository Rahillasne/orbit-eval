"""Ship-Gate v1 sequential layer (gate.sequential): frozen-spec e-processes.

Anchors the spec REQUIRES asserting: E_0 == 1 exactly for both processes;
e-process monotone under repeated identical wins; always-valid p monotone
nonincreasing in evidence. Exactness checks recompute every formula
independently in the test (mixture sums, Beta-mixture CS, half-normal
mixture E, Robbins CS) and compare against the class. Calibration is
TEST-VALIDATED by seeded, deterministic simulation (marked 'slow', still in
the default suite): H0 ever-cross rates within alpha + 3*MC-sigma at
>= 2000 sims x 2000 observations for both streams, > 80% sequential power
within 500 observations at the spec's alternatives (episodes p=0.65;
retrains delta=8, sigma_run=3.31, se=3), and CS coverage >= 1-alpha at
t in {10, 100, 1000} for both CS forms. No RNG in the library path — all
randomness lives in the tests, seeded."""

import math
import os
import random
import tempfile
import unittest

import pytest

from orbit_eval import stats
from orbit_eval.gate.engine import GateInputError
from orbit_eval.gate.records import (
    GateConfig, append_record, canonical_json, record_hash, verify_ledger)
from orbit_eval.gate.sequential import (
    DECISION_TO_VERDICT, LAMBDA_GRID, P0_GRID, EpisodeSequential,
    RetrainSequential, episode_p0_log_e, log_phi_cdf)

NOW = "2026-08-08T12:00:00Z"
ALPHA = 0.05
LOG_THRESH = -math.log(ALPHA)   # log(1/alpha)


def _rcfg(**kw):
    """Retrain-stream config: comparison_level='retrain' + overrides."""
    kw.setdefault("comparison_level", "retrain")
    return GateConfig(**kw)


def _episode_mix_e(xs):
    """Independent recomputation of the betting-mixture e-value for a full
    X sequence (ship direction): (1/9) * sum_lambda prod(1 + lam*(2X-1))."""
    total = 0.0
    for lam in LAMBDA_GRID:
        prod = 1.0
        for x in xs:
            prod *= 1.0 + lam * (2 * x - 1)
        total += prod
    return total / len(LAMBDA_GRID)


def _retrain_e(pairs, sigma_run, tau):
    """Independent recomputation of the half-normal mixture e-value (ship)
    from (delta, se) pairs: the spec formula, computed directly."""
    S = V = 0.0
    for d, se in pairs:
        var = 2.0 * sigma_run * sigma_run + se * se
        S += d / var
        V += 1.0 / var
    A = V + 1.0 / tau ** 2
    return (2.0 / (tau * math.sqrt(A))) * math.exp(S * S / (2.0 * A)) \
        * stats.Phi(S / math.sqrt(A))


# ------------------------------------------------------------------ config

class TestConfigSequentialTau(unittest.TestCase):
    def test_default_is_5(self):
        self.assertEqual(GateConfig().sequential_tau, 5.0)
        self.assertEqual(GateConfig().validate(), [])

    def test_validate_bounds(self):
        for bad in (0.0, -1.0, True, "5", None):
            cfg = GateConfig(sequential_tau=bad)
            self.assertTrue(any("sequential_tau" in m
                                for m in cfg.validate()),
                            "sequential_tau=%r not flagged" % (bad,))

    def test_sha256_sees_tau(self):
        self.assertNotEqual(GateConfig().sha256(),
                            GateConfig(sequential_tau=3.0).sha256())

    def test_invalid_config_raises_in_both_classes(self):
        with self.assertRaises(GateInputError):
            EpisodeSequential(GateConfig(alpha=0.0))
        with self.assertRaises(GateInputError):
            RetrainSequential(_rcfg(sequential_tau=-1.0))


# --------------------------------------------------------- episode anchors

class TestEpisodeAnchors(unittest.TestCase):
    def test_e0_exactly_one_full_cs_continue(self):
        st = EpisodeSequential(GateConfig()).state()
        self.assertEqual(st["e_ship"], 1.0)     # spec anchor: EXACT
        self.assertEqual(st["e_hold"], 1.0)
        self.assertEqual(st["log_e_ship"], 0.0)
        self.assertEqual(st["p_ship_av"], 1.0)
        self.assertEqual(st["p_hold_av"], 1.0)
        self.assertEqual(st["cs_p"], [0.001, 0.999])   # nothing excluded yet
        self.assertEqual((st["n"], st["b"], st["c"]), (0, 0, 0))
        self.assertIsNone(st["delta_hat_pp"])
        self.assertEqual(st["decision"], "CONTINUE")

    def test_hand_computed_e_sequence(self):
        # X sequence 1, 0, 1 (candidate-only, incumbent-only, candidate-only)
        g = EpisodeSequential(GateConfig())
        g.update(0, 1)
        e1 = _episode_mix_e([1])                 # = 1 + mean(lambda) = 1.5
        st = g.state()
        self.assertAlmostEqual(st["e_ship"], e1, places=12)
        self.assertAlmostEqual(st["e_ship"], 1.5, places=12)
        self.assertAlmostEqual(st["e_hold"], 0.5, places=12)  # 1 - mean(lam)
        g.update(1, 0)
        e2 = _episode_mix_e([1, 0])              # = 1 - mean(lambda^2)
        self.assertAlmostEqual(g.state()["e_ship"], e2, places=12)
        g.update(0, 1)
        e3 = _episode_mix_e([1, 0, 1])
        st = g.state()
        self.assertAlmostEqual(st["e_ship"], e3, places=12)
        # always-valid p = 1 / running max E (max includes E_0 = 1)
        self.assertAlmostEqual(st["p_ship_av"],
                               1.0 / max(1.0, e1, e2, e3), places=12)
        self.assertEqual((st["n"], st["b"], st["c"]), (3, 1, 2))

    def test_monotone_under_repeated_wins(self):
        # spec anchor: e-process monotone under repeated identical wins
        g = EpisodeSequential(GateConfig())
        prev_ship, prev_hold, prev_p = 0.0, 0.0, 1.0
        for _ in range(60):
            g.update(0, 1)
            st = g.state()
            self.assertGreater(st["log_e_ship"], prev_ship)   # strict up
            self.assertLess(st["log_e_hold"], prev_hold)      # strict down
            self.assertLessEqual(st["p_ship_av"], prev_p)     # nonincreasing
            prev_ship, prev_hold = st["log_e_ship"], st["log_e_hold"]
            prev_p = st["p_ship_av"]
        self.assertLessEqual(st["p_ship_av"], ALPHA)
        self.assertEqual(st["decision"], "SHIP-EVIDENCE")

    def test_always_valid_p_monotone_on_mixed_stream(self):
        # spec anchor: always-valid p monotone nonincreasing in evidence,
        # even when the raw e-process itself goes up AND down
        g = EpisodeSequential(GateConfig())
        rng = random.Random(3)
        prev_ps, prev_ph = 1.0, 1.0
        for _ in range(300):
            x = rng.random() < 0.5
            g.update(0, 1) if x else g.update(1, 0)
            st = g.state()
            self.assertLessEqual(st["p_ship_av"], prev_ps)
            self.assertLessEqual(st["p_hold_av"], prev_ph)
            prev_ps, prev_ph = st["p_ship_av"], st["p_hold_av"]

    def test_concordant_pairs_leave_e_untouched(self):
        g = EpisodeSequential(GateConfig())
        g.update(1, 1)
        g.update(0, 0)
        st = g.state()
        self.assertEqual(st["e_ship"], 1.0)
        self.assertEqual(st["e_hold"], 1.0)
        self.assertEqual((st["n"], st["b"], st["c"]), (2, 0, 0))
        self.assertEqual(st["p_d_hat"], 0.0)
        self.assertEqual(st["sr_incumbent_pp"], 50.0)
        self.assertEqual(st["sr_candidate_pp"], 50.0)

    def test_determinism_identical_streams_identical_records(self):
        # no RNG anywhere in the sequential path (frozen spec C)
        seq = [(0, 1)] * 7 + [(1, 0)] * 3 + [(1, 1)] * 5 + [(0, 0)] * 5
        g1, g2 = (EpisodeSequential(GateConfig()) for _ in range(2))
        for inc, cand in seq:
            g1.update(inc, cand)
            g2.update(inc, cand)
        self.assertEqual(g1.state(), g2.state())
        r1, r2 = g1.record(now=NOW), g2.record(now=NOW)
        self.assertEqual(canonical_json(r1), canonical_json(r2))
        self.assertEqual(r1["record_sha256"], r2["record_sha256"])


# -------------------------------------------------------------- episode CS

class TestEpisodeCS(unittest.TestCase):
    def test_grid_matches_direct_running_max(self):
        # replay the same discordant path and recompute the running max per
        # grid point straight from the spec formula
        rng = random.Random(17)
        xs = [1 if rng.random() < 0.7 else 0 for _ in range(40)]
        g = EpisodeSequential(GateConfig())
        direct_max = [0.0] * len(P0_GRID)
        s = 0
        for t, x in enumerate(xs, 1):
            s += x
            g.update(0, 1) if x else g.update(1, 0)
            for i, p0 in enumerate(P0_GRID):
                v = episode_p0_log_e(t, s, p0)
                if v > direct_max[i]:
                    direct_max[i] = v
        inside = [p0 for p0, m in zip(P0_GRID, direct_max) if m < LOG_THRESH]
        self.assertEqual(g.state()["cs_p"], [min(inside), max(inside)])

    def test_all_wins_cs_excludes_low_p_hand_check(self):
        # 200 straight candidate wins: S_t = t, so log E(p0) =
        # -log(201) - 200*ln(p0); the CS keeps p0 > exp(-(log20+log201)/200)
        g = EpisodeSequential(GateConfig())
        for _ in range(200):
            g.update(0, 1)
        cut = math.exp(-(math.log(1 / ALPHA) + math.log(201)) / 200)
        lo, hi = g.state()["cs_p"]
        self.assertEqual(hi, 0.999)
        self.assertAlmostEqual(lo, math.ceil(cut * 1000) / 1000.0, places=12)
        self.assertGreater(lo, 0.9)

    def test_delta_mapping_and_wilson_band(self):
        g = EpisodeSequential(GateConfig())
        for _ in range(30):
            g.update(0, 1)
        for _ in range(10):
            g.update(1, 0)
        for _ in range(60):
            g.update(1, 1)
        st = g.state()
        n, b, c = st["n"], st["b"], st["c"]
        self.assertEqual((n, b, c), (100, 10, 30))
        p_d = (b + c) / n
        self.assertEqual(st["p_d_hat"], p_d)
        lo, hi = st["cs_p"]
        self.assertEqual(st["cs_delta_pp_approx"],
                         [100.0 * p_d * (2 * lo - 1),
                          100.0 * p_d * (2 * hi - 1)])
        self.assertEqual(st["p_d_wilson"], list(stats.wilson(b + c, n)))

    def test_track_cs_false_reports_none_e_unaffected(self):
        g1 = EpisodeSequential(GateConfig(), track_cs=True)
        g2 = EpisodeSequential(GateConfig(), track_cs=False)
        for _ in range(25):
            g1.update(0, 1)
            g2.update(0, 1)
        s1, s2 = g1.state(), g2.state()
        self.assertIsNone(s2["cs_p"])
        self.assertIsNone(s2["cs_delta_pp_approx"])
        self.assertEqual(s1["log_e_ship"], s2["log_e_ship"])
        self.assertEqual(s1["p_ship_av"], s2["p_ship_av"])
        self.assertEqual(s1["decision"], s2["decision"])


# -------------------------------------------------------- episode decision

class TestEpisodeDecision(unittest.TestCase):
    def test_ship_requires_both_crossing_and_min_effect(self):
        # 180 wins / 60 losses: delta_hat = 50 pp, e_ship crossed long ago.
        # min_effect above delta -> CONTINUE with the direction-only reason;
        # min_effect below -> SHIP-EVIDENCE.
        stream = [(0, 1)] * 180 + [(1, 0)] * 60
        g = EpisodeSequential(GateConfig(min_effect_pp=60.0), track_cs=False)
        for inc, cand in stream:
            g.update(inc, cand)
        st = g.state()
        self.assertGreaterEqual(g.max_log_e_ship, LOG_THRESH)
        self.assertEqual(st["delta_hat_pp"], 50.0)
        self.assertEqual(st["decision"], "CONTINUE")
        rec = g.record(now=NOW)
        self.assertEqual(rec["verdict"], "COLLECT-MORE")
        self.assertTrue(any("min_effect" in r for r in rec["reasons"]))
        g2 = EpisodeSequential(GateConfig(min_effect_pp=10.0), track_cs=False)
        for inc, cand in stream:
            g2.update(inc, cand)
        self.assertEqual(g2.state()["decision"], "SHIP-EVIDENCE")
        self.assertEqual(g2.record(now=NOW)["verdict"], "SHIP")

    def test_hold_evidence_on_worse_candidate(self):
        g = EpisodeSequential(GateConfig(), track_cs=False)
        for _ in range(200):
            g.update(1, 0)
        st = g.state()
        self.assertEqual(st["decision"], "HOLD-EVIDENCE")
        self.assertLessEqual(st["p_hold_av"], ALPHA)
        self.assertEqual(g.record(now=NOW)["verdict"], "HOLD")

    def test_hold_takes_precedence_once_both_crossed(self):
        # ship evidence first (100 wins), then the candidate collapses (400
        # losses): both sup-crossings are latched, worse direction wins the
        # frozen decision order
        g = EpisodeSequential(GateConfig(), track_cs=False)
        for _ in range(100):
            g.update(0, 1)
        self.assertGreaterEqual(g.max_log_e_ship, LOG_THRESH)
        for _ in range(400):
            g.update(1, 0)
        self.assertGreaterEqual(g.max_log_e_hold, LOG_THRESH)
        self.assertEqual(g.state()["decision"], "HOLD-EVIDENCE")

    def test_curse_prior_applied_to_ship_leg_only(self):
        g = EpisodeSequential(
            GateConfig(selection="max-over-checkpoints", n_checkpoints=4),
            track_cs=False)
        for _ in range(50):
            g.update(0, 1)
        st = g.state()
        self.assertEqual(st["curse_correction_pp"], 4.0)  # J=4 prior exactly
        self.assertEqual(st["delta_effective_pp"], 100.0 - 4.0)
        rec = g.record(now=NOW)
        self.assertEqual(rec["statistics"]["curse_method"],
                         "atlas-prior-scaled")
        self.assertTrue(any("PI05_CELL1_RESULTS" in a
                            for a in rec["assumptions"]))

    def test_max_over_checkpoints_without_j_raises(self):
        with self.assertRaises(GateInputError):
            EpisodeSequential(GateConfig(selection="max-over-checkpoints"))
        with self.assertRaises(GateInputError):
            RetrainSequential(_rcfg(selection="max-over-checkpoints"))


# ---------------------------------------------------------- episode record

class TestEpisodeRecord(unittest.TestCase):
    def _gate(self):
        g = EpisodeSequential(GateConfig())
        for _ in range(30):
            g.update(0, 1)
        for _ in range(5):
            g.update(1, 0)
        for _ in range(65):
            g.update(1, 1)
        return g

    def test_schema_v2_shape_method_and_hash(self):
        rec = self._gate().record(now=NOW, source="stream.jsonl")
        self.assertEqual(set(rec), {
            "schema_version", "engine_version", "created", "verdict",
            "reasons", "gate_config", "gate_config_sha256", "inputs",
            "statistics", "caveats", "warnings", "assumptions",
            "record_sha256"})
        self.assertEqual(rec["schema_version"], 2)
        self.assertEqual(rec["statistics"]["method"], "e-process-episodes")
        self.assertEqual(rec["statistics"]["decision"], "SHIP-EVIDENCE")
        self.assertEqual(rec["verdict"],
                         DECISION_TO_VERDICT["SHIP-EVIDENCE"])
        self.assertEqual(rec["gate_config"]["sequential_tau"], 5.0)
        self.assertEqual(rec["inputs"]["stream_source"], "stream.jsonl")
        self.assertEqual(rec["record_sha256"], record_hash(rec))

    def test_ledger_append_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.jsonl")
            append_record(self._gate().record(now=NOW), path)
            res = verify_ledger(path)
            self.assertTrue(res["ok"])
            self.assertEqual(res["n_records"], 1)

    def test_disclosures_present(self):
        rec = self._gate().record(now=NOW)
        joined = " ".join(rec["assumptions"])
        self.assertIn("exact joint CS", joined)          # delta mapping v2
        self.assertIn("Ville", joined)                   # anytime validity
        self.assertIn("checkpoint-level comparison", joined)

    def test_retrain_level_config_discloses_unpriced_lottery(self):
        g = EpisodeSequential(GateConfig(comparison_level="retrain",
                                         sigma_run_pp=3.31))
        g.update(0, 1)
        joined = " ".join(g.record(now=NOW)["assumptions"])
        self.assertIn("NOT priced", joined)
        self.assertIn("RetrainSequential", joined)

    def test_json_safe_at_extreme_evidence(self):
        # 1200 straight wins push log E past the exp() overflow point; the
        # record must still be canonically hashable (no inf/nan)
        g = EpisodeSequential(GateConfig(), track_cs=False)
        for _ in range(1200):
            g.update(0, 1)
        rec = g.record(now=NOW)
        canonical_json(rec)     # allow_nan=False: raises on inf/nan
        st = rec["statistics"]
        self.assertTrue(math.isfinite(st["e_ship"]))
        self.assertGreater(st["log_e_ship"], 700.0)   # exact log kept
        self.assertEqual(st["p_ship_av"], 0.0)

    def test_extra_assumptions_recorded_verbatim(self):
        g = EpisodeSequential(GateConfig())
        rec = g.record(now=NOW, extra_assumptions=["ASSUMPTION: from-cli"])
        self.assertIn("ASSUMPTION: from-cli", rec["assumptions"])


# --------------------------------------------------------- retrain anchors

class TestRetrainAnchors(unittest.TestCase):
    def test_e0_exactly_one(self):
        # spec anchor, MUST be exact: t=0 -> A=1/tau^2 -> E = 1
        for tau in (5.0, 1.0, 3.31, 0.7):
            st = RetrainSequential(_rcfg(sequential_tau=tau)).state()
            self.assertEqual(st["e_ship"], 1.0)
            self.assertEqual(st["e_hold"], 1.0)
            self.assertEqual(st["log_e_ship"], 0.0)
            self.assertEqual(st["p_ship_av"], 1.0)
            self.assertIsNone(st["cs_delta_pp"])
            self.assertEqual(st["decision"], "CONTINUE")

    def test_hand_computed_single_pair(self):
        g = RetrainSequential(_rcfg(sigma_run_pp=3.31))
        g.add_pair(8.0, 3.0)
        st = g.state()
        e = _retrain_e([(8.0, 3.0)], 3.31, 5.0)
        self.assertAlmostEqual(st["e_ship"], e, delta=abs(e) * 1e-10)
        var = 2.0 * 3.31 ** 2 + 9.0
        self.assertAlmostEqual(st["delta_hat_pp"],
                               (8.0 / var) / (1.0 / var), places=10)

    def test_hand_computed_three_pairs_and_mirror(self):
        pairs = [(4.0, 3.0), (6.0, 2.0), (-1.0, 4.0)]
        g = RetrainSequential(_rcfg(sigma_run_pp=2.0, sequential_tau=5.0))
        for d, se in pairs:
            g.add_pair(d, se)
        st = g.state()
        e = _retrain_e(pairs, 2.0, 5.0)
        self.assertAlmostEqual(st["e_ship"], e, delta=abs(e) * 1e-10)
        # mirror: feeding the negated deltas swaps the two directions
        g2 = RetrainSequential(_rcfg(sigma_run_pp=2.0, sequential_tau=5.0))
        for d, se in pairs:
            g2.add_pair(-d, se)
        st2 = g2.state()
        self.assertAlmostEqual(st["e_ship"], st2["e_hold"], places=10)
        self.assertAlmostEqual(st["e_hold"], st2["e_ship"], places=10)

    def test_robbins_cs_hand_check(self):
        pairs = [(4.0, 3.0), (6.0, 2.0), (-1.0, 4.0)]
        tau = 5.0
        g = RetrainSequential(_rcfg(sigma_run_pp=2.0, sequential_tau=tau))
        S = V = 0.0
        for d, se in pairs:
            g.add_pair(d, se)
            var = 2.0 * 4.0 + se * se
            S += d / var
            V += 1.0 / var
        rad = (1.0 / V) * math.sqrt(
            (V + 1.0 / tau ** 2)
            * (math.log(1.0 + tau ** 2 * V) + 2.0 * math.log(1.0 / ALPHA)))
        lo, hi = g.state()["cs_delta_pp"]
        self.assertAlmostEqual(lo, S / V - rad, places=10)
        self.assertAlmostEqual(hi, S / V + rad, places=10)

    def test_always_valid_p_monotone(self):
        g = RetrainSequential(_rcfg(sigma_run_pp=3.31))
        rng = random.Random(9)
        prev_ps, prev_ph = 1.0, 1.0
        for _ in range(200):
            g.add_pair(rng.gauss(0.0, 5.0), 3.0)
            st = g.state()
            self.assertLessEqual(st["p_ship_av"], prev_ps)
            self.assertLessEqual(st["p_hold_av"], prev_ph)
            prev_ps, prev_ph = st["p_ship_av"], st["p_hold_av"]

    def test_nonnegative_and_monotone_under_repeated_gains(self):
        g = RetrainSequential(_rcfg(sigma_run_pp=3.31))
        prev = 0.0
        for _ in range(30):
            g.add_pair(8.0, 3.0)
            st = g.state()
            self.assertGreaterEqual(st["e_ship"], 0.0)
            self.assertGreater(st["log_e_ship"], prev)
            prev = st["log_e_ship"]
        self.assertEqual(st["decision"], "SHIP-EVIDENCE")

    def test_determinism(self):
        pairs = [(3.0, 2.0), (-1.5, 3.0), (7.0, 2.5)]
        g1, g2 = (RetrainSequential(_rcfg(regime="smolvla-ft"))
                  for _ in range(2))
        for d, se in pairs:
            g1.add_pair(d, se)
            g2.add_pair(d, se)
        self.assertEqual(g1.state(), g2.state())
        self.assertEqual(canonical_json(g1.record(now=NOW)),
                         canonical_json(g2.record(now=NOW)))

    def test_martingale_one_step_identity_by_quadrature(self):
        # spec sanity anchor: E is a nonnegative MARTINGALE under delta=0.
        # Deterministic verification (no MC noise): for any reached state
        # (S, V), integrating E(S + d/sigma^2, V + 1/sigma^2) against the
        # H0 law d ~ N(0, sigma^2) must give back E(S, V) exactly. A
        # sample-mean check is hopeless here — the mixture e-value is heavy-
        # tailed enough that 2000-sim means sit far below 1 — so the
        # identity is checked by trapezoid quadrature, in LOG space (the
        # e-factor alone overflows where the product with the density is
        # still tiny) over +/-30 sigma (the integrand's effective decay is
        # sqrt(1 - w/A) of the raw Gaussian's, so 12 sigma truncates ~1e-5
        # of mass when V0=0).
        tau = 5.0

        def log_e_sv(S, V):
            A = V + 1.0 / tau ** 2
            return (math.log(2.0 / (tau * math.sqrt(A)))
                    + S * S / (2.0 * A) + log_phi_cdf(S / math.sqrt(A)))

        for (S0, V0) in ((0.0, 0.0), (0.35, 0.6), (-0.2, 1.2)):
            for sigma in (5.56, 2.0):
                w = 1.0 / sigma ** 2
                n_pts, span = 12000, 30.0 * sigma
                h = 2.0 * span / n_pts
                log_norm = math.log(sigma * math.sqrt(2.0 * math.pi))
                total = 0.0
                for i in range(n_pts + 1):
                    d = -span + i * h
                    lv = (log_e_sv(S0 + d * w, V0 + w)
                          - 0.5 * (d / sigma) ** 2 - log_norm)
                    val = math.exp(lv) if lv > -700.0 else 0.0
                    total += val if 0 < i < n_pts else 0.5 * val
                integral = total * h
                expect = math.exp(log_e_sv(S0, V0))
                self.assertAlmostEqual(
                    integral / expect, 1.0, delta=1e-9,
                    msg="martingale identity broken at S=%.2f V=%.2f "
                        "sigma=%.2f" % (S0, V0, sigma))


# ------------------------------------------------- retrain config / priors

class TestRetrainSigmaResolution(unittest.TestCase):
    def test_explicit_beats_regime(self):
        g = RetrainSequential(_rcfg(sigma_run_pp=3.0, regime="smolvla-ft"))
        st = g.state()
        self.assertEqual(st["sigma_run_pp_used"], 3.0)
        self.assertEqual(st["sigma_run_source"], "explicit")

    def test_regime_resolves_atlas(self):
        g = RetrainSequential(_rcfg(regime="smolvla-ft"))
        st = g.state()
        self.assertEqual(st["sigma_run_pp_used"], 3.31)
        self.assertEqual(st["sigma_run_source"], "atlas:smolvla-ft")

    def test_unknown_regime_raises(self):
        with self.assertRaises(GateInputError):
            RetrainSequential(_rcfg(regime="no-such-regime"))

    def test_checkpoint_level_raises(self):
        with self.assertRaises(GateInputError):
            RetrainSequential(GateConfig())    # comparison_level=checkpoint

    def test_no_prior_runs_on_se_only_with_disclosure(self):
        g = RetrainSequential(_rcfg())
        g.add_pair(8.0, 3.0)
        st = g.state()
        self.assertIsNone(st["sigma_run_pp_used"])
        # variance is se^2 alone: cross-check against the direct formula
        e = _retrain_e([(8.0, 3.0)], 0.0, 5.0)
        self.assertAlmostEqual(st["e_ship"], e, delta=abs(e) * 1e-10)
        joined = " ".join(g.record(now=NOW)["assumptions"])
        self.assertIn("UNPRICED", joined)

    def test_zero_variance_pair_raises(self):
        g = RetrainSequential(_rcfg())
        with self.assertRaises(GateInputError):
            g.add_pair(1.0, 0.0)

    def test_non_finite_pair_raises_and_leaves_state_untouched(self):
        # a NaN delta would poison S_t forever; an infinite se would be a
        # zero-weight no-op that still counts toward n — both are rejected
        # (defense in depth behind the CLI stream validator's finite check)
        g = RetrainSequential(_rcfg(sigma_run_pp=3.31))
        g.add_pair(8.0, 3.0)
        before = g.state()
        for d, se in ((float("nan"), 3.0), (8.0, float("nan")),
                      (float("inf"), 3.0), (8.0, float("inf")),
                      (float("-inf"), 3.0)):
            with self.assertRaises(GateInputError):
                g.add_pair(d, se)
        self.assertEqual(g.state(), before)


# ---------------------------------------------------------- retrain record

class TestRetrainRecord(unittest.TestCase):
    def test_record_schema_method_ledger(self):
        g = RetrainSequential(_rcfg(sigma_run_pp=3.31))
        for _ in range(10):
            g.add_pair(8.0, 3.0)
        rec = g.record(now=NOW, source="pairs.jsonl")
        self.assertEqual(rec["statistics"]["method"], "e-process-retrains")
        self.assertEqual(rec["verdict"], "SHIP")
        self.assertEqual(rec["statistics"]["tau_pp"], 5.0)
        self.assertEqual(rec["inputs"]["n_retrain_pairs"], 10)
        self.assertEqual(rec["record_sha256"], record_hash(rec))
        joined = " ".join(rec["assumptions"])
        self.assertIn("sequential_tau", joined)
        self.assertIn("Robbins", joined)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.jsonl")
            append_record(rec, path)
            self.assertTrue(verify_ledger(path)["ok"])

    def test_hold_record_on_worse(self):
        g = RetrainSequential(_rcfg(sigma_run_pp=3.31))
        for _ in range(10):
            g.add_pair(-8.0, 3.0)
        rec = g.record(now=NOW)
        self.assertEqual(rec["verdict"], "HOLD")
        self.assertEqual(rec["statistics"]["decision"], "HOLD-EVIDENCE")

    def test_min_effect_floor_blocks_ship_with_curse(self):
        # true-ish delta 3 pp, curse J=4 removes 4 pp -> below min_effect
        g = RetrainSequential(_rcfg(sigma_run_pp=1.0,
                                    selection="max-over-checkpoints",
                                    n_checkpoints=4))
        for _ in range(200):
            g.add_pair(3.0, 1.0)
        st = g.state()
        self.assertGreaterEqual(g.max_log_e_ship, LOG_THRESH)
        self.assertAlmostEqual(st["delta_effective_pp"], -1.0, places=9)
        self.assertEqual(st["decision"], "CONTINUE")
        rec = g.record(now=NOW)
        self.assertEqual(rec["verdict"], "COLLECT-MORE")
        self.assertTrue(any("shippable margin" in r for r in rec["reasons"]))
        self.assertTrue(any("coin flip" in c for c in rec["caveats"]))


# ---------------------------------------------------------- log Phi helper

class TestLogPhiCdf(unittest.TestCase):
    def test_matches_direct_in_bulk(self):
        for x in (-5.5, -2.0, -1.0, 0.0, 1.0, 3.0):
            self.assertAlmostEqual(log_phi_cdf(x), math.log(stats.Phi(x)),
                                   places=10)

    def test_zero_is_log_half(self):
        self.assertEqual(log_phi_cdf(0.0), math.log(0.5))

    def test_deep_tail_asymptotic(self):
        # log Phi(-20) = -200 - log(20) - 0.5*log(2*pi) + log1p(-...)
        self.assertAlmostEqual(log_phi_cdf(-20.0), -203.9170, delta=0.01)
        # continuity across the branch switch at -6
        self.assertAlmostEqual(log_phi_cdf(-6.0 + 1e-9),
                               log_phi_cdf(-6.0 - 1e-9), delta=1e-3)
        # the exact boundary x = -6.0 takes the DIRECT branch (~1e-7-
        # accurate there), not the Mills asymptotic (~5e-5 at its worst)
        self.assertEqual(log_phi_cdf(-6.0), math.log(stats.Phi(-6.0)))
        # strictly increasing through the tail
        xs = [-30.0, -12.0, -6.5, -6.0, -5.0, -1.0, 0.0, 2.0]
        vals = [log_phi_cdf(x) for x in xs]
        self.assertEqual(vals, sorted(vals))


# ------------------------------------------------- calibration simulations
# Seeded and deterministic; 'slow' marks the heavy ones but they stay in
# the default suite (frozen spec E).

@pytest.mark.slow
class TestCalibrationEpisodesH0(unittest.TestCase):
    def test_h0_ever_cross_rate_within_bound(self):
        # p = 1/2 exactly (the boundary of H0_ship, where the per-lambda
        # products are exact martingales): the ship e-process must cross
        # 1/alpha within 2000 discordant steps in <= alpha + 3*MC-sigma of
        # 2000 seeded sims (Ville guarantees <= alpha even over infinite
        # horizons; finite horizon is strictly easier)
        n_sims, n_steps = 2000, 2000
        rng = random.Random(20260808)
        cfg = GateConfig()
        crossed = 0
        for _ in range(n_sims):
            g = EpisodeSequential(cfg, track_cs=False)
            for _ in range(n_steps):
                if rng.random() < 0.5:
                    g.update(0, 1)
                else:
                    g.update(1, 0)
                if g.max_log_e_ship >= LOG_THRESH:
                    crossed += 1
                    break
        rate = crossed / n_sims
        bound = ALPHA + 3.0 * math.sqrt(ALPHA * (1 - ALPHA) / n_sims)
        self.assertLessEqual(rate, bound,
                             "H0 episode e-process crossed at %.4f > %.4f"
                             % (rate, bound))


@pytest.mark.slow
class TestCalibrationRetrainsH0(unittest.TestCase):
    def test_h0_ever_cross_rate_within_bound(self):
        # delta = 0 with a MIX of per-pair sigmas (se cycles 2/3/5 pp on
        # sigma_run = 3.31): ever-cross rate over 2000 pairs bounded as in
        # the episode sim (the martingale property itself is verified
        # deterministically by quadrature in TestRetrainAnchors)
        n_sims, n_steps = 2000, 2000
        cfg = _rcfg(sigma_run_pp=3.31)
        ses = (2.0, 3.0, 5.0)
        sds = tuple(math.sqrt(2.0 * 3.31 ** 2 + s * s) for s in ses)
        rng = random.Random(910)
        crossed = 0
        for _ in range(n_sims):
            g = RetrainSequential(cfg)
            for j in range(n_steps):
                k = j % 3
                g.add_pair(rng.gauss(0.0, sds[k]), ses[k])
                if g.max_log_e_ship >= LOG_THRESH:
                    crossed += 1
                    break
        rate = crossed / n_sims
        bound = ALPHA + 3.0 * math.sqrt(ALPHA * (1 - ALPHA) / n_sims)
        self.assertLessEqual(rate, bound,
                             "H0 retrain e-process crossed at %.4f > %.4f"
                             % (rate, bound))


@pytest.mark.slow
class TestSequentialPower(unittest.TestCase):
    def test_episodes_power_at_p065(self):
        # spec E: p=0.65 must stop with SHIP-evidence in > 80% of sims
        # within 500 discordant observations
        rng = random.Random(4242)
        n_sims, horizon, stops = 300, 500, 0
        cfg = GateConfig()
        for _ in range(n_sims):
            g = EpisodeSequential(cfg, track_cs=False)
            for _ in range(horizon):
                if rng.random() < 0.65:
                    g.update(0, 1)
                else:
                    g.update(1, 0)
                if (g.max_log_e_ship >= LOG_THRESH
                        and g.state()["decision"] == "SHIP-EVIDENCE"):
                    stops += 1
                    break
        self.assertGreater(stops / n_sims, 0.8)

    def test_retrains_power_at_delta8(self):
        # spec E: delta=8, sigma_run=3.31, se=3 must stop with SHIP-evidence
        # in > 80% of sims within 500 observations
        rng = random.Random(777)
        n_sims, horizon, stops = 300, 500, 0
        cfg = _rcfg(sigma_run_pp=3.31)
        sd = math.sqrt(2.0 * 3.31 ** 2 + 9.0)
        for _ in range(n_sims):
            g = RetrainSequential(cfg)
            for _ in range(horizon):
                g.add_pair(rng.gauss(8.0, sd), 3.0)
                if (g.max_log_e_ship >= LOG_THRESH
                        and g.state()["decision"] == "SHIP-EVIDENCE"):
                    stops += 1
                    break
        self.assertGreater(stops / n_sims, 0.8)


@pytest.mark.slow
class TestCSCoverage(unittest.TestCase):
    CHECKPOINTS = (10, 100, 1000)

    def test_episodes_cs_coverage(self):
        # coverage of the Beta(1,1)-mixture CS at the TRUE p: p_true stays
        # inside CS_t iff max_{s<=t} E_s(p_true) < 1/alpha — evaluated
        # directly from the spec formula along the path (the class grid is
        # the same formula on 0.001 steps; grid-vs-direct equality is
        # asserted exactly in TestEpisodeCS)
        for p_true, seed in ((0.35, 5), (0.65, 6)):
            n_sims = 500
            cov = dict.fromkeys(self.CHECKPOINTS, 0)
            rng = random.Random(seed)
            for _ in range(n_sims):
                s, mx = 0, 0.0
                for t in range(1, 1001):
                    if rng.random() < p_true:
                        s += 1
                    v = episode_p0_log_e(t, s, p_true)
                    if v > mx:
                        mx = v
                    if t in cov and mx < LOG_THRESH:
                        cov[t] += 1
            for t in self.CHECKPOINTS:
                self.assertGreaterEqual(
                    cov[t] / n_sims, 1 - ALPHA,
                    "episode CS coverage %.3f < %.2f at t=%d, p=%.2f"
                    % (cov[t] / n_sims, 1 - ALPHA, t, p_true))

    def test_retrains_cs_coverage(self):
        # coverage of the Robbins normal-mixture CS at the true delta,
        # through the class surface, sigma mix as in the H0 sim
        delta_true = 4.0
        n_sims = 400
        cfg = _rcfg(sigma_run_pp=3.31)
        ses = (2.0, 3.0, 5.0)
        sds = tuple(math.sqrt(2.0 * 3.31 ** 2 + s * s) for s in ses)
        cov = dict.fromkeys(self.CHECKPOINTS, 0)
        rng = random.Random(1234)
        for _ in range(n_sims):
            g = RetrainSequential(cfg)
            for j in range(1000):
                k = j % 3
                g.add_pair(rng.gauss(delta_true, sds[k]), ses[k])
                t = j + 1
                if t in cov:
                    lo, hi = g.state()["cs_delta_pp"]
                    if lo <= delta_true <= hi:
                        cov[t] += 1
        for t in self.CHECKPOINTS:
            self.assertGreaterEqual(
                cov[t] / n_sims, 1 - ALPHA,
                "retrain CS coverage %.3f < %.2f at t=%d"
                % (cov[t] / n_sims, 1 - ALPHA, t))


if __name__ == "__main__":
    unittest.main()
