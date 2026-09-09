"""Ship-Gate engine tests on synthetic CRN data (shared-u construction from
test_stats): precheck ladder, all five verdicts, curse both methods, per-task
regression, sizing formulas, byte determinism."""

import math
import os
import random
import tempfile
import unittest

from orbit_eval import atlas, stats
from orbit_eval.gate import engine
from orbit_eval.gate.engine import (
    CAVEAT_FLOOR_2PP, GateDecision, GateInputError, curse_from_history,
    curse_from_prior, episodes_for_min_effect,
    episodes_for_min_effect_unpaired, gate, power_at_min_effect_paired,
    power_at_min_effect_unpaired)
from orbit_eval.gate.records import GateConfig, verify_ledger, append_record
from orbit_eval.io import EvalRun

NOW = "2026-08-08T12:00:00Z"


def _cfg(**kw):
    """Small n_boot keeps the seeded bootstrap fast; everything else is the
    frozen default unless a test overrides it."""
    kw.setdefault("n_boot", 200)
    return GateConfig(**kw)


def _mk(run_id, successes, seed=1000, groups=("pusht",), **kw):
    n = len(successes) if successes else None
    sr = kw.pop("sr", 100.0 * sum(map(bool, successes)) / n
                if successes else None)
    return EvalRun(run_id=run_id, fmt="lerobot", sr=sr, n_eval=n, seed=seed,
                   successes=successes, task_groups=list(groups or []), **kw)


def _crn_pair(n=400, p_inc=0.45, p_cand=0.60, seed=11, seed_inc=1000,
              seed_cand=1000, groups=("pusht",)):
    """Shared-u CRN construction: same u per episode, so the higher-p run's
    successes are a superset of the lower-p run's (b or c is exactly 0)."""
    rng = random.Random(seed)
    inc_s, cand_s = [], []
    for _ in range(n):
        u = rng.random()
        inc_s.append(u < p_inc)
        cand_s.append(u < p_cand)
    return (_mk("inc", inc_s, seed=seed_inc, groups=groups),
            _mk("cand", cand_s, seed=seed_cand, groups=groups))


def _bc_pair(n, b, c, concordant=0, **mkkw):
    """Explicit (b, c) construction: episodes 0..b-1 incumbent-only wins,
    b..b+c-1 candidate-only wins, then `concordant` shared wins, rest shared
    losses."""
    inc_s = [True] * b + [False] * c + [True] * concordant \
        + [False] * (n - b - c - concordant)
    cand_s = [False] * b + [True] * c + [True] * concordant \
        + [False] * (n - b - c - concordant)
    return _mk("inc", inc_s, **mkkw), _mk("cand", cand_s, **mkkw)


class TestPrechecks(unittest.TestCase):
    def test_invalid_config_raises(self):
        inc, cand = _crn_pair()
        with self.assertRaises(GateInputError):
            gate(inc, cand, GateConfig(alpha=0.0), now=NOW)
        with self.assertRaises(GateInputError):
            gate(inc, cand, GateConfig(selection="max-over-ckpts"), now=NOW)

    def test_both_seeds_recorded_unequal_invalid(self):
        inc, cand = _crn_pair(seed_inc=0, seed_cand=1)
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertEqual(d.exit_code, 2)
        self.assertTrue(any("seeds differ" in r for r in d.reasons))

    def test_len_mismatch_invalid(self):
        inc, cand = _crn_pair(n=200)
        cand.successes = cand.successes[:199]
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("episode counts differ" in r for r in d.reasons))

    def test_truncated_budget_invalid_directional(self):
        inc, cand = _crn_pair()
        inc.final_step, inc.design_steps = 35000, 100000
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("DIRECTIONAL" in r for r in d.reasons))
        # the record is still built and ledger-appendable
        self.assertEqual(d.record["verdict"], "INVALID")

    def test_successes_missing_invalid(self):
        inc, _ = _crn_pair()
        cand = _mk("cand", None, sr=50.0)
        cand.n_eval = 400
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("successes missing" in r for r in d.reasons))

    def test_both_seeds_none_degrade_then_assume_crn(self):
        inc, cand = _crn_pair(seed_inc=None, seed_cand=None)
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertNotEqual(d.verdict, "INVALID")
        self.assertFalse(d.record["statistics"]["paired"])
        self.assertTrue(any("UNPAIRED" in w for w in d.record["warnings"]))
        d2 = gate(inc, cand, _cfg(assume_crn=True), now=NOW)
        self.assertTrue(d2.record["statistics"]["paired"])
        self.assertTrue(any("ASSUMED" in w for w in d2.record["warnings"]))

    def test_mixed_recorded_1000_pairs_under_assume_crn(self):
        inc, cand = _crn_pair(seed_inc=1000, seed_cand=None)
        d = gate(inc, cand, _cfg(assume_crn=True), now=NOW)
        self.assertNotEqual(d.verdict, "INVALID")
        self.assertTrue(d.record["statistics"]["paired"])
        self.assertTrue(any("ASSUMED" in w for w in d.record["warnings"]))

    def test_mixed_recorded_42_invalid(self):
        inc, cand = _crn_pair(seed_inc=42, seed_cand=None)
        d = gate(inc, cand, _cfg(assume_crn=True), now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("almost certainly differ" in r
                            for r in d.reasons))

    def test_task_groups_unequal_invalid(self):
        inc, cand = _crn_pair()
        inc.task_groups, cand.task_groups = ["pusht"], ["libero_object"]
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("task_groups differ" in r for r in d.reasons))

    def test_libero_degrades_pairing_unless_sequential(self):
        inc, cand = _crn_pair(groups=("libero_object",))
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertNotEqual(d.verdict, "INVALID")
        self.assertFalse(d.record["statistics"]["paired"])
        self.assertIn(atlas.CAVEATS["libero-autoreset"], d.record["caveats"])
        self.assertNotIn(atlas.CAVEATS["pairing-validated"],
                         d.record["caveats"])

    def test_libero_sequential_eval_stays_paired(self):
        inc, cand = _crn_pair(groups=("libero_object",))
        d = gate(inc, cand, _cfg(sequential_eval=True), now=NOW)
        self.assertTrue(d.record["statistics"]["paired"])
        self.assertIn(atlas.CAVEATS["pairing-validated"], d.record["caveats"])
        self.assertIn(atlas.CAVEATS["async-envs"], d.record["caveats"])
        self.assertNotIn(atlas.CAVEATS["libero-autoreset"],
                         d.record["caveats"])


class TestVerdicts(unittest.TestCase):
    def test_ship(self):
        inc, cand = _crn_pair(n=400, p_inc=0.45, p_cand=0.60)
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "SHIP")
        self.assertEqual(d.exit_code, 0)
        st = d.record["statistics"]
        self.assertTrue(st["paired"])
        self.assertEqual(st["b"], 0)          # shared-u: cand is a superset
        self.assertGreater(st["delta_hat_pp"], 10.0)
        self.assertLessEqual(st["p_ship"], 0.05)
        self.assertTrue(any("no per-task regression flag" in r
                            for r in d.reasons))
        # sizing is computed even on SHIP records
        self.assertIsInstance(st["n_required"], int)
        self.assertIsInstance(st["episodes_needed"], int)

    def test_hold_via_p_worse(self):
        inc, cand = _crn_pair(n=400, p_inc=0.60, p_cand=0.45)
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "HOLD")
        self.assertEqual(d.exit_code, 1)
        self.assertLessEqual(d.record["statistics"]["p_worse"], 0.05)
        self.assertTrue(any("WORSE" in r for r in d.reasons))

    def test_hold_via_per_task_regression_despite_overall_gain(self):
        # 4 blocks of 100. Block 0: incumbent-only wins b=20 (delta -20 pp,
        # exact one-sided p = 2^-20). Blocks 1-3: candidate-only wins c=15
        # each. Overall b=20, c=45: delta +6.25 pp, p_ship significant — the
        # gate must still HOLD on the broken task.
        inc_s = ([True] * 20 + [False] * 80) + [False] * 300
        cand_s = [False] * 100 + ([True] * 15 + [False] * 85) * 3
        inc, cand = _mk("inc", inc_s), _mk("cand", cand_s)
        d = gate(inc, cand, _cfg(n_tasks=4), now=NOW)
        self.assertEqual(d.verdict, "HOLD")
        st = d.record["statistics"]
        self.assertGreater(st["delta_hat_pp"], 2.0)
        self.assertLessEqual(st["p_ship"], 0.05)
        rows = st["per_task"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows[0]["regression"])
        self.assertEqual(rows[0]["b"], 20)
        self.assertEqual(rows[0]["c"], 0)
        self.assertAlmostEqual(rows[0]["delta_pp"], -20.0, places=9)
        self.assertAlmostEqual(rows[0]["p_worse"], 0.5 ** 20, places=12)
        self.assertFalse(any(r["regression"] for r in rows[1:]))
        self.assertTrue(any(r.startswith("ASSUMPTION: task blocks "
                                         "contiguous equal")
                            for r in d.record["assumptions"]))

    def test_collect_more_small_effect_with_floor_caveat(self):
        # b=20, c=22 at n=200: delta_hat = 1.0 pp, p_d = 0.21 -> n_required
        # 3246 >> 200, and |delta_eff| < 2 pp triggers the measured floor.
        inc, cand = _bc_pair(200, b=20, c=22)
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "COLLECT-MORE")
        self.assertEqual(d.exit_code, 3)
        st = d.record["statistics"]
        self.assertAlmostEqual(st["delta_hat_pp"], 1.0, places=9)
        self.assertGreater(st["episodes_needed"], 0)
        self.assertEqual(st["n_required"],
                         episodes_for_min_effect(42 / 200, 2.0))
        self.assertEqual(st["episodes_needed"], st["n_required"] - 200)
        self.assertIn(CAVEAT_FLOOR_2PP, d.record["caveats"])

    def test_unresolvable_when_budget_capped(self):
        inc, cand = _bc_pair(200, b=20, c=22)
        d = gate(inc, cand, _cfg(max_budget=300), now=NOW)
        self.assertEqual(d.verdict, "UNRESOLVABLE")
        self.assertEqual(d.exit_code, 4)
        self.assertTrue(any("max_budget" in r for r in d.reasons))
        self.assertIn(CAVEAT_FLOOR_2PP, d.record["caveats"])

    def test_n_tasks_not_dividing_invalid(self):
        inc, cand = _crn_pair(n=200)
        d = gate(inc, cand, _cfg(n_tasks=3), now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("does not divide" in r for r in d.reasons))

    def test_unpaired_per_task_uses_fisher_exact(self):
        """On the unpaired degrade path the per-task test must be Fisher's
        exact one-sided tail (valid at any block size), not the pooled-z
        Gaussian (level drifts above nominal at ~20-episode blocks)."""
        inc_s = ([True] * 15 + [False] * 5) + ([True] * 10 + [False] * 10)
        cand_s = ([True] * 5 + [False] * 15) + ([True] * 10 + [False] * 10)
        inc = _mk("inc", inc_s, seed=None)
        cand = _mk("cand", cand_s, seed=None)
        d = gate(inc, cand, _cfg(n_tasks=2), now=NOW)   # no assume_crn
        st = d.record["statistics"]
        self.assertFalse(st["paired"])
        rows = st["per_task"]
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["b"])   # no discordant counts unpaired
        self.assertAlmostEqual(rows[0]["delta_pp"], -50.0, places=9)
        self.assertAlmostEqual(
            rows[0]["p_worse"],
            stats.fisher_exact_one_sided(5, 20, 15, 20), places=12)
        self.assertTrue(rows[0]["regression"])
        self.assertAlmostEqual(
            rows[1]["p_worse"],
            stats.fisher_exact_one_sided(10, 20, 10, 20), places=12)
        self.assertFalse(rows[1]["regression"])
        self.assertEqual(d.verdict, "HOLD")

    def test_multiplicity_assumption_disclosed_for_multi_task(self):
        """Per-task regression alpha is applied PER TASK, uncorrected —
        every multi-task record must disclose the family-wise false-flag
        probability; a single-block gate must not."""
        inc, cand = _crn_pair(n=200)
        d = gate(inc, cand, _cfg(n_tasks=10), now=NOW)
        fam = [a for a in d.record["assumptions"] if "family-wise" in a]
        self.assertEqual(len(fam), 1)
        self.assertIn("PER TASK", fam[0])
        # 1-(1-.05)^10 = 0.401 computed into the string
        self.assertIn("0.401", fam[0])
        d_single = gate(inc, cand, _cfg(), now=NOW)
        self.assertFalse(any("family-wise" in a
                             for a in d_single.record["assumptions"]))

    def test_collect_more_zero_episodes_gets_floor_reason(self):
        """Decisively measured but sub-floor effect at large n: the sizing
        would say 'collect 0 more episodes' — the verdict must instead say
        the effect is below the shippable floor (distinct reason-code)."""
        inc, cand = _bc_pair(3000, b=0, c=30)
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "COLLECT-MORE")
        st = d.record["statistics"]
        self.assertAlmostEqual(st["delta_hat_pp"], 1.0, places=9)
        self.assertLessEqual(st["p_ship"], 1e-8)
        self.assertEqual(st["episodes_needed"], 0)
        self.assertTrue(any("effect-below-floor" in r for r in d.reasons))
        self.assertTrue(any("will not change this verdict" in r
                            for r in d.reasons))
        self.assertFalse(any("collect 0 more" in r for r in d.reasons))
        self.assertIn(CAVEAT_FLOOR_2PP, d.record["caveats"])


class TestCurse(unittest.TestCase):
    def test_curse_from_prior_values(self):
        self.assertEqual(curse_from_prior(4), 4.0)
        self.assertAlmostEqual(curse_from_prior(16), 4.0 * math.sqrt(2.0),
                               places=12)
        self.assertEqual(curse_from_prior(1), 0.0)
        self.assertAlmostEqual(curse_from_prior(2), 4.0 * math.sqrt(0.5),
                               places=12)

    def test_prior_correction_turns_win_into_collect_more(self):
        # b=5, c=25 at n=400: delta_hat = 5.0 pp, p_ship ~ 1.6e-4 -> SHIP
        # under selection='final'; with J=4 max-over-checkpoints the prior
        # correction (4.0 pp) drops delta_eff to 1.0 < min_effect 2.0.
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        d_final = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d_final.verdict, "SHIP")
        self.assertEqual(d_final.record["statistics"]["curse_method"], "none")
        self.assertEqual(
            d_final.record["statistics"]["curse_correction_pp"], 0.0)
        d = gate(inc, cand, _cfg(selection="max-over-checkpoints",
                                 n_checkpoints=4), now=NOW)
        self.assertEqual(d.verdict, "COLLECT-MORE")
        st = d.record["statistics"]
        self.assertEqual(st["curse_method"], "atlas-prior-scaled")
        self.assertEqual(st["curse_correction_pp"], 4.0)
        self.assertAlmostEqual(st["delta_hat_pp"], 5.0, places=9)
        self.assertAlmostEqual(st["delta_effective_pp"], 1.0, places=9)
        self.assertIn(CAVEAT_FLOOR_2PP, d.record["caveats"])
        self.assertTrue(any("winner's-curse" in a
                            for a in d.record["assumptions"]))

    def test_curse_from_history_nonneg_and_positive_for_noisy_max(self):
        rng = random.Random(5)
        hist = [[rng.random() < 0.5 for _ in range(100)] for _ in range(4)]
        corr = curse_from_history(hist, n_boot=300, seed=7)
        self.assertGreaterEqual(corr, 0.0)
        self.assertGreater(corr, 0.0)   # 4 independent noisy ckpts: max is optimistic
        # deterministic under a fixed seed
        self.assertEqual(corr, curse_from_history(hist, n_boot=300, seed=7))
        # degenerate: identical checkpoints -> ties resolve to index 0 and
        # the 'optimism' is pure mean-zero bootstrap noise, clamped at 0 —
        # near zero, well below any real selection effect
        same = [[True] * 50 + [False] * 50] * 4
        self.assertLess(curse_from_history(same, n_boot=1000), 0.5)

    def test_history_bootstrap_method_no_prior_assumption(self):
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        rng = random.Random(9)
        hist = [_mk("ckpt%d" % j,
                    [rng.random() < 0.4 for _ in range(400)])
                for j in range(3)]
        d = gate(inc, cand, _cfg(selection="max-over-checkpoints"),
                 history=hist, now=NOW)
        st = d.record["statistics"]
        self.assertEqual(st["curse_method"], "history-bootstrap")
        self.assertGreaterEqual(st["curse_correction_pp"], 0.0)
        # the sqrt-log scaling ASSUMPTION belongs to the prior method only
        self.assertFalse(any("winner's-curse" in a
                             for a in d.record["assumptions"]))
        # estimand disclosure: the bootstrap measures eval-draw selection
        # optimism only, NOT retrain-lottery optimism — and the prior-scaled
        # reference value is recorded alongside for a record quoted alone
        est = [a for a in d.record["assumptions"]
               if "retrain-lottery" in a]
        self.assertEqual(len(est), 1)
        self.assertAlmostEqual(st["curse_prior_reference_pp"],
                               curse_from_prior(3), places=12)

    def test_curse_prior_reference_none_without_history(self):
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertIsNone(
            d.record["statistics"]["curse_prior_reference_pp"])

    def test_history_seed_mismatch_invalid(self):
        inc, cand = _bc_pair(400, b=5, c=25)
        hist = [_mk("ckpt0", [False] * 400, seed=5)]
        d = gate(inc, cand, _cfg(selection="max-over-checkpoints"),
                 history=hist, now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("history run ckpt0" in r for r in d.reasons))

    def test_history_n_mismatch_invalid(self):
        inc, cand = _bc_pair(400, b=5, c=25)
        hist = [_mk("ckpt0", [False] * 200)]
        d = gate(inc, cand, _cfg(selection="max-over-checkpoints"),
                 history=hist, now=NOW)
        self.assertEqual(d.verdict, "INVALID")

    def test_history_with_selection_final_raises(self):
        inc, cand = _bc_pair(400, b=5, c=25)
        hist = [_mk("ckpt0", [False] * 400)]
        with self.assertRaises(GateInputError):
            gate(inc, cand, _cfg(), history=hist, now=NOW)

    def test_max_over_checkpoints_needs_j_or_history(self):
        inc, cand = _bc_pair(400, b=5, c=25)
        with self.assertRaises(GateInputError):
            gate(inc, cand, _cfg(selection="max-over-checkpoints"), now=NOW)


class TestSizingAndDeterminism(unittest.TestCase):
    def test_sizing_formulas_match_inline_recomputation(self):
        za, zp = stats.norm_q(0.95), stats.norm_q(0.8)
        self.assertEqual(
            episodes_for_min_effect(0.21, 2.0),
            math.ceil(0.21 * ((za + zp) / 0.02) ** 2 - 1e-9))
        self.assertEqual(
            episodes_for_min_effect_unpaired(0.5, 2.0),
            math.ceil(2 * 0.5 * 0.5 * ((za + zp) / 0.02) ** 2 - 1e-9))
        self.assertAlmostEqual(
            power_at_min_effect_paired(200, 0.145, 2.0),
            stats.Phi(math.sqrt(200) * 0.02 / math.sqrt(0.145) - za),
            places=12)
        self.assertAlmostEqual(
            power_at_min_effect_unpaired(200, 0.3, 2.0),
            stats.Phi(0.02 / math.sqrt(2 * 0.3 * 0.7 / 200) - za),
            places=12)
        # the frozen-spec worked example (record_schema_json): p_d=0.145,
        # n=200 -> n_required 2242, power 0.1835
        self.assertEqual(episodes_for_min_effect(0.145, 2.0), 2242)
        self.assertAlmostEqual(power_at_min_effect_paired(200, 0.145, 2.0),
                               0.1835, delta=2e-4)

    def test_zero_discordance_uses_floor_with_assumption(self):
        s = [True] * 80 + [False] * 120
        inc, cand = _mk("inc", list(s)), _mk("cand", list(s))
        d = gate(inc, cand, _cfg(), now=NOW)
        st = d.record["statistics"]
        self.assertEqual(st["n_discordant"], 0)
        self.assertEqual(st["n_required"],
                         episodes_for_min_effect(1.0 / 200, 2.0))
        self.assertTrue(any(a.startswith("ASSUMPTION: zero discordant")
                            for a in d.record["assumptions"]))
        self.assertEqual(d.verdict, "COLLECT-MORE")

    def test_gate_is_byte_deterministic_and_ledger_verifies(self):
        inc, cand = _crn_pair(n=400, p_inc=0.45, p_cand=0.60)
        cfg = _cfg(selection="max-over-checkpoints", n_checkpoints=4)
        d1 = gate(inc, cand, cfg, now=NOW)
        d2 = gate(inc, cand, cfg, now=NOW)
        self.assertEqual(d1.record["record_sha256"],
                         d2.record["record_sha256"])
        self.assertEqual(d1.record, d2.record)
        self.assertIsInstance(d1, GateDecision)
        with tempfile.TemporaryDirectory() as td:
            ledger = os.path.join(td, "ledger.jsonl")
            append_record(d1.record, ledger)
            append_record(d2.record, ledger)
            rep = verify_ledger(ledger)
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["n_ok"], 2)

    def test_engine_exit_codes_frozen(self):
        for verdict, code in (("SHIP", 0), ("HOLD", 1), ("INVALID", 2),
                              ("COLLECT-MORE", 3), ("UNRESOLVABLE", 4)):
            self.assertEqual(
                GateDecision(verdict=verdict, record={}, reasons=[]).exit_code,
                code)


if __name__ == "__main__":
    unittest.main()
