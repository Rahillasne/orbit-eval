"""Retrain-priced per-task HOLD (HOLDINJ_PREREG 2026-08-15, sha256 938a3d46...).

The unpriced per-task rule HOLDs 96.4% of identical-data retrain pairs at
production k (HOLDRATE_DIAG_2026-08-15); these tests pin the priced rule's
contract: lottery-sized drops classify LOTTERY-EXPECTED and do not HOLD,
catastrophic drops still HOLD, unpriced paths stay byte-identical to v0.
"""
import unittest

from orbit_eval.gate.engine import gate, resolve_sigma_pertask
from orbit_eval.gate.records import GateConfig
from orbit_eval.io import EvalRun


def _run(rid, blocks, seed=1000):
    succ = [bool(x) for b in blocks for x in b]
    return EvalRun(run_id=rid, fmt="orbit", sr=100.0 * sum(succ) / len(succ),
                   n_eval=len(succ), seed=seed, successes=succ,
                   task_groups=["synthetic"] * len(blocks))


def _block(n, k, flip_prefix=0):
    """n episodes, k successes; optionally flip the first successes to 0."""
    b = [1] * k + [0] * (n - k)
    for i in range(flip_prefix):
        b[i] = 0
    return b


def _cfg(**kw):
    base = dict(comparison_level="retrain", n_tasks=10, assume_crn=True)
    base.update(kw)
    return GateConfig(**base)


class TestResolution(unittest.TestCase):
    def test_explicit_beats_regime(self):
        c = _cfg(regime="pi05-ft-k88-multitask", sigma_pertask_pp=7.5)
        self.assertEqual(resolve_sigma_pertask(c), (7.5, "explicit"))

    def test_atlas_regime(self):
        c = _cfg(regime="pi05-ft-k88-multitask")
        sp, src = resolve_sigma_pertask(c)
        self.assertEqual(sp, 11.00)
        self.assertEqual(src, "atlas:pi05-ft-k88-multitask")

    def test_atlas_regime_smolvla(self):
        # was df-starved None until PERTASK-3 Leg A' (2026-08-18) measured it
        # at df=7: binding 6.69 (raw 7.33 carried in the row)
        c = _cfg(regime="smolvla-ft-k88-multitask")
        sp, src = resolve_sigma_pertask(c)
        self.assertEqual(sp, 6.69)
        self.assertEqual(src, "atlas:smolvla-ft-k88-multitask")

    def test_unmeasured_regime_stays_unpriced(self):
        # a row with no sigma_pertask key resolves to unpriced v0 behavior
        c = _cfg(regime="smolvla-ft")
        self.assertEqual(resolve_sigma_pertask(c), (None, None))

    def test_checkpoint_level_never_prices(self):
        c = GateConfig(sigma_pertask_pp=11.0)
        self.assertEqual(resolve_sigma_pertask(c), (None, None))


class TestPricedFlags(unittest.TestCase):
    def _gate(self, drop_eps, sigma=11.0):
        # 10 tasks x 100 eps; candidate drops `drop_eps` episodes on task 0.
        inc = [_block(100, 70) for _ in range(10)]
        cand = [_block(100, 70) for _ in range(10)]
        cand[0] = _block(100, 70, flip_prefix=drop_eps)
        d = gate(_run("inc", inc), _run("cand", cand),
                 _cfg(sigma_pertask_pp=sigma, sigma_run_pp=4.05),
                 now="2026-08-15T00:00:00Z")
        return d

    def test_lottery_sized_drop_is_not_gate_actionable(self):
        # -12 pp on one task: eval-exact flags it (p ~ 2^-12), the lottery
        # (sigma 11 -> se_total ~ 15.9) does not.
        d = self._gate(12)
        row = d.record['statistics']['per_task'][0]
        self.assertEqual(row["tier"], "lottery-expected")
        self.assertFalse(row["regression"])
        self.assertNotEqual(d.verdict, "HOLD")

    def test_catastrophic_drop_still_holds(self):
        # -55 pp: beyond any lottery draw the atlas has measured.
        d = self._gate(55)
        row = d.record['statistics']['per_task'][0]
        self.assertEqual(row["tier"], "regression-beyond-lottery")
        self.assertTrue(row["regression"])
        self.assertEqual(d.verdict, "HOLD")
        self.assertIn("per-task regression flag", d.reasons[0])

    def test_priced_disclosure_and_budget_quote(self):
        d = self._gate(12)
        text = " ".join(d.record["assumptions"])
        self.assertIn("RETRAIN-PRICED", text)
        self.assertIn("retrains/side", text)

    def test_p_retrain_recorded_and_monotone(self):
        small, big = self._gate(8), self._gate(40)
        self.assertLess(big.record['statistics']['per_task'][0]['p_retrain'],
                        small.record['statistics']['per_task'][0]['p_retrain'])


class TestUnpricedPathsUnchanged(unittest.TestCase):
    def test_v0_checkpoint_rows_have_no_pricing_keys(self):
        inc = [_block(100, 70) for _ in range(10)]
        cand = [_block(100, 70) for _ in range(10)]
        cand[0] = _block(100, 70, flip_prefix=12)
        d = gate(_run("inc", inc), _run("cand", cand),
                 GateConfig(n_tasks=10, assume_crn=True),
                 now="2026-08-15T00:00:00Z")
        row = d.record['statistics']['per_task'][0]
        self.assertNotIn("p_retrain", row)
        self.assertNotIn("tier", row)
        # v0 eval-exact behavior: the 12 pp drop flags and HOLDs
        self.assertTrue(row["regression"])
        self.assertEqual(d.verdict, "HOLD")

    def test_retrain_without_pertask_prior_keeps_v0_flags_plus_disclosure(self):
        inc = [_block(100, 70) for _ in range(10)]
        cand = [_block(100, 70) for _ in range(10)]
        cand[0] = _block(100, 70, flip_prefix=12)
        d = gate(_run("inc", inc), _run("cand", cand),
                 _cfg(sigma_run_pp=4.05),  # suite prior only
                 now="2026-08-15T00:00:00Z")
        row = d.record['statistics']['per_task'][0]
        self.assertNotIn("p_retrain", row)
        self.assertTrue(row["regression"])
        self.assertIn("NOT priced into per-task flags",
                      " ".join(d.record["assumptions"]))


if __name__ == "__main__":
    unittest.main()
