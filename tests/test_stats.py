"""McNemar / bootstrap / helper statistics against hand-computed values."""

import math
import random
import unittest

from orbit_eval import stats
from orbit_eval.io import EvalRun


class TestHelpers(unittest.TestCase):
    def test_norm_q(self):
        self.assertAlmostEqual(stats.norm_q(0.975), 1.959964, places=5)
        self.assertAlmostEqual(stats.norm_q(0.5), 0.0, places=9)
        self.assertAlmostEqual(stats.norm_q(0.05), -1.644854, places=5)
        # round-trip with Phi
        for p in (0.01, 0.2, 0.7, 0.99):
            self.assertAlmostEqual(stats.Phi(stats.norm_q(p)), p, places=8)

    def test_wilson_hand_value(self):
        # k=50, n=100: Wilson 95% = (0.40384, 0.59616), hand-computed
        lo, hi = stats.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.40384, places=4)
        self.assertAlmostEqual(hi, 0.59616, places=4)
        self.assertAlmostEqual(stats.wilson_width_pp(50, 100), 19.232, places=2)

    def test_binom_interval_hand_value(self):
        # Binomial(10, .5): central 95% acceptance region is [2, 8]
        self.assertEqual(stats.binom_interval(10, 0.5), (2, 8))
        # property: the region really holds >= 95% mass
        for n, p in ((20, 0.3), (57, 0.8), (200, 0.5)):
            lo, hi = stats.binom_interval(n, p)
            pmf = stats.binom_pmf(n, p)
            self.assertGreaterEqual(sum(pmf[lo:hi + 1]), 0.95 - 1e-9)

    def test_sigma_from_gaps(self):
        # Var(gap) = 2 sigma^2  ->  sigma = sqrt(mean(g^2)/2)  [analyze_paired V4]
        self.assertAlmostEqual(stats.sigma_from_gaps([2, -2, 2, -2]),
                               math.sqrt(2.0), places=9)
        self.assertIsNone(stats.sigma_from_gaps([]))

    def test_set_hash_convention(self):
        # order-independent, int-normalising, sha1[:16]  [build_atlas.py]
        self.assertEqual(stats.set_hash([3, 1, 2]), stats.set_hash([1, 2, 3]))
        self.assertEqual(stats.set_hash([1.0, 2.0]), stats.set_hash([1, 2]))
        self.assertEqual(len(stats.set_hash([0])), 16)

    def test_budget_gate(self):
        ok, why = stats.budget_gate(100000, 100000)
        self.assertTrue(ok)
        self.assertIsNone(why)
        ok, why = stats.budget_gate(35000, 100000)
        self.assertFalse(ok)
        self.assertIn("35000", why)
        ok, why = stats.budget_gate(None, 100000)
        self.assertTrue(ok)          # unverifiable passes ...
        self.assertIn("unverifiable", why)   # ... but carries a caveat


class TestMcNemar(unittest.TestCase):
    def test_hand_value(self):
        # b=1, c=9: two-sided exact p = 2 * P(X <= 1 | n=10, 1/2) = 22/1024
        self.assertAlmostEqual(stats.mcnemar_exact(1, 9), 22 / 1024, places=12)

    def test_symmetry_and_edges(self):
        self.assertEqual(stats.mcnemar_exact(3, 8), stats.mcnemar_exact(8, 3))
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact(5, 5), 1.0, places=9)

    def test_one_sided(self):
        # P(X <= 1 | 10, .5) = 11/1024
        self.assertAlmostEqual(stats.mcnemar_one_sided(9, 1), 11 / 1024,
                               places=12)

    def test_discordant_counts(self):
        a = [1, 1, 0, 0, 1]
        b = [1, 0, 1, 0, 0]
        self.assertEqual(stats.discordant_counts(a, b), (1, 2, 1, 1))


class TestTwoProp(unittest.TestCase):
    def test_hand_value(self):
        # 60/100 vs 40/100: pooled p=.5, se=.070711, z=2.828427
        d, z, p = stats.two_prop_test(60, 100, 40, 100)
        self.assertAlmostEqual(d, 20.0, places=9)
        self.assertAlmostEqual(z, 2.828427, places=5)
        self.assertAlmostEqual(p, 2 * (1 - stats.Phi(2.828427)), places=6)
        self.assertAlmostEqual(p, 0.004678, places=5)

    def test_newcombe_contains_diff(self):
        lo, hi = stats.newcombe_ci(60, 100, 40, 100)
        self.assertLess(lo, 20.0)
        self.assertGreater(hi, 20.0)
        self.assertGreater(lo, 0.0)   # significant at ~95%


class TestPairedBootstrap(unittest.TestCase):
    def test_null_ci_covers_zero(self):
        rng = random.Random(3)
        a = [rng.random() < 0.5 for _ in range(300)]
        lo, hi = stats.paired_boot_ci(a, list(a), n_boot=500)
        self.assertEqual((lo, hi), (0.0, 0.0))   # identical runs: no width

    def test_signal_ci_excludes_zero(self):
        rng = random.Random(4)
        a, b = [], []
        for _ in range(400):
            u = rng.random()
            a.append(u < 0.6)
            b.append(u < 0.4)          # CRN-style: same u, A strictly better
        lo, hi = stats.paired_boot_ci(a, b, n_boot=1000)
        self.assertGreater(lo, 0.0)
        self.assertLess(abs((lo + hi) / 2 - 20.0), 6.0)


def _mk(run_id, **kw):
    return EvalRun(run_id=run_id, fmt=kw.pop("fmt", "lerobot"), **kw)


class TestCompareRuns(unittest.TestCase):
    def _paired_pair(self, seed_a=1000, seed_b=1000):
        rng = random.Random(11)
        sa, sb = [], []
        for _ in range(200):
            u = rng.random()
            sa.append(u < 0.55)
            sb.append(u < 0.45)
        a = _mk("A", sr=100 * sum(sa) / 200, n_eval=200, seed=seed_a,
                successes=sa)
        b = _mk("B", sr=100 * sum(sb) / 200, n_eval=200, seed=seed_b,
                successes=sb)
        return a, b

    def test_crn_paired_path(self):
        a, b = self._paired_pair()
        res = stats.compare_runs(a, b)
        self.assertTrue(res["paired"])
        self.assertEqual(res["n"], 200)
        self.assertLessEqual(res["p_mcnemar"], 0.05)
        self.assertGreater(res["diff_pp"], 0)
        # every B-only success would contradict the CRN construction above
        self.assertEqual(res["n01"], 0)

    def test_seed_mismatch_falls_back_unpaired(self):
        a, b = self._paired_pair(seed_a=0, seed_b=1)
        res = stats.compare_runs(a, b)
        self.assertFalse(res["paired"])
        self.assertTrue(any("seeds differ" in w for w in res["warnings"]))

    def test_unknown_seeds_need_assume_crn(self):
        a, b = self._paired_pair(seed_a=None, seed_b=None)
        res = stats.compare_runs(a, b)
        self.assertFalse(res["paired"])
        res = stats.compare_runs(a, b, assume_crn=True)
        self.assertTrue(res["paired"])
        self.assertTrue(any("ASSUMED" in w for w in res["warnings"]))

    def test_mixed_recorded_seed_never_pairs_under_assume_crn(self):
        # A recorded seed 42, B recorded nothing: 42 almost certainly differs
        # from B's silent default 1000, so --assume-crn must NOT pair them
        a, b = self._paired_pair(seed_a=42, seed_b=None)
        res = stats.compare_runs(a, b, assume_crn=True)
        self.assertFalse(res["paired"])
        self.assertTrue(any("almost certainly differ" in w
                            for w in res["warnings"]))
        res = stats.compare_runs(a, b)
        self.assertFalse(res["paired"])

    def test_mixed_seed_1000_pairs_only_under_assume_crn(self):
        # the one recorded seed IS the silent default: pairing is plausible,
        # but still only as an explicit assumption
        a, b = self._paired_pair(seed_a=1000, seed_b=None)
        res = stats.compare_runs(a, b)
        self.assertFalse(res["paired"])
        res = stats.compare_runs(a, b, assume_crn=True)
        self.assertTrue(res["paired"])
        self.assertTrue(any("ASSUMED" in w for w in res["warnings"]))

    def test_budget_gate_blocks_stats(self):
        a = _mk("A", fmt="orbit", sr=30.0, n_eval=500, seed=0,
                final_step=35000, design_steps=100000)
        b = _mk("B", fmt="orbit", sr=40.0, n_eval=500, seed=0,
                final_step=100000, design_steps=100000)
        res = stats.compare_runs(a, b)
        self.assertTrue(res["invalid"])
        self.assertNotIn("diff_pp", res)


if __name__ == "__main__":
    unittest.main()
