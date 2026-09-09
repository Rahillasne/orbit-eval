"""The selftest must pass (deterministic under its fixed seed), and its
exact-McNemar power prediction must be self-consistent."""

import unittest

from orbit_eval import selftest


class TestSelftest(unittest.TestCase):
    def test_selftest_passes(self):
        code, rows, text = selftest.run_selftest(quiet=True, n_rep=400)
        self.assertEqual(code, 0, "selftest FAILED:\n" + text)
        self.assertTrue(all(good for *_x, good in rows))
        self.assertIn("PASSED", text)

    def test_mcnemar_prediction_null_is_conservative(self):
        # exact test: predicted size can never exceed alpha
        for n, eps in ((100, 0.05), (200, 0.06), (500, 0.10)):
            self.assertLessEqual(
                selftest._mcnemar_predicted(n, eps, eps), 0.05 + 1e-12)

    def test_mcnemar_prediction_monotone_in_effect(self):
        p1 = selftest._mcnemar_predicted(200, 0.08, 0.06)
        p2 = selftest._mcnemar_predicted(200, 0.11, 0.03)
        self.assertLess(p1, p2)


if __name__ == "__main__":
    unittest.main()
