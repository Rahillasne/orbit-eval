"""Power math against hand-computed values (MDE = 2.80 * sigma * sqrt(2/m))."""

import unittest

from orbit_eval import atlas, power


class TestMDE(unittest.TestCase):
    def test_hand_values(self):
        # (1.959964+0.841621) * 2.09 * sqrt(2/2) = 2.801585*2.09 = 5.8553
        self.assertAlmostEqual(power.mde(2.09, 2), 5.8553, places=3)
        # 2.801585 * 11.43 * sqrt(2/8) = 32.0221 * 0.5 = 16.0111
        self.assertAlmostEqual(power.mde(11.43, 8), 16.0111, places=3)
        # 2.801585 * 3.31 * sqrt(2/1) = 9.2732*1.414214 = 13.1144
        self.assertAlmostEqual(power.mde(3.31, 1), 13.1144, places=3)

    def test_draws_needed_hand_values(self):
        # matches Table 2 of POWER_ANALYSIS.md: PushT sigma 2.09, 5 pp -> 3
        self.assertEqual(power.draws_needed(2.09, 5), 3)
        # LIBERO k=22 sigma 11.43, 20 pp -> ceil(2*(2.801585*11.43/20)^2)
        #   = ceil(2*1.60111^2) = ceil(5.1271) = 6
        self.assertEqual(power.draws_needed(11.43, 20), 6)

    def test_mde_draws_round_trip(self):
        for sigma in (2.09, 3.31, 11.43):
            for m in (2, 3, 5, 8, 12, 20):
                self.assertEqual(power.draws_needed(sigma, power.mde(sigma, m)),
                                 m)

    def test_power_at_mde_is_80pct(self):
        for sigma, m in ((2.09, 3), (11.43, 8)):
            self.assertAlmostEqual(
                power.two_sided_power(power.mde(sigma, m), sigma, m),
                0.80, places=3)


class TestProductRegime(unittest.TestCase):
    def test_59_draws_per_arm(self):
        """The frozen honesty number: LIBERO product regime, 10 pp effect.
        sigma = sqrt(19.01^2 + 3.31^2) = 19.2961 -> 59 draws/arm."""
        sig = power.method_sigma(atlas.SIGMA_SET_PRODUCT_PP,
                                 atlas.REGIMES["smolvla-ft"]["sigma_run"])
        self.assertAlmostEqual(sig, 19.2961, places=3)
        self.assertEqual(power.draws_needed(sig, 10.0), 59)

    def test_fixed_set_is_cheap_by_contrast(self):
        # same 10 pp question on a FIXED set: sigma_run only
        self.assertEqual(
            power.draws_needed(atlas.REGIMES["smolvla-ft"]["sigma_run"], 10.0),
            2)


class TestSigmaCI(unittest.TestCase):
    def test_hand_value(self):
        # sigma=2.09, df=8: lo = 2.09*sqrt(8/15.507), hi = 2.09*sqrt(8/2.733)
        lo, hi = power.sigma_ci(2.09, 8)
        self.assertAlmostEqual(lo, 1.5013, places=3)
        self.assertAlmostEqual(hi, 3.5757, places=3)
        self.assertIsNone(power.sigma_ci(2.09, None))
        self.assertIsNone(power.sigma_ci(2.09, 99))


class TestVRFTrue(unittest.TestCase):
    def test_limits(self):
        # w=0 (purely additive), sigma_0=0: VRF = d_bar/m  [power_paired.py]
        self.assertAlmostEqual(power.vrf_true(0.0, 4, 51.5, 3.2, 0.0),
                               51.5 / 4, places=9)
        # w=1 (pure set-idiosyncrasy): pairing buys nothing, VRF = 1
        self.assertAlmostEqual(power.vrf_true(1.0, 4, 51.5, 3.2, 0.0), 1.0,
                               places=9)


class TestAtlasIntegrity(unittest.TestCase):
    def test_measured_numbers_present(self):
        R = atlas.REGIMES
        self.assertEqual(R["pusht-dp"]["sigma_run"], 2.09)
        self.assertEqual(R["pusht-dp-rebuilt"]["sigma_run"], 1.292)   # founder-adopted 2026-08-20 (was 1.80, half the evidence)
        self.assertEqual(R["libero-dp-k22"]["sigma_run"], 11.43)
        self.assertEqual(R["libero-dp-k31"]["sigma_run"], 7.48)
        self.assertEqual(R["libero-dp-k44"]["sigma_run"], 6.24)
        self.assertEqual(R["smolvla-ft"]["sigma_run"], 3.31)
        self.assertEqual(R["smolvla-ft-pooled"]["sigma_run"], 2.82)
        # Tier-3 repair (TIER3_RESULTS.md resolved block, 2026-07-30): the
        # budget-contaminated per-suite sigmas were withdrawn; these are the
        # clean budget-matched values that accompany the restored pooled 2.82.
        self.assertEqual(R["smolvla-ft-pooled"]["per_suite"],
                         {"goal": 1.08, "spatial": 1.84, "long10": 2.94})
        self.assertEqual(R["smolvla-ft-pooled"]["df"], 8)
        self.assertEqual(R["pusht-dp"]["sigma_0"], 1.91)
        self.assertEqual(R["smolvla-ft"]["sigma_0"], 0.00)
        self.assertEqual(atlas.SIGMA_SET_PRODUCT_PP, 19.01)
        self.assertEqual(atlas.STACK_SHIFT_PUSHT_PP, -18.2)

    def test_get_regime(self):
        self.assertEqual(atlas.get_regime("smolvla_ft")["sigma_run"], 3.31)
        with self.assertRaises(KeyError):
            atlas.get_regime("nope")


if __name__ == "__main__":
    unittest.main()
