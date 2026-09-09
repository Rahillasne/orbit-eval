"""Regime-scope assertion (pre-check 11, 2026-08-16): a --regime whose
measured cell verifiably contradicts the comparison's own metadata must rule
INVALID instead of pricing with the wrong cell's sigma. Fires only on
verifiable contradictions; silent when inputs carry no metadata (operator
assertion, disclosed via the resolve source) — so all matched-input behavior
stays byte-identical to the pre-check-11 engine.
"""
import random
import unittest

from orbit_eval import atlas
from orbit_eval.gate.engine import GateInputError, gate
from orbit_eval.gate.records import GateConfig
from orbit_eval.io import EvalRun


def _run(rid, groups, n_per=100, seed=1000, episodes=None):
    succ = [True] * (n_per * max(1, len(groups)))
    succ[0] = False    # avoid degenerate all-success SR edge
    return EvalRun(run_id=rid, fmt="orbit",
                   sr=100.0 * sum(succ) / len(succ), n_eval=len(succ),
                   seed=seed, successes=succ, task_groups=list(groups),
                   episodes=episodes)


LIBERO10 = ["libero_object_task%d" % i for i in range(10)]


def _cfg(**kw):
    base = dict(comparison_level="retrain", assume_crn=True, n_boot=50)
    base.update(kw)
    return GateConfig(**base)


class TestRegimeScope(unittest.TestCase):
    def test_env_contradiction_rules_invalid(self):
        # pusht-measured regime priced onto a LIBERO eval: verifiable lie
        inc = _run("a", LIBERO10)
        cand = _run("b", LIBERO10)
        dec = gate(inc, cand, _cfg(regime="pusht-dp-rebuilt", n_tasks=10))
        self.assertEqual(dec.verdict, "INVALID")
        self.assertTrue(any("wrong-cell sigma" in r for r in dec.reasons))

    def test_task_count_contradiction_rules_invalid(self):
        # the judges' wrong-regime finale: 10-task pricing asserted on a
        # single-task eval whose n happens to tile
        inc = _run("a", ["libero_object_single"], n_per=200)
        cand = _run("b", ["libero_object_single"], n_per=200)
        dec = gate(inc, cand,
                   _cfg(regime="pi05-ft-k88-multitask", n_tasks=10))
        self.assertEqual(dec.verdict, "INVALID")
        self.assertTrue(any("mismatched battery" in r for r in dec.reasons))

    def test_k_contradiction_rules_invalid(self):
        # corpus-shaped input carrying its training-set episode list: k=22
        # data must not be priced by the k=88 cell
        eps22 = list(range(22))
        inc = _run("a", LIBERO10, episodes=eps22)
        cand = _run("b", LIBERO10, episodes=eps22)
        dec = gate(inc, cand,
                   _cfg(regime="pi05-ft-k88-multitask", n_tasks=10))
        self.assertEqual(dec.verdict, "INVALID")
        self.assertTrue(any("does not transfer across k" in r
                            for r in dec.reasons))

    def test_matched_metadata_is_silent(self):
        eps88 = list(range(88))
        inc = _run("a", LIBERO10, episodes=eps88)
        cand = _run("b", LIBERO10, episodes=eps88)
        dec = gate(inc, cand,
                   _cfg(regime="pi05-ft-k88-multitask", n_tasks=10))
        self.assertNotEqual(dec.verdict, "INVALID")
        self.assertFalse(any("wrong-cell" in r or "mismatched battery" in r
                             or "transfer across k" in r
                             for r in dec.reasons))

    def test_metadata_free_inputs_stay_silent(self):
        # no task_groups, no episodes: scope is an operator assertion —
        # pre-check-11 must not fire (byte-stability of the v0 contract)
        inc = _run("a", [], n_per=200)
        cand = _run("b", [], n_per=200)
        dec = gate(inc, cand, _cfg(regime="pusht-dp-rebuilt"))
        self.assertNotEqual(dec.verdict, "INVALID")

    def test_synthetic_group_names_stay_silent(self):
        # unrecognized vocabulary proves nothing (existing harness idiom)
        inc = _run("a", ["synthetic"] * 10)
        cand = _run("b", ["synthetic"] * 10)
        dec = gate(inc, cand,
                   _cfg(regime="pi05-ft-k88-multitask", n_tasks=10))
        self.assertFalse(any("wrong-cell" in r for r in dec.reasons))

    def test_unknown_regime_keeps_existing_error_path(self):
        inc = _run("a", LIBERO10)
        cand = _run("b", LIBERO10)
        with self.assertRaises(GateInputError):
            gate(inc, cand, _cfg(regime="act-made-up-regime", n_tasks=10))

    def test_checkpoint_level_never_fires(self):
        # scope check is a retrain-level concern only
        inc = _run("a", ["libero_object_single"], n_per=200)
        cand = _run("b", ["libero_object_single"], n_per=200)
        dec = gate(inc, cand,
                   GateConfig(comparison_level="checkpoint", assume_crn=True,
                              n_boot=50, regime="pusht-dp-rebuilt"))
        self.assertFalse(any("wrong-cell" in r for r in dec.reasons))


if __name__ == "__main__":
    unittest.main()


class TestSRRegimeScope(unittest.TestCase):
    """Pre-check 11b: a sigma measured on a near-floor cell must not price a
    healthy one. sigma in pp carries a binomial term scaling as sqrt(p(1-p)),
    so SR regime is part of a regime's scope, not prose beside it
    (PERTASK-3 audit, 2026-08-18)."""

    @staticmethod
    def _run(rid, p, n=500, seed=1000):
        rng = random.Random(sum(map(ord, rid)))   # literal: hash() varies with PYTHONHASHSEED
        s = [rng.random() < p for _ in range(n)]
        return EvalRun(run_id=rid, fmt="lerobot", sr=100.0 * sum(s) / n,
                       n_eval=n, seed=seed, successes=s,
                       task_groups=["pusht"], episodes=list(range(206)))

    def _gate(self, p_inc, p_cand, regime="pusht-act"):
        return gate(self._run("inc", p_inc), self._run("cand", p_cand),
                    GateConfig(n_tasks=1, n_boot=50,
                               comparison_level="retrain", regime=regime),
                    now="2026-08-17T00:00:00Z")

    def test_in_band_still_prices(self):
        d = self._gate(0.012, 0.030)
        self.assertNotEqual(d.verdict, "INVALID")
        self.assertEqual(d.record["statistics"]["sigma_run_source"],
                         "atlas:pusht-act")

    def test_healthy_policy_is_out_of_regime(self):
        d = self._gate(0.60, 0.632)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("out-of-regime pricing" in r for r in d.reasons))

    def test_mirror_side_is_not_a_false_match(self):
        # sqrt(p(1-p)) is symmetric about 50%: a 95% cell shares the 5% cell's
        # sampling floor. The floor test alone would pass it; the band is
        # wholly below the axis, so it must not.
        d = self._gate(0.95, 0.97)
        self.assertEqual(d.verdict, "INVALID")

    def test_just_outside_the_band_stays_silent(self):
        # the guard is a scope check, not a containment check: SRs a little
        # above the measured band still share its floor and must keep pricing
        d = self._gate(0.05, 0.07)
        self.assertNotEqual(d.verdict, "INVALID")

    def test_row_without_sr_range_is_unguarded(self):
        # pusht-dp-rebuilt declares no sr_range: scope stays an operator
        # assertion exactly as before this check existed
        self.assertIsNone(atlas.get_regime("pusht-dp-rebuilt").get("sr_range"))

    def test_checkpoint_level_never_fires(self):
        d = gate(self._run("inc", 0.60), self._run("cand", 0.632),
                 GateConfig(n_tasks=1, n_boot=50,
                            comparison_level="checkpoint", regime="pusht-act"),
                 now="2026-08-17T00:00:00Z")
        self.assertNotEqual(d.verdict, "INVALID")

    def test_floor_mismatch_helper_shape(self):
        band = (0.2, 3.0)
        self.assertEqual(atlas.sr_floor_mismatch(band, 1.35, 500), 1.0)
        self.assertLess(atlas.sr_floor_mismatch(band, 5.0, 500),
                        atlas.SR_SCOPE_FLOOR_RATIO)
        self.assertGreater(atlas.sr_floor_mismatch(band, 20.0, 500),
                           atlas.SR_SCOPE_FLOOR_RATIO)
        self.assertEqual(atlas.sr_floor_mismatch(band, 95.0, 500),
                         float("inf"))
        self.assertIsNone(atlas.sr_floor_mismatch(None, 50.0, 500))


class TestAlohaACTRowsScope(unittest.TestCase):
    """The two ALOHA ACT rows (PERTASK-3 B2) match on env, policy AND k, so
    every pre-check-11 condition passes between them — only 11b keeps a
    12%-SR cell from pricing a 78%-SR one. Locks in the scope that the
    prose LOW-SR stamp alone could not enforce."""

    @staticmethod
    def _run(rid, p, n=500):
        rng = random.Random(sum(map(ord, rid)))   # literal: hash() varies with PYTHONHASHSEED
        s = [rng.random() < p for _ in range(n)]
        return EvalRun(run_id=rid, fmt="lerobot", sr=100.0 * sum(s) / n,
                       n_eval=n, seed=1000, successes=s,
                       task_groups=["aloha"], episodes=list(range(50)))

    def _gate(self, p_inc, p_cand, regime):
        return gate(self._run("inc", p_inc), self._run("cand", p_cand),
                    GateConfig(n_tasks=1, n_boot=50,
                               comparison_level="retrain", regime=regime),
                    now="2026-08-17T00:00:00Z")

    def test_both_rows_declare_their_measured_band(self):
        self.assertEqual(
            atlas.get_regime("aloha-transfer-cube-act")["sr_range"], (69.6, 88.4))
        self.assertEqual(
            atlas.get_regime("aloha-insertion-act")["sr_range"], (9.0, 15.4))

    def test_each_row_prices_its_own_cell(self):
        for regime, p in (("aloha-transfer-cube-act", 0.779),
                          ("aloha-insertion-act", 0.120)):
            d = self._gate(p, p + 0.02, regime)
            self.assertNotEqual(d.verdict, "INVALID", regime)
            self.assertEqual(d.record["statistics"]["sigma_run_source"],
                             "atlas:" + regime)

    def test_insertion_row_cannot_price_the_transfer_cube_cell(self):
        d = self._gate(0.779, 0.80, "aloha-insertion-act")
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("out-of-regime pricing" in r for r in d.reasons))

    def test_transfer_cube_row_cannot_price_the_insertion_cell(self):
        d = self._gate(0.120, 0.14, "aloha-transfer-cube-act")
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("out-of-regime pricing" in r for r in d.reasons))

    def test_env_and_k_alone_would_not_have_caught_it(self):
        # both rows are env=aloha, policy=act, k=50 — pre-check 11's existing
        # conditions are all satisfied in the cross-pricing case above
        a = atlas.get_regime("aloha-transfer-cube-act")
        b = atlas.get_regime("aloha-insertion-act")
        self.assertEqual((a["env"], a["policy"], a["k"]),
                         (b["env"], b["policy"], b["k"]))


class TestCrossCellSigmaComparison(unittest.TestCase):
    """n_eval + sigma_run_convention exist so a sigma_run from one cell can be
    honestly set beside another's. compare_sigma_run() must REFUSE rather than
    return a number when it cannot (2026-08-20)."""

    def test_every_row_records_n_eval(self):
        missing = [k for k, r in atlas.REGIMES.items() if not r.get("n_eval")]
        self.assertEqual(missing, [], "rows without n_eval: %s" % missing)

    def test_every_row_records_a_convention(self):
        bad = [k for k, r in atlas.REGIMES.items()
               if r.get("sigma_run_convention") not in
               ("eval-inclusive", "unknown")]
        self.assertEqual(bad, [])

    def test_convention_resolved_for_all_but_one(self):
        unknown = [k for k, r in atlas.REGIMES.items()
                   if r["sigma_run_convention"] == "unknown"]
        self.assertEqual(unknown, ["smolvla-ft-pooled"])

    def test_act_vs_dp_is_refused_on_precision_and_floor_share(self):
        """The multiple this session quoted, withdrew, restored, and has now
        withdrawn for good. pusht-dp-rebuilt is a df=2 estimate from TWO gaps,
        one of them exactly 0.00, sitting at 85% sampling floor. The ratio's own
        95% F interval is [0.58, 9.26] and CONTAINS 1, so the two cells are not
        distinguishable at all. No hand-fed SR can make this quotable."""
        c = atlas.compare_sigma_run("aloha-transfer-cube-act", "pusht-dp-rebuilt")
        self.assertFalse(c["quotable"])
        self.assertIsNone(c["ratio_raw"])
        # at the founder-adopted 1.292/df=4 the cell is 169% sampling floor —
        # there is no retrain variance in it at all to compare against
        self.assertTrue(any("sampling floor" in r for r in c["reasons"]))

    def test_a_ratio_indistinguishable_from_one_is_refused(self):
        # h16 vs h8 are 4.839 vs 4.849 — genuinely equal. "A is 1.00x B" is not
        # a finding, and the CI check says so rather than printing 1.00.
        c = atlas.compare_sigma_run("pusht-act-h16", "pusht-act-h8")
        self.assertFalse(c["quotable"])
        self.assertTrue(any("CONTAINS" in r for r in c["reasons"]))

    def test_a_row_without_df_cannot_carry_a_ratio(self):
        import copy
        saved = atlas.REGIMES["pusht-act-h8"]["df"]
        try:
            atlas.REGIMES["pusht-act-h8"]["df"] = None
            c = atlas.compare_sigma_run("pusht-act-h16", "pusht-act-h8")
            self.assertFalse(c["quotable"])
            self.assertTrue(any("records no df" in r for r in c["reasons"]))
        finally:
            atlas.REGIMES["pusht-act-h8"]["df"] = saved

    def test_unknown_convention_is_still_refused(self):
        c = atlas.compare_sigma_run("aloha-transfer-cube-act",
                                    "smolvla-ft-pooled", sr_b=55.0)
        self.assertFalse(c["quotable"])
        self.assertTrue(any("convention" in r for r in c["reasons"]))

    def test_mismatched_sampling_floors_are_refused(self):
        # a near-floor cell against a healthy one: the ratio would be an artifact
        c = atlas.compare_sigma_run("aloha-transfer-cube-act",
                                    "pusht-act", sr_b=1.35)
        self.assertFalse(c["quotable"])
        self.assertTrue(any("sampling floors" in r for r in c["reasons"]))

    def test_readings_reproduce_each_row_published_floor(self):
        """sigma_run_readings must default to the MEASURED mean SR, not the
        midpoint of sr_range — the midpoint is not p_bar and did not reproduce
        the rows' own published floors (audit 2026-08-20)."""
        for key, published_floor in (("aloha-transfer-cube-act", 1.856),
                                     ("aloha-insertion-act", 1.452),
                                     ("pusht-act", 0.516),
                                     ("pusht-act-h8", 1.654),
                                     ("pusht-act-h16", 1.856)):
            r = atlas.sigma_run_readings(key)
            self.assertAlmostEqual(r["floor"], published_floor, places=3,
                                   msg="%s floor %r != published %r"
                                       % (key, r["floor"], published_floor))

    def test_degenerate_sr_cannot_slip_through_the_floor_check(self):
        # audit bug 3: floor==0.0 is falsy, so the MOST extreme mismatch used to
        # pass the guard silently. 0% and 100% must never be quotable.
        for sr in (0.0, 100.0):
            c = atlas.compare_sigma_run("aloha-transfer-cube-act",
                                        "pusht-dp-rebuilt", sr_b=sr)
            self.assertFalse(c["quotable"], "sr_b=%r slipped through" % sr)
            self.assertIsNone(c["ratio_raw"])

    def test_dp_row_is_floor_dominated_at_every_plausible_sr(self):
        # Regression guard for a real error: at the WRONG SR (60%, a figure the
        # repo carried) the floor exceeds sigma_run and looks like evidence of a
        # paired estimate. At the MEASURED post-rebuild SR (17.6%, from 12 runs
        # in experiments/forecast/*.csv) it does not. Never infer a spread
        # convention from a floor computed at an unverified SR.
        # The original error: a floor computed at an UNVERIFIED SR (60%) was
        # used to infer a spread convention. The lesson survives its own
        # example — at the adopted 1.292 the cell is below its floor at BOTH
        # SRs, so the 2026-08-20 "it sits above the floor" reasoning was itself
        # an artifact of the superseded 1.80. Never infer a convention from a
        # floor, full stop.
        for sr in (60.0, 17.05):
            r = atlas.sigma_run_readings("pusht-dp-rebuilt", sr_pp=sr)
            self.assertGreater(r["floor"], r["raw"])
            self.assertEqual(r["floor_subtracted"], 0.0)

    def test_readings_need_an_sr_to_compute_a_floor(self):
        r = atlas.sigma_run_readings("libero-dp-k22")   # declares no sr_range
        self.assertIsNone(r["floor"])
        self.assertIsNone(r["floor_subtracted"])
        self.assertEqual(r["n_eval"], 200)


class TestFloorDominatedPrior(unittest.TestCase):
    """Pre-check 11c: pricing from a regime whose sigma_run sits at or below its
    own binomial floor UNDER-prices the lottery. That is the one direction the
    gate exists to avoid, so it must be loud on the record (2026-08-20, after
    pusht-dp-rebuilt was adopted at 1.292/df=4)."""

    @staticmethod
    def _run(rid, p, n=500):
        rng = random.Random(sum(map(ord, rid)))
        s = [rng.random() < p for _ in range(n)]
        return EvalRun(run_id=rid, fmt="lerobot", sr=100.0 * sum(s) / n,
                       n_eval=n, seed=1000, successes=s,
                       task_groups=["pusht"], episodes=list(range(103)))

    def test_floor_dominated_regime_warns(self):
        d = gate(self._run("i", 0.17), self._run("c", 0.19),
                 GateConfig(n_tasks=1, n_boot=50, comparison_level="retrain",
                            regime="pusht-dp-rebuilt"),
                 now="2026-08-17T00:00:00Z")
        w = [x for x in d.record.get("warnings", [])
             if "ANTI-CONSERVATIVE PRIOR" in x]
        self.assertTrue(w, "floor-dominated prior priced with no warning")
        self.assertIn("1.292", w[0])
        self.assertIn("1.682", w[0])

    def test_healthy_regime_does_not_warn(self):
        d = gate(self._run("i", 0.779), self._run("c", 0.80),
                 GateConfig(n_tasks=1, n_boot=50, comparison_level="retrain",
                            regime="aloha-transfer-cube-act"),
                 now="2026-08-17T00:00:00Z")
        self.assertFalse([x for x in d.record.get("warnings", [])
                          if "ANTI-CONSERVATIVE PRIOR" in x])

    def test_explicit_sigma_bypasses_the_warning(self):
        # an operator who passes a sigma is not relying on the row
        d = gate(self._run("i", 0.17), self._run("c", 0.19),
                 GateConfig(n_tasks=1, n_boot=50, comparison_level="retrain",
                            regime="pusht-dp-rebuilt", sigma_run_pp=3.0),
                 now="2026-08-17T00:00:00Z")
        self.assertFalse([x for x in d.record.get("warnings", [])
                          if "ANTI-CONSERVATIVE PRIOR" in x])
