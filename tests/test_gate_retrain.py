"""Ship-Gate v0.1 "retrain-aware" mode (frozen math spec).

Motivation (measured, validation/REPLAY_REPORT.md): the v0 alpha bounds
eval-sampling error only — 62/386 (16%) false ships on the real seed-
replicate null pairs while the exact-null control sat at 4.0% <= alpha, so
the gap is SEMANTICS and v0.1 closes it with the measured atlas priors.

Covers: GateConfig v0.1 fields + validation, prior resolution order
(explicit > regime > none), the z-retrain-aware statistics against
hand-computed se_eval/se_total/mde_floor/retrains_needed values (e.g.
sigma_run=3.31 -> floor 2.4866*1.41421*3.31 = 11.64 pp), checkpoint-level
unchanged-verdict regression vs v0, schema v2 round-trip + v1 backward
ledger verify, replay's per-pair prior auto-resolution and BEFORE/AFTER
summary, and the CLI --level retrain end-to-end.
"""

import contextlib
import io as _io
import json
import math
import os
import shutil
import tempfile
import unittest

from orbit_eval import atlas, stats
from orbit_eval.gate import cli as gcli
from orbit_eval.gate import replay as greplay
from orbit_eval.gate.engine import (
    GateInputError, episodes_for_min_effect, gate, resolve_sigma_run)
from orbit_eval.gate.records import (
    GateConfig, build_record, canonical_json, record_hash, sha256_hex,
    verify_ledger)
from orbit_eval.io import EvalRun

NOW = "2026-08-08T12:00:00Z"

# One-sided sizing constants of the frozen spec (stats.norm_q, NOT
# power.py's two-sided 2.80).
Z_A = stats.norm_q(0.95)
Z_P = stats.norm_q(0.8)


def _cfg(**kw):
    kw.setdefault("n_boot", 150)
    return GateConfig(**kw)


def _mk(run_id, successes, seed=1000, groups=("pusht",), **kw):
    n = len(successes) if successes else None
    sr = kw.pop("sr", 100.0 * sum(map(bool, successes)) / n
                if successes else None)
    return EvalRun(run_id=run_id, fmt="lerobot", sr=sr, n_eval=n, seed=seed,
                   successes=successes, task_groups=list(groups or []), **kw)


def _bc_pair(n, b, c, concordant=0, **mkkw):
    inc_s = [True] * b + [False] * c + [True] * concordant \
        + [False] * (n - b - c - concordant)
    cand_s = [False] * b + [True] * c + [True] * concordant \
        + [False] * (n - b - c - concordant)
    return _mk("inc", inc_s, **mkkw), _mk("cand", cand_s, **mkkw)


# ------------------------------------------------------------------ config

class TestConfigV01(unittest.TestCase):
    def test_new_defaults(self):
        cfg = GateConfig()
        self.assertEqual(cfg.comparison_level, "checkpoint")
        self.assertIsNone(cfg.sigma_run_pp)
        self.assertIsNone(cfg.sigma_run_df)

    def test_round_trip_with_new_fields(self):
        cfg = GateConfig(comparison_level="retrain", sigma_run_pp=3.31,
                         sigma_run_df=6)
        back = GateConfig.from_dict(cfg.to_dict())
        self.assertEqual(back, cfg)
        self.assertEqual(back.sha256(), cfg.sha256())
        self.assertEqual(cfg.validate(), [])

    def test_validate_bad_level(self):
        errs = GateConfig(comparison_level="retrained").validate()
        self.assertTrue(any("comparison_level" in m for m in errs))
        self.assertEqual(len(errs), 1)

    def test_validate_negative_sigma(self):
        errs = GateConfig(comparison_level="retrain",
                          sigma_run_pp=-1.0).validate()
        self.assertTrue(any("sigma_run_pp" in m for m in errs))

    def test_validate_sigma_at_checkpoint_is_problem(self):
        errs = GateConfig(sigma_run_pp=3.31).validate()
        self.assertTrue(any("silently unused" in m for m in errs))
        # ... and setting the level clears it
        self.assertEqual(GateConfig(comparison_level="retrain",
                                    sigma_run_pp=3.31).validate(), [])


# --------------------------------------------------------- prior resolution

class TestPriorResolution(unittest.TestCase):
    def test_explicit_beats_regime(self):
        cfg = _cfg(comparison_level="retrain", sigma_run_pp=5.0,
                   regime="smolvla-ft")
        self.assertEqual(resolve_sigma_run(cfg), (5.0, "explicit"))
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        st = gate(inc, cand, cfg, now=NOW).record["statistics"]
        self.assertEqual(st["sigma_run_pp_used"], 5.0)
        self.assertEqual(st["sigma_run_source"], "explicit")

    def test_regime_resolves_from_atlas_including_pi05_ft(self):
        # pi05-ft updated 2026-08-09 by PI05_PAIRS_RESULTS: 2.02 (df=2) -> 9.74
        # (df=10) — the df=2 value was an optimistic low-df draw; the ladder is
        # not monotone. This test pins the MEASURED atlas values of record.
        for regime, sigma in (("pi05-ft", 9.74), ("smolvla-ft", 3.31),
                              ("pusht-dp", 2.09),
                              ("pusht-dp-rebuilt", 1.292)):   # adopted 2026-08-20
            cfg = _cfg(comparison_level="retrain", regime=regime)
            self.assertEqual(resolve_sigma_run(cfg),
                             (sigma, "atlas:%s" % regime))
        # the atlas entry itself: measured values + provenance quoted
        reg = atlas.get_regime("pi05-ft")
        self.assertEqual(reg["sigma_run"], 9.74)
        self.assertEqual(reg["df"], 10)
        self.assertEqual(reg["k"], 22)
        self.assertEqual(reg["env"], "libero")
        self.assertEqual(reg["policy"], "pi05-ft")
        self.assertIn("PI05_PAIRS_RESULTS", reg["provenance"])
        self.assertIn("[6.81, 17.10]", reg["provenance"])
        # the overturned df=2 value must remain disclosed in provenance
        self.assertIn("2.02", reg["provenance"])

    def test_none_runs_v0_stats_with_disclosure(self):
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        d = gate(inc, cand, _cfg(comparison_level="retrain"), now=NOW)
        st = d.record["statistics"]
        self.assertIsNone(st["sigma_run_pp_used"])
        self.assertIsNone(st["sigma_run_source"])
        self.assertIsNone(st["se_total_pp"])
        self.assertIsNone(st["mde_floor_pp"])
        self.assertEqual(st["method"], "mcnemar-exact")
        self.assertEqual(st["p_ship"], st["p_eval"])
        disc = [a for a in d.record["assumptions"]
                if "WITHOUT a sigma_run prior" in a]
        self.assertEqual(len(disc), 1)
        self.assertIn("16%", disc[0])
        # v0 statistics govern: same verdict as the checkpoint gate
        d0 = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, d0.verdict)
        self.assertEqual(st["p_ship"], d0.record["statistics"]["p_ship"])

    def test_unknown_regime_at_retrain_raises(self):
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        with self.assertRaises(GateInputError):
            gate(inc, cand, _cfg(comparison_level="retrain",
                                 regime="no-such-regime"), now=NOW)

    def test_checkpoint_never_resolves(self):
        self.assertEqual(resolve_sigma_run(_cfg(regime="smolvla-ft")),
                         (None, None))


# ------------------------------------------------------- retrain statistics

class TestRetrainStatistics(unittest.TestCase):
    def test_paired_se_total_and_governing_p_hand_computed(self):
        # b=0, c=30 at n=200: delta_hat +15 pp, p_d = 0.15.
        # se_eval = 100*sqrt(0.15/200) = 2.73861...; sigma_run 3.31 ->
        # se_total = sqrt(7.5 + 2*3.31^2) = sqrt(29.4122) = 5.42330...
        inc, cand = _bc_pair(200, b=0, c=30)
        d = gate(inc, cand, _cfg(comparison_level="retrain",
                                 sigma_run_pp=3.31), now=NOW)
        st = d.record["statistics"]
        self.assertEqual(st["method"], "z-retrain-aware")
        se_eval = 100.0 * math.sqrt(0.15 / 200)
        se_total = math.sqrt(se_eval ** 2 + 2 * 3.31 ** 2)
        self.assertAlmostEqual(st["se_eval_pp"], se_eval, places=12)
        self.assertAlmostEqual(st["se_total_pp"], se_total, places=12)
        self.assertAlmostEqual(st["se_total_pp"], 5.42330, places=4)
        # governing p-values: one-sided normal both directions
        self.assertAlmostEqual(st["p_ship"],
                               1 - stats.Phi(15.0 / se_total), places=12)
        self.assertAlmostEqual(st["p_worse"],
                               stats.Phi(15.0 / se_total), places=12)
        # the exact eval-level McNemar p is still computed and recorded
        self.assertAlmostEqual(st["p_eval"], 0.5 ** 30, places=15)
        self.assertLess(st["p_eval"], st["p_ship"])
        # ship rule unchanged in FORM: still significant here -> SHIP
        self.assertEqual(d.verdict, "SHIP")

    def test_hold_via_governing_p_worse(self):
        inc, cand = _bc_pair(200, b=30, c=0)   # delta -15 pp
        d = gate(inc, cand, _cfg(comparison_level="retrain",
                                 sigma_run_pp=3.31), now=NOW)
        self.assertEqual(d.verdict, "HOLD")
        self.assertLessEqual(d.record["statistics"]["p_worse"], 0.05)

    def test_unpaired_se_eval_hand_computed(self):
        # seeds unrecorded, no assume_crn -> unpaired degrade path.
        # inc 100/200, cand 130/200: se_eval =
        # 100*sqrt(.5*.5/200 + .65*.35/200) = 100*sqrt(0.0023875)
        inc_s = [True] * 100 + [False] * 100
        cand_s = [True] * 130 + [False] * 70
        inc = _mk("inc", inc_s, seed=None)
        cand = _mk("cand", cand_s, seed=None)
        d = gate(inc, cand, _cfg(comparison_level="retrain",
                                 sigma_run_pp=3.31), now=NOW)
        st = d.record["statistics"]
        self.assertFalse(st["paired"])
        self.assertEqual(st["method"], "z-retrain-aware")
        se_eval = 100.0 * math.sqrt(0.5 * 0.5 / 200 + 0.65 * 0.35 / 200)
        self.assertAlmostEqual(st["se_eval_pp"], se_eval, places=12)
        self.assertAlmostEqual(st["se_total_pp"],
                               math.sqrt(se_eval ** 2 + 2 * 3.31 ** 2),
                               places=12)

    def test_mde_floor_spec_worked_example(self):
        # sigma_run=3.31, alpha=.05 one-sided, power .8 ->
        # floor = 2.4866 * 1.41421 * 3.31 = 11.64 pp (frozen spec example)
        inc, cand = _bc_pair(200, b=0, c=30)
        st = gate(inc, cand, _cfg(comparison_level="retrain",
                                  sigma_run_pp=3.31),
                  now=NOW).record["statistics"]
        self.assertAlmostEqual(st["mde_floor_pp"],
                               (Z_A + Z_P) * math.sqrt(2.0) * 3.31,
                               places=12)
        self.assertAlmostEqual(st["mde_floor_pp"], 11.64, places=2)

    def test_retrains_needed_ceil(self):
        # min_effect 2.0 <= floor 11.64 -> retrains unit:
        # ceil(2*((z_a+z_p)*3.31/2)^2) = ceil(33.868...) = 34 per side
        inc, cand = _bc_pair(200, b=20, c=22)   # +1 pp, not significant
        d = gate(inc, cand, _cfg(comparison_level="retrain",
                                 sigma_run_pp=3.31), now=NOW)
        st = d.record["statistics"]
        self.assertEqual(d.verdict, "COLLECT-MORE")
        self.assertEqual(st["collect_more_unit"], "retrains")
        want = math.ceil(2 * ((Z_A + Z_P) * 3.31 / 2.0) ** 2)
        self.assertEqual(want, 34)
        self.assertEqual(st["retrains_needed"], 34)
        self.assertIsNone(st["episodes_needed"])
        self.assertIsNone(st["n_required"])
        self.assertTrue(any("retrains-needed" in r for r in d.reasons))
        self.assertTrue(any("ignores the eval-sampling term" in a
                            for a in d.record["assumptions"]))
        # max_budget is an EPISODE budget: never UNRESOLVABLE on this path
        d2 = gate(inc, cand, _cfg(comparison_level="retrain",
                                  sigma_run_pp=3.31, max_budget=300),
                  now=NOW)
        self.assertEqual(d2.verdict, "COLLECT-MORE")

    def test_episodes_path_when_min_effect_clears_floor(self):
        # sigma_run 0.5 -> floor 1.758 < min_effect 2.0 -> episodes unit:
        # n_required = ceil(p_d*10000 / ((min_effect/(z_a+z_p))^2 -
        # 2*sigma^2)), p_d = 42/200
        inc, cand = _bc_pair(200, b=20, c=22)
        d = gate(inc, cand, _cfg(comparison_level="retrain",
                                 sigma_run_pp=0.5), now=NOW)
        st = d.record["statistics"]
        self.assertEqual(st["collect_more_unit"], "episodes")
        self.assertIsNone(st["retrains_needed"])
        denom = (2.0 / (Z_A + Z_P)) ** 2 - 2 * 0.5 ** 2
        want = math.ceil(0.21 * 10000.0 / denom)
        self.assertEqual(st["n_required"], want)
        self.assertEqual(st["episodes_needed"], want - 200)
        self.assertEqual(d.verdict, "COLLECT-MORE")
        # the retrain price is visible: n_required exceeds the eval-only one
        self.assertGreater(st["n_required"],
                           episodes_for_min_effect(0.21, 2.0))

    def test_episodes_path_unresolvable_over_budget(self):
        inc, cand = _bc_pair(200, b=20, c=22)
        d = gate(inc, cand, _cfg(comparison_level="retrain",
                                 sigma_run_pp=0.5, max_budget=300), now=NOW)
        self.assertEqual(d.verdict, "UNRESOLVABLE")

    def test_per_task_regression_stays_eval_level_with_disclosure(self):
        # block 0 broken (b=20), blocks 1-3 candidate wins: HOLD both levels
        inc_s = ([True] * 20 + [False] * 80) + [False] * 300
        cand_s = [False] * 100 + ([True] * 15 + [False] * 85) * 3
        inc, cand = _mk("inc", inc_s), _mk("cand", cand_s)
        d = gate(inc, cand, _cfg(comparison_level="retrain",
                                 sigma_run_pp=3.31, n_tasks=4), now=NOW)
        self.assertEqual(d.verdict, "HOLD")
        rows = d.record["statistics"]["per_task"]
        self.assertTrue(rows[0]["regression"])
        # per-task p stays the exact eval-level McNemar tail
        self.assertAlmostEqual(rows[0]["p_worse"], 0.5 ** 20, places=12)
        disc = [a for a in d.record["assumptions"]
                if "per-task sigma_run is unmeasured" in a]
        self.assertEqual(len(disc), 1)
        d0 = gate(inc, cand, _cfg(n_tasks=4), now=NOW)
        self.assertFalse(any("per-task sigma_run is unmeasured" in a
                             for a in d0.record["assumptions"]))


# --------------------------------------------- checkpoint-level regression

class TestCheckpointUnchanged(unittest.TestCase):
    """comparison_level='checkpoint' (the default) must preserve v0
    behavior everywhere except the new unconditional disclosure string."""

    def test_ship_fixture_matches_v0_expectations(self):
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        d = gate(inc, cand, _cfg(), now=NOW)
        st = d.record["statistics"]
        self.assertEqual(d.verdict, "SHIP")
        self.assertEqual(st["method"], "mcnemar-exact")
        self.assertEqual(st["p_ship"], stats.mcnemar_one_sided(25, 5))
        self.assertEqual(st["p_eval"], st["p_ship"])
        self.assertIsNone(st["sigma_run_pp_used"])
        self.assertIsNone(st["sigma_run_source"])
        self.assertIsNone(st["se_eval_pp"])
        self.assertIsNone(st["se_total_pp"])
        self.assertIsNone(st["mde_floor_pp"])
        self.assertEqual(st["collect_more_unit"], "episodes")
        self.assertIsNone(st["retrains_needed"])
        self.assertEqual(st["n_required"],
                         episodes_for_min_effect(30 / 400, 2.0))

    def test_unpaired_method_string(self):
        inc, cand = _bc_pair(200, b=20, c=30, seed=None)
        d = gate(inc, cand, _cfg(), now=NOW)
        self.assertEqual(d.record["statistics"]["method"], "two-prop-z")

    def test_unconditional_checkpoint_disclosure_on_every_record(self):
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        for d in (gate(inc, cand, _cfg(), now=NOW),
                  gate(*_bc_pair(400, b=25, c=5), config=_cfg(), now=NOW)):
            disc = [a for a in d.record["assumptions"]
                    if "checkpoint-level comparison" in a]
            self.assertEqual(len(disc), 1)
            self.assertIn("16%", disc[0])
        # INVALID records carry it too (EVERY record, both levels)
        bad_inc, bad_cand = _bc_pair(400, b=5, c=25)
        bad_cand.seed = 7
        d = gate(bad_inc, bad_cand, _cfg(), now=NOW)
        self.assertEqual(d.verdict, "INVALID")
        self.assertTrue(any("checkpoint-level comparison" in a
                            for a in d.record["assumptions"]))


# --------------------------------------------------------------- schema

class TestSchemaV2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="shipgate_v2_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_v2_record_round_trip(self):
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        rec = gate(inc, cand, _cfg(comparison_level="retrain",
                                   sigma_run_pp=3.31, sigma_run_df=6),
                   now=NOW).record
        self.assertEqual(rec["schema_version"], 2)
        self.assertEqual(rec["gate_config"]["comparison_level"], "retrain")
        self.assertEqual(rec["gate_config"]["sigma_run_df"], 6)
        ledger = os.path.join(self.tmp, "v2.jsonl")
        from orbit_eval.gate.records import append_record
        append_record(rec, ledger)
        rep = verify_ledger(ledger)
        self.assertTrue(rep["ok"], rep["bad"])
        cfg2 = GateConfig.from_dict(rec["gate_config"])
        self.assertEqual(cfg2.sha256(), rec["gate_config_sha256"])

    def test_v1_record_still_verifies(self):
        """verify_ledger hashes whatever fields a record carries — a v1
        record (18-key config, v1 statistics, schema_version 1) written
        before the bump must keep verifying byte-for-byte."""
        v1_config = {
            "alpha": 0.05, "min_effect_pp": 2.0, "regression_alpha": 0.05,
            "regression_min_pp": 5.0, "selection": "final",
            "n_checkpoints": None, "curse_prior_pp": 4.0,
            "curse_prior_j": 4, "max_budget": None, "n_tasks": None,
            "task_blocks": None, "task_labels": None, "assume_crn": False,
            "sequential_eval": False, "regime": None, "n_boot": 4000,
            "boot_seed": 7, "power_target": 0.8}
        v1_rec = {
            "schema_version": 1, "engine_version": "0.1.0",
            "created": "2026-08-01T00:00:00Z", "verdict": "SHIP",
            "reasons": ["p_ship 5.19e-05 <= alpha 0.05"],
            "gate_config": v1_config,
            "gate_config_sha256": sha256_hex(canonical_json(v1_config)),
            "inputs": {"n_paired": 200},
            "statistics": {"paired": True, "b": 4, "c": 25,
                           "delta_hat_pp": 10.5, "p_ship": 5.19e-05},
            "caveats": [], "warnings": [], "assumptions": []}
        v1_rec["record_sha256"] = record_hash(v1_rec)
        ledger = os.path.join(self.tmp, "mixed.jsonl")
        with open(ledger, "w") as fh:
            fh.write(canonical_json(v1_rec) + "\n")
        # ... and a v2 record appended after the bump on the SAME ledger
        inc, cand = _bc_pair(400, b=5, c=25, concordant=150)
        from orbit_eval.gate.records import append_record
        append_record(gate(inc, cand, _cfg(), now=NOW).record, ledger)
        rep = verify_ledger(ledger)
        self.assertTrue(rep["ok"], rep["bad"])
        self.assertEqual(rep["n_records"], 2)

    def test_build_record_stamps_schema_2(self):
        rec = build_record("SHIP", GateConfig(), {}, {}, [], [], [],
                           [], NOW)
        self.assertEqual(rec["schema_version"], 2)


# --------------------------------------------------------------- replay

def _corpus_dict(run_id, successes, train_seed=0, wave="datamodel_pusht",
                 suite="pusht", policy="DP", set_hash="feed" * 4, k="22",
                 **over):
    n = len(successes)
    d = {"run_id": run_id, "wave": wave, "policy_class": policy,
         "suite": suite, "task": "task0", "k_episodes": k,
         "train_seed": str(train_seed), "set_hash": set_hash,
         "steps": "100000", "n_eval": n,
         "sr_atlas": 100.0 * sum(1 for s in successes if s) / n,
         "eval_seed": 1000,
         "successes": "".join("1" if s else "0" for s in successes)}
    d.update(over)
    return d


def _parse(d):
    return greplay.parse_corpus_line(json.dumps(d))


def _block(k, n):
    return [True] * k + [False] * (n - k)


class TestReplayPriorResolution(unittest.TestCase):
    def test_pusht_dp_by_stack_epoch(self):
        old = _parse(_corpus_dict("a/r", [True], wave="datamodel_pusht"))
        self.assertEqual(greplay.retrain_prior_regime(old), "pusht-dp")
        reb = _parse(_corpus_dict("a/r", [True], wave="phase4_pusht"))
        self.assertEqual(greplay.retrain_prior_regime(reb),
                         "pusht-dp-rebuilt")
        # unknown pusht epoch: neither prior is honest -> None
        unk = _parse(_corpus_dict("a/r", [True], wave="w1"))
        self.assertIsNone(greplay.retrain_prior_regime(unk))

    def test_libero_dp_nearest_k(self):
        for k, want in (("22", "libero-dp-k22"), ("30", "libero-dp-k31"),
                        ("31", "libero-dp-k31"), ("60", "libero-dp-k44")):
            cr = _parse(_corpus_dict("a/r", [True], suite="libero_object",
                                     wave="phase2", k=k))
            self.assertEqual(greplay.retrain_prior_regime(cr), want, k)

    def test_policy_map(self):
        sm = _parse(_corpus_dict("a/r", [True], suite="libero_object",
                                 wave="t2", policy="smolvla_ft"))
        self.assertEqual(greplay.retrain_prior_regime(sm), "smolvla-ft")
        pi = _parse(_corpus_dict("a/r", [True], suite="libero_object",
                                 wave="pi", policy="pi05"))
        self.assertEqual(greplay.retrain_prior_regime(pi), "pi05-ft")
        other = _parse(_corpus_dict("a/r", [True], policy="smolvla"))
        self.assertIsNone(greplay.retrain_prior_regime(other))


class TestReplayBeforeAfter(unittest.TestCase):
    def _pair(self):
        """Old-stack pusht/DP null pair with a +5 pp gap: false-ships under
        the v0 eval-only semantics (p = 2^-10), does NOT under the
        retrain-aware prior pusht-dp (2.09 pp): z = 5/sqrt(2.5+8.7362)."""
        n = 200
        return [_parse(_corpus_dict("eng/s0", _block(100, n), train_seed=0)),
                _parse(_corpus_dict("eng/s1", _block(110, n), train_seed=1))]

    def test_retrain_aware_stops_the_eval_only_false_ship(self):
        res = greplay.replay_null_pairs(self._pair(), _cfg(), now=NOW)
        self.assertEqual(res["comparison_level"], "retrain")
        self.assertEqual(res["n_pairs"], 1)
        self.assertEqual(res["n_gates"], 2)
        # BEFORE (v0 eval-only): one direction false-ships
        self.assertEqual(res["eval_only"]["n_false_ship"], 1)
        self.assertEqual(res["eval_only"]["verdict_counts"].get("SHIP"), 1)
        # AFTER (retrain-aware, primary): no false ship
        self.assertEqual(res["n_false_ship"], 0)
        self.assertNotIn("SHIP", res["verdict_counts"])
        # per-prior breakdown present on both sides
        self.assertIn("pusht-dp", res["by_prior"])
        self.assertIn("pusht-dp", res["eval_only"]["by_prior"])
        self.assertEqual(res["by_prior"]["pusht-dp"]["n_gates"], 2)
        # the primary records carry the resolved prior + method
        for rec in res["records"]:
            self.assertEqual(rec["gate_config"]["comparison_level"],
                             "retrain")
            self.assertEqual(rec["gate_config"]["regime"], "pusht-dp")
            st = rec["statistics"]
            self.assertEqual(st["method"], "z-retrain-aware")
            self.assertEqual(st["sigma_run_pp_used"], 2.09)
            self.assertEqual(st["sigma_run_source"], "atlas:pusht-dp")

    def test_no_prior_pairs_keep_v0_statistics_with_disclosure(self):
        n = 200
        cruns = [_parse(_corpus_dict("u/s0", _block(100, n), train_seed=0,
                                     wave="w1")),
                 _parse(_corpus_dict("u/s1", _block(130, n), train_seed=1,
                                     wave="w1"))]
        res = greplay.replay_null_pairs(cruns, _cfg(), now=NOW)
        # +15 pp: false-ships under BOTH semantics (no prior resolves)
        self.assertEqual(res["n_false_ship"], 1)
        self.assertEqual(res["eval_only"]["n_false_ship"], 1)
        self.assertIn("(no prior)", res["by_prior"])
        for rec in res["records"]:
            self.assertIsNone(rec["statistics"]["sigma_run_pp_used"])
            self.assertTrue(any("WITHOUT a sigma_run prior" in a
                                for a in rec["assumptions"]))

    def test_report_renders_before_after_table(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "c.jsonl")
        with open(path, "w") as fh:
            for d in (_corpus_dict("eng/s0", _block(100, 200), train_seed=0),
                      _corpus_dict("eng/s1", _block(110, 200), train_seed=1)):
                fh.write(json.dumps(d) + "\n")
        res = greplay.run_replay([path], config=_cfg(), now=NOW)
        for needle in ("BEFORE/AFTER", "eval-only", "retrain-aware",
                       "pusht-dp", "comparison_level: retrain",
                       # every v0 disclosure survives (frozen needles)
                       "dependence structure", "cluster bootstrap",
                       "exact-null calibration control",
                       "Retraining noise measured from THESE null pairs",
                       "cross-epoch replicate pairings excluded",
                       "cross-epoch pairings excluded"):
            self.assertIn(needle, res["report"], needle)

    def test_effect_mode_gates_at_retrain_level(self):
        n = 200
        cruns = [_parse(_corpus_dict("e/a", _block(100, n), train_seed=0,
                                     set_hash="aaaa" * 4)),
                 _parse(_corpus_dict("e/b", _block(130, n), train_seed=0,
                                     set_hash="bbbb" * 4))]
        res = greplay.replay_effects(cruns, _cfg(), now=NOW)
        self.assertEqual(res["comparison_level"], "retrain")
        self.assertEqual(res["n_gates"], 2)
        for rec in res["records"]:
            self.assertEqual(rec["gate_config"]["comparison_level"],
                             "retrain")
            self.assertEqual(rec["statistics"]["sigma_run_source"],
                             "atlas:pusht-dp")


# ------------------------------------------------------------------ CLI

def _run(argv):
    out, err = _io.StringIO(), _io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = gcli.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
    return code, out.getvalue(), err.getvalue()


class TestRetrainCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="shipgate_retrain_cli_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _eval_info(self, successes, seed=1000):
        n = len(successes)
        return {"per_task": [{"task_group": "pusht", "task_id": 0,
                              "metrics": {"successes": list(successes)}}],
                "overall": {"pc_success": 100.0 * sum(successes) / n,
                            "n_episodes": n},
                "seed": seed}

    def _pair(self):
        a = os.path.join(self.tmp, "inc_eval_info.json")
        b = os.path.join(self.tmp, "cand_eval_info.json")
        with open(a, "w") as fh:
            json.dump(self._eval_info(_block(220, 400)), fh)
        with open(b, "w") as fh:
            json.dump(self._eval_info(_block(240, 400)), fh)
        return a, b

    def test_level_retrain_sigma_run_end_to_end(self):
        # +5.0 pp (b=0, c=20): SHIP at checkpoint level, but at retrain
        # level with sigma_run 3.31 the governing z = 5/sqrt(1.25+21.9122)
        # is not significant -> COLLECT-MORE, priced in retrains (34/side).
        a, b = self._pair()
        code, _, _ = _run(["check", a, b, "--n-boot", "150"])
        self.assertEqual(code, 0)
        code, out, err = _run(["check", a, b, "--n-boot", "150",
                               "--level", "retrain", "--sigma-run", "3.31",
                               "--json"])
        self.assertEqual(code, 3, msg=out + err)
        rec = json.loads(out)
        self.assertEqual(rec["verdict"], "COLLECT-MORE")
        st = rec["statistics"]
        self.assertEqual(st["method"], "z-retrain-aware")
        self.assertEqual(st["sigma_run_source"], "explicit")
        self.assertEqual(st["sigma_run_pp_used"], 3.31)
        self.assertEqual(st["collect_more_unit"], "retrains")
        self.assertEqual(st["retrains_needed"], 34)
        self.assertIsNone(st["episodes_needed"])
        se_total = math.sqrt((100.0 * math.sqrt(0.05 / 400)) ** 2
                             + 2 * 3.31 ** 2)
        self.assertAlmostEqual(st["se_total_pp"], se_total, places=12)
        self.assertAlmostEqual(st["p_ship"],
                               1 - stats.Phi(5.0 / se_total), places=12)

    def test_level_retrain_regime_resolves_prior(self):
        # 2026-08-16: fixture data is pusht-labeled, so the regime must be
        # the env-consistent cell — pre-check 11 now refuses cross-env
        # pricing (the old smolvla-ft choice here was exactly that).
        a, b = self._pair()
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--level", "retrain",
                             "--regime", "pusht-dp-rebuilt", "--json"])
        rec = json.loads(out)
        self.assertEqual(rec["statistics"]["sigma_run_source"],
                         "atlas:pusht-dp-rebuilt")
        self.assertEqual(rec["statistics"]["sigma_run_pp_used"], 1.292)

    def test_sigma_run_at_checkpoint_level_exits_2(self):
        a, b = self._pair()
        code, _, err = _run(["check", a, b, "--sigma-run", "3.31"])
        self.assertEqual(code, 2)
        self.assertIn("silently unused", err)

    def test_text_output_discloses_retrain_mode(self):
        a, b = self._pair()
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--level", "retrain", "--sigma-run", "3.31"])
        self.assertEqual(code, 3)
        self.assertIn("retrain-aware", out)
        self.assertIn("RETRAINS per side", out)
        self.assertIn("mde_floor", out)

    def test_help_states_the_measurement(self):
        code, out, _ = _run(["check", "--help"])
        self.assertIn("16%", out)


if __name__ == "__main__":
    unittest.main()
