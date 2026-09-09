"""replay.py tests on synthetic corpus JSONL written to tempfiles that
reproduce the EXACT shapes of the real corpus files
(experiments/forecast/*_per_episode.jsonl): successes as '0'/'1' character
strings, k_episodes/train_seed/steps stored as STRINGS, extra harvester
fields (eval_seed_provenance, match_method, max_rewards) present. Covers the
reader normalization, the null/effect pair finders, all three replay modes
(including a null pair engineered to FALSE-SHIP one way and HOLD the other),
report rendering and the ledger round-trip.

Everything asserted here comes from the frozen Ship-Gate v0 spec — no
coupling to engine internals beyond its public constants.
"""

import json
import math
import os
import random
import shutil
import tempfile
import unittest

from orbit_eval import __version__ as ENGINE_VERSION
from orbit_eval import stats
from orbit_eval.gate import engine as gengine
from orbit_eval.gate import replay as greplay
from orbit_eval.gate.records import GateConfig, verify_ledger

NOW = "2026-08-08T00:00:00Z"
VERDICTS = {"SHIP", "HOLD", "INVALID", "COLLECT-MORE", "UNRESOLVABLE"}

# small bootstrap keeps the suite fast; every record stays deterministic
CFG = dict(n_boot=150)

REAL_CORPUS_DIR = os.path.join(
    os.path.expanduser("~"), "Desktop", "Orbit-Research",
    "Orbit-Robotic-Research", "experiments", "forecast")


def succ_str(successes):
    return "".join("1" if s else "0" for s in successes)


def bern(n, p, seed):
    rng = random.Random(seed)
    return [rng.random() < p for _ in range(n)]


def block(k, n):
    """k successes then n-k failures — engineered CRN-style patterns."""
    return [True] * k + [False] * (n - k)


def corpus_dict(run_id, successes, train_seed=0, set_hash="feedbeefcafe0123",
                suite="pusht", task="task0", wave="w1", policy="DP",
                eval_seed=1000, steps=100000, **over):
    """One corpus line dict in the exact real-file shape (string-typed ints,
    harvester extras)."""
    n = len(successes)
    d = {"run_id": run_id, "wave": wave, "policy_class": policy,
         "suite": suite, "task": task, "k_episodes": "22",
         "train_seed": str(train_seed), "set_hash": set_hash,
         "steps": str(steps), "n_eval": n,
         "sr_atlas": 100.0 * sum(1 for s in successes if s) / n,
         "sr_delta_pp": 0.0,
         "eval_seed": eval_seed,
         "eval_seed_provenance": "lerobot_default_1000_implicit",
         "successes": succ_str(successes),
         "source": "%s/eval_info.json" % run_id,
         "match_method": "wave+mask"}
    d.update(over)
    return d


def parse_dict(d, **kw):
    return greplay.parse_corpus_line(json.dumps(d), **kw)


def write_jsonl(path, dicts):
    with open(path, "w") as fh:
        for d in dicts:
            fh.write(json.dumps(d) + "\n")


# ------------------------------------------------------------------ reader

class TestCorpusReader(unittest.TestCase):
    def test_string_successes_and_string_ints(self):
        cr = parse_dict(corpus_dict("w1/r0", [False, True, False, True],
                                    train_seed=3, steps=100000))
        self.assertEqual(cr.run.successes, [False, True, False, True])
        self.assertEqual(cr.run.n_eval, 4)
        # corpus stores these as strings — must arrive as ints
        self.assertEqual(cr.k_episodes, 22)
        self.assertEqual(cr.train_seed, 3)
        self.assertEqual(cr.steps, 100000)
        self.assertEqual(cr.run.final_step, 100000)
        # the corpus never records the designed budget: no fabricated pass
        self.assertIsNone(cr.run.design_steps)
        self.assertEqual(cr.run.seed, 1000)
        self.assertEqual(cr.run.fmt, "corpus")
        self.assertEqual(cr.run.task_groups, ["pusht"])
        self.assertEqual(cr.run.policy_class, "DP")
        self.assertEqual(cr.run.set_hash, "feedbeefcafe0123")
        self.assertAlmostEqual(cr.run.sr, 50.0)
        self.assertEqual(cr.source, "w1/r0/eval_info.json")
        self.assertEqual(cr.raw["match_method"], "wave+mask")

    def test_list_successes(self):
        d = corpus_dict("w1/r1", [True] * 4)
        d["successes"] = [0, 1, True, False]
        cr = parse_dict(d)
        self.assertEqual(cr.run.successes, [False, True, True, False])

    def test_bad_successes_rejected(self):
        d = corpus_dict("w1/r1", [True] * 4)
        d["successes"] = "01x1"
        with self.assertRaises(ValueError):
            parse_dict(d)
        d["successes"] = [0, 1, 2, 1]
        with self.assertRaises(ValueError):
            parse_dict(d)

    def test_aliases(self):
        d = corpus_dict("w1/r2", [True, False], eval_seed=1000)
        del d["eval_seed"], d["suite"], d["k_episodes"], d["steps"]
        del d["policy_class"]
        d.update({"seed": 7, "task_group": "pusht", "k": "44",
                  "final_step": "5000", "policy": "smolvla_ft"})
        cr = parse_dict(d)
        self.assertEqual(cr.run.seed, 7)
        self.assertEqual(cr.suite, "pusht")
        self.assertEqual(cr.run.task_groups, ["pusht"])
        self.assertEqual(cr.k_episodes, 44)
        self.assertEqual(cr.steps, 5000)
        self.assertEqual(cr.policy_class, "smolvla_ft")
        self.assertEqual(cr.run.policy_class, "smolvla_ft")

    def test_policy_class_default_unknown(self):
        d = corpus_dict("w1/r3", [True, False])
        del d["policy_class"]
        cr = parse_dict(d)
        self.assertEqual(cr.policy_class, "unknown")

    def test_sr_preference_atlas_then_recomputed_then_sr(self):
        d = corpus_dict("w1/r4", [True, False, False, False])  # sr_atlas 25.0
        d.update({"sr_recomputed": 60.0, "sr": 70.0})
        self.assertAlmostEqual(parse_dict(d).run.sr, 25.0)
        del d["sr_atlas"]
        self.assertAlmostEqual(parse_dict(d).run.sr, 60.0)
        del d["sr_recomputed"]
        self.assertAlmostEqual(parse_dict(d).run.sr, 70.0)

    def test_len_mismatch_is_error_entry_not_raise(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        good = corpus_dict("w1/good", [True, False])
        bad = corpus_dict("w1/bad", [True, False, True])
        bad["n_eval"] = 5   # len(successes)=3 != 5
        path = os.path.join(tmp, "c.jsonl")
        write_jsonl(path, [good, bad])
        cruns, errors = greplay.load_corpus([path])
        self.assertEqual([cr.run.run_id for cr in cruns], ["w1/good"])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["path"], path)
        self.assertEqual(errors[0]["line"], 2)
        self.assertIn("n_eval", errors[0]["error"])

    def test_blank_lines_skipped(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "c.jsonl")
        with open(path, "w") as fh:
            fh.write("\n")
            fh.write(json.dumps(corpus_dict("w1/g", [True])) + "\n")
            fh.write("   \n")
        cruns, errors = greplay.load_corpus([path])
        self.assertEqual(len(cruns), 1)
        self.assertEqual(errors, [])

    def test_run_id_required(self):
        d = corpus_dict("w1/r5", [True])
        del d["run_id"]
        with self.assertRaises(ValueError):
            parse_dict(d)

    def test_malformed_json_raises_with_context(self):
        with self.assertRaises(ValueError) as cm:
            greplay.parse_corpus_line("{not json", path="f.jsonl", lineno=3)
        self.assertIn("f.jsonl:3", str(cm.exception))

    def test_n_tasks_hint(self):
        self.assertEqual(
            parse_dict(corpus_dict("w/r", [True], task="all10")).n_tasks_hint,
            10)
        self.assertIsNone(
            parse_dict(corpus_dict("w/r", [True], task="task0")).n_tasks_hint)

    def test_n_eval_from_successes_when_absent(self):
        d = corpus_dict("w1/r6", [True, False, True])
        del d["n_eval"]
        self.assertEqual(parse_dict(d).run.n_eval, 3)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(REAL_CORPUS_DIR, "per_episode.jsonl")),
        "real ORBIT corpus not present on this machine")
    def test_real_corpus_parses_clean(self):
        """The reader must swallow the founder's real files with ZERO parse
        errors and find null pairs in them (86 replicate groups measured)."""
        paths = [os.path.join(REAL_CORPUS_DIR, f)
                 for f in ("per_episode.jsonl", "pi05_per_episode.jsonl")]
        cruns, errors = greplay.load_corpus(paths)
        self.assertEqual(errors, [])
        self.assertEqual(len(cruns), 254)
        self.assertTrue(all(cr.run.successes is not None for cr in cruns))
        self.assertTrue(len(greplay.find_null_pairs(cruns)) > 0)


# ------------------------------------------------------------------ pair finders

class TestPairFinders(unittest.TestCase):
    def _replicates(self):
        base = bern(20, 0.5, 1)
        return [corpus_dict("g/r%d" % i, base, train_seed=i,
                            set_hash="aaaa000011112222")
                for i in range(3)]

    def test_null_pairs_matched(self):
        cruns = [parse_dict(d) for d in self._replicates()]
        pairs = greplay.find_null_pairs(cruns)
        self.assertEqual([(a.run.run_id, b.run.run_id) for a, b in pairs],
                         [("g/r0", "g/r1"), ("g/r0", "g/r2"),
                          ("g/r1", "g/r2")])

    def test_differing_cell_fields_excluded(self):
        a = corpus_dict("g/a", bern(20, 0.5, 2), train_seed=0,
                        set_hash="aaaa000011112222")
        for over in (dict(successes=succ_str(bern(24, 0.5, 3)), n_eval=24),
                     dict(eval_seed=2000),
                     dict(set_hash="bbbb000011112222")):
            b = corpus_dict("g/b", bern(20, 0.5, 4), train_seed=1,
                            set_hash="aaaa000011112222")
            b.update(over)
            pairs = greplay.find_null_pairs([parse_dict(a), parse_dict(b)])
            self.assertEqual(pairs, [], "should exclude on %s" % over)

    def test_same_or_unrecorded_train_seed_excluded(self):
        a = corpus_dict("g/a", bern(20, 0.5, 5), train_seed=0,
                        set_hash="cccc000011112222")
        b = corpus_dict("g/b", bern(20, 0.5, 6), train_seed=0,
                        set_hash="cccc000011112222")
        self.assertEqual(
            greplay.find_null_pairs([parse_dict(a), parse_dict(b)]), [])
        c = dict(b, run_id="g/c")
        del c["train_seed"]   # unrecorded: cannot certify a seed replicate
        self.assertEqual(
            greplay.find_null_pairs([parse_dict(a), parse_dict(c)]), [])

    def test_deterministic_ordering_input_order_independent(self):
        dicts = self._replicates()
        fwd = greplay.find_null_pairs([parse_dict(d) for d in dicts])
        rev = greplay.find_null_pairs([parse_dict(d) for d in dicts[::-1]])
        self.assertEqual(
            [(a.run.run_id, b.run.run_id) for a, b in fwd],
            [(a.run.run_id, b.run.run_id) for a, b in rev])

    def test_effect_representative_is_min_train_seed(self):
        base = bern(200, 0.6, 7)
        other = bern(200, 0.62, 8)
        dicts = [
            corpus_dict("e/a_s1", base, train_seed=1, set_hash="aaaa" * 4),
            corpus_dict("e/a_s0", base, train_seed=0, set_hash="aaaa" * 4),
            corpus_dict("e/b_s5", other, train_seed=5, set_hash="bbbb" * 4),
        ]
        cruns = [parse_dict(d) for d in dicts]
        res = greplay.replay_effects(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 1)          # one rep per set_hash
        self.assertEqual(res["n_gates"], 2)
        gated_ids = {p["a"] for p in res["pairs"]} | \
                    {p["b"] for p in res["pairs"]}
        self.assertEqual(gated_ids, {"e/a_s0", "e/b_s5"})   # min train_seed


# ------------------------------------------------------------------ null mode

class TestNullReplay(unittest.TestCase):
    def _null_group(self, n_runs=6, n=200, p=0.6):
        """Seed replicates as iid Bernoulli(p) draws — non-LIBERO suite,
        recorded eval_seed 1000 on every run, so gates are CRN-paired."""
        return [parse_dict(corpus_dict("null/s%d" % i, bern(n, p, 100 + i),
                                       train_seed=i, set_hash="dddd" * 4))
                for i in range(n_runs)]

    def test_structure_and_false_ship_rate(self):
        config = GateConfig(**CFG)
        res = greplay.replay_null_pairs(self._null_group(), config, now=NOW)
        self.assertEqual(res["n_pairs"], 15)
        self.assertEqual(res["n_gates"], 30)
        self.assertEqual(len(res["records"]), 30)
        self.assertEqual(sum(res["verdict_counts"].values()), 30)
        self.assertTrue(set(res["verdict_counts"]) <= VERDICTS)
        for rec in res["records"]:
            self.assertIn(rec["verdict"], VERDICTS)
            self.assertTrue(rec["statistics"]["paired"])
        # every SHIP on a null pair is a false ship, by construction
        self.assertEqual(res["n_false_ship"],
                         res["verdict_counts"].get("SHIP", 0))
        self.assertEqual(res["alpha"], config.alpha)
        rate = res["false_ship_rate"]
        self.assertAlmostEqual(rate, res["n_false_ship"] / 30.0)
        # the whole product claim: false-ship rate compatible with alpha
        bound = config.alpha + 3 * math.sqrt(
            config.alpha * (1 - config.alpha) / res["n_gates"])
        self.assertLessEqual(rate, bound)
        # wilson_ci is stats.wilson verbatim and brackets the rate
        self.assertEqual(res["wilson_ci"],
                         list(stats.wilson(res["n_false_ship"],
                                           res["n_gates"])))
        self.assertLessEqual(res["wilson_ci"][0], rate)
        self.assertGreaterEqual(res["wilson_ci"][1], rate)
        # gaps sorted by |delta| descending
        deltas = [abs(g["delta_hat_pp"]) for g in res["top_null_gaps"]]
        self.assertEqual(deltas, sorted(deltas, reverse=True))
        self.assertEqual(len(res["top_null_gaps"]), 30)

    def test_engineered_false_ship_and_hold(self):
        """A null pair with a +15 pp gap (pure 'retraining noise' by
        set_hash) must FALSE-SHIP in one direction and HOLD in the other —
        both counted, the SHIP flagged as false."""
        n = 200
        inc = block(100, n)                # 50 %
        cand = block(130, n)               # 65 %, superset: b=0, c=30
        cruns = [parse_dict(corpus_dict("eng/s0", inc, train_seed=0,
                                        set_hash="eeee" * 4)),
                 parse_dict(corpus_dict("eng/s1", cand, train_seed=1,
                                        set_hash="eeee" * 4))]
        res = greplay.replay_null_pairs(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 1)
        self.assertEqual(res["n_gates"], 2)
        self.assertEqual(res["verdict_counts"].get("SHIP"), 1)
        self.assertEqual(res["verdict_counts"].get("HOLD"), 1)
        self.assertEqual(res["n_false_ship"], 1)
        self.assertEqual(res["false_ship_rate"], 0.5)
        by_verdict = {r["verdict"]: r for r in res["records"]}
        self.assertAlmostEqual(
            by_verdict["SHIP"]["statistics"]["delta_hat_pp"], 15.0)
        self.assertAlmostEqual(
            by_verdict["HOLD"]["statistics"]["delta_hat_pp"], -15.0)
        # selection forced to 'final': no curse correction on null gates
        self.assertEqual(
            by_verdict["SHIP"]["statistics"]["curse_correction_pp"], 0.0)
        self.assertEqual(
            by_verdict["SHIP"]["gate_config"]["selection"], "final")

    def test_dependence_disclosure_cluster_ci_and_calibration(self):
        """The null result must expose the dependence structure the Wilson
        CI ignores: replicate-group count, pair-level Wilson CI, cluster
        bootstrap CI over groups, per-regime retraining noise
        (sigma_from_gaps) beside atlas references, and the seeded
        exact-null calibration control of the paired path."""
        config = GateConfig(**CFG)
        res = greplay.replay_null_pairs(self._null_group(), config, now=NOW)
        self.assertEqual(res["n_groups"], 1)     # one replicate group
        rate = res["false_ship_rate"]
        # pair-level: a pair false-ships iff either direction ships
        self.assertLessEqual(res["n_pairs_shipped"], res["n_pairs"])
        self.assertGreaterEqual(res["n_false_ship"], res["n_pairs_shipped"])
        plo, phi = res["pair_wilson_ci"]
        self.assertLessEqual(plo, res["n_pairs_shipped"] / res["n_pairs"])
        self.assertGreaterEqual(phi, res["n_pairs_shipped"] / res["n_pairs"])
        # cluster bootstrap over ONE group is degenerate: CI = [rate, rate]
        self.assertEqual(res["cluster_ci"], [rate, rate])
        # deterministic under the seeded config
        res2 = greplay.replay_null_pairs(self._null_group(), config, now=NOW)
        self.assertEqual(res["cluster_ci"], res2["cluster_ci"])
        self.assertEqual(res["calibration"], res2["calibration"])
        # retraining noise per env/policy regime, atlas refs attached
        rn = res["retrain_noise_pp"]
        self.assertEqual(list(rn), ["pusht/DP"])
        self.assertEqual(rn["pusht/DP"]["n_pairs"], res["n_pairs"])
        self.assertGreaterEqual(rn["pusht/DP"]["sigma_run_pp"], 0.0)
        self.assertEqual([r["regime"] for r in rn["pusht/DP"]["atlas_refs"]],
                         ["pusht-dp", "pusht-dp-rebuilt"])
        # exact-null calibration: seeded, n_boot sims, level at/below alpha
        cal = res["calibration"]
        self.assertEqual(cal["n_profiles"], res["n_pairs"])
        self.assertEqual(cal["n_sims"], config.n_boot)
        self.assertLessEqual(cal["ship_rule_rate"], cal["sig_rate"])
        # exact McNemar is conservative; allow MC slack on 150 sims
        self.assertLessEqual(cal["sig_rate"],
                             config.alpha
                             + 3 * math.sqrt(config.alpha * (1 - config.alpha)
                                             / cal["n_sims"]))
        # no cross-wave / cross-epoch pairs in this single-wave fixture
        self.assertEqual(res["n_cross_wave_pairs"], 0)
        self.assertEqual(res["n_cross_epoch_excluded"], 0)

    def test_identical_replicates_do_not_ship(self):
        """Bit-identical successes (sigma_0 = 0, the measured SmolVLA-ft
        floor): b = c = 0 must never SHIP."""
        s = bern(200, 0.7, 55)
        cruns = [parse_dict(corpus_dict("bit/s%d" % i, s, train_seed=i,
                                        set_hash="ffff" * 4))
                 for i in range(2)]
        res = greplay.replay_null_pairs(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_false_ship"], 0)
        for rec in res["records"]:
            self.assertIn(rec["verdict"], {"COLLECT-MORE", "UNRESOLVABLE"})


# ------------------------------------------------------ stack epochs

class TestStackEpochs(unittest.TestCase):
    """The wave->stack-epoch key: cross-rebuild pairs are never formed
    (atlas CAVEATS['cross-stack'], measured -18.2 pp gym_pusht shift);
    cross-wave pairs within an epoch carry a recorded ASSUMPTION."""

    def _pusht_pair(self, wave_a, wave_b, same_set=True):
        base = bern(60, 0.5, 41)
        other = bern(60, 0.55, 42)
        return [parse_dict(corpus_dict("a/s0", base, train_seed=0,
                                       set_hash="1111" * 4, wave=wave_a)),
                parse_dict(corpus_dict("b/s1", other, train_seed=1,
                                       set_hash="1111" * 4 if same_set
                                       else "2222" * 4, wave=wave_b))]

    def test_epoch_map_and_unknown_waves(self):
        cr = parse_dict(corpus_dict("x/r", [True], wave="datamodel_pusht"))
        self.assertEqual(greplay.stack_epoch(cr), "pusht-stack-old")
        cr = parse_dict(corpus_dict("x/r", [True], wave="phase4_pusht"))
        self.assertEqual(greplay.stack_epoch(cr), "pusht-stack-rebuilt")
        # unknown pusht wave: own epoch — never silently poolable
        cr = parse_dict(corpus_dict("x/r", [True], wave="mystery"))
        self.assertEqual(greplay.stack_epoch(cr), "pusht-wave:mystery")
        # non-pusht suites: one epoch (no rebuild measured)
        cr = parse_dict(corpus_dict("x/r", [True], suite="libero_object",
                                    wave="phase2_libero"))
        self.assertEqual(greplay.stack_epoch(cr), "no-rebuild-measured")

    def test_cross_epoch_null_pair_never_forms_and_is_counted(self):
        cruns = self._pusht_pair("datamodel_pusht", "phase4_pusht")
        self.assertEqual(greplay.find_null_pairs(cruns), [])
        res = greplay.replay_null_pairs(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 0)
        self.assertEqual(res["n_cross_epoch_excluded"], 1)

    def test_unknown_pusht_waves_do_not_pool(self):
        cruns = self._pusht_pair("w1", "w2")
        self.assertEqual(greplay.find_null_pairs(cruns), [])
        res = greplay.replay_null_pairs(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_cross_epoch_excluded"], 1)

    def test_cross_wave_same_epoch_null_pair_carries_assumption(self):
        cruns = self._pusht_pair("datamodel_pusht", "phase1c_pusht")
        res = greplay.replay_null_pairs(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 1)
        self.assertEqual(res["n_cross_wave_pairs"], 1)
        self.assertEqual(res["n_cross_epoch_excluded"], 0)
        for rec in res["records"]:
            cw = [a for a in rec["assumptions"]
                  if a.startswith("ASSUMPTION: cross-wave pair")]
            self.assertEqual(len(cw), 1)
            self.assertIn("pusht-stack-old", cw[0])
        # same-wave pairs carry no such assumption
        same = greplay.replay_null_pairs(
            self._pusht_pair("phase1c_pusht", "phase1c_pusht"),
            GateConfig(**CFG), now=NOW)
        for rec in same["records"]:
            self.assertFalse(any("cross-wave" in a
                                 for a in rec["assumptions"]))

    def test_libero_cross_wave_allowed_with_assumption(self):
        base = bern(60, 0.5, 43)
        cruns = [parse_dict(corpus_dict("a/s0", base, train_seed=0,
                                        suite="libero_object",
                                        wave="phase2_libero",
                                        set_hash="3333" * 4)),
                 parse_dict(corpus_dict("b/s1", bern(60, 0.5, 44),
                                        train_seed=1, suite="libero_object",
                                        wave="phase4_libero",
                                        set_hash="3333" * 4))]
        res = greplay.replay_null_pairs(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 1)
        self.assertEqual(res["n_cross_wave_pairs"], 1)
        self.assertTrue(any("cross-wave pair" in a
                            for a in res["records"][0]["assumptions"]))

    def test_cross_epoch_effect_pair_never_forms_and_is_counted(self):
        cruns = self._pusht_pair("datamodel_pusht", "phase1d_pusht",
                                 same_set=False)
        res = greplay.replay_effects(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 0)
        self.assertEqual(res["n_gates"], 0)
        self.assertEqual(res["n_cross_epoch_excluded"], 1)

    def test_cross_wave_same_epoch_effect_pair_carries_assumption(self):
        cruns = self._pusht_pair("datamodel_pusht", "phase1c_wave2",
                                 same_set=False)
        res = greplay.replay_effects(cruns, GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 1)
        self.assertEqual(res["n_cross_wave_pairs"], 1)
        self.assertEqual(res["n_cross_epoch_excluded"], 0)
        for rec in res["records"]:
            self.assertTrue(any("cross-wave pair" in a
                                for a in rec["assumptions"]))

    def test_report_renders_epoch_and_dependence_disclosures(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "c.jsonl")
        dicts = [corpus_dict("null/s%d" % i, bern(200, 0.6, 500 + i),
                             train_seed=i, set_hash="7777" * 4)
                 for i in range(3)]
        write_jsonl(path, dicts)
        res = greplay.run_replay([path], config=GateConfig(**CFG), now=NOW)
        for needle in ("dependence structure", "cluster bootstrap",
                       "exact-null calibration control",
                       "Retraining noise measured from THESE null pairs",
                       "cross-epoch replicate pairings excluded",
                       "cross-epoch pairings excluded"):
            self.assertIn(needle, res["report"], needle)


# ------------------------------------------------------ curse + effect modes

class TestCurseAndEffectReplay(unittest.TestCase):
    def _curse_cell(self):
        """4 runs in one cell; run c3 is the noise max (75 %). Distinct
        set_hash per run (different training sets do not break the cell
        key)."""
        n = 200
        rng = random.Random(9)
        dicts = []
        for i, k in enumerate((110, 120, 124, 150)):    # 55, 60, 62, 75 %
            s = block(k, n)
            rng.shuffle(s)
            dicts.append(corpus_dict("cell/c%d" % i, s, train_seed=i,
                                     set_hash=("%04x" % (4369 * (i + 1))) * 4,
                                     wave="pi", policy="pi05"))
        return [parse_dict(d) for d in dicts]

    def test_curse_planted_cell(self):
        config = GateConfig(**CFG)
        res = greplay.replay_curse(self._curse_cell(), config, now=NOW)
        self.assertEqual(res["n_cells"], 1)
        cell = res["cells"][0]
        # median-SR incumbent (lower median of 55,60,62,75 -> 60 -> c1),
        # candidate = argmax SR of the remaining pool -> c3, J = 3
        self.assertEqual(cell["incumbent"], "cell/c1")
        self.assertEqual(cell["candidate"], "cell/c3")
        self.assertEqual(cell["j"], 3)
        self.assertAlmostEqual(cell["sr_incumbent_pp"], 60.0)
        self.assertAlmostEqual(cell["sr_candidate_pp"], 75.0)
        self.assertAlmostEqual(
            cell["correction_prior_pp"],
            gengine.curse_from_prior(3, config.curse_prior_pp,
                                     config.curse_prior_j))
        self.assertGreaterEqual(cell["correction_history_pp"], 0.0)
        self.assertIn(cell["verdict_uncorrected"], VERDICTS)
        self.assertIn(cell["verdict_corrected"], VERDICTS)
        rec_unc, rec_cor = cell["records"]
        self.assertEqual(rec_unc["gate_config"]["selection"], "final")
        self.assertEqual(rec_unc["statistics"]["curse_method"], "none")
        self.assertEqual(rec_cor["gate_config"]["selection"],
                         "max-over-checkpoints")
        self.assertEqual(rec_cor["statistics"]["curse_method"],
                         "history-bootstrap")
        self.assertAlmostEqual(rec_cor["statistics"]["curse_correction_pp"],
                               cell["correction_history_pp"])
        # correction reduces the effective delta the ship rule sees
        self.assertAlmostEqual(
            rec_cor["statistics"]["delta_effective_pp"],
            rec_cor["statistics"]["delta_hat_pp"]
            - rec_cor["statistics"]["curse_correction_pp"])

    def test_curse_needs_three_runs(self):
        res = greplay.replay_curse(self._curse_cell()[:2], GateConfig(**CFG),
                                   now=NOW)
        self.assertEqual(res["n_cells"], 0)
        self.assertEqual(res["cells"], [])

    def test_effect_big_gap_resolves(self):
        n = 200
        dicts = [corpus_dict("eff/a", block(100, n), train_seed=0,
                             set_hash="a1a1" * 4),
                 corpus_dict("eff/b", block(130, n), train_seed=0,
                             set_hash="b2b2" * 4)]
        res = greplay.replay_effects([parse_dict(d) for d in dicts],
                                     GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 1)
        self.assertEqual(res["n_gates"], 2)
        self.assertEqual(res["resolved_rate"], 1.0)   # SHIP one way, HOLD back
        self.assertEqual(res["verdict_counts"].get("SHIP"), 1)
        self.assertEqual(res["verdict_counts"].get("HOLD"), 1)
        self.assertEqual(res["episodes_needed"],
                         {"min": None, "median": None, "max": None})

    def test_effect_small_gap_collects_more_with_sizing(self):
        """+1 pp at high discordance: unresolvable at n=200, and the sizing
        must say how many MORE episodes the paired test needs."""
        n = 200
        inc = [i < 120 for i in range(n)]
        cand = list(inc)
        for i in range(100, 120):
            cand[i] = False        # b = 20 incumbent-only
        for i in range(120, 142):
            cand[i] = True         # c = 22 candidate-only -> +1 pp
        dicts = [corpus_dict("sm/a", inc, train_seed=0, set_hash="c3c3" * 4),
                 corpus_dict("sm/b", cand, train_seed=0, set_hash="d4d4" * 4)]
        res = greplay.replay_effects([parse_dict(d) for d in dicts],
                                     GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_gates"], 2)
        self.assertEqual(res["verdict_counts"].get("COLLECT-MORE"), 2)
        self.assertEqual(res["resolved_rate"], 0.0)
        for p in res["pairs"]:
            self.assertIsInstance(p["episodes_needed"], int)
            self.assertGreater(p["episodes_needed"], 0)
        self.assertIsInstance(res["episodes_needed"]["min"], int)
        self.assertGreaterEqual(res["episodes_needed"]["max"],
                                res["episodes_needed"]["min"])
        # |delta_effective| < 2 pp: the corpus-measured floor caveat attaches
        for rec in res["records"]:
            self.assertIn(gengine.CAVEAT_FLOOR_2PP, rec["caveats"])

    def test_effect_pairs_sorted_by_abs_delta(self):
        n = 100
        dicts = [corpus_dict("s/a", block(50, n), train_seed=0,
                             set_hash="aaaa" * 4),
                 corpus_dict("s/b", block(60, n), train_seed=0,
                             set_hash="bbbb" * 4),
                 corpus_dict("s/c", block(52, n), train_seed=0,
                             set_hash="cccc" * 4)]
        res = greplay.replay_effects([parse_dict(d) for d in dicts],
                                     GateConfig(**CFG), now=NOW)
        self.assertEqual(res["n_pairs"], 3)
        deltas = [abs(p["delta_hat_pp"]) for p in res["pairs"]]
        self.assertEqual(deltas, sorted(deltas, reverse=True))


# ------------------------------------------------------ report + ledger

class TestReportAndLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.path = os.path.join(self.tmp, "corpus.jsonl")
        n = 200
        dicts = []
        # null group: 4 seed replicates (iid Bernoulli, non-LIBERO)
        for i in range(4):
            dicts.append(corpus_dict("null/s%d" % i, bern(n, 0.6, 200 + i),
                                     train_seed=i, set_hash="9999" * 4))
        # curse + effect cell: 4 runs, distinct set_hash, one noise max
        for i, k in enumerate((110, 120, 124, 150)):
            dicts.append(corpus_dict("cell/c%d" % i, block(k, n),
                                     train_seed=i,
                                     set_hash=("%04x" % (257 * (i + 1))) * 4,
                                     wave="pi", policy="pi05"))
        self.n_dicts = len(dicts)
        write_jsonl(self.path, dicts)

    def _libero_fixture(self):
        path = os.path.join(self.tmp, "libero.jsonl")
        n = 200
        write_jsonl(path, [
            corpus_dict("lib/s0", block(100, n), train_seed=0,
                        set_hash="e5e5" * 4, suite="libero_object",
                        policy="smolvla_ft"),
            corpus_dict("lib/s1", block(130, n), train_seed=1,
                        set_hash="e5e5" * 4, suite="libero_object",
                        policy="smolvla_ft"),
        ])
        return path

    def test_run_replay_report_and_ledger_roundtrip(self):
        config = GateConfig(**CFG)
        ledger = os.path.join(self.tmp, "ledger.jsonl")
        res = greplay.run_replay([self.path], config=config,
                                 out_dir=self.tmp, ledger_path=ledger,
                                 now=NOW)
        # meta reflects the inputs, verbatim timestamp, engine version
        self.assertEqual(res["meta"]["n_runs"], self.n_dicts)
        self.assertEqual(res["meta"]["n_errors"], 0)
        self.assertEqual(res["meta"]["created"], NOW)
        self.assertEqual(res["meta"]["engine_version"], ENGINE_VERSION)
        self.assertEqual(res["meta"]["config_sha256"], config.sha256())
        # report written and identical to the returned text
        rp = os.path.join(self.tmp, "REPLAY_REPORT.md")
        self.assertEqual(res["report_path"], rp)
        with open(rp) as fh:
            self.assertEqual(fh.read(), res["report"])
        # numbers in the report are computed from results, not asserted
        self.assertIn("%.4f" % res["null"]["false_ship_rate"], res["report"])
        self.assertIn(config.sha256()[:16], res["report"])
        self.assertIn(ENGINE_VERSION, res["report"])
        # every produced record landed in the ledger and re-verifies
        n_records = (len(res["null"]["records"])
                     + sum(len(c["records"]) for c in res["curse"]["cells"])
                     + len(res["effect"]["records"]))
        v = verify_ledger(ledger)
        self.assertTrue(v["ok"], v["bad"])
        self.assertEqual(v["n_records"], n_records)
        self.assertEqual(v["n_ok"], n_records)
        # all three modes ran on this fixture
        self.assertGreater(res["null"]["n_gates"], 0)
        self.assertGreater(res["curse"]["n_cells"], 0)
        self.assertGreater(res["effect"]["n_gates"], 0)
        # one shared timestamp on every record
        for rec in greplay._all_records(res):
            self.assertEqual(rec["created"], NOW)

    def test_modes_subset(self):
        res = greplay.run_replay([self.path], config=GateConfig(**CFG),
                                 modes=("null",), now=NOW)
        self.assertIsNotNone(res["null"])
        self.assertIsNone(res["curse"])
        self.assertIsNone(res["effect"])
        self.assertIsNone(res["report_path"])
        self.assertIn("_mode not run_", res["report"])

    def test_changing_alpha_changes_report(self):
        r1 = greplay.run_replay([self.path], config=GateConfig(**CFG),
                                now=NOW)
        r2 = greplay.run_replay([self.path],
                                config=GateConfig(alpha=0.01, **CFG),
                                now=NOW)
        self.assertNotEqual(r1["report"], r2["report"])
        self.assertNotEqual(r1["meta"]["config_sha256"],
                            r2["meta"]["config_sha256"])

    def test_parse_errors_surface_in_meta_and_report(self):
        bad_path = os.path.join(self.tmp, "bad.jsonl")
        with open(bad_path, "w") as fh:
            fh.write(json.dumps(corpus_dict("ok/r", [True, False])) + "\n")
            fh.write("{broken\n")
        res = greplay.run_replay([bad_path], config=GateConfig(**CFG),
                                 now=NOW)
        self.assertEqual(res["meta"]["n_runs"], 1)
        self.assertEqual(res["meta"]["n_errors"], 1)
        self.assertIn("Parse errors:", res["report"])

    def test_libero_degrade_noted_unless_sequential(self):
        path = self._libero_fixture()
        res = greplay.run_replay([path], config=GateConfig(**CFG), now=NOW)
        self.assertIn("DEGRADED", res["report"])
        for rec in res["null"]["records"]:
            self.assertFalse(rec["statistics"]["paired"])
        # declaring sequential (non-vectorized) eval restores pairing
        res_seq = greplay.run_replay(
            [path], config=GateConfig(sequential_eval=True, **CFG), now=NOW)
        self.assertNotIn("DEGRADED", res_seq["report"])
        self.assertIn("sequential", res_seq["report"])
        for rec in res_seq["null"]["records"]:
            self.assertTrue(rec["statistics"]["paired"])

    def test_all10_hint_populates_per_task_table(self):
        """A multi-task 'all10' corpus run (curated88 shape: n_eval=200,
        10 contiguous 20-episode blocks) must gate with a 10-row per-task
        table via the n_tasks hint, and disclose the equal-blocks
        assumption."""
        path = os.path.join(self.tmp, "multi.jsonl")
        n = 200
        write_jsonl(path, [
            corpus_dict("m/s0", bern(n, 0.6, 300), train_seed=0,
                        set_hash="abab" * 4, task="all10"),
            corpus_dict("m/s1", bern(n, 0.6, 301), train_seed=1,
                        set_hash="abab" * 4, task="all10"),
        ])
        res = greplay.run_replay([path], config=GateConfig(**CFG),
                                 modes=("null",), now=NOW)
        self.assertEqual(res["null"]["n_gates"], 2)
        for rec in res["null"]["records"]:
            self.assertEqual(len(rec["statistics"]["per_task"]), 10)
            self.assertTrue(all(row["n"] == 20
                                for row in rec["statistics"]["per_task"]))
            self.assertTrue(any(a.startswith("ASSUMPTION:")
                                for a in rec["assumptions"]))
            # caller's config was replaced per-gate, not mutated
            self.assertEqual(rec["gate_config"]["n_tasks"], 10)

    def test_ledger_tamper_detected(self):
        ledger = os.path.join(self.tmp, "ledger.jsonl")
        greplay.run_replay([self.path], config=GateConfig(**CFG),
                           modes=("null",), ledger_path=ledger, now=NOW)
        with open(ledger) as fh:
            lines = fh.readlines()
        doctored = json.loads(lines[0])
        doctored["verdict"] = "SHIP"     # forge a promotion
        lines[0] = json.dumps(doctored, sort_keys=True,
                              separators=(",", ":")) + "\n"
        with open(ledger, "w") as fh:
            fh.writelines(lines)
        v = verify_ledger(ledger)
        self.assertFalse(v["ok"])
        self.assertEqual(v["bad"][0]["line"], 1)


if __name__ == "__main__":
    unittest.main()
