"""End-to-end shipgate CLI behaviour: check verdict exit codes, config-file /
flag override precedence, ledger round-trips, replay modes + REPLAY_REPORT.md,
verify tamper detection.

Fixtures are DETERMINISTIC block constructions (no rng in the verdict path):
e.g. incumbent = 220 successes then failures, candidate = 240 then failures,
same eval seed -> b=0, c=20, delta exactly +5.0 pp — so every asserted exit
code is a consequence of the frozen decision rule, not of a lucky draw.
"""

import contextlib
import io as _io
import json
import os
import random
import shutil
import tempfile
import unittest

from orbit_eval import __version__
from orbit_eval.gate import cli, records


def _run(argv):
    out, err = _io.StringIO(), _io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as e:  # argparse --version / usage errors
            code = e.code if isinstance(e.code, int) else 0
    return code, out.getvalue(), err.getvalue()


def _blocks(n, k):
    """k successes then n-k failures."""
    return [True] * k + [False] * (n - k)


def _eval_info(successes, seed=1000, task_group="pusht"):
    n = len(successes)
    obj = {
        "per_task": [{"task_group": task_group, "task_id": 0, "metrics": {
            "successes": [bool(s) for s in successes]}}],
        "overall": {"pc_success": 100.0 * sum(successes) / n,
                    "n_episodes": n},
    }
    if seed is not None:
        obj["seed"] = seed
    return obj


class _TmpMixin:
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="shipgate_cli_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, obj):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as fh:
            json.dump(obj, fh)
        return p

    def _pair(self, inc_succ, cand_succ, seed=1000, seed_b="same",
              task_group="pusht"):
        a = self._write("inc_eval_info.json",
                        _eval_info(inc_succ, seed=seed, task_group=task_group))
        b = self._write("cand_eval_info.json",
                        _eval_info(cand_succ,
                                   seed=(seed if seed_b == "same" else seed_b),
                                   task_group=task_group))
        return a, b


class TestCheckVerdictExits(_TmpMixin, unittest.TestCase):
    def test_ship_exit_0(self):
        # +15 pp, b=0 c=60: p_ship ~ 2^-60, delta_effective 15 >= 2
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260))
        code, out, err = _run(["check", a, b, "--n-boot", "150"])
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("VERDICT: SHIP", out)
        self.assertIn("CRN-PAIRED", out)
        self.assertIn("CAVEAT", out)          # pairing-validated + async-envs

    def test_hold_exit_1_on_significant_drop(self):
        a, b = self._pair(_blocks(400, 260), _blocks(400, 200))
        code, out, _ = _run(["check", a, b, "--n-boot", "150"])
        self.assertEqual(code, 1)
        self.assertIn("VERDICT: HOLD", out)

    def test_collect_more_exit_3_with_floor_caveat(self):
        # delta 0 with maximal discordance: p_d = 1 -> n_required 15457 >> 400
        inc = [i % 2 == 0 for i in range(400)]
        cand = [i % 2 == 1 for i in range(400)]
        a, b = self._pair(inc, cand)
        code, out, _ = _run(["check", a, b, "--n-boot", "150"])
        self.assertEqual(code, 3)
        self.assertIn("VERDICT: COLLECT-MORE", out)
        self.assertIn("coin flip", out)       # CAVEAT_FLOOR_2PP (|delta|<2)

    def test_unresolvable_exit_4_over_budget(self):
        inc = [i % 2 == 0 for i in range(400)]
        cand = [i % 2 == 1 for i in range(400)]
        a, b = self._pair(inc, cand)
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--max-budget", "300"])
        self.assertEqual(code, 4)
        self.assertIn("VERDICT: UNRESOLVABLE", out)

    def test_invalid_exit_2_on_seed_mismatch(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260),
                          seed=5, seed_b=6)
        code, out, _ = _run(["check", a, b, "--n-boot", "150"])
        self.assertEqual(code, 2)
        self.assertIn("VERDICT: INVALID", out)

    def test_invalid_exit_2_on_mixed_nondefault_seed(self):
        # one seed recorded as 42, the other unrecorded: almost certainly
        # different seeds -> INVALID, not a silent unpaired fallback
        a, b = self._pair(_blocks(200, 100), _blocks(200, 120),
                          seed=42, seed_b=None)
        code, out, _ = _run(["check", a, b, "--n-boot", "150"])
        self.assertEqual(code, 2)
        self.assertIn("VERDICT: INVALID", out)

    def test_unreadable_input_exit_2(self):
        a = self._write("ok_eval_info.json", _eval_info(_blocks(50, 25)))
        code, _, err = _run(["check", a,
                             os.path.join(self.tmp, "no_such.json")])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_json_prints_verifiable_record(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260))
        code, out, _ = _run(["check", a, b, "--n-boot", "150", "--json"])
        self.assertEqual(code, 0)
        rec = json.loads(out)
        self.assertEqual(rec["verdict"], "SHIP")
        self.assertEqual(rec["engine_version"], __version__)
        self.assertEqual(len(rec["record_sha256"]), 64)
        # the printed record must verify against its own embedded hash
        self.assertEqual(rec["record_sha256"], records.record_hash(rec))
        st = rec["statistics"]
        self.assertTrue(st["paired"])
        self.assertEqual((st["b"], st["c"]), (0, 60))
        self.assertEqual(st["delta_hat_pp"], 15.0)


class TestCheckPairingModes(_TmpMixin, unittest.TestCase):
    def test_unrecorded_seeds_degrade_unless_assume_crn(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260),
                          seed=None, seed_b=None)
        code, out, _ = _run(["check", a, b, "--n-boot", "150", "--json"])
        rec = json.loads(out)
        self.assertFalse(rec["statistics"]["paired"])
        self.assertTrue(any("UNPAIRED" in w for w in rec["warnings"]))
        code2, out2, _ = _run(["check", a, b, "--n-boot", "150",
                               "--assume-crn", "--json"])
        rec2 = json.loads(out2)
        self.assertTrue(rec2["statistics"]["paired"])
        self.assertTrue(any("ASSUMED" in w for w in rec2["warnings"]))

    def test_libero_degrades_without_sequential_eval(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260),
                          task_group="libero_object")
        code, out, _ = _run(["check", a, b, "--n-boot", "150", "--json"])
        rec = json.loads(out)
        self.assertFalse(rec["statistics"]["paired"])
        self.assertTrue(any("autoreset" in c for c in rec["caveats"]))

    def test_libero_pairs_with_sequential_eval(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260),
                          task_group="libero_object")
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--sequential-eval", "--json"])
        rec = json.loads(out)
        self.assertTrue(rec["statistics"]["paired"])
        self.assertEqual(code, 0)


class TestCheckCurse(_TmpMixin, unittest.TestCase):
    def test_prior_correction_turns_ship_into_collect_more(self):
        # +5.0 pp exactly (b=0, c=20): SHIP as 'final', but J=4 prior takes
        # 4.0 pp off the effective delta -> 1.0 < min_effect 2.0
        a, b = self._pair(_blocks(400, 220), _blocks(400, 240))
        code, _, _ = _run(["check", a, b, "--n-boot", "150"])
        self.assertEqual(code, 0)
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--selection", "max-over-checkpoints",
                             "--n-checkpoints", "4", "--json"])
        self.assertEqual(code, 3)
        rec = json.loads(out)
        self.assertEqual(rec["verdict"], "COLLECT-MORE")
        self.assertEqual(rec["statistics"]["curse_method"],
                         "atlas-prior-scaled")
        self.assertEqual(rec["statistics"]["curse_correction_pp"], 4.0)
        self.assertTrue(any(a_.startswith("ASSUMPTION: ")
                            for a_ in rec["assumptions"]))

    def test_history_bootstrap_curse(self):
        a, b = self._pair(_blocks(400, 220), _blocks(400, 240))
        hist = [self._write("ckpt%d_eval_info.json" % j,
                            _eval_info(_blocks(400, 180 + 20 * j)))
                for j in range(4)]
        argv = ["check", a, b, "--n-boot", "150",
                "--selection", "max-over-checkpoints", "--json"]
        for h in hist:
            argv += ["--history", h]
        code, out, _ = _run(argv)
        rec = json.loads(out)
        self.assertEqual(rec["statistics"]["curse_method"],
                         "history-bootstrap")
        self.assertGreaterEqual(rec["statistics"]["curse_correction_pp"], 0.0)
        self.assertIn(rec["verdict"],
                      ("SHIP", "HOLD", "COLLECT-MORE", "UNRESOLVABLE"))

    def test_history_with_selection_final_is_config_misuse(self):
        a, b = self._pair(_blocks(400, 220), _blocks(400, 240))
        h = self._write("ckpt_eval_info.json", _eval_info(_blocks(400, 200)))
        code, _, err = _run(["check", a, b, "--history", h])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_max_over_checkpoints_needs_j_or_history(self):
        a, b = self._pair(_blocks(400, 220), _blocks(400, 240))
        code, _, err = _run(["check", a, b,
                             "--selection", "max-over-checkpoints"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_history_seed_mismatch_is_invalid_not_error(self):
        a, b = self._pair(_blocks(400, 220), _blocks(400, 240))
        h = self._write("ckpt_eval_info.json",
                        _eval_info(_blocks(400, 200), seed=77))
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--selection", "max-over-checkpoints",
                             "--history", h])
        self.assertEqual(code, 2)
        self.assertIn("VERDICT: INVALID", out)


class TestCheckPerTask(_TmpMixin, unittest.TestCase):
    def _regression_pair(self):
        # 4 blocks of 100. Block 0: candidate drops 80 -> 50 (b=30, c=0,
        # -30 pp, p ~ 2^-30). Blocks 1-3: candidate gains 40 -> 80 each.
        # Overall +22.5 pp and highly significant — but the broken task
        # must HOLD the release.
        inc = _blocks(100, 80) + 3 * _blocks(100, 40)
        cand = _blocks(100, 50) + 3 * _blocks(100, 80)
        return self._pair(inc, cand)

    def test_regression_flag_holds_despite_overall_win(self):
        a, b = self._regression_pair()
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--n-tasks", "4"])
        self.assertEqual(code, 1)
        self.assertIn("VERDICT: HOLD", out)
        self.assertIn("REGRESSION", out)

    def test_task_blocks_csv_and_labels(self):
        a, b = self._regression_pair()
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--task-blocks", "100,100,100,100",
                             "--task-labels", "lift,push,stack,open",
                             "--json"])
        self.assertEqual(code, 1)
        rec = json.loads(out)
        rows = rec["statistics"]["per_task"]
        self.assertEqual([r["task"] for r in rows],
                         ["lift", "push", "stack", "open"])
        self.assertTrue(rows[0]["regression"])
        self.assertFalse(any(r["regression"] for r in rows[1:]))

    def test_bad_task_blocks_csv_exits_2(self):
        a, b = self._regression_pair()
        code, _, err = _run(["check", a, b, "--task-blocks", "100,x"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_task_blocks_not_tiling_is_invalid(self):
        a, b = self._regression_pair()
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--task-blocks", "100,100"])
        self.assertEqual(code, 2)
        self.assertIn("VERDICT: INVALID", out)


class TestCheckConfigAndLedger(_TmpMixin, unittest.TestCase):
    def test_config_file_applies_and_flags_override(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260))
        cfgp = self._write("gate_config.json",
                           {"min_effect_pp": 50.0, "n_boot": 150})
        # config file alone: +15 pp < 50 pp min effect -> COLLECT-MORE
        code, _, _ = _run(["check", a, b, "--config", cfgp])
        self.assertEqual(code, 3)
        # explicit flag overrides the file -> SHIP
        code, _, _ = _run(["check", a, b, "--config", cfgp,
                           "--min-effect-pp", "2.0"])
        self.assertEqual(code, 0)

    def test_unknown_config_key_exits_2(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260))
        cfgp = self._write("gate_config.json", {"alhpa": 0.05})
        code, _, err = _run(["check", a, b, "--config", cfgp])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_invalid_config_value_exits_2(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260))
        code, _, err = _run(["check", a, b, "--alpha", "0.9"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_ledger_append_verify_and_tamper(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260))
        ledger = os.path.join(self.tmp, "gate_ledger.jsonl")
        code, _, _ = _run(["check", a, b, "--n-boot", "150",
                           "--ledger", ledger])
        self.assertEqual(code, 0)
        code, _, _ = _run(["check", b, a, "--n-boot", "150",
                           "--ledger", ledger])          # reversed -> HOLD
        self.assertEqual(code, 1)

        code, out, _ = _run(["verify", ledger, "--json"])
        self.assertEqual(code, 0)
        rep = json.loads(out)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["n_records"], 2)

        with open(ledger) as fh:
            lines = fh.readlines()
        lines[0] = lines[0].replace('"verdict":"SHIP"', '"verdict":"HOLD"')
        with open(ledger, "w") as fh:
            fh.writelines(lines)
        code, out, _ = _run(["verify", ledger, "--json"])
        self.assertEqual(code, 1)
        rep = json.loads(out)
        self.assertFalse(rep["ok"])
        self.assertEqual([bad["line"] for bad in rep["bad"]], [1])

    def test_verify_missing_path_exits_2(self):
        code, _, err = _run(["verify",
                             os.path.join(self.tmp, "no_such.jsonl")])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)


def _corpus_line(run_id, task, set_hash, train_seed, successes,
                 wave="w1", suite="pusht"):
    """One corpus per-episode JSONL line, with the string-typed ints and
    '0'/'1' successes string the real corpus files use."""
    return json.dumps({
        "run_id": run_id, "wave": wave, "policy_class": "smolvla",
        "suite": suite, "task": task, "k_episodes": "22",
        "train_seed": str(train_seed), "set_hash": set_hash,
        "n_eval": len(successes),
        "sr_atlas": 100.0 * sum(successes) / len(successes),
        "eval_seed": 1000, "steps": "20000",
        "successes": "".join("1" if s else "0" for s in successes),
    })


class TestCheckCorpusInput(_TmpMixin, unittest.TestCase):
    """The product contract says `check` accepts the corpus per-episode
    JSONL, not only eval_info/model_result — real-shape corpus lines
    ('0'/'1' successes strings, string-typed ints) must gate head-to-head."""

    def _write_lines(self, name, lines):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return p

    def _two_run_corpus(self):
        inc = [True] * 30 + [False] * 30          # 50 %
        cand = [True] * 45 + [False] * 15         # 75 %: b=0, c=15
        return self._write_lines("corpus.jsonl", [
            _corpus_line("w1/inc", "task0", "aaa111", 0, inc),
            _corpus_line("w1/cand", "task0", "aaa111", 1, cand),
        ])

    def test_check_two_corpus_runs_by_id(self):
        p = self._two_run_corpus()
        code, out, err = _run(["check", p, p,
                               "--incumbent-id", "w1/inc",
                               "--candidate-id", "w1/cand",
                               "--n-boot", "150"])
        self.assertEqual(code, 0, msg=out + err)     # +25 pp, p ~ 2^-15
        self.assertIn("SHIP", out)
        self.assertIn("CRN-PAIRED", out)

    def test_check_multi_run_corpus_without_id_exits_2(self):
        p = self._two_run_corpus()
        code, _, err = _run(["check", p, p, "--n-boot", "150"])
        self.assertEqual(code, 2)
        self.assertIn("--incumbent-id", err)

    def test_check_unknown_run_id_exits_2(self):
        p = self._two_run_corpus()
        code, _, err = _run(["check", p, p,
                             "--incumbent-id", "w1/inc",
                             "--candidate-id", "w1/nope"])
        self.assertEqual(code, 2)
        self.assertIn("w1/nope", err)

    def test_check_single_line_corpus_files_without_ids(self):
        inc = [True] * 30 + [False] * 30
        cand = [True] * 45 + [False] * 15
        pa = self._write_lines("inc.jsonl",
                               [_corpus_line("w1/inc", "task0", "aaa111", 0,
                                             inc)])
        pb = self._write_lines("cand.jsonl",
                               [_corpus_line("w1/cand", "task0", "aaa111", 1,
                                             cand)])
        code, out, err = _run(["check", pa, pb, "--n-boot", "150"])
        self.assertEqual(code, 0, msg=out + err)

    def test_check_corpus_history_ids_bootstrap_curse(self):
        inc = [True] * 30 + [False] * 30
        cand = [True] * 45 + [False] * 15
        ck = [True] * 40 + [False] * 20
        p = self._write_lines("corpus.jsonl", [
            _corpus_line("w1/inc", "task0", "aaa111", 0, inc),
            _corpus_line("w1/cand", "task0", "bbb222", 1, cand),
            _corpus_line("w1/ck", "task0", "bbb222", 1, ck),
        ])
        code, out, err = _run(["check", p, p,
                               "--incumbent-id", "w1/inc",
                               "--candidate-id", "w1/cand",
                               "--selection", "max-over-checkpoints",
                               "--history-id", "w1/cand",
                               "--history-id", "w1/ck",
                               "--n-boot", "150", "--json"])
        rec = json.loads(out)
        self.assertEqual(rec["statistics"]["curse_method"],
                         "history-bootstrap")
        # pool includes the candidate's own eval: no wrong-pool warning
        self.assertFalse(any("does not include the candidate" in w
                             for w in rec["warnings"]))

    def test_check_orbit_model_result_without_successes_is_invalid(self):
        """ORBIT model_result.json parses but carries no per-episode
        successes: the gate must refuse with an auditable INVALID (the gate
        is a paired per-episode instrument), not crash or silently degrade."""
        a = self._write("model_result_a.json",
                        {"mask_id": "wave/a", "sr": 40.0, "n_eval": 500,
                         "seed": 1000, "set_hash": "aaa111",
                         "final_step": 100000, "design_steps": 100000})
        b = self._write("model_result_b.json",
                        {"mask_id": "wave/b", "sr": 45.0, "n_eval": 500,
                         "seed": 1000, "set_hash": "bbb222",
                         "final_step": 100000, "design_steps": 100000})
        code, out, _ = _run(["check", a, b])
        self.assertEqual(code, 2)
        self.assertIn("INVALID", out)
        self.assertIn("successes missing", out)

    def test_check_cross_stack_declared_is_invalid(self):
        a, b = self._pair(_blocks(400, 200), _blocks(400, 260))
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--cross-stack"])
        self.assertEqual(code, 2)
        self.assertIn("INVALID", out)
        self.assertIn("harness drift", out)
        self.assertIn("Cross-stack comparisons are invalid", out)


class TestReplayCLI(_TmpMixin, unittest.TestCase):
    def _corpus(self, name="corpus.jsonl", with_bad_line=False):
        rng = random.Random(11)
        lines = []
        # cell A (task0): 4 seed replicates on set aaa111 -> 6 null pairs
        # (12 gates), and a >=3-run cell for curse mode (J=3).
        for ts in range(4):
            succ = [rng.random() < 0.6 for _ in range(60)]
            lines.append(_corpus_line("w1/null%02d" % ts, "task0", "aaa111",
                                      ts, succ))
        # cell B (task1): two different training sets -> 1 effect pair.
        succ_b = [rng.random() < 0.6 for _ in range(60)]
        succ_c = [rng.random() < 0.85 for _ in range(60)]
        lines.append(_corpus_line("w1/eff_b", "task1", "bbb222", 0, succ_b))
        lines.append(_corpus_line("w1/eff_c", "task1", "ccc333", 0, succ_c))
        if with_bad_line:
            bad = json.loads(_corpus_line("w1/bad", "task0", "ddd444", 9,
                                          [True] * 60))
            bad["n_eval"] = 61                     # len(successes) mismatch
            lines.append(json.dumps(bad))
        p = os.path.join(self.tmp, name)
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return p

    def test_replay_all_modes_report_and_ledger(self):
        corpus = self._corpus()
        outdir = os.path.join(self.tmp, "rep")
        ledger = os.path.join(self.tmp, "replay_ledger.jsonl")
        code, out, err = _run(["replay", corpus, "--n-boot", "200",
                               "--out", outdir, "--ledger", ledger])
        self.assertEqual(code, 0, msg=out + err)
        report_path = os.path.join(outdir, "REPLAY_REPORT.md")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path) as fh:
            self.assertTrue(fh.read().strip())
        # every produced record must land in the ledger and verify:
        # 12 null gates + 2 curse records (1 cell) + 2 effect gates = 16
        code, vout, _ = _run(["verify", ledger, "--json"])
        self.assertEqual(code, 0)
        rep = json.loads(vout)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["n_records"], 16)

    def test_replay_json_structure(self):
        corpus = self._corpus(with_bad_line=True)
        code, out, _ = _run(["replay", corpus, "--n-boot", "200", "--json"])
        self.assertEqual(code, 0)
        res = json.loads(out)
        self.assertNotIn("report", res)
        self.assertEqual(res["meta"]["n_runs"], 6)
        self.assertEqual(res["meta"]["n_errors"], 1)
        null = res["null"]
        self.assertEqual(null["n_pairs"], 6)
        self.assertEqual(null["n_gates"], 12)
        self.assertEqual(null["n_false_ship"],
                         null["verdict_counts"].get("SHIP", 0))
        if null["n_gates"]:
            lo, hi = null["wilson_ci"]
            self.assertLessEqual(lo, null["false_ship_rate"])
            self.assertGreaterEqual(hi, null["false_ship_rate"])
        self.assertEqual(res["curse"]["n_cells"], 1)
        self.assertEqual(res["curse"]["cells"][0]["j"], 3)
        self.assertEqual(res["effect"]["n_pairs"], 1)
        self.assertEqual(res["effect"]["n_gates"], 2)

    def test_replay_single_mode(self):
        corpus = self._corpus()
        code, out, _ = _run(["replay", corpus, "--mode", "null",
                             "--n-boot", "200", "--json"])
        self.assertEqual(code, 0)
        res = json.loads(out)
        self.assertIsNotNone(res["null"])
        self.assertIsNone(res["curse"])
        self.assertIsNone(res["effect"])

    def test_replay_unreadable_path_exits_2(self):
        code, _, err = _run(["replay",
                             os.path.join(self.tmp, "no_such.jsonl")])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_replay_no_parseable_runs_exits_2(self):
        p = os.path.join(self.tmp, "empty.jsonl")
        with open(p, "w") as fh:
            fh.write("\n\n")
        code, _, err = _run(["replay", p])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)


class TestParserSurface(unittest.TestCase):
    def test_version(self):
        code, out, _ = _run(["--version"])
        self.assertEqual(code, 0)
        self.assertIn("orbit-shipgate %s" % __version__, out)

    def test_no_args_usage_error(self):
        code, _, err = _run([])
        self.assertEqual(code, 2)

    def test_check_requires_two_runs(self):
        code, _, err = _run(["check", "only_one.json"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
