"""shipgate CLI v1 surface: the LeRobot training-output-dir adapter on
`check` (positional args and --history may be output/run dirs), `shipgate
demo` (compact DEMO.md + fresh verifiable ledger), and `shipgate seq` (the
anytime-valid sequential gate over a JSONL observation stream).

The seq tests import orbit_eval.gate.sequential, which is delivered by the
v1 sequential workstream: while the module is absent they SKIP cleanly
(HAVE_SEQ guard) and TestSeqUnavailable asserts the clean exit-2 error
instead; once the module lands, the skips invert automatically — no test
edits needed by the integrator.

Fixtures are DETERMINISTIC block constructions in the test_gate_cli style
(no rng in any verdict path): e.g. incumbent 200/400 vs candidate 260/400
on the same eval seed -> b=0, c=60, delta exactly +15.0 pp.
"""

import contextlib
import io as _io
import json
import os
import random
import shutil
import tempfile
import unittest

from orbit_eval.gate import adapter, cli, records

try:
    from orbit_eval.gate import sequential  # noqa: F401
    HAVE_SEQ = True
except ImportError:
    HAVE_SEQ = False


def _run(argv):
    out, err = _io.StringIO(), _io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as e:  # argparse usage errors
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
        self.tmp = tempfile.mkdtemp(prefix="shipgate_cli_v1_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_json(self, relpath, obj):
        p = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(obj, fh)
        return p

    def _output_dir(self, name, evals, seed=1000):
        """LeRobot-shaped training output dir: evals is a dict of
        {eval_dir_name: successes}; returns the output dir path."""
        root = os.path.join(self.tmp, name)
        for dname, succ in evals.items():
            self._write_json(os.path.join(name, dname, "eval_info.json"),
                             _eval_info(succ, seed=seed))
        return root


# ---------------------------------------------------------------- adapter

class TestLeRobotOutputDirAdapter(_TmpMixin, unittest.TestCase):
    def test_check_two_output_dirs_with_eval_final(self):
        a = self._output_dir("inc_run", {"eval_final": _blocks(400, 200)})
        b = self._output_dir("cand_run", {"eval_final": _blocks(400, 260)})
        code, out, err = _run(["check", a, b, "--n-boot", "150"])
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("VERDICT: SHIP", out)
        self.assertIn("CRN-PAIRED", out)

    def test_newest_eval_step_dir_chosen(self):
        a = self._output_dir("inc_run", {"eval_final": _blocks(400, 200)})
        b = self._output_dir("cand_run", {
            "eval_010000": _blocks(400, 205),   # old snapshot: no effect
            "eval_020000": _blocks(400, 260),   # newest: +15 pp
        })
        code, out, err = _run(["check", a, b, "--n-boot", "150", "--json"])
        self.assertEqual(code, 0, msg=out + err)
        rec = json.loads(out)
        self.assertIn("eval_020000", rec["inputs"]["candidate"]["path"])
        self.assertEqual(rec["statistics"]["delta_hat_pp"], 15.0)

    def test_eval_final_beats_newer_eval_step(self):
        a = self._output_dir("inc_run", {"eval_final": _blocks(400, 200)})
        b = self._output_dir("cand_run", {
            "eval_020000": _blocks(400, 300),
            "eval_final": _blocks(400, 260),
        })
        code, out, _ = _run(["check", a, b, "--n-boot", "150", "--json"])
        rec = json.loads(out)
        self.assertIn("eval_final", rec["inputs"]["candidate"]["path"])
        self.assertEqual(rec["statistics"]["delta_hat_pp"], 15.0)

    def test_plain_dir_with_single_root_eval_info_still_loads(self):
        # no eval*/ subdir: the adapter returns None and the pre-adapter
        # io.load_single dir behaviour applies unchanged
        self._write_json("inc_plain/eval_info.json",
                         _eval_info(_blocks(400, 200)))
        self._write_json("cand_plain/eval_info.json",
                         _eval_info(_blocks(400, 260)))
        code, out, err = _run(["check", os.path.join(self.tmp, "inc_plain"),
                               os.path.join(self.tmp, "cand_plain"),
                               "--n-boot", "150"])
        self.assertEqual(code, 0, msg=out + err)

    def test_dir_without_any_result_file_exits_2(self):
        empty = os.path.join(self.tmp, "empty_run")
        os.makedirs(empty)
        b = self._output_dir("cand_run", {"eval_final": _blocks(400, 260)})
        code, _, err = _run(["check", empty, b, "--n-boot", "150"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_multiple_eval_final_trees_are_ambiguous(self):
        # a path spanning TWO runs' output dirs must refuse, not pick one
        root = os.path.join(self.tmp, "many")
        self._write_json("many/runA/eval_final/eval_info.json",
                         _eval_info(_blocks(400, 200)))
        self._write_json("many/runB/eval_final/eval_info.json",
                         _eval_info(_blocks(400, 260)))
        b = self._output_dir("cand_run", {"eval_final": _blocks(400, 260)})
        code, _, err = _run(["check", root, b, "--n-boot", "150"])
        self.assertEqual(code, 2)
        self.assertIn("eval_final", err)

    def test_history_run_dir_expands_checkpoint_pool(self):
        a = self._output_dir("inc_run", {"eval_final": _blocks(400, 220)})
        b = self._output_dir("cand_run", {"eval_final": _blocks(400, 240)})
        # candidate's run dir also carries the checkpoint eval pool,
        # INCLUDING the winner's own vector (the engine's pool contract)
        for step, k in (("0000100", 180), ("0000200", 220),
                        ("0000300", 240)):
            self._write_json(os.path.join(
                "cand_run", "checkpoints", step, "eval_info.json"),
                _eval_info(_blocks(400, k)))
        code, out, err = _run(["check", a, b, "--n-boot", "150",
                               "--selection", "max-over-checkpoints",
                               "--history", b, "--json"])
        rec = json.loads(out)
        self.assertEqual(rec["statistics"]["curse_method"],
                         "history-bootstrap")
        self.assertFalse(any("does not include the candidate" in w
                             for w in rec["warnings"]))

    def test_history_dir_without_checkpoints_falls_back_to_single_eval(self):
        a = self._output_dir("inc_run", {"eval_final": _blocks(400, 220)})
        b = self._output_dir("cand_run", {"eval_final": _blocks(400, 240)})
        hist = os.path.join(self.tmp, "hist1")
        self._write_json("hist1/eval_info.json", _eval_info(_blocks(400, 240)))
        code, out, _ = _run(["check", a, b, "--n-boot", "150",
                             "--selection", "max-over-checkpoints",
                             "--history", hist, "--json"])
        rec = json.loads(out)
        self.assertEqual(rec["statistics"]["curse_method"],
                         "history-bootstrap")

    def test_history_run_dir_with_empty_checkpoints_exits_2(self):
        a = self._output_dir("inc_run", {"eval_final": _blocks(400, 220)})
        b = self._output_dir("cand_run", {"eval_final": _blocks(400, 240)})
        os.makedirs(os.path.join(self.tmp, "cand_run", "checkpoints"))
        code, _, err = _run(["check", a, b, "--n-boot", "150",
                             "--selection", "max-over-checkpoints",
                             "--history", b])
        self.assertEqual(code, 2)
        self.assertIn("checkpoints", err)

    def test_step_key_ordering(self):
        # unit sanity on the frozen 'newest' rule: largest embedded int,
        # digitless sorts oldest
        self.assertGreater(adapter._step_key("eval_020000"),
                           adapter._step_key("eval_010000"))
        self.assertEqual(adapter._step_key("eval"), -1)


# ---------------------------------------------------------------- demo

def _corpus_line(run_id, task, set_hash, train_seed, successes,
                 wave="w1", suite="pusht"):
    """One corpus per-episode JSONL line, real-shape (string-typed ints,
    '0'/'1' successes string) — the test_gate_cli fixture convention."""
    return json.dumps({
        "run_id": run_id, "wave": wave, "policy_class": "smolvla",
        "suite": suite, "task": task, "k_episodes": "22",
        "train_seed": str(train_seed), "set_hash": set_hash,
        "n_eval": len(successes),
        "sr_atlas": 100.0 * sum(successes) / len(successes),
        "eval_seed": 1000, "steps": "20000",
        "successes": "".join("1" if s else "0" for s in successes),
    })


class TestDemoCLI(_TmpMixin, unittest.TestCase):
    def _corpus_dir(self):
        rng = random.Random(11)
        lines = []
        # 4 seed replicates on one set -> 6 null pairs (12 gates) and a
        # J=3 curse cell; plus one different-set pair -> 2 effect gates.
        for ts in range(4):
            succ = [rng.random() < 0.6 for _ in range(60)]
            lines.append(_corpus_line("w1/null%02d" % ts, "task0", "aaa111",
                                      ts, succ))
        succ_b = [rng.random() < 0.6 for _ in range(60)]
        succ_c = [rng.random() < 0.85 for _ in range(60)]
        lines.append(_corpus_line("w1/eff_b", "task1", "bbb222", 0, succ_b))
        lines.append(_corpus_line("w1/eff_c", "task1", "ccc333", 0, succ_c))
        d = os.path.join(self.tmp, "corpus")
        os.makedirs(d)
        with open(os.path.join(d, "corpus.jsonl"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return d

    def test_demo_renders_and_ledger_verifies(self):
        corpus = self._corpus_dir()
        outdir = os.path.join(self.tmp, "demo_out")
        code, out, err = _run(["demo", "--corpus-dir", corpus,
                               "--out", outdir, "--n-boot", "200"])
        self.assertEqual(code, 0, msg=out + err)
        demo_path = os.path.join(outdir, "DEMO.md")
        ledger = os.path.join(outdir, "demo_ledger.jsonl")
        self.assertTrue(os.path.exists(demo_path))
        self.assertTrue(os.path.exists(ledger))
        with open(demo_path) as fh:
            md = fh.read()
        # headline BEFORE/AFTER table, three verbatim records, verify line
        self.assertIn("eval-only (before)", md)
        self.assertIn("retrain-aware (after)", md)
        self.assertEqual(md.count("```json"), 3)
        self.assertIn("all verify", md)
        # the quoted ledger really verifies: 12 null + 2 curse + 2 effect
        code, vout, _ = _run(["verify", ledger, "--json"])
        self.assertEqual(code, 0)
        rep = json.loads(vout)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["n_records"], 16)

    def test_demo_records_verify_against_embedded_hashes(self):
        corpus = self._corpus_dir()
        outdir = os.path.join(self.tmp, "demo_out2")
        code, _, err = _run(["demo", "--corpus-dir", corpus,
                             "--out", outdir, "--n-boot", "200"])
        self.assertEqual(code, 0, msg=err)
        with open(os.path.join(outdir, "DEMO.md")) as fh:
            md = fh.read()
        # each verbatim record block must round-trip and self-verify
        chunks = md.split("```json")[1:]
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            rec = json.loads(chunk.split("```")[0])
            self.assertEqual(rec["record_sha256"], records.record_hash(rec))

    def test_demo_empty_corpus_dir_exits_2(self):
        d = os.path.join(self.tmp, "nothing")
        os.makedirs(d)
        code, _, err = _run(["demo", "--corpus-dir", d,
                             "--out", os.path.join(self.tmp, "o")])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_demo_refuses_existing_ledger(self):
        corpus = self._corpus_dir()
        outdir = os.path.join(self.tmp, "demo_out3")
        os.makedirs(outdir)
        with open(os.path.join(outdir, "demo_ledger.jsonl"), "w") as fh:
            fh.write("{}\n")
        code, _, err = _run(["demo", "--corpus-dir", corpus,
                             "--out", outdir])
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)


# ---------------------------------------------------------------- seq

class _SeqMixin(_TmpMixin):
    def _episodes_stream(self, name, pairs):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as fh:
            for inc, cand in pairs:
                fh.write(json.dumps({"inc": inc, "cand": cand}) + "\n")
        return p

    def _retrains_stream(self, name, deltas, se=3.0):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as fh:
            for d in deltas:
                fh.write(json.dumps({"delta_pp": d, "se_eval_pp": se}) + "\n")
        return p


@unittest.skipUnless(HAVE_SEQ, "orbit_eval.gate.sequential not present yet "
                               "(v1 sequential workstream); skips invert "
                               "automatically once it lands")
class TestSeqCLI(_SeqMixin, unittest.TestCase):
    def test_episodes_stream_state_lines_and_ledger(self):
        # 40 candidate-only discordant wins + 40 concordant successes:
        # overwhelming evidence-to-ship at alpha 0.05
        pairs = [(0, 1)] * 40 + [(1, 1)] * 40
        p = self._episodes_stream("eps.jsonl", pairs)
        ledger = os.path.join(self.tmp, "seq_ledger.jsonl")
        code, out, err = _run(["seq", p, "--mode", "episodes",
                               "--ledger", ledger])
        self.assertEqual(code, 0, msg=out + err)
        lines = [ln for ln in out.splitlines() if ln.startswith("{")]
        self.assertEqual(len(lines), len(pairs))
        for ln in lines:
            st = json.loads(ln)
            self.assertIn("e_ship", st)
            self.assertIn("decision", st)
        final = json.loads(lines[-1])
        self.assertEqual(final["i"], len(pairs))
        self.assertEqual((final["b"], final["c"]), (0, 40))
        self.assertGreaterEqual(final["e_ship"], 20.0)     # 1/alpha
        # ledger record present, schema-v2-compatible, verifies
        code, vout, _ = _run(["verify", ledger, "--json"])
        self.assertEqual(code, 0)
        rep = json.loads(vout)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["n_records"], 1)
        with open(ledger) as fh:
            rec = json.loads(fh.readline())
        self.assertIn("e-process-episodes", json.dumps(rec))

    def test_retrains_stream_with_sigma_run(self):
        p = self._retrains_stream("ret.jsonl", [8.0] * 30, se=3.0)
        ledger = os.path.join(self.tmp, "seq_ledger_r.jsonl")
        code, out, err = _run(["seq", p, "--mode", "retrains",
                               "--sigma-run", "3.31", "--tau", "5.0",
                               "--ledger", ledger, "--json"])
        self.assertEqual(code, 0, msg=out + err)
        lines = out.splitlines()
        rec = json.loads(lines[-1])              # --json: final record last
        self.assertIn("e-process-retrains", json.dumps(rec))
        self.assertEqual(rec["record_sha256"], records.record_hash(rec))
        code, vout, _ = _run(["verify", ledger, "--json"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(vout)["ok"])

    def test_nondefault_tau_reaches_process_and_config_hash(self):
        # integration regression (found 2026-08-08): the seam formerly set
        # cfg.tau_pp/cfg.tau on TypeError fallback, attributes the frozen
        # v1 API never reads — a non-default --tau was silently ignored
        # (invisible at --tau 5.0, the default). Spec D + FROZEN_DEFAULTS
        # 22: --tau must land in GateConfig.sequential_tau, so it reaches
        # the half-normal mixture, the record's tau_pp disclosure, AND the
        # gate_config sha256.
        p = self._retrains_stream("ret_tau.jsonl", [8.0] * 10, se=3.0)
        code, out, err = _run(["seq", p, "--mode", "retrains",
                               "--sigma-run", "3.31", "--tau", "2.0",
                               "--json"])
        self.assertEqual(code, 0, msg=out + err)
        lines = out.splitlines()
        state = json.loads(lines[-2])            # --json: record is last
        rec = json.loads(lines[-1])
        self.assertEqual(state["tau_pp"], 2.0)
        self.assertEqual(rec["gate_config"]["sequential_tau"], 2.0)
        # a different tau is a different disclosed config -> different hash
        code2, out2, _ = _run(["seq", p, "--mode", "retrains",
                               "--sigma-run", "3.31", "--tau", "4.0",
                               "--json"])
        self.assertEqual(code2, 0)
        rec2 = json.loads(out2.splitlines()[-1])
        self.assertNotEqual(rec["gate_config_sha256"],
                            rec2["gate_config_sha256"])

    def test_invalid_tau_rejected_by_config_validation(self):
        # folding --tau into the config means cfg.validate() prices it:
        # a nonpositive prior scale must exit 2, not construct a process
        p = self._retrains_stream("ret_badtau.jsonl", [8.0] * 3, se=3.0)
        code, out, err = _run(["seq", p, "--mode", "retrains",
                               "--sigma-run", "3.31", "--tau", "-1.0"])
        self.assertEqual(code, 2, msg=out + err)
        self.assertIn("ERROR", err)

    def test_every_thins_state_lines(self):
        pairs = [(0, 1)] * 25
        p = self._episodes_stream("eps25.jsonl", pairs)
        code, out, err = _run(["seq", p, "--mode", "episodes",
                               "--every", "10"])
        self.assertEqual(code, 0, msg=out + err)
        lines = [ln for ln in out.splitlines() if ln.startswith("{")]
        # i = 10, 20, and the always-emitted final 25
        self.assertEqual([json.loads(ln)["i"] for ln in lines],
                         [10, 20, 25])

    def test_bad_stream_line_exits_2(self):
        p = os.path.join(self.tmp, "bad.jsonl")
        with open(p, "w") as fh:
            fh.write('{"inc": 1, "cand": 1}\n{"inc": 2, "cand": 0}\n')
        code, _, err = _run(["seq", p, "--mode", "episodes"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)
        self.assertIn(":2", err)                  # the offending line number

    def test_non_finite_retrains_line_exits_2_with_context(self):
        # json.loads parses the non-standard NaN/Infinity tokens; the stream
        # validator must reject them with file:line context (a NaN delta
        # would silently poison S_t, an infinite se would be a zero-weight
        # no-op that still counts as an observation)
        for i, bad in enumerate(('{"delta_pp": NaN, "se_eval_pp": 3.0}',
                                 '{"delta_pp": 8.0, "se_eval_pp": Infinity}',
                                 '{"delta_pp": -Infinity, "se_eval_pp": 3.0}')):
            p = os.path.join(self.tmp, "nonfinite%d.jsonl" % i)
            with open(p, "w") as fh:
                fh.write('{"delta_pp": 8.0, "se_eval_pp": 3.0}\n')
                fh.write(bad + "\n")
            code, _, err = _run(["seq", p, "--mode", "retrains",
                                 "--sigma-run", "3.31"])
            self.assertEqual(code, 2, msg=err)
            self.assertIn("ERROR", err)
            self.assertIn("FINITE", err)
            self.assertIn(":2", err)              # the offending line number

    def test_missing_stream_exits_2(self):
        code, _, err = _run(["seq", os.path.join(self.tmp, "no.jsonl"),
                             "--mode", "episodes"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_empty_stream_exits_2(self):
        p = os.path.join(self.tmp, "empty.jsonl")
        with open(p, "w") as fh:
            fh.write("\n")
        code, _, err = _run(["seq", p, "--mode", "episodes"])
        self.assertEqual(code, 2)
        self.assertIn("no observations", err)


@unittest.skipIf(HAVE_SEQ, "sequential module present — the clean-error "
                           "path is unreachable")
class TestSeqUnavailable(_SeqMixin, unittest.TestCase):
    def test_seq_without_module_is_clean_exit_2(self):
        p = self._episodes_stream("eps.jsonl", [(0, 1)] * 3)
        code, _, err = _run(["seq", p, "--mode", "episodes"])
        self.assertEqual(code, 2)
        self.assertIn("sequential", err)


# ---------------------------------------------------------------- parser

class TestSeqParserSurface(unittest.TestCase):
    def test_seq_requires_mode(self):
        code, _, _ = _run(["seq", "stream.jsonl"])
        self.assertEqual(code, 2)

    def test_demo_requires_corpus_dir_and_out(self):
        code, _, _ = _run(["demo"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
