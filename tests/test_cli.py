"""End-to-end CLI behaviour: compare regimes, regress exit codes, power text."""

import contextlib
import io as _io
import json
import os
import random
import shutil
import tempfile
import unittest

from orbit_eval import cli

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _run(argv):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _eval_info(successes, seed=1000):
    n = len(successes)
    return {
        "per_task": [{"task_group": "pusht", "task_id": 0, "metrics": {
            "sum_rewards": [0.0] * n, "max_rewards": [0.0] * n,
            "successes": [bool(s) for s in successes], "video_paths": []}}],
        "overall": {"pc_success": 100.0 * sum(successes) / n,
                    "n_episodes": n},
        "seed": seed,
    }


class TestCompareCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orbit_eval_cli_")
        rng = random.Random(5)
        self.sa, self.sb = [], []
        for _ in range(500):
            u = rng.random()
            self.sa.append(u < 0.55)
            self.sb.append(u < 0.45)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, obj):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as fh:
            json.dump(obj, fh)
        return p

    def test_crn_paired_compare(self):
        a = self._write("a_eval_info.json", _eval_info(self.sa))
        b = self._write("b_eval_info.json", _eval_info(self.sb))
        code, out = _run(["compare", a, b])
        self.assertEqual(code, 0)
        self.assertIn("CRN-PAIRED", out)
        self.assertIn("McNemar", out)
        self.assertIn("WHICH QUESTION IS THIS?", out)
        self.assertIn("CANNOT answer (b)", out)
        self.assertIn("VALIDATED 2026-08-02", out)  # pairing-validated provenance

    def test_unpaired_on_seed_mismatch(self):
        a = self._write("a_eval_info.json", _eval_info(self.sa, seed=0))
        b = self._write("b_eval_info.json", _eval_info(self.sb, seed=1))
        code, out = _run(["compare", a, b])
        self.assertEqual(code, 0)
        self.assertIn("UNPAIRED", out)
        self.assertIn("CRN pairing", out)

    def test_compare_real_orbit_records_blocks_on_nothing(self):
        # two real single-run files from the paired wave (no per-episode data)
        recs = json.load(open(os.path.join(FIX, "paired1_results.json")))
        a = self._write("a.model_result.json",
                        [r for r in recs if r["mask_id"] == "p_b0_m0"][0])
        b = self._write("b.model_result.json",
                        [r for r in recs if r["mask_id"] == "p_b1_m0"][0])
        code, out = _run(["compare", a, b])
        self.assertEqual(code, 0)
        self.assertIn("UNPAIRED", out)

    def test_truncated_run_is_rejected(self):
        recs = json.load(open(os.path.join(FIX, "paired1_results.json")))
        bad = dict([r for r in recs if r["mask_id"] == "p_b0_m0"][0])
        bad["final_step"] = 35000            # the OPS_RUNBOOK sec 3 defect
        a = self._write("bad.model_result.json", bad)
        b = self._write("ok.model_result.json",
                        [r for r in recs if r["mask_id"] == "p_b1_m0"][0])
        code, out = _run(["compare", a, b])
        self.assertEqual(code, 2)
        self.assertIn("INVALID", out)
        self.assertIn("directional", out)


class TestRegressCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orbit_eval_reg_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pair(self, n, p_base, drop_pp, seed=1000, seed_b=None):
        """CRN-style pair: candidate loses drop_pp of episodes one-sidedly."""
        rng = random.Random(9)
        sa, sb = [], []
        p_cand = p_base - drop_pp / 100.0
        for _ in range(n):
            u = rng.random()
            sa.append(u < p_base)
            sb.append(u < p_cand)
        pa = os.path.join(self.tmp, "base_eval_info.json")
        pb = os.path.join(self.tmp, "cand_eval_info.json")
        json.dump(_eval_info(sa, seed), open(pa, "w"))
        json.dump(_eval_info(sb, seed if seed_b is None else seed_b),
                  open(pb, "w"))
        return pa, pb

    def test_exit_1_on_regression(self):
        a, b = self._pair(500, 0.60, 20.0)
        code, out = _run(["regress", a, b, "--regime", "smolvla-ft"])
        self.assertEqual(code, 1)
        self.assertIn("REGRESSION", out)

    def test_exit_0_no_regression(self):
        a, b = self._pair(500, 0.60, 0.0)
        code, out = _run(["regress", a, b, "--regime", "smolvla-ft"])
        self.assertEqual(code, 0)
        self.assertIn("no regression", out)

    def test_exit_3_underpowered(self):
        # different eval seeds -> UNPAIRED; 50 episodes/arm cannot certify a
        # ~6.6 pp gate (unpaired MDE ~ 28 pp at p~.5). Exit 3, NOT 2: a CI
        # consumer must distinguish "inputs unusable" (2) from "design too
        # weak — collect more" (3, the shipgate COLLECT-MORE convention).
        a, b = self._pair(50, 0.50, 0.0, seed=0, seed_b=1)
        code, out = _run(["regress", a, b, "--regime", "smolvla-ft"])
        self.assertEqual(code, 3)
        self.assertIn("UNDERPOWERED", out)

    def test_exit_2_unknown_regime_clean_error(self):
        # a typo'd --regime must print the available regimes and exit 2,
        # never crash with a raw KeyError traceback (mirrors cmd_power)
        a, b = self._pair(500, 0.60, 0.0)
        code, out = _run(["regress", a, b, "--regime", "typo"])
        self.assertEqual(code, 2)
        self.assertIn("unknown regime", out)
        self.assertIn("smolvla-ft", out)     # lists the available names

    def test_exit_2_invalid_inputs(self):
        rec = {"mask_id": "x", "sr": 30.0, "n_eval": 500, "seed": 0,
               "final_step": 35000, "design_steps": 100000}
        pa = os.path.join(self.tmp, "bad.model_result.json")
        json.dump(rec, open(pa, "w"))
        ok = dict(rec, final_step=100000, mask_id="y")
        pb = os.path.join(self.tmp, "ok.model_result.json")
        json.dump(ok, open(pb, "w"))
        code, out = _run(["regress", pa, pb])
        self.assertEqual(code, 2)
        self.assertIn("INVALID", out)

    def test_gate_override(self):
        a, b = self._pair(500, 0.60, 20.0)
        # an absurdly high gate turns the same 20 pp drop into a pass
        code, _ = _run(["regress", a, b, "--gate", "40",
                        "--regime", "smolvla-ft"])
        self.assertEqual(code, 0)

    def test_paired_regress_prints_caveats_and_banner(self):
        # the CI gate is where an unqualified green light gets trusted: a
        # paired regress must carry the pairing-validated provenance and the
        # (a)/(b) fixed-set-vs-method banner
        a, b = self._pair(500, 0.60, 0.0)
        code, out = _run(["regress", a, b, "--regime", "smolvla-ft"])
        self.assertEqual(code, 0)
        self.assertIn("CAVEAT", out)
        self.assertIn("VALIDATED 2026-08-02", out)      # pairing-validated
        self.assertIn("WHICH QUESTION IS THIS?", out)
        self.assertIn("CANNOT answer (b)", out)

    def test_default_gate_names_its_atlas_regime(self):
        a, b = self._pair(500, 0.60, 0.0)
        code, out = _run(["regress", a, b])
        self.assertEqual(code, 0)
        self.assertIn("smolvla-ft", out)           # provenance of the 3.31
        self.assertIn("if your stack differs", out)

    def test_regress_json(self):
        a, b = self._pair(500, 0.60, 20.0)
        code, out = _run(["regress", a, b, "--regime", "smolvla-ft",
                          "--json"])
        self.assertEqual(code, 1)
        rep = json.loads(out)
        self.assertEqual(rep["verdict"], "REGRESSION")
        self.assertEqual(rep["exit"], 1)
        self.assertIn("gate_pp", rep)
        self.assertIn("mde_hat_pp", rep)


class TestPowerCLI(unittest.TestCase):
    def test_fixed_set(self):
        code, out = _run(["power", "--regime", "pusht-dp", "--effect", "5"])
        self.assertEqual(code, 0)
        self.assertIn("sigma_run = 2.09", out)
        self.assertIn("3 draws per arm", out)      # hand-checked table value
        self.assertIn("FIXED-SET", out)

    def test_method_arms_59(self):
        code, out = _run(["power", "--regime", "smolvla-ft",
                          "--effect", "10", "--arms", "method"])
        self.assertEqual(code, 0)
        self.assertIn("59 set draws per arm", out)
        self.assertIn("CRN pairing cannot rescue", out)
        self.assertIn("19.30", out)               # sqrt(19.01^2+3.31^2)

    def test_audit_cli_json(self):
        tmp = tempfile.mkdtemp(prefix="orbit_eval_audj_")
        try:
            p = os.path.join(tmp, "r", "model_result.json")
            os.makedirs(os.path.dirname(p))
            json.dump({"mask_id": "r", "sr": 50.0, "n_eval": 50, "seed": 1000,
                       "final_step": 100000, "design_steps": 100000},
                      open(p, "w"))
            code, out = _run(["audit", tmp, "--json"])
            self.assertEqual(code, 1)              # flags found
            rep = json.loads(out)
            codes = {f["code"] for f in rep["flags"]}
            self.assertEqual(codes, {"SEED_DEFAULT_1000", "TINY_N"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_audit_nonexistent_path_exits_2(self):
        # a typo'd path in CI must NOT produce a clean bill of health
        code, out = _run(["audit", "/no/such/dir/orbit_eval_xyz"])
        self.assertEqual(code, 2)
        self.assertIn("not a directory", out)

    def test_audit_empty_dir_exits_2(self):
        tmp = tempfile.mkdtemp(prefix="orbit_eval_empty_")
        try:
            code, out = _run(["audit", tmp])
            self.assertEqual(code, 2)
            self.assertIn("no recognisable result files", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_json(self):
        tmp = tempfile.mkdtemp(prefix="orbit_eval_cmpj_")
        try:
            rng = random.Random(5)
            sa, sb = [], []
            for _ in range(200):
                u = rng.random()
                sa.append(u < 0.55)
                sb.append(u < 0.45)
            pa = os.path.join(tmp, "a_eval_info.json")
            pb = os.path.join(tmp, "b_eval_info.json")
            json.dump(_eval_info(sa), open(pa, "w"))
            json.dump(_eval_info(sb), open(pb, "w"))
            code, out = _run(["compare", pa, pb, "--json"])
            self.assertEqual(code, 0)
            rep = json.loads(out)
            self.assertTrue(rep["paired"])
            self.assertIn("p_mcnemar", rep)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class TestDemoAndCover(unittest.TestCase):
    """`--demo` runs on bundled public metadata, so a cold machine has something to look at."""

    def test_status_demo_shows_the_clipped_bounds_and_exits_zero(self):
        code, out = _run(["status", "--demo"])
        self.assertEqual(code, 0)
        self.assertIn("DEMO", out)
        self.assertIn("svla_so101_pickplace", out)
        self.assertIn("shoulder_lift.pos", out)
        self.assertIn("elbow_flex.pos", out)
        self.assertIn("WHAT A BATTERY BUYS", out)
        self.assertIn("ALSO WORTH KNOWING", out)

    def test_status_demo_json_is_the_same_scan(self):
        code, out = _run(["status", "--demo", "--json"])
        self.assertEqual(code, 0)
        d = json.loads(out)
        self.assertEqual(d["dataset"]["n_episodes"], 50)
        self.assertEqual(sorted(f["joint"] for f in d["findings"]
                                if f["code"] == "saturated_action"),
                         ["elbow_flex.pos", "shoulder_lift.pos"])

    def test_cover_demo_reads_the_v21_table_with_no_dependency(self):
        code, out = _run(["cover", "--demo"])
        self.assertEqual(code, 0)
        self.assertIn("Grab {A} and place into pen holder", out)
        self.assertIn("THE LARGEST GAP", out)
        self.assertIn("never scores a demonstration", out)

    def test_cover_json_round_trips(self):
        code, out = _run(["cover", "--demo", "--json"])
        self.assertEqual(code, 0)
        d = json.loads(out)
        self.assertEqual(d["n_episodes"], 80)
        self.assertEqual(d["instructions"]["template"], "Grab {A} and place into pen holder")

    def test_cover_on_an_empty_directory_says_what_it_reads(self):
        tmp = tempfile.mkdtemp(prefix="orbit_eval_cover_")
        try:
            code, out = _run(["cover", tmp])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(code, 2)
        self.assertIn("meta/info.json", out)

    def test_status_lists_the_other_commands_without_a_menu(self):
        code, out = _run(["status", "--demo"])
        for cmd in ("orbit cover", "orbit body", "orbit freeze", "orbit check",
                    "orbit next", "orbit log"):
            self.assertIn(cmd, out)


class TestBodyFreezeAndLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orbit_eval_bfl_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_body_lists_the_database_with_sources_on_request(self):
        code, out = _run(["body"])
        self.assertEqual(code, 0)
        self.assertIn("so101", out)
        self.assertIn("robots", out)
        code, out = _run(["body", "SO-101"])
        self.assertEqual(code, 0)
        self.assertIn("https://", out)
        self.assertIn("not published", out)      # reach and payload, honestly

    def test_body_arithmetic_names_the_shortfall(self):
        code, out = _run(["body", "--reach", "500", "--payload", "1", "--budget", "5000",
                          "--job", "pick_place"])
        self.assertEqual(code, 0)
        self.assertIn("PASSES", out)
        self.assertIn("short by", out)
        self.assertIn("MEASURED ON THIS JOB SHAPE", out)
        self.assertNotIn("not right for", out.lower())

    def test_freeze_writes_and_verifies_a_manifest(self):
        from test_freeze import write_checkpoint
        ck = write_checkpoint(self.tmp)
        out_path = os.path.join(self.tmp, "release.json")
        code, out = _run(["freeze", "--checkpoint", ck, "--out", out_path])
        self.assertEqual(code, 0)
        self.assertIn("FROZEN", out)
        self.assertIn("sha256", out)
        code, out = _run(["freeze", "--verify", out_path])
        self.assertEqual(code, 0)
        self.assertIn("VERIFIED", out)
        with open(os.path.join(ck, "model.safetensors"), "wb") as fh:
            fh.write(b"changed")
        code, out = _run(["freeze", "--verify", out_path])
        self.assertEqual(code, 1)
        self.assertIn("model.safetensors", out)

    def test_body_skill_compares_a_manifest_with_a_recording(self):
        from test_freeze import write_checkpoint
        from test_dataset import write_dataset
        ck = write_checkpoint(self.tmp)
        ds = write_dataset(os.path.join(self.tmp, "data"))
        out_path = os.path.join(self.tmp, "release.json")
        _run(["freeze", "--checkpoint", ck, "--dataset", ds, "--out", out_path])
        code, out = _run(["body", "--skill", out_path, "--path", ds])
        self.assertEqual(code, 0)
        self.assertIn("SKILL FIT", out)
        self.assertIn("calibration match", out)

    def test_check_demo_audits_a_published_loop_as_rounds(self):
        code, out = _run(["check", "--demo"])
        self.assertEqual(code, 0)
        self.assertIn("DEMO", out)
        self.assertIn("2609.14633", out)
        self.assertIn("loop", out)
        self.assertNotIn("retrain and pick", out)
        self.assertIn("rounds", out)
