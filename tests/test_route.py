"""route: loader validation, abstention rule, per-task incumbents, blocks, held-out
estimator, budget, plan, CLI exit codes."""

import contextlib
import io as _io
import json
import os
import random
import shutil
import tempfile
import unittest

from orbit_eval import cli, route


def _run(argv):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _pool(rates, n, seed):
    rng = random.Random(seed)
    return {c: {t: [rng.random() < p for _ in range(n)] for t, p in ts.items()}
            for c, ts in rates.items()}


def _lottery(J=8, T=10, n=200, seed=0, swing=0.35, base=0.55):
    rates = {}
    for j in range(J):
        rates["c%d" % j] = {}
        for t in range(T):
            p = base + (swing if t % J == j else -swing / (T - 1))
            rates["c%d" % j]["t%d" % t] = min(0.98, max(0.02, p))
    return _pool(rates, n, seed)


def _write_json(path, data, blocks=None):
    obj = {"candidates": {c: {t: [int(b) for b in v] for t, v in ts.items()} for c, ts in data.items()}}
    if blocks:
        obj["blocks"] = blocks
    json.dump(obj, open(path, "w"))


class TestLoaderAndValidate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orbit_route_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_dir(self, data):
        d = os.path.join(self.tmp, "cands")
        for c, tasks in data.items():
            for i, (t, bits) in enumerate(sorted(tasks.items())):
                p = os.path.join(d, c, "t%d" % i)
                os.makedirs(p)
                json.dump({"per_task": [{"task_group": "g", "task_id": i,
                                         "metrics": {"successes": bits}}],
                           "overall": {"n_episodes": len(bits),
                                       "pc_success": 100.0 * sum(bits) / len(bits)}},
                          open(os.path.join(p, "eval_info.json"), "w"))
        return d

    def test_dir_and_json_load_agree(self):
        data = _lottery(J=3, T=4, n=40)
        d = self._write_dir(data)
        got, files, blocks = route.load_candidates(d)
        self.assertEqual(sorted(got), sorted(data))
        self.assertEqual(len(files), 12)
        self.assertIsNone(blocks)
        js = os.path.join(self.tmp, "m.json")
        _write_json(js, data)
        got2, _, _ = route.load_candidates(js)
        for c in data:
            for t in data[c]:
                self.assertEqual(data[c][t], got2[c][t])
        self.assertEqual(route.validate(got)[0], [])

    def test_csv_export_loads_like_json_with_blocks(self):
        data = _lottery(J=2, T=3, n=24, seed=4)
        cp = os.path.join(self.tmp, "export.csv")
        with open(cp, "w") as fh:
            fh.write("checkpoint,sku,episode,success,robot\n")
            for c, ts in data.items():
                for t, bits in ts.items():
                    for e, b in enumerate(bits):
                        fh.write("%s,%s,%d,%s,r%d\n" % (c, t, e, "true" if b else "false", e % 3))
        got, files, blocks = route.load_candidates(cp)
        self.assertEqual(files, [cp])
        for c in data:
            for t in data[c]:
                self.assertEqual(data[c][t], got[c][t])
                self.assertEqual(blocks[c][t][:4], ["r0", "r1", "r2", "r0"])
        inv, warn = route.validate(got, blocks)
        self.assertEqual(inv, [])
        sb = route.shared_blocks(got, blocks)
        self.assertEqual(sorted(sb), sorted(data["c0"]))

    def test_block_layout_mismatch_is_invalid(self):
        data = _lottery(J=2, T=2, n=20, seed=5)
        blocks = {c: {t: ["a"] * 10 + ["b"] * 10 for t in data[c]} for c in data}
        blocks["c1"]["t0"] = ["b"] * 10 + ["a"] * 10          # same composition, shifted
        inv, _ = route.validate(data, blocks)
        self.assertTrue(any("different blocks across candidates" in m for m in inv))
        blocks["c1"]["t0"] = ["a"] * 20                        # different composition
        inv, _ = route.validate(data, blocks)
        self.assertTrue(any("composition differs" in m for m in inv))

    def test_non_json_file_gives_clear_error(self):
        bad = os.path.join(self.tmp, "export.txt")
        open(bad, "w").write("checkpoint,task\n")
        with self.assertRaises(ValueError) as cm:
            route.load_candidates(bad)
        self.assertIn("Accepted inputs", str(cm.exception))

    def test_unequal_n_is_invalid(self):
        data = _lottery(J=2, T=2, n=40)
        data["c0"]["t0"] = data["c0"]["t0"][:30]
        self.assertTrue(any("unequal episode counts" in m for m in route.validate(data)[0]))

    def test_missing_task_is_invalid(self):
        data = _lottery(J=2, T=2, n=40)
        del data["c1"]["t1"]
        self.assertTrue(any("task sets differ" in m for m in route.validate(data)[0]))

    def test_single_candidate_is_invalid(self):
        self.assertTrue(route.validate({"only": {"t0": [True] * 20}})[0])


class TestIncumbentSpec(unittest.TestCase):
    def test_forms(self):
        cands, tasks = ["a", "b"], ["t0", "t1"]
        self.assertIsNone(route.parse_incumbent_spec(None, cands, tasks))
        self.assertEqual(route.parse_incumbent_spec("a", cands, tasks), {"t0": "a", "t1": "a"})
        self.assertEqual(route.parse_incumbent_spec("t0=a,t1=b", cands, tasks), {"t0": "a", "t1": "b"})
        self.assertEqual(route.parse_incumbent_spec("t1=b,*=a", cands, tasks), {"t0": "a", "t1": "b"})
        with self.assertRaises(ValueError):
            route.parse_incumbent_spec("t0=a", cands, tasks)          # t1 missing, no default
        with self.assertRaises(ValueError):
            route.parse_incumbent_spec("zzz", cands, tasks)
        tmp = tempfile.mkdtemp()
        try:
            js = os.path.join(tmp, "inc.json")
            json.dump({"t0": "b", "t1": "a"}, open(js, "w"))
            self.assertEqual(route.parse_incumbent_spec(js, cands, tasks), {"t0": "b", "t1": "a"})
            cp = os.path.join(tmp, "inc.csv")
            open(cp, "w").write("task,candidate\nt0,a\nt1,b\n")
            self.assertEqual(route.parse_incumbent_spec(cp, cands, tasks), {"t0": "a", "t1": "b"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestReleaseTable(unittest.TestCase):
    def test_defects_only_on_clear_advantage(self):
        data = {"inc": {"a": [True] * 100 + [False] * 100,
                        "b": [True] * 100 + [False] * 100},
                "cand": {"a": [True] * 180 + [False] * 20,
                         "b": [True] * 106 + [False] * 94}}
        rel = route.release_table(data, incumbent={"a": "inc", "b": "inc"})
        rows = {r["task"]: r for r in rel["rows"]}
        self.assertEqual(rows["a"]["decision"], "DEFECT")
        self.assertEqual(rows["a"]["chosen"], "cand")
        self.assertEqual(rows["b"]["decision"], "ABSTAIN")
        self.assertEqual(rows["b"]["chosen"], "inc")
        self.assertEqual(rel["n_defections"], 1)
        self.assertEqual(rows["a"]["regression_bound_pp"], 0.0)
        self.assertAlmostEqual(rel["expected_false_defections_under_null"], 0.05 * 2, places=3)

    def test_per_task_incumbent_and_blocks(self):
        # task a: cand wins in both blocks; task b: cand wins overall but loses in block y
        data = {"inc": {"a": [True] * 50 + [False] * 50 + [True] * 50 + [False] * 50,
                        "b": [False] * 100 + [True] * 90 + [False] * 10},
                "cand": {"a": [True] * 90 + [False] * 10 + [True] * 90 + [False] * 10,
                         "b": [True] * 100 + [True] * 60 + [False] * 40}}
        blocks = {"a": ["x"] * 100 + ["y"] * 100, "b": ["x"] * 100 + ["y"] * 100}
        rel = route.release_table(data, incumbent={"a": "inc", "b": "inc"}, blocks=blocks)
        rows = {r["task"]: r for r in rel["rows"]}
        self.assertEqual(rows["a"]["decision"], "DEFECT")
        self.assertTrue(rows["a"]["blocks_agree"])
        self.assertEqual(rows["b"]["decision"], "DEFECT")
        self.assertFalse(rows["b"]["blocks_agree"])
        self.assertLess(rows["b"]["min_block_advantage_pp"], 0)
        self.assertEqual(rel["defections_with_block_disagreement"], ["b"])
        self.assertEqual(rel["blocks"], ["x", "y"])

    def test_incumbent_default_is_best_suite(self):
        data = _lottery(J=3, T=3, n=50, seed=3)
        rel = route.release_table(data)
        best = max(rel["suite_sr"], key=lambda k: (rel["suite_sr"][k], k))
        self.assertEqual(set(rel["incumbent"].values()), {best})


class TestSplitHalf(unittest.TestCase):
    def test_lottery_pool_pays_and_null_pool_does_not(self):
        lot = route.split_half(_lottery(seed=1), draws=150, seed=7)
        self.assertGreater(lot["gain_route_abstain_vs_incumbent"], 5.0)
        self.assertGreaterEqual(lot["arms"]["ORACLE"]["heldout_sr"], lot["arms"]["ROUTE"]["heldout_sr"])
        self.assertLess(lot["arms"]["ROUTE+ABSTAIN"]["damaged_tasks"], lot["arms"]["INCUMBENT"]["damaged_tasks"] + 1e-9)
        rates = {"c%d" % j: {"t%d" % t: 0.5 for t in range(10)} for j in range(8)}
        nul = route.split_half(_pool(rates, 200, 11), draws=150, seed=7)
        self.assertLess(abs(nul["gain_route_abstain_vs_incumbent"]), 2.0)
        self.assertGreater(nul["abstain_rate"], 0.7)

    def test_blocks_stratify_and_per_task_incumbent(self):
        data = _lottery(J=3, T=3, n=60, seed=2)
        blocks = {t: ["r%d" % (i % 3) for i in range(60)] for t in data["c0"]}
        inc = {"t0": "c0", "t1": "c1", "t2": "c2"}
        a = route.split_half(data, draws=40, seed=5, incumbent=inc, blocks=blocks)
        self.assertTrue(a["stratified_by_block"])
        self.assertIn("INCUMBENT", a["arms"])

    def test_deterministic_in_seed(self):
        d = _lottery(J=3, T=3, n=40, seed=2)
        a = route.split_half(d, draws=50, seed=5)
        b = route.split_half(d, draws=50, seed=5)
        self.assertEqual(a["arms"]["ROUTE"]["heldout_sr"], b["arms"]["ROUTE"]["heldout_sr"])


class TestBudgetAndPlan(unittest.TestCase):
    def test_thresholds_shrink_with_budget(self):
        a = route.plan(8, 10, 20)
        b = route.plan(8, 10, 80)
        self.assertAlmostEqual(a["defection_threshold_pp"] / b["defection_threshold_pp"], 2.0, places=6)
        self.assertGreater(a["advantage_80pct_power_pp"], a["defection_threshold_pp"])
        self.assertEqual(a["selection_episodes"], 8 * 10 * 20)
        self.assertEqual(len(a["measured_cells"]), 2)

    def test_budget_targets_round_trip(self):
        bud = route.budget(20, 4, 8)
        ten = [r for r in bud["targets"] if r["advantage_pp"] == 10.0][0]
        # at n_needed the 80%-power advantage is <= 10 pp
        self.assertLessEqual(route.budget(ten["n_needed_per_task"], 4, 8)["advantage_80pct_power_pp"], 10.0 + 1e-9)
        self.assertEqual(ten["extra_episodes_total"], (ten["n_needed_per_task"] - 20) * 4 * 8)
        self.assertFalse(ten["already_resolvable"])


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orbit_route_cli_")
        self.data = _lottery(J=4, T=5, n=60, seed=9)
        self.js = os.path.join(self.tmp, "m.json")
        _write_json(self.js, self.data)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_writes_record_and_report(self):
        out = os.path.join(self.tmp, "release.json"); rep = os.path.join(self.tmp, "REPORT.md")
        code, text = _run(["route", "build", self.js, "--draws", "40", "--out", out, "--report", rep,
                           "--policy-path", "c0=/w/c0", "--policy-path", "c1=/w/c1"])
        self.assertEqual(code, 0, text)
        self.assertIn("Release table", text)
        self.assertIn("## Budget", text)
        rec = json.load(open(out))
        self.assertIn("record_sha256", rec)
        self.assertIn("built_at_utc", rec["meta"])
        self.assertIn("argv", rec["meta"])
        self.assertIn("budget", rec)
        self.assertEqual(rec["release"]["policy_paths"], {"c0": "/w/c0", "c1": "/w/c1"})
        self.assertEqual(len(rec["release"]["rows"]), 5)
        self.assertTrue(os.path.exists(rep))

    def test_build_per_task_incumbent_and_blocks_json(self):
        js = os.path.join(self.tmp, "b.json")
        blocks = {t: ["day%d" % (i % 2) for i in range(60)] for t in self.data["c0"]}
        _write_json(js, self.data, blocks)
        code, text = _run(["route", "build", js, "--draws", "20", "--incumbent", "t0=c1,*=c0"])
        self.assertEqual(code, 0, text)
        self.assertIn("blocks: day0, day1", text)
        self.assertIn("blocks agree", text)

    def test_build_json_mode(self):
        code, text = _run(["route", "build", self.js, "--draws", "20", "--json"])
        self.assertEqual(code, 0)
        self.assertIn("heldout", json.loads(text))

    def test_build_bad_path_exit2(self):
        code, _ = _run(["route", "build", os.path.join(self.tmp, "missing")])
        self.assertEqual(code, 2)

    def test_build_bad_incumbent_exit2(self):
        code, _ = _run(["route", "build", self.js, "--draws", "5", "--incumbent", "zzz"])
        self.assertEqual(code, 2)

    def test_build_bad_policy_path_exit2(self):
        code, _ = _run(["route", "build", self.js, "--draws", "5", "--policy-path", "nope=/x"])
        self.assertEqual(code, 2)

    def test_plan_text(self):
        code, text = _run(["route", "plan", "--candidates", "8", "--tasks", "10", "--select-eps", "50"])
        self.assertEqual(code, 0)
        self.assertIn("defects at a per-task advantage", text)
        self.assertIn("to resolve a per-task advantage", text)
        self.assertIn("NOT a forecast", text)

    def test_plan_reports_regression_risk(self):
        code, text = _run(["route", "plan", "--candidates", "3", "--tasks", "10", "--select-eps", "25"])
        self.assertEqual(code, 0)
        self.assertIn("REGRESSION RISK", text)
        self.assertIn("ship one retrain", text)
        # best-validation never reaches the 5% risk target in any measured cell
        self.assertIn("best-validation NEVER, at any J tested", text)
        # the honest scope note on the gain must travel with the plan
        self.assertIn("libero_spatial", text)

    def test_damage_rows_pick_nearest_task_count(self):
        small = route.damage_rows(J=3, T=10)
        big = route.damage_rows(J=3, T=50)
        self.assertEqual(len(small), 2)
        self.assertTrue(all(r["cell_T"] == 10 for r in small))
        self.assertEqual(big[0]["cell_T"], 50)

    def test_required_pool_grows_with_task_count(self):
        """The DAMAGE-J shape: more tasks needs a bigger pool to reach 5% risk."""
        t10 = [r for r in route.damage_rows(J=8, T=10) if r["cell_T"] == 10][0]
        t50 = [r for r in route.damage_rows(J=8, T=50) if r["cell_T"] == 50][0]
        self.assertIsNotNone(t10["J_for_5pct_route"])
        self.assertIsNotNone(t50["J_for_5pct_route"])
        self.assertLess(t10["J_for_5pct_route"], t50["J_for_5pct_route"])
        # and best-validation selection never gets there, in either
        self.assertIsNone(t10["J_for_5pct_best_val"])
        self.assertIsNone(t50["J_for_5pct_best_val"])

    def test_damage_curves_are_monotone_and_bounded(self):
        for name, cell in route.DAMAGE_J_CELLS.items():
            self.assertIn("T", cell, name)
            for arm in ("best_val", "route", "route_abstain"):
                vals = [cell[arm][j] for j in sorted(cell[arm])]
                self.assertTrue(all(0.0 <= v <= 1.0 for v in vals), (name, arm))
                # The routing arms must not get WORSE as the pool grows. Asserted on the
                # endpoints with a Monte-Carlo tolerance rather than pointwise: at 4000
                # draws the SE of a proportion near 0.02 is ~0.002, and on libero_spatial
                # the whole curve sits inside ~0.02, so pointwise monotonicity there is a
                # test of the RNG, not of the finding.
                if arm != "best_val":
                    self.assertLessEqual(vals[-1], vals[0] + 0.01, (name, arm))


if __name__ == "__main__":
    unittest.main()
