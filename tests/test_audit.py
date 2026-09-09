"""audit on a fixture tree containing every defect class it must catch:
a seed-1000 case, a truncated case, replicate sets, missing/tiny n_eval."""

import json
import os
import shutil
import tempfile
import unittest

from orbit_eval import audit

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


class TestAuditTree(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="orbit_eval_audit_")
        # 1) real LeRobot eval_info (no seed recorded) -> SEED_ABSENT
        shutil.copy(os.path.join(FIX, "cupid_e1_eval_info.json"),
                    self._mk("runA", "eval_info.json"))
        # 2) seed == 1000 (LeRobot's silent CRN default) -> SEED_DEFAULT_1000
        _write(self._mk("runB", "model_result.json"), {
            "mask_id": "runB", "sr": 42.0, "n_eval": 500, "seed": 1000,
            "final_step": 100000, "design_steps": 100000,
            "episodes": [1, 2, 3]})
        # 3) truncated budget (the OPS_RUNBOOK sec 3 defect) -> TRUNCATED
        _write(self._mk("runC", "model_result.json"), {
            "mask_id": "runC", "sr": 12.0, "n_eval": 500, "seed": 0,
            "final_step": 35000, "design_steps": 100000,
            "episodes": [4, 5, 6]})
        # 4+5) identical training sets -> REPLICATE_SET (via set_hash)
        for rid in ("runD", "runE"):
            _write(self._mk(rid, "model_result.json"), {
                "mask_id": rid, "sr": 55.0, "n_eval": 500, "seed": 0,
                "final_step": 100000, "design_steps": 100000,
                "episodes": [7, 8, 9]})
        # 6) missing n_eval -> N_EVAL_MISSING
        _write(self._mk("runF", "model_result.json"), {
            "mask_id": "runF", "sr": 60.0, "seed": 0,
            "final_step": 100000, "design_steps": 100000})
        # 7) tiny n -> TINY_N (n=50 at 50% has a ~27 pp Wilson CI)
        _write(self._mk("runG", "model_result.json"), {
            "mask_id": "runG", "sr": 50.0, "n_eval": 50, "seed": 0,
            "final_step": 100000, "design_steps": 100000,
            "episodes": [10, 11]})

    def _mk(self, run, name):
        p = os.path.join(self.root, run, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _codes(self, rep):
        by = {}
        for f in rep["flags"]:
            by.setdefault(f["code"], []).append(f["run_id"])
        return by

    def test_all_defect_classes_flagged(self):
        rep = audit.audit_tree(self.root)
        self.assertEqual(rep["n_runs"], 7)
        codes = self._codes(rep)
        self.assertIn("SEED_ABSENT", codes)          # the real eval_info
        self.assertEqual(codes["SEED_DEFAULT_1000"], ["runB"])
        self.assertEqual(codes["TRUNCATED"], ["runC"])
        self.assertIn("35000", rep["flags"][
            [f["code"] for f in rep["flags"]].index("TRUNCATED")]["message"])
        self.assertEqual(sorted(codes["REPLICATE_SET"]), ["runD", "runE"])
        self.assertEqual(codes["N_EVAL_MISSING"], ["runF"])
        self.assertIn("runG", codes["TINY_N"])

    def test_replicate_group_uses_set_hash(self):
        from orbit_eval.stats import set_hash
        rep = audit.audit_tree(self.root)
        h = set_hash([7, 8, 9])
        self.assertIn(h, rep["replicate_groups"])
        self.assertEqual(sorted(rep["replicate_groups"][h]), ["runD", "runE"])

    def test_clean_tree_has_no_flags(self):
        clean = tempfile.mkdtemp(prefix="orbit_eval_clean_")
        try:
            _write(os.path.join(clean, "run1", "model_result.json"), {
                "mask_id": "run1", "sr": 42.0, "n_eval": 500, "seed": 7,
                "final_step": 100000, "design_steps": 100000,
                "episodes": [1, 2, 3]})
            rep = audit.audit_tree(clean)
            self.assertEqual(rep["n_runs"], 1)
            self.assertEqual(rep["flags"], [])
        finally:
            shutil.rmtree(clean, ignore_errors=True)

    def test_fixture_dir_parses_all_four_files(self):
        # 46 (paired1_results) + 1 (model_result_p_b0_m0) + 1 (cupid) +
        # 1 (dropgate) — the *_eval_info.json names must NOT be skipped
        rep = audit.audit_tree(FIX)
        self.assertEqual(rep["n_files"], 4)
        self.assertEqual(rep["n_runs"], 49)
        self.assertEqual(rep["errors"], [])

    def test_human_report_and_json_shape(self):
        rep = audit.audit_tree(self.root)
        text = audit.format_report(rep)
        self.assertIn("TRUNCATED", text)
        self.assertIn("seed", text.lower())
        self.assertIn("replicate groups", text)
        # JSON-serialisable end to end
        json.dumps(rep)


if __name__ == "__main__":
    unittest.main()
