"""gate.records: canonical hashing, frozen GateConfig defaults, ledger audit.

Hash determinism is the whole point of the decision record: the same inputs
must hash to the same digest on any machine, in any process, at any time (no
clock reads, no dict-order dependence), and verify_ledger must catch a single
tampered byte on disk with a 1-based line number.
"""

import dataclasses
import json
import os
import tempfile
import unittest

import orbit_eval
from orbit_eval.gate.records import (
    SCHEMA_VERSION, GateConfig, append_record, build_record, canonical_json,
    iter_ledger, record_hash, sha256_hex, verify_ledger)

# The 24 spec-frozen GateConfig defaults, asserted literally below
# (18 from v0 + comparison_level / sigma_run_pp / sigma_run_df from the
# v0.1 retrain-aware spec + sequential_tau from the frozen v1 sequential
# spec).
FROZEN_DEFAULTS = {
    "alpha": 0.05,
    "min_effect_pp": 2.0,
    "regression_alpha": 0.05,
    "regression_min_pp": 5.0,
    "selection": "final",
    "n_checkpoints": None,
    "curse_prior_pp": 4.0,
    "curse_prior_j": 4,
    "max_budget": None,
    "n_tasks": None,
    "task_blocks": None,
    "task_labels": None,
    "assume_crn": False,
    "sequential_eval": False,
    "regime": None,
    "n_boot": 4000,
    "boot_seed": 7,
    "power_target": 0.8,
    "comparison_level": "checkpoint",
    "sigma_run_pp": None,
    "sigma_run_df": None,
    "sigma_pertask_pp": None,   # HOLDINJ 2026-08-15: per-task retrain pricing
    "sigma_pertask_df": None,
    "sequential_tau": 5.0,
}


class TestCanonicalJson(unittest.TestCase):
    def test_key_order_independence(self):
        a = {"a": 1, "b": [1, 2], "c": {"y": 0, "x": None}}
        b = {"c": {"x": None, "y": 0}, "b": [1, 2], "a": 1}
        self.assertEqual(canonical_json(a), canonical_json(b))

    def test_compact_separators_exact_bytes(self):
        self.assertEqual(canonical_json({"b": [1, 2], "a": True}),
                         '{"a":true,"b":[1,2]}')

    def test_ascii_only(self):
        # non-ASCII must be escaped so the byte stream is encoding-proof
        self.assertEqual(canonical_json({"a": "é"}), '{"a":"\\u00e9"}')

    def test_byte_determinism_across_calls(self):
        obj = {"z": 1.5, "a": {"k": [True, None, "s"]}, "n": 10}
        self.assertEqual(canonical_json(obj), canonical_json(obj))
        self.assertEqual(sha256_hex(canonical_json(obj)),
                         sha256_hex(canonical_json(obj)))

    def test_allow_nan_false_rejects_nan(self):
        with self.assertRaises(ValueError):
            canonical_json({"x": float("nan")})
        with self.assertRaises(ValueError):
            canonical_json({"x": float("inf")})

    def test_sha256_hex_known_value(self):
        # sha256 of the empty string — pins the digest algorithm itself
        self.assertEqual(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


class TestGateConfig(unittest.TestCase):
    def test_all_22_defaults_literal(self):
        d = GateConfig().to_dict()
        self.assertEqual(len(d), 24)
        self.assertEqual(set(d), set(FROZEN_DEFAULTS))
        for k, v in FROZEN_DEFAULTS.items():
            self.assertEqual(d[k], v, "default for %s" % k)
        # bool defaults must be real bools (False == 0 would slip through)
        self.assertIs(d["assume_crn"], False)
        self.assertIs(d["sequential_eval"], False)

    def test_to_dict_from_dict_round_trip(self):
        cfg = GateConfig(alpha=0.01, selection="max-over-checkpoints",
                         n_checkpoints=4, task_blocks=[20, 20, 160],
                         task_labels=["a", "b", "c"], n_tasks=3,
                         regime="smolvla-ft", max_budget=4000,
                         assume_crn=True, sequential_eval=True)
        back = GateConfig.from_dict(cfg.to_dict())
        self.assertEqual(back, cfg)
        self.assertEqual(back.sha256(), cfg.sha256())
        # to_dict deep-copies lists: mutating the dict must not touch cfg
        d = cfg.to_dict()
        d["task_blocks"].append(999)
        self.assertEqual(cfg.task_blocks, [20, 20, 160])

    def test_from_dict_partial_takes_defaults(self):
        cfg = GateConfig.from_dict({"alpha": 0.01})
        self.assertEqual(cfg.alpha, 0.01)
        self.assertEqual(cfg.min_effect_pp, 2.0)
        self.assertEqual(cfg.n_boot, 4000)

    def test_from_dict_unknown_key_raises(self):
        with self.assertRaises(ValueError) as cm:
            GateConfig.from_dict({"alpha": 0.05, "alfa": 0.01})
        self.assertIn("alfa", str(cm.exception))

    def test_sha256_stable(self):
        h = GateConfig().sha256()
        self.assertEqual(h, GateConfig().sha256())
        self.assertEqual(len(h), 64)
        self.assertEqual(h, sha256_hex(canonical_json(GateConfig().to_dict())))

    def test_sha256_changes_when_any_field_changes(self):
        base = GateConfig()
        h0 = base.sha256()
        changed = {
            "alpha": 0.01, "min_effect_pp": 3.0, "regression_alpha": 0.01,
            "regression_min_pp": 6.0, "selection": "max-over-checkpoints",
            "n_checkpoints": 4, "curse_prior_pp": 5.0, "curse_prior_j": 8,
            "max_budget": 1000, "n_tasks": 10, "task_blocks": [10, 10],
            "task_labels": ["x"], "assume_crn": True,
            "sequential_eval": True, "regime": "smolvla-ft", "n_boot": 2000,
            "boot_seed": 11, "power_target": 0.9,
            "comparison_level": "retrain", "sigma_run_pp": 3.31,
            "sigma_run_df": 6, "sigma_pertask_pp": 11.0,
            "sigma_pertask_df": 7, "sequential_tau": 3.0,
        }
        # every one of the 24 fields is covered, no more, no fewer
        self.assertEqual(set(changed),
                         {f.name for f in dataclasses.fields(GateConfig)})
        for name, val in changed.items():
            h = dataclasses.replace(base, **{name: val}).sha256()
            self.assertNotEqual(h, h0, "sha256 blind to field %s" % name)

    def test_validate_ok_on_defaults_and_full_config(self):
        self.assertEqual(GateConfig().validate(), [])
        full = GateConfig(alpha=0.05, min_effect_pp=2.0,
                          selection="max-over-checkpoints", n_checkpoints=4,
                          max_budget=4000, n_tasks=10,
                          task_blocks=[20] * 10,
                          task_labels=["t%d" % i for i in range(10)],
                          assume_crn=True, sequential_eval=True,
                          regime="smolvla-ft")
        self.assertEqual(full.validate(), [])

    def test_validate_flags(self):
        def problems(**kw):
            return dataclasses.replace(GateConfig(), **kw).validate()

        self.assertTrue(any("alpha" in m for m in problems(alpha=0.0)))
        self.assertTrue(any("alpha" in m for m in problems(alpha=0.5)))
        self.assertTrue(any("regression_alpha" in m
                            for m in problems(regression_alpha=0.7)))
        self.assertTrue(any("min_effect_pp" in m
                            for m in problems(min_effect_pp=0.0)))
        self.assertTrue(any("regression_min_pp" in m
                            for m in problems(regression_min_pp=-1.0)))
        self.assertTrue(any("selection" in m
                            for m in problems(selection="best")))
        self.assertTrue(any("n_checkpoints" in m
                            for m in problems(n_checkpoints=1)))
        self.assertTrue(any("task_blocks" in m
                            for m in problems(task_blocks=[0])))
        self.assertTrue(any("task_blocks" in m
                            for m in problems(task_blocks=[20, -5])))
        self.assertTrue(any("n_tasks" in m for m in problems(n_tasks=0)))
        self.assertTrue(any("n_boot" in m for m in problems(n_boot=0)))
        self.assertTrue(any("power_target" in m
                            for m in problems(power_target=1.0)))
        self.assertTrue(any("power_target" in m
                            for m in problems(power_target=0.0)))
        # one bad field -> exactly one problem, not a cascade
        self.assertEqual(len(problems(alpha=0.0)), 1)


def _record(**overrides):
    """A complete record with plausible content; overrides applied on top."""
    kw = dict(
        verdict="SHIP",
        config=overrides.pop("config", GateConfig()),
        inputs={"incumbent": {"run_id": "inc", "n_eval": 200,
                              "eval_seed": 1000},
                "candidate": {"run_id": "cand", "n_eval": 200,
                              "eval_seed": 1000},
                "n_paired": 200},
        statistics={"paired": True, "b": 4, "c": 25, "n_discordant": 29,
                    "delta_hat_pp": 10.5, "ci95_pp": [4.5, 16.5],
                    "p_ship": 5.19e-05},
        caveats=["caveat-1"],
        warnings=[],
        assumptions=["ASSUMPTION: task blocks contiguous equal"],
        reasons=["p_ship 5.19e-05 <= alpha 0.05"],
        created="2026-08-08T12:00:00Z",
    )
    kw.update(overrides)
    return build_record(**kw)


class TestBuildRecord(unittest.TestCase):
    def test_embedded_hash_matches_record_hash(self):
        rec = _record()
        self.assertEqual(rec["record_sha256"], record_hash(rec))
        # and record_hash is over the record MINUS its own hash field
        naked = {k: v for k, v in rec.items() if k != "record_sha256"}
        self.assertEqual(rec["record_sha256"],
                         sha256_hex(canonical_json(naked)))

    def test_identical_args_identical_bytes_proves_no_clock_read(self):
        a, b = _record(), _record()
        self.assertEqual(a["record_sha256"], b["record_sha256"])
        self.assertEqual(canonical_json(a), canonical_json(b))

    def test_created_stored_verbatim(self):
        rec = _record(created="2001-01-01T00:00:00Z")
        self.assertEqual(rec["created"], "2001-01-01T00:00:00Z")
        # and the timestamp is inside the hashed content
        self.assertNotEqual(rec["record_sha256"], _record()["record_sha256"])

    def test_gate_config_embedded_and_hashed(self):
        cfg = GateConfig(alpha=0.01, regime="smolvla-ft")
        rec = _record(config=cfg)
        self.assertEqual(rec["gate_config"], cfg.to_dict())
        self.assertEqual(rec["gate_config_sha256"], cfg.sha256())

    def test_schema_engine_version_and_key_set(self):
        rec = _record()
        self.assertEqual(rec["schema_version"], SCHEMA_VERSION)
        self.assertEqual(rec["engine_version"], orbit_eval.__version__)
        self.assertEqual(rec["verdict"], "SHIP")
        self.assertEqual(set(rec), {
            "schema_version", "engine_version", "created", "verdict",
            "reasons", "gate_config", "gate_config_sha256", "inputs",
            "statistics", "caveats", "warnings", "assumptions",
            "record_sha256"})

    def test_content_changes_change_the_hash(self):
        h0 = _record()["record_sha256"]
        self.assertNotEqual(_record(verdict="HOLD")["record_sha256"], h0)
        self.assertNotEqual(
            _record(statistics={"paired": False})["record_sha256"], h0)
        self.assertNotEqual(
            _record(config=GateConfig(alpha=0.01))["record_sha256"], h0)


class TestLedger(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(self.dir, "ledger.jsonl")

    def _append_three(self):
        recs = []
        for i, verdict in enumerate(("SHIP", "HOLD", "COLLECT-MORE")):
            rec = _record(verdict=verdict,
                          statistics={"delta_hat_pp": float(i)},
                          created="2026-08-08T0%d:00:00Z" % i)
            append_record(rec, self.path)
            recs.append(rec)
        return recs

    def test_append_creates_file_and_verify_ok(self):
        self.assertFalse(os.path.exists(self.path))
        self._append_three()
        res = verify_ledger(self.path)
        self.assertEqual(res["path"], self.path)
        self.assertEqual(res["n_records"], 3)
        self.assertEqual(res["n_ok"], 3)
        self.assertEqual(res["bad"], [])
        self.assertTrue(res["ok"])

    def test_ledger_lines_are_canonical_json(self):
        recs = self._append_three()
        with open(self.path) as fh:
            lines = fh.read().splitlines()
        self.assertEqual(lines, [canonical_json(r) for r in recs])

    def test_iter_ledger_yields_dicts_in_order(self):
        recs = self._append_three()
        got = list(iter_ledger(self.path))
        self.assertEqual(len(got), 3)
        for g in got:
            self.assertIsInstance(g, dict)
        # full round-trip through canonical JSON: content-identical, in order
        self.assertEqual(got, recs)
        self.assertEqual([g["verdict"] for g in got],
                         ["SHIP", "HOLD", "COLLECT-MORE"])

    def test_single_byte_tamper_detected_on_the_right_line(self):
        self._append_three()
        with open(self.path) as fh:
            lines = fh.readlines()
        # flip ONE byte in line 2: verdict "HOLD" -> "H0LD" (same length)
        self.assertIn('"verdict":"HOLD"', lines[1])
        lines[1] = lines[1].replace('"verdict":"HOLD"', '"verdict":"H0LD"')
        with open(self.path, "w") as fh:
            fh.writelines(lines)
        res = verify_ledger(self.path)
        self.assertFalse(res["ok"])
        self.assertEqual(res["n_records"], 3)
        self.assertEqual(res["n_ok"], 2)
        self.assertEqual([b["line"] for b in res["bad"]], [2])
        self.assertIn("hash", res["bad"][0]["reason"].lower())

    def test_gate_config_tamper_flags_config_hash_too(self):
        self._append_three()
        with open(self.path) as fh:
            lines = fh.readlines()
        # rewrite a threshold inside the embedded gate_config on line 1
        self.assertIn('"alpha":0.05', lines[0])
        lines[0] = lines[0].replace('"alpha":0.05', '"alpha":0.15', 1)
        with open(self.path, "w") as fh:
            fh.writelines(lines)
        res = verify_ledger(self.path)
        self.assertFalse(res["ok"])
        self.assertEqual([b["line"] for b in res["bad"]], [1])
        self.assertIn("gate_config", res["bad"][0]["reason"])
        self.assertIn("hash", res["bad"][0]["reason"].lower())

    def test_malformed_json_line_flagged_not_raised(self):
        self._append_three()
        with open(self.path, "a") as fh:
            fh.write("{this is not json}\n")
        res = verify_ledger(self.path)
        self.assertFalse(res["ok"])
        self.assertEqual(res["n_records"], 4)
        self.assertEqual(res["n_ok"], 3)
        self.assertEqual([b["line"] for b in res["bad"]], [4])
        self.assertIn("JSON", res["bad"][0]["reason"])

    def test_missing_hash_field_flagged(self):
        rec = _record()
        del rec["record_sha256"]
        with open(self.path, "w") as fh:
            fh.write(canonical_json(rec) + "\n")
        res = verify_ledger(self.path)
        self.assertFalse(res["ok"])
        self.assertIn("record_sha256", res["bad"][0]["reason"])

    def test_non_object_line_flagged(self):
        with open(self.path, "w") as fh:
            fh.write("[1,2,3]\n")
        res = verify_ledger(self.path)
        self.assertFalse(res["ok"])
        self.assertEqual(res["n_records"], 1)
        self.assertIn("object", res["bad"][0]["reason"])

    def test_blank_lines_skipped_line_numbers_still_physical(self):
        recs = self._append_three()
        with open(self.path) as fh:
            content = fh.read()
        lines = content.splitlines()
        # insert a blank line between records 1 and 2, then tamper record 3
        # (now on physical line 4)
        lines.insert(1, "")
        lines[3] = lines[3].replace('"verdict":"COLLECT-MORE"',
                                    '"verdict":"COLLECT-M0RE"')
        with open(self.path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        res = verify_ledger(self.path)
        self.assertEqual(res["n_records"], 3)
        self.assertEqual(res["n_ok"], 2)
        self.assertEqual([b["line"] for b in res["bad"]], [4])
        self.assertEqual(len(list(iter_ledger(self.path))), 3)
        self.assertEqual(recs[0], list(iter_ledger(self.path))[0])

    def test_missing_path_raises_oserror(self):
        missing = os.path.join(self.dir, "nope.jsonl")
        with self.assertRaises(OSError):
            verify_ledger(missing)
        with self.assertRaises(OSError):
            iter_ledger(missing)

    def test_verify_survives_nan_smuggled_via_json(self):
        # json.loads accepts bare NaN; canonical_json (allow_nan=False) must
        # not crash verify_ledger — the line is reported bad instead
        with open(self.path, "w") as fh:
            fh.write('{"gate_config":{"alpha":NaN},'
                     '"gate_config_sha256":"00","record_sha256":"00"}\n')
        res = verify_ledger(self.path)
        self.assertFalse(res["ok"])
        self.assertEqual(res["n_records"], 1)
        self.assertEqual(res["n_ok"], 0)


if __name__ == "__main__":
    unittest.main()
