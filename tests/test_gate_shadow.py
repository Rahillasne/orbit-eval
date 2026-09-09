"""Contract tests for the WS3 shadow-gating layer: capture (chained ledger,
idempotency, provenance, G3 metadata), shadow (the shipped retrain-level
gate beside the team's declared decision, matched-config refusals,
divergence semantics), the report (the kill-criterion instrument), and
verify_chain tamper detection — plus the CLI wiring end to end.

Fixtures are DETERMINISTIC block constructions (no rng in the verdict
path): e.g. incumbent = 50 successes then failures, candidate = 80 then
failures, same eval seed -> b=0, c=30, delta exactly +30.0 pp — so every
asserted verdict is a consequence of the frozen decision rule, not of a
lucky draw.
"""

import contextlib
import io as _io
import json
import os
import shutil
import tempfile
import unittest

from orbit_eval.gate import cli, records, shadow
from orbit_eval.gate.engine import GateInputError
from orbit_eval.gate.records import GateConfig

NOW = "2026-08-15T00:00:00Z"


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


def _eval_info(tasks, seed=None):
    """tasks: [(task_group, task_id, successes), ...] in eval order."""
    per_task = [{"task_group": g, "task_id": t,
                 "metrics": {"successes": [bool(x) for x in s]}}
                for g, t, s in tasks]
    all_s = [x for _g, _t, s in tasks for x in s]
    obj = {"per_task": per_task,
           "overall": {"pc_success": 100.0 * sum(all_s) / len(all_s),
                       "n_episodes": len(all_s)}}
    if seed is not None:
        obj["seed"] = seed
    return obj


class _TmpMixin:
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="shipgate_shadow_")
        self.ledger = os.path.join(self.tmp, "shadow_ledger.jsonl")
        self.tree = os.path.join(self.tmp, "wave")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_json(self, path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(obj, fh)
        return path

    def _mk_run(self, name, evals, sidecar=None, seed=1000):
        """One run dir under the tree: evals = {eval_name: tasks}."""
        run_dir = os.path.join(self.tree, "runs", name)
        for eval_name, tasks in evals.items():
            self._write_json(os.path.join(run_dir, eval_name,
                                          "eval_info.json"),
                             _eval_info(tasks, seed=seed))
        if sidecar is not None:
            self._write_json(os.path.join(run_dir, "model_result.json"),
                             sidecar)
        return run_dir

    def _ledger_lines(self):
        with open(self.ledger) as fh:
            return [ln.rstrip("\n") for ln in fh if ln.strip()]


# ---------------------------------------------------------------- capture

class TestCaptureLedger(_TmpMixin, unittest.TestCase):
    def _two_run_tree(self):
        side = {"mask_id": "maskA", "eval_seed": 1000, "seed": 7,
                "set_hash": "cafecafecafecafe", "final_step": 20000,
                "design_steps": 20000, "policy_class": "dp",
                "suite": "pusht"}
        self._mk_run("runA",
                     {"eval_005000": [("pusht", 0, _blocks(20, 10))],
                      "eval_final": [("pusht", 0, _blocks(20, 12))]},
                     sidecar=side, seed=None)
        self._mk_run("runB",
                     {"eval_005000": [("pusht", 0, _blocks(20, 8))],
                      "eval_final": [("pusht", 0, _blocks(20, 11))]},
                     sidecar=dict(side, mask_id="maskB"), seed=None)

    def test_capture_counts_chain_and_idempotency(self):
        self._two_run_tree()
        res = shadow.capture_tree(self.tree, self.ledger, now=NOW)
        self.assertEqual((res["captured"], res["skipped_duplicate"],
                          res["unreadable"]), (4, 0, 0))
        lines = self._ledger_lines()
        self.assertEqual(len(lines), 4)
        first = json.loads(lines[0])
        self.assertEqual(first["prev_sha256"], shadow.GENESIS_SHA256)
        second = json.loads(lines[1])
        self.assertEqual(second["prev_sha256"],
                         records.sha256_hex(lines[0]))
        rep = shadow.verify_chain(self.ledger)
        self.assertTrue(rep["ok"], msg=rep)
        self.assertEqual(rep["n_records"], 4)
        with open(self.ledger, "rb") as fh:
            before = fh.read()
        res2 = shadow.capture_tree(self.tree, self.ledger, now=NOW)
        self.assertEqual((res2["captured"], res2["skipped_duplicate"]),
                         (0, 4))
        with open(self.ledger, "rb") as fh:
            self.assertEqual(fh.read(), before)   # append-only, no dupes

    def test_capture_record_fields(self):
        self._two_run_tree()
        res = shadow.capture_tree(self.tree, self.ledger, now=NOW)
        by_key = {(r["run_id"], r["eval_name"]): r for r in res["records"]}
        snap = by_key[("maskA", "eval_005000")]
        self.assertEqual(snap["record_type"], "capture-eval")
        self.assertEqual(snap["step"], 5000)
        self.assertEqual(snap["n_eval"], 20)
        self.assertEqual(snap["successes"], "1" * 10 + "0" * 10)
        self.assertEqual(snap["eval_seed"], 1000)
        self.assertEqual(snap["eval_seed_provenance"],
                         shadow.SEED_SIDECAR)
        self.assertEqual(snap["train_seed"], 7)   # 'seed' disambiguated by
        self.assertEqual(snap["set_hash"], "cafecafecafecafe")
        self.assertEqual(snap["task_blocks"], [20])
        self.assertEqual(snap["task_order"], ["pusht:0"])
        self.assertEqual(snap["n_tasks"], 1)
        self.assertEqual(snap["content_sha256"],
                         shadow.content_sha256(snap))
        final = by_key[("maskA", "eval_final")]
        self.assertEqual(final["step"], 20000)    # sidecar final_step
        self.assertEqual(final["design_steps"], 20000)

    def test_eval_seed_provenance_ladder(self):
        # (a) recorded in eval_info beats the sidecar; (b) neither -> null
        # with the lerobot-default disclosure, never a fabricated 1000
        self._mk_run("recorded",
                     {"eval_final": [("pusht", 0, _blocks(10, 5))]},
                     sidecar={"mask_id": "rec", "eval_seed": 1000},
                     seed=42)
        self._mk_run("implicit",
                     {"eval_final": [("pusht", 0, _blocks(10, 5))]},
                     sidecar=None, seed=None)
        res = shadow.capture_tree(self.tree, self.ledger, now=NOW)
        by_id = {r["run_id"]: r for r in res["records"]}
        self.assertEqual(by_id["rec"]["eval_seed"], 42)
        self.assertEqual(by_id["rec"]["eval_seed_provenance"],
                         shadow.SEED_RECORDED)
        self.assertIsNone(by_id["implicit"]["eval_seed"])
        self.assertEqual(by_id["implicit"]["eval_seed_provenance"],
                         shadow.SEED_IMPLICIT)
        # no sidecar -> run_id falls back to the run dir name
        self.assertIn("implicit", by_id)

    def test_unreadable_disclosed_not_fatal(self):
        self._two_run_tree()
        bad = os.path.join(self.tree, "runs", "runA", "eval_005000",
                           "eval_info.json")
        with open(bad, "w") as fh:
            fh.write("{not json")
        res = shadow.capture_tree(self.tree, self.ledger, now=NOW)
        self.assertEqual(res["unreadable"], 1)
        self.assertEqual(res["captured"], 3)      # the other 3 still land
        self.assertIn(bad, res["unreadable_files"][0]["path"])
        self.assertTrue(shadow.verify_chain(self.ledger)["ok"])

    def test_corrupt_sidecar_disclosed_evals_kept(self):
        self._mk_run("runA",
                     {"eval_final": [("pusht", 0, _blocks(10, 5))]},
                     sidecar=None, seed=1000)
        with open(os.path.join(self.tree, "runs", "runA",
                               "model_result.json"), "w") as fh:
            fh.write("{nope")
        res = shadow.capture_tree(self.tree, self.ledger, now=NOW)
        self.assertEqual(res["unreadable"], 1)    # the sidecar
        self.assertEqual(res["captured"], 1)      # the eval still captured

    def test_empty_tree_is_an_error(self):
        os.makedirs(self.tree)
        with self.assertRaises(ValueError):
            shadow.capture_tree(self.tree, self.ledger, now=NOW)

    def test_deterministic_records_given_now(self):
        self._two_run_tree()
        other = os.path.join(self.tmp, "other_ledger.jsonl")
        shadow.capture_tree(self.tree, self.ledger, now=NOW)
        shadow.capture_tree(self.tree, other, now=NOW)
        with open(self.ledger, "rb") as fa, open(other, "rb") as fb:
            self.assertEqual(fa.read(), fb.read())

    def test_chain_tamper_detected(self):
        self._two_run_tree()
        shadow.capture_tree(self.tree, self.ledger, now=NOW)
        lines = self._ledger_lines()
        tampered = lines[0].replace('"sr_pp":', '"sr_pp_x":', 1)
        with open(self.ledger, "w") as fh:
            fh.write("\n".join([tampered] + lines[1:]) + "\n")
        rep = shadow.verify_chain(self.ledger)
        self.assertFalse(rep["ok"])
        reasons = " ".join(b["reason"] for b in rep["bad"])
        self.assertIn("record_sha256", reasons)
        self.assertIn("chain break", reasons)     # line 2's prev now dangles

    def test_chain_removal_detected(self):
        self._two_run_tree()
        shadow.capture_tree(self.tree, self.ledger, now=NOW)
        lines = self._ledger_lines()
        with open(self.ledger, "w") as fh:       # drop line 2
            fh.write("\n".join([lines[0]] + lines[2:]) + "\n")
        rep = shadow.verify_chain(self.ledger)
        self.assertFalse(rep["ok"])
        self.assertIn("chain break", rep["bad"][0]["reason"])

    def test_read_chain_refuses_corrupt_ledger(self):
        self._two_run_tree()
        shadow.capture_tree(self.tree, self.ledger, now=NOW)
        with open(self.ledger, "a") as fh:
            fh.write("{garbage\n")
        with self.assertRaises(ValueError):
            shadow.read_chain(self.ledger)
        with self.assertRaises(ValueError):       # capture must not extend it
            shadow.capture_tree(self.tree, self.ledger, now=NOW)


# ---------------------------------------------------------------- shadow

class TestShadowPair(_TmpMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.cfg = GateConfig(n_boot=150)
        side = {"final_step": 20000, "design_steps": 20000}
        self._mk_run("inc",
                     {"eval_final": [("pusht", 0, _blocks(100, 50))]},
                     sidecar=dict(side, mask_id="inc"))
        self._mk_run("cand",
                     {"eval_final": [("pusht", 0, _blocks(100, 80))]},
                     sidecar=dict(side, mask_id="cand"))
        self._mk_run("worse",
                     {"eval_final": [("pusht", 0, _blocks(100, 20))]},
                     sidecar=dict(side, mask_id="worse"))
        self._mk_run("twotask",
                     {"eval_final": [("pusht", 0, _blocks(50, 25)),
                                     ("pusht", 1, _blocks(50, 25))]},
                     sidecar=dict(side, mask_id="twotask"))
        shadow.capture_tree(self.tree, self.ledger, now=NOW)

    def _shadow(self, inc, cand, decided="none", regime=None):
        return shadow.shadow_pair(self.ledger, inc, cand, decided=decided,
                                  regime=regime, config=self.cfg, now=NOW)

    def test_ship_agree(self):
        # b=0, c=30, delta +30 pp: SHIP under the frozen rule
        rec = self._shadow("inc", "cand", decided="promote")
        self.assertEqual(rec["record_type"], "shadow-decision")
        self.assertEqual(rec["gate_verdict"], "SHIP")
        self.assertTrue(rec["matched_config"])
        self.assertEqual(rec["divergence"], shadow.AGREE)
        gr = rec["gate_record"]
        self.assertEqual(gr["gate_config"]["comparison_level"], "retrain")
        self.assertEqual(gr["gate_config"]["selection"], "final")
        self.assertEqual(gr["gate_config"]["task_blocks"], [100])
        # the embedded gate record is itself a signed, verifiable record
        self.assertEqual(gr["record_sha256"], records.record_hash(gr))
        self.assertTrue(shadow.verify_chain(self.ledger)["ok"])

    def test_reversed_ship_decision(self):
        rec = self._shadow("inc", "worse", decided="promote")
        self.assertEqual(rec["gate_verdict"], "HOLD")
        self.assertEqual(rec["divergence"],
                         shadow.GATE_HOLD_TEAM_PROMOTED)

    def test_gate_ship_team_held(self):
        rec = self._shadow("inc", "cand", decided="hold")
        self.assertEqual(rec["divergence"], shadow.GATE_SHIP_TEAM_HELD)

    def test_no_declared_decision_no_divergence(self):
        rec = self._shadow("inc", "cand", decided="none")
        self.assertIsNone(rec["divergence"])

    def test_task_set_mismatch_refused_recorded_nonblocking(self):
        rec = self._shadow("inc", "twotask", decided="promote")
        self.assertEqual(rec["gate_verdict"], "INVALID")
        self.assertFalse(rec["matched_config"])
        self.assertIsNone(rec["gate_record"])
        self.assertIn("task sets differ", rec["config_mismatch"][0])
        # a promote against a refusal still counts as a reversal, and the
        # verdict rides along so it stays distinguishable from a HOLD
        self.assertEqual(rec["divergence"],
                         shadow.GATE_HOLD_TEAM_PROMOTED)
        self.assertTrue(shadow.verify_chain(self.ledger)["ok"])

    def test_per_task_count_mismatch_refused(self):
        side = {"final_step": 20000, "design_steps": 20000}
        self._mk_run("blkA",
                     {"eval_final": [("pusht", 0, _blocks(10, 5)),
                                     ("pusht", 1, _blocks(30, 15))]},
                     sidecar=dict(side, mask_id="blkA"))
        self._mk_run("blkB",
                     {"eval_final": [("pusht", 0, _blocks(20, 10)),
                                     ("pusht", 1, _blocks(20, 10))]},
                     sidecar=dict(side, mask_id="blkB"))
        shadow.capture_tree(self.tree, self.ledger, now=NOW)
        rec = self._shadow("blkA", "blkB")
        self.assertEqual(rec["gate_verdict"], "INVALID")
        self.assertIn("per-task episode counts differ",
                      rec["config_mismatch"][0])

    def test_block_order_mismatch_refused(self):
        # same task set, same n per block — only the ORDER differs: the
        # engine's total-n and task_groups checks cannot see this; the
        # captured G3 metadata is the only instrument that can
        side = {"final_step": 20000, "design_steps": 20000}
        self._mk_run("ordA",
                     {"eval_final": [("pusht", 0, _blocks(20, 10)),
                                     ("pusht", 1, _blocks(20, 10))]},
                     sidecar=dict(side, mask_id="ordA"))
        self._mk_run("ordB",
                     {"eval_final": [("pusht", 1, _blocks(20, 10)),
                                     ("pusht", 0, _blocks(20, 10))]},
                     sidecar=dict(side, mask_id="ordB"))
        shadow.capture_tree(self.tree, self.ledger, now=NOW)
        rec = self._shadow("ordA", "ordB")
        self.assertEqual(rec["gate_verdict"], "INVALID")
        self.assertIn("ORDER differs", rec["config_mismatch"][0])

    def test_authoritative_eval_selection(self):
        # eval_final beats snapshots; without one the largest step wins
        self._mk_run("snaps",
                     {"eval_005000": [("pusht", 0, _blocks(100, 40))],
                      "eval_010000": [("pusht", 0, _blocks(100, 45))]},
                     sidecar=None)
        shadow.capture_tree(self.tree, self.ledger, now=NOW)
        rec = self._shadow("inc", "snaps")
        self.assertEqual(rec["candidate"]["eval_name"], "eval_010000")
        rec2 = self._shadow("inc", "cand")
        self.assertEqual(rec2["candidate"]["eval_name"], "eval_final")

    def test_unknown_run_id_lists_available(self):
        with self.assertRaises(ValueError) as ctx:
            self._shadow("inc", "nope")
        self.assertIn("available", str(ctx.exception))
        self.assertIn("cand", str(ctx.exception))

    def test_regime_resolution_includes_pertask_pricing(self):
        # the shipped engine, not a reimplementation: the regime's
        # sigma_run AND sigma_pertask both resolve into the gate record
        side = {"final_step": 20000, "design_steps": 20000}
        self._mk_run("mtA",
                     {"eval_final": [("objgrp", 0, _blocks(100, 50)),
                                     ("objgrp", 1, _blocks(100, 60))]},
                     sidecar=dict(side, mask_id="mtA"))
        self._mk_run("mtB",
                     {"eval_final": [("objgrp", 0, _blocks(100, 55)),
                                     ("objgrp", 1, _blocks(100, 40))]},
                     sidecar=dict(side, mask_id="mtB"))
        shadow.capture_tree(self.tree, self.ledger, now=NOW)
        rec = self._shadow("mtA", "mtB", regime="pi05-ft-k88-multitask")
        st = rec["gate_record"]["statistics"]
        self.assertEqual(st["sigma_run_pp_used"], 4.05)
        self.assertEqual(st["method"], "z-retrain-aware")
        for row in st["per_task"]:
            self.assertIn("p_retrain", row)       # per-task pricing is live
            self.assertIn("tier", row)

    def test_unknown_regime_is_an_input_error(self):
        with self.assertRaises(GateInputError):
            self._shadow("inc", "cand", regime="no-such-regime")

    def test_report_counts(self):
        self._shadow("inc", "cand", decided="promote")    # agree (SHIP)
        self._shadow("inc", "worse", decided="promote")   # reversed
        self._shadow("inc", "cand", decided="hold")       # gate-ship-held
        self._shadow("inc", "cand", decided="none")       # undeclared
        rep = shadow.shadow_report(self.ledger)
        self.assertEqual(rep["n_decisions"], 4)
        self.assertEqual(rep["n_declared"], 3)
        self.assertEqual(rep["n_undeclared"], 1)
        self.assertEqual(rep["n_agree"], 1)
        self.assertEqual(rep["n_gate_hold_team_promoted"], 1)
        self.assertEqual(rep["n_gate_ship_team_held"], 1)
        self.assertEqual(rep["verdict_counts"]["SHIP"], 3)
        self.assertEqual(rep["verdict_counts"]["HOLD"], 1)

    def test_capture_after_shadow_extends_the_same_chain(self):
        self._shadow("inc", "cand", decided="promote")
        self._mk_run("late",
                     {"eval_final": [("pusht", 0, _blocks(100, 60))]},
                     sidecar={"mask_id": "late", "final_step": 20000,
                              "design_steps": 20000})
        res = shadow.capture_tree(self.tree, self.ledger, now=NOW)
        self.assertEqual(res["captured"], 1)
        self.assertTrue(shadow.verify_chain(self.ledger)["ok"])


# ---------------------------------------------------------------- CLI

class TestShadowCLI(_TmpMixin, unittest.TestCase):
    def _tree_pair(self):
        side = {"final_step": 20000, "design_steps": 20000}
        self._mk_run("inc",
                     {"eval_final": [("pusht", 0, _blocks(100, 50))]},
                     sidecar=dict(side, mask_id="inc"))
        self._mk_run("worse",
                     {"eval_final": [("pusht", 0, _blocks(100, 20))]},
                     sidecar=dict(side, mask_id="worse"))

    def test_capture_shadow_report_end_to_end(self):
        self._tree_pair()
        code, out, err = _run(["capture", self.tree,
                               "--ledger", self.ledger])
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("captured=2 skipped-duplicate=0 unreadable=0", out)
        # gate says HOLD, team promoted: non-blocking exit 0 regardless
        code, out, err = _run(["shadow", "--ledger", self.ledger,
                               "--incumbent", "inc",
                               "--candidate", "worse",
                               "--decided", "promote"])
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("gate verdict  HOLD", out)
        self.assertIn("gate-hold-team-promoted", out)
        self.assertIn("NON-BLOCKING", out)
        code, out, err = _run(["shadow", "--report",
                               "--ledger", self.ledger])
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("gate-said-HOLD-team-promoted  1", out)
        self.assertIn("reversed ship decisions", out)
        self.assertIn("kill criterion", out)

    def test_shadow_json_record(self):
        self._tree_pair()
        _run(["capture", self.tree, "--ledger", self.ledger])
        code, out, _ = _run(["shadow", "--ledger", self.ledger,
                             "--incumbent", "inc", "--candidate", "worse",
                             "--json"])
        self.assertEqual(code, 0)
        rec = json.loads(out)
        self.assertEqual(rec["record_type"], "shadow-decision")
        self.assertEqual(rec["gate_verdict"], "HOLD")
        code, out, _ = _run(["shadow", "--report", "--ledger", self.ledger,
                             "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["n_decisions"], 1)

    def test_capture_unusable_tree_exit_2(self):
        os.makedirs(self.tree)
        code, _out, err = _run(["capture", self.tree,
                                "--ledger", self.ledger])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_shadow_missing_pair_and_report_exit_2(self):
        self._tree_pair()
        _run(["capture", self.tree, "--ledger", self.ledger])
        code, _out, err = _run(["shadow", "--ledger", self.ledger])
        self.assertEqual(code, 2)
        self.assertIn("--incumbent", err)

    def test_shadow_unknown_run_exit_2(self):
        self._tree_pair()
        _run(["capture", self.tree, "--ledger", self.ledger])
        code, _out, err = _run(["shadow", "--ledger", self.ledger,
                                "--incumbent", "inc",
                                "--candidate", "ghost"])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)

    def test_report_missing_ledger_exit_2(self):
        code, _out, err = _run(["shadow", "--report", "--ledger",
                                os.path.join(self.tmp, "absent.jsonl")])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", err)


if __name__ == "__main__":
    unittest.main()
