# orbit-eval — honest statistics for robot-policy evaluation.
# Copyright (C) 2026 ORBIT Research
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the license along with this program. If not,
# see <https://www.gnu.org/licenses/>.
#
# A commercial license, exempting you from the AGPL's source-disclosure terms, is
# available from ORBIT Research.

"""Shadow-gating layer: capture LeRobot output trees into a sha256-CHAINED
ledger, then run the SHIPPED gate() on captured pairs NEXT TO the team's real
decisions — the WS3 instrument for the thesis kill criterion ('zero reversed
ship decisions in 3 months of shadow-gating' kills the product).

Two moves, one append-only ledger:

  CAPTURE  walk a training-output tree (adapter.find_run_evals — the same
           eval*/eval_info.json discovery gating uses) and append one
           'capture-eval' record per checkpoint eval found: run_id, the
           per-episode success vector ('0'/'1' string, the corpus
           convention), n_eval, the eval seed WITH provenance (recorded in
           eval_info.json; else the run dir's model_result.json sidecar;
           else null with 'lerobot_default_1000_implicit' — LeRobot's
           silent default is DISCLOSED, never written down as if it had
           been recorded), steps, set_hash, and the G3 ordering metadata
           (task count, per-task episode counts, task-block order) that
           decides matched-config pairability later. IDEMPOTENT: records
           dedupe on content_sha256 over the run_id + eval identity, so
           re-capturing the same tree appends nothing.

  SHADOW   load two captured records (each run's authoritative eval:
           eval_final, else the largest step — the find_output_eval rule),
           REFUSE via the verdict-'INVALID' path — still recorded, still
           non-blocking — when the G3 metadata says the pair is not
           matched-config (different task sets, block order, or per-task
           episode counts: the total-n equality the engine checks cannot
           see a re-ordered or re-split eval), else run the SHIPPED
           engine.gate() at comparison_level='retrain' (regime resolution
           included — sigma_run AND the per-task sigma_pertask pricing),
           and append a 'shadow-decision' record: gate verdict + reasons +
           the team's DECLARED decision + the divergence between them.
           NON-BLOCKING by contract: shadow never gates anything for real —
           it only writes down what the gate WOULD have said.

CHAIN: every record carries prev_sha256 = sha256 of the previous ledger LINE
(canonical JSON, newline stripped); the first record chains from
GENESIS_SHA256 (64 zeros). record_sha256 is the usual content signature
(records.record_hash) over the record INCLUDING prev_sha256 — a record
cannot be re-signed without breaking the next record's chain link.
verify_chain() recomputes every link and trusts nothing on disk. The chain
runs over line BYTES, so it stays checkable across a malformed line.

Divergence semantics (frozen for the instrument): SHIP is the only 'gate
would promote'; every other verdict (HOLD / COLLECT-MORE / INVALID /
UNRESOLVABLE) is 'gate would not certify this promotion', so a declared
promote against any of them counts as a REVERSED SHIP DECISION
(gate-hold-team-promoted). The verdict itself rides along in the record, so
an INVALID-driven reversal stays distinguishable from a statistical HOLD.

Determinism: no clock reads inside record builders — capture_tree and
shadow_pair take now= (stamped once, verbatim into every record of the
call) exactly like engine.gate and replay.run_replay.
"""

import dataclasses
import json
import os
from collections import Counter
from datetime import datetime, timezone

from .. import io, __version__ as _ENGINE_VERSION
from . import adapter
from .engine import gate
from .records import (GateConfig, append_record, canonical_json, record_hash,
                      sha256_hex)

# Chain genesis: the first record's prev_sha256 (64 zeros, frozen).
GENESIS_SHA256 = "0" * 64

# Schema of the capture/shadow ledger records ('capture-eval' /
# 'shadow-decision'); independent of records.SCHEMA_VERSION — these are not
# gate decision records, though a shadow record EMBEDS one.
SHADOW_SCHEMA_VERSION = 1

# eval_seed_provenance values (frozen): where the recorded eval seed came
# from — or the disclosure that NOTHING recorded it and LeRobot's silent
# default (seed 1000) is the only candidate. In the implicit case eval_seed
# stays null: writing 1000 down would let the engine 'verify' CRN that was
# never verified (that is exactly what assume_crn is for).
SEED_RECORDED = "recorded"
SEED_SIDECAR = "model_result_sidecar"
SEED_IMPLICIT = "lerobot_default_1000_implicit"

# Divergence values (frozen; see the module docstring for the semantics).
AGREE = "agree"
GATE_HOLD_TEAM_PROMOTED = "gate-hold-team-promoted"
GATE_SHIP_TEAM_HELD = "gate-ship-team-held"

_SHADOW_ASSUMPTION = (
    "ASSUMPTION: shadow-gated from captured ledger records (recorded "
    "per-episode vectors, not a live eval); the verdict is advisory by "
    "contract — the shadow never blocks a promotion")

# The eval-identity fields the idempotency key hashes over. 'created' and
# the path fields are deliberately EXCLUDED: re-capturing the same eval
# tomorrow, or from a moved copy of the tree, must still dedupe.
_CONTENT_FIELDS = ("run_id", "eval_name", "step", "n_eval", "eval_seed",
                   "set_hash", "successes", "task_blocks", "task_order")


def content_sha256(rec):
    """Idempotency key: sha256 over the canonical eval-identity fields
    (_CONTENT_FIELDS) of a capture record."""
    return sha256_hex(canonical_json(
        {k: rec.get(k) for k in _CONTENT_FIELDS}))


# ---------------------------------------------------------------- chain

def read_chain(ledger_path):
    """Parse an existing capture/shadow ledger for APPENDING onto it ->
    (records, capture_content_hashes, prev_sha256). A missing file is an
    empty chain: ([], set(), GENESIS_SHA256). A malformed line RAISES
    ValueError with its 1-based line number — extending a corrupt chain
    would sign garbage; verify_chain() is the survey-without-raising tool."""
    if not os.path.exists(ledger_path):
        return [], set(), GENESIS_SHA256
    recs, hashes, prev = [], set(), GENESIS_SHA256
    with open(ledger_path) as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.rstrip("\n")
            if not text.strip():
                continue
            try:
                rec = json.loads(text)
            except ValueError as e:
                raise ValueError(
                    "%s:%d: shadow ledger line is not valid JSON (%s) — "
                    "refusing to extend a corrupt chain (verify_chain "
                    "surveys the damage without raising)"
                    % (ledger_path, lineno, e))
            if not isinstance(rec, dict):
                raise ValueError("%s:%d: shadow ledger line is not a JSON "
                                 "object record" % (ledger_path, lineno))
            recs.append(rec)
            if rec.get("record_type") == "capture-eval" \
                    and rec.get("content_sha256"):
                hashes.add(rec["content_sha256"])
            prev = sha256_hex(text)
    return recs, hashes, prev


def append_chained(record, ledger_path, prev_sha256):
    """Chain-link one record and append it: prev_sha256 stored verbatim,
    record_sha256 computed OVER it (records.record_hash), the line written
    via records.append_record (single O_APPEND write). Returns
    (signed_record, line_sha256) — the second value is the next record's
    prev. Read-then-append is not atomic across processes: a concurrent
    writer forks the chain, which verify_chain() then reports."""
    rec = dict(record)
    rec["prev_sha256"] = prev_sha256
    rec["record_sha256"] = record_hash(rec)
    append_record(rec, ledger_path)
    return rec, sha256_hex(canonical_json(rec))


def verify_chain(ledger_path):
    """Re-verify EVERY line of a capture/shadow ledger: the prev_sha256
    chain link (sha256 of the previous LINE; GENESIS for the first), the
    record_sha256 content signature, and — for capture-eval records — the
    content_sha256 idempotency key. Trusts nothing on disk. Returns
      {'path', 'n_records', 'n_ok', 'bad': [{'line': 1-based, 'reason'}], 'ok'}
    Raises OSError if the ledger itself is unreadable."""
    bad, n_records = [], 0
    expected_prev = GENESIS_SHA256
    with open(ledger_path) as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.rstrip("\n")
            if not text.strip():
                continue
            n_records += 1
            this_prev, expected_prev = expected_prev, sha256_hex(text)
            try:
                rec = json.loads(text)
            except ValueError as e:
                bad.append({"line": lineno,
                            "reason": "malformed JSON: %s" % e})
                continue
            if not isinstance(rec, dict):
                bad.append({"line": lineno,
                            "reason": "not a JSON object record"})
                continue
            if rec.get("prev_sha256") != this_prev:
                bad.append({"line": lineno,
                            "reason": "prev_sha256 chain break: stored %r, "
                                      "expected %s — a line above was "
                                      "altered, removed, or reordered"
                                      % (rec.get("prev_sha256"), this_prev)})
                continue
            try:
                got = record_hash(rec)
            except (TypeError, ValueError) as e:
                bad.append({"line": lineno,
                            "reason": "record not canonically hashable: %s"
                                      % e})
                continue
            if rec.get("record_sha256") != got:
                bad.append({"line": lineno,
                            "reason": "record_sha256 hash mismatch: stored "
                                      "%r, recomputed %s — content was "
                                      "altered after signing"
                                      % (rec.get("record_sha256"), got)})
                continue
            if rec.get("record_type") == "capture-eval" \
                    and rec.get("content_sha256") != content_sha256(rec):
                bad.append({"line": lineno,
                            "reason": "content_sha256 mismatch: the eval "
                                      "identity fields were altered after "
                                      "capture"})
    return {"path": ledger_path, "n_records": n_records,
            "n_ok": n_records - len(bad), "bad": bad, "ok": not bad}


# ---------------------------------------------------------------- capture

def _int_or_none(v, ctx):
    """Coerce a sidecar int field (the ORBIT harvester stores ints as
    STRINGS — the replay module measured this on the real files); garbage
    raises with the offending field named."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError("%s=%r is not an integer" % (ctx, v))


def _sidecar_meta(run_dir):
    """Raw dict of the run dir's model_result.json sidecar (the ORBIT
    harness drops one beside the eval snapshots), or ({}, error). Absence
    is normal ({} with no error); a sidecar that exists but does not parse
    is returned as an error STRING so capture can disclose it in the
    unreadable count without dropping the run's (valid) eval records."""
    p = os.path.join(run_dir, "model_result.json")
    if not os.path.isfile(p):
        return {}, None
    try:
        with open(p) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        return {}, "%s: %s" % (p, e)
    if not isinstance(d, dict):
        return {}, "%s: not a JSON object" % p
    return d, None


def _g3_metadata(path):
    """G3 ordering metadata from a raw eval_info.json -> (n_tasks,
    task_blocks, task_order) or (None, None, None) when the file carries no
    complete per_task successes structure. task_order entries are
    '<task_group>:<task_id>' in FILE order — the order IS the metadata
    (episode index i must mean the same task on both sides of a pair)."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None, None, None
    pts = d.get("per_task") if isinstance(d, dict) else None
    if not isinstance(pts, list) or not pts:
        return None, None, None
    blocks, order = [], []
    for pt in pts:
        if not isinstance(pt, dict):
            return None, None, None
        m = pt.get("metrics") or {}
        succ = m.get("successes")
        if not isinstance(succ, list):
            return None, None, None
        blocks.append(len(succ))
        order.append("%s:%s" % (pt.get("task_group"), pt.get("task_id")))
    return len(blocks), blocks, order


def _capture_record(created, root, run_dir, run_id, eval_name, step, path,
                    run, meta):
    """One 'capture-eval' record (unsigned — append_chained links and signs
    it). Eval-seed provenance ladder: eval_info.json itself; else the
    sidecar's eval_seed; else null + the lerobot-default disclosure. The
    sidecar's bare 'seed' is read as the TRAIN seed only when an explicit
    'eval_seed' key disambiguates it (the ORBIT model_result convention
    otherwise uses 'seed' FOR the eval seed — io.parse_model_result)."""
    if run.seed is not None:
        eval_seed, prov = int(run.seed), SEED_RECORDED
    elif meta.get("eval_seed") is not None:
        eval_seed, prov = _int_or_none(meta["eval_seed"],
                                       "%s: eval_seed" % run_dir), \
            SEED_SIDECAR
    else:
        eval_seed, prov = None, SEED_IMPLICIT
    final_step = _int_or_none(meta.get("final_step"),
                              "%s: final_step" % run_dir)
    design_steps = _int_or_none(meta.get("design_steps"),
                                "%s: design_steps" % run_dir)
    if step is None and eval_name == "eval_final":
        step = final_step
    train_seed = meta.get("train_seed")
    if train_seed is None and "eval_seed" in meta:
        train_seed = meta.get("seed")
    n_tasks, blocks, order = _g3_metadata(path)
    set_hash = meta.get("set_hash", run.set_hash)
    rec = {
        "record_type": "capture-eval",
        "schema_version": SHADOW_SCHEMA_VERSION,
        "engine_version": _ENGINE_VERSION,
        "created": created,
        "capture_root": root,
        "run_dir": run_dir,
        "run_id": run_id,
        "eval_name": eval_name,
        "eval_path": path,
        "step": step,
        "final_step": final_step,
        "design_steps": design_steps,
        "n_eval": run.n_eval,
        "sr_pp": run.sr,
        "successes": "".join("1" if s else "0" for s in run.successes)
        if run.successes is not None else None,
        "eval_seed": eval_seed,
        "eval_seed_provenance": prov,
        "set_hash": str(set_hash) if set_hash is not None else None,
        "train_seed": _int_or_none(train_seed, "%s: train_seed" % run_dir),
        "policy_class": meta.get("policy_class"),
        "suite": meta.get("suite"),
        "task_groups": list(run.task_groups or []),
        "n_tasks": n_tasks,
        "task_blocks": blocks,
        "task_order": order,
    }
    rec["content_sha256"] = content_sha256(rec)
    return rec


def capture_tree(root, ledger_path, now=None):
    """Walk a LeRobot training-output tree and append one 'capture-eval'
    record per eval*/eval_info.json to the chained ledger. Idempotent
    (content_sha256 dedupe against the EXISTING ledger); a corrupt snapshot
    or sidecar is counted unreadable and disclosed, never fatal. One
    'created' stamp for the whole sweep (``now`` verbatim, or current UTC
    computed exactly once). Returns
      {'root', 'ledger', 'captured', 'skipped_duplicate', 'unreadable',
       'unreadable_files': [{'path', 'error'}], 'records': [new records]}
    Raises ValueError when the tree holds no eval*/eval_info.json at all."""
    created = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_groups = adapter.find_run_evals(root)
    if not run_groups:
        raise ValueError("%s: no eval*/eval_info.json under this tree — "
                         "nothing to capture" % root)
    _recs, content_hashes, prev = read_chain(ledger_path)
    captured = skipped = 0
    unreadable, new_records = [], []
    for run_dir, evals in run_groups:
        meta, meta_err = _sidecar_meta(run_dir)
        if meta_err:
            unreadable.append({"path": os.path.join(run_dir,
                                                    "model_result.json"),
                               "error": meta_err})
        run_id = str(meta["mask_id"]) if meta.get("mask_id") \
            else os.path.basename(os.path.normpath(run_dir))
        for eval_name, step, path, runs, err in evals:
            if err:
                unreadable.append({"path": path, "error": err})
                continue
            if not runs or len(runs) != 1:
                unreadable.append({"path": path,
                                   "error": "holds %d runs, need exactly one"
                                            % len(runs or [])})
                continue
            try:
                rec = _capture_record(created, root, run_dir, run_id,
                                      eval_name, step, path, runs[0], meta)
            except ValueError as e:
                unreadable.append({"path": path, "error": str(e)})
                continue
            if rec["content_sha256"] in content_hashes:
                skipped += 1
                continue
            rec, prev = append_chained(rec, ledger_path, prev)
            content_hashes.add(rec["content_sha256"])
            new_records.append(rec)
            captured += 1
    return {"root": root, "ledger": ledger_path, "captured": captured,
            "skipped_duplicate": skipped, "unreadable": len(unreadable),
            "unreadable_files": unreadable, "records": new_records}


# ---------------------------------------------------------------- shadow

def latest_capture(recs, run_id):
    """The AUTHORITATIVE capture record for a run_id, mirroring
    adapter.find_output_eval: eval_final beats the largest step; remaining
    ties go to the record captured LAST (ledger order). Raises ValueError
    with the available run_ids when none match."""
    cands = [(i, r) for i, r in enumerate(recs)
             if r.get("record_type") == "capture-eval"
             and r.get("run_id") == run_id]
    if not cands:
        avail = sorted({r.get("run_id") for r in recs
                        if r.get("record_type") == "capture-eval"}
                       - {None})
        raise ValueError(
            "no capture-eval record with run_id %r in the ledger (%s)"
            % (run_id,
               "available: %s%s" % (", ".join(avail[:8]),
                                    "..." if len(avail) > 8 else "")
               if avail else "no capture records at all — run "
                             "'shipgate capture' first"))
    return max(cands, key=lambda ir: (
        ir[1].get("eval_name") == "eval_final",
        ir[1]["step"] if isinstance(ir[1].get("step"), int) else -1,
        ir[0]))[1]


def eval_run_from_capture(rec):
    """Reconstruct the io.EvalRun the gate consumes from a capture record —
    the inverse of _capture_record, nothing re-derived. final_step is the
    captured eval's OWN step (a mid-training snapshot of a longer design
    is honestly truncated at the budget gate); design_steps is the run's."""
    succ = rec.get("successes")
    run = io.EvalRun(
        run_id=rec.get("run_id"), fmt="capture", path=rec.get("eval_path"),
        sr=rec.get("sr_pp"), n_eval=rec.get("n_eval"),
        seed=rec.get("eval_seed"),
        successes=[ch == "1" for ch in succ] if succ is not None else None,
        set_hash=rec.get("set_hash"),
        final_step=rec.get("step"), design_steps=rec.get("design_steps"),
        task_groups=list(rec.get("task_groups") or []),
    )
    # engine reads getattr(run, 'policy_class', None) — recorded, not typed
    # into EvalRun (the replay convention).
    run.policy_class = rec.get("policy_class")
    return run


def matched_config_mismatch(a, b):
    """Reasons two capture records are NOT matched-config per the captured
    G3 ordering metadata; [] iff pairable. Checked in specificity order —
    set, then order, then per-task counts — because a set mismatch makes
    the later comparisons meaningless. Missing metadata on either side
    REFUSES: pairability that cannot be certified is not assumed."""
    missing = [r.get("run_id") for r in (a, b)
               if not r.get("task_blocks") or not r.get("task_order")]
    if missing:
        return ["captured G3 ordering metadata missing on %s: the "
                "eval_info.json carried no complete per_task successes, so "
                "matched-config pairability cannot be certified"
                % ", ".join(str(m) for m in missing)]
    ao, bo = list(a["task_order"]), list(b["task_order"])
    if sorted(ao) != sorted(bo):
        return ["task sets differ (%s vs %s): not the same eval battery — "
                "any delta would mix task identity with policy quality"
                % (ao, bo)]
    if ao != bo:
        return ["task-block ORDER differs (%s vs %s): episode index i is a "
                "different task on each side — CRN pairing would pair "
                "across tasks" % (ao, bo)]
    if list(a["task_blocks"]) != list(b["task_blocks"]):
        return ["per-task episode counts differ (%s vs %s): the block "
                "structure is not matched-config even though the task sets "
                "agree" % (a["task_blocks"], b["task_blocks"])]
    return []


def _divergence(gate_verdict, decided):
    """Gate-vs-team divergence; None when no decision was declared. SHIP is
    the only 'gate would promote' (module docstring)."""
    if decided not in ("promote", "hold"):
        return None
    if (gate_verdict == "SHIP") == (decided == "promote"):
        return AGREE
    return GATE_HOLD_TEAM_PROMOTED if decided == "promote" \
        else GATE_SHIP_TEAM_HELD


def _pair_ref(rec):
    """The sub-dict binding a shadow record to ONE captured eval — run_id
    plus the content hash, so the decision can never be quoted against a
    different vector than the one it gated."""
    return {"run_id": rec.get("run_id"), "eval_name": rec.get("eval_name"),
            "step": rec.get("step"),
            "content_sha256": rec.get("content_sha256")}


def shadow_pair(ledger_path, incumbent_id, candidate_id, decided="none",
                regime=None, config=None, now=None):
    """Shadow-gate one captured pair and append the 'shadow-decision'
    record. Runs the SHIPPED engine.gate() at comparison_level='retrain'
    with the captured task_blocks/task_order as the declared task structure
    and ``regime`` resolving the sigma_run / sigma_pertask priors (frozen
    resolution order; unknown regime raises GateInputError = invalid
    input). Refuses via the verdict-'INVALID' path — recorded, not raised —
    when matched_config_mismatch() finds the pair unpairable. ``config``
    (default GateConfig()) supplies every other threshold. Returns the
    appended record."""
    if decided not in ("promote", "hold", "none"):
        raise ValueError("decided must be 'promote', 'hold' or 'none', "
                         "got %r" % (decided,))
    recs, _hashes, prev = read_chain(ledger_path)
    if not recs:
        raise ValueError("%s: no records — run 'shipgate capture' first"
                         % ledger_path)
    inc_rec = latest_capture(recs, incumbent_id)
    cand_rec = latest_capture(recs, candidate_id)
    created = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mismatch = matched_config_mismatch(inc_rec, cand_rec)
    if mismatch:
        verdict, reasons, gate_record = "INVALID", list(mismatch), None
    else:
        base = config if config is not None else GateConfig()
        cfg = dataclasses.replace(
            base, comparison_level="retrain",
            regime=regime if regime is not None else base.regime,
            selection="final", n_checkpoints=None, n_tasks=None,
            task_blocks=list(inc_rec["task_blocks"]),
            task_labels=list(inc_rec["task_order"]))
        dec = gate(eval_run_from_capture(inc_rec),
                   eval_run_from_capture(cand_rec), cfg, now=created,
                   extra_assumptions=[_SHADOW_ASSUMPTION])
        verdict, reasons, gate_record = dec.verdict, list(dec.reasons), \
            dec.record
    rec = {
        "record_type": "shadow-decision",
        "schema_version": SHADOW_SCHEMA_VERSION,
        "engine_version": _ENGINE_VERSION,
        "created": created,
        "incumbent": _pair_ref(inc_rec),
        "candidate": _pair_ref(cand_rec),
        "matched_config": not mismatch,
        "config_mismatch": list(mismatch),
        "gate_verdict": verdict,
        "gate_reasons": reasons,
        "gate_record": gate_record,
        "decided": decided,
        "divergence": _divergence(verdict, decided),
    }
    rec, _line_sha = append_chained(rec, ledger_path, prev)
    return rec


def shadow_report(ledger_path):
    """The divergence table over every shadow-decision in the ledger — the
    literal kill-criterion instrument ('zero reversed ship decisions in 3
    months of shadow-gating'): reversed ship decisions =
    n_gate_hold_team_promoted. Returns
      {'path', 'n_decisions', 'n_declared', 'n_undeclared', 'n_agree',
       'n_gate_hold_team_promoted', 'n_gate_ship_team_held',
       'verdict_counts'}
    Raises ValueError when the ledger does not exist (there is nothing to
    report on, which is different from a ledger with zero decisions)."""
    if not os.path.exists(ledger_path):
        raise ValueError("%s: no such ledger" % ledger_path)
    recs, _hashes, _prev = read_chain(ledger_path)
    decisions = [r for r in recs
                 if r.get("record_type") == "shadow-decision"]
    n_declared = sum(1 for d in decisions
                     if d.get("decided") in ("promote", "hold"))
    div = Counter(d.get("divergence") for d in decisions)
    return {
        "path": ledger_path,
        "n_decisions": len(decisions),
        "n_declared": n_declared,
        "n_undeclared": len(decisions) - n_declared,
        "n_agree": div.get(AGREE, 0),
        "n_gate_hold_team_promoted": div.get(GATE_HOLD_TEAM_PROMOTED, 0),
        "n_gate_ship_team_held": div.get(GATE_SHIP_TEAM_HELD, 0),
        "verdict_counts": dict(Counter(d.get("gate_verdict")
                                       for d in decisions)),
    }
