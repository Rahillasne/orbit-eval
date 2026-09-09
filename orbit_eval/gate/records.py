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

"""Decision-record schema, GateConfig, canonical hashing, append-only ledger.

Base layer of the Ship-Gate (v0, fixed-sample; anytime-valid sequential is
explicitly v1): everything here is bookkeeping for HONESTY — a decision record
is only auditable if its bytes are reproducible, so hashing runs over a
canonical JSON serialisation and this module NEVER reads the clock ('created'
is caller-supplied verbatim; the engine stamps it, records.py cannot).

Imports: stdlib + orbit_eval.__version__ only (engine/replay/cli layer on top;
importing them here would cycle).

Conventions (frozen in the v0 spec):
  canonical_json  json.dumps(sort_keys=True, separators=(",", ":"),
                  ensure_ascii=True, allow_nan=False)
  record_sha256   sha256 hex of canonical_json(record minus 'record_sha256')
  ledger          append-only JSONL, one canonical record per line;
                  verify_ledger() recomputes EVERY hash, trusts nothing
"""

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Optional

from .. import __version__ as _ENGINE_VERSION

# v2 (Ship-Gate v0.1 "retrain-aware"): GateConfig gains comparison_level /
# sigma_run_pp / sigma_run_df; statistics gain method, p_eval, sigma_run_*,
# se_eval_pp, se_total_pp, mde_floor_pp, collect_more_unit, retrains_needed.
# verify_ledger stays version-blind: it hashes whatever fields a record
# carries, so v1 records keep verifying byte-for-byte.
# The v1 SEQUENTIAL layer (gate.sequential) adds GateConfig.sequential_tau;
# the schema stays 2 (sequential records are ordinary v2 records with
# method "e-process-episodes" / "e-process-retrains").
SCHEMA_VERSION = 2


def canonical_json(obj):
    """The one true serialisation every hash runs over: sorted keys, compact
    separators, ASCII-only, and allow_nan=False so a NaN/inf can never sneak
    into a signed record (json would emit non-standard tokens silently)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_hex(text):
    """sha256 hex digest of a text string (utf-8)."""
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class GateConfig:
    """Every threshold and every declared assumption the gate uses — nothing
    statistical is hard-coded in the engine. Defaults are the v0 frozen
    defaults; provenance per field:

      alpha             one-sided ship/hold alpha
      min_effect_pp     minimum shippable effect AFTER curse correction;
                        the corpus-measured resolvability floor — delta < 2 pp
                        is a coin flip at any budget [ADAPTIVE_EVAL_PROGRAMME]
      regression_alpha  per-task one-sided alpha, applied PER TASK with NO
                        family-wise correction (the frozen v0 rule): with T
                        tasks the family-wise false-flag probability is up
                        to 1-(1-alpha)^T (~0.40 at alpha=.05, T=10) — the
                        engine appends this as an ASSUMPTION to every
                        multi-task record rather than silently multiplying
                        HOLD chances
      regression_min_pp per-task drop that flags regression
      selection         how the candidate was chosen:
                        "final" | "max-over-checkpoints"
      n_checkpoints     J; required for max-over-checkpoints without history
      curse_prior_pp    measured J=4 mean inflation [PI05_CELL1_RESULTS]
      curse_prior_j     J at which the prior was measured
      max_budget        max affordable TOTAL paired eval episodes per policy;
                        None = never UNRESOLVABLE
      n_tasks           task count for per-task regression; None + no
                        task_blocks = single block, no check (disclosed)
      task_blocks       episodes per task in eval order; must sum to n_eval;
                        overrides the n_tasks equal-block default
      task_labels       display names; default "task%d" % i
      assume_crn        pair when seeds unrecorded / mixed-with-1000
                        (LeRobot's silent default); warning recorded
      sequential_eval   declares LIBERO eval was sequential/non-vectorized;
                        False on LIBERO degrades pairing to unpaired with the
                        libero-autoreset caveat
      regime            atlas regime key, informational, recorded
      n_boot            = stats.N_BOOT; bootstrap reps for CI + history-curse
      boot_seed         bootstrap RNG seed; makes records deterministic
      power_target      power used by COLLECT-MORE sizing
      comparison_level  "checkpoint": both evals come from the SAME training
                        run — question (a); sigma_run does not apply (v0
                        semantics preserved). "retrain": the two evals come
                        from INDEPENDENT training runs — the retraining
                        lottery must be priced (validation/REPLAY_REPORT.md:
                        the eval-only alpha false-shipped 62/386 = 16% of
                        the ORBIT seed-replicate null pairs while the
                        exact-null control sat at 4.0% <= alpha, so the gap
                        is SEMANTICS, not calibration)
      sigma_run_pp      retrain-noise prior (pp). Resolution order at
                        comparison_level='retrain': explicit sigma_run_pp;
                        else regime -> atlas get_regime(regime)['sigma_run'];
                        else None -> the gate still runs, with v0 statistics
                        plus the recorded no-prior disclosure
      sigma_run_df      df behind sigma_run_pp (informational, recorded)
      sigma_pertask_pp  PER-TASK retrain-noise prior (pp) for pricing the
                        retraining lottery into per-task regression flags at
                        comparison_level='retrain' [HOLDINJ_PREREG 2026-08-15;
                        measured: PERTASK2_RESULTS 11.00 pp (raw 11.82),
                        CI 6.86-15.13, scoped pi0.5 / libero_object / k=88
                        multi-task. sigma_pertask is SUITE-dependent
                        (PERTASK-4 T4-2 DIVERGENT 2026-08-21: libero_spatial
                        measured 3.29 at the same class/k) — never choose
                        this prior without naming the suite it was measured
                        on]. Resolution
                        order mirrors sigma_run_pp: explicit; else regime ->
                        atlas regime['sigma_pertask']; else None -> per-task
                        flags stay EVAL-level (v0 behavior, byte-identical)
                        with the existing unpriced disclosure
      sigma_pertask_df  df behind sigma_pertask_pp (informational, recorded)
      sequential_tau    v1 sequential layer (gate.sequential): prior scale
                        tau (pp) of the half-normal mixture e-process and
                        the Robbins normal-mixture CS on the retrain
                        stream; default 5.0 pp, DISCLOSED in every
                        sequential record [frozen v1 math spec]
    """
    alpha: float = 0.05
    min_effect_pp: float = 2.0
    regression_alpha: float = 0.05
    regression_min_pp: float = 5.0
    selection: str = "final"
    n_checkpoints: Optional[int] = None
    curse_prior_pp: float = 4.0
    curse_prior_j: int = 4
    max_budget: Optional[int] = None
    n_tasks: Optional[int] = None
    task_blocks: Optional[List[int]] = None
    task_labels: Optional[List[str]] = None
    assume_crn: bool = False
    sequential_eval: bool = False
    regime: Optional[str] = None
    n_boot: int = 4000
    boot_seed: int = 7
    power_target: float = 0.8
    comparison_level: str = "checkpoint"
    sigma_run_pp: Optional[float] = None
    sigma_run_df: Optional[int] = None
    sigma_pertask_pp: Optional[float] = None
    sigma_pertask_df: Optional[int] = None
    sequential_tau: float = 5.0

    def to_dict(self):
        """JSON-safe dict with EVERY field present (lists copied)."""
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, list) else v
        return out

    @classmethod
    def from_dict(cls, d):
        """Build from a dict; unknown keys raise ValueError (a typo'd
        threshold must never silently fall back to a default)."""
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError("unknown GateConfig key(s): %s (known: %s)"
                             % (", ".join(unknown), ", ".join(sorted(names))))
        return cls(**d)

    def sha256(self):
        """Hash of the canonical config — stamped into every record so a
        verdict can never be quoted apart from the thresholds that made it."""
        return sha256_hex(canonical_json(self.to_dict()))

    def validate(self):
        """List of problems; [] iff the config is usable."""
        errs = []
        if not (isinstance(self.alpha, (int, float)) and 0 < self.alpha < 0.5):
            errs.append("alpha must be in (0, 0.5), got %r" % (self.alpha,))
        if not (isinstance(self.regression_alpha, (int, float))
                and 0 < self.regression_alpha < 0.5):
            errs.append("regression_alpha must be in (0, 0.5), got %r"
                        % (self.regression_alpha,))
        if not (isinstance(self.min_effect_pp, (int, float))
                and self.min_effect_pp > 0):
            errs.append("min_effect_pp must be > 0, got %r"
                        % (self.min_effect_pp,))
        if not (isinstance(self.regression_min_pp, (int, float))
                and self.regression_min_pp > 0):
            errs.append("regression_min_pp must be > 0, got %r"
                        % (self.regression_min_pp,))
        if self.selection not in ("final", "max-over-checkpoints"):
            errs.append("selection must be 'final' or 'max-over-checkpoints',"
                        " got %r" % (self.selection,))
        if self.n_checkpoints is not None and self.n_checkpoints < 2:
            errs.append("n_checkpoints must be None or >= 2, got %r"
                        % (self.n_checkpoints,))
        if not (isinstance(self.curse_prior_pp, (int, float))
                and not isinstance(self.curse_prior_pp, bool)
                and self.curse_prior_pp >= 0):
            errs.append("curse_prior_pp must be a number >= 0 (a negative "
                        "prior would INCREASE the effective delta the SHIP "
                        "rule sees), got %r" % (self.curse_prior_pp,))
        if not (isinstance(self.curse_prior_j, int)
                and not isinstance(self.curse_prior_j, bool)
                and self.curse_prior_j >= 2):
            errs.append("curse_prior_j must be an int >= 2 (the sqrt-log "
                        "scaling divides by ln(curse_prior_j)), got %r"
                        % (self.curse_prior_j,))
        if self.max_budget is not None and self.max_budget < 1:
            errs.append("max_budget must be None or >= 1 (0 or negative "
                        "silently forces every COLLECT-MORE to "
                        "UNRESOLVABLE), got %r" % (self.max_budget,))
        for name in ("assume_crn", "sequential_eval"):
            v = getattr(self, name)
            if not isinstance(v, bool):
                errs.append("%s must be a bool — the engine tests identity "
                            "against True, so a truthy non-bool (e.g. 1) "
                            "would silently act as False, got %r"
                            % (name, v))
        if self.task_blocks is not None and any(b <= 0 for b in self.task_blocks):
            errs.append("task_blocks must be None or all > 0, got %r"
                        % (self.task_blocks,))
        if self.n_tasks is not None and self.n_tasks < 1:
            errs.append("n_tasks must be None or >= 1, got %r"
                        % (self.n_tasks,))
        if not self.n_boot > 0:
            errs.append("n_boot must be > 0, got %r" % (self.n_boot,))
        if not (0 < self.power_target < 1):
            errs.append("power_target must be in (0, 1), got %r"
                        % (self.power_target,))
        if not (isinstance(self.sequential_tau, (int, float))
                and not isinstance(self.sequential_tau, bool)
                and self.sequential_tau > 0):
            errs.append("sequential_tau must be a number > 0 (the "
                        "sequential retrain-stream mixture prior scale in "
                        "pp; the e-process normalisation divides by tau), "
                        "got %r" % (self.sequential_tau,))
        if self.comparison_level not in ("checkpoint", "retrain"):
            errs.append("comparison_level must be 'checkpoint' or 'retrain',"
                        " got %r" % (self.comparison_level,))
        if self.sigma_run_pp is not None:
            if not (isinstance(self.sigma_run_pp, (int, float))
                    and not isinstance(self.sigma_run_pp, bool)
                    and self.sigma_run_pp >= 0):
                errs.append("sigma_run_pp must be None or a number >= 0, "
                            "got %r" % (self.sigma_run_pp,))
            elif self.comparison_level == "checkpoint":
                errs.append("sigma_run_pp set while comparison_level is "
                            "'checkpoint' — the prior would be silently "
                            "unused (sigma_run applies only to independent "
                            "retrains; set comparison_level='retrain' or "
                            "drop the prior)")
        return errs


def build_record(verdict, config, inputs, statistics, caveats, warnings,
                 assumptions, reasons, created):
    """Assemble a decision record and sign it.

    'created' is stored VERBATIM — the caller supplies the timestamp
    (engine.gate stamps it; two calls with identical arguments MUST produce
    byte-identical records, which is only possible if this function never
    reads the clock)."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": _ENGINE_VERSION,
        "created": created,
        "verdict": verdict,
        "reasons": list(reasons),
        "gate_config": config.to_dict(),
        "gate_config_sha256": config.sha256(),
        "inputs": inputs,
        "statistics": statistics,
        "caveats": list(caveats),
        "warnings": list(warnings),
        "assumptions": list(assumptions),
    }
    record["record_sha256"] = record_hash(record)
    return record


def record_hash(record):
    """sha256 over the canonical record content, excluding the signature
    field itself."""
    return sha256_hex(canonical_json(
        {k: v for k, v in record.items() if k != "record_sha256"}))


def append_record(record, ledger_path):
    """Append one canonical record line to the JSONL ledger (created if
    absent). Append-only: nothing here ever rewrites an existing line.

    The line is serialised FIRST and emitted as a single os.write() on an
    O_APPEND fd — one syscall regardless of record size, so two concurrent
    writers cannot interleave chunks of each other's lines (a buffered
    text-mode write flushes in ~8 KB pieces once a record outgrows the
    default buffer, and interleaved chunks would corrupt BOTH signed
    records; verify_ledger would catch it only after the fact)."""
    data = (canonical_json(record) + "\n").encode()
    fd = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(fd, data)
        if written != len(data):   # regular-file short write: disk full etc.
            raise OSError("short ledger write: %d of %d bytes to %s"
                          % (written, len(data), ledger_path))
    finally:
        os.close(fd)


def iter_ledger(ledger_path):
    """Yield parsed records in ledger order (blank lines skipped). Raises
    OSError immediately if the path is unreadable (the file is opened
    eagerly, not on first iteration); malformed JSON raises ValueError —
    use verify_ledger() to survey damage without raising."""
    fh = open(ledger_path)

    def _gen():
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    return _gen()


def verify_ledger(ledger_path):
    """Re-verify EVERY line: recompute record_sha256 AND gate_config_sha256
    from the stored content and compare against the stored values. Trusts
    nothing on disk. Returns
      {'path', 'n_records', 'n_ok', 'bad': [{'line': 1-based, 'reason'}], 'ok'}
    Raises OSError if the ledger itself is unreadable."""
    bad, n_records = [], 0
    with open(ledger_path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            n_records += 1
            try:
                rec = json.loads(line)
            except ValueError as e:
                bad.append({"line": lineno, "reason": "malformed JSON: %s" % e})
                continue
            if not isinstance(rec, dict):
                bad.append({"line": lineno,
                            "reason": "not a JSON object record"})
                continue
            # gate_config first: a tampered threshold gets the SPECIFIC
            # reason (it necessarily also breaks record_sha256, but "your
            # alpha was rewritten" beats "some byte changed").
            try:
                gc_got = sha256_hex(canonical_json(rec.get("gate_config")))
            except (TypeError, ValueError) as e:
                bad.append({"line": lineno,
                            "reason": "gate_config not canonically hashable:"
                                      " %s" % e})
                continue
            if rec.get("gate_config_sha256") != gc_got:
                bad.append({"line": lineno,
                            "reason": "gate_config_sha256 hash mismatch:"
                                      " stored %r, recomputed %s — the "
                                      "thresholds behind this verdict were "
                                      "altered after signing"
                                      % (rec.get("gate_config_sha256"),
                                         gc_got)})
                continue
            try:
                got = record_hash(rec)
            except (TypeError, ValueError) as e:
                bad.append({"line": lineno,
                            "reason": "record not canonically hashable: %s" % e})
                continue
            if rec.get("record_sha256") != got:
                bad.append({"line": lineno,
                            "reason": "record_sha256 hash mismatch: stored %r,"
                                      " recomputed %s — content was altered"
                                      " after signing"
                                      % (rec.get("record_sha256"), got)})
    return {"path": ledger_path, "n_records": n_records,
            "n_ok": n_records - len(bad), "bad": bad, "ok": not bad}
