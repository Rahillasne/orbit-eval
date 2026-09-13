# orbit-eval — honest statistics for robot-policy evaluation.
# Copyright 2026 ORBIT Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""orbit-shipgate — the statistical promotion gate for robot policies (CLI).

Seven subcommands, one contract:

  check    gate a CANDIDATE eval against an INCUMBENT eval: CRN pre-checks,
           one-sided exact McNemar + paired bootstrap CI, winner's-curse
           handling, per-task regression, COLLECT-MORE sizing. Exit codes ARE
           the verdict: 0 SHIP / 1 HOLD / 2 INVALID / 3 COLLECT-MORE /
           4 UNRESOLVABLE (exit 2 also covers usage errors, unreadable input,
           and GateConfig misuse). Inputs: LeRobot eval_info.json / ORBIT
           model_result.json (io.load_single), the corpus per-episode
           JSONL (replay.load_corpus — '0'/'1' successes strings; a
           multi-run file needs --incumbent-id / --candidate-id to name the
           line), OR a LeRobot training OUTPUT DIR — the adapter locates
           eval_final/eval_info.json, else the newest eval*/eval_info.json
           (orbit_eval.gate.adapter). --history additionally accepts a run
           dir with checkpoints/<step>/eval_info.json: the full checkpoint
           pool for the history-bootstrap curse.
  seq      anytime-valid SEQUENTIAL gating (v1) over a JSONL observation
           stream: --mode episodes consumes {"inc":0/1,"cand":0/1} CRN
           episode lines (betting-mixture e-process on the discordant
           pairs), --mode retrains consumes {"delta_pp":..,"se_eval_pp":..}
           per-retrain-pair lines (half-normal mixture e-process with the
           sigma_run prior). One JSON state line per observation (--every N
           to thin), a final schema-v2 decision record to --ledger. The
           statistics live in orbit_eval.gate.sequential; alpha is honest
           at ANY stopping time (Ville's inequality). Exit 0 = stream
           consumed (the decision is in the emitted state), 2 = error.
  replay   run historical corpus per-episode JSONL through the gate
           (null-pair / curse / effect modes) and render REPLAY_REPORT.md —
           every number computed, none asserted.
  demo     replay a corpus DIRECTORY and render a compact DEMO.md for a
           design partner: the headline eval-only vs retrain-aware
           false-ship table, three verbatim decision records, and the
           ledger-verify line — every number computed from the corpus on
           the spot, none asserted.
  verify   recompute every record_sha256 and gate_config_sha256 in an
           append-only decision ledger: 0 all verify / 1 any bad / 2 unreadable.
  capture  harvest EVERY eval*/eval_info.json under a LeRobot training-
           output tree into a sha256-CHAINED append-only ledger (one
           'capture-eval' record per checkpoint eval: successes vector,
           eval seed with provenance, steps, set_hash, G3 ordering
           metadata; prev_sha256 chains each record to the previous line,
           genesis 64 zeros). IDEMPOTENT: dedupes on the content hash of
           the run_id+eval identity, so re-capturing appends nothing.
           Prints a one-line captured/skipped-duplicate/unreadable
           summary. Exit 0 / 2 unusable tree or ledger.
  shadow   run the SHIPPED gate() (comparison_level='retrain', regime
           resolution incl. per-task pricing) on two captured runs NEXT TO
           the team's declared decision (--decided promote|hold|none) and
           append a 'shadow-decision' record: gate verdict + reasons +
           the declared decision + the divergence between them. Refuses
           via the verdict-INVALID path (recorded, NOT an error) when the
           captured G3 metadata says the pair is not matched-config.
           NON-BLOCKING by contract: exit 0 always, 2 only for invalid
           inputs. --report prints the divergence table (decisions,
           agreements, gate-said-HOLD-team-promoted = reversed ship
           decisions, gate-said-SHIP-team-held) — the literal instrument
           for the thesis kill criterion 'zero reversed ship decisions in
           3 months of shadow-gating'.

`check` is FIXED-SAMPLE: one look at the data, thresholds from GateConfig —
do not peek at a running eval with `check` and pretend the alpha survived;
that is exactly what `seq` is for.

Style contract (mirrors orbit_eval.cli): every statistical choice the engine
made is disclosed in the output — warnings, ASSUMPTION strings, and the
measured-caveat list ride along with the verdict, in text and in --json.
"""

import argparse
import dataclasses
import json
import math
import os
import sys

from .. import __version__, io
from . import adapter, engine, records, replay, shadow


# ---------------------------------------------------------------- helpers

def _csv_ints(text):
    """'20,20,160' -> [20, 20, 160]; raises ValueError with the offending token."""
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            raise ValueError("--task-blocks: empty entry in %r" % text)
        try:
            out.append(int(tok))
        except ValueError:
            raise ValueError("--task-blocks: %r is not an integer" % tok)
    return out


def _build_config(args):
    """GateConfig from --config JSON (GateConfig.from_dict), then explicit
    flags override field by field. Flags left at their None defaults (or
    store_true flags left False) do NOT override the file."""
    if getattr(args, "config", None):
        with open(args.config) as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as e:
                raise ValueError("%s: invalid JSON: %s" % (args.config, e))
        if not isinstance(data, dict):
            raise ValueError("%s: GateConfig JSON must be an object"
                             % args.config)
        cfg = records.GateConfig.from_dict(data)
    else:
        cfg = records.GateConfig()

    ov = {}
    for name in ("alpha", "min_effect_pp", "regression_alpha",
                 "regression_min_pp", "selection", "n_checkpoints",
                 "max_budget", "n_tasks", "regime", "n_boot", "boot_seed",
                 "comparison_level", "sigma_run_pp"):
        v = getattr(args, name, None)
        if v is not None:
            ov[name] = v
    if getattr(args, "tau", None) is not None:
        # seq --tau is the half-normal mixture prior scale: the frozen v1
        # config field is GateConfig.sequential_tau (FROZEN_DEFAULTS 22),
        # folded in HERE so it passes cfg.validate(), enters the config
        # sha256, and is DISCLOSED as tau_pp in every sequential record
        ov["sequential_tau"] = args.tau
    if getattr(args, "task_blocks", None) is not None:
        ov["task_blocks"] = _csv_ints(args.task_blocks)
    if getattr(args, "task_labels", None) is not None:
        ov["task_labels"] = [s.strip() for s in args.task_labels.split(",")]
    if getattr(args, "assume_crn", False):
        ov["assume_crn"] = True
    if getattr(args, "sequential_eval", False):
        ov["sequential_eval"] = True
    return dataclasses.replace(cfg, **ov) if ov else cfg


def _fmt_p(p):
    if p is None:
        return "n/a"
    return "%.4f" % p if p >= 1e-4 else "%.2e" % p


def _load_eval(path, run_id=None):
    """Load one eval run for the gate, from EITHER supported input shape
    (the product contract: LeRobot eval_info.json / model_result.json, or
    the corpus per-episode JSONL).

    run_id given -> the file is a corpus JSONL; the named line is selected
    via replay.load_corpus (reused, not reimplemented). Without run_id, a
    DIRECTORY is first offered to the LeRobot output-dir adapter
    (adapter.find_output_eval: eval_final/eval_info.json, else the newest
    eval*/eval_info.json); a dir with no eval*/ tree falls through to
    io.load_single, so a dir holding exactly one result file keeps its
    pre-adapter behaviour. For files, io.load_single is tried first; if it
    fails — or yields a run with NO per-episode successes while the corpus
    reader finds them (early corpus generations alias 'sr', which io would
    misread as a bare model_result) — the corpus reader is tried, accepting
    exactly one run (more than one needs --incumbent-id/--candidate-id).
    Raises ValueError with every parse error collected, never a silent
    fallback."""
    if run_id is None and os.path.isdir(path):
        run = adapter.find_output_eval(path)
        if run is not None:
            return run
    if run_id is not None:
        cruns, errors = replay.load_corpus([path])
        matches = [cr for cr in cruns if cr.run.run_id == run_id]
        if len(matches) == 1:
            return matches[0].run
        if len(matches) > 1:
            raise ValueError("%s: run_id %r matches %d corpus lines — the "
                             "file is ambiguous" % (path, run_id,
                                                    len(matches)))
        avail = ", ".join(cr.run.run_id for cr in cruns[:8])
        raise ValueError(
            "%s: no corpus line with run_id %r (%d runs parsed%s%s)"
            % (path, run_id, len(cruns),
               "; available: %s%s" % (avail, "..." if len(cruns) > 8 else "")
               if cruns else "",
               "; parse errors: %d" % len(errors) if errors else ""))
    orig = None
    try:
        run = io.load_single(path)
    except ValueError as e:
        orig, run = e, None
    if run is not None and run.successes:
        return run
    cruns, errors = replay.load_corpus([path])
    if len(cruns) == 1:
        return cruns[0].run
    if len(cruns) > 1:
        raise ValueError(
            "%s: corpus JSONL holds %d runs — select one with "
            "--incumbent-id/--candidate-id (e.g. %s)"
            % (path, len(cruns),
               ", ".join(cr.run.run_id for cr in cruns[:4])))
    if run is not None:
        return run    # no corpus shape either; keep the io result verbatim
    msg = str(orig)
    if errors:
        msg += "; also not corpus per-episode JSONL (%s)" % errors[0]["error"]
    raise ValueError(msg)


# ---------------------------------------------------------------- check

def _print_decision(dec, ledger=None):
    rec = dec.record
    cfg = rec.get("gate_config") or {}
    st = rec.get("statistics") or {}
    ins = rec.get("inputs") or {}
    inc = ins.get("incumbent") or {}
    cand = ins.get("candidate") or {}
    print("orbit-shipgate check  incumbent=%s  candidate=%s"
          % (inc.get("run_id"), cand.get("run_id")))
    print("  engine %s  schema %s  gate-config sha256 %s"
          % (rec.get("engine_version"), rec.get("schema_version"),
             (rec.get("gate_config_sha256") or "")[:12]))
    for w in rec.get("warnings") or []:
        print("  WARNING: " + w)
    print()

    paired = st.get("paired")
    if paired is True:
        print("mode: CRN-PAIRED — one-sided exact McNemar + paired bootstrap")
        print("      (fixed-sample v0; anytime-valid sequential is v1)")
    elif paired is False:
        print("mode: UNPAIRED two-proportion (pairing unavailable or degraded")
        print("      — see warnings; the CI pays the full independent-draw "
              "price)")
    if st.get("sr_incumbent_pp") is not None and \
            st.get("sr_candidate_pp") is not None:
        if ins.get("n_paired") is not None:
            print("  n episodes         %d" % ins["n_paired"])
        print("  SR inc / cand      %.1f / %.1f pp"
              % (st["sr_incumbent_pp"], st["sr_candidate_pp"]))
    if st.get("delta_hat_pp") is not None:
        print("  delta (cand-inc)   %+.1f pp" % st["delta_hat_pp"])
    ci = st.get("ci95_pp")
    if ci:
        print("  95%% CI             [%+.1f, %+.1f] pp" % (ci[0], ci[1]))
    if st.get("b") is not None and st.get("c") is not None:
        print("  discordant         inc-only b=%d / cand-only c=%d "
              "(rate %.3f)" % (st["b"], st["c"],
                               st.get("discordance_rate") or 0.0))
    if st.get("p_ship") is not None:
        print("  p_ship / p_worse   %s / %s  (one-sided, alpha %s)"
              % (_fmt_p(st["p_ship"]), _fmt_p(st.get("p_worse")),
                 cfg.get("alpha")))
    if st.get("method") == "z-retrain-aware":
        print("  retrain-aware      se_eval %.2f pp, se_total %.2f pp "
              "(sigma_run %.2f pp, %s)"
              % (st.get("se_eval_pp") or 0.0, st.get("se_total_pp") or 0.0,
                 st.get("sigma_run_pp_used") or 0.0,
                 st.get("sigma_run_source")))
        print("                     eval-level p (recorded as p_eval) %s; "
              "mde_floor %.2f pp at 1 retrain/side"
              % (_fmt_p(st.get("p_eval")), st.get("mde_floor_pp") or 0.0))
        if st.get("retrains_needed") is not None:
            print("                     sizing unit: RETRAINS per side "
                  "(retrains_needed %d — episodes cannot resolve below the "
                  "floor)" % st["retrains_needed"])
    if st.get("curse_method") and st["curse_method"] != "none":
        print("  winner's curse     -%.2f pp (%s) -> delta_effective %+.1f pp"
              % (st.get("curse_correction_pp") or 0.0, st["curse_method"],
                 st.get("delta_effective_pp") or 0.0))
    per_task = st.get("per_task") or []
    if per_task:
        print("  per-task (flag: drop >= %s pp at one-sided alpha %s PER "
              "TASK, no family-wise correction — see ASSUMPTION below):"
              % (cfg.get("regression_min_pp"), cfg.get("regression_alpha")))
        for row in per_task:
            print("    %-14s n=%-4d delta %+7.1f pp  p_worse %-10s%s"
                  % (row.get("task"), row.get("n") or 0,
                     row.get("delta_pp") or 0.0, _fmt_p(row.get("p_worse")),
                     "  REGRESSION" if row.get("regression") else ""))
    if st.get("n_required") is not None:
        print("  sizing             power %.0f%% now at min_effect %s pp; "
              "n_required %d (additional episodes %s)"
              % (100.0 * (st.get("power_at_min_effect") or 0.0),
                 cfg.get("min_effect_pp"), st["n_required"],
                 st.get("episodes_needed")))
    print()
    print("VERDICT: %s -> exit %d" % (dec.verdict, dec.exit_code))
    for r in dec.reasons:
        print("  " + r)
    for a in rec.get("assumptions") or []:
        print("  " + a)
    for cv in rec.get("caveats") or []:
        print("  CAVEAT: " + cv)
    if ledger:
        print("  record appended to %s" % ledger)


def cmd_check(args):
    try:
        cfg = _build_config(args)
        inc = _load_eval(args.incumbent, run_id=args.incumbent_id)
        cand = _load_eval(args.candidate, run_id=args.candidate_id)
        history = None
        if args.history or args.history_id:
            # --history PATH may be (a) one eval file, (b) a run dir whose
            # checkpoints/<step>/ tree holds one eval_info.json per
            # checkpoint — the WHOLE pool, expanded in step order
            # (adapter.load_history_dir; None = no checkpoints/ subtree,
            # fall through to the single-eval loaders).
            history = []
            for p in (args.history or []):
                pool = adapter.load_history_dir(p) if os.path.isdir(p) \
                    else None
                if pool is not None:
                    history.extend(pool)
                else:
                    history.append(_load_eval(p))
            history += [_load_eval(args.candidate, run_id=rid)
                        for rid in (args.history_id or [])]
        dec = engine.gate(inc, cand, cfg, history=history,
                          cross_stack=args.cross_stack)
        if args.ledger:
            records.append_record(dec.record, args.ledger)
    except (engine.GateInputError, ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(dec.record, indent=2, default=str))
    else:
        _print_decision(dec, ledger=args.ledger)
    return dec.exit_code


# ---------------------------------------------------------------- seq

def _import_sequential():
    """Lazy import of the v1 anytime-valid module — the rest of the CLI
    (check/replay/verify/demo, all fixed-sample) must keep working when the
    sequential module is not shipped, with a clean error instead of an
    ImportError traceback."""
    try:
        from . import sequential
    except ImportError as e:
        raise ValueError(
            "'orbit-shipgate seq' needs orbit_eval.gate.sequential (the v1 "
            "anytime-valid e-process module), which is not importable "
            "here: %s" % e)
    return sequential


def _make_sequential(seq, mode, cfg):
    """Construct the sequential process for a mode.

    The frozen v1 API is EpisodeSequential(config) / RetrainSequential(
    config): one validated GateConfig argument. --tau (the half-normal
    mixture prior scale, default 5.0 pp, DISCLOSED in the record as
    tau_pp) is NOT a constructor keyword — _build_config folds it into
    GateConfig.sequential_tau (FROZEN_DEFAULTS 22) before validation, so
    by the time this seam runs the flag has already reached the config
    the constructor reads. With --tau unset the frozen default applies."""
    cls = seq.EpisodeSequential if mode == "episodes" \
        else seq.RetrainSequential
    return cls(cfg)


def _iter_seq_stream(path, mode):
    """Yield validated observations from a seq JSONL stream file.

    episodes mode: {"inc":0/1,"cand":0/1} per line -> (bool, bool).
    retrains mode: {"delta_pp":..,"se_eval_pp":..} per line -> (float,
    float), both required FINITE (json.loads parses the non-standard
    NaN/Infinity tokens, which would poison or zero-weight the e-process).
    Blank lines are skipped; ANY malformed line raises ValueError
    with its 1-based line number — unlike replay's collected corpus errors,
    a gating stream with a corrupt observation must abort, not silently
    shorten (the e-process indexes evidence by observation count)."""
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            ctx = "%s:%d" % (path, lineno)
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError("%s: invalid JSON: %s" % (ctx, e))
            if not isinstance(d, dict):
                raise ValueError("%s: line is not a JSON object" % ctx)
            if mode == "episodes":
                inc, cand = d.get("inc"), d.get("cand")
                for k, v in (("inc", inc), ("cand", cand)):
                    if v not in (0, 1):    # bool is an int: True/False pass
                        raise ValueError(
                            '%s: episodes line needs {"inc":0/1,"cand":0/1}'
                            ", got %s=%r" % (ctx, k, v))
                yield bool(inc), bool(cand)
            else:
                vals = []
                for k in ("delta_pp", "se_eval_pp"):
                    v = d.get(k)
                    try:
                        if v is None or isinstance(v, bool):
                            raise TypeError
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        raise ValueError(
                            '%s: retrains line needs {"delta_pp":..,'
                            '"se_eval_pp":..} numeric, got %s=%r'
                            % (ctx, k, v))
                    if not math.isfinite(vals[-1]):
                        # json.loads accepts the non-standard NaN/Infinity
                        # tokens; a NaN delta would silently poison S_t and
                        # an infinite se would be a silent zero-weight no-op
                        # that still counts as an observation — reject with
                        # the same file:line context as any malformed line
                        raise ValueError(
                            '%s: retrains line needs FINITE numbers, got '
                            '%s=%r' % (ctx, k, vals[-1]))
                yield vals[0], vals[1]


def cmd_seq(args):
    try:
        if args.every < 1:
            raise ValueError("--every must be >= 1, got %d" % args.every)
        seq = _import_sequential()
        if args.mode == "retrains" and args.comparison_level is None:
            # a retrain stream IS a retrain-level comparison; defaulting the
            # level lets --regime/--sigma-run resolve the sigma_run prior
            # exactly as in the fixed engine (explicit > regime > None)
            args.comparison_level = "retrain"
        cfg = _build_config(args)
        errs = cfg.validate()
        if errs:
            raise ValueError("invalid GateConfig: " + "; ".join(errs))
        proc = _make_sequential(seq, args.mode, cfg)
        n_obs = 0
        emitted_last = False
        for obs in _iter_seq_stream(args.stream, args.mode):
            if args.mode == "episodes":
                proc.update(obs[0], obs[1])
            else:
                proc.add_pair(obs[0], obs[1])
            n_obs += 1
            emitted_last = n_obs % args.every == 0
            if emitted_last:
                print(json.dumps(dict(proc.state(), i=n_obs), default=str))
        if n_obs == 0:
            raise ValueError("%s: no observations in the stream" % args.stream)
        state = proc.state()
        if not emitted_last:
            print(json.dumps(dict(state, i=n_obs), default=str))
        rec = proc.record()
        if args.ledger:
            records.append_record(rec, args.ledger)
    except (engine.GateInputError, ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rec, default=str))
    else:
        print("seq %s  n=%d  decision: %s" % (args.mode, n_obs,
                                              state.get("decision")))
        if args.ledger:
            print("record appended to %s" % args.ledger)
    return 0


# ---------------------------------------------------------------- replay

def cmd_replay(args):
    try:
        cfg = _build_config(args)
        modes = (("null", "curse", "effect") if args.mode == "all"
                 else (args.mode,))
        res = replay.run_replay(args.files, config=cfg, modes=modes,
                                out_dir=args.out, ledger_path=args.ledger)
    except (engine.GateInputError, ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    meta = res.get("meta") or {}
    if not meta.get("n_runs"):
        print("ERROR: no parseable corpus runs in: %s"
              % ", ".join(args.files), file=sys.stderr)
        for err in (meta.get("errors") or [])[:10]:
            print("  %s:%s: %s" % (err.get("path"), err.get("line"),
                                   err.get("error")), file=sys.stderr)
        return 2
    if args.json:
        out = {k: v for k, v in res.items() if k != "report"}
        print(json.dumps(out, indent=2, default=str))
    else:
        print(res["report"])
        if res.get("report_path"):
            print("report written to %s" % res["report_path"])
        if args.ledger:
            print("records appended to %s" % args.ledger)
    return 0


# ---------------------------------------------------------------- demo

def _find_corpus_files(corpus_dir):
    """Every *.jsonl under corpus_dir (recursive, sorted — deterministic
    input order regardless of filesystem enumeration)."""
    out = []
    for dirpath, _dirnames, filenames in sorted(os.walk(corpus_dir)):
        for name in sorted(filenames):
            if name.endswith(".jsonl"):
                out.append(os.path.join(dirpath, name))
    return out


def _demo_pick_records(res):
    """Three verbatim decision records for DEMO.md, deterministic and
    deduplicated by record_sha256: (1) a FALSE SHIP on a null pair at
    retrain level if one exists (the headline failure made concrete), else
    the first null gate call; (2) the first curse cell's CORRECTED gate
    (the max-over-checkpoints move with the optimism subtracted); (3) the
    first resolved effect-pair gate (SHIP/HOLD), else the first effect
    gate. Padded from the null records when a mode produced nothing."""
    picks = []
    null = res.get("null") or {}
    nrecs = null.get("records") or []
    ships = [r for r in nrecs if r.get("verdict") == "SHIP"]
    if ships:
        picks.append(("A FALSE SHIP on a seed-replicate null pair "
                      "(retrain-aware)", ships[0]))
    elif nrecs:
        picks.append(("A null-pair gate call (retrain-aware)", nrecs[0]))
    cells = (res.get("curse") or {}).get("cells") or []
    if cells and len(cells[0].get("records") or []) > 1:
        picks.append(("The winner's-curse CORRECTED gate on a "
                      "max-over-checkpoints cell", cells[0]["records"][1]))
    erecs = (res.get("effect") or {}).get("records") or []
    if erecs:
        resolved = [r for r in erecs
                    if r.get("verdict") in ("SHIP", "HOLD")]
        picks.append(("An effect pair — different training sets, "
                      "retrain-aware", (resolved or erecs)[0]))
    seen, out = set(), []
    for label, rec in picks:
        h = rec.get("record_sha256")
        if h in seen:
            continue
        seen.add(h)
        out.append((label, rec))
    for rec in nrecs:
        if len(out) >= 3:
            break
        h = rec.get("record_sha256")
        if h not in seen:
            seen.add(h)
            out.append(("A null-pair gate call (retrain-aware)", rec))
    return out[:3]


def _demo_rate_cell(row):
    """'rate [wilson_lo, wilson_hi]' cell for a pass-summary row dict."""
    if not row or not row.get("n_gates") \
            or row.get("false_ship_rate") is None:
        return "-"
    lo, hi = row["wilson_ci"]
    return "%.4f [%.4f, %.4f]" % (row["false_ship_rate"], lo, hi)


def render_demo(res, config, verify_report, ledger_path):
    """DEMO.md from a replay results dict + a verify_ledger report. Compact
    by design — the artifact a design partner reads in five minutes: the
    headline BEFORE/AFTER false-ship table, three verbatim records, the
    ledger-verify line. Every number is formatted from ``res`` and
    ``verify_report``; none hard-coded, none asserted (the replay's full
    REPLAY_REPORT.md carries the complete methodology)."""
    meta = res.get("meta") or {}
    null = res.get("null") or {}
    eo = null.get("eval_only") or {}
    L = []
    p = L.append
    p("# Ship-Gate demo")
    p("")
    p("A historical eval corpus replayed through the Ship-Gate, on the "
      "spot: %s run(s) parsed from %s file(s), gate engine %s, every "
      "verdict below computed from the input files during this render — "
      "none asserted."
      % (meta.get("n_runs", 0), len(meta.get("paths") or []),
         meta.get("engine_version")))
    p("")
    p("## Headline — false-ship rate on seed-replicate null pairs")
    p("")
    if null.get("n_gates"):
        p("Null pairs are runs retrained on the IDENTICAL episode set with "
          "a different train seed: the true difference is zero, so every "
          "SHIP verdict is a FALSE SHIP. The same %d gate calls (%d pairs, "
          "both directions), two semantics: eval-only (alpha bounds "
          "eval-sampling error, the checkpoint-level question) vs "
          "retrain-aware (the retraining lottery priced with the measured "
          "sigma_run prior). One-sided alpha %g."
          % (null["n_gates"], null.get("n_pairs", 0), null.get("alpha")))
        p("")
        p("| semantics | gates | false ships | rate [Wilson 95% CI] |")
        p("|---|---:|---:|---|")
        p("| eval-only (before) | %d | %d | %s |"
          % (eo.get("n_gates", 0), eo.get("n_false_ship", 0),
             _demo_rate_cell(eo)))
        p("| retrain-aware (after) | %d | %d | %s |"
          % (null["n_gates"], null.get("n_false_ship", 0),
             _demo_rate_cell(null)))
        p("")
        p("(Per-gate Wilson CIs; the gates are mirrored direction-pairs "
          "sharing runs within replicate groups, so these intervals run "
          "narrow — the full REPLAY_REPORT.md carries the cluster "
          "bootstrap and the per-regime breakdown.)")
    else:
        p("_No gateable null pairs in this corpus — the headline table "
          "needs seed replicates (same set_hash, different train_seed)._")
    p("")
    p("## Three decision records, verbatim")
    p("")
    p("Every verdict ships as a signed, schema-versioned record: full "
      "config, statistics, caveats, warnings, assumptions, hashed over "
      "canonical JSON.")
    picks = _demo_pick_records(res)
    if not picks:
        p("")
        p("_No records produced by this replay._")
    for i, (label, rec) in enumerate(picks, 1):
        p("")
        p("### %d. %s — verdict %s" % (i, label, rec.get("verdict")))
        p("")
        p("```json")
        p(json.dumps(rec, indent=2, sort_keys=True))
        p("```")
    p("")
    p("## Ledger")
    p("")
    n_rec = verify_report.get("n_records", 0)
    if verify_report.get("ok"):
        p("- `%s`: %d append-only record(s); `orbit-shipgate verify` recomputed "
          "every record_sha256 and gate_config_sha256 from the stored "
          "content — all verify."
          % (os.path.basename(ledger_path), n_rec))
    else:
        p("- `%s`: %d record(s), %d FAILED verification — do not trust "
          "this ledger."
          % (os.path.basename(ledger_path), n_rec,
             len(verify_report.get("bad") or [])))
    p("- gate_config sha256: `%s`" % config.sha256())
    p("")
    return "\n".join(L)


def cmd_demo(args):
    try:
        cfg = _build_config(args)
        files = _find_corpus_files(args.corpus_dir)
        if not files:
            raise ValueError("%s: no *.jsonl corpus files found"
                             % args.corpus_dir)
        os.makedirs(args.out, exist_ok=True)
        ledger = os.path.join(args.out, "demo_ledger.jsonl")
        if os.path.exists(ledger):
            raise ValueError(
                "%s already exists — demo writes a FRESH append-only "
                "ledger whose record count DEMO.md quotes; remove it or "
                "pick another --out" % ledger)
        res = replay.run_replay(files, config=cfg,
                                modes=("null", "curse", "effect"),
                                out_dir=None, ledger_path=ledger)
        meta = res.get("meta") or {}
        if not meta.get("n_runs"):
            print("ERROR: no parseable corpus runs in: %s"
                  % ", ".join(files), file=sys.stderr)
            for err in (meta.get("errors") or [])[:10]:
                print("  %s:%s: %s" % (err.get("path"), err.get("line"),
                                       err.get("error")), file=sys.stderr)
            return 2
        rep = records.verify_ledger(ledger) if os.path.exists(ledger) \
            else {"path": ledger, "n_records": 0, "n_ok": 0, "bad": [],
                  "ok": True}
        demo_md = render_demo(res, cfg, rep, ledger)
        demo_path = os.path.join(args.out, "DEMO.md")
        with open(demo_path, "w") as fh:
            fh.write(demo_md)
    except (engine.GateInputError, ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    print("demo written to %s" % demo_path)
    print("ledger written to %s (%d records, %s)"
          % (ledger, rep.get("n_records", 0),
             "all verify" if rep.get("ok") else "VERIFICATION FAILED"))
    return 0 if rep.get("ok") else 1


# ---------------------------------------------------------------- verify

def cmd_verify(args):
    try:
        rep = records.verify_ledger(args.ledger)
    except OSError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print("orbit-shipgate verify  %s" % rep.get("path"))
        print("  records %d  ok %d  bad %d"
              % (rep.get("n_records") or 0, rep.get("n_ok") or 0,
                 len(rep.get("bad") or [])))
        for b in rep.get("bad") or []:
            print("  line %s: %s" % (b.get("line"), b.get("reason")))
        if rep.get("ok"):
            print("VERDICT: LEDGER OK — every record_sha256 and "
                  "gate_config_sha256 verifies -> exit 0")
        else:
            print("VERDICT: LEDGER BAD — hash mismatch or malformed record "
                  "-> exit 1")
    return 0 if rep.get("ok") else 1


# ---------------------------------------------------------------- capture

def cmd_capture(args):
    try:
        res = shadow.capture_tree(args.root, args.ledger)
    except (ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    for u in res["unreadable_files"]:
        print("UNREADABLE: %s: %s" % (u["path"], u["error"]),
              file=sys.stderr)
    print("orbit-shipgate capture %s  captured=%d skipped-duplicate=%d "
          "unreadable=%d -> %s"
          % (args.root, res["captured"], res["skipped_duplicate"],
             res["unreadable"], args.ledger))
    return 0


# ---------------------------------------------------------------- shadow

def _print_shadow_report(rep):
    order = ["SHIP", "HOLD", "INVALID", "COLLECT-MORE", "UNRESOLVABLE"]
    vc = rep.get("verdict_counts") or {}
    vline = ", ".join("%s=%d" % (v, vc[v]) for v in order if v in vc) \
        or "none"
    print("orbit-shipgate shadow report  %s" % rep.get("path"))
    print("  shadow decisions              %d  (declared %d, undeclared %d)"
          % (rep["n_decisions"], rep["n_declared"], rep["n_undeclared"]))
    print("  gate verdicts                 %s" % vline)
    print("  agreements                    %d" % rep["n_agree"])
    print("  gate-said-HOLD-team-promoted  %d  (reversed ship decisions)"
          % rep["n_gate_hold_team_promoted"])
    print("  gate-said-SHIP-team-held      %d"
          % rep["n_gate_ship_team_held"])
    print("  kill criterion ('zero reversed ship decisions in 3 months of "
          "shadow-gating'): %d reversed so far"
          % rep["n_gate_hold_team_promoted"])


def cmd_shadow(args):
    try:
        if args.report:
            rep = shadow.shadow_report(args.ledger)
        else:
            if not args.incumbent or not args.candidate:
                raise ValueError("shadow needs --incumbent and --candidate "
                                 "run_ids (or --report)")
            rec = shadow.shadow_pair(args.ledger, args.incumbent,
                                     args.candidate, decided=args.decided,
                                     regime=args.regime)
    except (engine.GateInputError, ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    if args.report:
        if args.json:
            print(json.dumps(rep, indent=2, default=str))
        else:
            _print_shadow_report(rep)
        return 0
    if args.json:
        print(json.dumps(rec, indent=2, default=str))
        return 0
    print("orbit-shipgate shadow  incumbent=%s  candidate=%s"
          % (args.incumbent, args.candidate))
    print("  gate verdict  %s%s"
          % (rec["gate_verdict"],
             "" if rec["matched_config"] else "  (matched-config REFUSAL)"))
    for r in rec["gate_reasons"]:
        print("    " + r)
    print("  team decided  %s" % rec["decided"])
    print("  divergence    %s"
          % (rec["divergence"] or "n/a (no declared decision)"))
    print("  record appended to %s" % args.ledger)
    print("  NON-BLOCKING: shadow verdicts never gate a promotion -> exit 0")
    return 0


# ---------------------------------------------------------------- parser

def _add_config_flags(sp, full=True):
    """Flags shared with GateConfig; None defaults mean 'no override'."""
    sp.add_argument("--alpha", type=float, default=None,
                    help="one-sided ship/hold alpha (default 0.05)")
    sp.add_argument("--min-effect-pp", dest="min_effect_pp", type=float,
                    default=None,
                    help="minimum shippable effect AFTER curse correction "
                         "(default 2.0 pp — the measured resolvability floor)")
    sp.add_argument("--max-budget", dest="max_budget", type=int, default=None,
                    help="max affordable TOTAL paired eval episodes per "
                         "policy; exceeding it makes COLLECT-MORE "
                         "UNRESOLVABLE")
    sp.add_argument("--regime", default=None,
                    help="atlas regime key (recorded; at --level retrain it "
                         "also resolves the sigma_run prior unless "
                         "--sigma-run is given)")
    sp.add_argument("--n-boot", dest="n_boot", type=int, default=None,
                    help="bootstrap replicates for CI and history-curse "
                         "(default 4000)")
    sp.add_argument("--boot-seed", dest="boot_seed", type=int, default=None,
                    help="bootstrap RNG seed (default 7; makes records "
                         "deterministic)")
    sp.add_argument("--assume-crn", dest="assume_crn", action="store_true",
                    help="pair when seeds are unrecorded / mixed-with-1000 "
                         "(LeRobot silent default); the assumption is "
                         "recorded as a warning")
    sp.add_argument("--sequential-eval", dest="sequential_eval",
                    action="store_true",
                    help="declare the LIBERO eval was sequential "
                         "(non-vectorized); without this, LIBERO pairing "
                         "degrades to unpaired (issue #4152)")
    sp.add_argument("--config", default=None,
                    help="GateConfig JSON file (GateConfig.from_dict); "
                         "explicit flags override its fields")
    if full:
        sp.add_argument("--regression-alpha", dest="regression_alpha",
                        type=float, default=None,
                        help="per-task one-sided alpha (default 0.05)")
        sp.add_argument("--regression-min-pp", dest="regression_min_pp",
                        type=float, default=None,
                        help="per-task drop that flags regression "
                             "(default 5.0 pp)")
        sp.add_argument("--selection",
                        choices=["final", "max-over-checkpoints"],
                        default=None,
                        help="how the candidate was chosen (default final)")
        sp.add_argument("--n-checkpoints", dest="n_checkpoints", type=int,
                        default=None,
                        help="J for max-over-checkpoints when no --history "
                             "is given (atlas prior correction, scaled)")
        sp.add_argument("--n-tasks", dest="n_tasks", type=int, default=None,
                        help="task count for per-task regression (contiguous "
                             "equal blocks, disclosed as an assumption)")
        sp.add_argument("--task-blocks", dest="task_blocks", default=None,
                        help="episodes per task in eval order, CSV ints "
                             "(e.g. 20,20,160); overrides --n-tasks")
        sp.add_argument("--task-labels", dest="task_labels", default=None,
                        help="display names for tasks, CSV")
        sp.add_argument("--level", dest="comparison_level",
                        choices=["checkpoint", "retrain"], default=None,
                        help="comparison level (default checkpoint): "
                             "'retrain' prices the retraining lottery with "
                             "a sigma_run prior — the eval-only checkpoint "
                             "alpha false-shipped 16%% of the ORBIT retrain "
                             "null pairs (validation/REPLAY_REPORT.md)")
        sp.add_argument("--sigma-run", dest="sigma_run_pp", type=float,
                        default=None, metavar="PP",
                        help="explicit retrain-noise prior sigma_run in pp "
                             "for --level retrain (beats the --regime atlas "
                             "value in the resolution order)")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="orbit-shipgate",
        description="Statistical promotion gate for robot policies: CRN "
                    "pre-checks, one-sided exact McNemar, winner's-curse "
                    "handling, auditable decision records (fixed-sample "
                    "'check'; anytime-valid sequential 'seq', v1).")
    ap.add_argument("--version", action="version",
                    version="orbit-shipgate %s" % __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser(
        "check",
        help="gate CANDIDATE against INCUMBENT (exit 0 SHIP / 1 HOLD / "
             "2 INVALID / 3 COLLECT-MORE / 4 UNRESOLVABLE)")
    c.add_argument("incumbent", metavar="INCUMBENT",
                   help="incumbent eval (eval_info.json / model_result.json "
                        "/ corpus per-episode JSONL)")
    c.add_argument("candidate", metavar="CANDIDATE",
                   help="candidate eval (same accepted shapes)")
    c.add_argument("--incumbent-id", dest="incumbent_id", default=None,
                   metavar="RUN_ID",
                   help="run_id selecting the incumbent line out of a "
                        "multi-run corpus JSONL INCUMBENT")
    c.add_argument("--candidate-id", dest="candidate_id", default=None,
                   metavar="RUN_ID",
                   help="run_id selecting the candidate line out of a "
                        "multi-run corpus JSONL CANDIDATE")
    c.add_argument("--history", action="append", default=None, metavar="PATH",
                   help="checkpoint eval file (repeatable); enables the "
                        "direct history-bootstrap curse estimate. Must list "
                        "ALL J checkpoint evals INCLUDING the candidate's "
                        "own — the bootstrap measures the actual selection "
                        "event, and a pool without the winner "
                        "underestimates the optimism (a warning is "
                        "recorded if no history vector matches the "
                        "candidate's)")
    c.add_argument("--history-id", dest="history_id", action="append",
                   default=None, metavar="RUN_ID",
                   help="run_id of a checkpoint eval line inside the "
                        "CANDIDATE corpus JSONL (repeatable); same "
                        "full-pool-including-the-candidate contract as "
                        "--history")
    c.add_argument("--cross-stack", dest="cross_stack", action="store_true",
                   help="declare the two runs were evaluated on DIFFERENT "
                        "environment/simulator builds: the gate rules "
                        "INVALID with the measured cross-stack caveat (the "
                        "gym_pusht rebuild shifted SR -18.2 pp mean, "
                        "non-uniform — OPS_RUNBOOK sec 9)")
    _add_config_flags(c, full=True)
    c.add_argument("--ledger", default=None, metavar="PATH.jsonl",
                   help="append the decision record to this JSONL ledger")
    c.add_argument("--json", action="store_true",
                   help="print the decision record instead of the text "
                        "report")
    c.set_defaults(fn=cmd_check)

    s = sub.add_parser(
        "seq",
        help="anytime-valid sequential gate (v1 e-process) over a JSONL "
             "observation stream: alpha honest at ANY stopping time "
             "(exit 0 stream consumed — the decision is in the emitted "
             "state — / 2 error)")
    s.add_argument("stream", metavar="STREAM.jsonl",
                   help="observation stream: episodes mode "
                        '{"inc":0/1,"cand":0/1} per CRN episode; retrains '
                        'mode {"delta_pp":..,"se_eval_pp":..} per '
                        "retrain pair")
    s.add_argument("--mode", choices=["episodes", "retrains"], required=True,
                   help="episodes: betting-mixture e-process on the "
                        "discordant CRN pairs; retrains: half-normal "
                        "mixture e-process on per-retrain-pair deltas with "
                        "the sigma_run prior")
    s.add_argument("--alpha", type=float, default=None,
                   help="one-sided evidence threshold (E >= 1/alpha ships "
                        "or holds; default 0.05)")
    s.add_argument("--min-effect-pp", dest="min_effect_pp", type=float,
                   default=None,
                   help="minimum shippable effect AFTER curse correction "
                        "(default 2.0 pp; SHIP-evidence needs the effect "
                        "estimate to clear it, same semantics as the fixed "
                        "engine)")
    s.add_argument("--tau", type=float, default=None, metavar="PP",
                   help="half-normal mixture prior scale for the retrain "
                        "e-process and CS (module default 5.0 pp, "
                        "DISCLOSED in the record)")
    s.add_argument("--level", dest="comparison_level",
                   choices=["checkpoint", "retrain"], default=None,
                   help="comparison level (retrains mode defaults to "
                        "'retrain', which lets --regime/--sigma-run "
                        "resolve the sigma_run prior exactly as in "
                        "'check')")
    s.add_argument("--regime", default=None,
                   help="atlas regime key resolving the sigma_run prior at "
                        "retrain level (explicit --sigma-run beats it)")
    s.add_argument("--sigma-run", dest="sigma_run_pp", type=float,
                   default=None, metavar="PP",
                   help="explicit retrain-noise prior sigma_run in pp "
                        "(resolution order identical to 'check': explicit "
                        "> --regime > none-with-disclosure)")
    s.add_argument("--config", default=None,
                   help="GateConfig JSON file (GateConfig.from_dict); "
                        "explicit flags override its fields")
    s.add_argument("--every", type=int, default=1, metavar="N",
                   help="emit a state line every N observations (the final "
                        "state is always emitted; default 1)")
    s.add_argument("--ledger", default=None, metavar="PATH.jsonl",
                   help="append the final decision record to this JSONL "
                        "ledger")
    s.add_argument("--json", action="store_true",
                   help="print the final decision record as the last "
                        "stdout line (after the state stream) instead of "
                        "the text footer")
    s.set_defaults(fn=cmd_seq)

    r = sub.add_parser(
        "replay",
        help="replay corpus per-episode JSONL through the gate "
             "(null-pair / curse / effect) and render REPLAY_REPORT.md")
    r.add_argument("files", nargs="+", metavar="FILE.jsonl",
                   help="corpus per-episode JSONL file(s)")
    r.add_argument("--mode", choices=["null", "curse", "effect", "all"],
                   default="all")
    r.add_argument("--out", default=None, metavar="DIR",
                   help="write DIR/REPLAY_REPORT.md")
    r.add_argument("--ledger", default=None, metavar="PATH.jsonl",
                   help="append every produced record to this JSONL ledger")
    _add_config_flags(r, full=False)
    r.add_argument("--json", action="store_true",
                   help="print the replay result dict (minus the rendered "
                        "report) instead of the report")
    r.set_defaults(fn=cmd_replay)

    d = sub.add_parser(
        "demo",
        help="replay a corpus directory and render a compact DEMO.md for "
             "a design partner (headline before/after table, three "
             "verbatim records, ledger-verify line; exit 0 ok / 1 ledger "
             "failed verification / 2 error)")
    d.add_argument("--corpus-dir", dest="corpus_dir", required=True,
                   metavar="DIR",
                   help="directory holding corpus per-episode *.jsonl "
                        "files (searched recursively)")
    d.add_argument("--out", required=True, metavar="DIR",
                   help="write DIR/DEMO.md and DIR/demo_ledger.jsonl "
                        "(the ledger must not already exist — DEMO.md "
                        "quotes its record count)")
    _add_config_flags(d, full=False)
    d.set_defaults(fn=cmd_demo)

    v = sub.add_parser(
        "verify",
        help="re-verify every record hash in a decision ledger "
             "(exit 0 ok / 1 bad / 2 unreadable)")
    v.add_argument("ledger", metavar="LEDGER.jsonl")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_verify)

    cap = sub.add_parser(
        "capture",
        help="harvest every eval*/eval_info.json under a LeRobot "
             "training-output tree into a sha256-chained append-only "
             "ledger — idempotent (content-hash dedupe), one-line "
             "captured/skipped-duplicate/unreadable summary (exit 0 / "
             "2 unusable tree or ledger)")
    cap.add_argument("root", metavar="OUTPUT_TREE",
                     help="LeRobot training-output tree (may span many "
                          "runs' output dirs); every run's checkpoint "
                          "evals are captured, with the run dir's "
                          "model_result.json sidecar as metadata when "
                          "present")
    cap.add_argument("--ledger", default="shadow_ledger.jsonl",
                     metavar="PATH.jsonl",
                     help="chained capture/shadow ledger to append to "
                          "(default ./shadow_ledger.jsonl)")
    cap.set_defaults(fn=cmd_capture)

    sh = sub.add_parser(
        "shadow",
        help="run the SHIPPED retrain-level gate on two captured runs "
             "NEXT TO the team's declared decision and record the "
             "divergence (non-blocking: exit 0 always, 2 only for "
             "invalid inputs); --report prints the divergence table — "
             "the kill-criterion instrument")
    sh.add_argument("--ledger", required=True, metavar="PATH.jsonl",
                    help="chained capture/shadow ledger ('orbit-shipgate "
                         "capture' writes it)")
    sh.add_argument("--incumbent", default=None, metavar="RUN_ID",
                    help="run_id of the captured incumbent (its "
                         "authoritative eval is used: eval_final, else "
                         "the largest step)")
    sh.add_argument("--candidate", default=None, metavar="RUN_ID",
                    help="run_id of the captured candidate (same "
                         "authoritative-eval rule)")
    sh.add_argument("--decided", choices=["promote", "hold", "none"],
                    default="none",
                    help="the team's ACTUAL decision for this pair, "
                         "recorded beside the gate's verdict ('none' "
                         "records the verdict without a divergence)")
    sh.add_argument("--regime", default=None,
                    help="atlas regime key resolving the sigma_run and "
                         "sigma_pertask priors at retrain level (unknown "
                         "key = input error, exit 2)")
    sh.add_argument("--report", action="store_true",
                    help="print the divergence table over every shadow "
                         "decision in the ledger instead of gating a pair")
    sh.add_argument("--json", action="store_true",
                    help="print the shadow-decision record (or the report "
                         "dict) instead of the text summary")
    sh.set_defaults(fn=cmd_shadow)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
