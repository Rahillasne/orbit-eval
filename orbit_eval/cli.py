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

"""orbit-eval — honest statistics for robot-policy evaluation.

Every command keeps the same distinction in view:
  (a) fixed-checkpoint / fixed-dataset comparison under common random numbers
      — cheap, powered at ~2 seeds; and
  (b) selection-method comparison across independent retrains — dominated by
      sigma_run and sigma_set. The price is k-dependent: ~59 set draws per arm
      for 10 pp at k=22 single-task, falling to ~3-4 draws/arm at k>=88 on a
      multi-task pool (KCURVE_RESULTS.md). CRN pairing cannot rescue (b) at
      any k.
"""

import argparse
import json
import os
import sys

from . import __version__, atlas, audit, io, power, route, stats
from . import release as release_mod


# ------------------------------------------------------------------ audit

def cmd_audit(args):
    # A typo'd path must NOT produce a clean bill of health: os.walk on a
    # missing dir (or a plain file) silently yields nothing.
    if not os.path.isdir(args.dir):
        print("ERROR: %s: not a directory" % args.dir, file=sys.stderr)
        return 2
    rep = audit.audit_tree(args.dir)
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(audit.format_report(rep))
    if rep["n_runs"] == 0:
        print("ERROR: no recognisable result files under %s" % args.dir,
              file=sys.stderr)
        return 2
    return 1 if rep["flags"] or rep["errors"] else 0


# ------------------------------------------------------------------ compare

def _regime_banner(paired):
    if paired:
        price = ["      answers, and it is the CHEAP question (~2 eval seeds)."]
    else:
        # the ~2-seed price holds only UNDER CRN pairing, which this
        # comparison just declared unavailable — do not quote it as current
        price = [
            "      answers. It is the CHEAP question (~2 eval seeds) — but ONLY",
            "      under CRN pairing, which THIS comparison lacked: unpaired,",
            "      the CI above pays the full independent eval-draw price.",
        ]
    lines = [
        "-" * 72,
        "WHICH QUESTION IS THIS? (the answer decides what this number means)",
        "  (a) same-data checkpoint regression — 'did MY checkpoint/dataset",
        "      change move the metric?' That is what this %s comparison"
        % ("CRN-PAIRED" if paired else "eval-level"),
    ] + price + [
        "  (b) method comparison across retrains — 'is selection method A",
        "      better than B?' This comparison CANNOT answer (b): a single",
        "      training run per arm confounds the method with sigma_run",
        "      (and sigma_set if the sets were redrawn). Run",
        "      `orbit-eval power --arms method` for the real price (~59 set",
        "      draws/arm at 10 pp at k=22 single-task; ~3-4 at k>=88 on a",
        "      multi-task pool — sigma_set is k-dependent, KCURVE_RESULTS.md).",
        "-" * 72,
    ]
    return "\n".join(lines)


def _load_two(args):
    ra = io.load_single(args.runA)
    rb = io.load_single(args.runB)
    return ra, rb


def cmd_compare(args):
    try:
        ra, rb = _load_two(args)
    except (ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    res = stats.compare_runs(ra, rb, assume_crn=args.assume_crn,
                             n_boot=args.n_boot)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        _print_compare(ra, rb, res)
    return 2 if res["invalid"] else 0


def _print_compare(ra, rb, res):
    print("orbit-eval compare  A=%s  B=%s" % (res["run_a"], res["run_b"]))
    if res["invalid"]:
        print("\nINVALID — no statistics produced (analyze_paired rule: drop a")
        print("run the budget gate rejects BEFORE any statistic touches it):")
        for m in res["invalid"]:
            print("  " + m)
        print("\nTruncation is directional, not mean-zero: an undertrained")
        print("model scores LOW and impersonates whichever hypothesis")
        print("predicts a lower score. [OPS_RUNBOOK sec 3]")
        return
    for w in res["warnings"]:
        print("  WARNING: " + w)
    print()
    if res["paired"]:
        print("regime: CRN-PAIRED (same eval seed -> same initial states;")
        print("        eval-draw noise cancels down to the harness floor sigma_0)")
        print("  n episodes         %d" % res["n"])
        print("  SR A / B           %.1f / %.1f pp" % (res["sr_a"], res["sr_b"]))
        print("  diff (A - B)       %+.1f pp" % res["diff_pp"])
        ci = res["ci"]
        if ci:
            print("  95%% paired bootstrap CI  [%+.1f, %+.1f] pp" % ci)
        print("  discordant pairs   A-only %d / B-only %d (of %d)"
              % (res["n10"], res["n01"], res["n"]))
        print("  exact McNemar p    %.4f  -> %s at alpha=%.2f"
              % (res["p_mcnemar"],
                 "SIGNIFICANT" if res["p_mcnemar"] <= stats.ALPHA
                 else "not significant", stats.ALPHA))
        print()
        print("  CAVEAT: " + atlas.CAVEATS["pairing-validated"])
        if ra.is_libero or rb.is_libero:
            print("  CAVEAT: " + atlas.CAVEATS["libero-autoreset"])
            print("  CAVEAT: " + atlas.CAVEATS["libero-task-ids"])
        elif not (ra.task_groups or rb.task_groups):
            print("  CAVEAT: env unknown (no task_group recorded) — if this is "
                  "LIBERO, vectorized autoreset (LeRobot #4152) breaks CRN "
                  "pairing; sequential eval required.")
        print("  CAVEAT: " + atlas.CAVEATS["async-envs"])
    else:
        print("regime: UNPAIRED two-proportion comparison (CRN pairing was")
        print("        unavailable — see warnings above; the CI below pays the")
        print("        full independent-draw price)")
        print("  n A / B            %d / %d" % (res["n_a"], res["n_b"]))
        print("  SR A / B           %.1f / %.1f pp" % (res["sr_a"], res["sr_b"]))
        print("  diff (A - B)       %+.1f pp" % res["diff_pp"])
        print("  95%% Newcombe CI    [%+.1f, %+.1f] pp" % res["ci"])
        print("  two-prop z, p      z=%.2f, p=%.4f -> %s at alpha=%.2f"
              % (res["z"], res["p_two_prop"],
                 "SIGNIFICANT" if res["p_two_prop"] <= stats.ALPHA
                 else "not significant", stats.ALPHA))
    print()
    print(_regime_banner(res["paired"]))
    print("  " + atlas.CAVEATS["cross-stack"])


# ------------------------------------------------------------------ power

def cmd_power(args):
    reg = None
    if args.regime:
        try:
            reg = atlas.get_regime(args.regime)
        except KeyError as e:
            print("ERROR: %s" % e.args[0], file=sys.stderr)
            return 2
    sigma_run = args.sigma_run
    if sigma_run is None:
        if reg is None:
            print("ERROR: pass --regime or --sigma-run", file=sys.stderr)
            return 2
        sigma_run = reg["sigma_run"]
    sigma_set = args.sigma_set
    sigma_set_defaulted = False
    if sigma_set is None:
        sigma_set = (reg or {}).get("sigma_set")
        if sigma_set is None:
            sigma_set = atlas.SIGMA_SET_PRODUCT_PP
            sigma_set_defaulted = True

    print("orbit-eval power — MDE = 2.80 * sigma * sqrt(2/m)  "
          "(alpha=.05 two-sided, power=.80)")
    if reg:
        print("regime: %s" % reg["label"])
        print("  provenance: %s" % reg["provenance"])
        ci = power.sigma_ci(sigma_run, reg.get("df"))
        print("  sigma_run = %.2f pp%s" % (
            sigma_run,
            "  (90%% chi-square CI [%.2f, %.2f], df=%d)" % (ci[0], ci[1], reg["df"])
            if ci else ""))
        if reg.get("sigma_0") is not None:
            print("  sigma_0 (harness floor, CRN d=0) = %.2f pp" % reg["sigma_0"])
    else:
        print("  sigma_run = %.2f pp (user-supplied)" % sigma_run)

    if args.arms == "method":
        sig = power.method_sigma(sigma_set, sigma_run)
        print("\narms: METHOD — each arm is an independent retrain on an "
              "independently\ndrawn episode set, so per-draw sigma = "
              "sqrt(sigma_set^2 + sigma_run^2)\n  = sqrt(%.2f^2 + %.2f^2) = %.2f pp"
              % (sigma_set, sigma_run, sig))
        if sigma_set_defaulted:
            print("  NOTE: sigma_set defaulted to %.2f pp — the measured "
                  "LIBERO product-regime\n  value; pass --sigma-set if your "
                  "setting differs." % atlas.SIGMA_SET_PRODUCT_PP)
    else:
        sig = sigma_run
        print("\narms: FIXED-SET — both arms retrain on the SAME fixed episode "
              "set,\nso only sigma_run separates them (per-draw sigma = %.2f pp)"
              % sig)

    grid = [1, 2, 3, 5, 8, 12, 20, 59]
    print("\n  %-12s %s" % ("draws/arm", "  ".join("m=%-3d" % m for m in grid)))
    print("  %-12s %s" % ("MDE (pp)",
                          "  ".join("%-5.1f" % power.mde(sig, m) for m in grid)))
    if args.effect:
        n = power.draws_needed(sig, args.effect)
        print("\n  to detect a %.1f pp effect: %d draws per arm"
              % (args.effect, n))
        print("  power at 2 draws/arm for this effect: %.0f%%"
              % (100 * power.two_sided_power(args.effect, sig, 2)))

    if args.arms == "method":
        n10 = power.draws_needed(sig, 10.0)
        print("\nHONESTY NOTE — the method question is the EXPENSIVE one:")
        print("  ~%d set draws per arm for a 10 pp effect in this regime." % n10)
        print("  CRN pairing cannot rescue this: pairing cancels EVAL-draw")
        print("  noise only, while sigma_run and sigma_set live in training.")
        print("  (Fixed-checkpoint regressions under CRN are the cheap")
        print("  question — powered at ~2 eval seeds against sigma_0.)")
    else:
        print("\nNOTE: these draws are independent RETRAINS (training seeds),")
        print("  not eval seeds. A fixed-checkpoint comparison under CRN is")
        print("  cheaper still: it pays only the sigma_0 harness floor.")
    print("\n" + atlas.HONESTY_NOTE)
    return 0


# ------------------------------------------------------------------ regress

def cmd_regress(args):
    try:
        ra, rb = _load_two(args)
    except (ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    res = stats.compare_runs(ra, rb, assume_crn=args.assume_crn,
                             n_boot=args.n_boot)
    if res["invalid"]:
        if args.json:
            print(json.dumps(dict(res, verdict="INVALID", exit=2),
                             indent=2, default=str))
        else:
            print("orbit-eval regress: INVALID INPUTS -> exit 2")
            for m in res["invalid"]:
                print("  " + m)
        return 2

    reg = None
    if args.regime:
        try:
            reg = atlas.get_regime(args.regime)
        except KeyError as e:
            print("ERROR: %s" % e.args[0], file=sys.stderr)
            return 2
    sigma_run = args.sigma_run
    if sigma_run is None:
        # explicit `is None` so --sigma-run 0 is honoured, not discarded
        sigma_run = (reg or {}).get("sigma_run")
        if sigma_run is None:
            sigma_run = atlas.REGIMES[atlas.DEFAULT_REGIME]["sigma_run"]
    gate = args.gate if args.gate is not None else 2.0 * sigma_run
    # Default gate = 2*sigma_run: a drop smaller than two retrain-sigmas is
    # indistinguishable from retrain luck, so it is not actionable as signal.
    if args.gate is not None:
        gate_note = "user --gate"
    elif args.sigma_run is None and args.regime is None:
        gate_note = ("default 2*sigma_run, sigma_run=%.2f (atlas default "
                     "regime '%s', LIBERO SmolVLA-ft — pass --regime or "
                     "--sigma-run if your stack differs)"
                     % (sigma_run, atlas.DEFAULT_REGIME))
    else:
        gate_note = "default 2*sigma_run, sigma_run=%.2f" % sigma_run

    drop = -(res["sr_b"] - res["sr_a"])          # positive = B worse
    p_one = res["p_one_sided_b_worse"]
    # achievable MDE of THIS design (80% power) against the gate, using the
    # same ONE-sided alpha as the verdict's test (z_{1-alpha}, not z_{1-a/2})
    z_a = stats.norm_q(1 - args.alpha)
    if res["paired"]:
        # per-episode diff variance from the discordant counts
        n = res["n"]
        pd_ = (res["n10"] + res["n01"]) / n
        dbar = (res["n10"] - res["n01"]) / n
        sd = (max(pd_ - dbar * dbar, 1e-9)) ** 0.5 * 100.0
        mde_hat = (z_a + stats.Z80) * sd / (n ** 0.5)
    else:
        na, nb = res["n_a"], res["n_b"]
        pbar = (res["sr_a"] / 100 + res["sr_b"] / 100) / 2
        sd = (max(pbar * (1 - pbar), 1e-9)) ** 0.5 * 100.0
        mde_hat = (z_a + stats.Z80) * sd * (1 / na + 1 / nb) ** 0.5

    if drop >= gate and p_one <= args.alpha:
        verdict, exit_code = "REGRESSION", 1
    elif mde_hat > gate:
        # exit 3, NOT 2: invalid inputs (exit 2) mean "fix your files";
        # UNDERPOWERED means "the design cannot detect the gate — collect
        # more" (matching the shipgate COLLECT-MORE convention). A CI
        # consumer must be able to tell the two remediations apart without
        # parsing text.
        verdict, exit_code = "UNDERPOWERED", 3
    else:
        verdict, exit_code = "NO_REGRESSION", 0

    if args.json:
        print(json.dumps(dict(res, drop_pp=drop, gate_pp=gate,
                              gate_note=gate_note, mde_hat_pp=mde_hat,
                              alpha=args.alpha, verdict=verdict,
                              exit=exit_code),
                         indent=2, default=str))
        return exit_code

    print("orbit-eval regress  baseline=%s  candidate=%s"
          % (res["run_a"], res["run_b"]))
    for w in res["warnings"]:
        print("  WARNING: " + w)
    print("  mode              %s"
          % ("CRN-PAIRED" if res["paired"] else "UNPAIRED"))
    print("  SR baseline/cand  %.1f / %.1f pp" % (res["sr_a"], res["sr_b"]))
    print("  drop              %+.1f pp (positive = candidate worse)" % drop)
    print("  gate              %.1f pp (%s)" % (gate, gate_note))
    print("  one-sided p       %.4f (candidate worse)" % p_one)
    print("  design MDE~80%%    %.1f pp (one-sided at alpha=%.2f — what this "
          "n can reliably detect)" % (mde_hat, args.alpha))
    if res["paired"]:
        print("  CAVEAT: " + atlas.CAVEATS["pairing-validated"])
        if ra.is_libero or rb.is_libero:
            print("  CAVEAT: " + atlas.CAVEATS["libero-autoreset"])

    if verdict == "REGRESSION":
        print("VERDICT: REGRESSION (drop >= gate and significant) -> exit 1")
    elif verdict == "UNDERPOWERED":
        print("VERDICT: UNDERPOWERED -> exit 3")
        print("  this design cannot reliably detect the %.1f pp gate "
              "(MDE %.1f pp);" % (gate, mde_hat))
        if res["paired"]:
            print("  'no regression detected' would be vacuous — add eval "
                  "episodes")
            print("  (pairing is already in use; only more episodes shrink "
                  "this MDE)")
        else:
            print("  'no regression detected' would be vacuous — add eval "
                  "episodes")
            print("  or use CRN pairing before trusting a green light")
    else:
        print("VERDICT: no regression at the gate -> exit 0")
    print(_regime_banner(res["paired"]))
    return exit_code


# ------------------------------------------------------------------ selftest

def cmd_selftest(args):
    from . import selftest
    code, _rows, _text = selftest.run_selftest(n_rep=args.n_rep)
    return code


# ------------------------------------------------------------------ route

def cmd_route_build(args):
    try:
        data, files, blocks_by_cand = route.load_candidates(args.candidates)
    except (ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    bad, warns = route.validate(data, blocks_by_cand)
    if bad:
        print("orbit-eval route build: INVALID INPUTS -> exit 2")
        for m in bad:
            print("  " + m)
        return 2
    blocks = route.shared_blocks(data, blocks_by_cand)
    tasks = sorted(data[sorted(data)[0]])
    try:
        inc = route.parse_incumbent_spec(args.incumbent, sorted(data), tasks)
        rel = route.release_table(data, z_abstain=args.z_abstain, incumbent=inc, blocks=blocks)
    except ValueError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    policy_paths = {}
    for item in args.policy_path or []:
        if "=" not in item:
            print("ERROR: --policy-path wants name=path, got %r" % item, file=sys.stderr)
            return 2
        k, v = item.split("=", 1)
        if k.strip() not in data:
            print("ERROR: --policy-path names unknown candidate %r" % k.strip(), file=sys.stderr)
            return 2
        policy_paths[k.strip()] = v.strip()
    rel["policy_paths"] = policy_paths
    sh = route.split_half(data, draws=args.draws, seed=args.seed, z_abstain=args.z_abstain,
                          incumbent=inc, blocks=blocks)
    bud = route.budget(rel["min_n"], rel["J"], rel["T"], z_abstain=args.z_abstain)
    digest, n_files = route.inputs_digest(files)
    import datetime
    meta = {"version": __version__, "inputs_sha256": digest, "n_files": n_files,
            "crn_asserted_by_user": bool(args.assume_crn),
            "built_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "argv": list(sys.argv[1:]) if sys.argv else [],
            "incumbent_arg": args.incumbent, "z_abstain": args.z_abstain, "draws": args.draws,
            "seed": args.seed, "warnings": warns}
    record = {"release": rel, "heldout": sh, "budget": bud, "meta": meta}
    body = json.dumps(record, sort_keys=True, default=str).encode()
    record["record_sha256"] = __import__("hashlib").sha256(body).hexdigest()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=2, default=str)
    report = route.format_build_report(rel, sh, meta, bud)
    if warns:
        report += "\n\n## Warnings\n" + "\n".join("- " + w for w in warns)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(report + "\n")
    if args.json:
        print(json.dumps(record, indent=2, default=str))
    else:
        print(report)
        print("record sha256: %s" % record["record_sha256"])
    return 0


def cmd_release(args):
    """Which of your skills did the new model break? One file in, one answer out."""
    try:
        data, versions = release_mod.load(args.log, args.order)
    except (ValueError, OSError) as e:
        print("orbit-eval release: %s" % e)
        return 2
    if not args.order and len(versions) >= 2 and not args.json:
        print("  version order taken from the file, oldest first: %s"
              % " -> ".join(versions))
        print("  if that is backwards (a newest-first export reverses every comparison),")
        print("  pass --order with the right sequence.\n")
    if len(versions) < 2:
        print("orbit-eval release: found %d version(s) in %s (%s).\n"
              "  Two or more are needed: the model you run now, and the one you are "
              "thinking of shipping." % (len(versions), args.log, ", ".join(versions)))
        return 2

    ids = getattr(route._load_csv, "last_episode_ids", None)
    kw = {"drop_pp": args.drop_pp, "z_dmg": args.z_damage, "episode_ids": ids}
    if len(versions) > 2 and not (args.incumbent and args.candidate):
        h = release_mod.history(data, versions, **kw)
        out = release_mod.format_history(h)
        latest = release_mod.check(data, versions[-2], versions[-1], **kw)
        out += "\n\n" + release_mod.format_check(latest)
        payload = {"history": h, "latest": latest, "versions": versions}
    else:
        inc = args.incumbent or versions[-2]
        cand = args.candidate or versions[-1]
        for v in (inc, cand):
            if v not in data:
                print("orbit-eval release: no version %r in %s (present: %s)"
                      % (v, args.log, ", ".join(sorted(data))))
                return 2
        c = release_mod.check(data, inc, cand, **kw)
        out = release_mod.format_check(c)
        payload = {"latest": c, "versions": versions}

    payload["thresholds"] = {"drop_pp": args.drop_pp, "z_dmg": args.z_damage,
                             "min_episodes": release_mod.MIN_EPISODES}
    rec = release_mod.record(payload, ["release", args.log], inputs=[args.log])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rec, fh, indent=1)
    if args.json:
        print(json.dumps(rec, indent=1))
    else:
        print(out)
        if args.out:
            print("\n  signed record: %s" % args.out)
    latest = payload["latest"]
    # Exit codes a CI job can trust. 0 must mean "checked, and nothing broke".
    if latest["all_underpowered"]:
        return 3                      # could not judge; matches `regress`'s underpowered code
    if latest["n_regressed"] or latest["skills_dropped"]:
        return 1
    if latest["n_underpowered"]:
        # A skill was never judged. Exiting 0 here asserts it was checked and
        # passed, which is the one claim this tool exists to refuse to make.
        return 3
    return 0


def cmd_route_plan(args):
    pl = route.plan(args.candidates, args.tasks, args.select_eps, p=args.rate,
                    z_abstain=args.z_abstain)
    if args.json:
        print(json.dumps(pl, indent=2))
    else:
        print(route.format_plan(pl))
    return 0


# ------------------------------------------------------------------ parser

def build_parser():
    ap = argparse.ArgumentParser(
        prog="orbit-eval",
        description="Honest statistics for robot-policy evaluation "
                    "(ships the measured ORBIT noise atlas).")
    ap.add_argument("--version", action="version",
                    version="orbit-eval %s" % __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rel = sub.add_parser("release", help="which of your skills did the new model "
                                         "break? (start here)")
    rel.add_argument("log", help="episode-level CSV/TSV: version,skill,episode,success "
                                 "(column names are auto-detected)")
    rel.add_argument("--incumbent", help="the version you run now (default: second newest)")
    rel.add_argument("--candidate", help="the version you are thinking of shipping "
                                         "(default: newest)")
    rel.add_argument("--order", help="comma-separated version order, oldest first, when "
                                     "the file order is not chronological")
    rel.add_argument("--drop-pp", type=float, default=release_mod.DROP_PP,
                     help="a skill must fall at least this far to count (default 5.0)")
    rel.add_argument("--z-damage", type=float, default=release_mod.Z_DMG,
                     help="...and at least this many standard errors (default 2.0)")
    rel.add_argument("--out", help="write the signed record (JSON)")
    rel.add_argument("--json", action="store_true")
    rel.set_defaults(fn=cmd_release)

    a = sub.add_parser("audit", help="scan eval outputs for silent defects "
                                     "(exit 0 clean / 1 flags or parse "
                                     "errors / 2 unusable path or no result "
                                     "files)")
    a.add_argument("dir")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_audit)

    c = sub.add_parser("compare", help="compare two eval runs (CRN-paired "
                                       "when possible)")
    c.add_argument("runA")
    c.add_argument("runB")
    c.add_argument("--assume-crn", action="store_true",
                   help="treat unrecorded seeds as equal (LeRobot default 1000)")
    c.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    c.add_argument("--json", action="store_true",
                   help="machine-readable result instead of the text report")
    c.set_defaults(fn=cmd_compare)

    p = sub.add_parser("power", help="MDE / draws-per-arm design rules")
    p.add_argument("--regime", help="atlas regime: %s"
                   % ", ".join(sorted(atlas.REGIMES)))
    p.add_argument("--sigma-run", type=float)
    p.add_argument("--sigma-set", type=float)
    p.add_argument("--effect", type=float, help="target effect in pp")
    p.add_argument("--arms", choices=["fixed-set", "method"],
                   default="fixed-set")
    p.set_defaults(fn=cmd_power)

    r = sub.add_parser("regress", help="CI regression gate "
                                       "(exit 0 ok / 1 regression / 2 invalid "
                                       "inputs / 3 underpowered design)")
    r.add_argument("runA", help="baseline")
    r.add_argument("runB", help="candidate")
    r.add_argument("--gate", type=float,
                   help="regression gate in pp (default 2*sigma_run)")
    r.add_argument("--regime")
    r.add_argument("--sigma-run", type=float)
    r.add_argument("--alpha", type=float, default=stats.ALPHA)
    r.add_argument("--assume-crn", action="store_true")
    r.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    r.add_argument("--json", action="store_true",
                   help="machine-readable result (verdict/exit/gate/MDE) "
                        "instead of the text report")
    r.set_defaults(fn=cmd_regress)

    s = sub.add_parser("selftest", help="verify size/power/unbiasedness on "
                                        "synthetic data")
    s.add_argument("--n-rep", type=int, default=600)
    s.set_defaults(fn=cmd_selftest)

    rt = sub.add_parser("route", help="task-conditioned release selection over "
                                      "candidate checkpoints (build a routed "
                                      "release with abstention; plan a budget)")
    rsub = rt.add_subparsers(dest="route_cmd", required=True)
    rb = rsub.add_parser("build", help="decide, per task, which candidate ships; "
                                       "held-out gain, damage count, regression bound")
    rb.add_argument("candidates", help="dir of <candidate>/**/eval_info.json, or a "
                                       "JSON {\"candidates\": {name: {task: [0/1..]}}}")
    rb.add_argument("--incumbent", help="candidate you would otherwise ship (default: best "
                                        "suite mean); per task: 'taskA=cand1,taskB=cand2' or a "
                                        "JSON/CSV file mapping task -> candidate ('*' = default)")
    rb.add_argument("--policy-path", action="append", metavar="NAME=PATH",
                    help="record where each candidate's weights live (repeatable); "
                         "RoutedPolicy.from_release uses it to serve the release")
    rb.add_argument("--z-abstain", type=float, default=route.Z_ABSTAIN,
                    help="one-sided z to defect from the incumbent (default 1.645)")
    rb.add_argument("--draws", type=int, default=route.N_DRAWS)
    rb.add_argument("--seed", type=int, default=route.RNG_SEED)
    rb.add_argument("--assume-crn", action="store_true",
                    help="assert that episode k of each task used the same initial "
                         "state for every candidate")
    rb.add_argument("--out", help="write the signed release record (JSON)")
    rb.add_argument("--report", help="write the markdown report")
    rb.add_argument("--json", action="store_true")
    rb.set_defaults(fn=cmd_route_build)
    rp = rsub.add_parser("plan", help="what a selection budget can resolve, before spending it")
    rp.add_argument("--candidates", type=int, required=True, help="J")
    rp.add_argument("--tasks", type=int, required=True, help="T")
    rp.add_argument("--select-eps", type=int, required=True, help="selection episodes per task")
    rp.add_argument("--rate", type=float, default=0.5, help="assumed task success rate (0..1)")
    rp.add_argument("--z-abstain", type=float, default=route.Z_ABSTAIN)
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_route_plan)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
