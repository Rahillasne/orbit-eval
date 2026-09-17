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
from . import check as check_mod
from . import cover as cover_mod
from . import dataset as dataset_mod
from . import judge as judge_mod
from . import logtrials
from . import nextstep
from . import release as release_mod
from . import status as status_mod


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


def cmd_check(args):
    """Run it where the evaluations already are."""
    try:
        res = check_mod.run(args.path, incumbent=args.incumbent, candidate=args.candidate,
                            target_pp=args.target_pp, drop_pp=args.drop_pp,
                            min_episodes=args.min_episodes, crn=args.crn,
                            want_history=True if args.history else None)
    except ValueError as e:
        print("orbit check: %s" % e)
        return 2
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print("\n  scanning %s ..." % os.path.abspath(args.path))
        print(check_mod.format_check(res, args.path, why=args.why))
    if not res.get("ok"):
        return 2
    rec = check_mod.signed(res, ["check", args.path])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rec, fh, indent=2)
    sig = rec.get("record_sha256") or rec.get("meta", {}).get("sha256")
    if args.markdown:
        try:
            with open(args.markdown, "w") as fh:
                fh.write(check_mod.markdown_report(res, sig=sig) + "\n")
        except OSError as e:
            print("  (could not write markdown: %s)" % e)
    if not args.no_report:
        path = args.report or os.path.join(os.path.abspath(args.path), "orbit-report.html")
        try:
            check_mod.write_report(res, path, sig=sig, open_it=not args.no_open)
            if not args.json:
                print("\n  report  file://%s" % path)
                if args.out:
                    print("  record  %s" % args.out)
        except OSError as e:
            print("  (could not write the report: %s)" % e)
    c = res.get("check")
    return 1 if (c and c["n_regressed"]) else 0


def _demo_header(command):
    path, hub_id, blurb = dataset_mod.demo_path(command)
    print()
    print("  DEMO  %s" % hub_id)
    print("        %s, bundled with this package (Apache-2.0)." % blurb)
    print("        Run `orbit %s` in a folder with your own dataset to see yours."
          % command)
    return path


def cmd_status(args):
    """Where this robot is, and the one thing to do next."""
    demo = getattr(args, "demo", False)
    path = _demo_header("status") if demo and not args.json else (
        dataset_mod.demo_path("status")[0] if demo else args.path)
    st = status_mod.scan(path, target_pp=args.target_pp)
    if args.json:
        print(json.dumps(st.as_dict(), indent=2, default=str))
    else:
        print(status_mod.format_status(st, budget=demo))
    if demo:
        # The demo's job is to show the output. Its exit code is not a verdict
        # on anybody's robot, so it does not carry one.
        return 0
    return 1 if st.blocking else 0


def cmd_cover(args):
    """What the recording is missing: the factor combinations never recorded."""
    demo = getattr(args, "demo", False)
    path = _demo_header("cover") if demo and not args.json else (
        dataset_mod.demo_path("cover")[0] if demo else args.path)
    try:
        rep = cover_mod.scan(path)
    except ValueError as e:
        print("orbit cover: %s" % e)
        print("It reads a LeRobot dataset: meta/info.json plus the per-episode table "
              "(meta/episodes.jsonl in v2.x, meta/episodes/*.parquet in v3.0).")
        return 2
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(cover_mod.format_cover(rep))
    return 2 if rep.get("needs") else 0


def cmd_next(args):
    """What to do tomorrow, per job."""
    try:
        res = check_mod.run(args.path, incumbent=args.incumbent, candidate=args.candidate,
                            target_pp=args.target_pp)
    except ValueError as e:
        print("orbit next: %s" % e)
        return 2
    if not res.get("ok"):
        print(check_mod.format_nothing(res, args.path))
        return 2
    d = res["next"]
    if args.json:
        print(json.dumps(d, indent=2, default=str))
    else:
        print()
        print(nextstep.format_next(d))
    return 0


def cmd_log(args):
    """Stand at the robot and press one key per trial."""
    s = logtrials.Session(args.out, args.policy, args.job, block=args.block or "")
    if s.n:
        print("\n  resuming: %d trial%s already recorded for %s / %s"
              % (s.n, "" if s.n == 1 else "s", args.job, args.policy))
    try:
        logtrials.loop(s, target_pp=args.target_pp)
    except KeyboardInterrupt:
        print("\n" + logtrials.summary(s, args.target_pp))
    return 0


def cmd_quickstart(args=None):
    print(QUICKSTART)
    return 0


def cmd_judge(args):
    """Price a success detector against a reference."""
    try:
        recs = judge_mod.load(args.labels)
    except (OSError, ValueError) as e:
        print("orbit judge: %s" % e)
        return 2
    v = judge_mod.validate(recs, target_pp=args.target_pp)
    if args.json:
        print(json.dumps(v, indent=2))
    else:
        print()
        print(judge_mod.format_judge(v))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(v, fh, indent=2)
    # 1 when the judge errs differently depending on the policy, which is the one
    # failure that cannot be corrected for.
    return 1 if v["differential"]["differential"] else 0


AGENT_SKILL_DIRS = [
    ("Claude Code", os.path.join("~", ".claude", "skills")),
    ("Codex", os.path.join("~", ".codex", "skills")),
    ("OpenCode", os.path.join("~", ".opencode", "skills")),
    ("Cursor", os.path.join("~", ".cursor", "skills")),
    ("generic agents", os.path.join("~", ".agents", "skills")),
]


def _skill_text():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "skills", "SKILL.md")) as fh:
        return fh.read()


def cmd_skill(args):
    """Print the agent skill, or install it where a coding agent will find it.

    Robotics engineers increasingly drive their work through a coding agent. An
    agent that has never heard of this tool will answer "did my checkpoint get
    worse" by averaging two success rates, which is the exact mistake the tool
    exists to prevent. Installing the skill is how the right answer reaches
    somebody who never went looking for it.
    """
    text = _skill_text()
    if not args.install:
        print(text)
        return 0
    wrote, skipped = [], []
    for name, d in AGENT_SKILL_DIRS:
        base = os.path.expanduser(d)
        if not (os.path.isdir(base) or args.all):
            skipped.append(name)
            continue
        dest = os.path.join(base, "orbit-eval")
        try:
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "SKILL.md"), "w") as fh:
                fh.write(text)
            wrote.append((name, os.path.join(dest, "SKILL.md")))
        except OSError as e:
            print("  could not write %s: %s" % (dest, e))
    if not wrote:
        print("No agent skills directory found. Looked for:")
        for name, d in AGENT_SKILL_DIRS:
            print("  %-16s %s" % (name, d))
        print("\nRun with --all to create them anyway, or `orbit skill` to print")
        print("the skill and paste it wherever your agent reads skills from.")
        return 1
    for name, path in wrote:
        print("  installed for %-14s %s" % (name, path))
    if skipped:
        print("  (not installed, no directory: %s)" % ", ".join(skipped))
    print("\nYour agent will pick it up on its next session. Ask it "
          "\"did my new checkpoint get worse?\" in a directory with evaluations.")
    return 0


# ------------------------------------------------------------------ parser

def build_parser():
    ap = argparse.ArgumentParser(
        prog="orbit",
        description="Which of your jobs did the new policy break, and what to do "
                    "about it. Run `orbit` on its own in the directory your "
                    "evaluations are already in.",
        epilog="also: release (the same question from a CSV you name), compare, "
               "audit, power, regress, route, selftest. `orbit <command> --help` "
               "for any of them.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version="orbit-eval %s" % __version__)
    # Eleven commands at the top level is a wall. The four anyone needs are named;
    # the rest still work and are listed once, below, for the people who want them.
    sub = ap.add_subparsers(dest="cmd",
                            metavar="{status,cover,check,next,log,judge,skill}")

    sk = sub.add_parser("skill", help="print or install the agent skill, so a coding "
                                      "agent knows when to run this")
    sk.add_argument("--install", action="store_true",
                    help="write it into every agent skills directory that exists")
    sk.add_argument("--all", action="store_true",
                    help="create the directories too, not just the ones present")
    sk.set_defaults(fn=cmd_skill)

    jg = sub.add_parser("judge", help="price your success detector: what does judging "
                                      "by model or by eye cost you?")
    jg.add_argument("labels", help="CSV with one row per episode carrying both the "
                                   "judge's label and a reference label")
    jg.add_argument("--target-pp", type=float, default=10.0,
                    help="the size of regression you care about (default 10.0)")
    jg.add_argument("--out", help="write the result as JSON")
    jg.add_argument("--json", action="store_true")
    jg.set_defaults(fn=cmd_judge)

    lg = sub.add_parser("log", help="record trials at the robot, one keypress each")
    lg.add_argument("--job", required=True, help="the named job being run")
    lg.add_argument("--policy", required=True, help="which trained policy is in the robot")
    lg.add_argument("--block", help="robot, cell, day or operator, so a later check can "
                                    "ask whether a regression is site-wide or local")
    lg.add_argument("--out", default="trials.csv", help="CSV to append to (default trials.csv)")
    lg.add_argument("--target-pp", type=float, default=nextstep.TARGET_PP,
                    help="the drop you want to be able to see (default 10.0)")
    lg.set_defaults(fn=cmd_log)

    stt = sub.add_parser("status", help="where this robot is and the one thing "
                                         "to do next: what the recording got "
                                         "wrong, and what a battery of trials "
                                         "can actually resolve")
    stt.add_argument("path", nargs="?", default=".",
                     help="a LeRobot dataset, a run directory, or the folder "
                          "holding both (default: here)")
    stt.add_argument("--target-pp", type=float, default=nextstep.TARGET_PP,
                     help="the drop you want to be able to see, in points "
                          "(default: %(default)s)")
    stt.add_argument("--demo", action="store_true",
                     help="run on a bundled public SO-101 dataset instead, so "
                          "there is something to look at before you have data")
    stt.add_argument("--json", action="store_true")
    stt.set_defaults(fn=cmd_status)

    cv = sub.add_parser("cover", help="what the recording is missing: the "
                                      "instruction and workspace combinations "
                                      "that were never recorded, counted, never "
                                      "scored")
    cv.add_argument("path", nargs="?", default=".",
                    help="a LeRobot dataset, or the folder holding one (default: here)")
    cv.add_argument("--demo", action="store_true",
                    help="run on a bundled public SO-101 dataset instead")
    cv.add_argument("--json", action="store_true")
    cv.set_defaults(fn=cmd_cover)

    ck = sub.add_parser("check", help="run this where your evaluations already are "
                                      "(start here)")
    ck.add_argument("path", nargs="?", default=".",
                    help="directory to scan (default: the current one)")
    ck.add_argument("--incumbent", help="the policy you run today (default: guessed)")
    ck.add_argument("--candidate", help="the policy you might ship (default: guessed)")
    ck.add_argument("--target-pp", type=float, default=nextstep.TARGET_PP,
                    help="the drop you want to be able to see (default 10.0)")
    ck.add_argument("--drop-pp", type=float, default=release_mod.DROP_PP,
                    help="a job must fall at least this far to count (default 5.0)")
    ck.add_argument("--min-episodes", type=int, default=release_mod.MIN_EPISODES)
    ck.add_argument("--report", help="where to write the HTML report")
    ck.add_argument("--no-report", action="store_true")
    ck.add_argument("--no-open", action="store_true",
                    help="write the report but do not open a browser")
    ck.add_argument("--why", action="store_true",
                    help="the long form: every caveat, what the battery could not see, "
                         "the full round-by-round audit and the episode worklist")
    ck.add_argument("--history", action="store_true",
                    help="force the round-by-round audit (it runs automatically once "
                         "three or more rounds are present)")
    ck.add_argument("--crn", action="store_true",
                    help="assert that episode k is the same starting state for every "
                         "policy (same eval seed). Enables the paired comparison and "
                         "makes the episode worklist a counterexample rather than a lead")
    ck.add_argument("--markdown", help="write the result as markdown, for a PR comment")
    ck.add_argument("--out", help="write the signed record (JSON)")
    ck.add_argument("--json", action="store_true")
    ck.set_defaults(fn=cmd_check)

    nx = sub.add_parser("next", help="what to do tomorrow, per job: retrain, collect "
                                     "or measure")
    nx.add_argument("path", nargs="?", default=".")
    nx.add_argument("--incumbent")
    nx.add_argument("--candidate")
    nx.add_argument("--target-pp", type=float, default=nextstep.TARGET_PP,
                    help="the drop you want to be able to see (default 10.0)")
    nx.add_argument("--json", action="store_true")
    nx.set_defaults(fn=cmd_next)

    rel = sub.add_parser("release", help="the same question as check, from a CSV you "
                                         "name yourself")
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


QUICKSTART = """
orbit: which of your jobs did the new policy break, and what to do about it.

  Nothing was found here to check, so here is the whole tool in six lines.

    orbit                   run it where your evaluations already are. It finds
                            them (lerobot eval_info.json at any depth, a CSV, a
                            JSON matrix), works out which policy you ship today,
                            and answers per job. --why for every caveat.

    orbit status            where this robot is: what the recording got
                            wrong, and what a battery of trials can resolve.
                            --demo shows it on a public SO-101 dataset.

    orbit cover             what the recording is missing: the instruction
                            and workspace combinations never recorded.

    orbit next              what to do tomorrow, per job: retrain, collect
                            demonstrations, or run more trials, and how many.

    orbit log --job J --policy P
                            standing at the robot. One keypress per trial.

    orbit judge labels.csv  what is your success detector costing you?

  Everything runs on your machine. Nothing is uploaded and there is no account.
"""


class _Bare(object):
    """Defaults for a bare `orbit`, which runs a check here."""
    path = "."
    incumbent = candidate = report = out = markdown = None
    target_pp = nextstep.TARGET_PP
    drop_pp = release_mod.DROP_PP
    min_episodes = release_mod.MIN_EPISODES
    crn = history = why = json = no_report = False
    no_open = True


def main(argv=None):
    if not (argv if argv is not None else __import__("sys").argv[1:]):
        # `orbit` on its own answers the question here, the way `git status` does.
        # Printing a menu instead wastes the one interaction a newcomer gives you.
        if check_mod.discover.scan(".")[:1]:
            return cmd_check(_Bare())
        # No evaluation here yet. Somebody standing in a dataset or a run
        # directory still asked a real question, so answer that one instead of
        # printing a menu.
        st = status_mod.scan(".")
        if st.dataset is not None or st.policies:
            print(status_mod.format_status(st))
            return 1 if st.blocking else 0
        return cmd_quickstart()
    args = build_parser().parse_args(argv)
    if not getattr(args, "fn", None):
        return cmd_quickstart()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
