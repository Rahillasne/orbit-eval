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

"""``orbit-eval audit`` — recursively scan eval outputs for the silent failure
modes that corrupted real ORBIT waves (each flag cost real GPU-days):

  SEED_ABSENT        no eval seed recorded. LeRobot's silent default is
                     seed=1000: every default eval scores initial states
                     1000..1000+n-1 — common random numbers whether you asked
                     or not, and indistinguishable from a deliberate choice.
  SEED_DEFAULT_1000  seed == 1000 recorded: almost certainly the silent
                     default, not a choice.
  TRUNCATED          final_step != design_steps. A crashed run emits a
                     complete-looking result with a plausible sr; truncation
                     is DIRECTIONAL (undertrained scores LOW), not mean-zero.
                     [OPS_RUNBOOK sec 3/4 — corrupted 12/414 runs]
  N_EVAL_MISSING     number of eval episodes unknown: the SR has no CI.
  SR_VECTOR_MISMATCH per-episode successes disagreed with the file's own
                     overall header (n_episodes / pc_success): the vector was
                     dropped at parse time — any paired statistic would have
                     run on episodes that do not correspond to the reported
                     n_eval or SR.
  TINY_N             n_eval so small the 95% Wilson CI is wider than 10 pp.
  REPLICATE_SET      >1 run trained on the identical episode set (matched
                     mechanically via set_hash = sha1(sorted episodes)[:16]).
                     Replicates are great for sigma_run — but fatal if you
                     thought the runs were independent draws.
"""

from collections import defaultdict

from . import io
from .stats import wilson_width_pp, budget_gate

TINY_N_WIDTH_PP = 10.0


def audit_tree(root):
    """Scan root recursively. Returns a plain-dict report."""
    runs, errors = [], []
    for path, parsed, err in io.discover(root):
        if err:
            errors.append({"path": path, "error": err})
            continue
        runs.extend(parsed)

    flags = []

    def flag(run, code, msg):
        flags.append({"run_id": run.run_id, "path": run.path,
                      "code": code, "message": msg})

    for r in runs:
        # (a) seed hygiene
        if r.seed is None:
            flag(r, "SEED_ABSENT",
                 "no eval seed recorded — LeRobot silently defaults to seed "
                 "1000 (initial states 1000..1000+n-1): this run is on common "
                 "random numbers with every other default eval, which is great "
                 "for pairing and poison for independence assumptions")
        elif int(r.seed) == 1000:
            flag(r, "SEED_DEFAULT_1000",
                 "seed == 1000, LeRobot's silent common-random-numbers "
                 "default — treat as 'seed was never chosen'. Every default "
                 "eval ecosystem-wide scores the SAME initial states "
                 "1000..1000+n-1, so this run shares its eval draw with every "
                 "other default eval: great for CRN pairing, poison for any "
                 "independence assumption")
        # (a2) header/vector consistency (parse already dropped the vector)
        if getattr(r, "sr_vector_mismatch", None):
            flag(r, "SR_VECTOR_MISMATCH", r.sr_vector_mismatch)
        # (b) budget gate
        ok, why = budget_gate(r.final_step, r.design_steps)
        if not ok:
            flag(r, "TRUNCATED",
                 "%s — crashed runs eval to plausible but systematically LOW "
                 "scores; exclude before any statistic touches it "
                 "[OPS_RUNBOOK sec 3/4]" % why)
        # (c) n_eval / CI width
        n = r.n_eval or (len(r.successes) if r.successes else None)
        if n is None:
            flag(r, "N_EVAL_MISSING",
                 "n_eval unknown: sr=%s has no confidence interval" % r.sr)
        elif r.sr is not None:
            k = int(round(r.sr / 100.0 * n))
            w = wilson_width_pp(k, n)
            if w > TINY_N_WIDTH_PP:
                flag(r, "TINY_N",
                     "n_eval=%d gives a 95%% Wilson CI %.1f pp wide at "
                     "sr=%.1f — differences inside that band are noise"
                     % (n, w, r.sr))

    # (d) replicate sets via set_hash
    groups = defaultdict(list)
    for r in runs:
        if r.set_hash:
            groups[r.set_hash].append(r)
    replicate_groups = {}
    for h, rs in sorted(groups.items()):
        if len(rs) > 1:
            replicate_groups[h] = [r.run_id for r in rs]
            for r in rs:
                flag(r, "REPLICATE_SET",
                     "set_hash %s shared by %d runs (%s) — identical training "
                     "set; only sigma_run/sigma_0 separates them"
                     % (h, len(rs), ", ".join(x.run_id for x in rs)))

    return {
        "root": root,
        "n_files": len(runs) and len({r.path for r in runs}) or 0,
        "n_runs": len(runs),
        "runs": [_summ(r) for r in runs],
        "flags": flags,
        "replicate_groups": replicate_groups,
        "errors": errors,
    }


def _summ(r):
    n = r.n_eval or (len(r.successes) if r.successes else None)
    ci = None
    if n and r.sr is not None:
        ci = round(wilson_width_pp(int(round(r.sr / 100.0 * n)), n), 2)
    return {"run_id": r.run_id, "fmt": r.fmt, "path": r.path, "sr": r.sr,
            "n_eval": n, "seed": r.seed, "set_hash": r.set_hash,
            "final_step": r.final_step, "design_steps": r.design_steps,
            "wilson_ci_width_pp": ci}


def format_report(rep):
    lines = []
    p = lines.append
    p("=" * 72)
    p("orbit-eval audit — %s" % rep["root"])
    p("=" * 72)
    p("%d runs parsed from %d files, %d parse errors"
      % (rep["n_runs"], rep["n_files"], len(rep["errors"])))
    for e in rep["errors"]:
        p("  ERROR %s: %s" % (e["path"], e["error"]))
    p("")
    p("%-28s %-8s %6s %6s %6s  %s" % ("run", "fmt", "sr", "n", "seed", "CI width"))
    for r in rep["runs"]:
        p("%-28s %-8s %6s %6s %6s  %s"
          % (r["run_id"][:28], r["fmt"],
             "%.1f" % r["sr"] if r["sr"] is not None else "-",
             r["n_eval"] if r["n_eval"] is not None else "-",
             r["seed"] if r["seed"] is not None else "-",
             "%.1f pp" % r["wilson_ci_width_pp"]
             if r["wilson_ci_width_pp"] is not None else "-"))
    p("")
    if not rep["flags"]:
        p("no flags — nothing silently wrong that this tool can see")
    else:
        by = defaultdict(list)
        for f in rep["flags"]:
            by[f["code"]].append(f)
        p("%d flags:" % len(rep["flags"]))
        for code in sorted(by):
            p("")
            p("[%s] x%d" % (code, len(by[code])))
            for f in by[code]:
                p("  %s: %s" % (f["run_id"], f["message"]))
    if rep["replicate_groups"]:
        p("")
        p("replicate groups (identical training set, via set_hash):")
        for h, ids in rep["replicate_groups"].items():
            p("  %s: %s" % (h, ", ".join(ids)))
    return "\n".join(lines)
