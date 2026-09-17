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

"""orbit-eval history — does the loop actually get better, round after round?

A single release is checked against the one before it. A policy that retrains
itself, or a team that ships every fortnight, is a SEQUENCE, and a sequence has a
failure the pair does not: damage compounds silently. A round that quietly loses
one job reports a higher average, the next round trains on top of it, and by
round ten nobody can say when the thing stopped working.

The measured version of that risk: shipping the best-validation model damaged at
least one task in 100% of draws on a fifty-task suite, while the suite mean got
three times steadier (research/cleanrel50). The certifying number stabilises
exactly as it stops certifying anything, and in a loop that repeats every round.

So this reports three things a pairwise check cannot:

  PER-ROUND RISK       how often a round broke at least one job, and what that
                       rate compounds to over the rounds already run
  SILENT ROT           jobs that went down while the suite average went up, which
                       is the shape a loop optimising its own score produces
  NET, END TO END      first round against last, per job, with intervals, because
                       ten rounds of noise can look like progress

Nothing here is a new statistic. Every round is the same priced comparison
`release.check` makes, applied along a chain.
"""

import math

from . import release as release_mod


def _rate(bits):
    return 100.0 * sum(1 for b in bits if b) / len(bits) if bits else float("nan")


def audit(data, order, drop_pp=release_mod.DROP_PP,
          min_episodes=release_mod.MIN_EPISODES, paired=False, episode_ids=None):
    """`order` is the rounds oldest first. Returns the loop's record."""
    order = [v for v in order if v in data]
    if len(order) < 2:
        return {"rounds": len(order), "steps": [], "order": order,
                "too_short": True}
    kw = {"drop_pp": drop_pp, "min_episodes": min_episodes, "paired": paired}
    steps = []
    for a, b in zip(order, order[1:]):
        ids = None
        if episode_ids:
            ids = episode_ids
        c = release_mod.check(data, a, b, episode_ids=ids, **kw)
        steps.append({
            "from": a, "to": b,
            "n_skills": c["n_skills"],
            "regressed": [r["skill"] for r in c["rows"] if r["verdict"] == "REGRESSED"],
            "suspect": [r["skill"] for r in c["rows"] if r["verdict"] == "SUSPECT"],
            "improved": [r["skill"] for r in c["rows"] if r["verdict"] == "IMPROVED"],
            "suite_delta_pp": c["suite_delta_pp"],
            "worst_skill": c["rows"][0]["skill"] if c["rows"] else None,
            "worst_delta_pp": c["rows"][0]["delta_pp"] if c["rows"] else 0.0,
            "resolvable_drop_pp": c["median_resolvable_drop_pp"],
        })

    n = len(steps)
    bad = sum(1 for s in steps if s["regressed"])
    per_round = bad / n if n else float("nan")

    # Compounding. The observed rate is itself an estimate from n rounds, so the
    # projection carries the interval of that rate rather than pretending to a
    # point. Rounds are treated as independent, which is optimistic: a loop that
    # trains on its own output correlates them.
    def compound(p, k):
        return 1.0 - (1.0 - p) ** k
    lo_hi = _wilson(bad, n)
    proj = {k: {"point": compound(per_round, k),
                "lo": compound(lo_hi[0], k), "hi": compound(lo_hi[1], k)}
            for k in (5, 10, 25, 50) if k >= n}

    # Silent rot: the round's average went UP while a job went down past the rule.
    rot = []
    for s in steps:
        if s["suite_delta_pp"] > 0 and s["regressed"]:
            rot.append({"from": s["from"], "to": s["to"],
                        "suite_delta_pp": s["suite_delta_pp"],
                        "jobs": s["regressed"]})

    # Net, first round against last.
    first, last = order[0], order[-1]
    net = release_mod.check(data, first, last, episode_ids=episode_ids, **kw)

    # Which jobs never recovered: regressed at some round and still below the
    # first round at the end.
    ever = set()
    for s in steps:
        ever.update(s["regressed"])
    unrecovered = sorted(r["skill"] for r in net["rows"]
                         if r["skill"] in ever and r["delta_pp"] < 0)

    # CREEPING ROT, the failure only a sequence can see. A job that loses a few
    # points a round is under the resolution of every single round and over it by
    # the end. A pairwise check run forever would never once flag it; this is the
    # whole reason the chain is worth auditing.
    step_flagged = set()
    for s_ in steps:
        step_flagged.update(s_["regressed"])
    creeping = [r["skill"] for r in net["rows"]
                if r["verdict"] == "REGRESSED" and r["skill"] not in step_flagged]

    trails = {}
    for skill in sorted(set(net_row["skill"] for net_row in net["rows"])):
        trails[skill] = [_rate(data[v][skill]) if skill in data.get(v, {}) else float("nan")
                         for v in order]

    return {
        "rounds": len(order), "order": order, "steps": steps, "too_short": False,
        "rounds_with_a_regression": bad,
        "per_round_risk": per_round,
        "per_round_risk_ci": list(lo_hi),
        "projection": proj,
        "silent_rot": rot,
        "net": net,
        "net_suite_delta_pp": net["suite_delta_pp"],
        "jobs_never_recovered": unrecovered,
        "creeping_rot": creeping,
        "flagged_in_a_single_round": sorted(step_flagged),
        "trails": trails,
    }


def _wilson(k, n, z=1.959964):
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def format_history(h):
    if h.get("too_short"):
        return ("  Only %d round found. A loop needs at least two to have a record."
                % h["rounds"])
    L = []
    n = len(h["steps"])
    L.append("  %d rounds, %d comparisons." % (h["rounds"], n))
    lo, hi = h["per_round_risk_ci"]
    L.append("  %d of %d rounds broke at least one job: %.0f%% per round, 95%% [%.0f, %.0f]."
             % (h["rounds_with_a_regression"], n, 100 * h["per_round_risk"],
                100 * lo, 100 * hi))
    if h["projection"]:
        L.append("")
        L.append("  AT THAT RATE, THE CHANCE SOMETHING IS BROKEN BY ROUND")
        for k, v in sorted(h["projection"].items()):
            L.append("    %-4d  %3.0f%%   95%% [%.0f, %.0f]"
                     % (k, 100 * v["point"], 100 * v["lo"], 100 * v["hi"]))
        L.append("    Rounds are counted as independent, which flatters a loop that")
        L.append("    trains on its own output.")
    L.append("")
    if h["silent_rot"]:
        L.append("  SILENT ROT: %d round%s reported a HIGHER average while breaking a job."
                 % (len(h["silent_rot"]), "" if len(h["silent_rot"]) == 1 else "s"))
        for r in h["silent_rot"][:5]:
            L.append("    %s -> %s   suite %+.1f pp   broke %s"
                     % (r["from"], r["to"], r["suite_delta_pp"], ", ".join(r["jobs"])))
        L.append("    This is the shape a loop scoring itself produces. The average is")
        L.append("    the number that improves; the job is the number a customer notices.")
    else:
        L.append("  No round reported a higher average while breaking a job.")
    if h["creeping_rot"]:
        L.append("")
        L.append("  CREEPING ROT: %s"
                 % ", ".join(h["creeping_rot"]))
        L.append("    NO single round broke %s. %d rounds together did."
                 % ("this job" if len(h["creeping_rot"]) == 1 else "these jobs", n))
        L.append("    A few points a round is under the resolution of every round and")
        L.append("    over it by the end. A pairwise check would never once have fired.")
    L.append("")
    L.append("  END TO END: %s -> %s, suite %+.1f pp"
             % (h["order"][0], h["order"][-1], h["net_suite_delta_pp"]))
    rows = sorted(h["net"]["rows"], key=lambda r: r["delta_pp"])
    for r in rows[:4]:
        L.append("    %-22s %5.0f%% -> %5.0f%%  %+6.1f pp  95%% [%+.0f, %+.0f]  %s"
                 % (r["skill"][:22], r["incumbent_pct"], r["candidate_pct"],
                    r["delta_pp"], r["ci95"][0], r["ci95"][1],
                    "" if r["verdict"] == "held" else r["verdict"]))
    if len(rows) > 6:
        L.append("    ... %d jobs between ..." % (len(rows) - 6))
    for r in rows[-2:] if len(rows) > 4 else []:
        L.append("    %-22s %5.0f%% -> %5.0f%%  %+6.1f pp  95%% [%+.0f, %+.0f]  %s"
                 % (r["skill"][:22], r["incumbent_pct"], r["candidate_pct"],
                    r["delta_pp"], r["ci95"][0], r["ci95"][1],
                    "" if r["verdict"] == "held" else r["verdict"]))
    if h["jobs_never_recovered"]:
        L.append("")
        L.append("  NEVER RECOVERED: %s" % ", ".join(h["jobs_never_recovered"]))
        L.append("    These broke at some round and are still below where they started.")
    return "\n".join(L)
