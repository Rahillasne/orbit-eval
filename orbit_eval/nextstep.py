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

"""orbit-eval next — what to do tomorrow, per job.

`release` says which jobs broke. That is the question a release engineer has. The
question the person who trains the policy actually has is "my bin-picking is at 55%,
what do I do about it", and the three answers are not interchangeable:

  WEAK          every copy you trained is bad at this job. More trials will not help
                and neither will retraining. This job needs demonstrations or a
                different recipe.
  UNLUCKY       your copies disagree about this job by more than sampling can
                explain. That is the training lottery, and it is bought off with
                retrains plus a per-job pick, not with data.
  UNDERPOWERED  you cannot yet see the change you care about. Buy trials.

Telling them apart needs the noise floor: how far apart J copies of the SAME policy
would look purely by sampling. Without that floor a spread of 20 points reads as a
finding when at twenty episodes it is the expected disagreement between identical
models, which is how a team ends up chasing a seed for a week.

The floor here is the expected range of J normal approximations to the per-job rate,
E[range] = d(J) * sigma, with Hartley's constants. At a hundred episodes it agrees
with simulation to a tenth of a point; at twenty it runs slightly HIGH, which makes
the lottery estimate conservative, and that is the direction to be wrong in.

Retrain advice quotes the measured DAMAGE-J cells shipped in `route`, nearest in task
count, and names the cell. It does not extrapolate: there is no transferable law for
run-to-run spread (research/NO_SIGMA_LAW), which is exactly why this command measures
your pool instead of looking your model up in a table.
"""

import math

from . import release as release_mod
from . import route

# Hartley's d2 and d3: the MEAN and the SD of the range of J iid standard normals.
# d2 alone answers "how far apart would identical copies look on average", which is
# the wrong question for a decision: half of all identical pools exceed it. The call
# is made against the upper tail instead, d2 + 1.28*d3, so telling somebody to spend
# a week of GPU time on a lottery that is not there happens about one time in ten
# rather than one time in two. Measured against simulation with identical copies, the
# retrain call fires on 8.5% to 12.3% of pools across J in {3, 8} and n in {20, 100};
# the high corner is J=3 at n=20, where the normal approximation to a 20-trial rate is
# at its worst. test_check.py holds that rate to 15%.
D2 = {1: 0.000, 2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
      7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173, 12: 3.258}
D3 = {1: 0.000, 2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848,
      7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797, 11: 0.787, 12: 0.778}
Z90 = 1.2816

TARGET_PP = 10.0     # the drop a team says it wants to be able to see
LOTTERY_PP = 5.0     # spread above the noise floor before we call it a lottery
WEAK_PCT = 60.0      # nobody clears this on the job
MAX_EPISODES = 100000


def _rate(bits):
    return 100.0 * sum(1 for b in bits if b) / len(bits) if bits else float("nan")


def _sigma_pp(n, pct):
    p = min(max(pct / 100.0, 0.0), 1.0)
    return 100.0 * math.sqrt(max(p * (1.0 - p), 0.0) / n) if n > 0 else 0.0


def noise_floor_pp(J, n, pct):
    """How far apart J identical policies would look ON AVERAGE, in points."""
    if J < 2 or n <= 0:
        return 0.0
    d = D2.get(J) or (D2[12] + 0.06 * (J - 12))   # gentle extension past the table
    return d * _sigma_pp(n, pct)


def noise_ceiling_pp(J, n, pct):
    """The spread identical copies exceed only about one time in ten.

    This, not the mean floor, is what a retrain recommendation is tested against.
    """
    if J < 2 or n <= 0:
        return 0.0
    d2 = D2.get(J) or (D2[12] + 0.06 * (J - 12))
    d3 = D3.get(J) or D3[12]
    return (d2 + Z90 * d3) * _sigma_pp(n, pct)


def episodes_for(pct, target_pp, z=release_mod.Z_DMG):
    """Episodes per arm needed before a drop of `target_pp` is callable.

    Inverts the same resolution statement `release` prints: a drop is callable when
    z * SE no longer exceeds it, with SE the unpaired two-proportion error at equal
    episode counts. Common random numbers make this cheaper; this is the number for a
    team that has not paired, which is most of them.
    """
    if target_pp <= 0:
        return None
    p = min(max(pct / 100.0, 0.0), 1.0)
    var = max(p * (1.0 - p), 0.01)          # a floor, so 0% and 100% still size
    n = 2.0 * var * (100.0 * z / target_pp) ** 2
    return min(int(math.ceil(n)), MAX_EPISODES)


def retrain_advice(n_skills, arm="route_abstain"):
    """The measured DAMAGE-J row nearest this team's job count, named, not extrapolated."""
    best, best_gap = None, None
    for name, cell in route.DAMAGE_J_CELLS.items():
        gap = abs(cell["T"] - n_skills)
        if best_gap is None or gap < best_gap:
            best, best_gap = (name, cell), gap
    name, cell = best
    curve = cell.get(arm, {})
    enough = None
    for J in sorted(curve):
        if curve[J] <= 0.05:
            enough = J
            break
    return {"cell": name, "cell_tasks": cell["T"], "your_tasks": n_skills,
            "exact_match": cell["T"] == n_skills,
            "ship_one_risk": cell.get("ship_one"),
            "curve": {int(k): v for k, v in curve.items()},
            "retrains_for_5pct_risk": enough,
            "best_val_is_flat": cell.get("best_val")}


def merge_reasons(reasons, skills):
    """Pool the per-candidate failure tallies into one per job."""
    out = {}
    for _cand, by_skill in (reasons or {}).items():
        for s, counts in by_skill.items():
            if s not in skills:
                continue
            bucket = out.setdefault(s, {})
            for why, n in counts.items():
                bucket[why] = bucket.get(why, 0) + n
    return {s: dict(sorted(v.items(), key=lambda kv: -kv[1])) for s, v in out.items()}


def dominant_reason(counts, min_share=0.4, min_n=5):
    """The one failure mode worth naming, or None.

    A job at 45% tells you to collect demonstrations. A job at 45% where two thirds
    of the failures were the grasp tells you which demonstrations, and that is the
    difference between a week of collection and an afternoon of it. Named only when
    it is both common enough to act on and counted often enough to believe.
    """
    total = sum(counts.values()) if counts else 0
    if total < min_n:
        return None
    named = {k: v for k, v in counts.items() if k and k != "unrecorded"}
    if not named:
        return None
    why, n = max(named.items(), key=lambda kv: kv[1])
    return (why, n, total) if n / float(total) >= min_share else None


def diagnose(data, target_pp=TARGET_PP, lottery_pp=LOTTERY_PP, weak_pct=WEAK_PCT,
             reasons=None, rounds=False):
    """Per job: unlucky, weak, thin or fine, under one suite-level power statement.

    Power is deliberately NOT a per-job verdict. Teams run the same battery on every
    job, so an episode count that cannot see the drop they care about is true of all
    ten jobs at once, and printing it ten times is one sentence repeated, not advice.
    It is stated once at the top. What the per-job rows carry is what makes jobs
    DIFFER from each other, which is the only thing that can change what you do
    tomorrow.
    """
    cands = sorted(data)
    J = len(cands)
    skills = sorted(set.intersection(*(set(data[c]) for c in cands))) if cands else []
    pooled = merge_reasons(reasons, set(skills))
    ns = [min(len(data[c][s]) for c in cands) for s in skills]
    median_n = sorted(ns)[len(ns) // 2] if ns else 0
    rows = []
    for s in skills:
        rates = {c: _rate(data[c][s]) for c in cands}
        n = min(len(data[c][s]) for c in cands)
        best_c = max(rates, key=lambda c: rates[c])
        worst_c = min(rates, key=lambda c: rates[c])
        best, worst = rates[best_c], rates[worst_c]
        pool = sum(rates.values()) / J
        spread = best - worst
        floor = noise_floor_pp(J, n, pool)
        ceiling = noise_ceiling_pp(J, n, pool)
        lottery = max(0.0, spread - floor)
        se = 100.0 * math.sqrt(2.0 * max(pool / 100.0 * (1 - pool / 100.0), 1e-9) / n)
        resolvable = max(release_mod.DROP_PP, release_mod.Z_DMG * se)
        need = episodes_for(pool, target_pp)
        thin = median_n > 0 and n < 0.6 * median_n
        if thin:
            diag = "THIN"
            action = ("only %d trials here against %d elsewhere: even out the battery"
                      % (n, median_n))
        elif J >= 2 and spread > ceiling and lottery >= lottery_pp and rounds:
            # The candidates are successive rounds of one loop, not copies of one
            # recipe. A spread past the floor means the loop moved this job; it
            # is not a lottery to be re-drawn.
            diag = "MOVED"
            action = ("rounds range %.0f%% to %.0f%%, a %.0f-pt spread past the %.0f "
                      "identical copies would show: the loop moved this job; the "
                      "round-by-round audit says whether it held"
                      % (worst, best, spread, ceiling))
        elif J >= 2 and spread > ceiling and lottery >= lottery_pp:
            diag = "UNLUCKY"
            action = ("copies range %.0f%% to %.0f%%, a %.0f-pt spread where identical "
                      "copies would rarely pass %.0f: retrain and pick this job separately"
                      % (worst, best, spread, ceiling))
        elif best < weak_pct:
            diag = "WEAK"
            dom = dominant_reason(pooled.get(s, {}))
            if dom:
                why, cnt, tot = dom
                action = ("best copy only reaches %.0f%%, and %d of its %d recorded "
                          "failures were the %s: collect demonstrations of that"
                          % (best, cnt, tot, why))
            else:
                action = ("best copy only reaches %.0f%%: needs demonstrations or a "
                          "recipe change, not trials or seeds" % best)
        else:
            diag, action = "FINE", "nothing"
        rows.append({
            "skill": s, "n_episodes": n, "n_candidates": J,
            "best_pct": best, "best_candidate": best_c,
            "worst_pct": worst, "worst_candidate": worst_c,
            "pool_pct": pool, "spread_pp": spread,
            "noise_floor_pp": floor, "noise_ceiling_pp": ceiling,
            "lottery_pp": lottery,
            "resolvable_drop_pp": resolvable,
            "episodes_for_target": need,
            "extra_episodes": max(0, (need or 0) - n),
            # route.wilson, not stats.wilson: the former returns points and every
            # other number in this row is points. The latter returns fractions.
            "ci95_best": list(route.wilson(sum(1 for b in data[best_c][s] if b),
                                           len(data[best_c][s]))),
            "diagnosis": diag, "action": action,
            "failure_reasons": pooled.get(s, {}),
            "dominant_reason": dominant_reason(pooled.get(s, {})),
        })
    order = {"THIN": 0, "UNLUCKY": 1, "MOVED": 1, "WEAK": 2, "FINE": 3}
    rows.sort(key=lambda r: (order[r["diagnosis"]],
                             -r["lottery_pp"] if r["diagnosis"] in ("UNLUCKY", "MOVED")
                             else r["best_pct"]))
    counts = {}
    for r in rows:
        counts[r["diagnosis"]] = counts.get(r["diagnosis"], 0) + 1
    worst_res = max((r["resolvable_drop_pp"] for r in rows), default=float("nan"))
    need_all = max((r["episodes_for_target"] or 0) for r in rows) if rows else 0
    return {
        "n_candidates": J, "candidates": cands, "n_skills": len(rows), "rounds": bool(rounds),
        "target_pp": target_pp, "lottery_pp": lottery_pp, "weak_pct": weak_pct,
        "rows": rows, "counts": counts,
        "median_episodes": median_n,
        "resolvable_drop_pp": worst_res,
        "episodes_for_target": need_all,
        "extra_episodes_per_job": max(0, need_all - median_n),
        "extra_episodes_total": max(0, need_all - median_n) * len(rows),
        "underpowered_for_target": worst_res > target_pp,
        "retrain": retrain_advice(len(rows)) if rows else None,
        "single_candidate": J < 2,
    }


def _plural(n, one, many=None):
    return one if n == 1 else (many or one + "s")


def format_next(d):
    """Reads like advice, not like a table. That is the whole point of the command."""
    if not d["rows"]:
        return "orbit next: no jobs shared by every policy, so there is nothing to compare."
    c = d["counts"]
    L = ["What to do next", "=" * 74, ""]
    unit = ("round", "rounds") if d.get("rounds") else ("copy", "copies")
    L.append("%d %s, %d %s, %d trials each."
             % (d["n_skills"], _plural(d["n_skills"], "job"), d["n_candidates"],
                (unit[0] if d["n_candidates"] == 1 else unit[1]) if d.get("rounds")
                else "trained " + _plural(d["n_candidates"], "copy", "copies"),
                d["median_episodes"]))
    if d["underpowered_for_target"]:
        L.append("At %d trials a job has to drop %.0f points before you can call it. To "
                 "call a" % (d["median_episodes"], d["resolvable_drop_pp"]))
        L.append("%.0f-point drop you need %d trials per copy per job (%d more, %d in total)."
                 % (d["target_pp"], d["episodes_for_target"],
                    d["extra_episodes_per_job"], d["extra_episodes_total"]))
        L.append("Running every copy on the SAME starting states buys part of that back for")
        L.append("free. `orbit check` tells you whether yours already are.")
    else:
        L.append("At %d trials you can already call a %.0f-point drop. The battery is not"
                 % (d["median_episodes"], d["target_pp"]))
        L.append("what is holding you back.")
    L.append("")
    if d["single_candidate"]:
        L.append("Only one trained copy was found, so on no job can the training lottery be")
        L.append("told apart from sampling noise. Train the same recipe twice more and")
        L.append("re-run. It is the cheapest measurement in this tool.")
        L.append("")
    label = {"UNLUCKY": "retrain", "MOVED": "audit", "WEAK": "collect", "THIN": "even out",
             "FINE": "-"}
    L.append("%-20s %6s %7s  %-9s %s" % ("job", "best", "spread", "do", "why"))
    L.append("-" * 74)
    for r in d["rows"]:
        sp = "-" if r["n_candidates"] < 2 else "%.0f" % r["spread_pp"]
        L.append("%-20s %5.0f%% %7s  %-9s %s"
                 % (r["skill"][:20], r["best_pct"], sp, label[r["diagnosis"]], r["action"]))
    L.append("")
    if c.get("UNLUCKY"):
        L.append("RETRAIN   %d %s %s further than sampling explains. That is the training"
                 % (c["UNLUCKY"], _plural(c["UNLUCKY"], "job"),
                    _plural(c["UNLUCKY"], "swings", "swing")))
        L.append("          lottery and not your data: train the recipe again and let")
        L.append("          `orbit check` pick per job, keeping the incumbent where the")
        L.append("          evidence is not there.")
    if c.get("MOVED"):
        L.append("AUDIT     %d %s moved further across rounds than sampling explains. The"
                 % (c["MOVED"], _plural(c["MOVED"], "job")))
        L.append("          loop changed %s; `orbit check --history` says whether any"
                 % _plural(c["MOVED"], "it", "them"))
        L.append("          single round could see it and whether it held to the end.")
    if c.get("WEAK"):
        L.append("COLLECT   %d %s %s bad in every copy you trained. Neither trials nor seeds"
                 % (c["WEAK"], _plural(c["WEAK"], "job"), _plural(c["WEAK"], "is", "are")))
        L.append("          will move %s. Demonstrations or a recipe change might."
                 % _plural(c["WEAK"], "it", "them"))
        named = [r for r in d["rows"] if r["diagnosis"] == "WEAK" and r["dominant_reason"]]
        if named:
            for r in named:
                why, cnt, tot = r["dominant_reason"]
                L.append("          %s: %d of %d failures were the %s."
                         % (r["skill"], cnt, tot, why))
        elif any(r["diagnosis"] == "WEAK" for r in d["rows"]):
            L.append("          Record why each trial failed (`orbit log` asks) and this")
            L.append("          line will say which demonstrations to collect.")
    if c.get("THIN"):
        L.append("EVEN OUT  %d %s %s far fewer trials than the rest, so they are the jobs you"
                 % (c["THIN"], _plural(c["THIN"], "job"), _plural(c["THIN"], "has", "have")))
        L.append("          are least able to defend.")
    if c.get("FINE"):
        L.append("LEAVE     %d %s %s nothing."
                 % (c["FINE"], _plural(c["FINE"], "job"), _plural(c["FINE"], "needs", "need")))
    rt = d.get("retrain")
    if rt and rt["retrains_for_5pct_risk"]:
        L.append("")
        L.append("How many copies to train: %d." % rt["retrains_for_5pct_risk"])
        if rt.get("ship_one_risk") is not None:
            L.append("  Training once and shipping it broke a job in %.0f%% of releases"
                     % (100 * rt["ship_one_risk"]))
            L.append("  measured on %d tasks; picking per job across %d copies took that"
                     % (rt["cell_tasks"], rt["retrains_for_5pct_risk"]))
            L.append("  under 5%.")
        L.append("  Cell: %s%s."
                 % (rt["cell"], "" if rt["exact_match"]
                    else " (nearest to your %d jobs)" % rt["your_tasks"]))
        L.append("  Copy counts do not transfer between setups. This is a starting point,")
        L.append("  not a law; your own pool is what settles it.")
    return "\n".join(L)
