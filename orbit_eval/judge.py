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

"""orbit-eval judge — how much is your success detector costing you?

Every number this package prints sits on top of a bit that says the robot did the
task. In simulation that bit is a threshold on a reward: across 1,257 per-task
blocks harvested from public repositories, 99% of them are exactly
`max_reward >= 1.0`. On a real robot there is no such scalar, so the bit comes
from a person watching, or from a model watching. Both can be wrong, and nobody
measures by how much.

The important consequence is not that a judge is inaccurate. It is that an
inaccurate judge SHRINKS the difference you are trying to measure, by an exact
and knowable amount.

Write se for the judge's sensitivity, P(judge says yes | it really succeeded),
and sp for its specificity, P(judge says no | it really failed). For a policy
whose true success rate is p, the judge reports

    p_obs = p * se + (1 - p) * (1 - sp)

so for two policies scored by the SAME judge the reported gap is

    observed difference = true difference * (se + sp - 1)

The factor (se + sp - 1) is Youden's J. It is not an approximation and it does
not depend on p. A ten-point regression seen through a judge with J = 0.54
arrives as 5.4 points, and the episodes needed to resolve it grow by 1/J squared.

That turns the published reliability of vision-language judges into a price. The
strongest general model measured across fourteen public sources reaches 0.77
balanced accuracy, and no model exceeds 0.60 where success turns on fine contact
(arXiv 2609.03611). At se = sp = 0.77, J = 0.54 and the bill is 3.4x the trials.
At 0.60, J = 0.20 and the bill is 25x. At 0.52, the reported figure for
contact-rich assembly, J = 0.04 and a ten-point regression arrives as 0.4
points, which is to say it does not arrive.

The attenuation above assumes the judge errs the same way for every policy. When
it does not, the direction of the bias is no longer knowable and a difference can
be invented rather than shrunk. That case is tested for separately and reported
loudly, because it is the one that cannot be corrected.

This module never labels anything. It takes episodes that carry BOTH a judge's
label and a trusted reference label, and prices the judge against the reference.
"""

import csv
import math
import os

from . import discover
from . import stats

Z = stats.Z975


def _rate(xs):
    return sum(1 for x in xs if x) / len(xs) if xs else float("nan")


def confusion(judge, ref):
    """(tp, fn, fp, tn) over aligned boolean lists."""
    tp = fn = fp = tn = 0
    for j, r in zip(judge, ref):
        if r and j:
            tp += 1
        elif r and not j:
            fn += 1
        elif not r and j:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


def kappa(judge, ref):
    """Cohen's kappa: agreement above what guessing at the same rates would give."""
    n = len(ref)
    if not n:
        return float("nan")
    tp, fn, fp, tn = confusion(judge, ref)
    po = (tp + tn) / n
    pj, pr = (tp + fp) / n, (tp + fn) / n
    pe = pj * pr + (1 - pj) * (1 - pr)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def rates(judge, ref):
    """Sensitivity and specificity with Wilson intervals, and Youden's J."""
    tp, fn, fp, tn = confusion(judge, ref)
    n_pos, n_neg = tp + fn, fp + tn
    se = tp / n_pos if n_pos else float("nan")
    sp = tn / n_neg if n_neg else float("nan")
    se_ci = stats.wilson(tp, n_pos) if n_pos else (float("nan"),) * 2
    sp_ci = stats.wilson(tn, n_neg) if n_neg else (float("nan"),) * 2
    j = se + sp - 1 if n_pos and n_neg else float("nan")
    # J is a sum of two independent proportions, so its variance adds.
    var = ((se * (1 - se) / n_pos) if n_pos else 0.0) + \
          ((sp * (1 - sp) / n_neg) if n_neg else 0.0)
    half = Z * math.sqrt(var) if var > 0 else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "n": tp + fn + fp + tn, "n_true_success": n_pos, "n_true_failure": n_neg,
            "sensitivity": se, "sensitivity_ci": list(se_ci),
            "specificity": sp, "specificity_ci": list(sp_ci),
            "youden_j": j, "youden_j_ci": [max(-1.0, j - half), min(1.0, j + half)],
            "agreement": (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) else float("nan"),
            "kappa": kappa(judge, ref),
            "judge_rate": _rate(judge), "reference_rate": _rate(ref),
            "rate_bias": _rate(judge) - _rate(ref)}


def episode_multiplier(j):
    """How many more trials a judge with this J costs, to the same resolution."""
    if not (j and j == j) or j <= 0:
        return float("inf")
    return 1.0 / (j * j)


def attenuated(true_diff_pp, j):
    return true_diff_pp * j


def corrected(observed_diff_pp, j):
    """Undo the shrinkage. Only legitimate when the judge errs the same way for
    every policy, which `differential` is what tests."""
    if not (j and j == j) or j <= 0:
        return float("nan")
    return observed_diff_pp / j


def differential(by_policy):
    """Does the judge err differently depending on which policy it is watching?

    Non-differential error shrinks a difference by a known factor and can be
    corrected. Differential error cannot: it can invent a gap that is not there.
    This compares each pair of policies' sensitivity and specificity.
    """
    names = sorted(by_policy)
    worst = None
    pairs = []
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            a, b = by_policy[names[i]], by_policy[names[k]]
            row = {"a": names[i], "b": names[k]}
            for metric, num, den in (("sensitivity", "tp", "n_true_success"),
                                     ("specificity", "tn", "n_true_failure")):
                if not (a[den] and b[den]):
                    row[metric + "_p"] = float("nan")
                    continue
                _d, _z, p = stats.two_prop_test(a[num], a[den], b[num], b[den])
                row[metric + "_p"] = p
                row[metric + "_delta"] = a[metric] - b[metric]
            ps = [row.get("sensitivity_p"), row.get("specificity_p")]
            ps = [p for p in ps if p == p]
            row["min_p"] = min(ps) if ps else float("nan")
            pairs.append(row)
            if ps and (worst is None or row["min_p"] < worst["min_p"]):
                worst = row
    n_tests = sum(1 for r in pairs for k in ("sensitivity_p", "specificity_p")
                  if r.get(k) == r.get(k))
    # ONE verdict over many tests, so the family-wise rate is what matters and it is
    # corrected explicitly. This is the opposite of the per-skill rule in `release`,
    # which reports many findings and states the expected false-flag count instead of
    # shrinking alpha: there, correcting measurably removed the flags that were doing
    # the work. Here a single uncorrected verdict fires on 10-12% of even-handed
    # judges, which would send somebody to rewrite a detector that was fine.
    alpha = 0.05 / n_tests if n_tests else 0.05
    flagged = bool(worst) and worst["min_p"] < alpha
    return {"pairs": pairs, "worst": worst, "n_tests": n_tests,
            "alpha_per_test": alpha, "differential": flagged,
            "expected_false_flags": 0.05 * n_tests}


def labels_needed(se_guess=0.8, sp_guess=0.8, half_width=0.05):
    """Reference labels needed to pin sensitivity and specificity to +/- half_width.

    Returns the count of TRUE SUCCESSES and TRUE FAILURES separately, because a
    reference set of a hundred episodes that are all successes measures
    sensitivity and says nothing at all about specificity.
    """
    def n(p):
        return int(math.ceil(Z * Z * p * (1 - p) / (half_width * half_width)))
    return {"true_successes": n(se_guess), "true_failures": n(sp_guess),
            "half_width": half_width}


def validate(records, target_pp=10.0):
    """`records` = [{policy, job, judge: bool, reference: bool}, ...]."""
    judge = [bool(r["judge"]) for r in records]
    ref = [bool(r["reference"]) for r in records]
    overall = rates(judge, ref)
    by_policy = {}
    for r in records:
        by_policy.setdefault(r.get("policy") or "all", []).append(r)
    per = {k: rates([bool(x["judge"]) for x in v], [bool(x["reference"]) for x in v])
           for k, v in by_policy.items()}
    diff = differential(per) if len(per) > 1 else {"differential": False, "pairs": [],
                                                   "worst": None, "n_tests": 0,
                                                   "expected_false_flags": 0.0}
    j = overall["youden_j"]
    return {"n_records": len(records), "n_policies": len(per),
            "overall": overall, "per_policy": per, "differential": diff,
            "target_pp": target_pp,
            "attenuated_target_pp": attenuated(target_pp, j),
            "episode_multiplier": episode_multiplier(j),
            "sizing": labels_needed()}


def format_judge(v):
    o = v["overall"]
    L = ["How much is your success detector costing you?", "=" * 72, ""]
    if not o["n_true_success"] or not o["n_true_failure"]:
        L.append("  The reference set contains only one kind of outcome (%d successes,"
                 % o["n_true_success"])
        L.append("  %d failures). Sensitivity and specificity each need examples of"
                 % o["n_true_failure"])
        L.append("  their own kind, so this cannot be priced yet. Label some of both.")
        return "\n".join(L)
    L.append("  %d episodes carry both a judge label and a reference label, across %d"
             % (o["n"], v["n_policies"]))
    L.append("  polic%s." % ("y" if v["n_policies"] == 1 else "ies"))
    L.append("")
    L.append("  agreement            %5.1f%%        (kappa %.2f)"
             % (100 * o["agreement"], o["kappa"]))
    L.append("  sensitivity          %5.1f%%   95%% [%.0f, %.0f]   of %d real successes,"
             % (100 * o["sensitivity"], 100 * o["sensitivity_ci"][0],
                100 * o["sensitivity_ci"][1], o["n_true_success"]))
    L.append("                                            it missed %d" % o["fn"])
    L.append("  specificity          %5.1f%%   95%% [%.0f, %.0f]   of %d real failures,"
             % (100 * o["specificity"], 100 * o["specificity_ci"][0],
                100 * o["specificity_ci"][1], o["n_true_failure"]))
    L.append("                                            it passed %d" % o["fp"])
    L.append("  success rate bias    %+5.1f pp      the judge reports %.0f%%, the "
             "reference %.0f%%" % (100 * o["rate_bias"], 100 * o["judge_rate"],
                                   100 * o["reference_rate"]))
    L.append("")
    j = o["youden_j"]
    lo, hi = o["youden_j_ci"]
    L.append("  WHAT THIS DOES TO A COMPARISON")
    L.append("    Youden's J = sensitivity + specificity - 1 = %.2f  95%% [%.2f, %.2f]"
             % (j, lo, hi))
    if j <= 0:
        L.append("    At or below zero this judge carries no usable signal about which")
        L.append("    policy is better. Nothing measured through it can be trusted.")
        return "\n".join(L)
    L.append("    Every real difference arrives shrunk by that factor. A %.0f-point"
             % v["target_pp"])
    L.append("    regression is reported as %.1f points." % v["attenuated_target_pp"])
    m = v["episode_multiplier"]
    if m == float("inf"):
        L.append("    No number of episodes recovers it.")
    else:
        L.append("    Resolving it anyway costs %.1fx the episodes you would need with"
                 % m)
        L.append("    a perfect detector.")
    L.append("")
    d = v["differential"]
    if d["differential"]:
        w = d["worst"]
        L.append("  THE JUDGE IS NOT EVEN-HANDED")
        L.append("    It errs differently on %s than on %s: p = %.4f against a %.4f"
                 % (w["a"], w["b"], w["min_p"], d["alpha_per_test"]))
        L.append("    threshold, corrected for the %d comparisons made."
                 % d["n_tests"])
        L.append("    Shrinkage is correctable; this is not.")
        L.append("    A gap measured through this judge may not exist at all. Do not")
        L.append("    correct for J here; fix the judge or score by hand.")
    elif v["n_policies"] > 1:
        L.append("  It errs about the same way on every policy tested (%d comparisons),"
                 % d["n_tests"])
        L.append("  so the shrinkage above is a factor rather than a fabrication, and a")
        L.append("  measured difference can be divided by J to recover the real one.")
    else:
        L.append("  Only one policy is present, so whether the judge is even-handed")
        L.append("  ACROSS policies is untested. That is the failure mode that invents")
        L.append("  differences. Label episodes from a second policy before trusting it.")
    s = v["sizing"]
    L.append("")
    L.append("  To pin both rates to within %.0f points you need about %d reference"
             % (100 * s["half_width"], s["true_successes"]))
    L.append("  labels on episodes that really succeeded and %d on episodes that really"
             % s["true_failures"])
    L.append("  failed. A reference set that is all successes measures nothing about")
    L.append("  specificity.")
    return "\n".join(L)


# ------------------------------------------------------------------ loading

JUDGE_COLS = ("judge", "predicted", "prediction", "pred", "auto", "automatic",
              "detector", "vlm", "model_label", "judge_label", "machine")
REF_COLS = ("reference", "truth", "ground_truth", "groundtruth", "human", "gold",
            "actual", "operator", "label", "human_label", "reviewed")


def load(path):
    """Read a CSV carrying both a judge label and a reference label per episode.

    Column names are matched loosely, as everywhere else in this package, and
    comment lines before the header are skipped.
    """
    lines = discover._data_lines(path)
    if not lines:
        raise ValueError("%s: empty" % path)
    sample = "".join(lines[:40])
    delim = "\t" if (path.lower().endswith(".tsv") or
                      sample.count("\t") > sample.count(",")) else ","
    rdr = csv.DictReader(lines, delimiter=delim)
    fn = rdr.fieldnames or []
    jc = discover._pick(fn, JUDGE_COLS)
    rc = discover._pick(fn, REF_COLS)
    if not jc or not rc:
        raise ValueError(
            "%s: need one column for the judge's label and one for the reference.\n"
            "Found: %s\nJudge names tried: %s\nReference names tried: %s"
            % (os.path.basename(path), ", ".join(fn) or "(none)",
               ", ".join(JUDGE_COLS[:5]), ", ".join(REF_COLS[:5])))
    if jc == rc:
        raise ValueError("%s: the judge and reference columns are the same column (%r)"
                         % (os.path.basename(path), jc))
    pc = discover._pick(fn, discover.VERSION_COLS)
    sc = discover._pick(fn, discover.SKILL_COLS)
    ec = discover._pick(fn, discover.EPISODE_COLS)
    out = []
    for i, row in enumerate(rdr):
        jv, rv = row.get(jc), row.get(rc)
        if jv in (None, "") or rv in (None, ""):
            continue                       # unlabelled episodes are simply not scored
        try:
            j, r = discover._truthy(jv), discover._truthy(rv)
        except ValueError as e:
            raise ValueError("%s row %d: %s" % (os.path.basename(path), i + 2, e))
        out.append({"policy": (row.get(pc) or "").strip() if pc else "",
                    "job": (row.get(sc) or "").strip() if sc else "",
                    "episode": (row.get(ec) or "").strip() if ec else str(i),
                    "judge": j, "reference": r})
    if not out:
        raise ValueError("%s: no rows carry both labels" % os.path.basename(path))
    return out
