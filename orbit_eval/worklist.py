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

"""orbit-eval worklist — the episodes that changed, which are the ones to watch.

`check` says a job lost eighteen points. That is a number, and a number is not a
bug report. The next question is always "show me", and the honest answer is
already sitting in the evaluation: if both policies started episode 7 of
`bin_pick` from the same state, and the old one solved it and the new one did
not, then episode 7 is a reproducible counterexample with two videos of it.

Those are the DISCORDANT pairs, the same n10 and n01 McNemar already counts to
decide whether the drop is real. The statistics use them to price the change;
nobody has been using them to point at the change. Both readings come from one
number and only one of them has ever been printed.

What this does NOT do, stated plainly because the gap matters:

  It does not say WHY. It says which episodes, and where the recordings are. A
  survey of roughly 150 robot verifiers (arXiv 2609.09250) finds credibility
  falls as availability rises, and the strongest general vision-language judge
  measured across fourteen public sources reaches 0.77 balanced accuracy, with
  no model above 0.60 where success depends on fine contact (arXiv 2609.03611).
  A judge that is wrong one time in four cannot supply the label; it can only
  order the queue. So this orders the queue and leaves the verdict to a person.

  It is only a worklist if episode k means the same starting state for both
  policies. Equal length is not alignment. Without recorded episode ids the
  caller must assert that, and the record says who asserted it.
"""

import os

BROKE = "BROKE"      # the incumbent solved it, the candidate did not
FIXED = "FIXED"      # the candidate solved it, the incumbent did not


def _rate(bits):
    return 100.0 * sum(1 for b in bits if b) / len(bits) if bits else float("nan")


def build(data, incumbent, candidate, media=None, crn_asserted=False, limit_per_job=6):
    """Per job, the episodes whose outcome flipped between two policies.

    `data` is {policy: {job: [bool, ...]}}; `media` is the optional
    {policy: {job: [path, ...]}} carried by the discovery layer. Episodes are
    matched by index, which is only meaningful under common random numbers.
    """
    media = media or {}
    jobs = sorted(set(data.get(incumbent, {})) & set(data.get(candidate, {})))
    out, n_broke, n_fixed = [], 0, 0
    for job in jobs:
        A, B = data[incumbent][job], data[candidate][job]
        n = min(len(A), len(B))
        broke, fixed = [], []
        for i in range(n):
            if bool(A[i]) and not bool(B[i]):
                broke.append(i)
            elif bool(B[i]) and not bool(A[i]):
                fixed.append(i)
        if not broke and not fixed:
            continue
        n_broke += len(broke)
        n_fixed += len(fixed)

        def clip(pol, j, i):
            v = (media.get(pol) or {}).get(j)
            return v[i] if isinstance(v, list) and i < len(v) else None

        rows = []
        for kind, idxs in ((BROKE, broke), (FIXED, fixed)):
            for i in idxs[:limit_per_job]:
                rows.append({"episode": i, "kind": kind,
                             "incumbent_video": clip(incumbent, job, i),
                             "candidate_video": clip(candidate, job, i)})
        out.append({
            "job": job, "n_episodes": n,
            "broke": len(broke), "fixed": len(fixed),
            "net": len(fixed) - len(broke),
            "incumbent_pct": _rate(A[:n]), "candidate_pct": _rate(B[:n]),
            "episodes": rows,
            "truncated": max(0, len(broke) - limit_per_job) +
                         max(0, len(fixed) - limit_per_job),
        })
    out.sort(key=lambda r: (r["net"], -r["broke"]))
    have_media = any(e.get("incumbent_video") or e.get("candidate_video")
                     for r in out for e in r["episodes"])
    return {
        "incumbent": incumbent, "candidate": candidate,
        "jobs": out, "n_broke": n_broke, "n_fixed": n_fixed,
        "n_discordant": n_broke + n_fixed,
        "crn_asserted": bool(crn_asserted),
        "has_media": have_media,
    }


def format_worklist(w, limit_jobs=6):
    if not w["jobs"]:
        return ("  Every episode came out the same way for both policies. There is\n"
                "  nothing to watch, which is itself worth knowing.")
    L = []
    L.append("  %d episode%s changed outcome: %d broke, %d %s fixed."
             % (w["n_discordant"], "" if w["n_discordant"] == 1 else "s",
                w["n_broke"], w["n_fixed"], "was" if w["n_fixed"] == 1 else "were"))
    if not w["crn_asserted"]:
        L.append("  Episodes are matched by POSITION and nobody has asserted that")
        L.append("  position k is the same starting state for both. Re-run with --crn")
        L.append("  if it is; until then treat this as a lead, not a counterexample.")
    L.append("")
    for r in w["jobs"][:limit_jobs]:
        head = ("  %s   %.0f%% -> %.0f%%   %d broke, %d fixed"
                % (r["job"], r["incumbent_pct"], r["candidate_pct"],
                   r["broke"], r["fixed"]))
        L.append(head)
        for e in r["episodes"]:
            mark = "broke" if e["kind"] == BROKE else "fixed"
            L.append("      episode %-4d %s" % (e["episode"], mark))
            if e["incumbent_video"]:
                L.append("         was  %s" % e["incumbent_video"])
            if e["candidate_video"]:
                L.append("         now  %s" % e["candidate_video"])
        if r["truncated"]:
            L.append("      ... and %d more" % r["truncated"])
        L.append("")
    if len(w["jobs"]) > limit_jobs:
        L.append("  (%d more jobs changed; --json has all of them)"
                 % (len(w["jobs"]) - limit_jobs))
    if not w["has_media"]:
        L.append("  No per-episode recordings were found next to these results. The")
        L.append("  episode numbers still reproduce: re-run those indices with the")
        L.append("  same eval seed.")
    return "\n".join(L)


def markdown(w, limit_jobs=6):
    if not w["jobs"]:
        return ""
    L = ["", "<details><summary>Episodes that changed (%d broke, %d fixed)</summary>"
         % (w["n_broke"], w["n_fixed"]), ""]
    if not w["crn_asserted"]:
        L.append("_Matched by position; nobody asserted that position k is the same "
                 "starting state for both policies._")
        L.append("")
    for r in w["jobs"][:limit_jobs]:
        L.append("**`%s`** %.0f%% &rarr; %.0f%% &mdash; %d broke, %d fixed"
                 % (r["job"], r["incumbent_pct"], r["candidate_pct"],
                    r["broke"], r["fixed"]))
        for e in r["episodes"]:
            bits = ["episode %d %s" % (e["episode"],
                                       "broke" if e["kind"] == BROKE else "fixed")]
            if e["incumbent_video"]:
                bits.append("was `%s`" % os.path.basename(e["incumbent_video"]))
            if e["candidate_video"]:
                bits.append("now `%s`" % os.path.basename(e["candidate_video"]))
            L.append("- " + ", ".join(bits))
        L.append("")
    L.append("</details>")
    return "\n".join(L)
