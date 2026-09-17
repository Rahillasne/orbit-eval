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

"""orbit-eval check — run it in the folder your evaluations are already in.

`release` is the same question and it asks you to hand it a tidy CSV first. This
one asks for nothing: it walks the directory, works out which files are
evaluations, which rows belong to the same trained policy, which policy is the one
you are already running, and then answers per job whether the new one broke
anything.

Everything it decides about the INPUTS is a heuristic and is printed so it can be
overridden. Everything it decides about the ANSWER comes from `release.check`,
unchanged, and lands in a signed record that reproduces without this command.
"""

import html as _html
import json
import os
import time
import webbrowser

from . import discover
from . import history as history_mod
from . import nextstep
from . import release as release_mod
from . import worklist as worklist_mod

VERDICT_ORDER = ["REGRESSED", "SUSPECT", "UNDERPOWERED", "held", "IMPROVED"]


def choose(source, incumbent=None, candidate=None):
    """Pick the two versions to compare and say why, so the guess can be argued with."""
    order = discover.order_by_time(source)
    if incumbent and incumbent not in source.data:
        raise ValueError("no policy named %r (found: %s)"
                         % (incumbent, ", ".join(order)))
    if candidate and candidate not in source.data:
        raise ValueError("no policy named %r (found: %s)"
                         % (candidate, ", ".join(order)))
    if incumbent and candidate:
        return incumbent, candidate, "named on the command line"
    if len(order) < 2:
        return (order[0] if order else None), None, "only one policy found"
    if candidate:
        inc = next((c for c in reversed(order) if c != candidate), order[0])
        return inc, candidate, "incumbent is the next-oldest policy"
    if incumbent:
        cand = next((c for c in reversed(order) if c != incumbent), order[-1])
        return incumbent, cand, "candidate is the newest policy"
    same = len({source.mtimes.get(c, 0.0) for c in order}) == 1
    why = ("alphabetical: every policy has the same timestamp, so file order cannot say "
           "which is newer" if same else
           "oldest file is taken as what you ship today, newest as the candidate")
    return order[0], order[-1], why


def run(root=".", incumbent=None, candidate=None, target_pp=nextstep.TARGET_PP,
        drop_pp=release_mod.DROP_PP, min_episodes=release_mod.MIN_EPISODES,
        crn=False, want_history=None):
    """Scan, choose, check, diagnose. Returns everything the formatters need."""
    sources, rejected = discover.scan(root, with_rejects=True)
    rejects = [{"path": p, "why": w} for p, w in rejected[:8]]
    if not sources:
        return {"ok": False, "root": os.path.abspath(root), "sources": [],
                "rejected": rejects}
    src = sources[0]
    members, skills, dropped = discover.comparable_group(src.data)
    named = [n for n in (incumbent, candidate) if n]
    if named and all(n in src.data for n in named):
        # an explicit pair outranks the automatic grouping
        pair_skills = sorted(set.intersection(*(set(src.data[n]) for n in named))) \
            if len(named) == 2 else skills
        if pair_skills:
            members, skills = sorted(set(named) | set(members)), pair_skills
            members = [m for m in members if pair_skills and
                       set(pair_skills).issubset(set(src.data[m]))]
            dropped = sorted(set(src.data) - set(members))
    if not skills or len(members) < 1:
        return {"ok": False, "root": os.path.abspath(root), "sources": sources,
                "rejected": rejects, "jobs_by_policy":
                    {c: sorted(v)[:6] for c, v in list(src.data.items())[:12]},
                "error": "no two policies here ran the same jobs"}
    group = discover.Source(src.kind, src.path, {c: src.data[c] for c in members},
                            src.mtimes, src.note, paired_ok=src.paired_ok,
                            reasons=src.reasons, media=src.media,
                            seeds=getattr(src, "seeds", None))
    data = discover.trim_to_common(group.data, skills)
    inc, cand, why = choose(group, incumbent, candidate)
    # lerobot-eval records every episode's seed. Same seed at every index IS
    # common random numbers, verified rather than asserted; different seeds mean
    # episode k is a different start state for the two runs and no
    # episode-level comparison exists, whatever the caller asserts.
    verified, mismatch, seed_note = _seed_alignment(group.seeds, inc, cand, skills, data)
    if verified:
        crn = True
    if mismatch:
        crn = False
    out = {
        "ok": True, "root": os.path.abspath(root),
        "source": {"kind": src.kind, "path": os.path.relpath(src.path, root)
                   if src.path.startswith(os.path.abspath(root)) else src.path,
                   "summary": src.summary(), "note": src.note},
        "other_sources": [{"kind": s.kind, "path": s.path, "summary": s.summary()}
                          for s in sources[1:6]],
        "policies": discover.order_by_time(group),
        "policies_dropped": dropped,
        "incumbent": inc, "candidate": cand, "pairing_reason": why,
        "n_skills": len(skills),
        "paired": src.paired_ok,
        "rejected": rejects,
        "trimmed": {s: min(len(group.data[c][s]) for c in group.data) for s in skills},
        "next": nextstep.diagnose(data, target_pp=target_pp,
                                  reasons={c: v for c, v in (group.reasons or {}).items()
                                           if c in group.data},
                                  rounds=(len(members) >= 3 if want_history is None
                                          else bool(want_history))),
    }
    if cand is None:
        out["check"] = None
        return out
    # Counts expanded back into trials look perfectly paired and are not: the
    # individual episodes are gone. Pairing them would invent agreement nobody
    # measured and report an interval far too tight.
    # --crn is the caller asserting that episode k is the same starting state for
    # every policy. It is the one fact this tool cannot check and the one that makes
    # a paired comparison legitimate, so it is asserted explicitly and recorded as an
    # assertion rather than inferred from equal list lengths.
    ids = None
    if crn and group.paired_ok:
        ids = {c: {s: list(range(len(data[c][s]))) for s in data[c]} for c in data}
    out["crn_asserted"] = bool(crn)
    out["crn_verified"] = bool(verified)
    out["seed_mismatch"] = bool(mismatch)
    out["seed_note"] = seed_note
    out["check"] = release_mod.check(data, inc, cand, drop_pp=drop_pp,
                                     min_episodes=min_episodes,
                                     paired=group.paired_ok, episode_ids=ids)
    # No episode list when the episodes are not real: trials rebuilt from a
    # rate have no identity, and runs with different seeds have no alignment.
    out["worklist"] = None if (mismatch or not group.paired_ok) else worklist_mod.build(
        data, inc, cand, media=group.media, crn_asserted=crn)
    # What would asserting common random numbers buy? Computed here so the flag can
    # advertise itself at the moment it is relevant, instead of living in a manual.
    out["crn_gain_pp"] = None
    if group.paired_ok and not crn and not mismatch and out["check"] and out["check"]["rows"]:
        pid = {c: {sk: list(range(len(data[c][sk]))) for sk in data[c]} for c in data}
        try:
            alt = release_mod.check(data, inc, cand, drop_pp=drop_pp,
                                    min_episodes=min_episodes, paired=True,
                                    episode_ids=pid)
            def _w(c):
                return sum(r["ci95"][1] - r["ci95"][0] for r in c["rows"]) / len(c["rows"])
            gain = _w(out["check"]) - _w(alt)
            out["crn_gain_pp"] = gain if gain > 1.0 else None
        except Exception:
            pass
    # Auto: three or more rounds IS a chain, and the failure a chain has is the one
    # nobody would think to ask for. Asking for it with a flag means never seeing it.
    if want_history is None:
        want_history = len(members) >= 3
    if want_history:
        out["history"] = history_mod.audit(
            data, discover.order_by_time(group), drop_pp=drop_pp,
            min_episodes=min_episodes, paired=group.paired_ok and crn,
            episode_ids=ids)
    return out


def _seed_alignment(seeds, inc, cand, skills, data):
    """(verified, mismatch, note) from the per-episode seeds two runs recorded."""
    seeds = seeds or {}
    a_all, b_all = seeds.get(inc) or {}, seeds.get(cand) or {}
    matched, differ = [], []
    for sk in skills:
        a, b = a_all.get(sk), b_all.get(sk)
        if not a or not b:
            continue
        n = min(len(a), len(b), len(data.get(inc, {}).get(sk, [])),
                len(data.get(cand, {}).get(sk, [])))
        if n and a[:n] == b[:n]:
            matched.append(sk)
        elif n:
            differ.append((sk, a[0], b[0]))
    if differ:
        sk, sa, sb = differ[0]
        return False, True, ("the two runs used different eval seeds (%s vs %s%s), so "
                             "episode k is not the same start state for both and no "
                             "episode-level comparison is shown"
                             % (sa, sb, "" if len(differ) == 1 else
                                ", and %d more jobs" % (len(differ) - 1)))
    if matched and len(matched) == len(skills):
        return True, False, ("both runs recorded the same eval seed for every episode, "
                             "so the comparison is paired (common random numbers, "
                             "verified from the files)")
    return False, False, None


# ------------------------------------------------------------------ terminal

def format_check(res, root=".", why=False):
    """Short by default. A tool that prints eighty lines to answer one question
    gets read once. `why` restores every caveat, and they are all still in the
    report and in --json."""
    if not res.get("ok"):
        return format_nothing(res, root)
    if why:
        return format_check_full(res, root)
    src = res["source"]
    L = ["", "  %s" % src["summary"]]
    if res["candidate"] is None:
        L.append("")
        L.append("  Only one trained policy here, so there is nothing to compare it")
        L.append("  against. Train the same recipe twice more and re-run.")
        return "\n".join(L)
    c = res["check"]
    L.append("  %s -> %s   (%s)" % (res["incumbent"], res["candidate"],
                                     "named" if "command line" in res["pairing_reason"]
                                     else "guessed; --incumbent/--candidate to change"))
    L.append("")
    shown = [r for r in c["rows"] if r["verdict"] != "held"]
    for r in (shown or c["rows"][:1]):
        L.append("    %-12s %-20s %3.0f%% -> %3.0f%%  %+6.1f  [%+.0f, %+.0f]"
                 % (r["verdict"], r["skill"][:20], r["incumbent_pct"],
                    r["candidate_pct"], r["delta_pp"], r["ci95"][0], r["ci95"][1]))
    held = c["n_skills"] - len(shown)
    if held > 0:
        L.append("    %-12s %d job%s" % ("held", held, "" if held == 1 else "s"))
    L.append("")
    verdict = ("%d regressed" % c["n_regressed"] if c["n_regressed"]
               else "nothing regressed beyond your measurement noise")
    L.append("  %s. Smallest drop this battery can call: %.0f points."
             % (verdict, c["median_resolvable_drop_pp"]))
    h = res.get("history")
    if h and not h.get("too_short"):
        bits = ["%d rounds" % h["rounds"],
                "%d broke a job" % h["rounds_with_a_regression"]]
        L.append("  loop: " + ", ".join(bits) + ".")
        if h["creeping_rot"]:
            L.append("  CREEPING ROT: %s. No single round saw it; %d rounds together did."
                     % (", ".join(h["creeping_rot"]), len(h["steps"])))
        if h["silent_rot"]:
            L.append("  SILENT ROT: %d round(s) gained on average while breaking a job."
                     % len(h["silent_rot"]))
    w = res.get("worklist")
    if w and w["n_broke"]:
        L.append("  %d episode%s that used to work now fail%s%s."
                 % (w["n_broke"], "" if w["n_broke"] == 1 else "s",
                    "s" if w["n_broke"] == 1 else "",
                    "" if w["crn_asserted"] else " (matched by position; --crn if "
                                                 "position k shares a seed)"))
    if res.get("seed_mismatch") and res.get("seed_note"):
        L.append("  " + res["seed_note"] + ".")
    nx = res.get("next")
    if nx and nx["rows"]:
        top = nx["rows"][0]
        if top["diagnosis"] != "FINE":
            L.append("")
            L.append("  next: %s -- %s" % (top["skill"], top["action"]))
    if res.get("crn_gain_pp"):
        L.append("  tip:  same eval seed for every policy? --crn narrows these "
                 "intervals by %.0f points." % res["crn_gain_pp"])
    return "\n".join(L)


def format_check_full(res, root="."):
    L = []
    src = res["source"]
    L.append("  found  %s" % src["summary"])
    L.append("         %s" % src["path"])
    for o in res["other_sources"]:
        L.append("  also   %s  (%s)" % (o["summary"], os.path.basename(o["path"])))
    for r in res.get("rejected", []):
        L.append("  skip   %s: %s" % (os.path.basename(r["path"]), r["why"]))
    if res.get("policies_dropped"):
        L.append("  left out %s: %s"
                 % ("it ran" if len(res["policies_dropped"]) == 1 else "they ran",
                    ", ".join(res["policies_dropped"][:6])))
        L.append("         (different jobs from the group above; name one with "
                 "--candidate to compare it instead)")
    if not res.get("paired", True):
        L.append("         trials were rebuilt from summary counts, so this is an "
                 "unpaired comparison")
    if res.get("seed_note"):
        L.append("         " + res["seed_note"])
    if res["candidate"] is None:
        L.append("")
        L.append("Only one trained policy is here, so there is nothing to compare it")
        L.append("against. `orbit check` answers 'did the new one break anything', which")
        L.append("needs the old one too. What can be said about this one alone is below.")
        L.append("")
        L.append(nextstep.format_next(res["next"]))
        return "\n".join(L)
    L.append("")
    L.append("  shipping today : %s" % res["incumbent"])
    L.append("  candidate      : %s" % res["candidate"])
    L.append("  (%s; override with --incumbent / --candidate)" % res["pairing_reason"])
    L.append("")
    L.append(release_mod.format_check(res["check"]))
    h = res.get("history")
    if h and not h.get("too_short"):
        L.append("")
        L.append("  " + "=" * 72)
        L.append("  THE LOOP, ROUND BY ROUND")
        L.append("  " + "=" * 72)
        L.append(history_mod.format_history(h))
    w = res.get("worklist")
    if w and w["n_discordant"]:
        L.append("")
        L.append("  " + "=" * 72)
        L.append("  WHICH EPISODES TO WATCH")
        L.append("  " + "=" * 72)
        L.append(worklist_mod.format_worklist(w))
    return "\n".join(L)


def format_nothing(res, root=".", prog="orbit check"):
    jbp = res.get("jobs_by_policy")
    if jbp:
        L = ["%s: %d polic%s found under %s, but no two of them ran the same jobs."
             % (prog, len(jbp), "y was" if len(jbp) == 1 else "ies were", res.get("root", root)),
             ""]
        for c, js in jbp.items():
            L.append("  %-28s %s" % (c[:28], ", ".join(js) or "(no jobs)"))
        L += ["",
              "A comparison needs at least two policies evaluated on the same named job.",
              "If these are the same job under different names, rename them and re-run;",
              "if they are genuinely different benchmarks, they cannot be compared."]
        return "\n".join(L)
    rej = res.get("rejected") or []
    near = ([""] + ["These came close and were skipped:", ""] +
            ["  %s" % r["path"] for r in []] +
            ["  %-42s %s" % (os.path.basename(r["path"]), r["why"]) for r in rej] +
            [""]) if rej else []
    return "\n".join([
        "%s: no evaluations found under %s" % (prog, res.get("root", root)),
        ] + near + [
        "",
        "It looks for three things, all of which it can read as they are:",
        "",
        "  eval_info.json   anything lerobot-eval wrote, at any depth",
        "  a CSV            with columns for the policy, the job, the trial and whether",
        "                   it succeeded. Header names are matched loosely, so",
        "                   model/task/trial/passed works as well as",
        "                   version/skill/episode/success",
        "  a JSON matrix    {\"candidates\": {name: {job: [1,0,1, ...]}}}",
        "",
        "Point it somewhere else with `orbit check PATH`. If you have trials on paper",
        "and nothing on disk yet, `orbit log` records them as you run them.",
    ])


# ---------------------------------------------------------------------- html

_CSS = """
:root{--bg:#fbfaf9;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e4e1dd;--card:#fff;
--bad:#b3261e;--warn:#9a6700;--good:#1a7f37;--thin:#57534e;--accent:#6d4a7c}
@media(prefers-color-scheme:dark){:root{--bg:#131211;--fg:#eceae8;--mut:#9b9793;
--line:#2c2a28;--card:#1b1a18;--bad:#f2857c;--warn:#e0b341;--good:#5fd07f;--thin:#a8a29e;
--accent:#c79ad8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:var(--mut);font-size:14px;margin-bottom:24px}
.head{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 26px}
.kpi{flex:1 1 150px;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px;letter-spacing:-.02em}
.kpi span{color:var(--mut);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font-weight:600;color:var(--mut);font-size:12px;
text-transform:uppercase;letter-spacing:.06em;padding:0 8px 8px;border-bottom:1px solid var(--line)}
td{padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
font-weight:600;letter-spacing:.03em}
.REGRESSED{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.SUSPECT{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.UNDERPOWERED{background:color-mix(in srgb,var(--thin) 16%,transparent);color:var(--thin)}
.IMPROVED{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.held{background:color-mix(in srgb,var(--mut) 14%,transparent);color:var(--mut)}
.bar{position:relative;height:8px;background:color-mix(in srgb,var(--mut) 18%,transparent);
border-radius:4px;min-width:90px}
.bar i{position:absolute;top:0;bottom:0;left:0;border-radius:4px;background:var(--accent)}
.bar u{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--fg);opacity:.55}
h2{font-size:15px;margin:38px 0 10px;letter-spacing:-.01em}
.note{color:var(--mut);font-size:13px;margin:10px 0}
.adv{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:8px 0}
.adv b{font-size:13px}.adv p{margin:4px 0 0;color:var(--mut);font-size:13px}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;
background:color-mix(in srgb,var(--mut) 12%,transparent);padding:1px 5px;border-radius:4px}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
color:var(--mut);font-size:12px;word-break:break-all}
@media(max-width:560px){.wrap{padding:20px 14px 48px}h1{font-size:21px}
th:nth-child(3),td:nth-child(3){display:none}}
.alarm{background:color-mix(in srgb,var(--bad) 10%,var(--card));
border:1px solid color-mix(in srgb,var(--bad) 35%,var(--line));border-radius:10px;
padding:14px 16px;margin:12px 0}
.alarm b{color:var(--bad)}
.chartbox{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 14px 6px;margin:12px 0;overflow-x:auto}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);
margin:8px 2px 4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;
vertical-align:-1px}
.rounds{display:flex;gap:3px;flex-wrap:wrap;margin:8px 0}
.rounds span{flex:1 1 14px;height:22px;border-radius:3px;min-width:14px;
background:color-mix(in srgb,var(--good) 30%,transparent)}
.rounds span.bad{background:color-mix(in srgb,var(--bad) 55%,transparent)}
"""


def _e(s):
    return _html.escape(str(s))


def _trend_svg(h, w=880, hgt=250, pad=34, rpad=96):
    """One line per job across the rounds. Regressed jobs in the alarm colour."""
    trails = h.get("trails") or {}
    if not trails:
        return ""
    n = h["rounds"]
    if n < 2:
        return ""
    regressed = {r["skill"] for r in h["net"]["rows"] if r["verdict"] == "REGRESSED"}
    iw, ih = w - pad - rpad, hgt - 2 * pad
    def X(i):
        return pad + iw * i / float(n - 1)
    def Y(v):
        return pad + ih * (1.0 - max(0.0, min(100.0, v)) / 100.0)
    P = ['<div class=chartbox><svg viewBox="0 0 %d %d" width="100%%" height="%d" '
         'role="img" aria-label="success rate per job across rounds">' % (w, hgt, hgt)]
    for g in (0, 25, 50, 75, 100):
        P.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" stroke="currentColor" '
                 'stroke-opacity=".13"/>' % (pad, Y(g), X(n - 1), Y(g)))
        P.append('<text x="%d" y="%.1f" font-size="10" fill="currentColor" '
                 'fill-opacity=".45" text-anchor="end">%d%%</text>'
                 % (pad - 6, Y(g) + 3, g))
    # Lay the end labels out so they never sit on top of each other: sort by final
    # height and push each down until it clears the one above.
    ends = []
    for skill, vals in trails.items():
        finite = [v for v in vals if v == v]
        if len(finite) >= 2:
            ends.append((Y(finite[-1]), skill))
    ends.sort()
    label_y, prev = {}, None
    for y, skill in ends:
        y = y if prev is None else max(y, prev + 13)
        label_y[skill] = y
        prev = y
    for skill, vals in sorted(trails.items()):
        pts = [(X(i), Y(v)) for i, v in enumerate(vals) if v == v]
        if len(pts) < 2:
            continue
        bad = skill in regressed
        col = "var(--bad)" if bad else "var(--accent)"
        P.append('<polyline fill="none" stroke="%s" stroke-width="%s" '
                 'stroke-opacity="%s" stroke-linejoin="round" points="%s"/>'
                 % (col, "2.6" if bad else "1.7", "1" if bad else ".5",
                    " ".join("%.1f,%.1f" % p for p in pts)))
        ly = label_y.get(skill, pts[-1][1])
        if abs(ly - pts[-1][1]) > 2:      # leader line when the label had to move
            P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                     'stroke-opacity=".35" stroke-width="1"/>'
                     % (pts[-1][0], pts[-1][1], pts[-1][0] + 6, ly, col))
        P.append('<text x="%.1f" y="%.1f" font-size="11" fill="%s" '
                 'fill-opacity="%s">%s</text>'
                 % (pts[-1][0] + 9, ly + 3.5, col, "1" if bad else ".65",
                    _e(skill[:16])))
    P.append('<text x="%d" y="%d" font-size="10" fill="currentColor" '
             'fill-opacity=".45">round 1</text>' % (pad, hgt - 8))
    P.append('<text x="%.1f" y="%d" font-size="10" fill="currentColor" '
             'fill-opacity=".45" text-anchor="end">round %d</text>'
             % (X(n - 1), hgt - 8, n))
    P.append("</svg></div>")
    return "".join(P)


def _bar(inc, cand):
    return ('<div class="bar"><i style="width:%.1f%%"></i><u style="left:%.1f%%"></u></div>'
            % (max(0.0, min(100.0, cand)), max(0.0, min(100.0, inc))))


def html_report(res, sig=None):
    c = res.get("check")
    nx = res["next"]
    t = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    P = ['<!doctype html><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">',
         "<title>orbit check</title><style>%s</style><div class=wrap>" % _CSS]
    P.append("<h1>Which jobs did the new policy break?</h1>")
    if c:
        P.append('<div class=sub>%s against %s &middot; %d jobs &middot; %s</div>'
                 % (_e(res["candidate"]), _e(res["incumbent"]), res["n_skills"], t))
        P.append('<div class=head>')
        for n, lab in ((c["n_regressed"], "regressed"), (c["n_suspect"], "suspect"),
                       (c["n_improved"], "improved"), (c["n_underpowered"], "underpowered")):
            P.append('<div class=kpi><b>%d</b><span>%s</span></div>' % (n, lab))
        P.append('</div>')
        P.append("<table><tr><th>job</th><th>shipping &rarr; candidate</th><th></th>"
                 "<th class=num>change</th><th class=num>95% interval</th><th></th></tr>")
        for r in c["rows"]:
            P.append("<tr><td>%s</td><td class=num>%.0f%% &rarr; %.0f%%</td><td>%s</td>"
                     "<td class=num>%+.1f</td><td class=num>[%+.0f, %+.0f]</td>"
                     "<td><span class='chip %s'>%s</span></td></tr>"
                     % (_e(r["skill"]), r["incumbent_pct"], r["candidate_pct"],
                        _bar(r["incumbent_pct"], r["candidate_pct"]), r["delta_pp"],
                        r["ci95"][0], r["ci95"][1], r["verdict"], r["verdict"]))
        P.append("</table>")
        P.append('<p class=note>The bar is the candidate; the tick is what you ship today. '
                 'A job is called <b>regressed</b> only when the whole interval sits at or '
                 'below &minus;%.0f points, so a drop inside your own measurement noise is '
                 'reported as <b>suspect</b> rather than asserted. With %d jobs tested at 5%%, '
                 'about %.1f false flags are expected and no correction has been applied '
                 'silently.</p>'
                 % (release_mod.DROP_PP, c["n_skills"], c["expected_false_flags"]))
        P.append('<p class=note>Suite average moved %+.1f points. That number is the mean of '
                 'the column above and cannot disagree with it in sign, which is the only '
                 'reason it is printed.</p>' % c["suite_delta_pp"])
    else:
        P.append('<div class=sub>One policy found, nothing to compare against &middot; %s</div>'
                 % t)
    h = res.get("history")
    if h and not h.get("too_short"):
        P.append("<h2>The loop, round by round</h2>")
        if h["creeping_rot"]:
            P.append('<div class=alarm><b>Creeping rot: %s.</b><p>No single round broke '
                     '%s. %d rounds together did. A few points a round sits under the '
                     'resolution of every round and over it by the end, so a pairwise '
                     'check would never once have fired.</p></div>'
                     % (_e(", ".join(h["creeping_rot"])),
                        "it" if len(h["creeping_rot"]) == 1 else "them",
                        len(h["steps"])))
        if h["silent_rot"]:
            P.append('<div class=alarm><b>%d round(s) reported a higher average while '
                     'breaking a job.</b><p>That is the shape a loop scoring itself '
                     'produces: the average is the number that improves, the job is the '
                     'number a customer notices.</p></div>' % len(h["silent_rot"]))
        lo, hi = h["per_round_risk_ci"]
        P.append('<div class=head>')
        P.append('<div class=kpi><b>%d</b><span>rounds</span></div>' % h["rounds"])
        P.append('<div class=kpi><b>%.0f%%</b><span>rounds that broke a job</span></div>'
                 % (100 * h["per_round_risk"]))
        pr = h["projection"].get(25) or h["projection"].get(10) or {}
        if pr:
            P.append('<div class=kpi><b>%.0f%%</b><span>chance something is broken by '
                     'round %d</span></div>' % (100 * pr["point"],
                                                25 if 25 in h["projection"] else 10))
        P.append('<div class=kpi><b>%+.1f</b><span>suite, end to end (pts)</span></div>'
                 % h["net_suite_delta_pp"])
        P.append('</div>')
        P.append(_trend_svg(h))
        P.append('<div class=rounds>' + "".join(
            '<span class="%s" title="%s to %s"></span>'
            % ("bad" if st["regressed"] else "", _e(st["from"]), _e(st["to"]))
            for st in h["steps"]) + '</div>')
        P.append('<p class=note>One block per round, oldest first. Red is a round that '
                 'broke at least one job. Rounds are counted as independent when '
                 'projecting forward, which flatters a loop that trains on its own '
                 'output.</p>')
    P.append("<h2>What to do next</h2>")
    if nx["underpowered_for_target"]:
        P.append('<p class=note>At %d trials a job has to drop %.0f points before it can be '
                 'called. To call a %.0f-point drop you need %d trials per copy per job, '
                 '%d more each and %d in total.</p>'
                 % (nx["median_episodes"], nx["resolvable_drop_pp"], nx["target_pp"],
                    nx["episodes_for_target"], nx["extra_episodes_per_job"],
                    nx["extra_episodes_total"]))
    # Grouped, not one card per job: eight cards that all say "retrain" is the same
    # sentence eight times, which is how a report stops being read.
    groups = [
        ("UNLUCKY", "Retrain these",
         "They swing further apart than sampling explains, so this is the training "
         "lottery rather than your data. Train the recipe again and let the per-job "
         "pick keep the incumbent wherever the evidence is not there."),
        ("WEAK", "Collect demonstrations for these",
         "Every copy is bad here. Neither more trials nor more seeds will move them."),
        ("THIN", "Even out the battery on these",
         "They carry far fewer trials than the rest, so they are the jobs you are "
         "least able to defend."),
    ]
    for key, title, why in groups:
        rows = [r for r in nx["rows"] if r["diagnosis"] == key]
        if not rows:
            continue
        P.append('<div class=adv><b>%s (%d)</b><p>%s</p><p>' % (_e(title), len(rows), _e(why)))
        P.append(" &middot; ".join(
            "<b>%s</b> %s" % (_e(r["skill"]),
                              ("%.0f&ndash;%.0f%%, %.0f pts past a %.0f-pt floor"
                               % (r["worst_pct"], r["best_pct"], r["lottery_pp"],
                                  r["noise_floor_pp"])) if key == "UNLUCKY" else
                              ("best %.0f%%" % r["best_pct"]) if key == "WEAK" else
                              ("%d trials" % r["n_episodes"]))
            for r in rows))
        P.append("</p></div>")
    if nx["counts"].get("FINE"):
        P.append('<p class=note>%d job(s) need nothing: %s.</p>'
                 % (nx["counts"]["FINE"],
                    _e(", ".join(r["skill"] for r in nx["rows"] if r["diagnosis"] == "FINE"))))
    rt = nx.get("retrain")
    if rt and rt["retrains_for_5pct_risk"]:
        P.append('<p class=note>Copies to train: <b>%d</b>. Measured on %d tasks in %s. '
                 'Copy counts do not transfer between setups; this is a starting point, not '
                 'a law.</p>' % (rt["retrains_for_5pct_risk"], rt["cell_tasks"], _e(rt["cell"])))
    w = res.get("worklist")
    if w and w["n_discordant"]:
        P.append("<h2>Which episodes to watch</h2>")
        P.append('<p class=note>%d episode(s) changed outcome: %d broke, %d were fixed.%s'
                 '</p>' % (w["n_discordant"], w["n_broke"], w["n_fixed"],
                           "" if w["crn_asserted"] else
                           " Matched by position; nobody asserted that position k is the "
                           "same starting state for both policies."))
        for r in w["jobs"][:6]:
            P.append('<div class=adv><b>%s</b> &mdash; %.0f%% &rarr; %.0f%%, %d broke, '
                     '%d fixed<p>%s</p></div>'
                     % (_e(r["job"]), r["incumbent_pct"], r["candidate_pct"],
                        r["broke"], r["fixed"],
                        " &middot; ".join(
                            "ep %d %s%s" % (e["episode"],
                                            "broke" if e["kind"] == "BROKE" else "fixed",
                                            (" <code>%s</code>" % _e(os.path.basename(
                                                e["candidate_video"])))
                                            if e["candidate_video"] else "")
                            for e in r["episodes"])))
    P.append("<footer>Read from <code>%s</code> (%s). Reproduce with "
             "<code>orbit check %s</code>."
             % (_e(res["source"]["path"]), _e(res["source"]["kind"]), _e(res["root"])))
    if sig:
        P.append("<br>Signed record sha256 <code>%s</code>." % _e(sig))
    P.append("<br>orbit-eval, Apache-2.0. The decision above is stdlib arithmetic and "
             "reproduces without this page.</footer></div>")
    return "".join(P)


def markdown_report(res, sig=None, title="orbit check"):
    """A PR comment. Short by construction: a comment nobody finishes is a comment
    nobody acts on, so the table is the finding and everything else is one line."""
    c = res.get("check")
    L = ["### %s" % title, ""]
    if not res.get("ok"):
        L.append("No comparable evaluations found in this tree.")
        for r in res.get("rejected", [])[:5]:
            L.append("- skipped `%s`: %s" % (os.path.basename(r["path"]), r["why"]))
        return "\n".join(L)
    nx = res["next"]
    if c is None:
        L.append("Only one trained policy was found, so there is nothing to compare.")
        return "\n".join(L)
    verdict = ("**%d job%s regressed.**" % (c["n_regressed"],
                                            "" if c["n_regressed"] == 1 else "s")
               if c["n_regressed"] else "**No job regressed beyond measurement noise.**")
    L.append("%s `%s` &rarr; `%s`, %d jobs."
             % (verdict, res["incumbent"], res["candidate"], c["n_skills"]))
    L.append("")
    L.append("| job | shipping | candidate | change | 95% interval | |")
    L.append("|---|--:|--:|--:|:--:|---|")
    for r in c["rows"]:
        if r["verdict"] == "held" and len(c["rows"]) > 12:
            continue
        L.append("| `%s` | %.0f%% | %.0f%% | %+.1f | [%+.0f, %+.0f] | %s |"
                 % (r["skill"], r["incumbent_pct"], r["candidate_pct"], r["delta_pp"],
                    r["ci95"][0], r["ci95"][1],
                    "" if r["verdict"] == "held" else r["verdict"]))
    L.append("")
    L.append("Smallest drop this battery can call real: about %.0f points."
             % c["median_resolvable_drop_pp"])
    todo = [r for r in nx["rows"] if r["diagnosis"] != "FINE"]
    if todo:
        L.append("")
        L.append("<details><summary>What to do next (%d job%s)</summary>"
                 % (len(todo), "" if len(todo) == 1 else "s"))
        L.append("")
        for r in todo:
            L.append("- **`%s`** &mdash; %s: %s"
                     % (r["skill"], r["diagnosis"].lower(), r["action"]))
        L.append("")
        L.append("</details>")
    w = res.get("worklist")
    if w and w["n_discordant"]:
        L.append(worklist_mod.markdown(w))
    L.append("")
    L.append("<sub>orbit-eval %s%s. The decision is stdlib arithmetic and reproduces "
             "with <code>orbit check</code>.</sub>"
             % (__import__("orbit_eval").__version__,
                (", record `%s`" % sig[:12]) if sig else ""))
    return "\n".join(L)


def write_report(res, path, sig=None, open_it=False):
    # Render BEFORE opening the file. Opening for write truncates it, so a render
    # that raises used to destroy the previous report and leave nothing behind.
    body = html_report(res, sig)
    with open(path, "w") as fh:
        fh.write(body)
    if open_it:
        try:
            webbrowser.open("file://" + os.path.abspath(path))
        except Exception:
            pass
    return path


def signed(res, argv):
    payload = {"check": res.get("check"), "next": res["next"],
               "source": res["source"], "incumbent": res["incumbent"],
               "candidate": res["candidate"], "pairing_reason": res["pairing_reason"],
               "thresholds": {"drop_pp": release_mod.DROP_PP,
                              "min_episodes": release_mod.MIN_EPISODES,
                              "target_pp": res["next"]["target_pp"]}}
    return release_mod.record(payload, argv)
