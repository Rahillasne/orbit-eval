#!/usr/bin/env python3
"""orbit-battle: two robot policies, same tasks, same seeds, a referee that prices the noise.

Referee only, from evaluations you already ran (LeRobot eval_info.json, one per policy):

    python orbit_battle.py --from-eval-info runs/A/eval_info.json runs/B/eval_info.json --names A B

Run and referee on your own GPU (LeRobot installed; both policies evaluated with the SAME seed, so the
initial states are shared and the comparison is paired):

    python orbit_battle.py --run --a outputs/train/A/checkpoints/last/pretrained_model \\
        --b outputs/train/B/checkpoints/last/pretrained_model --env libero --episodes 50 --seed 1000 \\
        --extra --env.task=libero_object --policy.device=cuda

    add --batch 10 to run in rounds and stop each task as soon as the referee has resolved it.

Per task the referee reports both success rates with 95% Wilson intervals, the difference with a 95%
Newcombe interval, and a call: A AHEAD, B AHEAD, or UNRESOLVED at these counts (with the episodes that
would settle it). The scoreboard is printed and written as JSON and Markdown; the two eval_info.json files
can be dropped into the chat at orbiteval.com/chat.html for the same scoreboard with an explanation.

Only the standard library is needed for the referee. `--run` shells out to `lerobot-eval`.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from statistics import NormalDist

Z95 = NormalDist().inv_cdf(0.975)


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=Z95):
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k_a, n_a, k_b, n_b):
    p_a, p_b = k_a / n_a, k_b / n_b
    l_a, u_a = wilson(k_a, n_a)
    l_b, u_b = wilson(k_b, n_b)
    d = p_a - p_b
    return (d - math.sqrt((p_a - l_a) ** 2 + (u_b - p_b) ** 2), d + math.sqrt((u_a - p_a) ** 2 + (p_b - l_b) ** 2))


def episodes_to_resolve(p_a, p_b, max_n=5000):
    """Smallest equal n at which the observed rates would give an interval that excludes zero."""
    if abs(p_a - p_b) < 1e-9:
        return None
    for n in list(range(5, 200)) + list(range(200, max_n + 1, 10)):
        lo, hi = newcombe(round(p_a * n), n, round(p_b * n), n)
        if lo > 0 or hi < 0:
            return n
    return None


# ----------------------------------------------------------------------------- referee
def load_eval_info(path):
    with open(path) as f:
        info = json.load(f)
    tasks = {}
    if "per_task" in info:
        for e in info["per_task"]:
            s = (e.get("metrics") or {}).get("successes") or e.get("successes")
            if s is None:
                raise ValueError(f"{path}: a per_task entry has no successes")
            tasks[f"{e.get('task_group')}/{e.get('task_id')}"] = [bool(x) for x in s]
    elif "per_episode" in info:
        tasks["all"] = [bool(ep.get("success")) for ep in info["per_episode"]]
    else:
        raise ValueError(f"{path}: not an eval_info.json")
    return tasks


def referee(a_tasks, b_tasks, name_a="A", name_b="B", min_episodes=10):
    shared = sorted(set(a_tasks) & set(b_tasks))
    rows = []
    for t in shared:
        a, b = a_tasks[t], b_tasks[t]
        ka, na, kb, nb = sum(a), len(a), sum(b), len(b)
        pa, pb = ka / na, kb / nb
        lo, hi = newcombe(ka, na, kb, nb)
        if na < min_episodes or nb < min_episodes:
            call, need = "UNDERPOWERED", None
        elif lo > 0:
            call, need = f"{name_a} AHEAD", None
        elif hi < 0:
            call, need = f"{name_b} AHEAD", None
        else:
            call, need = "UNRESOLVED", episodes_to_resolve(pa, pb)
        rows.append({"task": t, "a": {"k": ka, "n": na, "rate_pct": round(100 * pa, 1), "ci95": [round(100 * x, 1) for x in wilson(ka, na)]},
                     "b": {"k": kb, "n": nb, "rate_pct": round(100 * pb, 1), "ci95": [round(100 * x, 1) for x in wilson(kb, nb)]},
                     "delta_pp": round(100 * (pa - pb), 1), "ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)], "call": call,
                     "episodes_per_side_to_resolve": need})
    ka, na = sum(sum(a_tasks[t]) for t in shared), sum(len(a_tasks[t]) for t in shared)
    kb, nb = sum(sum(b_tasks[t]) for t in shared), sum(len(b_tasks[t]) for t in shared)
    lo, hi = newcombe(ka, na, kb, nb) if na and nb else (math.nan, math.nan)
    overall = {"a_pct": round(100 * ka / na, 1) if na else None, "b_pct": round(100 * kb / nb, 1) if nb else None,
               "delta_pp": round(100 * (ka / na - kb / nb), 1) if na and nb else None, "ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
               "call": (f"{name_a} AHEAD" if lo > 0 else f"{name_b} AHEAD" if hi < 0 else "UNRESOLVED") if na and nb else "n/a"}
    counts = {c: sum(1 for r in rows if r["call"] == c) for c in (f"{name_a} AHEAD", f"{name_b} AHEAD", "UNRESOLVED", "UNDERPOWERED")}
    return {"names": [name_a, name_b], "tasks": rows, "overall": overall, "counts": counts,
            "only_in_a": sorted(set(a_tasks) - set(b_tasks)), "only_in_b": sorted(set(b_tasks) - set(a_tasks)),
            "rule": "AHEAD only when the whole 95% Newcombe interval of the difference is on one side of zero. Sharing the seed gives both policies the same initial states; the interval is the independent-samples one, so it is conservative in that case."}


def format_scoreboard(sb):
    a, b = sb["names"]
    lines = [f"orbit-battle  {a}  vs  {b}", "", f"{'task':<26} {a:>16} {b:>16} {'diff (pp)':>10} {'95% interval':>18}  call"]
    for r in sb["tasks"]:
        ra = f"{r['a']['k']}/{r['a']['n']} {r['a']['rate_pct']:.0f}%"
        rb = f"{r['b']['k']}/{r['b']['n']} {r['b']['rate_pct']:.0f}%"
        ci = f"[{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}]"
        extra = f"  (~{r['episodes_per_side_to_resolve']} eps/side to settle)" if r["call"] == "UNRESOLVED" and r["episodes_per_side_to_resolve"] else ""
        lines.append(f"{r['task']:<26} {ra:>16} {rb:>16} {r['delta_pp']:>+10.1f} {ci:>18}  {r['call']}{extra}")
    o = sb["overall"]
    lines += ["", f"overall (pooled): {a} {o['a_pct']}%  {b} {o['b_pct']}%  diff {o['delta_pp']:+.1f} pp  95% [{o['ci95_pp'][0]:+.1f}, {o['ci95_pp'][1]:+.1f}]  {o['call']}",
              "  ".join(f"{k}: {v}" for k, v in sb["counts"].items()), sb["rule"]]
    if sb["only_in_a"] or sb["only_in_b"]:
        lines.append(f"not compared: only in {a}: {sb['only_in_a']}; only in {b}: {sb['only_in_b']}")
    return "\n".join(lines)


def format_markdown(sb):
    a, b = sb["names"]
    out = [f"| task | {a} | {b} | diff (pp) | 95% interval | call |", "|---|---|---|---:|---|---|"]
    for r in sb["tasks"]:
        out.append(f"| {r['task']} | {r['a']['k']}/{r['a']['n']} ({r['a']['rate_pct']}%) | {r['b']['k']}/{r['b']['n']} ({r['b']['rate_pct']}%) | {r['delta_pp']:+.1f} | [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}] | {r['call']} |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- runner
def run_lerobot_eval(policy_path, env, out_dir, seed, n_episodes, extra, dry_run=False):
    exe = shutil.which("lerobot-eval")
    cmd = [exe or "lerobot-eval", f"--policy.path={policy_path}", f"--env.type={env}", f"--seed={seed}",
           f"--eval.n_episodes={n_episodes}", f"--output_dir={out_dir}"] + list(extra or [])
    print("  $", " ".join(cmd), flush=True)
    if dry_run:
        return None
    if not exe:
        sys.exit("lerobot-eval not found on PATH; install LeRobot with the environment extra first")
    subprocess.run(cmd, check=True)
    return os.path.join(out_dir, "eval_info.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-eval-info", nargs=2, metavar=("A_JSON", "B_JSON"), help="referee two existing eval_info.json files")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--run", action="store_true", help="evaluate both policies with lerobot-eval, then referee")
    ap.add_argument("--a", help="policy A path or hub id (with --run)")
    ap.add_argument("--b", help="policy B path or hub id (with --run)")
    ap.add_argument("--env", default="libero")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--batch", type=int, default=0, help="run in rounds of this many episodes per task and stop when resolved (0 = one run)")
    ap.add_argument("--seed", type=int, default=1000, help="the same seed for both policies = shared initial states = paired comparison")
    ap.add_argument("--out", default="battle")
    ap.add_argument("--dry-run", action="store_true", help="print the lerobot-eval commands without running them")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra arguments passed to lerobot-eval (e.g. --env.task=libero_object)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.from_eval_info:
        a_tasks, b_tasks = load_eval_info(args.from_eval_info[0]), load_eval_info(args.from_eval_info[1])
    elif args.run:
        if not (args.a and args.b):
            sys.exit("--run needs --a and --b")
        if args.batch and args.batch < args.episodes:
            print(f"[rounds of {args.batch} episodes per task; stopping early when every task is resolved]", flush=True)
        n_done, a_tasks, b_tasks = 0, {}, {}
        step = args.batch or args.episodes
        while n_done < args.episodes:
            n = min(step, args.episodes - n_done)
            rnd = f"round{n_done // step + 1}"
            pa = run_lerobot_eval(args.a, args.env, os.path.join(args.out, rnd, args.names[0]), args.seed + n_done, n, args.extra, args.dry_run)
            pb = run_lerobot_eval(args.b, args.env, os.path.join(args.out, rnd, args.names[1]), args.seed + n_done, n, args.extra, args.dry_run)
            n_done += n
            if args.dry_run:
                continue
            for tasks, path in ((a_tasks, pa), (b_tasks, pb)):
                for t, s in load_eval_info(path).items():
                    tasks.setdefault(t, []).extend(s)
            sb = referee(a_tasks, b_tasks, *args.names)
            print(format_scoreboard(sb), flush=True)
            if args.batch and sb["counts"]["UNRESOLVED"] == 0 and sb["counts"]["UNDERPOWERED"] == 0:
                print("[every task resolved; stopping early]", flush=True)
                break
        if args.dry_run:
            return 0
    else:
        ap.error("use --from-eval-info A B, or --run --a A --b B")

    sb = referee(a_tasks, b_tasks, *args.names)
    sb["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "scoreboard.json"), "w") as f:
        json.dump(sb, f, indent=1)
    with open(os.path.join(args.out, "scoreboard.md"), "w") as f:
        f.write(format_markdown(sb) + "\n")
    print(format_scoreboard(sb))
    print(f"\nwritten: {args.out}/scoreboard.json, {args.out}/scoreboard.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
