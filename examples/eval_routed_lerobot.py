#!/usr/bin/env python3
"""Evaluate an orbit-eval release as ONE policy with LeRobot's own rollout loop.

    python examples/eval_routed_lerobot.py --release release.json \
        --policy-path o_mask_pc88_s101=/weights/s101/pretrained_model ... \
        --env libero --task libero_object --task-ids 3,4,6 --n-episodes 10 --seed 4242 \
        --rename-map '{"observation.images.image":"observation.images.camera1",...}' \
        --out routed_eval/ [--shim /path/eval_shim.py] [--banked /path/results]

For every task in the release the RoutedPolicy dispatches to the candidate the release
chose; each candidate carries its own pre/post processor pipelines, so `eval_policy_all`
runs unchanged with identity pipelines. Writes one eval_info.json per task under --out
(same schema as lerobot-eval), plus routed_summary.json. With --banked, compares each
task's per-episode outcomes with the chosen candidate's own banked eval on the same
initial states (needs the same --shim, seed and n).
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--policy-path", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--env", default="libero")
    ap.add_argument("--task", default="libero_object")
    ap.add_argument("--task-ids", default=None, help="comma list; default = every task in the release")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rename-map", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shim", default=None, help="eval_shim.py to import first (positional CRN init states)")
    ap.add_argument("--banked", default=None, help="results dir <cand>/t<id>/eval_info.json to compare against")
    a = ap.parse_args()

    if a.shim:
        sys.path.insert(0, os.path.dirname(os.path.abspath(a.shim)))
        os.environ.setdefault("SEGMENT_SHIM_LOG", os.path.join(a.out, "shim_log.json"))
        import importlib
        importlib.import_module(os.path.splitext(os.path.basename(a.shim))[0])   # installs the shim

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from orbit_eval.routed_policy import LeRobotCandidate, RoutedPolicy, as_lerobot_policy, identity_pipelines

    rename_map = json.loads(a.rename_map)
    paths = {}
    for item in a.policy_path:
        k, v = item.split("=", 1)
        paths[k.strip()] = v.strip()
    rec = json.load(open(a.release))
    rel = rec.get("release", rec)
    table = {r["task"]: r["chosen"] for r in rel["rows"]}
    if a.task_ids:
        want = [int(x) for x in a.task_ids.split(",")]
    else:
        want = sorted(int(t.split("/")[-1]) for t in table)
    keys = {t: "%s/%d" % (a.task, t) for t in want}
    for t, k in keys.items():
        if k not in table:
            sys.exit("task %s not in release table" % k)

    env_cfg0 = make_env_config(a.env, task=a.task, task_ids=[want[0]])
    loaded_cfg = {}

    def loader(path):
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = a.device
        policy = make_policy(cfg=cfg, env_cfg=env_cfg0, rename_map=rename_map)
        policy.eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                    "rename_observations_processor": {"rename_map": rename_map}})
        loaded_cfg.setdefault("cfg", cfg)
        return LeRobotCandidate(policy, pre, post)

    rp = RoutedPolicy.from_release(a.release, loader=loader, policy_paths=paths)
    lp = as_lerobot_policy(rp)                                    # PreTrainedPolicy view for lerobot-eval
    print("release:", lp, "| tasks", [keys[t] for t in want], flush=True)
    os.makedirs(a.out, exist_ok=True)
    ident_pre, ident_post = identity_pipelines()
    summary = {"tasks": {}, "seed": a.seed, "n_episodes": a.n_episodes}
    t_all = time.time()
    for t in want:
        key = keys[t]
        rp.set_task(key)
        cand = rp.candidate_for(key)
        rp.policy_for(key)                                        # load now (so env processors see a cfg)
        env_cfg = make_env_config(a.env, task=a.task, task_ids=[t])
        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=loaded_cfg["cfg"])
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        t0 = time.time()
        with torch.no_grad():
            info = eval_policy_all(envs=envs, policy=lp, env_preprocessor=env_pre, env_postprocessor=env_post,
                                   preprocessor=ident_pre, postprocessor=ident_post, n_episodes=a.n_episodes,
                                   start_seed=a.seed, max_parallel_tasks=1)
        for g in envs.values():
            for e in g.values():
                e.close()
        json.dump(info, open(os.path.join(a.out, "t%d_eval_info.json" % t), "w"))
        succ = info["per_task"][0]["metrics"]["successes"]
        row = {"candidate": cand, "sr": 100.0 * sum(succ) / len(succ), "n": len(succ),
               "seconds": time.time() - t0, "loaded": rp.loaded()}
        if a.banked:
            bp = os.path.join(a.banked, cand, "t%d" % t, "eval_info.json")
            if os.path.exists(bp):
                b = json.load(open(bp))["per_task"][0]["metrics"]["successes"][: len(succ)]
                row["banked_sr_same_episodes"] = 100.0 * sum(b) / len(b)
                row["episode_agreement"] = sum(int(x) == int(y) for x, y in zip(succ, b)) / len(b)
        summary["tasks"][key] = row
        print(key, "->", cand, "SR %.1f" % row["sr"], "(%.0fs)" % row["seconds"],
              ("agree %.2f vs banked %.1f" % (row["episode_agreement"], row["banked_sr_same_episodes"])
               if "episode_agreement" in row else ""), flush=True)
    summary["seconds_total"] = time.time() - t_all
    summary["loaded_candidates"] = rp.loaded()
    json.dump(summary, open(os.path.join(a.out, "routed_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
