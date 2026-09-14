#!/usr/bin/env python3
"""orbit-serve: put a LeRobot policy into an OrbitEval Arena battle from your own machine.

The model never leaves your machine. This shim pulls observations from the arena runtime with a
long-poll, runs your policy on them, and sends the actions back, so it works behind NAT and needs no
open port. Both policies in a battle run on their owners' hardware; the arena runs the world and the
referee and streams the rollouts to the page.

    # join slot "a" of session 3f9a2c with the public Diffusion Policy for PushT
    python orbit_serve.py --join 3f9a2c/a --policy lerobot/diffusion_pusht --env pusht --name "DP DDIM-10" \\
        --override noise_scheduler_type=DDIM num_inference_steps=10

    # your own checkpoint (a LeRobot pretrained_model directory)
    python orbit_serve.py --join 3f9a2c/b --policy outputs/train/run7/checkpoints/last/pretrained_model --env pusht

Needs `pip install 'lerobot[pusht]'` (plus the policy's extra, e.g. `lerobot[diffusion]`). Normalisation
statistics come from the checkpoint's processors when present, else from buffers inside the checkpoint, else from the training dataset
(`--dataset`, default lerobot/pusht for PushT). Runtime defaults to https://arena.orbiteval.com; override with --runtime.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

DEFAULT_RUNTIME = os.environ.get("ORBIT_ARENA", "https://orbiteval-arena-199802234044.us-central1.run.app")
DATASET_FOR_ENV = {"pusht": "lerobot/pusht"}


def http(method, url, body=None, timeout=40):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_policy(path, env_key, device, overrides, dataset):
    import numpy as np  # noqa: F401
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import configs as envcfgs
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    env_cls = {c.__name__.lower().replace("env", ""): c for n, c in vars(envcfgs).items() if isinstance(c, type) and n.endswith("Env")}
    if env_key not in env_cls:
        sys.exit(f"unknown --env {env_key}; known: {sorted(env_cls)}")
    env_cfg = env_cls[env_key]()
    pcfg = PreTrainedConfig.from_pretrained(path)
    pcfg.pretrained_path = path
    pcfg.device = device
    for k, v in overrides.items():
        setattr(pcfg, k, v)
    policy = make_policy(pcfg, env_cfg=env_cfg)
    policy.eval()
    try:
        pre, post = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
        source = "checkpoint processors"
    except Exception:  # noqa: BLE001  (older checkpoints have no processor files)
        stats = stats_from_checkpoint(path, pcfg)
        if stats:
            source = "normalisation buffers inside the checkpoint"
        else:
            from lerobot.datasets import LeRobotDatasetMetadata

            ds = dataset or DATASET_FOR_ENV.get(env_key)
            if not ds:
                sys.exit("this checkpoint has no processor files; pass --dataset <repo used for training> for normalisation stats")
            stats = LeRobotDatasetMetadata(ds).stats
            source = f"dataset stats from {ds}"
        pre, post = make_pre_post_processors(policy_cfg=pcfg, dataset_stats=stats, preprocessor_overrides={"device_processor": {"device": device}})
    epre, epost = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
    return policy, pre, post, epre, epost, source, torch


def stats_from_checkpoint(path, pcfg):
    """Normalisation statistics from the buffers an older checkpoint keeps inside model.safetensors.

    Checkpoints trained before LeRobot moved normalisation into processors carry mean/std or min/max
    buffers per feature. Using them reproduces exactly what the policy saw in training; dataset
    statistics can differ (the public PushT Diffusion Policy uses ImageNet image statistics, and the
    dataset's own image mean is ~0.97 because of the white background, which silently breaks it).
    """
    from safetensors import safe_open

    f = os.path.join(path, "model.safetensors") if os.path.isdir(path) else None
    if f is None:
        from huggingface_hub import hf_hub_download

        try:
            f = hf_hub_download(path, "model.safetensors")
        except Exception:  # noqa: BLE001
            return {}
    if not os.path.exists(f):
        return {}
    feats = {k.replace(".", "_"): k for k in list(pcfg.input_features) + list(pcfg.output_features)}
    stats = {}
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if ".buffer_" not in k or not k.startswith(("normalize_inputs.", "normalize_targets.")):
                continue
            rest = k.split(".buffer_", 1)[1]
            fkey, stat = rest.rsplit(".", 1)
            feat = feats.get(fkey)
            if feat is None or stat not in ("mean", "std", "min", "max"):
                continue
            t = sf.get_tensor(k)
            if t.numel() and (t == float("inf")).any():
                continue  # unused buffer placeholder
            stats.setdefault(feat, {})[stat] = t
    return stats


def decode_obs(payload):
    import numpy as np
    from PIL import Image

    raw = {}
    if "pixels" in payload:
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload["pixels"]))).convert("RGB"), dtype=np.uint8)
        raw["pixels"] = img[None]
    for k in ("agent_pos", "environment_state", "state"):
        if k in payload:
            raw[k] = np.asarray(payload[k], dtype=np.float32)[None]
    return raw


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--join", required=True, help="<session>/<slot> from the battle page, e.g. 3f9a2c/a")
    ap.add_argument("--policy", required=True, help="hub id or local pretrained_model directory")
    ap.add_argument("--env", default="pusht", help="embodiment key (matches the session's)")
    ap.add_argument("--name", help="display name on the page")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dataset", help="training dataset repo, for normalisation stats when the checkpoint has no processors")
    ap.add_argument("--override", nargs="*", default=[], help="policy config overrides, key=value (JSON values)")
    ap.add_argument("--runtime", default=DEFAULT_RUNTIME)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    base = f"{args.runtime.rstrip('/')}/s/{args.join}"
    print(f"loading {args.policy} for {args.env} on {args.device} ...", flush=True)
    policy, pre, post, epre, epost, source, torch = load_policy(args.policy, args.env, args.device, parse_overrides(args.override), args.dataset)
    torch.set_num_threads(args.threads)
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    name = args.name or os.path.basename(args.policy.rstrip("/"))
    info = {"policy": args.policy, "type": policy.config.type, "device": args.device, "normalisation": source, "overrides": json.dumps(parse_overrides(args.override))}
    r = http("POST", f"{base}/join", {"name": name, "info": info})
    emb = r["embodiment"]
    print(f"joined as '{name}' | embodiment {emb['label']} | action dim {emb['action_dim']} | status: {r['status']}", flush=True)

    ep_seen = -1
    n_actions = 0
    t_start = time.time()
    while True:
        try:
            nxt = http("GET", f"{base}/next?wait=20", timeout=45)
        except urllib.error.HTTPError as e:
            print(f"runtime error {e.code}: {e.read().decode()[:200]}", flush=True)
            time.sleep(2)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"connection problem: {e}; retrying", flush=True)
            time.sleep(2)
            continue
        req = nxt.get("request")
        if req is None:
            st = nxt.get("status", "")
            if st in ("finished", "error"):
                print(f"session {st}{(': ' + str(nxt.get('error'))) if nxt.get('error') else ''}. {n_actions} actions served in {time.time() - t_start:.0f}s.", flush=True)
                return 0
            continue
        if req["episode"] != ep_seen or req.get("reset"):
            policy.reset()
            ep_seen = req["episode"]
            print(f"episode {ep_seen + 1} (seed {req['seed']})", flush=True)
        raw = decode_obs(req["obs"])
        obs = preprocess_observation(raw)
        obs["task"] = [str(req.get("task") or "")]
        obs = epre(obs)
        obs = pre(obs)
        with torch.no_grad():
            act = policy.select_action(obs)
        act = post(act)
        act = epost({ACTION: act})[ACTION]
        act_np = act.detach().cpu().numpy() if hasattr(act, "detach") else act
        actions = [[float(x) for x in act_np.reshape(-1)[: emb["action_dim"]]]]
        try:
            http("POST", f"{base}/action", {"request_id": req["request_id"], "actions": actions})
            n_actions += 1
        except urllib.error.HTTPError as e:
            print(f"action rejected {e.code}: {e.read().decode()[:160]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
