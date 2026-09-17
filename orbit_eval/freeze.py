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

"""Pin a checkpoint so every later comparison is the cheap one.

Training variance is the large term: 20 points of standard deviation on a
single LIBERO task from the seed alone. Freezing the checkpoint removes it,
and what is left is a fixed checkpoint compared under common random numbers,
which is powered at about two seeds instead of about sixty retrains.

A frozen checkpoint is still not deterministic. Diffusion and flow-matching
policies sample their actions, so the inference seed is pinned too, and the
evaluation seed sequence is written down so that episode k means the same
starting state in every run that uses this manifest. Three sources of noise,
one file.

The manifest is also the skill listing. It carries the dataset's joint names
and recorded ranges, which is what lets `orbit body --skill` say whether
another arm's calibration overlaps the one this checkpoint was trained on,
before anybody runs it.

Nothing in the manifest says the checkpoint works. It says what it is.
"""

import hashlib
import json
import os
import time

from . import __version__, dataset

MANIFEST_VERSION = 1
DEFAULT_INFERENCE_SEED = 1000
DEFAULT_EVAL_START = 1000
DEFAULT_EVAL_COUNT = 100
CHUNK = 1 << 20

INFERENCE_NOTE = ("action sampling is stochastic for diffusion and flow-matching "
                  "policies, so the same checkpoint on the same episode can act "
                  "differently; this seed pins it")
EVAL_NOTE = ("episode k uses seed start+k; any two runs that follow this manifest "
             "are paired by construction (common random numbers)")


# ------------------------------------------------------------------ hashing

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _files_of(checkpoint):
    """(base directory, [relative file names]) for a directory or a single file."""
    checkpoint = os.path.abspath(checkpoint)
    if os.path.isfile(checkpoint):
        return os.path.dirname(checkpoint), [os.path.basename(checkpoint)]
    if not os.path.isdir(checkpoint):
        raise ValueError("no such checkpoint: %s" % checkpoint)
    names = sorted(n for n in os.listdir(checkpoint)
                   if os.path.isfile(os.path.join(checkpoint, n)) and not n.startswith("."))
    if not names:
        raise ValueError("no files directly inside %s" % checkpoint)
    return checkpoint, names


def _hash_files(base, names):
    out = {}
    for n in names:
        p = os.path.join(base, n)
        out[n] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    return out


def _combined(files):
    h = hashlib.sha256()
    for n in sorted(files):
        h.update(("%s:%s\n" % (n, files[n]["sha256"])).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ building

def build(checkpoint, dataset_path=None, inference_seed=DEFAULT_INFERENCE_SEED,
          eval_start=DEFAULT_EVAL_START, eval_count=DEFAULT_EVAL_COUNT, record=None):
    base, names = _files_of(checkpoint)
    files = _hash_files(base, names)
    m = {
        "orbit_freeze": MANIFEST_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "orbit_eval": __version__,
        "checkpoint": {
            "path": base if os.path.isdir(os.path.abspath(checkpoint)) else
                    os.path.abspath(checkpoint),
            "files": files,
            "sha256": _combined(files),
        },
        "policy": _policy_type(base),
        "inference": {"seed": int(inference_seed), "note": INFERENCE_NOTE},
        "eval": {"seeds": {"start": int(eval_start), "count": int(eval_count)},
                 "note": EVAL_NOTE},
        "dataset": None,
        "record": None,
    }
    if dataset_path:
        m["dataset"] = _dataset_block(dataset_path)
    if record:
        m["record"] = {"path": os.path.abspath(record), "sha256": sha256_file(record)}
    return m


def _policy_type(base):
    for name in ("config.json", "train_config.json"):
        p = os.path.join(base, name)
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            t = d.get("type")
            if isinstance(t, str):
                return t
            pol = d.get("policy")
            if isinstance(pol, dict) and isinstance(pol.get("type"), str):
                return pol["type"]
    return None


def _dataset_block(path):
    meta = dataset.load(path)
    files = {}
    for rel in ("meta/info.json", "meta/stats.json", "meta/episodes_stats.jsonl",
                "meta/tasks.jsonl", "meta/episodes.jsonl"):
        p = os.path.join(meta.root, rel)
        if os.path.isfile(p):
            files[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    ranges = {}
    for key in (dataset.STATE, dataset.ACTION):
        block = meta.block(key)
        if block:
            lo, hi = dataset._vec(block.get("min")), dataset._vec(block.get("max"))
            if lo is not None and hi is not None:
                ranges[key] = {"min": list(lo), "max": list(hi)}
    return {
        "path": meta.root,
        "repo_id": meta.info.get("repo_id"),
        "version": meta.version,
        "robot_type": meta.robot_type,
        "n_episodes": meta.n_episodes,
        "fps": meta.fps,
        "joints": meta.joint_names,
        "ranges": ranges,
        "files": files,
    }


def eval_seeds(manifest):
    s = manifest["eval"]["seeds"]
    return list(range(int(s["start"]), int(s["start"]) + int(s["count"])))


# ------------------------------------------------------------------ files

def write(manifest, path):
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load(path):
    with open(path) as fh:
        m = json.load(fh)
    if not isinstance(m, dict) or "orbit_freeze" not in m:
        raise ValueError("%s is not an orbit freeze manifest" % path)
    return m


# ------------------------------------------------------------------ verify

def verify(manifest, base=None):
    """Re-hash every file the manifest names and say which ones moved."""
    ck = manifest["checkpoint"]
    root = ck["path"]
    if os.path.isfile(root):
        root = os.path.dirname(root)
    if base and not os.path.isdir(root):
        root = os.path.join(base, os.path.basename(root))
    changed, missing, ok = [], [], []
    for name, want in sorted(ck["files"].items()):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            missing.append(name)
            continue
        got = sha256_file(p)
        (ok if got == want["sha256"] else changed).append(name)
    return {"ok": not changed and not missing, "checked": len(ck["files"]),
            "unchanged": ok, "changed": changed, "missing": missing, "root": root}


# ------------------------------------------------------------------ printing

def format_freeze(m, out_path):
    L = [""]
    L.append("  FROZEN  %s" % out_path)
    ck = m["checkpoint"]
    L.append("    %-12s %s" % ("checkpoint", ck["path"]))
    for name, f in sorted(ck["files"].items()):
        L.append("    %-12s %-22s sha256 %s  (%s)"
                 % ("", name[:22], f["sha256"][:12], _size(f["bytes"])))
    L.append("    %-12s %s" % ("all files", ck["sha256"][:16] + "..."))
    if m.get("policy"):
        L.append("    %-12s %s" % ("policy", m["policy"]))
    L.append("    %-12s seed %d" % ("inference", m["inference"]["seed"]))
    for line in _wrap(m["inference"]["note"], 58):
        L.append("    %-12s %s" % ("", line))
    s = m["eval"]["seeds"]
    L.append("    %-12s %d to %d" % ("eval seeds", s["start"], s["start"] + s["count"] - 1))
    for line in _wrap(m["eval"]["note"], 58):
        L.append("    %-12s %s" % ("", line))
    d = m.get("dataset")
    if d:
        bits = []
        if d.get("n_episodes"):
            bits.append("%d episodes" % d["n_episodes"])
        if d.get("robot_type"):
            bits.append(str(d["robot_type"]))
        if d.get("joints"):
            bits.append("%d joints" % len(d["joints"]))
        L.append("    %-12s %s; recorded ranges ride along for the skill listing"
                 % ("dataset", ", ".join(bits)))
    if m.get("record"):
        L.append("    %-12s %s" % ("record", m["record"]["path"]))
    L.append("")
    for line in _wrap("Every later comparison against this checkpoint is the cheap "
                      "question: a fixed checkpoint under common random numbers, "
                      "powered at about two seeds. Re-check the files any time with "
                      "`orbit freeze --verify %s`. The manifest says what the "
                      "checkpoint is; it does not say that it works." % out_path, 74):
        L.append("  " + line)
    L.append("")
    return "\n".join(line.rstrip() for line in L)


def format_verify(v, path):
    L = [""]
    if v["ok"]:
        L.append("  VERIFIED  %s: all %d file%s unchanged" % (
            path, v["checked"], "" if v["checked"] == 1 else "s"))
    else:
        L.append("  CHANGED   %s" % path)
        for n in v["changed"]:
            L.append("    %-22s content differs from the manifest" % n)
        for n in v["missing"]:
            L.append("    %-22s missing under %s" % (n, v["root"]))
        L.append("")
        L.append("  A comparison against this manifest is no longer a comparison against")
        L.append("  the frozen checkpoint. Freeze again, or restore the files.")
    L.append("")
    return "\n".join(L)


def _size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return "%d B" % n


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w) if line else w
    if line:
        out.append(line)
    return out or [""]
