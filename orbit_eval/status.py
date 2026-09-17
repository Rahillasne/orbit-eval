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

"""Where this robot is, and the one thing to do next.

The question people ask on the LeRobot issue tracker is not whether a
difference is significant. It is "why doesn't my robot work" and "how many
episodes do I need". This command answers those, and in answering them arrives
at the other question anyway, because the trial count is the same arithmetic.

There are four places to be: nothing recorded yet, a recording, a checkpoint,
and a result. Each has exactly one next action, and a defect in the recording
outranks every one of them: a servo that never moved cannot be fixed by
training longer, and a success rate measured on that data is not about the
policy.

What this module refuses to do is predict. It reports what was recorded, what
is measurably wrong with it, and what the next step costs in trials. It never
says a policy will work. Version 0.7.2 of this project's predecessor was
retired for making exactly that claim, and the retraction is in its README.
"""

import json
import math
import os

from . import dataset, discover, nextstep
from . import release as release_mod


# The drop a team usually says it wants to be able to see.
TARGET_PP = 10.0

# What LeRobot's own AGENT_GUIDE.md puts in front of every user and every
# coding agent: `--dataset.num_episodes=10` for a real-robot evaluation.
GUIDE_TRIALS = 10

POLICY_WEIGHTS = ("model.safetensors", "pytorch_model.bin", "model.bin",
                  "diffusion_pytorch_model.safetensors")
POLICY_MARKERS = ("train_config.json",)
SKIP_DIRS = {".git", ".hg", "node_modules", "__pycache__", ".venv", "venv",
             ".tox", ".mypy_cache", ".pytest_cache", "meta", "data", "videos"}
# lerobot-train writes to outputs/train/<job>/checkpoints/<step>/pretrained_model,
# which is six levels below where a user stands when they run this.
MAX_DEPTH = 9

STAGES = ("no_data", "data", "trained", "evaluated")


class Status(object):
    """One robot's position in the loop, and the next move."""

    def __init__(self, root):
        self.root = root
        self.stage = "no_data"
        self.dataset = None
        self.findings = []
        self.policies = []
        self.evals = []
        self.observed_rate = None
        self.observed_n = None
        self.trials_needed = None
        self.smallest_callable = None
        self.next_action = ""
        self.notes = []

    @property
    def blocking(self):
        return [f for f in self.findings if f.severity == "blocking"]

    @property
    def advisories(self):
        return [f for f in self.findings if f.severity != "blocking"]

    def as_dict(self):
        return {
            "root": self.root,
            "stage": self.stage,
            "dataset": None if self.dataset is None else {
                "path": self.dataset.root,
                "robot_type": self.dataset.robot_type,
                "n_episodes": self.dataset.n_episodes,
                "n_frames": self.dataset.n_frames,
                "fps": self.dataset.fps,
                "seconds": self.dataset.seconds,
                "joints": self.dataset.joint_names,
                "version": self.dataset.version,
            },
            "findings": [f.as_dict() for f in self.findings],
            "policies": list(self.policies),
            "evals": list(self.evals),
            "observed_rate": self.observed_rate,
            "observed_n": self.observed_n,
            "trials_needed": self.trials_needed,
            "smallest_callable": self.smallest_callable,
            "next_action": self.next_action,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------- the maths

def smallest_callable_pp(n, pct=50.0):
    """The smallest drop a battery of `n` trials per arm could call real.

    The same rule `release` prints and `orbit next` inverts, so the three
    commands never disagree with each other in front of a user.
    """
    if not n or n <= 0:
        return None
    p = min(max(pct / 100.0, 0.0), 1.0)
    var = max(p * (1.0 - p), 0.01)
    se = 100.0 * math.sqrt(2.0 * var / float(n))
    return max(release_mod.DROP_PP, release_mod.Z_DMG * se)


def trials_for(pct, target_pp=TARGET_PP):
    """Trials per arm before a drop of `target_pp` becomes callable."""
    return nextstep.episodes_for(pct, target_pp)


# ------------------------------------------------------------------ scanning

def scan(path=".", target_pp=TARGET_PP):
    root = os.path.abspath(path)
    st = Status(root)

    ds_root = dataset.find(root)
    if ds_root:
        try:
            st.dataset = dataset.load(ds_root)
            st.findings = dataset.checks(st.dataset)
        except ValueError:
            st.dataset = None

    st.policies = find_policies(root, skip=ds_root)
    st.evals, st.observed_rate, st.observed_n = _eval_summary(root)

    if st.evals:
        st.stage = "evaluated"
    elif st.policies:
        st.stage = "trained"
    elif st.dataset is not None:
        st.stage = "data"
    else:
        st.stage = "no_data"

    rate = st.observed_rate if st.observed_rate is not None else 50.0
    st.trials_needed = trials_for(rate, target_pp)
    if st.observed_n:
        st.smallest_callable = smallest_callable_pp(st.observed_n, rate)

    st.next_action = _next_action(st, target_pp)
    return st


def find_policies(root, skip=None, max_depth=MAX_DEPTH):
    """Directories that hold trained weights or a LeRobot training config."""
    out = []
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        if dirpath.rstrip(os.sep).count(os.sep) - base >= max_depth:
            dirnames[:] = []
        if (skip and skip != root
                and (dirpath == skip or dirpath.startswith(skip + os.sep))):
            dirnames[:] = []
            continue
        names = set(files)
        if names & set(POLICY_WEIGHTS) or names & set(POLICY_MARKERS):
            out.append(dirpath)
            dirnames[:] = []
    return out


def read_eval_counts(path):
    """(trials, successes) from one eval file, or None if it is not one.

    A single-task `lerobot-eval` writes `per_episode` and `aggregated` and no
    `per_task`, and that file is the commonest evaluation artifact in public.
    `discover._read_lerobot` reads the same shapes for `orbit check`; this is
    the pooled count `status` needs, kept separate because it also accepts
    the looser `overall` and `per_episode` spellings other tools write.
    """
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None

    tasks = d.get("per_task")
    if isinstance(tasks, list) and tasks:
        n = k = 0
        for pt in tasks:
            succ = ((pt or {}).get("metrics") or {}).get("successes")
            if isinstance(succ, list):
                n += len(succ)
                k += sum(1 for x in succ if x)
        if n:
            return n, k

    for block in (d.get("aggregated"), d.get("overall"), d):
        if not isinstance(block, dict):
            continue
        n = block.get("n_episodes")
        if not isinstance(n, (int, float)) or n <= 0:
            continue
        n = int(n)
        k = block.get("n_success")
        if isinstance(k, (int, float)):
            return n, int(k)
        pct = block.get("pc_success")
        if isinstance(pct, (int, float)):
            return n, int(round(n * float(pct) / 100.0))

    eps = d.get("per_episode")
    if isinstance(eps, list) and eps:
        flags = [e.get("success") for e in eps if isinstance(e, dict)]
        flags = [f for f in flags if f is not None]
        if flags:
            return len(flags), sum(1 for f in flags if f)
    return None


def _eval_summary(root, max_depth=MAX_DEPTH):
    """Any evaluation already sitting under `root`, pooled to one rate."""
    paths, n_tot, k_tot = [], 0, 0
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        if dirpath.rstrip(os.sep).count(os.sep) - base >= max_depth:
            dirnames[:] = []
        for name in sorted(files):
            if not name.lower().endswith(".json"):
                continue
            if name in ("train_config.json", "config.json", "info.json",
                        "stats.json"):
                continue
            got = read_eval_counts(os.path.join(dirpath, name))
            if got:
                paths.append(os.path.join(dirpath, name))
                n_tot += got[0]
                k_tot += got[1]

    if not n_tot:
        try:
            sources, _rejected = discover.scan(root, with_rejects=True)
        except Exception:
            sources = []
        for src in sources:
            data = getattr(src, "data", None) or {}
            for _cand, skills in data.items():
                if not isinstance(skills, dict):
                    continue
                for _skill, bits in skills.items():
                    if isinstance(bits, (list, tuple)):
                        n_tot += len(bits)
                        k_tot += sum(1 for b in bits if b)
                break
            if n_tot:
                paths.append(getattr(src, "path", "") or "")
                break

    if not n_tot:
        return [], None, None
    return paths, 100.0 * k_tot / n_tot, n_tot


# ------------------------------------------------------------- the one move

def _next_action(st, target_pp):
    block = st.blocking
    if block:
        f = block[0]
        what = f.joint or "the recording"
        more = ("  %d other%s like it." % (len(block) - 1,
                                           "" if len(block) == 2 else "s")
                if len(block) > 1 else "")
        return ("fix %s first: %s. Re-record those episodes before you read "
                "anything into a success rate.%s" % (what, f.short, more))

    if st.stage == "no_data":
        return ("record demonstrations. `lerobot-record` writes the dataset "
                "this command reads; LeRobot's guide starts a single task at "
                "%d episodes." % dataset.FEW_EPISODES)

    if st.stage == "data":
        n = st.dataset.n_episodes if st.dataset else None
        short = [f for f in st.advisories if f.code == "few_episodes"]
        if short:
            return ("record %d more episodes, then train. %s"
                    % (dataset.FEW_EPISODES - (n or 0), short[0].fix))
        return ("nothing is measurably wrong with the recording, so train on "
                "it. The next honest number comes from trials, not from the "
                "training loss.")

    if st.stage == "trained":
        need = st.trials_needed
        floor = smallest_callable_pp(GUIDE_TRIALS, st.observed_rate or 50.0)
        return ("run the policy and log the trials. To tell a %.0f point change "
                "from noise you need about %s trials; at the %d the LeRobot "
                "guide suggests, the smallest drop you could call is about "
                "%.0f points." % (target_pp, need if need else "many",
                                  GUIDE_TRIALS, floor))

    call = st.smallest_callable
    if call is not None and call > target_pp:
        return ("add trials before you conclude anything. %d trials resolve "
                "about %.0f points; %s would resolve %.0f."
                % (st.observed_n, call,
                   st.trials_needed if st.trials_needed else "more",
                   target_pp))
    return ("compare the runs properly: `orbit check` in this directory reads "
            "what is already here and answers per job, with the interval.")


# ------------------------------------------------------------------ printing

# The other commands, each with its one-line answer, so nobody has to learn a
# menu. Only commands that exist are listed.
ALSO = (
    ("orbit cover", "which instruction and workspace combinations were never recorded"),
    ("orbit body", "which robot can physically do this, and what is measured on it"),
    ("orbit freeze", "pin a checkpoint and its seeds so later comparisons are cheap"),
    ("orbit check", "did the new policy break any job, with the interval"),
    ("orbit next", "retrain, collect demonstrations, or run more trials, per job"),
    ("orbit log", "one keypress per trial, standing at the robot"),
)


def format_status(st, budget=False, also=True):
    L = []
    L.append("")
    L.append("  WHERE YOU ARE")
    for line in _where(st):
        L.append("    " + line)

    if st.findings:
        L.append("")
        L.append("  WHAT'S WRONG")
        for f in st.blocking:
            L.append("    %s" % f.message)
            if f.detail:
                L.append("      %s" % f.detail)
        for f in st.advisories:
            L.append("    %s" % f.message)

    L.append("")
    L.append("  NEXT ACTION")
    for line in _wrap(st.next_action, 72):
        L.append("    " + line)

    rows = _budget(st, force=budget)
    if rows:
        L.append("")
        L.append("  WHAT A BATTERY BUYS")
        for line in rows:
            L.append("    " + line)

    if st.notes:
        L.append("")
        L.append("  NOTE")
        for note in st.notes:
            for line in _wrap(note, 72):
                L.append("    " + line)

    if also:
        L.append("")
        L.append("  ALSO WORTH KNOWING")
        for cmd, what in ALSO:
            L.append("    %-12s %s" % (cmd, what))
    L.append("")
    return "\n".join(line.rstrip() for line in L)


def _where(st):
    out = []
    ds = st.dataset
    if ds is not None:
        bits = ["%s episodes" % ds.n_episodes if ds.n_episodes else "a dataset"]
        if ds.seconds:
            bits.append("%.0f seconds" % ds.seconds)
        if ds.robot_type:
            bits.append(str(ds.robot_type))
        out.append("recorded: " + ", ".join(bits))
        if ds.joint_names:
            out.append("joints:   %d (%s)" % (len(ds.joint_names),
                                              ", ".join(ds.joint_names[:3]) +
                                              (", ..." if len(ds.joint_names) > 3 else "")))
    else:
        out.append("recorded: no LeRobot dataset found here")
    out.append("trained:  %s" % (
        "%d checkpoint%s" % (len(st.policies), "" if len(st.policies) == 1 else "s")
        if st.policies else "nothing yet"))
    if st.observed_n:
        out.append("evaluated: %d trials at %.0f%%" % (st.observed_n, st.observed_rate))
    else:
        out.append("evaluated: no trials logged")
    return out


def _budget(st, force=False):
    rate = st.observed_rate if st.observed_rate is not None else 50.0
    if st.stage in ("no_data", "data") and not force:
        return []
    rows = []
    for n in (10, 20, 50, 100, 200):
        call = smallest_callable_pp(n, rate)
        mark = "  <- the guide's default" if n == GUIDE_TRIALS else ""
        if st.observed_n and n == st.observed_n:
            mark = "  <- what you ran"
        rows.append("%4d trials  ->  resolves a drop of %5.1f points%s"
                    % (n, call, mark))
    rows.append("")
    rows.append("a trial is about a minute of standing at the robot, so the "
                "row you pick")
    rows.append("is a time budget, not a statistical preference.")
    return rows


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
