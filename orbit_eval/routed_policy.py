# orbit-eval — honest statistics for robot-policy evaluation.
# Copyright (C) 2026 ORBIT Research
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the license along with this program. If not,
# see <https://www.gnu.org/licenses/>.
#
# A commercial license, exempting you from the AGPL's source-disclosure terms, is
# available from ORBIT Research.

"""RoutedPolicy — serve an orbit-eval release: task in, the right candidate's action out.

    from orbit_eval.routed_policy import RoutedPolicy
    rp = RoutedPolicy.from_release("release.json", loader=my_loader)   # loader(path) -> policy
    rp.set_task("libero_object/3")            # or pass task=... per call, or a task_key fn
    action = rp.select_action(observation)    # delegates to the candidate chosen for that task

The wrapper is pure Python and framework-agnostic: a "policy" is anything with
`select_action(observation)`; `reset()`, `eval()` and `to(device)` are forwarded when
present. Candidate policies are loaded lazily on first use, so a release that keeps the
incumbent on most tasks loads only what it serves.

For LeRobot, `LeRobotCandidate` bundles one policy with its own pre/post processor
pipelines (each candidate has its own normalisation statistics), so a RoutedPolicy can
be passed to `lerobot.scripts.lerobot_eval.eval_policy` with identity pipelines and the
rollout loop needs no change. See `examples/eval_routed_lerobot.py`.
"""

import json


class RoutedPolicy(object):
    def __init__(self, table, candidates, default=None, task_key=None):
        """table: {task_key: candidate_name}. candidates: {name: policy | callable -> policy}.
        default: candidate for tasks missing from the table (None = refuse). task_key:
        optional callable(observation) -> task_key, for batches that carry the task."""
        self.table = dict(table)
        self._cands = dict(candidates)
        self._loaded = {}
        self.default = default
        self._task = None
        self._task_key_fn = task_key
        missing = sorted(set(self.table.values()) - set(self._cands))
        if missing:
            raise ValueError("release names candidates with no policy: %s" % ", ".join(missing))
        if default is not None and default not in self._cands:
            raise ValueError("default candidate %r has no policy" % default)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_release(cls, path, loader=None, policy_paths=None, default=None, task_key=None,
                     only_used=True):
        """Build from an `orbit-eval route build --out release.json` record.

        loader(path) -> policy (called lazily); policy_paths {candidate: path} overrides or
        completes the record's `policy_paths` (set at build time with --policy-path).
        Without a loader the candidates are the paths themselves (dispatch table only).
        """
        with open(path) as fh:
            rec = json.load(fh)
        rel = rec.get("release", rec)
        table = {r["task"]: r["chosen"] for r in rel["rows"]}
        paths = dict(rel.get("policy_paths") or {})
        paths.update(policy_paths or {})
        names = sorted(set(table.values()) | ({default} if default else set()))
        if not only_used:
            names = sorted(set(names) | set(rel.get("candidates", [])))
        missing = [n for n in names if n not in paths]
        if missing:
            raise ValueError("no policy path for candidate(s) %s — pass policy_paths or build "
                             "with --policy-path name=path" % ", ".join(missing))
        if loader is None:
            cands = {n: paths[n] for n in names}
        else:
            cands = {n: (lambda p=paths[n]: loader(p)) for n in names}
        inc = rel.get("incumbent")
        if default is None and isinstance(inc, str) and inc in paths:
            default = inc
        return cls(table, cands, default=default, task_key=task_key)

    # ---------------------------------------------------------------- dispatch
    @property
    def tasks(self):
        return sorted(self.table)

    @property
    def candidates_used(self):
        return sorted(set(self.table.values()))

    def candidate_for(self, task):
        key = self._resolve(task)
        if key is None:
            if self.default is not None:
                return self.default
            raise KeyError("task %r is not in the release table (%s) and no default candidate is set"
                           % (task, ", ".join(self.tasks[:8]) + ("..." if len(self.tasks) > 8 else "")))
        return self.table[key]

    def _resolve(self, task):
        if task is None:
            return None
        t = str(task)
        if t in self.table:
            return t
        # tolerate "<group>/<id>" vs bare id, and ints
        tail = [k for k in self.table if k.endswith("/" + t) or k.split("/")[-1] == t]
        if len(tail) == 1:
            return tail[0]
        return None

    def _get(self, name):
        if name not in self._loaded:
            obj = self._cands[name]
            self._loaded[name] = obj() if callable(obj) and not hasattr(obj, "select_action") else obj
        return self._loaded[name]

    def set_task(self, task):
        self.candidate_for(task)          # validate now, fail early
        self._task = task
        return self

    def policy_for(self, task=None):
        return self._get(self.candidate_for(self._task if task is None else task))

    def select_action(self, observation, task=None, **kw):
        if task is None and self._task_key_fn is not None:
            task = self._task_key_fn(observation)
        if task is None:
            task = self._task
        if task is None:
            raise ValueError("no task: call set_task(), pass task=, or give a task_key function")
        return self._get(self.candidate_for(task)).select_action(observation, **kw)

    # ---------------------------------------------------------------- passthroughs
    def reset(self):
        for p in self._loaded.values():
            if hasattr(p, "reset"):
                p.reset()

    def eval(self):
        for p in self._loaded.values():
            if hasattr(p, "eval"):
                p.eval()
        return self

    def to(self, *args, **kw):
        for p in self._loaded.values():
            if hasattr(p, "to"):
                p.to(*args, **kw)
        return self

    def loaded(self):
        return sorted(self._loaded)

    def __repr__(self):
        return "RoutedPolicy(%d tasks -> %d candidates%s)" % (
            len(self.table), len(self.candidates_used),
            (", default=%s" % self.default) if self.default else "")


class LeRobotCandidate(object):
    """One LeRobot policy with its own preprocessor/postprocessor pipelines."""

    def __init__(self, policy, preprocessor=None, postprocessor=None):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def select_action(self, observation, **kw):
        if self.preprocessor is not None:
            observation = self.preprocessor(observation)
        action = self.policy.select_action(observation, **kw)
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        return action

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        for p in (self.preprocessor, self.postprocessor):
            if hasattr(p, "reset"):
                p.reset()

    def eval(self):
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        return self

    def to(self, *args, **kw):
        if hasattr(self.policy, "to"):
            self.policy.to(*args, **kw)
        return self


def load_lerobot_candidate(path, device="cuda"):
    """Load a pretrained LeRobot policy directory (or hub id) with its pipelines, the way
    `lerobot-eval` does. Requires lerobot >= 0.4 (PolicyProcessorPipeline)."""
    from lerobot.policies import make_policy, make_pre_post_processors           # lazy
    from lerobot.policies.factory import get_policy_class
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(path, config=cfg)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    return LeRobotCandidate(policy, pre, post)


_LEROBOT_CLS = {}


def as_lerobot_policy(routed, config=None):
    """Wrap a RoutedPolicy as a LeRobot `PreTrainedPolicy` so `lerobot-eval`'s rollout loop
    (which type-checks its policy) accepts the release as one policy object.

    `config` defaults to the config of the first candidate the release uses (loaded on
    demand). The wrapper is inference-only: `forward` raises; `select_action` and
    `reset` dispatch through the RoutedPolicy; `predict_action_chunk` delegates to the
    current candidate's policy when it has one (through that candidate's preprocessor).
    """
    try:
        from lerobot.policies.pretrained import PreTrainedPolicy
        from lerobot.configs.policies import PreTrainedConfig
    except ImportError as e:
        raise ImportError("as_lerobot_policy needs lerobot (>= 0.4): %s" % e)
    if "cls" not in _LEROBOT_CLS:
        class OrbitRoutedPolicy(PreTrainedPolicy):
            config_class = PreTrainedConfig
            name = "orbit_routed"

            def __init__(self, config, routed):
                super().__init__(config)
                self.routed = routed

            def get_optim_params(self):
                return {}

            def reset(self):
                self.routed.reset()

            def forward(self, batch):
                raise NotImplementedError("a routed release is inference-only; train the candidates")

            def predict_action_chunk(self, batch, **kw):
                cand = self.routed.policy_for()
                pol = getattr(cand, "policy", cand)
                pre = getattr(cand, "preprocessor", None)
                if pre is not None:
                    batch = pre(batch)
                return pol.predict_action_chunk(batch, **kw)

            def select_action(self, batch, **kw):
                return self.routed.select_action(batch, **kw)

            def __repr__(self):
                return "OrbitRoutedPolicy(%r)" % (self.routed,)
        _LEROBOT_CLS["cls"] = OrbitRoutedPolicy
    if config is None:
        first = routed.candidates_used[0] if routed.candidates_used else routed.default
        cand = routed._get(first)
        config = getattr(getattr(cand, "policy", cand), "config", None)
        if config is None:
            raise ValueError("cannot infer a PreTrainedConfig from candidate %r; pass config=" % first)
    return _LEROBOT_CLS["cls"](config, routed)


def identity_pipelines():
    """(preprocessor, postprocessor) that pass observations and actions through unchanged,
    with LeRobot's own converters — what to hand `eval_policy_all` when the candidates
    inside a RoutedPolicy already carry their pipelines."""
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, policy_action_to_transition,
                                              transition_to_batch, transition_to_policy_action)
    pre = PolicyProcessorPipeline(steps=[], to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline(steps=[], to_transition=policy_action_to_transition,
                                   to_output=transition_to_policy_action)
    return pre, post
