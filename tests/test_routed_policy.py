"""RoutedPolicy: dispatch, lazy loading, defaults, key tolerance, release loading."""

import json
import os
import shutil
import tempfile
import unittest

from orbit_eval.routed_policy import LeRobotCandidate, RoutedPolicy


class Fake(object):
    def __init__(self, name):
        self.name = name; self.resets = 0; self.device = None
    def select_action(self, obs):
        return "%s:%s" % (self.name, obs)
    def reset(self):
        self.resets += 1
    def to(self, device):
        self.device = device; return self
    def eval(self):
        return self


class TestRoutedPolicy(unittest.TestCase):
    def test_dispatch_and_lazy_load(self):
        loads = []
        def mk(name):
            def _l():
                loads.append(name); return Fake(name)
            return _l
        rp = RoutedPolicy({"g/0": "a", "g/1": "b", "g/2": "a"}, {"a": mk("a"), "b": mk("b"), "c": mk("c")})
        self.assertEqual(rp.select_action("obs", task="g/0"), "a:obs")
        self.assertEqual(loads, ["a"])
        self.assertEqual(rp.select_action("obs", task="1"), "b:obs")     # bare id tolerated
        self.assertEqual(rp.select_action("obs", task=2), "a:obs")       # int tolerated
        self.assertEqual(rp.loaded(), ["a", "b"])                        # c never loaded
        rp.set_task("g/1")
        self.assertEqual(rp.select_action("o"), "b:o")
        rp.reset()
        self.assertEqual(rp.policy_for("g/1").resets, 1)
        rp.to("cpu")
        self.assertEqual(rp.policy_for("g/0").device, "cpu")
        self.assertEqual(rp.candidates_used, ["a", "b"])

    def test_unknown_task_refuses_without_default(self):
        rp = RoutedPolicy({"t0": "a"}, {"a": Fake("a")})
        with self.assertRaises(KeyError):
            rp.select_action("o", task="t9")
        rp2 = RoutedPolicy({"t0": "a"}, {"a": Fake("a"), "inc": Fake("inc")}, default="inc")
        self.assertEqual(rp2.select_action("o", task="t9"), "inc:o")
        with self.assertRaises(ValueError):
            rp.select_action("o")                                        # no task at all

    def test_missing_policy_is_an_error(self):
        with self.assertRaises(ValueError):
            RoutedPolicy({"t0": "a"}, {"b": Fake("b")})

    def test_task_key_fn(self):
        rp = RoutedPolicy({"t0": "a", "t1": "b"}, {"a": Fake("a"), "b": Fake("b")},
                          task_key=lambda obs: obs["task"])
        self.assertEqual(rp.select_action({"task": "t1", "x": 1}), "b:{'task': 't1', 'x': 1}")

    def test_from_release(self):
        tmp = tempfile.mkdtemp()
        try:
            rec = {"release": {"incumbent": {"t0": "inc", "t1": "inc"},
                               "candidates": ["cand", "inc", "unused"],
                               "rows": [{"task": "t0", "chosen": "cand"}, {"task": "t1", "chosen": "inc"}],
                               "policy_paths": {"cand": "/w/cand", "inc": "/w/inc"}}}
            p = os.path.join(tmp, "release.json"); json.dump(rec, open(p, "w"))
            rp = RoutedPolicy.from_release(p, loader=lambda path: Fake(os.path.basename(path)))
            self.assertEqual(rp.select_action("o", task="t0"), "cand:o")
            self.assertEqual(rp.select_action("o", task="t1"), "inc:o")
            self.assertEqual(rp.tasks, ["t0", "t1"])
            rp2 = RoutedPolicy.from_release(p)                            # no loader: paths only
            self.assertEqual(rp2.policy_for("t0"), "/w/cand")
            rec["release"]["policy_paths"] = {}
            json.dump(rec, open(p, "w"))
            with self.assertRaises(ValueError):
                RoutedPolicy.from_release(p)
            rp3 = RoutedPolicy.from_release(p, policy_paths={"cand": "/x", "inc": "/y"})
            self.assertEqual(rp3.policy_for("t1"), "/y")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_as_lerobot_policy_needs_lerobot(self):
        from orbit_eval.routed_policy import as_lerobot_policy
        try:
            import lerobot  # noqa: F401
            have = True
        except ImportError:
            have = False
        if not have:
            with self.assertRaises(ImportError):
                as_lerobot_policy(RoutedPolicy({"t0": "a"}, {"a": Fake("a")}))

    def test_kwargs_forwarded(self):
        class KW(object):
            def select_action(self, obs, **kw):
                return (obs, sorted(kw))
        rp = RoutedPolicy({"t0": "a"}, {"a": KW()})
        self.assertEqual(rp.select_action("o", task="t0", foo=1), ("o", ["foo"]))

    def test_lerobot_candidate_applies_pipelines(self):
        c = LeRobotCandidate(Fake("p"), preprocessor=lambda o: "pre(%s)" % o, postprocessor=lambda a: "post(%s)" % a)
        self.assertEqual(c.select_action("o"), "post(p:pre(o))")
        c.reset(); self.assertEqual(c.policy.resets, 1)


if __name__ == "__main__":
    unittest.main()
