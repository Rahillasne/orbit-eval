"""`orbit cover` counts what a recording never saw. It never scores what it did.

Factor coverage from design of experiments: the task is a set of factors, the
recording is a set of episodes, and the report is the cells with nothing in
them. Every number here is a count, an expected count under even coverage, or
an arithmetic step on the metadata. None is a judgement about a demonstration.
"""

import json
import os
import shutil
import tempfile
import unittest

from orbit_eval import cover, dataset

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def write_v21(root, tasks, episodes, joints=JOINTS, fps=30, stats=True):
    """A v2.1 dataset: episodes.jsonl, tasks.jsonl, episodes_stats.jsonl.

    `episodes` is a list of (task, length, state_min, state_max, state_mean).
    """
    meta = os.path.join(root, "meta")
    os.makedirs(meta, exist_ok=True)
    info = {
        "codebase_version": "v2.1", "robot_type": "so101_follower",
        "total_episodes": len(episodes), "total_tasks": len(tasks), "fps": fps,
        "total_frames": sum(e[1] for e in episodes),
        "features": {
            "action": {"dtype": "float32", "shape": [len(joints)], "names": joints},
            "observation.state": {"dtype": "float32", "shape": [len(joints)],
                                  "names": joints},
        },
    }
    with open(os.path.join(meta, "info.json"), "w") as fh:
        json.dump(info, fh)
    with open(os.path.join(meta, "tasks.jsonl"), "w") as fh:
        for i, t in enumerate(tasks):
            fh.write(json.dumps({"task_index": i, "task": t}) + "\n")
    with open(os.path.join(meta, "episodes.jsonl"), "w") as fh:
        for i, e in enumerate(episodes):
            fh.write(json.dumps({"episode_index": i, "tasks": [e[0]], "length": e[1]}) + "\n")
    if stats:
        with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as fh:
            for i, e in enumerate(episodes):
                lo, hi, mean = e[2], e[3], e[4]
                std = [(h - l) / 4.0 for l, h in zip(lo, hi)]
                block = {"min": lo, "max": hi, "mean": mean, "std": std, "count": [e[1]]}
                fh.write(json.dumps({"episode_index": i, "stats": {
                    "observation.state": block, "action": block}}) + "\n")
    return root


def flat(lo, hi, mean=None, n=6):
    lo, hi = [float(lo)] * n, [float(hi)] * n
    mean = [(a + b) / 2.0 for a, b in zip(lo, hi)] if mean is None else [float(mean)] * n
    return lo, hi, mean


class Temp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orbit-cover-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------ the template

class TestTemplate(unittest.TestCase):
    def test_one_slot_of_varying_width(self):
        t = cover.induce_template([
            "pick up the milk and place it in the basket",
            "pick up the chocolate pudding and place it in the basket",
            "pick up the bbq sauce and place it in the basket"])
        self.assertEqual(t.template, "pick up the {A} and place it in the basket")
        self.assertEqual(t.slots, ["A"])
        self.assertEqual(t.levels["pick up the chocolate pudding and place it in the basket"],
                         {"A": "chocolate pudding"})

    def test_two_slots(self):
        t = cover.induce_template([
            "put the cup on the plate", "put the cup on the stove",
            "put the bowl on the plate"])
        self.assertEqual(t.template, "put the {A} on the {B}")
        self.assertEqual(t.levels["put the bowl on the plate"], {"A": "bowl", "B": "plate"})

    def test_a_slot_can_be_at_the_front(self):
        t = cover.induce_template([
            "open the top drawer", "open the bottom drawer", "close the top drawer"])
        self.assertEqual(t.template, "{A} the {B} drawer")

    def test_unrelated_instructions_fall_back_to_one_factor(self):
        t = cover.induce_template([
            "turn on the stove", "put the bowl in the drawer",
            "push the plate to the front of the table"])
        self.assertIsNone(t.template)
        self.assertEqual(t.slots, ["instruction"])

    def test_a_single_instruction_has_no_slots(self):
        t = cover.induce_template(["pink lego brick into the transparent box"])
        self.assertEqual(t.slots, [])
        self.assertEqual(t.template, "pink lego brick into the transparent box")


# ------------------------------------------------------------- the reading

class TestReadV21(Temp):
    def test_reads_tasks_lengths_and_per_episode_state(self):
        write_v21(self.dir, ["a", "b"], [("a", 100) + flat(-50, 50), ("b", 120) + flat(-40, 60)])
        meta = dataset.load(self.dir)
        eps, how, reason = cover.load_episodes(meta)
        self.assertIsNone(reason)
        self.assertIn("episodes.jsonl", how)
        self.assertEqual([e.tasks for e in eps], [["a"], ["b"]])
        self.assertEqual([e.length for e in eps], [100, 120])
        self.assertEqual(eps[1].state["max"][0], 60.0)

    def test_tasks_file_lists_tasks_that_have_no_episode(self):
        write_v21(self.dir, ["a", "b", "c"], [("a", 100) + flat(-50, 50)])
        self.assertEqual(cover.load_tasks(dataset.load(self.dir)), ["a", "b", "c"])

    def test_without_stats_the_episode_still_has_its_task_and_length(self):
        write_v21(self.dir, ["a"], [("a", 100) + flat(-50, 50)], stats=False)
        eps, _how, reason = cover.load_episodes(dataset.load(self.dir))
        self.assertIsNone(reason)
        self.assertIsNone(eps[0].state)


class TestReadV30(Temp):
    def _write_v30(self, with_stats=True):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        meta = os.path.join(self.dir, "meta")
        os.makedirs(os.path.join(meta, "episodes", "chunk-000"), exist_ok=True)
        info = {"codebase_version": "v3.0", "robot_type": "so101_follower",
                "total_episodes": 3, "total_tasks": 2, "fps": 30, "total_frames": 330,
                "features": {"observation.state": {"dtype": "float32", "shape": [6],
                                                   "names": JOINTS}}}
        with open(os.path.join(meta, "info.json"), "w") as fh:
            json.dump(info, fh)
        pq.write_table(pa.table({"task_index": [0, 1],
                                 "__index_level_0__": ["grab pens", "grab tapes"]}),
                       os.path.join(meta, "tasks.parquet"))
        cols = {"episode_index": [0, 1, 2],
                "tasks": [["grab pens"], ["grab pens"], ["grab tapes"]],
                "length": [100, 110, 120]}
        if with_stats:
            cols["stats/observation.state/min"] = [[-50.0] * 6, [-40.0] * 6, [-30.0] * 6]
            cols["stats/observation.state/max"] = [[50.0] * 6, [60.0] * 6, [70.0] * 6]
            cols["stats/observation.state/mean"] = [[0.0] * 6, [10.0] * 6, [20.0] * 6]
            cols["stats/observation.state/std"] = [[25.0] * 6] * 3
        pq.write_table(pa.table(cols),
                       os.path.join(meta, "episodes", "chunk-000", "file-000.parquet"))

    def test_reads_the_parquet_tables_when_pyarrow_is_present(self):
        self._write_v30()
        meta = dataset.load(self.dir)
        eps, how, reason = cover.load_episodes(meta)
        self.assertIsNone(reason)
        self.assertIn("parquet", how)
        self.assertEqual([e.length for e in eps], [100, 110, 120])
        self.assertEqual(eps[2].tasks, ["grab tapes"])
        self.assertEqual(eps[2].state["mean"][0], 20.0)
        self.assertEqual(cover.load_tasks(meta), ["grab pens", "grab tapes"])

    def test_says_what_it_needs_when_pyarrow_is_missing(self):
        self._write_v30()
        real = cover._pyarrow
        cover._pyarrow = lambda: None
        try:
            rep = cover.scan(self.dir)
        finally:
            cover._pyarrow = real
        self.assertEqual(rep["needs"], "pyarrow")
        self.assertEqual(rep["n_episodes"], 3)
        text = cover.format_cover(rep)
        self.assertIn("orbit-eval[coverage]", text)


# ------------------------------------------------------------- the counting

class TestCoverage(Temp):
    def test_a_task_with_no_episodes_is_the_largest_gap(self):
        eps = [("put the cup on the plate", 100) + flat(-50, 50)] * 6 + \
              [("put the bowl on the plate", 100) + flat(-50, 50)] * 6
        write_v21(self.dir, ["put the cup on the plate", "put the bowl on the plate",
                             "put the cup on the stove"], eps)
        rep = cover.scan(self.dir)
        self.assertEqual(rep["instructions"]["empty_tasks"], ["put the cup on the stove"])
        self.assertEqual(rep["largest_gap"]["kind"], "task")
        self.assertIn("put the cup on the stove", rep["largest_gap"]["sentence"])

    def test_pairwise_cells_never_recorded_are_listed(self):
        eps = [("put the cup on the plate", 100) + flat(-50, 50)] * 4 + \
              [("put the cup on the stove", 100) + flat(-50, 50)] * 4 + \
              [("put the bowl on the plate", 100) + flat(-50, 50)] * 4
        write_v21(self.dir, sorted(set(e[0] for e in eps)), eps)
        rep = cover.scan(self.dir)
        pairs = rep["instructions"]["pairs"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["slots"], ["A", "B"])
        self.assertEqual(pairs[0]["cells"], 4)
        self.assertEqual(pairs[0]["empty"], [["bowl", "stove"]])

    def test_per_task_counts_are_reported(self):
        eps = [("a", 100) + flat(-50, 50)] * 5 + [("b", 100) + flat(-50, 50)] * 2
        write_v21(self.dir, ["a", "b"], eps)
        rep = cover.scan(self.dir)
        self.assertEqual({r["task"]: r["episodes"] for r in rep["instructions"]["per_task"]},
                         {"a": 5, "b": 2})

    def test_length_outliers_are_named_by_episode_index(self):
        eps = [("a", 100) + flat(-50, 50)] * 9 + [("a", 900) + flat(-50, 50)]
        write_v21(self.dir, ["a"], eps)
        rep = cover.scan(self.dir)
        self.assertEqual(rep["lengths"]["median"], 100)
        self.assertEqual(rep["lengths"]["long"], [9])
        self.assertEqual(rep["lengths"]["short"], [])

    def test_a_joint_that_never_moves_in_a_task_is_counted(self):
        def ep(task, roll_lo, roll_hi):
            lo, hi, mean = flat(-50, 50)
            lo[4], hi[4] = roll_lo, roll_hi
            mean[4] = (roll_lo + roll_hi) / 2.0
            return (task, 100, lo, hi, mean)
        eps = [ep("a", -50, 50)] * 5 + [ep("b", 0.0, 0.5)] * 5
        write_v21(self.dir, ["a", "b"], eps)
        rep = cover.scan(self.dir)
        roll = [j for j in rep["joints"] if j["name"] == "wrist_roll.pos"][0]
        self.assertEqual(roll["idle_episodes"], 5)
        self.assertEqual(roll["idle_tasks"], ["b"])
        pan = [j for j in rep["joints"] if j["name"] == "shoulder_pan.pos"][0]
        self.assertEqual(pan["idle_episodes"], 0)

    def test_where_the_arm_sat_is_split_into_thirds_of_the_recorded_range(self):
        eps = [("a", 100) + flat(-90, -70, mean=-80)] * 3 + \
              [("a", 100) + flat(70, 90, mean=80)] * 3
        write_v21(self.dir, ["a"], eps)
        rep = cover.scan(self.dir)
        pan = [j for j in rep["joints"] if j["name"] == "shoulder_pan.pos"][0]
        self.assertEqual(pan["thirds"], [3, 0, 3])
        empty = rep["regions"]["empty"]
        self.assertTrue(any(c["task"] == "a" and c["joint"] == "shoulder_pan.pos"
                            and c["third"] == "middle" for c in empty))
        self.assertEqual(rep["largest_gap"]["kind"], "region")

    def test_even_coverage_has_no_gap_to_name(self):
        eps = []
        for task in ("a", "b"):
            for m in (-80, 0, 80):
                eps.append((task, 100) + flat(-90, 90, mean=m))
        write_v21(self.dir, ["a", "b"], eps)
        rep = cover.scan(self.dir)
        self.assertIsNone(rep["largest_gap"])
        self.assertIn("no empty cell", cover.format_cover(rep).lower())


# ---------------------------------------------------------------- the words

class TestWords(Temp):
    def test_it_counts_and_never_judges(self):
        eps = [("put the cup on the plate", 100) + flat(-50, 50)] * 3 + \
              [("put the bowl on the stove", 100) + flat(-50, 50)] * 3
        write_v21(self.dir, ["put the cup on the plate", "put the bowl on the stove"], eps)
        out = cover.format_cover(cover.scan(self.dir)).lower()
        for forbidden in ("low quality", "bad demonstration", "drop this", "will fail",
                          "caused the", "should use", "recommend", "quality"):
            self.assertNotIn(forbidden, out)
        self.assertIn("never scores a demonstration", out)

    def test_the_report_names_its_sections(self):
        eps = [("a", 100) + flat(-50, 50)] * 3
        write_v21(self.dir, ["a"], eps)
        out = cover.format_cover(cover.scan(self.dir))
        self.assertIn("COVERAGE", out)
        self.assertIn("INSTRUCTIONS", out)
        self.assertIn("EPISODES", out)

    def test_json_round_trips(self):
        eps = [("a", 100) + flat(-50, 50)] * 3
        write_v21(self.dir, ["a"], eps)
        json.dumps(cover.scan(self.dir))

    def test_no_dataset_is_an_error_not_a_report(self):
        with self.assertRaises(ValueError):
            cover.scan(self.dir)


if __name__ == "__main__":
    unittest.main()
