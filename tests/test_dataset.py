"""Zero-dependency checks over LeRobot dataset metadata.

Every check here reads only meta/info.json and meta/stats.json, which are plain
JSON in both the v2.x and v3.0 dataset layouts. Nothing opens a parquet file, a
video, or the network: a defect a user can act on has to be findable in 6 KB.
"""

import json
import os
import shutil
import tempfile
import unittest

from orbit_eval import dataset


JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

# lerobot/svla_so101_pickplace, a real SO-101 set, with the two clipped action
# bounds (-100.0 and +100.0) pulled in to 99.0 so this fixture reads as healthy.
HEALTHY_STATE = {
    "min": [-92.655, -98.911, 16.809, 34.040, -92.283, 1.208],
    "max": [88.104, 8.794, 99.372, 98.643, -20.195, 32.685],
    "mean": [7.987, -55.181, 66.776, 69.253, -53.439, 8.112],
    "std": [44.492, 36.827, 27.708, 13.013, 17.703, 8.405],
    "count": [11939],
}
HEALTHY_ACTION = {
    "min": [-93.456, -99.0, 12.972, 33.532, -92.772, 0.0],
    "max": [88.015, 8.126, 99.0, 99.490, -20.0, 32.998],
    "mean": [8.021, -55.962, 65.256, 69.182, -53.420, 6.849],
    "std": [44.563, 36.485, 29.012, 13.238, 17.764, 8.999],
    "count": [11939],
}


def write_dataset(root, state=None, action=None, episodes=50, frames=11939, fps=30):
    meta = os.path.join(root, "meta")
    os.makedirs(meta, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "total_episodes": episodes,
        "total_frames": frames,
        "fps": fps,
        "features": {
            "action": {"dtype": "float32", "shape": [6], "names": JOINTS},
            "observation.state": {"dtype": "float32", "shape": [6], "names": JOINTS},
            "observation.images.up": {"dtype": "video", "shape": [480, 640, 3],
                                      "names": ["height", "width", "channels"]},
        },
    }
    with open(os.path.join(meta, "info.json"), "w") as fh:
        json.dump(info, fh)
    st = json.loads(json.dumps(state if state is not None else HEALTHY_STATE))
    ac = json.loads(json.dumps(action if action is not None else HEALTHY_ACTION))
    with open(os.path.join(meta, "stats.json"), "w") as fh:
        json.dump({"observation.state": st, "action": ac,
                   "timestamp": {"min": [0.0], "max": [10.2], "mean": [4.0],
                                 "std": [2.4], "count": [frames]}}, fh)
    return root


def codes(findings):
    return sorted(f.code for f in findings)


class TempDataset(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orbit-ds-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestLoad(TempDataset):
    def test_loads_a_v3_dataset(self):
        write_dataset(self.dir)
        meta = dataset.load(self.dir)
        self.assertEqual(meta.n_episodes, 50)
        self.assertEqual(meta.fps, 30)
        self.assertEqual(meta.joint_names, JOINTS)
        self.assertEqual(meta.robot_type, "so101_follower")

    def test_finds_the_dataset_from_a_parent_directory(self):
        inner = os.path.join(self.dir, "data", "my_task")
        write_dataset(inner)
        self.assertEqual(dataset.find(self.dir), inner)

    def test_find_returns_none_when_there_is_no_dataset(self):
        self.assertIsNone(dataset.find(self.dir))

    def test_load_accepts_the_meta_directory_itself(self):
        write_dataset(self.dir)
        meta = dataset.load(os.path.join(self.dir, "meta"))
        self.assertEqual(meta.n_episodes, 50)


class TestHealthyDataset(TempDataset):
    def test_a_healthy_dataset_reports_no_joint_defect(self):
        write_dataset(self.dir)
        found = codes(dataset.checks(dataset.load(self.dir)))
        for bad in ("dead_joint", "saturated_action", "not_following", "nan_stats"):
            self.assertNotIn(bad, found)


class TestDeadJoint(TempDataset):
    def test_zero_variance_joint_is_dead(self):
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["std"][3] = 0.0
        state["min"][3] = 41.7
        state["max"][3] = 41.7
        write_dataset(self.dir, state=state)
        f = [x for x in dataset.checks(dataset.load(self.dir)) if x.code == "dead_joint"]
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].joint, "wrist_flex.pos")
        self.assertEqual(f[0].severity, "blocking")

    def test_the_message_names_the_joint_and_says_what_to_do(self):
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["std"][0] = 0.0
        state["min"][0] = state["max"][0] = 3.0
        write_dataset(self.dir, state=state)
        f = [x for x in dataset.checks(dataset.load(self.dir)) if x.code == "dead_joint"][0]
        self.assertIn("shoulder_pan.pos", f.message)
        self.assertTrue(f.fix)

    def test_a_barely_moving_joint_is_not_called_dead(self):
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["std"][3] = 2.0
        write_dataset(self.dir, state=state)
        self.assertNotIn("dead_joint", codes(dataset.checks(dataset.load(self.dir))))


class TestSaturatedAction(TempDataset):
    def test_action_pinned_at_the_normalised_limit_is_flagged(self):
        action = json.loads(json.dumps(HEALTHY_ACTION))
        action["min"][1] = -100.0
        write_dataset(self.dir, action=action)
        f = [x for x in dataset.checks(dataset.load(self.dir))
             if x.code == "saturated_action"]
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].joint, "shoulder_lift.pos")

    def test_a_gripper_at_either_end_of_its_range_is_not_saturation(self):
        # The gripper runs 0..100 and both ends are the calibrated closed and
        # open positions (validation/NORMALISATION_BOUND_2026-09-17.md). A
        # gripper closing on nothing or opening fully sits on them by design.
        action = json.loads(json.dumps(HEALTHY_ACTION))
        action["max"][5] = 100.0
        action["min"][5] = 0.0
        write_dataset(self.dir, action=action)
        self.assertNotIn("saturated_action", codes(dataset.checks(dataset.load(self.dir))))

    def test_a_gripper_resting_at_zero_is_not_saturation(self):
        # 0.0 is the closed rest position, not a clipped command.
        write_dataset(self.dir)
        self.assertNotIn("saturated_action", codes(dataset.checks(dataset.load(self.dir))))

    def test_no_saturation_check_when_the_units_are_not_normalised(self):
        action = json.loads(json.dumps(HEALTHY_ACTION))
        action["min"] = [-3.14] * 6
        action["max"] = [3.14] * 6
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["min"] = [-3.14] * 6
        state["max"] = [3.14] * 6
        write_dataset(self.dir, state=state, action=action)
        self.assertNotIn("saturated_action", codes(dataset.checks(dataset.load(self.dir))))


class TestNotFollowing(TempDataset):
    def test_commanded_range_far_beyond_reached_range_is_flagged(self):
        action = json.loads(json.dumps(HEALTHY_ACTION))
        action["max"][2] = 99.0
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["max"][2] = 40.0          # commanded to 99, never got past 40
        write_dataset(self.dir, state=state, action=action)
        f = [x for x in dataset.checks(dataset.load(self.dir)) if x.code == "not_following"]
        self.assertEqual([x.joint for x in f], ["elbow_flex.pos"])


class TestNaN(TempDataset):
    def test_nan_in_the_stats_is_blocking(self):
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["std"][2] = None
        write_dataset(self.dir, state=state)
        f = [x for x in dataset.checks(dataset.load(self.dir)) if x.code == "nan_stats"]
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].severity, "blocking")


class TestSize(TempDataset):
    def test_too_few_episodes_is_advisory_not_blocking(self):
        write_dataset(self.dir, episodes=12, frames=3000)
        f = [x for x in dataset.checks(dataset.load(self.dir)) if x.code == "few_episodes"]
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].severity, "advisory")

    def test_fifty_episodes_is_not_flagged(self):
        write_dataset(self.dir, episodes=50)
        self.assertNotIn("few_episodes", codes(dataset.checks(dataset.load(self.dir))))


if __name__ == "__main__":
    unittest.main()
