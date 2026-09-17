"""`orbit body`: reach, payload, dof and price against a spec sheet. Arithmetic, never a judgement."""

import json
import os
import shutil
import tempfile
import unittest

from orbit_eval import body, dataset, freeze
from test_dataset import write_dataset, HEALTHY_STATE
from test_freeze import write_checkpoint

DB = {"generated": "2026-09-17", "robots": [
    {"id": "so101", "name": "SO-101", "maker": "The Robot Studio", "class": "fixed_arm",
     "dof": 5, "gripper": "parallel", "reach_mm": 400, "payload_kg": 0.5, "price_usd": 250,
     "lerobot_driver": "so_follower", "mujoco_menagerie": "robotstudio_so101",
     "source": {"reach_mm": "https://example.org/so101", "payload_kg": "https://example.org/so101"},
     "notes": ""},
    {"id": "ur5e", "name": "UR5e", "maker": "Universal Robots", "class": "fixed_arm",
     "dof": 6, "gripper": None, "reach_mm": 850, "payload_kg": 5.0, "price_usd": 35000,
     "lerobot_driver": None, "mujoco_menagerie": "universal_robots_ur5e",
     "source": {"reach_mm": "https://example.org/ur5e"}, "notes": ""},
    {"id": "mystery", "name": "Mystery Arm", "maker": None, "class": "fixed_arm",
     "dof": 6, "gripper": None, "reach_mm": 600, "payload_kg": None, "price_usd": None,
     "lerobot_driver": None, "mujoco_menagerie": None, "source": {}, "notes": ""},
    {"id": "go2", "name": "Go2", "maker": "Unitree", "class": "legged",
     "dof": 12, "gripper": "none", "reach_mm": None, "payload_kg": 8.0, "price_usd": 1600,
     "lerobot_driver": None, "mujoco_menagerie": "unitree_go2", "source": {}, "notes": ""},
]}

MAP = {"cells": [
    {"job_shape": "pick_place", "embodiment": "so101", "model": "act", "sr": 70.0,
     "ci95": [60.0, 78.0], "n_trials": 100, "n_seeds": 1, "n_sources": 1,
     "status": "ONE_SEED", "sim_or_real": "real"}],
    "skills": [{"hub_id": "x/act_so101_test", "url": "https://huggingface.co/x/act_so101_test",
                "embodiment": "so101", "model": "act", "job_shape": "unknown", "measured": False}]}


class TestFit(unittest.TestCase):
    def test_a_short_payload_is_a_number_not_a_judgement(self):
        rows = body.fit(DB, {"payload_kg": 3.0})
        so = [r for r in rows if r["id"] == "so101"][0]
        self.assertEqual(so["verdict"], "fails")
        c = [c for c in so["checks"] if c["field"] == "payload_kg"][0]
        self.assertAlmostEqual(c["margin"], -2.5)
        out = body.format_fit(rows, {"payload_kg": 3.0})
        self.assertIn("short by 2.5 kg", out)
        self.assertNotIn("not right for", out.lower())

    def test_passing_robots_come_first_and_cheapest_first(self):
        rows = body.fit(DB, {"reach_mm": 300, "payload_kg": 0.3})
        self.assertEqual(rows[0]["id"], "so101")
        self.assertEqual(rows[0]["verdict"], "passes")
        self.assertEqual(rows[1]["id"], "ur5e")

    def test_an_unpublished_figure_is_unknown_not_a_pass(self):
        rows = body.fit(DB, {"payload_kg": 1.0})
        mys = [r for r in rows if r["id"] == "mystery"][0]
        self.assertEqual(mys["verdict"], "unknown")
        self.assertIn("not published", body.format_fit(rows, {"payload_kg": 1.0}))

    def test_budget_is_an_upper_bound(self):
        rows = body.fit(DB, {"price_usd": 5000})
        self.assertEqual([r["verdict"] for r in rows if r["id"] == "ur5e"], ["fails"])
        self.assertEqual([r["verdict"] for r in rows if r["id"] == "so101"], ["passes"])

    def test_class_filters(self):
        rows = body.fit(DB, {"class": "legged"})
        self.assertEqual([r["id"] for r in rows], ["go2"])

    def test_the_map_block_says_when_nobody_has_measured(self):
        rows = body.fit(DB, {"reach_mm": 300})
        out = body.format_fit(rows, {"reach_mm": 300}, m=MAP, job_shape="insertion")
        self.assertIn("nobody has measured", out)
        out2 = body.format_fit(rows, {"reach_mm": 300}, m=MAP, job_shape="pick_place")
        self.assertIn("ONE_SEED", out2)
        self.assertIn("SO-101", out2)


class TestSheet(unittest.TestCase):
    def test_find_is_forgiving_about_the_name(self):
        self.assertEqual(body.find(DB, "SO-101")["id"], "so101")
        self.assertEqual(body.find(DB, "so_101")["id"], "so101")
        self.assertIsNone(body.find(DB, "nothing"))

    def test_the_sheet_names_its_sources_and_its_skills(self):
        out = body.format_sheet(body.find(DB, "so101"), m=MAP)
        self.assertIn("https://example.org/so101", out)
        self.assertIn("not published", out)          # repeatability
        self.assertIn("x/act_so101_test", out)
        self.assertIn("none with a measured number", out)

    def test_the_list_counts_measured_cells(self):
        out = body.format_list(DB, m=MAP)
        self.assertIn("so101", out)
        self.assertIn("$35,000", out)


class TestSkillFit(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orbit-body-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _manifest(self):
        ck = write_checkpoint(self.dir)
        ds = write_dataset(os.path.join(self.dir, "skill_data"))
        return freeze.build(ck, dataset_path=ds)

    def test_a_matching_recording_is_inside_on_every_joint(self):
        m = self._manifest()
        mine = dataset.load(write_dataset(os.path.join(self.dir, "mine")))
        f = body.skill_fit(m, mine)
        self.assertTrue(f["comparable"])
        self.assertEqual(f["n_outside"], 0)
        self.assertIn("calibration match", body.format_skill_fit(f, "release.json"))

    def test_a_recording_past_the_trained_range_is_named_joint_by_joint(self):
        m = self._manifest()
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["max"][4] = 40.0            # wrist_roll goes far past -20
        mine = dataset.load(write_dataset(os.path.join(self.dir, "mine"), state=state))
        f = body.skill_fit(m, mine)
        self.assertEqual(f["n_outside"], 1)
        row = [r for r in f["rows"] if r["joint"] == "wrist_roll.pos"][0]
        self.assertEqual(row["verdict"], "outside")
        out = body.format_skill_fit(f, "release.json")
        self.assertIn("wrist_roll.pos", out)
        self.assertIn("OUTSIDE on the high end", out)
        self.assertIn("calibration", out.lower())

    def test_a_manifest_without_ranges_cannot_be_compared(self):
        ck = write_checkpoint(self.dir)
        m = freeze.build(ck)
        mine = dataset.load(write_dataset(os.path.join(self.dir, "mine")))
        f = body.skill_fit(m, mine)
        self.assertFalse(f["comparable"])


if __name__ == "__main__":
    unittest.main()
