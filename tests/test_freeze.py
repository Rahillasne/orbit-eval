"""`orbit freeze` pins a checkpoint, its sampling seed and its eval seeds.

Freezing removes training variance, the large term, from every later
comparison. What is left is the cheap question: a fixed checkpoint under common
random numbers. A frozen checkpoint is not deterministic, because diffusion and
flow-matching policies sample their actions, so the inference seed is pinned
too, and the eval seed sequence is written down so any two runs are paired by
construction.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from orbit_eval import freeze
from test_dataset import write_dataset


def write_checkpoint(root, name="pretrained_model", weights=b"\x00" * 1024):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model.safetensors"), "wb") as fh:
        fh.write(weights)
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump({"type": "act", "n_action_steps": 100}, fh)
    with open(os.path.join(d, "train_config.json"), "w") as fh:
        json.dump({"seed": 1000, "steps": 100000, "policy": {"type": "act"}}, fh)
    return d


class Temp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orbit-freeze-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestManifest(Temp):
    def test_every_file_in_the_checkpoint_is_hashed(self):
        ck = write_checkpoint(self.dir)
        m = freeze.build(ck)
        files = m["checkpoint"]["files"]
        self.assertEqual(sorted(files), ["config.json", "model.safetensors", "train_config.json"])
        want = hashlib.sha256(b"\x00" * 1024).hexdigest()
        self.assertEqual(files["model.safetensors"]["sha256"], want)
        self.assertEqual(files["model.safetensors"]["bytes"], 1024)

    def test_the_checkpoint_has_one_hash_over_all_its_files(self):
        ck = write_checkpoint(self.dir)
        a = freeze.build(ck)["checkpoint"]["sha256"]
        b = freeze.build(ck)["checkpoint"]["sha256"]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_policy_type_is_read_from_the_config(self):
        ck = write_checkpoint(self.dir)
        self.assertEqual(freeze.build(ck)["policy"], "act")

    def test_seeds_are_written_down(self):
        ck = write_checkpoint(self.dir)
        m = freeze.build(ck, inference_seed=7, eval_start=2000, eval_count=50)
        self.assertEqual(m["inference"]["seed"], 7)
        self.assertEqual(m["eval"]["seeds"], {"start": 2000, "count": 50})
        self.assertEqual(freeze.eval_seeds(m)[:3], [2000, 2001, 2002])
        self.assertEqual(len(freeze.eval_seeds(m)), 50)

    def test_a_single_file_can_be_frozen(self):
        p = os.path.join(self.dir, "model.safetensors")
        with open(p, "wb") as fh:
            fh.write(b"abc")
        m = freeze.build(p)
        self.assertEqual(list(m["checkpoint"]["files"]), ["model.safetensors"])

    def test_the_dataset_ranges_ride_along_for_the_skill_listing(self):
        ck = write_checkpoint(self.dir)
        ds = write_dataset(os.path.join(self.dir, "data"))
        m = freeze.build(ck, dataset_path=ds)
        d = m["dataset"]
        self.assertEqual(d["n_episodes"], 50)
        self.assertEqual(d["robot_type"], "so101_follower")
        self.assertEqual(len(d["joints"]), 6)
        self.assertEqual(len(d["ranges"]["observation.state"]["min"]), 6)
        self.assertEqual(len(d["files"]["meta/info.json"]["sha256"]), 64)

    def test_the_manifest_never_promises_success(self):
        ck = write_checkpoint(self.dir)
        text = json.dumps(freeze.build(ck)).lower()
        for forbidden in ("will work", "guarantee", "certified"):
            self.assertNotIn(forbidden, text)

    def test_it_says_a_frozen_checkpoint_is_not_deterministic(self):
        ck = write_checkpoint(self.dir)
        m = freeze.build(ck)
        self.assertIn("stochastic", m["inference"]["note"].lower())


class TestVerify(Temp):
    def test_an_untouched_checkpoint_verifies(self):
        ck = write_checkpoint(self.dir)
        m = freeze.build(ck)
        v = freeze.verify(m, base=self.dir)
        self.assertTrue(v["ok"])
        self.assertEqual(v["changed"], [])
        self.assertEqual(v["missing"], [])

    def test_a_changed_weight_file_is_named(self):
        ck = write_checkpoint(self.dir)
        m = freeze.build(ck)
        with open(os.path.join(ck, "model.safetensors"), "wb") as fh:
            fh.write(b"\x01" * 1024)
        v = freeze.verify(m, base=self.dir)
        self.assertFalse(v["ok"])
        self.assertEqual(v["changed"], ["model.safetensors"])

    def test_a_missing_file_is_named(self):
        ck = write_checkpoint(self.dir)
        m = freeze.build(ck)
        os.remove(os.path.join(ck, "train_config.json"))
        v = freeze.verify(m, base=self.dir)
        self.assertFalse(v["ok"])
        self.assertEqual(v["missing"], ["train_config.json"])

    def test_verify_reads_the_manifest_from_disk(self):
        ck = write_checkpoint(self.dir)
        out = os.path.join(self.dir, "release.json")
        freeze.write(freeze.build(ck), out)
        v = freeze.verify(freeze.load(out), base=self.dir)
        self.assertTrue(v["ok"])


class TestWords(Temp):
    def test_the_report_names_the_cheap_question(self):
        ck = write_checkpoint(self.dir)
        out = freeze.format_freeze(freeze.build(ck), "release.json").lower()
        self.assertIn("sha256", out)
        self.assertIn("eval seeds", out)
        self.assertIn("paired", out)

    def test_json_round_trips(self):
        ck = write_checkpoint(self.dir)
        json.dumps(freeze.build(ck))


if __name__ == "__main__":
    unittest.main()
