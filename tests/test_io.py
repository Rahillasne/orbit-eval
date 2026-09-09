"""io parsing against REAL fixtures copied from the ORBIT repo.

  cupid_e1_eval_info.json       LeRobot eval_info, 100 episodes (cupid e1)
  dropgate_full_eval_info.json  LeRobot eval_info, 500 episodes (phase_d)
  paired1_results.json          ORBIT results list, 46 records (paired leg 1)
  model_result_p_b0_m0.json     ORBIT model_result with episodes (leg-1 mask
                                p_b0_m0 episodes + its result record; the
                                recorded set_hash is the wave's own)
"""

import os
import unittest

from orbit_eval import io as oio
from orbit_eval import stats

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


class TestLeRobotEvalInfo(unittest.TestCase):
    def test_cupid_e1(self):
        runs = oio.parse_file(os.path.join(FIX, "cupid_e1_eval_info.json"))
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual(r.fmt, "lerobot")
        self.assertEqual(r.n_eval, 100)
        self.assertEqual(len(r.successes), 100)
        self.assertAlmostEqual(r.sr, 28.0, places=6)
        # pc_success must agree with the per-episode successes
        self.assertEqual(sum(r.successes), 28)
        # stock LeRobot does not record the eval seed
        self.assertIsNone(r.seed)
        self.assertEqual(len(r.sum_rewards), 100)
        self.assertEqual(len(r.max_rewards), 100)
        self.assertIn("pusht", r.task_groups)

    def test_dropgate_500(self):
        runs = oio.parse_file(os.path.join(FIX, "dropgate_full_eval_info.json"))
        r = runs[0]
        self.assertEqual(r.n_eval, 500)
        self.assertEqual(len(r.successes), 500)
        self.assertAlmostEqual(r.sr, 100.0 * sum(r.successes) / 500, places=6)
        self.assertIsNone(r.sr_vector_mismatch)

    def test_partial_successes_vector_is_dropped(self):
        # one per_task block lacks 'successes': the concatenated vector is
        # SHORTER than overall.n_episodes — running McNemar on it against the
        # header's n/SR would be statistics on the wrong episodes. The parser
        # must drop the vector, keep the header, and record why.
        data = {
            "per_task": [
                {"task_group": "pusht", "task_id": 0,
                 "metrics": {"successes": [True, False, True, False]}},
                {"task_group": "pusht", "task_id": 1,
                 "metrics": {"sum_rewards": [1.0, 2.0, 3.0, 4.0]}},
            ],
            "overall": {"pc_success": 50.0, "n_episodes": 8},
        }
        r = oio.parse_eval_info(data)
        self.assertIsNone(r.successes)                 # dropped, not trusted
        self.assertEqual(r.n_eval, 8)                  # header kept
        self.assertAlmostEqual(r.sr, 50.0, places=6)
        self.assertIn("4 entries", r.sr_vector_mismatch)
        self.assertIn("8", r.sr_vector_mismatch)

    def test_header_sr_disagreeing_with_vector_is_dropped(self):
        data = {
            "per_task": [{"task_group": "pusht", "task_id": 0,
                          "metrics": {"successes": [True] * 3 + [False] * 7}}],
            "overall": {"pc_success": 80.0, "n_episodes": 10},
        }
        r = oio.parse_eval_info(data)
        self.assertIsNone(r.successes)
        self.assertAlmostEqual(r.sr, 80.0, places=6)   # header kept, disclosed
        self.assertIn("30.00", r.sr_vector_mismatch)
        self.assertIn("80.00", r.sr_vector_mismatch)

    def test_consistent_file_keeps_its_vector(self):
        data = {
            "per_task": [{"task_group": "pusht", "task_id": 0,
                          "metrics": {"successes": [True] * 3 + [False] * 7}}],
            "overall": {"pc_success": 30.0, "n_episodes": 10},
        }
        r = oio.parse_eval_info(data)
        self.assertEqual(len(r.successes), 10)
        self.assertIsNone(r.sr_vector_mismatch)


class TestOrbitModelResult(unittest.TestCase):
    def test_results_list(self):
        runs = oio.parse_file(os.path.join(FIX, "paired1_results.json"))
        self.assertEqual(len(runs), 46)
        by_id = {r.run_id: r for r in runs}
        r = by_id["p_b0_m0"]
        self.assertEqual(r.fmt, "orbit")
        self.assertEqual(r.sr, 16.0)
        self.assertEqual(r.final_step, 100000)
        self.assertEqual(r.design_steps, 100000)
        self.assertEqual(r.n_eval, 500)
        self.assertEqual(r.seed, 0)
        self.assertEqual(r.set_hash, "1715bd6b98c20238")
        self.assertEqual(r.arm, "anchor")
        self.assertTrue(r.budget_ok)
        # arms present in the real wave
        arms = {r.arm for r in runs}
        self.assertEqual(arms,
                         {"anchor", "replicate", "reseed", "rung_m4", "rung_m16"})

    def test_single_model_result_with_episodes(self):
        runs = oio.parse_file(os.path.join(FIX, "model_result_p_b0_m0.json"))
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual(len(r.episodes), 103)
        # our set_hash implementation must reproduce the wave's recorded hash
        self.assertEqual(stats.set_hash(r.episodes), "1715bd6b98c20238")
        self.assertEqual(r.set_hash, "1715bd6b98c20238")

    def test_set_hash_computed_when_absent(self):
        import json
        with open(os.path.join(FIX, "model_result_p_b0_m0.json")) as fh:
            rec = json.load(fh)
        del rec["set_hash"]
        r = oio.parse_model_result(rec)
        self.assertEqual(r.set_hash, "1715bd6b98c20238")


class TestDiscovery(unittest.TestCase):
    def test_discover_fixture_dir(self):
        found = {os.path.basename(p): runs
                 for p, runs, err in oio.discover(FIX) if runs}
        # ALL four fixture files must be discovered, including the
        # LeRobot-style *_eval_info.json suffixed names — silently skipping
        # eval outputs is the exact failure mode audit exists to catch
        self.assertEqual(set(found), {"cupid_e1_eval_info.json",
                                      "dropgate_full_eval_info.json",
                                      "paired1_results.json",
                                      "model_result_p_b0_m0.json"})
        self.assertEqual(len(found["paired1_results.json"]), 46)
        self.assertEqual(len(found["cupid_e1_eval_info.json"]), 1)
        self.assertEqual(len(found["dropgate_full_eval_info.json"]), 1)

    def test_load_single_rejects_many(self):
        with self.assertRaises(ValueError):
            oio.load_single(os.path.join(FIX, "paired1_results.json"))

    def test_malformed_json_error_names_the_file(self):
        import tempfile
        fd, p = tempfile.mkstemp(suffix="_eval_info.json")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write("{bad json")
            with self.assertRaises(ValueError) as cm:
                oio.parse_file(p)
            self.assertIn(p, str(cm.exception))
            self.assertIn("invalid JSON", str(cm.exception))
        finally:
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()
