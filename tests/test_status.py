"""`orbit status` — where this robot is, and the one thing to do next.

The command exists because the question people actually ask is "why doesn't my
robot work", not "is this difference significant". It answers the first and
leads to the second.
"""

import json
import os
import shutil
import tempfile
import unittest

from orbit_eval import status
from test_dataset import write_dataset, HEALTHY_STATE


def write_policy(root, name="my_policy", steps=100000):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump({"type": "act"}, fh)
    open(os.path.join(d, "model.safetensors"), "w").close()
    with open(os.path.join(d, "train_config.json"), "w") as fh:
        json.dump({"steps": steps, "policy": {"type": "act"}, "seed": 1000}, fh)
    return d


def write_eval(root, name="eval_a", n=10, k=7):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "eval_info.json"), "w") as fh:
        json.dump({"aggregated": {"pc_success": 100.0 * k / n, "n_episodes": n,
                                  "n_success": k}}, fh)
    return d


class Temp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orbit-status-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestStages(Temp):
    def test_an_empty_directory_has_nothing_to_report(self):
        st = status.scan(self.dir)
        self.assertEqual(st.stage, "no_data")
        self.assertIn("record", st.next_action.lower())

    def test_a_dataset_alone_is_the_data_stage(self):
        write_dataset(self.dir)
        st = status.scan(self.dir)
        self.assertEqual(st.stage, "data")
        self.assertEqual(st.dataset.n_episodes, 50)

    def test_a_dataset_plus_a_checkpoint_is_the_trained_stage(self):
        write_dataset(self.dir)
        write_policy(self.dir)
        st = status.scan(self.dir)
        self.assertEqual(st.stage, "trained")
        self.assertEqual(len(st.policies), 1)

    def test_an_eval_result_is_the_evaluated_stage(self):
        write_dataset(self.dir)
        write_policy(self.dir)
        write_eval(self.dir)
        st = status.scan(self.dir)
        self.assertEqual(st.stage, "evaluated")

    def test_a_checkpoint_without_a_dataset_still_counts_as_trained(self):
        write_policy(self.dir)
        st = status.scan(self.dir)
        self.assertEqual(st.stage, "trained")


class TestBlockingComesFirst(Temp):
    def test_a_dead_joint_outranks_the_advice_to_train(self):
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["std"][3] = 0.0
        state["min"][3] = state["max"][3] = 41.7
        write_dataset(self.dir, state=state)
        st = status.scan(self.dir)
        self.assertTrue(st.blocking)
        self.assertIn("wrist_flex.pos", st.next_action)
        self.assertNotIn("train", st.next_action.lower())

    def test_a_clean_dataset_advances_to_training(self):
        write_dataset(self.dir)
        st = status.scan(self.dir)
        self.assertFalse(st.blocking)
        self.assertIn("train", st.next_action.lower())

    def test_a_broken_dataset_blocks_even_after_training(self):
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["std"][0] = 0.0
        state["min"][0] = state["max"][0] = 3.0
        write_dataset(self.dir, state=state)
        write_policy(self.dir)
        st = status.scan(self.dir)
        self.assertTrue(st.blocking)
        self.assertIn("shoulder_pan.pos", st.next_action)


class TestTrialBudget(Temp):
    def test_ten_trials_cannot_see_a_thirty_point_drop(self):
        # The number the whole company exists to say out loud.
        self.assertGreater(status.smallest_callable_pp(10, 50.0), 30.0)

    def test_the_budget_agrees_with_orbit_next(self):
        from orbit_eval import nextstep
        self.assertEqual(status.trials_for(50.0, 20.0),
                         nextstep.episodes_for(50.0, 20.0))

    def test_more_trials_resolve_smaller_drops(self):
        self.assertLess(status.smallest_callable_pp(100, 50.0),
                        status.smallest_callable_pp(10, 50.0))

    def test_a_trained_policy_is_told_what_a_battery_costs(self):
        write_dataset(self.dir)
        write_policy(self.dir)
        st = status.scan(self.dir)
        self.assertIsNotNone(st.trials_needed)
        self.assertGreater(st.trials_needed, 0)
        self.assertIn("trial", status.format_status(st).lower())

    def test_an_underpowered_eval_is_named_as_such(self):
        write_dataset(self.dir)
        write_policy(self.dir)
        write_eval(self.dir, n=10, k=7)
        st = status.scan(self.dir)
        self.assertEqual(st.stage, "evaluated")
        self.assertIsNotNone(st.smallest_callable)
        self.assertGreater(st.smallest_callable, 20.0)


class TestOutput(Temp):
    def test_the_report_names_every_section_it_has_content_for(self):
        write_dataset(self.dir, episodes=12, frames=3000)
        write_policy(self.dir)
        out = status.format_status(status.scan(self.dir))
        self.assertIn("WHERE YOU ARE", out)
        self.assertIn("NEXT ACTION", out)

    def test_broken_datasets_get_a_whats_wrong_section(self):
        state = json.loads(json.dumps(HEALTHY_STATE))
        state["std"][3] = 0.0
        state["min"][3] = state["max"][3] = 41.7
        write_dataset(self.dir, state=state)
        out = status.format_status(status.scan(self.dir))
        self.assertIn("WHAT'S WRONG", out)
        self.assertIn("wrist_flex.pos", out)

    def test_it_never_claims_a_policy_will_work(self):
        write_dataset(self.dir)
        write_policy(self.dir)
        out = status.format_status(status.scan(self.dir)).lower()
        for forbidden in ("will work", "will succeed", "should work", "guaranteed"):
            self.assertNotIn(forbidden, out)

    def test_json_round_trips(self):
        write_dataset(self.dir)
        write_policy(self.dir)
        d = status.scan(self.dir).as_dict()
        json.dumps(d)
        self.assertEqual(d["stage"], "trained")


if __name__ == "__main__":
    unittest.main()
