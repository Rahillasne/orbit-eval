"""The commonest evaluation artifact in public is a single-task eval_info.json.

A single-task `lerobot-eval` writes `per_episode` and `aggregated` and nothing
else: no `per_task`. `orbit status` could read it and `orbit check` could not,
which meant the command most people would try first was blind to the file most
people have.
"""

import json
import os

from orbit_eval import check, discover


def single_task(successes, seed0=1000, videos=None, with_episodes=True, n_success=None):
    n = len(successes)
    k = sum(1 for s in successes if s)
    d = {
        "aggregated": {
            "avg_sum_reward": 10.0, "avg_max_reward": 0.5,
            "pc_success": 100.0 * k / n, "n_episodes": n,
            "eval_s": 3.0, "eval_ep_s": 0.3,
        },
    }
    if n_success is not None:
        d["aggregated"]["n_success"] = n_success
    if with_episodes:
        d["per_episode"] = [
            {"episode_ix": i, "sum_reward": 1.0, "max_reward": 1.0,
             "success": bool(s), "seed": seed0 + i}
            for i, s in enumerate(successes)]
    if videos is not None:
        d["video_paths"] = videos
    return d


def write(tmp, rel, obj):
    p = os.path.join(str(tmp), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(obj, fh)
    return p


def test_per_episode_successes_are_read_as_one_job(tmp_path):
    write(tmp_path, "runA/eval_info.json", single_task([1, 0, 1, 1, 0]))
    srcs = discover.scan(str(tmp_path))
    assert srcs and srcs[0].kind == "lerobot"
    src = srcs[0]
    assert src.candidates == ["runA"]
    assert src.data["runA"] == {discover.SINGLE_TASK: [True, False, True, True, False]}


def test_per_episode_order_is_preserved_so_crn_pairing_can_be_asserted(tmp_path):
    write(tmp_path, "a/eval_info.json", single_task([1, 0, 0, 1]))
    write(tmp_path, "b/eval_info.json", single_task([0, 0, 1, 1]))
    src = discover.scan(str(tmp_path))[0]
    assert src.paired_ok is True
    assert src.data["a"][discover.SINGLE_TASK] == [True, False, False, True]
    assert src.data["b"][discover.SINGLE_TASK] == [False, False, True, True]


def test_top_level_video_paths_follow_the_episodes(tmp_path):
    vids = ["v/eval_episode_0.mp4", "v/eval_episode_1.mp4"]
    write(tmp_path, "a/eval_info.json", single_task([1, 0, 1], videos=vids))
    src = discover.scan(str(tmp_path))[0]
    media = src.media["a"][discover.SINGLE_TASK]
    assert media == vids + [None]


def test_counts_alone_are_expanded_and_never_treated_as_paired(tmp_path):
    # The public Hub shape: pc_success and n_episodes, no per-episode list.
    write(tmp_path, "a/eval_info.json", single_task([1] * 7 + [0] * 3, with_episodes=False))
    write(tmp_path, "b/eval_info.json", single_task([1] * 5 + [0] * 5, with_episodes=False))
    src = discover.scan(str(tmp_path))[0]
    assert sorted(src.data) == ["a", "b"]
    assert sum(src.data["a"][discover.SINGLE_TASK]) == 7
    assert len(src.data["a"][discover.SINGLE_TASK]) == 10
    assert src.paired_ok is False
    assert "counts" in src.note


def test_n_success_is_preferred_over_a_rounded_rate(tmp_path):
    write(tmp_path, "a/eval_info.json",
          single_task([1] * 2 + [0] * 1, with_episodes=False, n_success=2))
    src = discover.scan(str(tmp_path))[0]
    assert src.data["a"][discover.SINGLE_TASK] == [True, True, False]


def test_check_answers_from_two_single_task_files(tmp_path):
    good = [1] * 80 + [0] * 20
    bad = [1] * 40 + [0] * 60
    write(tmp_path, "old/eval_info.json", single_task(good))
    write(tmp_path, "new/eval_info.json", single_task(bad))
    os.utime(os.path.join(str(tmp_path), "old", "eval_info.json"), (1_000_000, 1_000_000))
    res = check.run(str(tmp_path))
    assert res["ok"]
    assert res["incumbent"] == "old" and res["candidate"] == "new"
    rows = res["check"]["rows"]
    assert len(rows) == 1
    assert rows[0]["skill"] == discover.SINGLE_TASK
    assert rows[0]["verdict"] == "REGRESSED"


def test_a_multi_task_file_still_wins_over_the_single_task_reading(tmp_path):
    d = single_task([1, 0, 1])
    d["per_task"] = [{"task_group": "libero", "task_id": 3,
                      "metrics": {"successes": [True, True, False]}}]
    write(tmp_path, "a/eval_info.json", d)
    src = discover.scan(str(tmp_path))[0]
    assert list(src.data["a"]) == ["libero/3"]


def test_a_json_that_is_not_an_evaluation_is_still_not_one(tmp_path):
    write(tmp_path, "config.json", {"n_episodes": 3, "policy": "act"})
    write(tmp_path, "notes.json", {"aggregated": {"note": "nothing here"}})
    assert discover.scan(str(tmp_path)) == []


# ---------------------------------------------------------------- the seeds

def test_matching_per_episode_seeds_verify_crn_without_a_flag(tmp_path):
    # lerobot-eval records the seed of every episode. When two runs carry the
    # same seed at every index, episode k IS the same starting state, and the
    # tool can say so instead of asking the caller to assert it.
    write(tmp_path, "old/eval_info.json", single_task([1] * 60 + [0] * 40, seed0=1000))
    write(tmp_path, "new/eval_info.json", single_task([1] * 50 + [0] * 50, seed0=1000))
    src = discover.scan(str(tmp_path))[0]
    assert src.seeds["old"][discover.SINGLE_TASK][:3] == [1000, 1001, 1002]
    res = check.run(str(tmp_path))
    assert res["crn_verified"] is True
    assert res["crn_asserted"] is True
    assert res["seed_mismatch"] is False
    text = check.format_check(res, str(tmp_path), why=True)
    assert "nobody has asserted" not in text
    assert "seed" in text.lower()


def test_different_seeds_mean_no_episode_level_comparison(tmp_path):
    write(tmp_path, "old/eval_info.json", single_task([1] * 60 + [0] * 40, seed0=1000))
    write(tmp_path, "new/eval_info.json", single_task([1] * 50 + [0] * 50, seed0=100000))
    os.utime(os.path.join(str(tmp_path), "old", "eval_info.json"), (1_000_000, 1_000_000))
    res = check.run(str(tmp_path))
    assert res["crn_verified"] is False
    assert res["seed_mismatch"] is True
    assert res["worklist"] is None
    assert res["crn_gain_pp"] is None
    short = check.format_check(res, str(tmp_path))
    assert "used to work now fail" not in short
    assert "different eval seeds" in short
    assert "--crn" not in short


def test_files_without_seeds_still_need_the_assertion(tmp_path):
    a = single_task([1, 0, 1, 1]); b = single_task([0, 0, 1, 1])
    for d in (a, b):
        for e in d["per_episode"]:
            del e["seed"]
    write(tmp_path, "old/eval_info.json", a)
    write(tmp_path, "new/eval_info.json", b)
    res = check.run(str(tmp_path))
    assert res["crn_verified"] is False
    assert res["seed_mismatch"] is False
    assert res["crn_asserted"] is False
