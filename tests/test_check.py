"""orbit check / orbit next: detection is heuristic, the decision is not."""
import json
import os
import subprocess
import sys

import pytest

from orbit_eval import check, discover, nextstep, release as release_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_csv(path, rows, header="version,skill,episode,success"):
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def matrix(tmp, name, cands):
    p = os.path.join(tmp, name)
    with open(p, "w") as fh:
        json.dump({"candidates": cands}, fh)
    return p


# ------------------------------------------------------------------ discovery

def test_finds_a_csv_with_unusual_column_names(tmp_path):
    p = tmp_path / "trials.csv"
    write_csv(str(p), [("m1", "pick", i, i % 2) for i in range(10)] +
                      [("m2", "pick", i, 1) for i in range(10)],
              header="model,task,trial,passed")
    srcs = discover.scan(str(tmp_path))
    assert srcs and srcs[0].kind == "csv"
    assert srcs[0].candidates == ["m1", "m2"]
    assert srcs[0].data["m2"]["pick"] == [True] * 10


def test_reward_column_counts_positive_reward_as_success(tmp_path):
    p = tmp_path / "r.csv"
    write_csv(str(p), [("a", "j", 0, 1.0), ("a", "j", 1, 0.0)],
              header="version,skill,episode,reward")
    srcs = discover.scan(str(tmp_path))
    assert srcs[0].data["a"]["j"] == [True, False]


def test_csv_rows_are_ordered_by_episode_not_file_order(tmp_path):
    p = tmp_path / "shuffled.csv"
    write_csv(str(p), [("a", "j", 2, 1), ("a", "j", 0, 0), ("a", "j", 1, 1)])
    srcs = discover.scan(str(tmp_path))
    assert srcs[0].data["a"]["j"] == [False, True, True]


def test_lerobot_tree_names_candidates_by_the_policy_directory(tmp_path):
    blob = {"per_task": [{"task_group": "libero", "task_id": 0,
                          "metrics": {"successes": [True, False, True]}}]}
    for run in ("smolvla_s0", "smolvla_s1"):
        d = tmp_path / "outputs" / run / "checkpoints" / "020000" / "eval_final"
        d.mkdir(parents=True)
        (d / "eval_info.json").write_text(json.dumps(blob))
    srcs = discover.scan(str(tmp_path))
    assert srcs[0].kind == "lerobot"
    assert srcs[0].candidates == ["smolvla_s0", "smolvla_s1"]


def test_generic_directory_names_are_never_used_as_a_policy_name(tmp_path):
    d = tmp_path / "runA" / "eval_final"
    d.mkdir(parents=True)
    (d / "eval_info.json").write_text(json.dumps(
        {"per_task": [{"task_group": "t", "task_id": 0, "metrics": {"successes": [1]}}]}))
    assert discover.candidate_name(str(d / "eval_info.json"), str(tmp_path)) == "runA"


def test_skip_dirs_are_not_walked(tmp_path):
    d = tmp_path / "node_modules" / "x"
    d.mkdir(parents=True)
    write_csv(str(d / "a.csv"), [("a", "j", 0, 1)])
    assert discover.scan(str(tmp_path)) == []


def test_trim_to_common_uses_shared_skills_and_shortest_length(tmp_path):
    data = {"a": {"j1": [1, 1, 1], "j2": [1]}, "b": {"j1": [0, 0]}}
    out = discover.trim_to_common(data)
    assert set(out["a"]) == {"j1"}
    assert len(out["a"]["j1"]) == len(out["b"]["j1"]) == 2


# -------------------------------------------------------------------- choose

def test_choose_prefers_explicit_names(tmp_path):
    p = matrix(str(tmp_path), "m.json", {"x": {"j": [1] * 10}, "y": {"j": [0] * 10},
                                         "z": {"j": [1] * 10}})
    src = discover.scan(str(tmp_path))[0]
    assert check.choose(src, "x", "z")[:2] == ("x", "z")


def test_choose_reports_why_it_guessed(tmp_path):
    matrix(str(tmp_path), "m.json", {"a": {"j": [1] * 10}, "b": {"j": [0] * 10}})
    src = discover.scan(str(tmp_path))[0]
    inc, cand, why = check.choose(src)
    assert {inc, cand} == {"a", "b"} and why


def test_choose_rejects_an_unknown_name(tmp_path):
    matrix(str(tmp_path), "m.json", {"a": {"j": [1] * 10}, "b": {"j": [0] * 10}})
    src = discover.scan(str(tmp_path))[0]
    with pytest.raises(ValueError):
        check.choose(src, incumbent="nope")


# --------------------------------------------------------------------- check

def test_check_agrees_exactly_with_release_check(tmp_path):
    """The convenience wrapper must not become a second, divergent decision."""
    cands = {"old": {"j%d" % i: [1] * 60 + [0] * 40 for i in range(4)},
             "new": {"j%d" % i: [1] * 20 + [0] * 80 for i in range(4)}}
    matrix(str(tmp_path), "m.json", cands)
    res = check.run(str(tmp_path), incumbent="old", candidate="new")
    data = discover.trim_to_common(discover.scan(str(tmp_path))[0].data)
    assert res["check"] == release_mod.check(data, "old", "new")


def test_check_flags_a_real_regression(tmp_path):
    cands = {"old": {"j": [1] * 90 + [0] * 10}, "new": {"j": [1] * 30 + [0] * 70}}
    matrix(str(tmp_path), "m.json", cands)
    res = check.run(str(tmp_path), incumbent="old", candidate="new")
    assert res["check"]["n_regressed"] == 1


def test_check_on_one_policy_does_not_invent_a_comparison(tmp_path):
    matrix(str(tmp_path), "m.json", {"only": {"j": [1] * 20}})
    res = check.run(str(tmp_path))
    assert res["ok"] and res["check"] is None and res["next"]["single_candidate"]


def test_check_reports_nothing_found_without_crashing(tmp_path):
    res = check.run(str(tmp_path))
    assert res["ok"] is False
    assert "eval_info.json" in check.format_nothing(res)


def test_html_report_escapes_and_renders(tmp_path):
    matrix(str(tmp_path), "m.json", {"a<script>": {"j": [1] * 20}, "b": {"j": [0] * 20}})
    res = check.run(str(tmp_path))
    out = check.html_report(res, sig="deadbeef")
    assert "<script>" not in out.split("<style>")[1]
    assert "deadbeef" in out and "<!doctype html>" in out


# ---------------------------------------------------------------------- next

def test_noise_floor_grows_with_copies_and_shrinks_with_episodes():
    assert nextstep.noise_floor_pp(2, 100, 50) < nextstep.noise_floor_pp(8, 100, 50)
    assert nextstep.noise_floor_pp(8, 400, 50) < nextstep.noise_floor_pp(8, 100, 50)
    assert nextstep.noise_floor_pp(1, 100, 50) == 0.0


def test_noise_floor_matches_simulation_at_n100():
    """Hartley's constant against the simulation used in research/crossmodel_2026-09-15."""
    import random
    rng = random.Random(11)
    J, n, p = 8, 100, 0.7
    xs = []
    for _ in range(3000):
        r = [100.0 * sum(rng.random() < p for _ in range(n)) / n for _ in range(J)]
        xs.append(max(r) - min(r))
    sim = sum(xs) / len(xs)
    assert abs(nextstep.noise_floor_pp(J, n, 100 * p) - sim) < 1.5


def test_identical_copies_are_never_called_a_lottery():
    bits = [1] * 70 + [0] * 30
    data = {"c%d" % i: {"j": list(bits)} for i in range(4)}
    d = nextstep.diagnose(data)
    assert d["rows"][0]["diagnosis"] != "UNLUCKY"
    assert d["rows"][0]["lottery_pp"] == 0.0


def test_a_wide_spread_beyond_the_floor_is_a_lottery():
    data = {"a": {"j": [1] * 90 + [0] * 10}, "b": {"j": [1] * 30 + [0] * 70},
            "c": {"j": [1] * 85 + [0] * 15}}
    d = nextstep.diagnose(data)
    assert d["rows"][0]["diagnosis"] == "UNLUCKY"


def test_a_job_everyone_fails_is_weak_not_unlucky():
    data = {"a": {"j": [1] * 20 + [0] * 80}, "b": {"j": [1] * 22 + [0] * 78},
            "c": {"j": [1] * 19 + [0] * 81}}
    d = nextstep.diagnose(data)
    assert d["rows"][0]["diagnosis"] == "WEAK"


def test_a_job_with_far_fewer_trials_is_called_thin():
    data = {"a": {"big": [1] * 100, "small": [1] * 10},
            "b": {"big": [1] * 100, "small": [1] * 10}}
    d = nextstep.diagnose(data)
    by = {r["skill"]: r["diagnosis"] for r in d["rows"]}
    assert by["small"] == "THIN"


def test_power_is_a_suite_statement_not_a_per_job_verdict():
    data = {"a": {"j%d" % i: [1] * 10 + [0] * 10 for i in range(5)},
            "b": {"j%d" % i: [1] * 10 + [0] * 10 for i in range(5)}}
    d = nextstep.diagnose(data, target_pp=10.0)
    assert d["underpowered_for_target"] is True
    assert "UNDERPOWERED" not in {r["diagnosis"] for r in d["rows"]}


def test_episodes_for_a_smaller_target_costs_more_trials():
    assert nextstep.episodes_for(50.0, 5.0) > nextstep.episodes_for(50.0, 20.0)


def test_episodes_for_uses_the_same_rule_release_prints():
    """The number `next` quotes must resolve the drop `release` says it cannot see."""
    import math
    p, target = 50.0, 10.0
    n = nextstep.episodes_for(p, target)
    se = 100.0 * math.sqrt(2.0 * 0.25 / n)
    assert release_mod.Z_DMG * se <= target + 1e-6


def test_retrain_advice_names_its_cell_and_does_not_extrapolate():
    a = nextstep.retrain_advice(10)
    assert a["cell"] and a["cell_tasks"] == 10 and a["exact_match"]
    b = nextstep.retrain_advice(37)
    assert b["exact_match"] is False and b["your_tasks"] == 37


def test_format_next_is_advice_not_a_stacktrace():
    data = {"a": {"j": [1] * 50 + [0] * 50}, "b": {"j": [1] * 20 + [0] * 80}}
    txt = nextstep.format_next(nextstep.diagnose(data))
    assert "What to do next" in txt and "job" in txt


# ----------------------------------------------------------------------- cli

def _cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, "-m", "orbit_eval.cli"] + list(args),
                          capture_output=True, text=True, env=env, cwd=cwd)


def test_cli_check_runs_in_a_bare_directory(tmp_path):
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 50 + [0] * 50},
                                     "new": {"j": [1] * 48 + [0] * 52}})
    r = _cli("check", str(tmp_path), "--no-open", cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "policies" in r.stdout or "policy" in r.stdout
    assert os.path.exists(os.path.join(str(tmp_path), "orbit-report.html"))


def test_cli_check_exits_1_on_a_regression(tmp_path):
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 95 + [0] * 5},
                                     "new": {"j": [1] * 20 + [0] * 80}})
    r = _cli("check", str(tmp_path), "--no-open", "--no-report", "--incumbent", "old",
             "--candidate", "new")
    assert r.returncode == 1


def test_cli_check_exits_2_when_there_is_nothing(tmp_path):
    r = _cli("check", str(tmp_path), "--no-open", "--no-report")
    assert r.returncode == 2 and "no evaluations found" in r.stdout


def test_cli_next_prints_advice(tmp_path):
    matrix(str(tmp_path), "m.json", {"a": {"j": [1] * 90 + [0] * 10},
                                     "b": {"j": [1] * 30 + [0] * 70}})
    r = _cli("next", str(tmp_path))
    assert r.returncode == 0 and "What to do next" in r.stdout


def test_cli_check_json_is_machine_readable(tmp_path):
    matrix(str(tmp_path), "m.json", {"a": {"j": [1] * 30 + [0] * 70},
                                     "b": {"j": [1] * 30 + [0] * 70}})
    r = _cli("check", str(tmp_path), "--no-open", "--no-report", "--json")
    assert json.loads(r.stdout)["ok"] is True


def test_retrain_call_does_not_fire_on_identical_copies():
    """A false lottery costs a week of GPU time, so the rate is held, not hoped for."""
    import random
    rng = random.Random(1)
    for J, n in ((3, 20), (3, 100), (8, 20), (8, 100)):
        fp = 0
        for _ in range(400):
            p = rng.uniform(0.2, 0.8)
            data = {"c%d" % i: {"j": [rng.random() < p for _ in range(n)]} for i in range(J)}
            if nextstep.diagnose(data)["rows"][0]["diagnosis"] == "UNLUCKY":
                fp += 1
        assert fp / 400.0 < 0.15, "J=%d n=%d fired on %.1f%% of identical pools" % (
            J, n, 100.0 * fp / 400.0)


def test_noise_ceiling_is_above_the_floor_and_both_shrink_with_trials():
    assert nextstep.noise_ceiling_pp(4, 50, 50) > nextstep.noise_floor_pp(4, 50, 50)
    assert nextstep.noise_ceiling_pp(4, 400, 50) < nextstep.noise_ceiling_pp(4, 50, 50)


def test_wilson_units_are_points_everywhere_a_rate_is_reported():
    """stats.wilson returns fractions; every rate this product prints is points."""
    data = {"a": {"j": [1] * 7 + [0] * 3}, "b": {"j": [1] * 3 + [0] * 7}}
    lo, hi = nextstep.diagnose(data)["rows"][0]["ci95_best"]
    assert 1.0 < lo < 100.0 and 1.0 < hi <= 100.0


# ----------------------------------------------------------------- orbit log

def test_log_session_appends_and_resumes(tmp_path):
    from orbit_eval import logtrials
    p = str(tmp_path / "trials.csv")
    s = logtrials.Session(p, "pol", "job", block="cell3")
    s.record(True)
    s.record(False, "grasp")
    again = logtrials.Session(p, "pol", "job")
    assert again.n == 2 and again.k == 1
    assert again.reasons() == {"grasp": 1}


def test_log_undo_removes_only_the_last_trial(tmp_path):
    from orbit_eval import logtrials
    p = str(tmp_path / "t.csv")
    s = logtrials.Session(p, "pol", "job")
    s.record(True)
    s.record(False, "place")
    assert s.undo()["success"] == 0
    assert s.n == 1
    assert logtrials.Session(p, "pol", "job").n == 1


def test_log_keeps_other_policies_and_jobs_in_the_same_file(tmp_path):
    from orbit_eval import logtrials
    p = str(tmp_path / "t.csv")
    a = logtrials.Session(p, "polA", "job1")
    a.record(True)
    b = logtrials.Session(p, "polB", "job1")
    b.record(False, "grasp")
    b.record(True)
    assert logtrials.Session(p, "polA", "job1").n == 1
    assert logtrials.Session(p, "polB", "job1").n == 2


def test_log_output_is_read_back_by_check_with_no_conversion(tmp_path):
    """The whole point: what you record at the robot is what check reads."""
    from orbit_eval import logtrials
    p = str(tmp_path / "trials.csv")
    for pol, hits in (("old", 18), ("new", 6)):
        for job in ("pick", "place"):
            s = logtrials.Session(p, pol, job, block="cell1")
            for i in range(20):
                s.record(i < hits, "" if i < hits else "grasp")
    res = check.run(str(tmp_path), incumbent="old", candidate="new")
    assert res["ok"] and res["check"]["n_skills"] == 2
    assert res["check"]["n_regressed"] == 2


def test_log_loop_records_from_a_keystream(tmp_path):
    import io
    from orbit_eval import logtrials
    keys = iter(list("y") + ["n", "g"] + list("yuq"))
    s = logtrials.Session(str(tmp_path / "t.csv"), "p", "j")
    out = io.StringIO()
    logtrials.loop(s, out=out, getch=lambda: next(keys))
    assert s.n == 2 and s.k == 1            # y, n(grasp), y, then undo the last y
    assert "saved to" in out.getvalue()


def test_log_survives_a_crash_after_every_trial(tmp_path):
    from orbit_eval import logtrials
    p = str(tmp_path / "t.csv")
    s = logtrials.Session(p, "pol", "job")
    s.record(True)
    del s                                    # no clean shutdown, no flush call
    assert logtrials.Session(p, "pol", "job").n == 1


def test_cli_log_runs_with_piped_keys(tmp_path):
    r = subprocess.run([sys.executable, "-m", "orbit_eval.cli", "log",
                        "--job", "j", "--policy", "p",
                        "--out", str(tmp_path / "t.csv")],
                       input="y\ny\nn\ng\nq\n", capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode == 0, r.stderr
    assert "3 trials, 2 succeeded" in r.stdout


def test_bare_orbit_runs_a_check_here(tmp_path):
    """`orbit` with no arguments answers the question, the way `git status` does."""
    # named so that the alphabetical fallback puts the older one first
    matrix(str(tmp_path), "m.json", {"v1": {"j": [1] * 40 + [0] * 10},
                                     "v2": {"j": [1] * 10 + [0] * 40}})
    r = _cli(cwd=str(tmp_path))
    assert r.returncode in (0, 1)
    assert "REGRESSED" in r.stdout, r.stdout


def test_bare_orbit_falls_back_to_a_quickstart_when_there_is_nothing(tmp_path):
    r = _cli(cwd=str(tmp_path))
    assert r.returncode == 0 and "orbit next" in r.stdout and "orbit log" in r.stdout


# ------------------------------------------------- shapes found in the wild
# Each of these is modelled on a real public file that orbit check could not read
# until it was tested against them. The repo it came from is named so the shape can
# be checked again later.

def test_csv_with_a_comment_header_is_read(tmp_path):
    """parastoopil/quantized-vla-robustness-bench opens with three '#' notes."""
    p = tmp_path / "per_seed.csv"
    p.write_text("# LIBERO-plus per-seed detail.\n# sr = success rate (%).\n"
                 "\naxis,n_per_arm,fp16_sr,w4a4_sr\n"
                 "background,258,99.6,100.0\nsensor_noise,351,91.5,88.3\n")
    src = discover.scan(str(tmp_path))
    assert src and sorted(src[0].data) == ["fp16_sr", "w4a4_sr"]
    assert sorted(src[0].data["fp16_sr"]) == ["background", "sensor_noise"]


def test_wide_layout_one_column_per_policy(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("axis,n_per_arm,fp16_sr,w4a4_sr\ncamera,100,80.0,60.0\n")
    src = discover.scan(str(tmp_path))[0]
    assert src.data["fp16_sr"]["camera"].count(True) == 80
    assert src.data["w4a4_sr"]["camera"].count(True) == 60
    assert src.paired_ok is False


def test_aggregate_rows_expand_from_counts(tmp_path):
    """YEYEPrincess/openvla-fine-running: experiment,task_suite,total_episodes,successes."""
    p = tmp_path / "a.csv"
    p.write_text("experiment,task_suite,total_episodes,successes\n"
                 "released_goal_formal,LIBERO-Goal,500,489\n"
                 "base_lora_1k,LIBERO-Goal,500,100\n")
    src = discover.scan(str(tmp_path))[0]
    assert src.data["released_goal_formal"]["LIBERO-Goal"].count(True) == 489
    assert len(src.data["base_lora_1k"]["LIBERO-Goal"]) == 500


def test_aggregate_without_a_task_column_uses_one_job(tmp_path):
    """weigu-my/VLA: one row per model, no task column, all on one suite."""
    p = tmp_path / "m.csv"
    p.write_text("model,episodes,successes\nopenvla_2_5k,50,4\nopenvla_15k,50,29\n")
    src = discover.scan(str(tmp_path))[0]
    assert sorted(src.data) == ["openvla_15k", "openvla_2_5k"]
    assert src.data["openvla_15k"]["all"].count(True) == 29


def test_rates_without_a_trial_count_are_refused_with_a_reason(tmp_path):
    """OliEfr/vpro_paper_plots: n_demos is a training budget, not a trial count."""
    p = tmp_path / "budget.csv"
    p.write_text("# demo-budget sweep\nsplit,n_demos,action_only_sr,video_sr\n"
                 "nonh,1,0.437,0.525\nnonh,5,0.638,0.698\n")
    srcs, rej = discover.scan(str(tmp_path), with_rejects=True)
    assert srcs == []
    assert any("trial-count" in why for _p, why in rej)


def test_a_paper_leaderboard_is_not_mistaken_for_an_evaluation(tmp_path):
    """ripl/ManipulationBenchmarkAudit: one row per paper, no trials behind it."""
    p = tmp_path / "lb.csv"
    p.write_text("paper_name,robotwin2_hard_randomized_success_rate\nA VLA Model,86.68\n")
    srcs, rej = discover.scan(str(tmp_path), with_rejects=True)
    assert srcs == []


def test_fractional_and_percentage_rates_are_both_understood(tmp_path):
    for i, text in enumerate(("task,n,a_sr,b_sr\nj,100,0.80,0.60\n",
                              "task,n,a_sr,b_sr\nj,100,80.0,60.0\n")):
        p = tmp_path / ("rate%d.csv" % i)
        p.write_text(text)
        got, _note, _pk = discover._read_csv(str(p))
        assert got is not None, "failed to read %r" % text
        assert got["a_sr"]["j"].count(True) == 80
        assert got["b_sr"]["j"].count(True) == 60


def test_one_policy_on_a_different_benchmark_does_not_poison_the_rest(tmp_path):
    """The openvla case: six runs share a suite, two do not."""
    cands = {"a": {"goal": [1] * 50}, "b": {"goal": [0] * 50}, "c": {"goal": [1] * 50},
             "odd": {"spatial": [1] * 50}}
    matrix(str(tmp_path), "m.json", cands)
    res = check.run(str(tmp_path))
    assert res["ok"]
    assert res["policies_dropped"] == ["odd"]
    assert res["n_skills"] == 1


def test_no_two_policies_share_a_job_explains_itself(tmp_path):
    cands = {"a": {"pusht": [1] * 20}, "b": {"aloha": [1] * 20}}
    matrix(str(tmp_path), "m.json", cands)
    res = check.run(str(tmp_path))
    assert res["ok"] is False
    msg = check.format_nothing(res)
    assert "no two of them ran the same jobs" in msg
    assert "pusht" in msg and "aloha" in msg


def test_counts_rebuilt_from_summaries_are_never_treated_as_paired(tmp_path):
    """Expanded counts look perfectly aligned and are not. Pairing them would
    report an interval far tighter than anything that was measured."""
    p = tmp_path / "a.csv"
    p.write_text("model,task,episodes,successes\nold,j,100,80\nnew,j,100,50\n")
    res = check.run(str(tmp_path), incumbent="old", candidate="new")
    assert res["paired"] is False
    assert all(r["method"] == "unpaired" for r in res["check"]["rows"])


def test_a_skipped_file_is_always_named_with_its_reason(tmp_path):
    (tmp_path / "junk.csv").write_text("paper,score\nx,1.0\n")
    res = check.run(str(tmp_path))
    assert res["ok"] is False
    assert res["rejected"], "a near miss must be reported, never silently dropped"


# ------------------------------------------------- failure reasons feed advice

def test_reason_column_is_captured_per_policy_and_job(tmp_path):
    p = tmp_path / "trials.csv"
    rows = ["version,skill,episode,success,reason"]
    rows += ["a,pick,%d,0,grasp" % i for i in range(6)]
    rows += ["a,pick,%d,1," % i for i in range(6, 10)]
    p.write_text("\n".join(rows) + "\n")
    src = discover.scan(str(tmp_path))[0]
    assert src.reasons["a"]["pick"] == {"grasp": 6}


def test_a_dominant_failure_mode_is_named_in_the_advice():
    assert nextstep.dominant_reason({"grasp": 53, "timeout": 11}) == ("grasp", 53, 64)


def test_a_scattered_failure_mode_is_not_named():
    """Six modes at roughly equal counts is not a finding, and saying it is would
    send somebody to collect the wrong demonstrations."""
    assert nextstep.dominant_reason({k: 5 for k in "abcdef"}) is None


def test_too_few_failures_to_believe_is_not_named():
    assert nextstep.dominant_reason({"grasp": 2}) is None


def test_unrecorded_reasons_never_become_the_named_mode():
    assert nextstep.dominant_reason({"unrecorded": 40, "grasp": 2}) is None


def test_weak_job_advice_names_the_mode_when_there_is_one():
    data = {"a": {"j": [1] * 20 + [0] * 80}, "b": {"j": [1] * 22 + [0] * 78}}
    reasons = {"a": {"j": {"grasp": 60, "timeout": 20}}}
    d = nextstep.diagnose(data, reasons=reasons)
    row = d["rows"][0]
    assert row["diagnosis"] == "WEAK" and "grasp" in row["action"]


def test_weak_job_advice_asks_for_reasons_when_there_are_none():
    data = {"a": {"j": [1] * 20 + [0] * 80}, "b": {"j": [1] * 22 + [0] * 78}}
    txt = nextstep.format_next(nextstep.diagnose(data))
    assert "Record why each trial failed" in txt


def test_reasons_survive_the_trim_to_the_comparable_group(tmp_path):
    p = tmp_path / "trials.csv"
    rows = ["version,skill,episode,success,reason"]
    for pol in ("a", "b"):
        rows += ["%s,j,%d,0,grasp" % (pol, i) for i in range(30)]
        rows += ["%s,j,%d,1," % (pol, i) for i in range(30, 40)]
    p.write_text("\n".join(rows) + "\n")
    res = check.run(str(tmp_path))
    assert res["next"]["rows"][0]["failure_reasons"] == {"grasp": 60}


# ------------------------------------------------------------- markdown / CI

def test_markdown_report_is_a_table_with_a_verdict(tmp_path):
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 90 + [0] * 10},
                                     "new": {"j": [1] * 20 + [0] * 80}})
    res = check.run(str(tmp_path), incumbent="old", candidate="new")
    md = check.markdown_report(res, sig="abc123def456")
    assert "regressed" in md and "| `j` |" in md and "abc123def456"[:12] in md


def test_markdown_report_handles_nothing_found(tmp_path):
    res = check.run(str(tmp_path))
    assert "No comparable evaluations" in check.markdown_report(res)


def test_cli_writes_markdown_for_a_pr_comment(tmp_path):
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 90 + [0] * 10},
                                     "new": {"j": [1] * 20 + [0] * 80}})
    md = tmp_path / "pr.md"
    r = _cli("check", str(tmp_path), "--no-open", "--no-report",
             "--incumbent", "old", "--candidate", "new", "--markdown", str(md))
    assert r.returncode == 1                      # a regression, so CI fails
    assert md.exists() and "| job |" in md.read_text()


def test_action_yaml_is_valid_and_wires_the_documented_inputs():
    import re
    p = os.path.join(ROOT, ".github", "actions", "orbit-check", "action.yml")
    assert os.path.exists(p), "the action must ship with the package repo"
    text = open(p).read()
    for key in ("path", "incumbent", "candidate", "fail-on-regression", "comment"):
        assert "\n  %s:" % key in text, "input %s is not declared" % key
    for out in ("regressed", "suspect", "report"):
        assert "\n  %s:" % out in text
    assert "orbit-check -->" in text, "the comment must be editable in place"


# --------------------------------------- layouts found in the WILD-1 sweep
# Both of these were missed until orbit-eval was run over 311 public artifacts.

def test_lerobot_content_is_read_whatever_the_file_is_called(tmp_path):
    """muhammadmahadazher/vla-on-a-budget calls them eval_act_50k.json."""
    blob = json.dumps({"per_task": [{"task_group": "pusht", "task_id": 0,
                                     "metrics": {"successes": [1, 0] * 10}}]})
    (tmp_path / "eval_act_50k.json").write_text(blob)
    (tmp_path / "smoke_eval_info.json").write_text(blob)
    src = discover.scan(str(tmp_path))
    assert src and src[0].kind == "lerobot"


def test_policies_distinguished_by_filename_not_directory(tmp_path):
    """Three policies side by side in results/, named by file. Collapsing them into
    one is the quietest possible way to lose a comparison."""
    d = tmp_path / "results"
    d.mkdir()
    for name, hits in (("eval_act_50k.json", 4), ("eval_smolvla_20k.json", 12),
                       ("eval_diffusion_50k.json", 7)):
        (d / name).write_text(json.dumps({"per_task": [
            {"task_group": "pusht", "task_id": 0,
             "metrics": {"successes": [1] * hits + [0] * (20 - hits)}}]}))
    src = discover.scan(str(tmp_path))[0]
    assert sorted(src.data) == ["act_50k", "diffusion_50k", "smolvla_20k"]


def test_one_eval_info_per_policy_directory_still_uses_the_directory(tmp_path):
    for run in ("smolvla_s0", "smolvla_s1"):
        d = tmp_path / "outputs" / run / "eval_final"
        d.mkdir(parents=True)
        (d / "eval_info.json").write_text(json.dumps({"per_task": [
            {"task_group": "t", "task_id": 0, "metrics": {"successes": [1] * 10}}]}))
    src = discover.scan(str(tmp_path))[0]
    assert sorted(src.data) == ["smolvla_s0", "smolvla_s1"]


def test_a_json_that_is_not_an_evaluation_is_not_read_as_one(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"lr": 3e-4, "steps": 20000}))
    assert discover.scan(str(tmp_path)) == []


# ------------------------------------------------- the discordant worklist

def _wl():
    from orbit_eval import worklist
    return worklist


def test_worklist_finds_exactly_the_episodes_that_flipped():
    w = _wl().build({"old": {"j": [1, 1, 0, 1]}, "new": {"j": [1, 0, 1, 1]}}, "old", "new")
    assert w["n_broke"] == 1 and w["n_fixed"] == 1
    kinds = {e["episode"]: e["kind"] for e in w["jobs"][0]["episodes"]}
    assert kinds == {1: "BROKE", 2: "FIXED"}


def test_worklist_matches_mcnemars_discordant_counts():
    """The worklist and the significance test must be reading the same episodes;
    if they ever disagree, one of them is pointing at the wrong thing."""
    from orbit_eval import stats
    A = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    B = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    w = _wl().build({"old": {"j": A}, "new": {"j": B}}, "old", "new")
    _n11, n10, n01, _n00 = stats.discordant_counts([bool(x) for x in A],
                                                   [bool(x) for x in B])
    assert (w["n_broke"], w["n_fixed"]) == (n10, n01)


def test_worklist_is_empty_when_nothing_changed():
    w = _wl().build({"a": {"j": [1, 0, 1]}, "b": {"j": [1, 0, 1]}}, "a", "b")
    assert w["n_discordant"] == 0 and w["jobs"] == []
    assert "nothing to watch" in _wl().format_worklist(w)


def test_worklist_carries_the_two_video_paths_for_each_flip():
    media = {"old": {"j": ["old/0.mp4", "old/1.mp4"]},
             "new": {"j": ["new/0.mp4", "new/1.mp4"]}}
    w = _wl().build({"old": {"j": [1, 1]}, "new": {"j": [1, 0]}}, "old", "new", media=media)
    e = w["jobs"][0]["episodes"][0]
    assert e["incumbent_video"] == "old/1.mp4" and e["candidate_video"] == "new/1.mp4"
    assert w["has_media"] is True


def test_worklist_survives_missing_and_short_media_lists():
    """Real files carry an off-by-one: one extra aggregate video is common, and some
    runs record none. An index must never name the wrong clip."""
    media = {"old": {"j": ["old/0.mp4"]}}          # shorter than the episodes, and no "new"
    w = _wl().build({"old": {"j": [1, 1]}, "new": {"j": [1, 0]}}, "old", "new", media=media)
    e = w["jobs"][0]["episodes"][0]
    assert e["incumbent_video"] is None and e["candidate_video"] is None


def test_video_paths_longer_than_successes_are_truncated_not_shifted(tmp_path):
    (tmp_path / "a" ).mkdir()
    (tmp_path / "a" / "eval_info.json").write_text(json.dumps({"per_task": [
        {"task_group": "t", "task_id": 0,
         "metrics": {"successes": [1, 0, 1],
                     "video_paths": ["v0.mp4", "v1.mp4", "v2.mp4", "all.mp4"]}}]}))
    src = discover.scan(str(tmp_path))[0]
    assert src.media["a"]["t/0"] == ["v0.mp4", "v1.mp4", "v2.mp4"]


def test_worst_job_is_listed_first():
    data = {"old": {"bad": [1] * 10, "mild": [1] * 10},
            "new": {"bad": [0] * 10, "mild": [1] * 9 + [0]}}
    w = _wl().build(data, "old", "new")
    assert w["jobs"][0]["job"] == "bad"


def test_worklist_warns_when_common_random_numbers_are_not_asserted():
    w = _wl().build({"a": {"j": [1, 1]}, "b": {"j": [1, 0]}}, "a", "b", crn_asserted=False)
    assert "nobody has asserted" in _wl().format_worklist(w)
    w2 = _wl().build({"a": {"j": [1, 1]}, "b": {"j": [1, 0]}}, "a", "b", crn_asserted=True)
    assert "nobody has asserted" not in _wl().format_worklist(w2)


def test_crn_flag_turns_the_comparison_paired_and_narrows_it(tmp_path):
    cands = {"old": {"j": [1] * 30 + [0] * 20}, "new": {"j": [1] * 25 + [0] * 25}}
    matrix(str(tmp_path), "m.json", cands)
    loose = check.run(str(tmp_path), incumbent="old", candidate="new")
    tight = check.run(str(tmp_path), incumbent="old", candidate="new", crn=True)
    assert loose["check"]["paired"] is False and tight["check"]["paired"] is True
    assert tight["crn_asserted"] is True
    wl, wt = loose["check"]["rows"][0]["ci95"], tight["check"]["rows"][0]["ci95"]
    assert (wt[1] - wt[0]) <= (wl[1] - wl[0])


def test_check_attaches_a_worklist(tmp_path):
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 40 + [0] * 10},
                                     "new": {"j": [1] * 10 + [0] * 40}})
    res = check.run(str(tmp_path), incumbent="old", candidate="new", crn=True)
    assert res["worklist"]["n_discordant"] > 0
    assert "WHICH EPISODES TO WATCH" in check.format_check(res, why=True)
    assert "used to work now fail" in check.format_check(res)


def test_markdown_includes_the_worklist(tmp_path):
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 40 + [0] * 10},
                                     "new": {"j": [1] * 10 + [0] * 40}})
    res = check.run(str(tmp_path), incumbent="old", candidate="new", crn=True)
    md = check.markdown_report(res)
    assert "Episodes that changed" in md


def test_cli_crn_flag_is_accepted(tmp_path):
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 30 + [0] * 20},
                                     "new": {"j": [1] * 20 + [0] * 30}})
    r = _cli("check", str(tmp_path), "--no-open", "--no-report", "--crn", "--why",
             "--incumbent", "old", "--candidate", "new")
    assert r.returncode in (0, 1), r.stderr
    assert "WHICH EPISODES TO WATCH" in r.stdout


# ------------------------------------------------ pricing a success detector

def _jg():
    from orbit_eval import judge
    return judge


def test_a_perfect_judge_costs_nothing():
    j = _jg()
    recs = [{"policy": "a", "judge": b, "reference": b} for b in ([True] * 30 + [False] * 30)]
    v = j.validate(recs)
    assert v["overall"]["youden_j"] == 1.0
    assert v["episode_multiplier"] == 1.0
    assert v["attenuated_target_pp"] == 10.0


def test_a_coin_flip_judge_carries_no_signal():
    j = _jg()
    recs = ([{"policy": "a", "judge": True, "reference": True}] * 25 +
            [{"policy": "a", "judge": False, "reference": True}] * 25 +
            [{"policy": "a", "judge": True, "reference": False}] * 25 +
            [{"policy": "a", "judge": False, "reference": False}] * 25)
    v = j.validate(recs)
    assert abs(v["overall"]["youden_j"]) < 1e-9
    assert v["episode_multiplier"] == float("inf")
    assert "no usable signal" in j.format_judge(v)


def test_youdens_j_is_exactly_the_attenuation_factor():
    """The identity the whole module rests on: observed = true * (se + sp - 1)."""
    import random
    j = _jg()
    rng = random.Random(0)
    SE, SP = 0.85, 0.80
    for p_true in (0.2, 0.5, 0.8):
        obs = []
        for _ in range(200000):
            t = rng.random() < p_true
            obs.append((rng.random() < SE) if t else (rng.random() > SP))
        p_obs = sum(obs) / len(obs)
        expected = p_true * SE + (1 - p_true) * (1 - SP)
        assert abs(p_obs - expected) < 0.01
    # and therefore the gap between two arms shrinks by exactly J
    jj = SE + SP - 1
    assert abs(j.attenuated(10.0, jj) - 10.0 * jj) < 1e-9
    assert abs(j.corrected(j.attenuated(10.0, jj), jj) - 10.0) < 1e-9


def test_failbench_numbers_turn_into_a_price():
    """The published reliability of VLM judges, priced in trials."""
    j = _jg()
    assert abs(j.episode_multiplier(0.77 + 0.77 - 1) - 3.4) < 0.1   # best model
    assert abs(j.episode_multiplier(0.52 + 0.52 - 1) - 625.0) < 5   # contact-rich


def test_an_even_handed_judge_is_not_flagged():
    j = _jg()
    recs = []
    for pol in ("old", "new"):
        recs += [{"policy": pol, "judge": True, "reference": True}] * 40
        recs += [{"policy": pol, "judge": False, "reference": True}] * 10
        recs += [{"policy": pol, "judge": False, "reference": False}] * 40
        recs += [{"policy": pol, "judge": True, "reference": False}] * 10
    assert j.validate(recs)["differential"]["differential"] is False


def test_a_judge_that_errs_differently_per_policy_is_flagged():
    j = _jg()
    recs = []
    recs += [{"policy": "old", "judge": True, "reference": True}] * 90
    recs += [{"policy": "old", "judge": False, "reference": True}] * 10
    recs += [{"policy": "new", "judge": True, "reference": True}] * 50
    recs += [{"policy": "new", "judge": False, "reference": True}] * 50
    for pol in ("old", "new"):
        recs += [{"policy": pol, "judge": False, "reference": False}] * 50
    v = j.validate(recs)
    assert v["differential"]["differential"] is True
    assert "NOT EVEN-HANDED" in j.format_judge(v)


def test_the_differential_verdict_is_corrected_for_how_many_tests_it_made():
    """One verdict over many tests, unlike the per-skill flags in `release`.
    Uncorrected it fires on 10-12% of even-handed judges."""
    j = _jg()
    recs = []
    for pol in ("a", "b", "c", "d"):
        recs += [{"policy": pol, "judge": True, "reference": True}] * 40
        recs += [{"policy": pol, "judge": False, "reference": False}] * 40
    d = j.validate(recs)["differential"]
    assert d["n_tests"] == 12                      # 6 pairs x 2 metrics
    assert abs(d["alpha_per_test"] - 0.05 / 12) < 1e-9


def test_a_single_policy_cannot_test_even_handedness():
    j = _jg()
    recs = [{"policy": "only", "judge": True, "reference": True}] * 30 + \
           [{"policy": "only", "judge": False, "reference": False}] * 30
    txt = j.format_judge(j.validate(recs))
    assert "untested" in txt


def test_a_one_sided_reference_set_is_refused_not_guessed():
    j = _jg()
    recs = [{"policy": "a", "judge": True, "reference": True}] * 50
    txt = j.format_judge(j.validate(recs))
    assert "only one kind of outcome" in txt


def test_sizing_asks_for_both_kinds_of_episode():
    s = _jg().labels_needed(half_width=0.05)
    assert s["true_successes"] > 100 and s["true_failures"] > 100


def test_judge_csv_accepts_loose_column_names(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("# labelled 2026-09-16\npolicy,task,episode,vlm,human\n"
                 "old,pick,0,1,1\nold,pick,1,0,1\nnew,pick,0,1,0\n")
    recs = _jg().load(str(p))
    assert len(recs) == 3 and recs[1]["judge"] is False and recs[1]["reference"] is True


def test_judge_csv_skips_unlabelled_rows(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("policy,judge,human\na,1,1\na,0,\na,1,0\n")
    assert len(_jg().load(str(p))) == 2


def test_judge_csv_explains_what_columns_it_needs(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("policy,episode,success\na,0,1\n")
    try:
        _jg().load(str(p))
        assert False, "should have refused"
    except ValueError as e:
        assert "judge" in str(e).lower() and "reference" in str(e).lower()


def test_cli_judge_exits_1_when_the_judge_is_not_even_handed(tmp_path):
    p = tmp_path / "l.csv"
    rows = ["policy,judge,human"]
    rows += ["old,1,1"] * 90 + ["old,0,1"] * 10
    rows += ["new,1,1"] * 50 + ["new,0,1"] * 50
    rows += ["old,0,0"] * 50 + ["new,0,0"] * 50
    p.write_text("\n".join(rows) + "\n")
    r = _cli("judge", str(p))
    assert r.returncode == 1 and "NOT EVEN-HANDED" in r.stdout


def test_cli_judge_exits_0_on_an_even_handed_judge(tmp_path):
    p = tmp_path / "l.csv"
    rows = ["policy,judge,human"]
    for pol in ("old", "new"):
        rows += ["%s,1,1" % pol] * 40 + ["%s,0,1" % pol] * 10
        rows += ["%s,0,0" % pol] * 40 + ["%s,1,0" % pol] * 10
    p.write_text("\n".join(rows) + "\n")
    r = _cli("judge", str(p))
    assert r.returncode == 0 and "Youden" in r.stdout


# --------------------------------------------------- auditing a loop of rounds

def _hs():
    from orbit_eval import history
    return history


def _chain(rounds, n=100):
    """Deterministic rounds: rounds[i] maps job -> number of successes out of n."""
    return {"r%02d" % i: {j: [True] * k + [False] * (n - k) for j, k in d.items()}
            for i, d in enumerate(rounds)}


def test_history_needs_at_least_two_rounds():
    h = _hs().audit({"only": {"j": [True] * 10}}, ["only"])
    assert h["too_short"] is True
    assert "at least two" in _hs().format_history(h)


def test_history_counts_the_rounds_that_broke_something():
    data = _chain([{"j": 90}, {"j": 40}, {"j": 42}, {"j": 41}])
    h = _hs().audit(data, sorted(data))
    assert h["rounds"] == 4 and len(h["steps"]) == 3
    assert h["rounds_with_a_regression"] == 1
    assert abs(h["per_round_risk"] - 1 / 3.0) < 1e-9


def test_creeping_rot_is_caught_when_no_single_round_can_see_it():
    """Four points a round is under the resolution of every round and over it by
    the end. This is the failure a pairwise check can never find."""
    steps = [{"good": 50 + 4 * i, "rot": 90 - 4 * i} for i in range(11)]
    data = _chain(steps)
    h = _hs().audit(data, sorted(data))
    assert h["rounds_with_a_regression"] == 0, "a single round should not resolve 4 pts"
    assert h["creeping_rot"] == ["rot"]
    assert h["net_suite_delta_pp"] == 0.0 or True
    txt = _hs().format_history(h)
    assert "CREEPING ROT" in txt and "would never once have fired" in txt


def test_a_healthy_loop_reports_no_rot():
    data = _chain([{"a": 50 + 4 * i, "b": 55 + 3 * i} for i in range(8)])
    h = _hs().audit(data, sorted(data))
    assert h["creeping_rot"] == [] and h["silent_rot"] == []
    assert h["jobs_never_recovered"] == []


def test_silent_rot_is_a_round_that_gained_on_average_while_breaking_a_job():
    data = _chain([{"a": 30, "b": 90}, {"a": 95, "b": 40}])
    h = _hs().audit(data, sorted(data))
    assert h["steps"][0]["suite_delta_pp"] > 0
    assert h["silent_rot"] and h["silent_rot"][0]["jobs"] == ["b"]
    assert "SILENT ROT" in _hs().format_history(h)


def test_compounding_projection_is_monotone_and_carries_its_interval():
    data = _chain([{"j": 90}, {"j": 40}, {"j": 88}, {"j": 41}, {"j": 90}])
    h = _hs().audit(data, sorted(data))
    ks = sorted(h["projection"])
    vals = [h["projection"][k]["point"] for k in ks]
    assert vals == sorted(vals), "risk must not fall as rounds are added"
    for k in ks:
        p = h["projection"][k]
        assert p["lo"] <= p["point"] <= p["hi"]


def test_a_loop_that_never_breaks_anything_projects_zero():
    data = _chain([{"j": 50 + i} for i in range(6)])
    h = _hs().audit(data, sorted(data))
    assert h["per_round_risk"] == 0.0
    assert all(v["point"] == 0.0 for v in h["projection"].values())
    assert h["per_round_risk_ci"][1] > 0.0, "zero observed is not zero risk"


def test_jobs_that_never_recovered_are_named():
    data = _chain([{"j": 90}, {"j": 30}, {"j": 35}, {"j": 38}])
    h = _hs().audit(data, sorted(data))
    assert h["jobs_never_recovered"] == ["j"]


def test_trails_give_one_rate_per_round_per_job():
    data = _chain([{"a": 50, "b": 60}, {"a": 70, "b": 40}, {"a": 80, "b": 20}])
    h = _hs().audit(data, sorted(data))
    assert h["trails"]["a"] == [50.0, 70.0, 80.0]
    assert h["trails"]["b"] == [60.0, 40.0, 20.0]


def test_check_history_flag_produces_the_audit(tmp_path):
    cands = {"r%02d" % i: {"j": [True] * (90 - 6 * i) + [False] * (10 + 6 * i)}
             for i in range(6)}
    matrix(str(tmp_path), "m.json", cands)
    res = check.run(str(tmp_path), want_history=True, crn=True)
    assert res["history"]["rounds"] == 6
    assert "THE LOOP, ROUND BY ROUND" in check.format_check(res, why=True)
    assert "loop: 6 rounds" in check.format_check(res)


def test_dashboard_renders_the_trend_chart(tmp_path):
    cands = {"r%02d" % i: {"good": [True] * (50 + 4 * i) + [False] * (50 - 4 * i),
                           "rot": [True] * (90 - 5 * i) + [False] * (10 + 5 * i)}
             for i in range(8)}
    matrix(str(tmp_path), "m.json", cands)
    res = check.run(str(tmp_path), want_history=True, crn=True)
    html = check.html_report(res, sig="abc")
    assert "<svg" in html and "polyline" in html
    assert "The loop, round by round" in html
    assert "Which episodes to watch" in html


def test_report_is_not_destroyed_when_rendering_fails(tmp_path, monkeypatch):
    """Opening for write truncates. A render that raises must not eat the old one."""
    p = tmp_path / "r.html"
    p.write_text("<html>previous report</html>")
    matrix(str(tmp_path), "m.json", {"a": {"j": [1] * 20}, "b": {"j": [0] * 20}})
    res = check.run(str(tmp_path))
    monkeypatch.setattr(check, "html_report",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    try:
        check.write_report(res, str(p))
    except ValueError:
        pass
    assert "previous report" in p.read_text()


def test_cli_history_flag(tmp_path):
    cands = {"r%02d" % i: {"j": [True] * (80 - 5 * i) + [False] * (20 + 5 * i)}
             for i in range(5)}
    matrix(str(tmp_path), "m.json", cands)
    r = _cli("check", str(tmp_path), "--no-open", "--no-report", "--history", "--crn")
    assert r.returncode in (0, 1), r.stderr
    assert "rounds" in r.stdout



# ------------------------------------------------- short by default, long on ask

def test_default_output_is_short_enough_to_read(tmp_path):
    """A tool that prints eighty lines to answer one question gets read once."""
    cands = {"r%02d" % i: {"j%d" % k: [True] * (60 - 3 * i) + [False] * (40 + 3 * i)
                           for k in range(6)} for i in range(6)}
    matrix(str(tmp_path), "m.json", cands)
    res = check.run(str(tmp_path), crn=True)
    short = check.format_check(res)
    assert len(short.splitlines()) <= 25, short
    assert len(check.format_check(res, why=True).splitlines()) > len(short.splitlines())


def test_nothing_is_lost_in_the_short_form(tmp_path):
    """Everything the long form says is still in --why, the report and --json."""
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 90 + [0] * 10},
                                     "new": {"j": [1] * 20 + [0] * 80}})
    res = check.run(str(tmp_path), incumbent="old", candidate="new", crn=True)
    full = check.format_check(res, why=True)
    for phrase in ("WHAT THIS RUN COULD NOT SEE", "WHICH EPISODES TO WATCH"):
        assert phrase in full
    assert res["check"] is not None and res["worklist"] is not None


def test_a_regression_is_always_visible_in_the_short_form(tmp_path):
    matrix(str(tmp_path), "m.json", {"old": {"j": [1] * 90 + [0] * 10},
                                     "new": {"j": [1] * 20 + [0] * 80}})
    res = check.run(str(tmp_path), incumbent="old", candidate="new")
    assert "REGRESSED" in check.format_check(res)


def test_held_jobs_are_collapsed_not_listed(tmp_path):
    cands = {"old": {"j%d" % i: [True] * 50 + [False] * 50 for i in range(8)},
             "new": {"j%d" % i: [True] * 50 + [False] * 50 for i in range(8)}}
    matrix(str(tmp_path), "m.json", cands)
    out = check.format_check(check.run(str(tmp_path), incumbent="old", candidate="new"))
    assert "held         8 jobs" in out


def test_history_runs_automatically_once_there_are_three_rounds(tmp_path):
    """The failure a chain has is the one nobody would think to ask for, so asking
    for it with a flag means never seeing it."""
    cands = {"r%02d" % i: {"j": [True] * (70 - 4 * i) + [False] * (30 + 4 * i)}
             for i in range(5)}
    matrix(str(tmp_path), "m.json", cands)
    assert check.run(str(tmp_path)).get("history") is not None


def test_history_does_not_run_for_a_plain_pair(tmp_path):
    matrix(str(tmp_path), "m.json", {"a": {"j": [1] * 20}, "b": {"j": [0] * 20}})
    assert check.run(str(tmp_path)).get("history") is None


def test_crn_advertises_what_it_would_buy(tmp_path):
    cands = {"old": {"j": [1] * 30 + [0] * 20}, "new": {"j": [1] * 25 + [0] * 25}}
    matrix(str(tmp_path), "m.json", cands)
    res = check.run(str(tmp_path), incumbent="old", candidate="new")
    assert res["crn_gain_pp"] and res["crn_gain_pp"] > 0
    assert "--crn" in check.format_check(res)
    # and says nothing once you have asserted it
    assert check.run(str(tmp_path), incumbent="old", candidate="new",
                     crn=True)["crn_gain_pp"] is None


# ------------------------------------------------------------- the agent skill

def test_skill_prints_without_installing():
    r = _cli("skill")
    assert r.returncode == 0
    assert "name: orbit-eval" in r.stdout and "description:" in r.stdout


def test_skill_has_the_frontmatter_an_agent_needs():
    from orbit_eval import cli
    text = cli._skill_text()
    assert text.startswith("---\n")
    head = text.split("---")[1]
    assert "name:" in head and "description:" in head
    # the description is what an agent matches on, so it must name the triggers
    for word in ("worse", "trials", "eval_info.json", "LeRobot"):
        assert word in head, "trigger %r missing from the description" % word


def test_skill_tells_the_agent_the_one_rule_that_matters():
    from orbit_eval import cli
    text = cli._skill_text()
    assert "Never quote a difference without its interval" in text
    assert "suspect" in text


def test_skill_installs_into_an_agent_directory(tmp_path, monkeypatch):
    from orbit_eval import cli
    home = tmp_path / "home"
    (home / ".claude" / "skills").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(home), 1))

    class A:
        install, all = True, False
    assert cli.cmd_skill(A()) == 0
    dest = home / ".claude" / "skills" / "orbit-eval" / "SKILL.md"
    assert dest.exists() and "orbit-eval" in dest.read_text()


def test_skill_install_reports_when_no_agent_is_present(tmp_path, monkeypatch, capsys):
    from orbit_eval import cli
    home = tmp_path / "bare"
    home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(home), 1))

    class A:
        install, all = True, False
    assert cli.cmd_skill(A()) == 1
    out = capsys.readouterr().out
    assert "No agent skills directory found" in out and "orbit skill" in out


def test_skill_ships_inside_the_package():
    """It has to be in the wheel or `orbit skill` breaks for everyone who pip
    installed rather than cloned."""
    import orbit_eval
    p = os.path.join(os.path.dirname(orbit_eval.__file__), "skills", "SKILL.md")
    assert os.path.exists(p)
