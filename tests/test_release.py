"""release: per-skill regression check, priced damage, pairing, history, CLI."""

import contextlib
import io as _io
import json
import os
import random
import shutil
import tempfile
import unittest

from orbit_eval import cli, release


def _run(argv):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def _csv(path, spec, n=60, seed=0, header="version,skill,episode,success"):
    """spec = {version: {skill: p}}"""
    rng = random.Random(seed)
    lines = [header]
    for v, skills in spec.items():
        for s, p in skills.items():
            for e in range(n):
                lines.append("%s,%s,%d,%d" % (v, s, e, 1 if rng.random() < p else 0))
    open(path, "w").write("\n".join(lines) + "\n")
    return path


class TestCheck(unittest.TestCase):
    def test_verdict_is_the_interval_clearing_the_floor(self):
        """REGRESSED means: 95% confident the skill lost at least drop_pp. A drop that
        clears the floor on the point estimate but not on the interval is SUSPECT,
        however many episodes there are."""
        # 12 pp drop, tight at n=4000: the interval clears -5 pp.
        for n, drop, expect in ((4000, 0.12, "REGRESSED"),
                                (20, 0.06, "SUSPECT"),      # wide interval
                                (4000, 0.055, "SUSPECT")):  # tight, but too close to -5
            k_a, k_b = int(0.70 * n), int((0.70 - drop) * n)
            data = {"old": {"s": [True] * k_a + [False] * (n - k_a)},
                    "new": {"s": [True] * k_b + [False] * (n - k_b)}}
            r = release.check(data, "old", "new")["rows"][0]
            self.assertEqual(r["verdict"], expect, (n, drop, r))

    def test_small_drop_never_flags_however_many_episodes(self):
        """Below the absolute floor, no sample size makes it a regression."""
        n = 20000
        k_a, k_b = int(0.700 * n), int(0.680 * n)      # 2 pp, way inside the floor
        data = {"old": {"s": [True] * k_a + [False] * (n - k_a)},
                "new": {"s": [True] * k_b + [False] * (n - k_b)}}
        c = release.check(data, "old", "new")
        self.assertEqual(c["rows"][0]["verdict"], "held")
        self.assertEqual(c["n_regressed"], 0)

    def test_interval_sign_matches_the_delta(self):
        """A drop must report a negative interval; the paired CI is easy to flip."""
        rng = random.Random(5)
        A = [rng.random() < 0.85 for _ in range(400)]
        B = [rng.random() < 0.45 for _ in range(400)]
        data = {"old": {"s": A}, "new": {"s": B}}
        r = release.check(data, "old", "new")["rows"][0]
        self.assertLess(r["delta_pp"], 0)
        self.assertLess(r["ci95"][1], 0, r)          # whole interval below zero
        self.assertLessEqual(r["ci95"][0], r["ci95"][1])

    def test_pairing_is_detected_and_beats_unpaired_precision(self):
        rng = random.Random(9)
        A = [rng.random() < 0.6 for _ in range(300)]
        B = list(A)
        for i in range(0, 300, 10):                   # a few honest flips
            B[i] = not B[i]
        data = {"old": {"s": A}, "new": {"s": B}}
        ids = {"old": {"s": list(range(300))}, "new": {"s": list(range(300))}}
        paired = release.check(data, "old", "new", episode_ids=ids)["rows"][0]
        unpaired = release.check(data, "old", "new")["rows"][0]
        self.assertEqual(paired["method"], "paired")
        self.assertEqual(unpaired["method"], "unpaired")
        self.assertLess(paired["se_pp"], unpaired["se_pp"])

    def test_equal_length_is_not_alignment(self):
        """Two versions run on different initial states have equal length and nothing
        else; pairing them would understate every interval."""
        rng = random.Random(21)
        A = [rng.random() < 0.6 for _ in range(200)]
        B = [rng.random() < 0.6 for _ in range(200)]
        data = {"old": {"s": A}, "new": {"s": B}}
        # no ids at all -> must refuse to pair
        self.assertEqual(release.check(data, "old", "new")["rows"][0]["method"], "unpaired")
        # ids that do not match -> must refuse to pair
        bad = {"old": {"s": list(range(200))}, "new": {"s": list(range(1000, 1200))}}
        self.assertEqual(release.check(data, "old", "new", episode_ids=bad)["rows"][0]["method"],
                         "unpaired")
        # ids that match -> paired
        good = {"old": {"s": list(range(200))}, "new": {"s": list(range(200))}}
        self.assertEqual(release.check(data, "old", "new", episode_ids=good)["rows"][0]["method"],
                         "paired")

    def test_improvement_is_priced_like_damage(self):
        """Demanding evidence for bad news and taking good news on the point estimate
        would be self-serving."""
        n = 30
        k_a, k_b = int(0.50 * n), int(0.58 * n)      # +8 pp, well inside the noise at n=30
        data = {"old": {"s": [True] * k_a + [False] * (n - k_a)},
                "new": {"s": [True] * k_b + [False] * (n - k_b)}}
        r = release.check(data, "old", "new")["rows"][0]
        self.assertGreater(r["delta_pp"], release.DROP_PP)
        self.assertEqual(r["verdict"], "held")       # not IMPROVED

    def test_detection_floor_never_understates(self):
        """The honest floor is max(absolute floor, interval width), not the width alone."""
        data = {"old": {"s": [True] * 5000 + [False] * 5000},
                "new": {"s": [True] * 5000 + [False] * 5000}}
        r = release.check(data, "old", "new")["rows"][0]
        self.assertGreaterEqual(r["resolvable_drop_pp"], release.DROP_PP)

    def test_unequal_episode_counts_fall_back_to_unpaired(self):
        data = {"old": {"s": [True] * 100}, "new": {"s": [True] * 80}}
        c = release.check(data, "old", "new")
        self.assertEqual(c["rows"][0]["method"], "unpaired")
        self.assertFalse(c["paired"])

    def test_suite_can_hold_while_a_skill_collapses(self):
        """The reason this command exists."""
        n = 400
        old = {"a": 0.80, "b": 0.45, "c": 0.45, "d": 0.45}
        new = {"a": 0.30, "b": 0.70, "c": 0.70, "d": 0.70}
        data = {}
        # fixed seeds: hash() is salted per process, which made this test flaky
        for seed, (v, rates) in enumerate((("old", old), ("new", new))):
            rng = random.Random(1000 + seed)
            data[v] = {s: [rng.random() < p for _ in range(n)] for s, p in rates.items()}
        c = release.check(data, "old", "new")
        self.assertGreaterEqual(c["n_regressed"], 1)
        self.assertGreater(c["suite_delta_pp"], 5.0)   # the average went UP, clearly
        self.assertEqual(c["rows"][0]["skill"], "a")

    def test_skills_missing_from_one_side_are_reported_not_dropped_silently(self):
        data = {"old": {"a": [True] * 50, "gone": [True] * 50},
                "new": {"a": [True] * 50, "brand_new": [True] * 50}}
        c = release.check(data, "old", "new")
        self.assertEqual(c["n_skills"], 1)
        self.assertEqual(c["skills_missing_from_one_side"], ["brand_new", "gone"])

    def test_expected_false_flags_scales_with_skill_count(self):
        mk = lambda k: {"old": {"s%d" % i: [True] * 50 for i in range(k)},
                        "new": {"s%d" % i: [True] * 50 for i in range(k)}}
        self.assertAlmostEqual(release.check(mk(10), "old", "new")["expected_false_flags"], 0.5)
        self.assertAlmostEqual(release.check(mk(50), "old", "new")["expected_false_flags"], 2.5)


class TestHistory(unittest.TestCase):
    def test_counts_releases_with_a_regression(self):
        n = 600
        rng = random.Random(2)
        vs = ["v1", "v2", "v3"]
        rates = [{"a": 0.8, "b": 0.6}, {"a": 0.3, "b": 0.6}, {"a": 0.3, "b": 0.6}]
        data = {v: {s: [rng.random() < p for _ in range(n)] for s, p in r.items()}
                for v, r in zip(vs, rates)}
        h = release.history(data, vs)
        self.assertEqual(h["releases"], 2)
        self.assertEqual(h["with_a_regression"], 1)
        self.assertIn("a", h["steps"][0]["regressed_skills"])
        self.assertEqual(h["steps"][1]["regressed_skills"], [])


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orbit_release_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_two_versions_exit_1_on_regression(self):
        p = _csv(os.path.join(self.tmp, "e.csv"),
                 {"v1": {"a": 0.85, "b": 0.6}, "v2": {"a": 0.25, "b": 0.6}}, n=200, seed=1)
        code, text = _run(["release", p])
        self.assertEqual(code, 1, text)
        self.assertIn("REGRESSED", text)
        self.assertIn("WHAT THIS RUN COULD NOT SEE", text)

    def test_clean_release_exits_0(self):
        p = _csv(os.path.join(self.tmp, "e.csv"),
                 {"v1": {"a": 0.6, "b": 0.6}, "v2": {"a": 0.6, "b": 0.6}}, n=200, seed=2)
        code, text = _run(["release", p])
        self.assertEqual(code, 0, text)
        self.assertIn("NO SKILL REGRESSED", text)

    def test_column_aliases_are_autodetected(self):
        p = _csv(os.path.join(self.tmp, "e.csv"),
                 {"v1": {"a": 0.85}, "v2": {"a": 0.25}}, n=200, seed=3,
                 header="checkpoint,sku,trial,passed")
        code, text = _run(["release", p])
        self.assertEqual(code, 1, text)

    def test_single_version_is_a_clear_error_not_a_crash(self):
        p = _csv(os.path.join(self.tmp, "e.csv"), {"only": {"a": 0.6}}, n=20)
        code, text = _run(["release", p])
        self.assertEqual(code, 2)
        self.assertIn("Two or more are needed", text)

    def test_missing_columns_names_every_accepted_alias(self):
        p = os.path.join(self.tmp, "bad.csv")
        open(p, "w").write("foo,bar\n1,2\n")
        code, text = _run(["release", p])
        self.assertEqual(code, 2)
        for word in ("model:", "skill:", "episode:", "outcome:"):
            self.assertIn(word, text)

    def test_history_mode_and_signed_record(self):
        p = _csv(os.path.join(self.tmp, "h.csv"),
                 {"v1": {"a": 0.85, "b": 0.6}, "v2": {"a": 0.25, "b": 0.6},
                  "v3": {"a": 0.25, "b": 0.6}}, n=200, seed=4)
        out = os.path.join(self.tmp, "rec.json")
        code, text = _run(["release", p, "--out", out])
        self.assertIn("RELEASE HISTORY", text)
        rec = json.load(open(out))
        self.assertIn("record_sha256", rec)
        self.assertEqual(len(rec["record_sha256"]), 64)
        self.assertIn("history", rec)

    def test_explicit_version_pair_overrides_history(self):
        p = _csv(os.path.join(self.tmp, "h.csv"),
                 {"v1": {"a": 0.85}, "v2": {"a": 0.85}, "v3": {"a": 0.25}}, n=200, seed=5)
        code, text = _run(["release", p, "--incumbent", "v1", "--candidate", "v3"])
        self.assertEqual(code, 1)
        self.assertIn("v1  ->  v3", text)
        self.assertNotIn("RELEASE HISTORY", text)

    def test_unknown_version_is_exit_2(self):
        p = _csv(os.path.join(self.tmp, "h.csv"), {"v1": {"a": 0.6}, "v2": {"a": 0.6}}, n=20)
        code, text = _run(["release", p, "--incumbent", "nope", "--candidate", "v2"])
        self.assertEqual(code, 2)
        self.assertIn("nope", text)

    def test_json_mode_is_machine_readable(self):
        p = _csv(os.path.join(self.tmp, "e.csv"),
                 {"v1": {"a": 0.85}, "v2": {"a": 0.25}}, n=200, seed=6)
        code, text = _run(["release", p, "--json"])
        obj = json.loads(text)
        self.assertEqual(obj["latest"]["n_regressed"], 1)
        self.assertIn("rows", obj["latest"])

    def test_underpowered_skills_are_refused_not_passed(self):
        """A false all-clear is the worst output this tool could produce."""
        data = {"old": {"a": [True] * 8}, "new": {"a": [False] * 8}}
        c = release.check(data, "old", "new")
        self.assertEqual(c["rows"][0]["verdict"], "UNDERPOWERED")
        self.assertEqual(c["n_regressed"], 0)
        self.assertEqual(c["n_underpowered"], 1)
        self.assertTrue(c["all_underpowered"])

    def test_underpowered_uses_the_smaller_side(self):
        data = {"old": {"a": [True] * 500}, "new": {"a": [False] * 4}}
        self.assertEqual(release.check(data, "old", "new")["rows"][0]["verdict"],
                         "UNDERPOWERED")

    def test_min_episodes_is_tunable_and_off_by_default_nowhere(self):
        data = {"old": {"a": [True] * 8}, "new": {"a": [False] * 8}}
        c = release.check(data, "old", "new", min_episodes=4)
        self.assertEqual(c["rows"][0]["verdict"], "REGRESSED")

    def test_n1_never_reports_a_zero_width_interval_as_a_verdict(self):
        data = {"old": {"a": [True]}, "new": {"a": [False]}}
        c = release.check(data, "old", "new")
        self.assertEqual(c["rows"][0]["verdict"], "UNDERPOWERED")
        out = release.format_check(c)
        self.assertIn("interval not reportable", out)
        self.assertNotIn("95% [-100.0, -100.0]", out)


class TestUnderpoweredCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orbit_release_up_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_low_episode_suite_does_not_read_as_an_all_clear(self):
        p = _csv(os.path.join(self.tmp, "low.csv"),
                 {"v1": {"a": 0.9, "b": 0.9}, "v2": {"a": 0.1, "b": 0.9}}, n=8, seed=8)
        code, text = _run(["release", p])
        # exit 3, not 0: a CI job must not go green on a file that answered nothing.
        # This mirrors `orbit-eval regress`, which already uses 3 for underpowered.
        self.assertEqual(code, 3)
        self.assertIn("CANNOT ANSWER THE QUESTION", text)
        self.assertIn("not an all-clear", text)
        self.assertNotIn("NO SKILL REGRESSED", text)
        self.assertIn("episodes per skill per version", text)

    def test_one_unjudged_skill_blocks_a_green_ci_run(self):
        """The dangerous shape is MIXED, not all-underpowered.

        A well-sampled skill holds while a thinly-sampled one collapses to zero.
        Through 0.8.2 this printed NO SKILL REGRESSED and exited 0, because the
        all-clear was suppressed only when EVERY skill was underpowered.
        """
        rows = ["version,skill,episode,success"]
        for v in ("v1", "v2"):
            rows += ["%s,packing,%d,1" % (v, e) for e in range(20)]
        rows += ["v1,unload,%d,1" % e for e in range(4)]
        rows += ["v2,unload,%d,0" % e for e in range(4)]
        p = os.path.join(self.tmp, "mixed.csv")
        with open(p, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        code, text = _run(["release", p])
        self.assertEqual(code, 3, text)
        self.assertNotIn("NO SKILL REGRESSED", text)
        self.assertIn("COULD JUDGE", text)
        flat = " ".join(text.split())          # the notice wraps across lines
        self.assertIn("This is not an all-clear", flat)
        self.assertIn("exits 3 rather than 0", flat)

    def test_mixed_headline_names_only_the_judged_skills(self):
        data = {"old": {"a": [True] * 40, "b": [True] * 6},
                "new": {"a": [True] * 40, "b": [False] * 6}}
        c = release.check(data, "old", "new")
        self.assertFalse(c["all_underpowered"])
        self.assertEqual(c["n_underpowered"], 1)
        self.assertEqual(c["n_regressed"], 0)
        out = release.format_check(c)
        self.assertIn("NO REGRESSION AMONG THE 1 SKILL THIS FILE COULD JUDGE", out)
        self.assertNotIn("NO SKILL REGRESSED", out)



class TestReviewRegressions(unittest.TestCase):
    """Bugs an adversarial review found in the first cut. Each cost a wrong answer on
    input shapes that are normal, not exotic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orbit_rev_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_boundary_rate_does_not_manufacture_a_verdict(self):
        """A 100% incumbent used to give the Wald SE zero variance, so the tool printed
        BROKE on a row whose own interval contained zero."""
        data = {"v1": {"seat": [True] * 20},
                "v2": {"seat": [True] * 21 + [False] * 4}}
        r = release.check(data, "v1", "v2")["rows"][0]
        self.assertGreater(r["ci95"][1], 0)              # interval contains zero
        self.assertNotEqual(r["verdict"], "REGRESSED")   # so it must not be a verdict

    def test_verdict_and_interval_never_disagree(self):
        """The rule IS the interval, so the two halves of a row cannot contradict."""
        rng = random.Random(31)
        for trial in range(60):
            n_a = rng.choice([12, 40, 200])
            n_b = rng.choice([12, 40, 200])
            pa, pb = rng.random(), rng.random()
            data = {"a": {"s": [rng.random() < pa for _ in range(n_a)]},
                    "b": {"s": [rng.random() < pb for _ in range(n_b)]}}
            r = release.check(data, "a", "b")["rows"][0]
            if r["verdict"] == "REGRESSED":
                self.assertLessEqual(r["ci95"][1], -release.DROP_PP, (trial, r))

    def test_suite_average_cannot_disagree_in_sign_with_every_skill(self):
        """Weighting each version by its own episode counts made the headline an
        allocation artifact: every skill fell 10 pp and the suite reported +47.65."""
        def block(n, k):
            return [True] * k + [False] * (n - k)
        data = {"v1": {"stow": block(20, 18), "peg": block(1000, 300)},
                "v2": {"stow": block(1000, 800), "peg": block(20, 4)}}
        c = release.check(data, "v1", "v2")
        self.assertTrue(all(r["delta_pp"] < 0 for r in c["rows"]))
        self.assertLess(c["suite_delta_pp"], 0)
        # with one common weight the headline IS the mean of the per-skill deltas
        self.assertAlmostEqual(c["suite_delta_pp"],
                               sum(r["delta_pp"] for r in c["rows"]) / len(c["rows"]),
                               places=6)

    def test_duplicate_episode_rows_are_refused_not_silently_dropped(self):
        """Concatenated eval shards each numbering episodes from 0 used to discard
        150 of 200 episodes and report a +50 pp improvement on an unchanged skill."""
        tmp = tempfile.mkdtemp(prefix="orbit_dupe_")
        try:
            p = os.path.join(tmp, "shards.csv")
            lines = ["version,skill,episode,success"]
            for v, (k1, k2) in (("v1", (50, 0)), ("v2", (25, 25))):
                for shard, k in enumerate((k1, k2)):
                    for e in range(50):
                        lines.append("%s,cap,%d,%d" % (v, e, 1 if e < k else 0))
            open(p, "w").write("\n".join(lines) + "\n")
            code, text = _run(["release", p])
            self.assertEqual(code, 2)
            self.assertIn("duplicated", text)
            self.assertIn("renumber", text.lower())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


    def test_a_vanished_skill_is_a_finding_and_not_a_green_build(self):
        """Dropping a hard skill raises the average of what is left; that must not
        read as a pass."""
        p = _csv(os.path.join(self.tmp, "v.csv"),
                 {"v1": {"easy": 0.9, "hard": 0.2}, "v2": {"easy": 0.9}}, n=100, seed=12)
        code, text = _run(["release", p])
        self.assertEqual(code, 1)
        self.assertIn("GONE FROM THE NEW MODEL", text)
        self.assertIn("hard", text)

    def test_version_order_is_disclosed_not_assumed_silently(self):
        p = _csv(os.path.join(self.tmp, "o.csv"),
                 {"v1": {"a": 0.6}, "v2": {"a": 0.6}, "v3": {"a": 0.6}}, n=50, seed=13)
        _code, text = _run(["release", p])
        self.assertIn("version order taken from the file", text)
        self.assertIn("--order", text)

    def test_thresholds_are_honoured_in_history_mode(self):
        """--drop-pp used to be silently ignored whenever the file had 3+ versions."""
        p = _csv(os.path.join(self.tmp, "t.csv"),
                 {"v1": {"a": 0.80}, "v2": {"a": 0.80}, "v3": {"a": 0.62}}, n=400, seed=14)
        strict, _ = _run(["release", p, "--drop-pp", "40"])
        loose, _ = _run(["release", p, "--drop-pp", "2"])
        self.assertEqual(strict, 0)
        self.assertEqual(loose, 1)


if __name__ == "__main__":
    unittest.main()
