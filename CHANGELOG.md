# Changelog

## 0.17.0 — 2026-09-17

The five commands of the spec are all shipped, and the map is real.

**`orbit body`: which robot can physically do this.** Reach, payload, degrees
of freedom and price against a robot database of thirty bodies, every figure
read from a page named per field and nulls where a figure was never published
(the SO-101's reach and payload are not published anywhere; the database says
so instead of guessing). Output is three blocks: what the task needs, what
passes, cannot be checked, or fails and by how much, and what the map has
measured for that job shape on each body that passes. It never says an arm is
wrong for a task; it says "payload 0.5 kg, short by 2.5 kg".

**`orbit body --skill release.json`** compares a frozen skill's trained joint
ranges with your own recording, joint by joint. LeRobot normalises commands to
each arm's own calibration, so two SO-101s are not the same body until the
ranges are shown to overlap. This is the check a downloaded skill needs before
it runs.

**`orbit freeze`: pin it so every later comparison is cheap.** One manifest
holds the checkpoint's file hashes, the inference seed (diffusion and
flow-matching policies sample their actions, so a frozen checkpoint is still
not deterministic), the eval seed sequence so any two runs are paired by
construction, and the dataset's joint ranges for the skill listing. `--verify`
re-hashes and names what moved. The manifest says what the checkpoint is; it
does not say that it works.

**The map ships.** `orbit_eval/map/map.json`, built from the banked corpus by
`research/map/build_map.py`: the noise atlas, the preregistered cross-model
wave, nine public `eval_info.json` files from the Hub (one excluded as a
byte-identical duplicate, with the reason), and the thirteen public SO-101
fine-tunes that carry no number at all. Each cell carries the pooled rate, the
interval, trials, independent sources and independent training seeds, and a
status of UNMEASURED, ONE_SEED, ONE_SOURCE or MEASURED. Ten cells; 99 percent
of the possible cells are empty, and an empty cell is the product. `orbit body
<robot>` reads it.

**`orbit check --demo`: a published self-improvement loop, audited.** Table II
of REVOLVE (arXiv 2609.14633, five iterations, four real tasks, 100 rollouts
each) is bundled as counts. The end-to-end gain is real on one task and inside
the noise on three; no single round's step is callable at 100 rollouts. When
the policies in a folder are successive rounds of one loop, `next` now says
"rounds" and "moved" instead of "copies" and "retrain".

**Removed nothing.** 0.16.0's commands are unchanged.

## 0.16.0 — 2026-09-17

**`orbit cover`: what the recording is missing.** Factor coverage from design of
experiments. The instruction text is read into a template ("Grab {A} and place
into pen holder", "put the {A} on the {B}"), every cell gets its episode count,
and the cells that hold none are named, including instructions the tasks file
lists that no episode carries. Per episode it counts length outliers, which
joints moved (a joint that travelled under 5% of its recorded range was idle in
that episode), and where each joint sat by thirds of its recorded range, crossed
with the instruction. One sentence names the largest gap: the empty cell that
even coverage would have filled the most, with the arithmetic shown.

It counts. It never scores a demonstration and never says a gap caused a
failure; the output says so in those words, and the tests forbid the
alternatives.

v2.1 datasets are read with no dependency (`meta/episodes.jsonl`,
`meta/episodes_stats.jsonl`). v3.0 keeps the per-episode table as parquet, which
is read through `pyarrow` when it is installed and otherwise reported as the one
thing needed: `pip install 'orbit-eval[coverage]'`. The base install stays
dependency-free.

**`--demo` on `status` and `cover`.** Two public SO-101 datasets' metadata ship
with the package (`orbit_eval/demo/SOURCES.md`, both Apache-2.0):
`lerobot/svla_so101_pickplace` for `status`, where two joints' commands sit
exactly on the calibration bound, and `youliangtan/so101-table-cleanup` for
`cover`, eighty episodes over four instructions. `uvx --from orbit-eval orbit
status --demo` now works on a machine with nothing on it.

**`orbit check` reads the single-task `eval_info.json`.** A single-task
`lerobot-eval` writes `per_episode` and `aggregated` and no `per_task`, and that
is the commonest evaluation artifact in public. `check` and `next` now read it
(job name `task/0`), keep the episode order so `--crn` can be asserted, and pick
up the top-level `video_paths`. A file that carries only `pc_success` and
`n_episodes` is expanded from the counts and the whole source is marked
unpairable, because trials rebuilt from a rate must never be paired.

**Common random numbers, verified from the files.** `lerobot-eval` records the
seed of every episode. When two runs carry the same seed at every index, episode
k is the same starting state for both, and `check` now pairs them and says so,
with no `--crn` needed. When the seeds differ, no episode-level comparison exists
whatever anybody asserts, so the "episodes that used to work now fail" list is
withheld and the two seeds are printed instead. Files without per-episode seeds
still need the assertion, and the short output now says that the list is matched
by position until they get it.

**The normalisation bound is verified.** LeRobot's motor bus clamps each raw
reading to the range recorded at calibration and maps that to -100..100 (arm
joints) or 0..100 (gripper); nothing rescales per dataset. So a recorded command
at exactly the bound was clipped, and the finding on the official example
stands. The gripper is now exempt from the check, because its two bounds are
the calibrated closed and open positions. `validation/NORMALISATION_BOUND_2026-09-17.md`.

**`status` ends with `ALSO WORTH KNOWING`**: the other four commands, one line
each, so nobody has to learn a menu.

PyPI metadata gained keywords and Source, Changelog and Issues links.

## 0.15.0 — 2026-09-17

**`orbit status`: where this robot is, and the one thing to do next.** Reads
`meta/info.json` and `meta/stats.json`, a few kilobytes of plain JSON present in
both the v2.x and v3.0 LeRobot layouts, and reports four stages (nothing
recorded, a recording, a checkpoint, a result) with one next action each. A
defect in the recording outranks every other piece of advice: dead joints,
commands clipped at the normalised bound, joints commanded past where the arm
reached, NaN in the statistics, datasets under fifty episodes. Robot-agnostic by
construction: it reads joint names and ranges out of the dataset and never asks
what robot produced them.

The trial table is where this meets the rest of the tool. At the
`--dataset.num_episodes=10` that LeRobot's own `AGENT_GUIDE.md` puts in front of
every user, the smallest callable drop is 44.7 points. Bare `orbit` falls
through to `status` when there is no evaluation here yet.

## 0.14.0 — 2026-09-16

**`orbit skill --install`** writes an agent skill into `~/.claude/skills`,
`~/.codex/skills`, `~/.opencode/skills`, `~/.cursor/skills` and `~/.agents/skills`
for whichever of them exist.

Robotics engineers increasingly drive their work through a coding agent, and an
agent that has never heard of this tool answers "did my checkpoint get worse?" by
averaging two success rates, which is the exact mistake the tool exists to
prevent. The skill tells the agent when to run `orbit`, how to read the verdicts,
and the one rule that matters: never quote a difference without its interval.

`orbit skill` prints it for pasting anywhere else.

## 0.13.0 — 2026-09-16

Measured the friction and removed it. Before: eleven commands in `--help`,
thirteen flags on `check`, 65 to 124 lines of output to answer one question, and
a bare `orbit` that printed a menu.

- **`orbit` on its own runs the check here**, the way `git status` does. It falls
  back to a quickstart only when there is nothing to check. Printing a menu
  instead wastes the one interaction a newcomer gives you.
- **Short by default.** The same ten-round loop went from 124 lines to 14: the
  verdict, the jobs that moved, the loop's state, and one next step. `--why`
  restores every caveat, and nothing was removed from the report or `--json`.
- **The round-by-round audit runs automatically** once three or more rounds are
  present. The failure a chain has is the one nobody would think to ask for, so
  putting it behind a flag means never seeing it.
- **`--crn` advertises itself.** When the episodes look alignable the output says
  what asserting it would buy, in points, instead of leaving the flag in a manual.
- **Four commands in `--help`**, with the other seven listed once in the epilog.

## 0.12.0 — 2026-09-16

**`orbit check --history` audits the whole chain, not just the last pair**, and the
HTML report becomes a dashboard.

A policy that retrains itself, or a team that ships every fortnight, is a
sequence, and a sequence has a failure a pair does not. A job that loses four
points a round sits under the resolution of every single round and over it by the
end. A pairwise check run forever would never once fire.

On a ten-round simulated loop this reports zero rounds broken, a suite average up
5.8 points, and:

    CREEPING ROT: insert/2
      NO single round broke this job. 9 rounds together did.

- **Per-round risk** and what it compounds to by round 10, 25 and 50, carrying the
  interval of the rate rather than a point, and saying plainly that treating
  rounds as independent flatters a loop that trains on its own output.
- **Creeping rot**: jobs regressed end to end that no single round flagged.
- **Silent rot**: rounds whose average went UP while a job broke, which is the
  shape a loop scoring itself produces.
- **Never recovered**: jobs that broke at some round and are still below where
  they started.
- The report now carries a trend chart (one line per job across rounds, regressed
  jobs in the alarm colour, end labels staggered so they never collide), a
  round-by-round strip, and the episode worklist.

### Fixed

- `write_report` opened the file before rendering, so a render that raised
  truncated the previous report to zero bytes. It renders first now.
- A missing format argument in the chart's leader lines.

## 0.11.0 — 2026-09-16

**`orbit judge` prices your success detector.** Every number this package prints
sits on a bit that says the robot did the task. In simulation that bit is a
reward threshold; across 1,257 per-task blocks harvested from public
repositories, 99% of them are exactly `max_reward >= 1.0`. On a real robot the
bit comes from a person or a model watching, and nobody measures how wrong it is.

The point is not that a judge is inaccurate. It is that an inaccurate judge
shrinks the difference you are trying to measure, by an exact amount. With
sensitivity se and specificity sp, two policies scored by the same judge show

    observed difference = true difference x (se + sp - 1)

That factor is Youden's J, it does not depend on the success rate, and it is not
an approximation. It turns published judge reliability into a price in trials:

| judge | J | a 10-point regression reads | trials needed |
|---|---|---|---|
| perfect | 1.00 | 10.0 points | 1x |
| best VLM measured across 14 public sources (0.77) | 0.54 | 5.4 points | 3.4x |
| VLMs on contact-heavy tasks (0.60) | 0.20 | 2.0 points | 25x |
| VLMs on contact-rich assembly (0.52) | 0.04 | 0.4 points | 625x |

`orbit judge labels.csv` takes episodes carrying both a judge label and a
reference label and reports sensitivity, specificity, Youden's J, the bias in
the reported success rate, the attenuation, the cost in trials, and how many
reference labels would pin it down. It never labels anything itself.

It also tests whether the judge errs the SAME way for every policy. Even
shrinkage is correctable; uneven shrinkage is not, and can invent a difference
that is not there. That verdict is one decision over many comparisons, so unlike
the per-skill flags in `release` it is explicitly corrected for multiplicity:
uncorrected it fired on 10-12% of even-handed judges, and corrected it fires on
5.2% while still catching a genuinely uneven judge 98.7% of the time at sixty
episodes per policy. Exit code 1 when the judge is uneven.

## 0.10.0 — 2026-09-16

**`orbit check` now says which episodes to watch.** A number is not a bug report.
If both policies started episode 7 of a job from the same state and the old one
solved it while the new one did not, episode 7 is a reproducible counterexample,
and LeRobot already recorded a video of it. Those are the discordant pairs that
McNemar has always counted to decide whether a drop is real; only one of the two
readings had ever been printed.

- The per-job list of episodes that broke and that were fixed, with the two
  recordings for each, old and new. On a public repository with 40 jobs this
  turns "libero_goal/9 dropped 60 points" into twelve mp4 files to open.
- `--crn` asserts that episode k is the same starting state for every policy.
  It is the one fact the tool cannot check, so it is asserted explicitly and
  recorded as an assertion. It also enables the paired comparison, which on a
  measured example narrowed one interval from 52 to 40 points for free.
- The worklist appears in the terminal, the HTML report and the PR comment.

What it deliberately does not do is say WHY. A survey of ~150 robot verifiers
finds credibility falls as availability rises, and the strongest general
vision-language judge measured across fourteen public sources reaches 0.77
balanced accuracy, with no model above 0.60 where success depends on fine
contact. A judge wrong one time in four can order a queue; it cannot supply a
label. So this orders the queue and leaves the verdict to a person.

### Fixed

- A `{"candidates": ...}` matrix was marked unpairable, so `--crn` could not
  pair it. A matrix is an explicit claim that column k is the same episode.
- Video paths longer than the success list (an extra aggregate clip is common)
  are truncated rather than shifted, so an index never names the wrong clip.

## 0.9.1 — 2026-09-15

Found by running 0.9.0 over 311 public evaluation artifacts from 141 repositories
(WILD-1). Both of these silently lost real data.

- **Read LeRobot content whatever the file is called.** In the wild it is
  `eval_act_50k.json`, `smoke_eval_info.json`, `eval_100K.json` or
  `eval_info-checkpoint.json` at least as often as `eval_info.json`. Fifty
  artifacts were refused for having the right content under the wrong name.
- **Name policies by filename when they share a directory.** Projects put
  `results/eval_act_50k.json` and `results/eval_smolvla_20k.json` side by side.
  The old rule collapsed those into one policy, losing the comparison without
  saying so. The one-directory-per-policy layout is unchanged.

Coverage of plausible public evaluations went from 87% to 89% per directory and
from 66% to 74% per repository.

## 0.9.0 — 2026-09-15

The engine was right and unusable. This release is about being run.

### Added

- **`orbit check [PATH]`** — run it in the directory your evaluations are already
  in. Walks the tree, recognises what it finds, works out which policy is the one
  you ship today, and answers per named job. Writes a self-contained HTML report
  and a signed record. Exits 1 on a regression, so CI can use it.
- **`orbit next [PATH]`** — what to do tomorrow, per job: retrain, collect
  demonstrations, or run more trials and how many. The three answers are not
  interchangeable and telling them apart needs the noise floor, so one ships:
  Hartley's expected range, with the retrain call gated on the upper tail so it
  fires on 8.5–12.3% of pools of identical copies rather than about half.
- **`orbit log`** — record trials at the robot, one keypress each, with an
  optional key for why a trial failed and a `--block` for robot, cell, day or
  operator. Every trial is fsynced before the next key is read; re-running
  resumes from the file.
- **A GitHub Action** (`.github/actions/orbit-check`) — one pull-request comment,
  edited in place on each push, failing the job only on a regression beyond
  measurement noise.
- An `orbit` console script alongside `orbit-eval`. A bare `orbit` prints a
  quickstart instead of an argparse error.

### Input formats

Tested against 21 real evaluation files from 11 public repositories rather than
invented fixtures. Now read: `eval_info.json` at any depth, one row per trial,
one row per policy with a trial count and successes, one row per job with one
column per policy plus a count, and a `{"candidates": ...}` matrix. Comment lines
before a CSV header are skipped. Rates are understood as `0.82` or `82.0`.

Where trials were rebuilt from summary counts the comparison is marked unpaired:
the episodes are gone, and pairing them would invent an agreement nobody
measured. Files that look like evaluations and cannot be used are named with the
reason rather than silently dropped.

### Fixed

- Policies that did not all run the same jobs emptied the comparison for
  everybody. The largest comparable group is now used and the rest are named.
- `stats.wilson` returns fractions and `route.wilson` returns points; two new
  call sites used the wrong one.
- A literal `%%` in the release footer, and a crash formatting markdown when
  nothing comparable was found.
