---
name: orbit-eval
description: Use when someone is training a robot policy and it is not working, asks why their robot does not move, asks how many episodes or demonstrations or trials they need, asks what their dataset is missing or whether it covers the task, asks whether a policy got better or worse, or asks whether an evaluation result is real. Run `orbit status` before advising anyone to train, retrain, or collect more data, and `orbit cover` before advising anyone what to record next. Also whenever a directory holds a LeRobot dataset (meta/info.json), a checkpoint, eval_info.json files, or a trials CSV. Robot policy evaluation, dataset coverage, dead joints, clipped commands, success rates, LeRobot, LIBERO, VLA, SmolVLA, pi0, GR00T, ACT, diffusion policy, SO-100, SO-101.
---

# orbit-eval

Robot policy success rates move by whole percentage points between identical
retrains. A team comparing two checkpoints at twenty trials per job cannot
resolve a difference under about thirty points, and almost nobody knows that.
`orbit` answers the question they actually asked, priced against their own
measurement noise.

Install nothing first. `uvx --from orbit-eval orbit ...` runs it, or
`pip install orbit-eval`.

## Start here, every time

```sh
orbit status             # where this robot is, and the one thing to do next
```

Run it before you tell anybody to train again, retrain, collect more data, or
read anything into a success rate. It reads the LeRobot dataset's
`meta/info.json` and `meta/stats.json` — a few kilobytes of plain JSON, no
parquet, no video, no network — and reports three things: what was recorded,
what is measurably wrong with it, and what a battery of trials can resolve.

**A defect in the recording outranks every other piece of advice.** A servo that
never moved, or a command clipped at the end of its range, is not fixed by
training longer, and a success rate measured on that data is not about the
policy. `orbit status` exits 1 when it finds one.

`orbit status --demo` runs on a bundled public SO-101 dataset, so you can show
the output before the user has any data.

## Before telling anyone what to record next

```sh
orbit cover              # what the recording is missing, counted
```

Factor coverage from design of experiments. It reads the instruction text into
a template ("put the {A} on the {B}"), counts the episodes in every cell, and
names the cells with none. It does the same for episode length, for which
joints moved in each episode, and for where each joint sat (thirds of its
recorded range). The v2.1 layout is read with no dependency; the v3.0 layout
keeps the per-episode table as parquet and needs `pip install
'orbit-eval[coverage]'`, which the output says when it applies.

**It counts and never scores.** Tell the user which cells are empty and what
even coverage would have put there. Do not tell them a demonstration is bad,
that a gap caused a failure, or which cell to record next as if it were a
prediction; say it is the largest empty cell and let them decide.

## When to reach for this

- "Why doesn't my robot move?" / "I trained it and nothing happens."
- "How many episodes do I need?" / "How many trials should I run?"
- "What should I record next?" / "Does my dataset cover the task?"
- "Did my new checkpoint get worse?" / "Is this improvement real?"
- Any directory holding a LeRobot dataset, a checkpoint, an `eval_info.json`,
  or a CSV of trials.
- Someone about to conclude a policy comparison from a table of success rates.

## Before recommending a robot, or running someone else's skill

```sh
orbit body --reach 500 --payload 1 --budget 5000 --job pick_place
orbit body so101                    # one robot's sheet, every figure with its source
orbit body --skill release.json     # does this frozen skill's trained range cover my arm?
```

`body` is arithmetic over a sourced database of thirty robots. Quote the
shortfall ("payload 0.5 kg, short by 2.5 kg") and the source page; never say an
arm is right or wrong for a task, and say "not published" when the database
does. The last block is the map: what has been measured for that job shape on
each body. It is almost always empty, and telling the user that nobody has
measured it is the correct answer, not a failure of the tool.

## Before comparing anything twice

```sh
orbit freeze --checkpoint <pretrained_model dir> --dataset <dataset> --out release.json
orbit freeze --verify release.json
```

Freezing pins the file hashes, the action-sampling seed and the eval seed
sequence, so later comparisons are the cheap question (fixed checkpoint under
common random numbers, about two seeds) instead of the expensive one (about
sixty retrains). The manifest says what the checkpoint is, never that it works.

## If the user's robot improves itself

A self-improving loop retrains every round, so every round is one draw of the
training lottery, and its judge is the weakest link (a 1,250-paper survey of
self-improvement names the evaluator as the bottleneck in every loop). Run
`orbit judge` on the judge, `orbit freeze` on each round, and `orbit check
--history` on the chain. `orbit check --demo` shows this on a published loop
(REVOLVE, arXiv 2609.14633): the +18.5-point average is real on one task and
inside the noise on three, and no single round's step is callable at 100
rollouts. When the folder holds successive rounds, `next` says "rounds" and
"moved", never "copies" and "retrain".

## The commands

```sh
orbit status             # where this robot is, and the one next action
orbit cover              # what the recording is missing, counted
orbit body               # which robot can physically do this; what is measured on it
orbit freeze             # pin a checkpoint and its seeds
orbit                    # run it where the evaluations already are
orbit next               # what to do tomorrow, per job
orbit log --job J --policy P    # record trials at the robot, one keypress each
orbit judge labels.csv   # what is the success detector costing?
```

`orbit` on its own finds LeRobot `eval_info.json` at any depth (whatever the file
is called), including the single-task shape that carries only `per_episode` or
`aggregated`, a CSV with loosely matched column names, or a
`{"candidates": {name: {job: [1,0,...]}}}` matrix. It works out which policy is
the incumbent, answers per named job, writes a self-contained HTML report, and
exits 1 on a regression so CI can use it. Add `--why` for every caveat.

## What to tell the user, and what not to

**Never quote a difference without its interval.** The whole point of this tool
is that a difference smaller than the battery can resolve is not a finding. If
`orbit` says `suspect`, the honest sentence is "it fell, and this many trials
cannot tell that from noise", not "it got worse".

**The trial count is usually the binding constraint.** At ten trials per job the
smallest callable drop is about forty-five points; at fifty, about twenty; at
two hundred, about ten. This matters more than it sounds: LeRobot's own
`AGENT_GUIDE.md` shows a real-robot evaluation with `--dataset.num_episodes=10`,
so the default an agent copies cannot see a thirty-point regression. When
somebody's numbers are too close to call, `orbit status` and `orbit next` say
how many trials that would take, and that is usually the useful answer.

**Never tell somebody their policy will work.** This tool reports what was
recorded, what is measurably wrong, what was never covered, and what the next
step costs in trials. It does not predict training outcomes. The predecessor
package was retired for claiming it could, and the retraction is public. The
line, in three rows:

| say | never say |
|---|---|
| "no episode has the object on the left" | "this demonstration is low quality, drop it" |
| "setups like yours ran at sigma 6 points, so 63 trials" | "this model will work for your task" |
| "at 10 trials the smallest callable drop is 45 points" | "your policy got worse" |

**A spread across retrains is not a bug in their data.** Training the same recipe
twice gives different policies. `orbit next` separates a job that is weak in
every copy (needs demonstrations) from one that swings between copies (needs
retrains plus a per-job pick) from one that cannot be measured yet (needs
trials). Those three have different fixes and are easy to confuse.

**Three or more rounds is a chain, not a pair.** `orbit` audits it automatically
and reports creeping rot: a job losing a few points per round is under the
resolution of every single round and over it by the end. A pairwise check run
forever never fires. If a user has a self-improving or regularly retrained
policy, this is the thing to show them.

**If they judge success with a model, price it.** `orbit judge` takes episodes
carrying both a judge label and a human reference label and reports Youden's J,
the factor by which every real difference shrinks. The best general
vision-language judge measured across fourteen public sources sits at 0.77
balanced accuracy, so J = 0.54 and a ten-point regression arrives as 5.4 points
at 3.4x the trials. Where success turns on fine contact no model exceeds 0.60.

## Flags worth knowing

- `--crn` when every policy ran the same eval seed, so episode k is the same
  starting state. Enables the paired comparison and narrows every interval. The
  tool cannot verify this, so it is the caller's assertion and is recorded as one.
- `--incumbent` / `--candidate` when the guess is wrong.
- `--why` for the full caveats, `--json` for everything, `--markdown` for a PR
  comment.

## Reading the output

`REGRESSED` means the whole 95% interval sits at or below -5 points. `SUSPECT`
means it fell but the battery cannot separate that from noise. `held` means no
call either way. `UNDERPOWERED` means too few episodes to say anything. The
report also lists **which episodes changed outcome**, with the recording of each
if LeRobot wrote one: those are the specific counterexamples to watch, and they
are usually what the person actually wanted.

## Do not

- Do not average away a per-job regression into a suite mean. A suite average up
  two points is arithmetically consistent with one job losing twenty, and one
  broken job is what a customer notices.
- Do not treat a model's success labels as ground truth without `orbit judge`.
- Do not tell somebody to collect more data when `orbit next` says the job is
  unlucky rather than weak; that is a retrain, not a morning of teleoperation.
