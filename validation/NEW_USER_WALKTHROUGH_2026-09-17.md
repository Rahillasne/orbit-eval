# A stranger with a robot: every command, cold, 2026-09-17

Method: `orbit-eval` 0.17.0 wheel installed into an empty virtualenv on a
machine with no LeRobot, no pyarrow and no data. A folder `my_robot/` holding
only the metadata of the official SO-101 example (`meta/info.json`,
`meta/stats.json`) stands in for a newcomer's recording. Every command was run
as written below and its output read as a first-time user would read it.

## What worked without explanation

| command | what a newcomer sees | verdict |
|---|---|---|
| `orbit` in the folder | where you are, two clipped joints named, the one next action, the other commands | clear |
| `orbit --help` | nine commands, one line each; the seven legacy commands are hidden and named once in the footer | clear after the fix below |
| `orbit cover` (v3.0, no pyarrow) | "the per-episode table is parquet ... `pip install 'orbit-eval[coverage]'`" | clear; the one dependency is named at the moment it matters |
| `orbit body` | thirty robots, id / class / reach / payload / price / driver / what is on the map | clear |
| `orbit body so-101` | the sheet with a source URL per figure, "not published" for reach and payload, 13 public skills, none measured | clear, and honest |
| `orbit body so10` (typo) | "no robot called 'so10'. `orbit body` lists the 30 known." exit 2 | clear |
| `orbit body --reach 500 --payload 1 --budget 5000 --job pick_place` | 1 passes (PiPER), 19 cannot be checked, 10 fail with the shortfall in units | clear; the long "cannot be checked" list is the truth about spec sheets |
| `orbit freeze` with no checkpoint | one line saying what to pass | clear |
| `orbit check` with no evaluations | what it looks for, in three shapes, and `orbit log` for trials on paper | clear |
| `orbit check --demo` | a published loop, five rounds, one task improved beyond noise, three inside it | clear |
| `orbit status --json`, `orbit cover --json` | machine-readable, same numbers | fine |
| `orbit statsu` (typo) | argparse's invalid-choice list | acceptable |

## What was rough, and what changed

1. **`orbit check` in a dataset folder listed `info.json` as a near-miss
   evaluation** ("JSON without per_task successes"). A newcomer reads that as
   "my dataset is broken". A dataset's own metadata files are now never
   reported as near misses.
2. **`orbit next` with nothing to compare spoke as `orbit check`.** It now
   speaks as itself.
3. **`orbit body --skill release.json` with no manifest printed a raw
   `[Errno 2]`.** It now says there is no manifest at that path, how `orbit
   freeze` writes one, and that a downloaded skill should ship with its own.
   The same for a missing dataset under `--path`.
4. **An empty `meta/episodes.jsonl` hid a real v3.0 parquet table** from
   `orbit cover`. A JSON-lines file with no rows now falls through to the
   parquet path.
5. **`--help` was a wall of seventeen commands.** The seven older commands
   (`release`, `audit`, `compare`, `power`, `regress`, `selftest`, `route`)
   still run and still have their own `--help`, but no longer appear in the
   list; the footer names them once.
6. **`orbit body`'s list truncated ids** (`trossen_wx`) and did not show that a
   body with no measured cell can still have public skills. The column is wider
   and reads "none, 13 public skills".

## What is still true and was left alone

- `orbit` and `orbit status` exit 1 when the recording has a blocking defect.
  That is the design: a shell script that gates training on it should stop.
- `orbit log` and `orbit judge` with no arguments print argparse usage. They
  need a job name and a file respectively; there is no sensible default.
- The "cannot be checked" list in `orbit body` is long because most makers do
  not publish reach or payload for hobby arms. Printing "not published" is
  the honest output; inventing a number is the thing this company retired a
  package for.
- `orbit cover` on a v3.0 dataset without pyarrow cannot count per-episode
  factors. It says so and names the one install that fixes it. The base
  install stays dependency-free, which is what makes the cold start 0.65 s.
