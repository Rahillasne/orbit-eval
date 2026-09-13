# orbit-eval

**Measure your sigma once — ~8 replicate retrains — and the gate prices every
ship decision after that.** Robot-policy success rates move by whole
percentage points between *identical-data retrains*; a gate that ignores that
lottery false-shipped 16% of ORBIT's seed-replicate null pairs. orbit-eval is
the statistical decision layer that prices it: calibrate your cell's
`sigma_run` one time, feed it to the gate (`--sigma-run` or an atlas regime),
and every promote/hold verdict after that is priced against measured
retraining noise instead of eval luck.

Honest statistics for robot-policy evaluation. A thin, stdlib-only
pip-installable port of the validated statistics from the ORBIT research
program, shipping the program's **measured noise atlas** as data.

## The one distinction that matters

Every command in this tool keeps two different questions separated, because
they differ in price by more than an order of magnitude:

| | question | noise that governs it | price |
|---|---|---|---|
| **(a)** | *"Did my checkpoint / dataset change move the metric?"* — fixed checkpoints, fixed data, compared under **common random numbers** (CRN: same eval seed → same initial states) | `sigma_0`, the harness floor (PushT/DP 1.91 pp; SmolVLA-ft 20K **0.00** — bit-deterministic) | **~2 eval seeds** |
| **(b)** | *"Is selection method A better than method B?"* — every arm is an **independent retrain**, usually on an independently drawn episode set | `sigma_run` (0.78–11.43 pp measured across the 16 atlas cells; the low extremes are floor-dominated LOW-SR cells, not stability results) and `sigma_set` — **k-dependent**: 19.01 pp at k=22 single-task, falling to ~2.4–2.8 pp (pure) at k≥88 on a multi-task pool | **~59 set draws per arm** for a 10 pp effect at k=22 single-task; **~3–4** at k≥88 multi-task (k=176 re-measured: K4B band B3, pooled σ_run 3.36 df=6; prices unchanged) |

**CRN pairing cannot rescue (b).** Pairing cancels eval-draw noise only;
`sigma_run` and `sigma_set` live in training, not in the eval draw. Any tool
that quotes you a McNemar p-value as evidence about a *selection method* is
lying to you. This one refuses to.

## Install

```bash
pip install orbit-eval
```

or, from a source checkout:

```bash
pip install -e .
```

No dependencies (pure stdlib). Python ≥ 3.9.

## Quickstart (60 seconds)

Price a design before you run it — what can m retrains per arm actually
detect, and how many does your claimed effect need?

```console
$ orbit-eval power --regime smolvla-ft --effect 10
orbit-eval power — MDE = 2.80 * sigma * sqrt(2/m)  (alpha=.05 two-sided, power=.80)
regime: LIBERO / SmolVLA fine-tune k=22 (Tier 2)
  sigma_run = 3.31 pp  (90% chi-square CI [2.28, 6.34], df=6)
  sigma_0 (harness floor, CRN d=0) = 0.00 pp

arms: FIXED-SET — both arms retrain on the SAME fixed episode set,
so only sigma_run separates them (per-draw sigma = 3.31 pp)

  draws/arm    m=1    m=2    m=3    m=5    m=8    m=12   m=20   m=59
  MDE (pp)     13.1   9.3    7.6    5.9    4.6    3.8    2.9    1.7

  to detect a 10.0 pp effect: 2 draws per arm
  power at 2 draws/arm for this effect: 86%
[... provenance + honesty note trimmed ...]
```

Gate a candidate against an incumbent — two LeRobot `eval_info.json` files
(or training output dirs; sequential LIBERO eval declared so CRN pairing is
licensed):

```console
$ orbit-shipgate check incumbent_eval_info.json candidate_eval_info.json --sequential-eval
orbit-shipgate check  incumbent=incumbent_eval_info  candidate=candidate_eval_info
  engine 0.8.1  schema 2  gate-config sha256 62f4a20c1892

mode: CRN-PAIRED — one-sided exact McNemar + paired bootstrap
      (fixed-sample v0; anytime-valid sequential is v1)
  n episodes         500
  SR inc / cand      52.8 / 58.4 pp
  delta (cand-inc)   +5.6 pp
  95% CI             [+3.6, +7.6] pp
  discordant         inc-only b=0 / cand-only c=28 (rate 0.056)
  p_ship / p_worse   3.73e-09 / 1.0000  (one-sided, alpha 0.05)

VERDICT: SHIP -> exit 0
[... assumptions + caveats trimmed — the decision record carries them in full ...]
```

**Calibrate your own cell — this is the product loop.** The atlas prices only
the cells it measured; your stack is not one of them until you measure it.
Run ~8 replicate retrains of your training config (same data, same budget,
new training seeds), evaluate each on a fixed common-seed battery, and take
`sigma_run` from the pairwise gaps (`orbit_eval.stats.sigma_from_gaps`) —
a one-time cost that prices every ship decision after it. Feed it back as
`orbit-shipgate check --level retrain --sigma-run <pp>` (or a custom atlas regime
with an `sr_range`): the gate then prices the retraining lottery into every
verdict, and without it the gate still runs but records the no-prior
disclosure instead of inventing a number.

## Commands

```bash
orbit-eval audit <dir>            # scan eval outputs for silent defects
                                  #   (exit 0 clean / 1 flags or parse errors
                                  #    / 2 unusable path or no result files)
orbit-eval compare <runA> <runB>  # CRN-paired if possible, else unpaired
orbit-eval power --regime smolvla-ft --effect 10 [--arms method]
orbit-eval regress <base> <cand>  # CI gate: exit 0 ok / 1 regression
                                  #   / 2 invalid inputs / 3 underpowered
                                  #   (the design cannot detect the gate —
                                  #    a different remediation than 2)
orbit-eval selftest               # the tool verifies its own size/power/bias
orbit-eval release <log.csv>      # which skills did the new model break? (exit
                                  #   0 clean / 1 regressed / 3 cannot answer)
orbit-eval route build <cands>    # task-conditioned release over candidate
                                  #   checkpoints, with abstention (exit 0 /
                                  #   2 invalid inputs)
orbit-eval route plan --candidates 8 --tasks 10 --select-eps 50
```

`compare` and `regress` accept `--json` for machine-readable output (same
exit codes).

### `audit` — the defects that corrupted real waves

* **seed absent or seed == 1000** — LeRobot's *silent* CRN default: every
  default eval scores initial states 1000..1000+n−1. Great for pairing,
  poison for independence assumptions — either way you need to *know*.
* **truncated budgets** (`final_step != design_steps`) — a crashed run emits a
  complete-looking result with a plausible `sr`. Truncation is **directional**
  (undertrained scores LOW), not mean-zero; it corrupted 12/414 ORBIT runs and
  flipped the sign of one suite's headline result.
* **missing / tiny `n_eval`** — reported with the Wilson CI width it implies.
* **header/vector mismatch** — a file whose per-episode successes disagree
  with its own `overall` header (partial vector, or a header that contradicts
  the episodes) has the vector dropped at parse time and is flagged
  `SR_VECTOR_MISMATCH`: no paired statistic may run on episodes that do not
  correspond to the reported `n_eval`/SR.
* **replicate sets** — detected mechanically via
  `set_hash = sha1(sorted episode ids)[:16]`.

### `compare` — which regime is this number from?

If per-episode successes exist on both runs and the eval seeds match, you get
a **CRN-PAIRED** comparison: exact McNemar on the discordant pairs plus a
paired bootstrap CI on the SR difference. Otherwise you get an unpaired
two-proportion comparison with an explicit warning. Either way the output
states which of questions (a)/(b) the comparison can and cannot answer.

### `power` — both directions of the design rule

`MDE = 2.80 · sigma · sqrt(2/m)` (α=.05 two-sided, power .80), from the
Tier-0 power analysis. `--arms fixed-set` prices retrains on a fixed episode
set (`sigma_run`); `--arms method` prices independent set draws
(`sqrt(sigma_set² + sigma_run²)` → ~59 draws/arm at 10 pp in the k=22
single-task regime; the price is k-dependent and falls to ~3–4 draws/arm at
k≥88 on a multi-task pool) and prints the honesty note.

### `release` — which of your skills did the new model break?

**Start here.** One command, on an evaluation log you already have.

```bash
orbit-eval release evals.csv     # version,skill,episode,success (names auto-detected)
```

It compares the new model against the one it replaces, **per named skill**, and leads
with the answer. A skill is `REGRESSED` only when the whole 95 % interval sits at or
below −5 pp, so the verdict and the interval beside it can never disagree. A drop that
clears the floor but not the interval is `SUSPECT`. A skill with fewer than 10 episodes
is `UNDERPOWERED`: not judged, and explicitly **not** a pass.

Exit codes are meant for CI: `0` checked and clean, `1` a skill regressed or vanished
from the new model, `3` the file could not answer the question, `2` unusable input.

Why per skill and not the suite average: on a 50-task suite a best-validation release
damaged at least one task in 100 % of measured draws while the suite mean got three
times steadier (`research/cleanrel50/`). An average over many skills absorbs one
collapse, and it gets *steadier* as you add skills, not more sensitive.

With three or more versions it reads release history and answers the question a
deployer can act on: how many of your last N releases broke a skill, and how many of
those had a suite average that held or improved.

### `route` — the release step, measured

`cp checkpoint_final prod/` leaves success rate on the table in some cells and
not others. **How much is a measurement about your cell, never a rate to
expect.** Where it paid: eight identical-recipe retrains, same data, same eval —
ship one at random 62.3 %; ship the best-validation one 68.3 %; ship a
**task-conditioned** choice over the same eight 73.9 %, with damaged tasks
falling 1.82 → 0.22 (π₀.₅ / libero_object, held out on disjoint episodes;
`research/ROUTE1_RESULTS_2026-08-29.md`). The shape reproduced on Meta-World
MT10 (+6.9 pp median, 3 replicates) and MT50 (+10 pp).

Where it did not: on SmolVLA-ft / `libero_spatial` the same estimator reads
**+1.7 pp, 95 % CI [−2.1, +5.4]** at 100 selection episodes per task — zero
inside the interval — tracking σ_pertask (3.29 pp there against 6.69 on
`libero_object`, same wave; `experiments/influence/route1_spatial_scope_check.py`).
Routing pays in proportion to the per-task complementarity a cell actually has,
and `plan` exists to tell you which case you are in before you spend anything.

The claim that does travel is about the **tail**, not the mean. Shipping the
best-validation retrain reduced the probability of a damaged task far less than
per-task selection did at every pool size tested — and not at all on the 50-task
pools, where it stayed at 1.000 from J=1 through J=8, while one ten-task
replicate got *worse* with more candidates (0.786 → 0.929). Per-task selection
with abstention cut it 3–8× at J=3 in every ten-task cell, and the pool size
needed grew with the task count (`research/damage_jcurve/`, declaration
`fb1c760f`).

```bash
# candidates/<name>/**/eval_info.json (LeRobot), or a JSON matrix, or your own
# CSV export: candidate,task,episode,success[,block]
orbit-eval route build candidates/ --incumbent final --assume-crn \
    --policy-path final=/weights/final --policy-path s3=/weights/s3 \
    --out release.json --report REPORT.md
orbit-eval route build eval_export.csv --incumbent fleet_incumbents.csv   # per-SKU incumbents
orbit-eval route plan --candidates 4 --tasks 8 --select-eps 20              # before spending
```

What `build` decides and reports:

- **per task**, the candidate that ships: the per-task winner only when its
  advantage over that task's incumbent clears one-sided z ≥ 1.645 on the
  selection episodes, otherwise **ABSTAIN** (keep the incumbent) — routing must
  never itself be a lottery. Abstention is a deliberate trade and the report
  prints both sides: never abstaining is the better arm against the candidate
  *pool*, but it swaps far more of the fleet and makes a swap landing ≥ 10 pp
  below the model it replaced 14× to 48× more likely across the banked cells at
  J = 3. The default protects the incumbent; `--z-abstain 0` takes the other
  side knowingly. `--incumbent` is one candidate, a
  `task=candidate,...` list, or a JSON/CSV mapping (`*` = default), because a
  fleet already runs different models per SKU;
- the **held-out gain**: a stratified split-half over episodes (select on one
  half, score on the disjoint other half, 2000 draws) for RANDOM / INCUMBENT /
  ROUTE / ROUTE+ABSTAIN / ORACLE, with **priced damage counts** (a task is
  damaged when it sits ≥ 5 pp + 2·SE below the candidate-pool mean) and the
  worst task; the in-sample plug-in gain is printed too, labelled optimistic;
- **blocks**: give each episode a block label (robot, day, cell — a `block`
  column in the CSV or `"blocks"` in the JSON). Halves are stratified within
  block and every defection reports the advantage inside each block and whether
  all blocks agree; a defection that reverses sign on one robot is flagged as
  drift-sensitive. Candidates evaluated on different block mixes are refused;
- the **budget**: at your episodes per task, the per-task advantage the rule can
  defect on and the advantage it detects with 80 % power, and for 5/10/15/20 pp
  targets the episodes per task you would need and the total extra episodes
  that implies. At LeRobot's documented 10–20 episodes per task the threshold
  is 26–37 pp, so an INCUMBENT-STANDS verdict there means *unresolved*, and the
  report says so;
- the **regression bound**: for every defection the one-sided 95 % bound on
  "worse than the incumbent" is ≤ 0 by construction of the rule; abstained
  tasks keep the incumbent exactly; the per-task alpha is stated as per-task,
  with the expected false defections under a no-difference null;
- a signed record (`--out`: inputs sha256, decisions, held-out estimate, budget,
  timestamp, argv, and `--policy-path` locations for serving).

**Serving the release.** `orbit_eval.routed_policy.RoutedPolicy` turns the
record into one policy object: task in, the chosen candidate's action out, with
candidates loaded lazily and each carrying its own pre/post processing. For
LeRobot, `examples/eval_routed_lerobot.py` evaluates a release through
`lerobot-eval`'s own rollout loop unchanged (identity pipelines outside, each
candidate's pipelines inside), and checks the per-task outcomes against the
candidates' own evaluations.

```python
from orbit_eval.routed_policy import RoutedPolicy
rp = RoutedPolicy.from_release("release.json", loader=load_my_policy)  # loader(path) -> policy
rp.set_task("libero_object/3"); action = rp.select_action(observation)
```

`plan` costs nothing: given J candidates, T tasks and a selection budget it
prints the same budget lines plus the measured cells' J-curve / budget-curve
rows nearest your design, **stamped with their cell** — they are not a forecast
for yours (no transferable-sigma law). Below ~20 selection episodes per task
the advantage was unreliable in every measured cell.

Scope: task identity must be known at inference; J candidates are held and
served (a fine-tune that freezes its backbone shares it — the SmolVLA retrains
shared 378/500 tensors bitwise, so the routed release is one backbone plus
per-candidate expert modules).

### The measured atlas (`orbit_eval.atlas`)

All **16 measured regimes** ship as data — sigma_run spans **0.78–11.43 pp**
across cells (a ~15× range: pricing from the wrong cell is the exact failure
this tool refuses), and the extremes are LOW-SR floor-dominated cells, not
stability results.

**Suite-level `sigma_run` — DiffusionPolicy / SmolVLA / pi0.5:**

| regime | sigma_run (pp) | df | n_eval | provenance / stamps |
|---|---|---|---|---|
| `pusht-dp` k=103 | **2.09** | 7 | 500 | Phase 1c, 8 noise pairs, old gym_pusht stack |
| `pusht-dp-rebuilt` k=103 | **1.292** | 4 | 500 | **FLOOR-DOMINATED — measurement, not a pricing prior.** Resolved 2026-08-20 pooling all four post-rebuild replicate pairs; sigma sits BELOW its own 1.682 pp binomial floor (169% floor share, floor-subtracted 0.00) — no retrain variance is detectable above sampling. Do not price a gate from it and do not use it as a comparator (`compare_sigma_run` refuses it on floor share); for *pricing*, the corpus policy is max(atlas, EB-shrunk) = **1.80** — a policy choice, not a measurement. Absolute SR shifted −18.2 pp across the rebuild: never compare across it. |
| `libero-dp-k22` | **11.43** | 7 | 200 | Phase 2 |
| `libero-dp-k31` | **7.48** | 5 | 200 | Tier 1 |
| `libero-dp-k44` | **6.24** | 7 | 200 | Tier 1 |
| `smolvla-ft` k=22 | **3.31** | 6 | 200 | Tier 2 fine-tune; truncated-schedule stamp (20k steps / 30k decay) |
| `smolvla-ft-k88-multitask` | **3.35** (df=2, direction-only) / **2.11** re-measured | 2 / 7 | 200 / 2000 | kcurve replicate pairs; 2.11 re-measured at n=2000 on the Leg A′ fresh finals. ANNEAL-1 (2026-08-21, prereg-frozen): completing the LR schedule at this cell is **NULL** — the per-task lottery survives (LOTTERY-SURVIVES, CONCORDANT) |
| `smolvla-ft-pooled` | **2.82** | 8 | 200 | pooled across three suites (goal 1.08 / spatial 1.84 / long10 2.94); spread convention UNKNOWN and no single binomial floor exists for it — `compare_sigma_run` refuses the row; use the per-suite values |
| `pi05-ft` k=22 | **9.74** | 10 | 200 | PI05-PAIRS, 16 runs / 6 masks, within-mask ANOVA; χ² CI [6.81, 17.10]. **Overturns** the cell-1 df=2 reading of 2.02 (CI [1.05, 12.67] — the old point estimate was misleading, its interval was not). Excluding the collapsed mask null04: **5.23** (df=9, CI [3.60, 9.55]) |
| `pi05-ft-k22-multitask` | **6.02** | 7 | 200 | PI05_SCALE W2 |
| `pi05-ft-k88-multitask` | **4.05** | 7 | 200 | PI05_SCALE W1 (production k; 4.23 re-measured at n=1000, suite noise confirmed real) |

**ACT rows (PERTASK-3 legs B1/B2 + ACT-HORIZON)** — every row declares
`sr_range`, the SR band its 8 retrains actually landed in, and Ship-Gate
pre-check 11b refuses to price an eval outside that band:

| regime | sigma_run (pp) | df | SR range (mean) | stamp |
|---|---|---|---|---|
| `pusht-act` (defaults, h=100) | **0.784** | 7 | 0.2–3.0 (1.35) | **LOW-SR**: the binomial floor is ~43% of the variance (floor-subtracted ~0.59) — do NOT read as "ACT is low-noise" outside the near-floor regime |
| `pusht-act-h8` | **4.849** | 7 | 9.2–21.4 (16.35) | eval-horizon stamp: prices retrains *evaluated at* n_action_steps=8 (same checkpoints as `pusht-act`, re-evaluated) |
| `pusht-act-h16` | **4.839** | 7 | 14.6–30.0 (22.1) | selection disclosure: h=16 was the argmax of a one-checkpoint sweep, so its **SR level is optimistically biased** (the sigma is not — h=8, non-argmax, reads essentially the same) |
| `aloha-transfer-cube-act` | **6.515** | 7 | 69.6–88.4 (77.9) | the healthy-SR ACT cell (floor only 8% of variance, floor-subtracted 6.245); common-seed-list UNPAIRED row (G2 pairing unlicensed on gym-aloha) |
| `aloha-insertion-act` | **2.261** | 7 | 9.0–15.4 (12.0) | **LOW-SR**: floor 41% of variance (floor-subtracted 1.734). In *relative* terms the ordering vs transfer-cube reverses (18.9% of mean vs 8.4%) — the pp gap is SR regime, not stability |

**Per-task retrain noise (`sigma_pertask`) — always quoted with its suite.**
PERTASK-4 (T4-2 DIVERGENT, 2026-08-21): on libero_spatial the same
SmolVLA class/k measured 3.29 pp against libero_object's 6.69 pp —
non-overlapping CIs, so sigma_pertask is suite-dependent in magnitude *and*
shape and must never be quoted without naming its suite. Both readings are
always carried: the floor-subtracted binding value and the raw fixed-battery
value beside it.

| regime | suite | sigma_pertask (raw) pp | CI 95% | df |
|---|---|---|---|---|
| `pi05-ft-k88-multitask` | libero_object | **11.00** (11.82) | [6.86, 15.13] | 7 |
| `pi05-ft-k22-multitask` | libero_object | **12.01** (15.33) | [8.70, 15.31] | 7 |
| `smolvla-ft-k88-multitask` | libero_object | **6.69** (7.33) | [5.60, 7.77] | 7 |
| same class/k (PERTASK-4) | libero_spatial | **3.29** (4.52) | [1.18, 5.41] | 7 |

**The pretraining ladder is not monotone.** DP 11.43 → SmolVLA-ft 3.31 → **pi05-ft 9.74**.
The 4B fine-tune is *not* the quiet regime the two-pair reading suggested: PI05-PAIRS resolved
its pre-registered null04 rule on the **seed-attributable** branch — the same episode set scored
50.0 at seed 0 and 87.5 at seed 1, a **37.5 pp gap on byte-identical data**, the largest null gap
in the corpus. Any gate pricing a pi0.5 promotion against the old 2.02 prior was under-pricing
the retraining lottery by ~4.8×.

Plus `sigma_set` — 19.01 pp at k=22 single-task, with the measured k-curve
`SIGMA_SET_PURE_BY_K` (12.47 → 4.67 → 2.75 → 3.86 pp at k=22/44/88/176,
multi-task pool P=454; the k=176 entry uses the K4B-pooled σ_run — it was 2.43
under the original df=2 reading, the verdict flipped on re-measurement, and
both readings are kept) — and the
`sigma_0` harness floors (PushT/DP 1.91 pp; SmolVLA-ft 20K 0.00,
bit-deterministic). At small k, set-level draws are bimodal (a failure mode,
not a Gaussian spread); Gaussian MDEs overstate precision there.

#### Cross-cell comparisons: `compare_sigma_run` refuses before it quotes

A multiple like "ACT is N× noisier than DP" is only a fact about the
policies when both cells are on the same footing — which is why the atlas
never states one in prose. Two functions are the public API:

* `atlas.sigma_run_readings(regime)` — both readings of a row's sigma_run
  with its sampling floor separated out: raw, binomial floor at the row's own
  measured SR and `n_eval`, floor-subtracted, floor share of variance, and
  the row's spread convention.
* `atlas.compare_sigma_run(a, b)` — is a cross-cell multiple quotable, and if
  so what is it? It **refuses** (returns `quotable: False` with explicit
  reasons) rather than printing a number when:
  * either row records no `n_eval` — the raw sigma carries a battery-sampling
    term scaling as 1/√n, so 1.80-at-n=500 and 1.80-at-n=2000 are different
    quantities;
  * either row's `sigma_run_convention` is not a known `'eval-inclusive'`
    (an `'unknown'` convention, e.g. `smolvla-ft-pooled`, cannot be set
    beside a known one);
  * either row records no `df` — a ratio without both dfs has no uncertainty;
  * either row is ≥ 75% sampling floor by variance (its floor-subtracted
    reading is noise, not a measurement — e.g. `pusht-dp-rebuilt`);
  * the two rows' binomial floors differ by more than 2× (SR_SCOPE_FLOOR_RATIO)
    — the ratio would partly be a floor artifact;
  * the ratio's own 95% F confidence interval contains 1 — the cells are not
    distinguishable, and "A is 1.00× B" is not a finding.

  A quotable multiple therefore requires both rows to share `n_eval` and a
  known eval-inclusive convention, carry dfs, sit well above their floors on
  comparable floor scales, and have a ratio CI excluding 1. When any of that
  fails you get the reasons, not a number — quote those instead.

## Ship-Gate (`shipgate`)

A statistical promotion gate for robot policies, built on the same machinery.
`check` is **fixed-sample** — one look at a pre-sized eval; `seq` is the
**anytime-valid sequential** gate (v1) whose alpha survives peeking at every
observation. Input: an incumbent and a candidate eval (LeRobot
`eval_info.json` / `model_result.json`, the corpus per-episode JSONL, or a
LeRobot training **output dir** — the adapter locates
`eval_final/eval_info.json`, else the newest `eval*/eval_info.json`). The
gate verifies CRN comparability, runs the paired verdict engine, applies
winner's-curse handling, and rules with the exit code:

```bash
orbit-shipgate check INCUMBENT CANDIDATE [--ledger gate.jsonl] [--json]
    # exit 0 SHIP / 1 HOLD / 2 INVALID / 3 COLLECT-MORE / 4 UNRESOLVABLE
    # INCUMBENT/CANDIDATE may be LeRobot training output dirs
orbit-shipgate seq STREAM.jsonl --mode episodes|retrains [--tau 5.0] [--ledger g.jsonl]
    # anytime-valid sequential gate (v1): one state line per observation,
    # final decision record to the ledger; exit 0 stream consumed / 2 error
orbit-shipgate replay corpus.jsonl --out DIR    # historical replay -> REPLAY_REPORT.md
orbit-shipgate demo --corpus-dir DIR --out DIR  # compact DEMO.md + verifiable ledger
orbit-shipgate verify gate.jsonl                # recompute every record hash
orbit-shipgate capture OUT_TREE --ledger shadow.jsonl
    # harvest every checkpoint eval in a LeRobot training-output tree into a
    # sha256-CHAINED append-only ledger; idempotent (content-hash dedupe)
orbit-shipgate shadow --ledger shadow.jsonl --incumbent A --candidate B \
    [--decided promote|hold] [--regime R]
    # non-blocking: record what the shipped retrain-level gate WOULD have
    # said next to the team's real decision; always exit 0 unless inputs
    # are invalid
orbit-shipgate shadow --report --ledger shadow.jsonl
    # the divergence table — the kill-criterion instrument
```

* **Pre-checks** — budget gate (truncation is directional), per-episode
  successes present, equal `n_eval`, matching task sets, matching eval seeds
  (unrecorded / mixed-with-1000 seeds pair only under `--assume-crn`, with the
  assumption recorded). On LIBERO, pairing degrades to unpaired unless
  `--sequential-eval` declares a non-vectorized eval (issue #4152).
  **Regime-scope guards**: check 11 rules INVALID when a `--regime`'s
  measured cell verifiably contradicts the eval's own metadata (env, task
  count, trained-k); check **11b** (SR-scope, 2026-08-18) **refuses to price
  an eval whose SR sits outside the regime's declared `sr_range`** — sigma in
  pp carries a binomial component scaling as √(p(1−p)), so when the eval's
  binomial floor differs from every point of the band the regime was measured
  in by more than the 2.0× tolerance (`SR_SCOPE_FLOOR_RATIO`), the verdict is
  INVALID rather than a wrong-cell price (any LOW-SR row — `pusht-act`,
  `aloha-insertion-act` — will exercise this); check **11c** (2026-08-20)
  stamps an **ANTI-CONSERVATIVE PRIOR** warning on the record whenever the
  pricing regime's own sigma_run sits at/below its own binomial floor (e.g.
  `pusht-dp-rebuilt`) — pricing from such a cell *understates* the lottery,
  the one direction this gate exists to avoid. Bad run data yields an INVALID
  record, never a crash.
* **Primary test** — one-sided exact McNemar on the discordant pairs
  (candidate beats incumbent), `delta = candidate − incumbent` in pp with a
  paired bootstrap CI; degraded comparisons use the unpaired two-proportion
  test and Newcombe CI and say so.
* **Winner's curse** — if the candidate was picked as a max over J
  checkpoints, the measured +4.0 pp mean inflation (J=4, SmolVLA-ft,
  PI05_CELL1_RESULTS, measured prospectively) is subtracted from the
  effective delta; other J scale as `4.0·sqrt(ln J / ln 4)` pp, disclosed as
  an assumption in the record. With `--history` (the checkpoint evals) the
  bias is bootstrapped from the data instead. The correction gates SHIP only;
  it never converts a loss into a HOLD escape.
* **min effect 2.0 pp default** — the corpus-measured resolvability floor:
  delta < 2 pp resolves at 42–63% accuracy at every budget tested — a coin
  flip (ADAPTIVE_EVAL_PROGRAMME).
* **Per-task regression** — a task whose paired drop is ≥ 5 pp at one-sided
  p ≤ 0.05 HOLDs the release even when the overall delta is a win. Task
  blocks default to contiguous equal blocks (LeRobot orders eval episodes by
  task_id — recorded as an assumption); pass `--task-blocks` to override.
  **Retrain-priced since 2026-08-15** (HOLDINJ, prereg-frozen): at
  `comparison_level='retrain'` with a measured per-task σ (explicit
  `sigma_pertask_pp` or an atlas regime carrying `sigma_pertask` — measured:
  π₀.₅ / **libero_object** / k=88 at 11.00 pp, raw 11.82 beside it;
  sigma_pertask is suite-dependent — PERTASK-4 T4-2 DIVERGENT: libero_spatial
  measured 3.29 for the same class/k, so it must always name its suite), the
  governing per-task p prices the
  retraining lottery; the unpriced eval-exact rule HOLDs **96.4% of
  identical-data retrain pairs** at production k (HOLDRATE_DIAG_2026-08-15),
  the priced rule is calibrated (null flag rate 0.043 ≈ α; validated
  HOLDINJ_RESULTS). Verdicts are two-tier: `regression-beyond-lottery`
  (gate-actionable) vs `lottery-expected` (real difference a no-change
  retrain produces anyway); 5–20 pp regressions are below single-pair
  resolution (≈39 pp at σ=11) and route to COLLECT-MORE quoting the
  retrain-draw budget (≈15 retrains/side at 10 pp, ≈4 at 20 pp).
* **COLLECT-MORE sizing** — episodes needed for 80% power at the minimum
  effect, from the observed discordance rate (`var(delta_hat) ~ p_d/n`);
  UNRESOLVABLE when that exceeds `--max-budget`.
* **Sequential gating (`orbit-shipgate seq`, v1)** — e-process gating whose alpha
  is honest at *any* stopping time (Ville's inequality): `--mode episodes`
  consumes `{"inc":0/1,"cand":0/1}` CRN episode lines and runs a
  betting-mixture e-process over the discordant pairs; `--mode retrains`
  consumes `{"delta_pp":..,"se_eval_pp":..}` per-retrain-pair lines and runs
  a one-sided half-normal mixture e-process (prior scale `--tau`, default
  5.0 pp, disclosed) on `sigma_j² = 2·sigma_run² + se_eval_j²`, with the
  `sigma_run` prior resolved exactly as in `check` (explicit `--sigma-run` >
  `--regime` > none-with-disclosure). Evidence-to-ship when the running
  e-value reaches `1/alpha` and the corrected effect clears the minimum;
  mirror process for the worse direction; always-valid p-values and
  confidence sequences ride along in every emitted state line (`--every N`
  to thin). Deterministic — no RNG anywhere in the sequential path. The
  statistics live in `orbit_eval.gate.sequential` (Robbins normal-mixture
  martingales; Ramdas et al. safe anytime-valid inference).
* **LeRobot output dirs** — `check` positional args may be training output
  dirs (`eval_final/eval_info.json` preferred, else the newest
  `eval*/eval_info.json`, "newest" = the largest step embedded in the dir
  name — deterministic, no mtime); `--history` may be a run dir whose
  `checkpoints/<step>/eval_info.json` files form the full selection pool
  for the history-bootstrap curse measurement.
* **Decision records** — every verdict is a schema-versioned record carrying
  the full config, statistics, caveats, warnings, and assumption strings,
  hashed over canonical JSON (`record_sha256`, `gate_config_sha256`) and
  appended to an append-only JSONL ledger. `orbit-shipgate verify` recomputes every
  hash and trusts nothing on disk.
* **Replay** — `orbit-shipgate replay` runs corpus per-episode JSONL through the
  gate: null-pair mode (seed replicates trained on identical data — every
  SHIP is a false ship, reported with a Wilson CI against the configured
  alpha *plus* the disclosed dependence structure, a cluster bootstrap CI
  over replicate groups, the retraining noise the null gaps imply beside
  the atlas `sigma_run`, and a regenerated exact-null calibration control),
  curse mode (max-over-checkpoints verdicts with and without correction),
  and effect mode (different-training-set pairs: resolved vs COLLECT-MORE,
  at what episode budgets). Pair keys include the environment **stack
  epoch** (the measured `gym_pusht` rebuild), so cross-rebuild pairs are
  never formed — excluded pairings are counted in the report, and
  cross-wave pairs within an epoch carry a recorded assumption. Every
  number in the report is computed, none asserted.
* **Demo** — `orbit-shipgate demo --corpus-dir DIR --out DIR` replays a corpus
  directory and renders a compact `DEMO.md`: the headline eval-only vs
  retrain-aware false-ship table, three verbatim decision records, and the
  ledger-verify line — every number computed from the corpus at render
  time, over a fresh append-only ledger that `orbit-shipgate verify` re-checks.
* **Shadow-gating (`capture` / `shadow`)** — the deployment mode that costs
  a team nothing. `capture` walks a LeRobot training-output tree (the same
  `eval*/eval_info.json` discovery `check` uses) and appends one record per
  checkpoint eval: per-episode successes, eval seed **with provenance**
  (recorded in `eval_info.json`, the run dir's `model_result.json` sidecar,
  or null with LeRobot's silent default disclosed as
  `lerobot_default_1000_implicit` — never written down as if it had been
  recorded), steps, `set_hash`, and the G3 ordering metadata (task count,
  per-task episode counts, block order) that decides matched-config
  pairability. The ledger is **sha256-chained**: each record carries
  `prev_sha256` of the previous line (genesis 64 zeros), so a line cannot
  be altered, removed, or reordered without breaking the chain
  (`orbit_eval.gate.verify_chain` recomputes every link); capture is
  idempotent — re-capturing the same tree appends nothing. `shadow` then
  runs the shipped retrain-level gate on a captured pair **next to the
  team's declared decision** (`--decided promote|hold`) and records the
  divergence; it refuses via the verdict-INVALID path — recorded, still
  exit 0 — when the G3 metadata says the pair is not matched-config
  (different task sets, block order, or per-task counts: mismatches the
  engine's total-n check cannot see). Non-blocking by contract.
  `shadow --report` prints the divergence table (decisions, agreements,
  gate-said-HOLD-team-promoted = **reversed ship decisions**,
  gate-said-SHIP-team-held) — the literal instrument for the thesis kill
  criterion (*zero reversed ship decisions in 3 months of shadow-gating*).

## Runtime caveats baked into the output

Each of these cost the research program real GPU-days:

* LIBERO env `task_ids` ≠ dataset episode order — match the task **language
  string**, or you train one task and eval another (SR ≈ 0).
* `use_async_envs` must be **false** for reproducible seeding.
* LIBERO vectorized autoreset makes the init-state sequence outcome-dependent
  under early termination (LeRobot issue #4152) → CRN pairing on LIBERO
  requires **sequential** eval.
* Cross-stack comparisons are invalid: a `gym_pusht` rebuild shifted SR by
  −18.2 pp (measured, non-uniform).
* Episode-index → initial-state alignment was **validated 2026-08-02** (6/6
  comparisons, lerobot 0.6.0): reset-time state at episode index *i* is
  identical across fresh processes, sync/async vector envs, and batch sizes,
  for gym_pusht and LIBERO. Scope: the reset-time mapping — on LIBERO, full
  closed-loop evals with early termination still shift later init states
  (issue #4152; see the autoreset bullet above).

## Provenance

Ported from the ORBIT research repo: `analyze_paired.py` (budget gate,
bootstrap/randomisation machinery, the self-test pattern), `power_paired.py`
(generative model), `build_atlas.py` (set_hash, replicate detection),
`fit_forecaster.py` (stdlib stats helpers), `power_analysis_phase2.py`
(MDE design rules), `OPS_RUNBOOK.md` (the defects).

## Licensing

`orbit-eval` is released under the **Apache License, Version 2.0** — see
[`LICENSE`](LICENSE).

Use it, ship it, embed it, modify it, build a product on top of it. There is no
source-disclosure obligation, no separate commercial licence to negotiate, and
the patent grant is explicit.

It was AGPL-3.0-only for versions 0.8.1 and 0.8.2. That licence obliged anyone
offering a modified version over a network to publish their changes — which is
precisely the adoption this tool needs and cannot buy. Relicensed at 0.8.3.

The measurement corpus behind the shipped atlas — the replicate-retraining runs,
the pre-registration record, and the per-cell variance estimates — is a separate
work and is not covered by this license.
