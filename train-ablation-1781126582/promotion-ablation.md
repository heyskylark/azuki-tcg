# Promotion evaluation, archive ablation, and league optimization

Status: retrospective implementation, evaluator/SPS validation, and the
15M-step Task-3-derived shadow are complete. The monitored archive-admission
continuation is running. Automatic production-anchor replacement remains
disabled.

Last updated: 2026-07-17.

This document turns the promotion-redesign discussion in `final-report.md`
sections 26.4, 27, and 28.9 into an ablation and rollout plan. Promotion is
currently useful as matchup-cycle telemetry, but it is not a reliable label
for overall model improvement. The redesign must establish a valid decision
rule before optimizing the evaluator or allowing it to mutate the league.

## Decision summary

1. Redefine promotion as admission to a protected quality archive, not an
   automatic replacement of one champion by another.
2. Evaluate each candidate against a frozen panel of four unique policy
   opponents with deterministic, seat-paired, gate-balanced games.
3. Use pooled strength together with matchup, seat, timeout, and external
   non-inferiority checks. Do not use a single hard champion veto.
4. Keep reference decks measurement-only in v1. They are an external
   yardstick to beat, never an imitation target or a training reward.
5. Validate the rule retrospectively, then in shadow mode. Only after its
   decisions are credible should archive mutation be enabled.
6. Optimize the validated evaluator with native/vector execution while
   preserving the matchup protocol and raw telemetry.

## Implemented v1 result

The retrospective phase changed the proposed decision rule in an important
way. Absolute panel thresholds are invalid for this payoff surface: the Task 3
production anchor scores only 48.4% and 45.3% against panel v1 on the two
predeclared paired schedules. An absolute 52% admission floor therefore rejects
the anchor that defines current production quality. The selected P4 rule is
anchor-relative, with separate absolute floors used only to catch catastrophic
failures.

The immutable evidence is under
`results/promotion_ablation_v1/`. The policy-pair matrix contains 13 policies,
two paired seeds, 182 matchup artifacts, and 5,824 games. Eight games reached
the retrospective 400-step timeout. Excluding those censored games, step count
was p99=325, p99.9=383, and max=396, supporting the live 600-step cap and a 1%
timeout allowance rather than treating every rare long game as a loss of model
quality.

Panel v1 is frozen in `results/promotion_ablation_v1/panel-v1.json`:

| Role | Policy | Selection evidence |
| --- | --- | --- |
| Production anchor | Task 3 final | Confirmed campaign checkpoint |
| Recent quality | S14 final | Most recent prior qualified endpoint |
| Hardest retained | Task 2 final | Anchor score 29.7% in the payoff matrix |
| Historically distinct | S13 final | Largest eligible payoff-vector distance, 0.1094 |

Task 2 is deliberately a hard evaluation opponent but is not marked
quality-qualified. Panel membership and quality admission are different
concepts.

At refresh time, a bootstrap member's frozen `anchor_score` remains eligible
as hardness evidence until a live payoff observation for that same member
supersedes it. A new checkpoint replaces the hardest-retained member only when
its measured anchor score is actually lower; merely being the first policy in
the new live payoff table is insufficient. The one-member-per-refresh rule
still applies.

### Selected anchored rule

The live confirmation decision compares the candidate's 128 panel games with
the production anchor's cached games on the identical schedule. It requires:

- pooled score delta at least -2 percentage points;
- at least three of four opponent deltas at least -5 points;
- at least two of four opponent deltas at or above break-even;
- no opponent delta below -10 points;
- neither seat delta below -8 points;
- paired 80% bootstrap lower bound on the pooled delta at least -5 points;
- absolute catastrophic floors of 40% pooled, 20% per opponent, and 35% per
  seat;
- candidate timeout rate at most 1% and no more than 1 point above the anchor;
- self-controlled reference score at least 50%;
- reference delta at least -15 points with its paired 80% lower bound at least
  -18 points; and
- two fixed reference seeds, totaling 288 games per policy.

The screen uses the same structure with intentionally looser anchored floors:
pooled delta -8 points, opponent delta -20, seat delta -15, and absolute floors
of 35%/15%/25%. A screen pass only authorizes confirmation.

The external-leap route can waive the ordinary relative pooled, quorum,
break-even, relative-confidence, and reference-noninferiority checks after a
confirmed reference improvement of at least 5 points. It cannot waive the 50%
reference floor, catastrophic panel/seat/opponent floors, worst-opponent
relative floor, or timeout checks.

### Reference controls

The primary live reference comparison is self-controlled: each checkpoint is
loaded for both policy seats while one seat receives the fixed reference deck.
This prevents the candidate's score from being confounded by having the
production anchor control the reference side. The older fixed-anchor variant
is retained as diagnostic telemetry, and the odd-indexed nine-deck split is a
periodic holdout rather than another live veto.

This distinction is necessary. The fixed-anchor comparator nearly admits Task
2, while the self-controlled primary reference exposes its regression. The
holdout by itself also cannot cleanly separate Task 2 from Task 3 ep2500, so it
must remain an audit rather than replacing the multi-axis live rule.

### Retrospective sentinel result

| Candidate | Panel score | Panel delta | Primary reference | Reference delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Task 1 final | 39.8% | -8.6 pt | 27.4% | -37.8 pt | Reject |
| Task 2 final | 66.4% | +18.0 pt | 45.1% | -20.1 pt | Reject |
| Task 3 ep2500 | 52.0% | +3.5 pt | 54.5% | -10.8 pt | Admit |
| Task 3 final | 48.4% | 0.0 pt | 65.3% | 0.0 pt | Admit |
| Task 4 final | 52.3% | +3.9 pt | 24.0% | -41.3 pt | Reject |
| S14 ep2000 | 27.7% | -20.7 pt | 85.1% | +19.8 pt | Reject |

All five predeclared Task 1-4 sentinel expectations match on both panel seeds
and in the combined live rule. S14 ep2000 is not merely a stale-champion false
negative: its external performance is excellent, but it falls below broad
panel, opponent, and seat catastrophic floors. It should remain available as
strategic evidence, but admitting it to the production-quality archive would
erase a real matchup collapse.

Twenty-four of 81 threshold combinations in the predeclared local sensitivity
grid reproduce every sentinel decision, and reference candidate floors of
48%, 50%, and 52% give the same sentinel results. The selected defaults sit
inside that stable region rather than on a single threshold edge.

Training opponent sampling remains randomized and unchanged throughout this
ablation. Promotion evaluation must not silently change the PPO training
distribution.

## Confirmed parent model

All promotion work builds on the latest confirmed model, not the raw defaults
in `python/config/azuki_deckbuild_native_3090.ini` and not an older ablation.
The parent is:

`experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt`

Its stack is the complete S14 production stack:

- reward-shaping anneal from 1.0 to 0.15 with the existing 12/40 episode
  warmup/ramp schedule;
- portal-GP bonus 0.3;
- cross-gate masking/replay 0.15;
- PFSP and effective frozen ratio 0.4;
- league retention 6/4/3;
- early-tempo bonus 0.1 with cap 4;
- damage-mitigation bonus 0.15 with cap 10;
- text-only gates, pick smoothing 0.02, same-element oversampling 0.35;
- seed 42;
- temporary-Charge realization bonus 0.08; and
- temporary-attack realization 0.025 per effective damage, capped at 4.

Do not add the rejected entity-damage, generated-IKZ-conversion, or
paid-response-availability rewards. Offline promotion replay does not require
training. If a live shadow run is needed, it must resume from or exactly
reproduce this Task 3 stack and record the precise parent checkpoint.

## Why the current gate fails

The present rule is anchored to one champion and two scalar-Elo baselines.
With the current configured episode counts and `z=1.0`, its nominal thresholds
are substantially stricter than their labels imply:

| Check | Games | Configured lower bound | Minimum passing record |
| --- | ---: | ---: | ---: |
| Quick champion | 32 | 0.55 | 21/32 = 65.6% |
| Full champion | 48 | 0.55 | 30/48 = 62.5% |
| Each baseline | 32 | 0.45 | 18/32 = 56.3% |

The candidate must also pass every baseline independently. A full evaluation
runs a new 48-game champion match instead of reusing the 32 quick games.
Current evaluation alternates seats, but each seat receives a different seed
and randomly sampled gates, so the observations are not paired controls.

The implementation has additional measurement problems:

- the champion can remain a stale early checkpoint in an intransitive meta;
- baselines are selected by a scalar Elo whose update treats an aggregate
  match as one rating game rather than recording every episode;
- the legacy serial evaluator samples policy actions stochastically;
- `MatchResult` retains only aggregate wins/losses/draws, so seat, gate,
  seed, timeout, and matchup failures cannot be audited; and
- serial gates have paused recent 45M runs for roughly 15-23 minutes even
  though steady rollout SPS remains healthy.

The evidence is not simply that the gate is too strict. It often measures a
real head-to-head relationship on the wrong strategic axis. S14 reached a
campaign-best external checkpoint while candidate-vs-champion performance
fell, and Task 3 improved broadly while receiving oscillating legacy
promotion decisions. Lowering the current threshold alone would preserve the
stale-champion and luck problems.

## Goal and non-goals

The goal is to admit policies that are competitively robust across the live
meta, retained past strategies, seats, gates, and an external reference
yardstick despite draw luck and intransitive matchups.

The first version does not try to prove that policies have a total ordering.
It also does not redesign PFSP, add reference-seat training, change reward
shaping, optimize the legacy decision rule, or use action/deck similarity to
human decks. Those are separate experiments.

## League state semantics

Maintain distinct concepts instead of overloading `champion_policy_id`:

- **Quality archive:** four to six protected policies that passed the
  validated admission gate.
- **Production anchor:** the currently selected best checkpoint for external
  use and the first opponent in the promotion panel.
- **Recent training pool:** normal checkpoints available to PFSP and recency
  retention; failure to enter the quality archive does not make a checkpoint
  unusable for training.
- **Diversity archive:** deferred. A later payoff-matrix ablation may retain a
  strategically distinct policy that is useful for training without calling
  it production quality.

An admitted candidate joins the quality archive but does not automatically
replace the production anchor. A compatibility champion pointer may mirror
the production anchor for existing code, but the pointer must not remain the
sole promotion opponent or quality label.

## Four-policy panel v1

Every panel contains four unique policy checkpoints with explicit roles:

1. the production anchor;
2. the most recent quality-qualified policy;
3. the hardest retained opponent according to the raw payoff matrix; and
4. a historically distinct qualified policy, selected by payoff-vector
   distance rather than behavior-rate novelty.

Fallbacks must be deterministic when the archive is young. Prefer a
well-separated historical checkpoint, then the oldest active qualified
policy. Do not fill the panel with several highly correlated checkpoints from
one short trajectory merely because their Elo differs.

Bootstrap the first panel from the retrospective payoff matrix: keep Task 3
as the production anchor, then choose the other roles from historically
supported checkpoints after the matrix is measured. Record this as panel
version 1. Do not seed the initial panel from the current scalar-Elo ordering
or choose all four members by hand before seeing their matchup coverage.

Freeze a panel version for approximately 600 training epochs. At a refresh,
replace at most one member and never remove the production anchor implicitly.
Panel membership, role, checkpoint hash, selection evidence, activation
epoch, and retirement epoch must be persisted. The 600-epoch interval and
one-member replacement rule are initial proposals to validate, not hidden
constants to bake into the first implementation.

## Deterministic paired match protocol

Quality evaluation uses explicit legal-action argmax with deterministic tie
breaking. It must not inherit training-time sampling temperature or smoothing.
The candidate and opponent are frozen in eval mode.

### Screen: same-gate pairs

Run 16 games against each panel opponent:

- one two-game block for each of the eight gates;
- both policies receive the same forced gate within a game;
- game A places the candidate in seat 0 and the opponent in seat 1;
- game B swaps the policies between seats; and
- both games use the same episode seed and gate.

This yields 64 screen games for a four-policy panel. Every candidate is tested
once in each seat at every gate against every opponent. The paired block, not
an individual game, is the independent unit for uncertainty estimates.

### Confirmation: balanced cross-gate pairs

A screen pass adds 16 games per opponent using a fixed eight-edge cycle:

1. Surge vs Stormchain;
2. Stormchain vs Ragefire;
3. Ragefire vs Rushfire;
4. Rushfire vs Devotion;
5. Devotion vs Stonehaven;
6. Stonehaven vs EchoedWaves;
7. EchoedWaves vs Hydromancy; and
8. Hydromancy vs Surge.

For each edge, game A gives the candidate seat 0 and the first gate. Game B
gives the candidate seat 1 and the second gate while the opponent receives
the opposite assignment. The pair shares an episode seed. Across all eight
edges, each policy receives every gate twice and every gate once in each
seat. The schedule intentionally contains four sibling and four cross-element
matchups and must be versioned.

Confirmation reuses all screen results. The panel total is therefore 128
games, not 64 discarded games followed by a fresh 128. A second independently
seeded schedule is used during retrospective validation, not automatically
for every live candidate.

## Raw result contract

Store one record per completed game with at least:

- run, candidate, opponent, panel version, and schedule version;
- screen/confirmation phase and paired-block identifier;
- episode seed and any explicit battle/world seed;
- candidate/opponent seat and gate;
- policy action mode and checkpoint hashes;
- winner, draw, steps, truncation, and timeout reason; and
- evaluator implementation/version and wall-clock duration.

Summaries must expose pooled score, each opponent, both seats, every gate,
same-gate versus cross-gate blocks, timeout rate, confidence result, and the
exact reason for admission or rejection. A draw scores 0.5 for strength
statistics but remains separately visible. Historical decisions must be
recomputable from immutable raw records without rerunning games.

## Original provisional decision rules (superseded)

These values were the hypotheses frozen before retrospective calibration.
They are retained to make the ablation auditable, but they are superseded by
the selected anchored rule above and are not the configured live constants.

### Cost-control screen

Continue to confirmation when all of the following hold over 64 games:

- pooled panel score is at least 48%;
- no opponent score is below 25%;
- neither seat shows a severe failure; and
- there is no timeout or truncation regression.

The screen only saves compute. Passing it is never sufficient for archive
admission, and early stopping may occur only after a complete balanced block.

### Standard archive admission

After all 128 panel games, require:

- pooled panel point estimate at least 52%;
- at least three of four opponent scores at least 45%;
- at least two of four opponent scores at least 50%;
- no opponent score below 35%;
- both seat scores at least 40%;
- no meaningful timeout/truncation regression; and
- external reference performance no more than 5 percentage points below the
  production anchor on the same fixed schedule.

Do not require a confidence lower bound itself to exceed 52%; that would
recreate the current hidden 62-66% hurdle at these sample sizes. The primary
confidence candidate to backtest is a stratified paired-block bootstrap whose
80% lower bound is at least 48%. A Bayesian paired-block rule such as
`P(panel mean > 50%) >= 0.80` is a comparator arm, not a second simultaneous
veto. Select one only after calibration and sensitivity analysis.

### External-leap route

A policy with a novel strategy may be strong externally without immediately
beating most of the current policy panel. A separate route may admit it when:

- it improves at least 5 percentage points over the production anchor on a
  fixed, seat/deck-balanced reference schedule with at least 96 games;
- its pooled panel score is at least 48%;
- no opponent or seat falls below the standard catastrophic floors; and
- the improvement repeats on the second seed schedule or clears a calibrated
  paired confidence condition.

This route is deliberately not a waiver for a narrow seat, gate, or timeout
exploit.

## Reference-deck role

Reference decks remain outside the four-policy panel. Use a designated
promotion-reference split for repeated measurement and retain a separate
holdout split for periodic audit. The existing 9/9 split can seed this design,
but the deck indices, deck hashes, seat counts, and seeds must live in a
versioned manifest.

The production anchor's result on a fixed schedule can be cached; every
candidate is evaluated on the identical schedule so the relevant statistic
is the paired delta. If a nine-deck split is used, prefer a game count that
balances every deck and seat exactly, or document the deterministic remainder
rather than relying on random deck draws.

No reference result creates an imitation loss, reward, forced draft target,
or action-similarity objective. Enabling the dormant S4 reference-seat
training mechanism is not part of this promotion ablation.

## Production-anchor selection

Archive admission and production selection answer different questions. The
production pointer should change less often and continue using the campaign's
checkpoint-selection evidence: windowed external reference performance is
primary, while policy-panel/H2H robustness is the secondary agreement check.
Confirm a proposed production peak at `n >= 288`, check both seats and
elements, and reject catastrophic panel regression before replacing the
anchor. An archive member may therefore remain valuable without becoming the
new production checkpoint.

## Retrospective ablation matrix

Generate the richest raw matchup matrix once, then score multiple rules from
the same paired games where their protocols permit it:

| Arm | Opponents | Pairing | Robustness checks | Reference check |
| --- | --- | --- | --- | --- |
| P0 | Current champion + Elo baselines | Legacy unpaired | Current hard vetoes | None |
| P1 | Current opponents | New paired schedule | Current hard vetoes | None |
| P2 | Frozen K=4 panel | New paired schedule | Pooled score only | None |
| P3 | Frozen K=4 panel | New paired schedule | Quorum, floors, seats | None |
| P4 | Frozen K=4 panel | New paired schedule | Quorum, floors, seats | Non-inferiority + leap route |

P4 is the proposed design, but it is selected only if the backtest supports
it. P1 isolates the value of controlling seat/gate luck; P2 isolates panel
breadth; P3 measures whether quorum and floors prevent specialist false
positives; P4 tests whether the external anchor resolves internal/external
divergence.

### Historical checkpoints

Backtest all available S9-S14 and reward-campaign windows, with these sentinel
expectations:

- Task 3 ep2500 and final should qualify;
- S14 ep2000 should not fail solely because it loses to the stale ep100
  champion;
- Task 1 and Task 2 endpoints should not qualify;
- Task 4 should remain rejected; and
- adjacent checkpoints should not oscillate between accept/reject merely
  because their seats or gates were sampled differently.

These are sentinel checks backed by the broader campaign evidence, not a tiny
supervised label set to overfit. Also compare rule scores with external
reference rankings, within-run retention H2H, per-seat/per-element breadth,
timeouts, and repeated trajectory windows.

### Selection metrics

Choose the simplest rule that performs well on:

- agreement with the broader historical quality assessment;
- rank correlation with external reference results without blindly copying
  that ranking;
- decision stability on a second paired-seed schedule;
- lower adjacent-checkpoint flip rate than P0;
- sensitivity to dropping one panel opponent or one gate block;
- correct detection of seat/gate/matchup collapses; and
- evaluation games and wall time per useful decision.

Thresholds must receive a sensitivity table. A rule that works only at one
exact percentage is not ready for live control.

## Rollout order

Steps 1-5 are complete as of 2026-07-17. Step 6 is the next controlled
experiment; steps 7-8 remain intentionally deferred.

1. Freeze the checkpoint set, panel manifests, two schedule manifests,
   reference split, metrics, and provisional rules before reading new results.
2. Add a replay-only evaluator and verify deterministic reruns and complete raw
   telemetry. It must not update league state.
3. Run the retrospective matrix and threshold sensitivity analysis. Document
   every sentinel miss and select the rule before live training.
4. Implement the optimized evaluator only after the matchup semantics are
   fixed. Prove one-env versus vector-native invariance on forced schedules and
   compare aggregate behavior with the legacy evaluator.
5. Run one Task-3-derived 15M shadow period. The new gate logs decisions but
   cannot admit, evict, or retarget training policies; the existing league
   state remains authoritative.
6. Enable quality-archive admission for one monitored run while production
   anchor replacement remains review-gated.
7. Automate production-anchor changes only after archive decisions and
   reference rankings remain stable across the monitored run.
8. Revisit diversity retention and payoff-matrix-driven PFSP only after the
   promotion signal itself is validated.

Do not simultaneously change promotion semantics, opponent sampling, reward
shaping, and retention. That would make both quality and performance results
unattributable.

## Evaluator optimization plan

The semantic prototype may be slow, but the live evaluator must not preserve
the legacy serial bottleneck. Use the available 12 CPU cores and RTX 3090 as
follows:

- native vector environments, initially 8-12 concurrent games;
- batched policy forwards across currently acting seats;
- keep the candidate resident on the GPU and cache/load one panel opponent at
  a time to control memory;
- reuse screen games during confirmation;
- stop only at balanced-block boundaries; and
- keep evaluation synchronous so PPO is paused instead of competing for GPU
  time and depressing rollout SPS.

Correctness requirements precede speed: fixed schedules must be deterministic
within the chosen native path; serial-native and vector-native runs must
produce identical per-game outcomes; and raw records must agree before
accepting a benchmark.

Performance targets on this machine:

- 64-game screen in under 3 minutes;
- cumulative 128-game panel confirmation in under 8 minutes;
- reference results cached for the anchor and candidate-only evaluation
  batched into the same gate where practical;
- no more than 1-2% regression in steady rollout SPS outside evaluation;
- preserve approximately 3K SPS early and 1.3K-1.6K SPS with the full league
  snapshot; and
- report effective end-to-end SPS including promotion pauses, not only the
  rollout-only counter.

Profile environment stepping, policy forward time, weight loading, and result
serialization separately before changing architecture. CPU parallelism that
starves the training data path or a background evaluator that contends for the
3090 is a failed optimization even if its standalone benchmark is faster.

### Implemented evaluator and league performance

The evaluator now uses deterministic native evaluation controls, stable legal
argmax, up to 12 vector environments, raw per-game telemetry, and one shared
batched forward when both seats use the same checkpoint. On a 16-game forced
schedule, batch size 1 took 37.26s and batch size 12 took 5.35s. A repeat took
5.30s. Batch-1, batch-12, repeat, and the pre-optimization evaluator all agree
exactly on every semantic game field, with score 87.5% and zero timeouts. The
invariance artifact is
`results/promotion_ablation_v1/evaluator-invariance-v1.json`.

The full 144-game self-controlled reference schedule now takes roughly 32-41s
per checkpoint instead of approximately 55-72s. Two full Task 3 reference
replays fell from 66.10s/64.63s to 35.46s/34.10s with exact record equality.

League rollout refresh now reuses checkpoint-resident opponent modules,
preserves per-opponent PFSP and LSTM state by checkpoint identity, and retains
a removed policy only until its in-flight games end. New games sample only the
current retained set, so a pool refresh never swaps an opponent mid-game. The
production anchor stays evaluation-only in a fresh league and does not become
an artificial PPO opponent at epoch zero.

Measured SPS is recorded in `results/promotion_sps_v1/summary.json`:

| State | Tail window | Mean SPS | Median SPS | Result |
| --- | ---: | ---: | ---: | --- |
| Fresh pool, evaluation-only anchor | 20 epochs | 3,530 | 3,800 | Passes 3K target |
| Mature pool, 13 resident opponents | 20 epochs | 1,565 | 1,608 | Matches observed full-league band |
| Diagnostic: anchor inserted into PPO pool | 20 epochs | 1,849 | 1,908 | Rejected initialization |

The diagnostic result is not evidence that opponent play should be removed;
it shows that an external production anchor must not silently alter a new
run's training distribution. Real training checkpoints enter the PPO pool on
the normal checkpoint schedule.

### Completed 15M shadow

The live shadow completed 977 updates, or 15,006,720 sampled rows, from the
exact Task 3 parent and reward stack. Because frozen-opponent rows are not
optimized as learner actions, the final optimizer counter is 9,660,131. The
machine-readable record is
`results/promotion_shadow_v1/summary.json`.

The run was split at epoch 300 after the first live gate exposed a real
compatibility defect: S13 and the newer checkpoints have different recorded
policy layouts. The checkpoint and matching trainer state were already durable.
The evaluator was fixed to construct every historical policy from that
checkpoint's own recorded layout, the epoch-300 screen was rerun against the
unchanged state, and training resumed from the epoch-300 optimizer, scheduler,
global step, and critic without resetting them.

That recovery also exposed a resume-accounting gap. Older checkpoint metadata
did not persist the aggregate native completed-episode counter. The resumed run
therefore supplied the observed ceiling of 26 explicitly. Resume tracking now
recognizes `environment/completed_episodes` and rounds a native vector mean
upward, preventing curriculum or reward shaping from becoming denser after a
restart. Shaping was 0.7272 at epoch 300, 0.7025 at the first resumed aggregate
with 27 completed episodes, and continued normally to 0.15. Every saved
checkpoint from epoch 400 through 977 has exact serialization parity with zero
missing, unexpected, shape-mismatched, or numerically changed tensors.

| Epoch | Gate | Candidate | Anchor | Delta | Relative 80% LCB | Reference | Outcome |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 300 | Screen | 50.0% | 53.1% | -3.1 pt | -9.4 pt | Not scheduled | Screen passed |
| 600 | Full | 57.0% | 53.9% | +3.1 pt | -0.8 pt | 32.6% vs 64.6%, -31.9 pt | Rejected by all three reference checks |
| 900 | Screen | 67.2% | 53.1% | +14.1 pt | +6.3 pt | Not scheduled | Screen passed |

The epoch-600 candidate was not rejected for an unlucky panel result. It beat
the anchor's pooled panel score, stayed within every relative opponent floor,
scored 56.3% and 57.8% by seat, and had no timeout. It was rejected because its
self-controlled reference score was below 50%, its reference delta was below
-15 points, and its paired reference lower bound was below -18 points. This is
the separation the redesign was intended to make: broad internal strength does
not erase a large independent external regression. The epoch-900 result is a
screen pass, not an admission, because confirmation was deliberately not
scheduled at that cadence. None of the 544 candidate games in the three stored
decisions timed out.

Shadow mode preserved the control invariants. It wrote immutable decisions,
payoff observations, audit history, and reusable anchor caches, but admitted or
evicted no policy, left the three-member bootstrap quality archive unchanged,
kept panel v1 unchanged, did not move the Task 3 production anchor, and left the
compatibility champion null. The ordinary training league independently
ingested ten checkpoints and ended with the configured 6 recent, 1 mid, and 3
old policies. In this document, "no shadow mutation" means no mutation of these
control decisions; telemetry and cache writes are expected.

### Live throughput and pauses

Steady rollout throughput met the requested machine envelope throughout the
shadow. Windows below omit startup, checkpoint-refresh transients, and the
synchronous gate pause immediately after epochs 600 and 900.

| Active PPO opponents | Epoch window | Mean SPS | Median SPS |
| ---: | ---: | ---: | ---: |
| 0 | 80-99 | 3,711 | 3,721 |
| 1 | 180-199 | 1,894 | 1,951 |
| 2 | 280-299 | 1,860 | 1,927 |
| 3 | 310-399 | 1,572 | 1,647 |
| 4 | 402-500 | 1,571 | 1,615 |
| 5 | 502-600 | 1,536 | 1,513 |
| 6 | 602-700 | 1,473 | 1,484 |
| 7 | 702-800 | 1,553 | 1,635 |
| 8 | 802-900 | 1,523 | 1,504 |
| 9 | 902-977 | 1,469 | 1,482 |
| 9, final tail | 958-977 | 1,511 | 1,496 |

These measurements prove the absolute no-regression guard supplied for this
machine: early training remained above 3K and the mature league remained in the
observed 1.3K-1.6K band. They do not isolate a causal 1-2% delta against an
otherwise identical pre-change worktree, because no such matched executable
was retained. The relative 1-2% target is therefore unmeasured rather than
silently claimed as a pass; the full live soak is the stronger practical check
that promotion-v2 did not significantly harm rollout speed.

The epoch-600 full gate paused training for 278.1s: 58.5s screen, 54.7s
confirmation, 144.5s reference evaluation, 5.3s policy loading, and normal
checkpoint/control overhead. This passes the eight-minute full-gate budget.
With the anchor panel and reference caches warm, the epoch-900 screen paused for
74.1s and passes the three-minute screen budget. The 33 SPS and 124 SPS logged
on the first updates after those gates are pause-inclusive interval metrics,
not rollout regressions.

Across both trainer processes, including startup and the scheduled live gates,
effective learner-action throughput was 1,581 SPS. Including the separately
recovered epoch-300 gate as compute time lowers it to 1,548 SPS. The analogous
all-row rates are 2,457 and 2,405 sampled rows/s. Successful gate stages used
456.7s total. The split excludes human debugging/calendar idle time and the
discarded partial failed gate, so the raw logs and per-gate pauses remain the
authoritative operational measurements.

### Shadow decision

Promotion-v2 passes the shadow criteria for archive admission, but the shadow
does not justify automatic production-anchor replacement. Only one scheduled
full confirmation occurred, and its correct rejection shows why archive and
anchor semantics must stay separate. The next run should:

- enable quality-archive admission while keeping anchor replacement manual;
- freeze the Task 3 reward stack, panel v1, thresholds, schedules, PFSP, and
  retention so admission behavior remains attributable;
- preserve the same 300-epoch screen and 600-epoch confirmation cadence;
- halt for review if steady mature-pool SPS falls below 1.3K, a screen exceeds
  three minutes, a full gate exceeds eight minutes, or timeout rate exceeds 1%;
  and
- require review of the raw panel/reference record before changing the
  production anchor, even when a candidate enters the archive.

### Monitored archive-admission continuation

The controlled continuation launched from the matching epoch-977 model and
trainer state as
`promotionv2_archive45_final_resume977_178433179898`. Its launcher is
`run_promotion_archive_continue977.sh`, its JSONL is under
`experiments/runlogs/`, and its copied control state is under
`experiments/league/promotionv2_archive45_final/`. The source model hash is
`6baac2e2451269f95fbf1a6b93c032029742b018a209a24f127074aaff842e17` and
the source trainer-state hash is
`cc461689f4057801a5337f1fc3e790f6edc4436313a883f545cfd2f9d2a86e7c`.

The run adds 1,953 updates, or 29,998,080 sampled rows, to the completed
15,006,720-row shadow for a cumulative aligned horizon of 45,004,800 rows.
Epoch numbering remains continuous from 977 through 2,930. Full gates occur
at epochs 1,200, 1,800, and 2,400; screen-only checks occur at 1,500, 2,100,
and 2,700. Panel v1 is frozen through the endpoint by setting its refresh
interval to 3,000 epochs. Quality admission is enabled, production-anchor
replacement remains manual, and `promotion_archive_affects_training_pool`
is false, so admission cannot change PFSP or retention membership.

Two continuation controls were added before launch:

- synchronous evaluation restores Python, NumPy, Torch CPU, and Torch CUDA RNG
  state before PPO resumes; and
- archive protection is independent from ordinary league retention unless an
  explicit opponent-distribution ablation enables it.

The epoch-977 scheduler had completed its original 977-epoch cosine and stored
zero LR. Loading it unchanged under a 2,930-epoch trainer caused PyTorch's
cosine schedule to rebound after its old endpoint. Restarting at the original
0.003 LR was also rejected: the earlier combo45 model-only continuation in
`research-notes-01.md` collapsed entropy and external strength under that
exact mature-policy shock. This run therefore preserves optimizer moments but
uses a new cosine from 0.0003 to zero over only the 1,953 remaining updates.
Entropy stays at its existing 0.002 tail, and reward shaping stays at 0.15.
The diagnostic rebound run was stopped before a promotion gate and its output
was discarded.

The first stable pre-checkpoint window, epochs 979-999, averaged 1,644 SPS
with a 1,675 median over 21 updates. Epoch 1,000 saved successfully and entered
the ordinary league as `p000011`; its first pause-inclusive post-refresh sample
is not treated as steady throughput. With 11 active opponents, epochs
1,002-1,020 averaged 1,372 SPS with a 1,445 median over 19 updates, remaining
inside the full-league operating band.

`monitor_promotion_archive_sps.sh` runs outside the trainer and polls only the
JSONL once per minute. It interrupts the trainer after two distinct 20-update
windows below a 1,300 median SPS; checkpoint and gate transients therefore do
not trigger on a single low sample. The first guarded window at epoch 1,035
had mean/median SPS of 1,562/1,573 with 11 active opponents.

The epoch-1,200 full gate rejected `p000013`. This was not an internal-panel
failure: the candidate scored 69.5% versus the anchor's 53.9%, a +15.6-point
paired delta with an +11.7-point 80% lower bound. All four opponent deltas
were positive, its two seat scores were 68.8% and 70.3%, and none of the 128
panel games timed out. The external reference check instead scored 36.1%
versus the anchor's 64.6%, a -28.5-point delta with a -31.3-point lower bound.
It therefore failed the reference floor, non-inferiority, and confidence
checks. All 416 stored candidate games completed normally. Screen,
confirmation, reference, and policy-load stages used 60.4s, 57.5s, 82.5s,
and 5.4s respectively, below the full-gate budget. The archive remained at
its three bootstrap members, panel v1 and the production anchor stayed fixed,
and ordinary retention independently reached the intended 13-policy 6/4/3
pool.

The epoch-1,500 screen passed `p000016`, but correctly made no archive change
because confirmation was not scheduled at that checkpoint. The candidate
scored 68.8% versus the anchor's 53.1%, a +15.6-point paired delta with a
+10.9-point 80% lower bound. Its seat scores were 59.4% and 78.1%; it beat the
production anchor, S13, and Task 2 panel rows while trailing S14 by 12.5
points. All 64 balanced-seat games ended in `gameover` with no timeouts. The
screen used 59.9s plus 3.1s of policy loading, below the three-minute budget.
RNG isolation and evaluation-only archive semantics remained enabled, and the
three-member bootstrap archive, panel v1, and production anchor were unchanged.

The original external guard stopped the trainer at epoch 1,518 after observing
overlapping rolling windows with 1,268 and 1,298 median SPS. Review showed that
both windows covered the epoch-1,500 screen pause: the first post-screen row was
122 SPS because its interval included evaluation, while the last five rows had
already recovered to 1,440--1,474 SPS. The 17 steady rows after that pause had a
1,304 median, so this was not a sustained below-budget regime. The uncheckpointed
epochs 1,501--1,518 were discarded and the run resumed from the atomic p1500
model/trainer-state pair as `promotionv2_archive45_final_resume1500_178433625308`.
That pair pins model hash
`d33a9d5f1ed9de9fad5284ce4ae1d9f51e825aa1c55edbf19196d2c6fac2ff0c` and
trainer-state hash
`ccf1ee134683bf8e449e3c828b4ecbbf87c7b01973b1bee03989a1a288bcf7c5`.
Optimizer and scheduler restoration resumed at LR 0.000249974 with cosine
`T_max=1953` and `last_epoch=523`; the schedule was not restarted. The guard
now requires its two low 20-update windows to be non-overlapping. The first
complete recovered window, epochs 1,505--1,524, averaged 1,551 SPS with a 1,485
median, confirming that the stop was a gate-transient false positive rather
than a steady learner regression.

The epoch-1,800 full gate rejected `p000019` on the same external axis, despite
another clear internal-panel pass. The candidate scored 76.6% versus the
anchor's 53.9%, a +22.7-point paired delta with a +19.4-point 80% lower bound.
All four opponent deltas (+15.6 to +31.3 points) and both seat deltas (+20.3
and +25.0 points) were positive. Reference performance improved materially
from 36.1% at p1200 to 46.5%, but remained below the 50% absolute floor; its
-18.1-point anchor delta and -21.2-point lower bound also missed the
non-inferiority and confidence limits. All 416 games ended normally with
balanced seats and no timeouts. Screen, confirmation, reference, and loading
used 55.4s, 53.2s, 77.1s, and 5.3s, remaining below the full-gate budget.

The reference gain was broad rather than a single favorable draw: eight of
nine fixed-deck rows improved from p1200, candidate-seat scores rose from
31.9%/40.3% to 44.4%/48.6%, and all eight gate slices improved. The
remaining relative deficit is strategically concentrated. Versus the cached
Task 3 anchor, p1800 was substantially weaker on starter Raizan, both Earth
references, and the two evaluated Lightning constructed decks; it was near
parity on one Fire reference and stronger on both Water references. This is
evidence of continued but specialized learning, not a global regression or a
promotion-luck artifact. The three-member archive, panel v1, and production
anchor therefore correctly remained unchanged.

The epoch-2,100 screen passed `p000022` with the strongest screen result so
far, but made no archive change because confirmation was not scheduled. The
candidate scored 81.3% versus the anchor's 53.1%, a +28.1-point paired delta
with a +23.4-point 80% lower bound. Both candidate seats scored 81.3%, and
all four opponent deltas were positive (+6.3 to +43.8 points). All 64 games
ended in `gameover` with no timeout or truncation. The screen and policy load
used 56.6s and 3.1s, respectively. RNG isolation and evaluation-only archive
semantics remained active; the bootstrap archive, panel v1, and production
anchor remained unchanged.

The epoch-2,400 full gate admitted `p000025` through the standard route, the
first live quality-archive admission. The candidate scored 75.0% versus the
anchor's 53.9%, a +21.1-point paired delta with a +17.2-point 80% lower bound.
All four opponent deltas (+6.3 to +31.3 points), both seat deltas (+17.2 and
+25.0 points), and every absolute panel floor passed. Reference score reached
51.0%, clearing the 50% candidate floor; its -13.5-point anchor delta cleared
the -15-point non-inferiority limit, and its -16.3-point paired lower bound
cleared the -18-point confidence limit. All 416 games ended normally with
balanced seats and no timeout or truncation. Screen, confirmation, reference,
and policy-load stages used 54.1s, 54.5s, 70.3s, and 5.3s, below the full-gate
budget.

Reference progression across the three full gates was 36.1% -> 46.5% ->
51.0%. From p1800 to p2400, six of nine fixed-deck rows improved, two held,
and one declined; both candidate-seat scores rose again to 47.9% and 54.2%.
Compared with p1200, all nine fixed-deck rows improved. This repeated paired
schedule is evidence of broad continued learning despite persistent matchup
heterogeneity. Admission added `p000025` as the fourth active quality member
but did not replace the Task 3 production anchor, refresh panel v1, or set the
compatibility champion. Ordinary PPO retention remained exactly six recent,
four mid, and three old policies; `p000025` occupied a normal recent slot, and
archive admission supplied no additional training-pool protection.

The epoch-2,700 screen passed `p000028` but exhibited the expected late-cycle
oscillation. The candidate scored 67.2% versus the anchor's 53.1%, a
+14.1-point paired delta with a +7.8-point 80% lower bound. All four opponent
deltas and both seat deltas remained positive, but the pooled result was lower
than the p2100 and p2400 panel results. All 64 balanced-seat games ended in
`gameover` with no timeout or truncation; the screen and policy load used
52.4s and 3.1s. Because confirmation was not scheduled, `p000025` remained the
only live admission and the archive, anchor, panel, and PPO-pool semantics were
otherwise unchanged.

### Continuation outcome

The continuation completed at epoch 2,930 with all 1,953 expected epochs
present exactly once from 978 through 2,930. It added 29,998,080 sampled rows
to the 15,006,720-row shadow parent, for 45,004,800 cumulative sampled rows,
and added 17,991,418 actual environment actions. The final checkpoint is
`experiments/azuki_local_promotionv2_archive45_final_resume1500_178433625308/model_azuki_local_002930.pt`
with model hash
`7196734b2250ea1901e0e74e7bd693ebe6cf0918a5e51e5d231a3424b5d045d4`
and trainer-state hash
`0aa5da87f27801ed008c984ef64373874eec843e17558b1af2710d7e9b2bc5e1`.
Checkpoint parity had zero missing, unexpected, shape-mismatched, or differing
parameters. The scheduler ended exactly at `last_epoch=T_max=1953` and LR
zero. The final row retained 0.356 entropy, 0.0366 approximate KL, 0.00985 clip
fraction, and 0.821 explained variance; it had 100% game-over completion and
no timeout.

Performance stayed inside the declared full-league band. Across all 1,953
authoritative rows, mean/median SPS was 1,410/1,432. With the six known
startup/gate pause rows removed, full-pool mean/median was 1,412/1,430. The
final 100 updates averaged 1,354 SPS with a 1,406 median. Matched 100-update
windows immediately before and after the p2400 admission had medians of 1,406
and 1,427, so evaluation-only archive admission did not reduce PPO throughput.
The corrected guard observed transient low windows but never two
non-overlapping failures; its final rolling median was 1,342 SPS. A tmux
prefix-matching cleanup bug that kept the completed guard alive was fixed by
using exact session targets; it did not touch the trainer or metrics.

All six scheduled live records exist at epochs 1,200, 1,500, 1,800, 2,100,
2,400, and 2,700. Together they contain 1,440 games, all ending in `gameover`
with zero timeout or truncation. The slowest full gate used 205.9s of measured
evaluation/loading stages, below the eight-minute budget, and the slowest
screen used 63.0s, below the three-minute budget. The final archive has four
active quality members: the three bootstrap members plus admitted `p000025`.
The Task 3 production anchor, panel v1, and null compatibility champion are
unchanged. PPO retention ends at exactly 6/4/3 with 13 active policies;
`p000025` has already left that ordinary active pool despite remaining in the
quality archive, directly confirming that admission does not protect or alter
training membership.

The endpoint p2930 model was not on a scheduled full-confirmation epoch, so it
was initially left unqualified rather than selected by recency. A subsequent
frozen-state full qualification admitted p2930 through the standard route. It
scored 71.9% on the panel with a 68.8% paired lower bound, 65.6% to 78.1%
across opponents, and 71.9% from both seats. Its 50.35% fixed-reference score,
-14.24-point anchor delta, and -17.36-point paired lower bound cleared all
three reference checks, with no timeout. The immutable result is
`results/promotion_p2930_qualification_v1/qualification_metrics.json`.
p2930 is therefore the qualified common parent for the reference-deck ladder;
the production anchor remains unchanged.

Final CPU-only validation passed 74 Python tests and 19 subtests. Both
continuation launchers and the SPS guard pass `bash -n` and `shellcheck`, and
the worktree passes `git diff --check`. No trainer, guard, or GPU compute
process remains.

## Acceptance criteria

The redesign can control quality-archive admission only when all of these are
true:

- historical sentinel decisions and their explanations are acceptable;
- decisions remain materially stable on the second paired-seed schedule;
- the rule improves agreement with broad external/strategic evidence over P0;
- raw records make every rejection attributable to opponent, seat, gate,
  timeout, or confidence rather than an opaque aggregate;
- deterministic and vectorization tests pass;
- shadow mode produces no archive, anchor, compatibility-champion, or training
  membership mutation; audit history, payoff telemetry, and cache writes are
  expected;
- the SPS and gate wall-time budgets pass on both early and full-pool states;
  and
- Task 3 remains the exact parent stack for any live validation run.

These criteria are now met for the monitored archive-admission stage. The
legacy `promotion_accepted` signal remains contextual matchup telemetry only;
it must not reject an otherwise confirmed model improvement or select a
production checkpoint by itself. Production-anchor changes remain manual: this
run validates archive admission, but one admitted checkpoint does not validate
an automatic anchor-replacement rule.
