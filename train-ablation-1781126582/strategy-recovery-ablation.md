# Promotion-v2 Strategy Recovery Ablation

Status: complete sequence (Steps 1-5 complete, Step 5 rejected, Step 6 skipped,
Step 7 broad zero-tail rejected, late 20.5% zero-tail revisit viable-neutral,
Step 5b delayed estimator rejected on throughput with retained-window
diagnostics complete; 2026-07-21)

## Objective

Retain promotion-v2 p2930's real gains in deck consistency, portal conversion,
Fire pressure, Stonehaven top end, and external strength while testing whether
Water spell play and Garden-only On Play realization can recover. The high IKZ
curve is not itself a defect; it is treated as a legitimate learned meta unless
matched counterfactuals show that a lower curve is stronger.

Promotion is not a primary outcome for this sequence. Current promotion events
are too sparse and luck-sensitive to distinguish many meaningful improvements.
Use paired fixed-seed evaluation, confidence bounds, mechanics, and deck-policy
diagnostics first; report promotion only as secondary telemetry.

## Required order

Do not skip forward based only on suggestive self-play rates.

1. Make episode-driven reward annealing monotonic across run resume.
2. Measure opportunity-normalized mechanics along the p2930 lineage.
3. Run frozen-battle-policy deck counterfactuals.
4. Run a matched 15M continuation ladder from p2930:
   control, early-tempo deduplication, and portal Gate-Power tail reduction.
5. Only if a supplied-deck counterfactual identifies a stronger draft choice,
   test phase-separated terminal-only draft credit. Water and leader choices
   are evaluated separately.
6. Only if Garden effects are available but missed, test realized effect-credit
   redistribution.
7. Validate the surviving recipe for 45-80 minutes with a mature shaping floor
   against a zero-tail schedule.

## Step 1: resume invariant

Implemented in `python/src/train.py` without modifying the native rollout hot
loop.

- New checkpoints persist the ceiling of the latest native completed-episode
  aggregate in both trainer state and checkpoint metadata.
- Resume selects the maximum exact count if sidecars disagree.
- A lower `AZK_RESUME_COMPLETED_EPISODES` override is ignored unless
  `AZK_RESUME_ALLOW_SCHEDULE_REWIND=1` is explicitly set.
- A legacy checkpoint with no exact counter starts at the completion boundary
  of every enabled monotonic episode schedule. This can conservatively skip a
  remaining legacy ramp, but cannot replay full-strength shaping.
- Fresh runs ignore inherited resume progression.

For p2930, the restored count is `242`. Under its 12-episode warmup and
40-episode reward ramp, shaping therefore resumes at the configured final
scale rather than at `1.0`.

## Step 2: opportunity-normalized lineage diagnostic

Raw action counts confound policy choice with shorter games, card availability,
legal windows, and different deck composition. Reanalyze the matched S14 and
p2930 games, plus available p2930 lineage checkpoints, using denominators that
answer whether a mechanic was actually declined.

Required readouts:

- Spell play: legal spell-play selections / decision windows with at least one
  legal spell, split by proactive versus response window and by Water gate.
- Direct Garden play: direct-Garden selections / entity-play windows where both
  Garden and Alley destinations were legal for the selected or comparable
  entity.
- Garden-only On Play: effects realized / times a drafted copy was drawn and
  legally playable to the Garden; also report drafted, drawn, and played counts.
- Gate riders and leader abilities: rider confirmations / legal rider offers,
  and ability selections / legal ability windows.
- Portal reward exposure: early-tempo bonus, flat portal Gate-Power bonus, and
  any overlapping action bonus per eligible action and per episode.
- Retain outcome, turn, seat, gate, deck curve, and early dead-hand strata.

Decision: if legal opportunity rates remain stable while raw counts fall, the
change is mainly ecology/deck availability. If conditional selection falls,
the battle policy itself has narrowed.

Completed result: Water has both a proactive policy regression and later deck
starvation, while its response policy remains healthy. Direct Garden choice
collapsed conditionally from 25.4% in S14 and 8.0% in the Task 3 parent to
0.2% by promotion p100 and 0.1% at p2930. Rider acceptance is saturated once
offered. Promotion p100 received about 2.14 gross early-tempo plus portal-GP
reward per game at scale 1.0, versus 0.29 at p2930's mature floor. Full details
are in `results/promotionv2_p2930_analysis/opportunity_lineage/report.md`.

## Step 3: frozen deck-policy counterfactual

Freeze p2930's battle policy and change only supplied decks. Compare:

- native p2930 decks;
- curve-balanced two/four-of variants of the same learned cores;
- Water spell-restored variants that preserve the p2930 entity backbone;
- old S14 element lists as a diagnostic bound, not as human imitation targets.

Pair seats, gates, draws, and seeds. Use the same external panel and report
paired win difference with a bootstrap confidence interval, early dead hands,
spell opportunity/use, rider realization, Garden-only effect realization, and
game length.

Decision: better play with supplied balanced or spell-restored decks indicates
draft credit failure. No strength gain means the high curve/entity core is
mostly legitimate and intervention should target only missed mechanics.

Completed result: the curve-balanced p2930-core arm lost 3.5 paired points
overall and 11.1 in Water despite fewer early dead hands. The Water
spell-restored arm cast proactive spells but lost 16.7 Water points; exact S14
Water was neutral. Water-specific draft credit is therefore not justified.

The complete S14 lists gained 6.9 points only because Fire gained 44.4 while
Earth lost 22.2. A Fire leader/main factorial isolated a genuine leader-choice
gap: Kagoro with p2930's unchanged main gained 33.3 points against the old
fixed panel and 31.3 against p2930's own learned-deck panel. The S14 main under
Zero did not reliably help. This supports a later phase-separated leader-credit
test, not a Kagoro identity reward or a low-cost reward. Direct Garden choice
remained below 0.5% in every supplied-deck arm. See
`results/promotionv2_p2930_analysis/frozen_deck_counterfactual/report.md`.

## Step 4: 15-minute shaping ladder

All arms start from the same p2930 model and matching trainer state, use the
same seeds and league snapshot, restore episode progression, restart a true
15M learning-rate schedule with a conservative continuation peak, and retain
the mature shaping floor. The screen should end after LR decay; it must not
evaluate one third of the way through a 45M schedule.

The prior reference control demonstrates why this matters. It used a 0.0003
peak on a 45M-compatible schedule and stopped at p3900 while LR was still about
0.00023. Its median PPO KL was roughly 0.04-0.05 and it scored only 26.0%
head-to-head against p2930. Treat that as an optimizer transient, not a valid
control result. Start the new control at peak LR 0.0001; if its paired strength
gate still fails, stop before candidate arms and repeat at 0.00003.

Arms:

- `control`: current successful recipe unchanged.
- `tempo_dedup`: keep equal early-tempo credit for hand development, but remove
  generic tempo payments from portal, ability activation, and affirmative
  confirmation. Portal GP and realized state changes already provide closer
  signals, and the Fire leader factorial makes outcome-agnostic ability credit
  a concrete concern.
- `portal_gp_tail`: keep early teaching but reduce or eliminate the flat portal
  Gate-Power action bonus at the mature floor. Do not penalize portals or high
  Gate Power.

Primary acceptance requires no meaningful paired strength regression and a
measurable recovery in at least one conditional blind spot without collapsing
the portal backbone. Deck diversity, sibling-gate KL, Water performance, and
SPS are guardrails. Target rollout throughput remains about 3K SPS early and
1.3K-1.6K SPS under the full league snapshot; investigate any repeatable loss
above 5% before continuing.

Implementation is in `run_strategy_recovery_ladder15_v1.sh`. It pins the
qualified p2930 model, trainer state, league state, and promotion-state hashes;
asserts restoration at 242 completed episodes; and uses 59,904,000 configured
timesteps so the restored p2930 trainer has exactly 970 remaining updates and
ends at p3900. The control uses a `0.0001` cosine peak ending near zero. It must
score at least 0.40 against p2930, with an 80% bootstrap upper bound reaching
0.45, before either candidate can train. If it fails, rerun under a new
campaign name with peak LR `0.00003` rather than interpreting candidate data.

Every completed arm receives 192-game paired H2H against p2930, candidate H2H
against control, fixed train/holdout reference panels as secondary evidence,
24 sampled drafts per gate, sibling-gate KL, and 64 legal-opportunity self-play
games. The report records Water proactive and response spell use, direct Garden
choice, Garden-only effect opportunities, Fire Kagoro/Zero selection, portal
retention, deck concentration, actual reward overlap, PPO KL, and SPS. Runtime
guards stop after two sustained sub-1235 SPS windows (5% below the historical
1300 SPS lower bound) or a candidate median SPS below 95% of control. Promotion
is disabled as a decision signal.

The first control process exposed why the absolute guard needs that 5% margin:
it stopped at p3326 on adjacent medians of 1286.8 and 1298.0 despite a 1431 SPS
run-wide median. Its p3300 model, trainer, metadata, and league state were an
atomic matched set, so the control resumed from p3300 without restarting LR or
reward annealing. A process restart resets vector-environment episode state
even when optimizer and schedules are exact; therefore both candidate arms use
the same planned p3300 process boundary. Final metrics merge only p2931-p3300
from part one and p3301-p3900 from part two for every arm.

The tempo arm exposed a separate process-lifecycle issue at that boundary:
interrupting the trainer could leave its multiprocessing forkserver and workers
alive after the parent exited. Two stopped pools were consuming roughly eleven
CPU cores while the resumed process was starting. They were terminated without
touching the active trainer, and the split helper now cleans processes matching
the exact completed segment tag after `wait`. The tempo resume retained the
exact expected p3301 LR (`6.804431700223059e-05`) and reported shaping scale
`0.15` at its first completed post-resume episode; the cleanup affects wall
speed only, not checkpoint or schedule state.

Completed result: all three arms reached p3900 with LR exactly zero, shaping
scale `0.15`, zero timeouts, and median throughput inside the established
league band. Control passed its startup strength gate at 43.8% against p2930.
`tempo_dedup` is the clear strength-only survivor: it scored 61.5% directly
against control, 53.1% against p2930, and improved the paired parent-panel
score by 9.4 points with an 80% interval of +3.6 to +15.1 points. Its median
SPS was 1459, 1.0% above control. It also produced less concentrated decks
(19.75 deterministic unique cards and 78% four-of slots versus 15.0 and 90%)
without weakening the portal backbone or heldout Water score.

The preregistered status is nevertheless `neutral_no_conditional_recovery`:
Water proactive spell use fell from 12.9% to 5.5% of legal main windows,
direct Garden choice rose by only 0.6 points, and the Kagoro share was
unchanged. `portal_gp_tail` recovered Water mechanics toward control and
retained 9.97 portals/game, but scored only 43.2% directly against control,
lost 4.2 heldout-reference points, and failed the Water guard. It is rejected.
Do not infer that an intermediate portal-GP coefficient fixes both objectives
without another matched test. The complete screen is in
`results/strategy_recovery_ladder15_v1/ladder_report.md`.

## Conditional interventions

### Step 5: terminal-only draft credit

Water did not pass this condition. Fire leader selection did: p2930's frozen
policy is substantially stronger with Kagoro on its unchanged learned main,
and stochastic self-play already associates Kagoro with better Fire outcomes,
yet the draft policy selects Zero slightly more often and its greedy choice is
Zero.

Run this only after Step 4, because removing nearby action-reward bias may shift
leader selection on its own. If the gap remains, test a leader-decision-only
Monte Carlo terminal advantage or a separate draft value head. Keep battle PPO
on its existing GAE stream. Do not reward Kagoro, spell count, or low cost by
name. This is a credit-assignment correction: at gamma 0.99 and lambda 0.95,
terminal GAE influence at an early leader pick is attenuated by approximately
`0.9405^N` across the remaining draft and battle decisions.

Do not reuse the current win-probability auxiliary head as that baseline without
repairing and validating its labels. Although the policy config enables it,
the native packed league logs report a zero labeled fraction throughout. The
packed environment supplies aggregate info plus per-row terminal reward arrays,
while the current league labeler expects nested per-seat info dictionaries.
Consequently the present head is untrained, not a learned potential to freeze.
If Step 5 remains necessary, first derive win labels from terminal rows, add a
nonzero-label regression test and telemetry guard, and then apply terminal
credit only to leader-choice rows. Do not change that path during the matched
Step 4 ladder.

Step 4 did not resolve the leader distribution. Both control and tempo-dedup
sampled Kagoro on 75% of Fire seats in the 64-game opportunity set, but their
larger per-gate deck dumps remained identical at 10 Kagoro versus 14 Zero
samples for each Fire gate and both greedy decks still chose Zero. Refresh the
leader/main counterfactual with the p3900 tempo policy and its own learned main
before implementing this step. Proceed only if Kagoro remains stronger on both
the fixed external panel and the learned-deck panel.

Completed condition refresh: the p3900 tempo policy still has a large causal
leader gap. Replacing only Zero with Kagoro on the unchanged p3900 main gained
27.8 paired points on the fixed panel (80% interval +16.7 to +38.9) and 50.0
points against p3900's own learned-deck panel (+37.5 to +62.5). Both Fire gates
improved in both panels. Replacing only the main while retaining Zero lost 8.3
points on the fixed panel, so this is not merely an old low-curve deck effect.
Step 5 therefore proceeds. See
`results/strategy_recovery_ladder15_v1/tempo_dedup/fire_leader_counterfactual/report.md`.

Implementation keeps the battle learner and league opponent distribution
unchanged. Native per-row terminal rewards now supply binary win labels only
where the true terminal mask is set; truncations are explicitly unlabeled and
the old nested-info path remains available to non-native callers. For an
eligible Fire leader decision made by the current policy, the trainer retains
the packed observation, sampled action and behavior log probability, and the
pre-decision recurrent state until that exact episode terminates. Frozen-policy
actions are never retained. A clipped contextual-bandit actor loss uses the
terminal outcome minus the detached win-probability baseline. The baseline BCE
is diagnostic in this path; both matched arms train it through the same regular
rollout auxiliary loss, so the candidate differs only by actor credit. PFSP,
battle PPO/GAE, native rewards, and promotion state are untouched.

The variable number of completed leader records initially caused Torch to
compile a new graph for different auxiliary batch sizes. A two-smoke check
caught 184/577 SPS outliers. The auxiliary now uses the eager encoder/decoder
methods already retained by the policy while normal rollout and PPO paths stay
compiled, and batches records every four epochs with an endpoint flush. In the
post-fix smoke, 6,756 rollout rows received labels, 126 Fire leader records
trained, median importance ratios remained near 1.0 with zero clipping, and
median SPS from epoch 5 onward was 1,667. Warm auxiliary passes took about
0.28-0.36 seconds. See `results/leader_credit_smoke_v2/report.md`.

The matched efficacy ladder is `strategy_recovery_leader15_v1`. Control and
candidate both resume the exact p3900 tempo model, trainer state, episode count
323, league state, tempo-deduplicated reward stack at scale 0.15, and a new
p3900-p4870 cosine schedule with peak LR `0.00003`. The candidate alone uses
coefficient 0.05 on the two Fire gates, PPO clip 0.2, and a four-update batch
interval. Promotion is excluded. Advance requires nonzero labels and credit
records, at least 95% of control SPS, direct and parent-panel strength gates,
non-losing Fire H2H, a sampled or greedy Kagoro distribution shift, and Water
and portal retention. The registered runner and report are
`run_leader_credit_ladder15_v1.sh` and `leader_credit_ladder_report.py`.

Completed result: reject terminal-only Fire leader credit. Both arms reached
p4870 with zero final LR, no timeouts, and valid terminal labels. The candidate
trained 10,194 eligible leader records and retained 96.9% of control SPS, but
did not change the learned Fire leader distribution: both arms sampled Kagoro
44/96 times, selected Zero greedily for both Fire gates, and had effectively
zero sibling-gate draft KL. Candidate strength against the p3900 parent fell
3.6 paired points (80% interval -7.8 to +0.5), its holdout score fell 2.1
points, and its holdout Water score fell 1.4 points. Its 51.0% direct score
against control and 58.3% Fire subset do not rescue the missing causal
distribution shift or the parent-panel regression. The candidate also produced
more concentrated heldout decks. See
`results/strategy_recovery_leader15_v1/ladder_report.md`.

The Step 7 parent is therefore the p4870 control, which is a clean continuation
of the accepted tempo-dedup recipe with leader credit disabled. Its model,
trainer state, and league state are treated as one atomic parent snapshot.

### Step 5b: whole-draft long-horizon terminal credit [fallback rejected]

Status: the required native impulse preflight selected the delayed sampled
fallback after Step 7 retained the p4870 control as the last accepted atomic
parent. Implementation and the matched performance/integrity smoke completed,
but the 15M ladder stopped on its preregistered throughput guard at p5503. The
retained p5200/p5500 diagnostics found no causal draft-context improvement.
Do not build the next experiment on the rejected leader-credit, zero-tail, or
delayed-credit candidates.

The leader-only result does not reject long-horizon draft credit. It updated one
decision while leaving the 50 following main-deck picks on the existing weak
trace, so leader choice and deck composition could not co-adapt. At `gamma=0.99`
and `lambda=0.95`, direct terminal influence retains about 4.7% after 50
decisions, 0.22% after 100, and 0.01% after 150. The next question is therefore
whether the true outcome can update the leader and the whole autoregressive
draft without that exponential attenuation.

This is an objective-consistent credit-assignment experiment, not another
native reward. Do not reward a leader, card, curve, spell count, or deck style
by name. Do not force a random leader: the gate is already assigned randomly,
and removing the leader choice shortens the trajectory by only one action while
preventing the leader and subsequent main picks from learning jointly.

#### Required preflight: actual credit-path impulse test

Before choosing an implementation, test the real native rollout and BPTT
layout with a synthetic episode whose only nonzero component is one true
terminal reward. Record actor advantage at the leader row and main picks 1, 10,
25, 40, and 50, including rows on opposite sides of every segment/update
boundary. Run the diagnostic with the current `0.99/0.95` trace and proposed
terminal-only settings. Also verify that truncations produce no win label.

This preflight is a hard branch condition. If leader and early-pick rows remain
in an advantage graph that reaches the terminal row, use the phase-aware trace
arm below. If those rows have already been trained or evicted before the game
terminates, changing `gamma` or `lambda` cannot repair the horizon; use the
delayed sampled estimator below instead. Do not run a 15M efficacy ladder until
the impulse test proves nonzero early-draft credit through the chosen path.

Completed result: select the delayed sampled whole-draft estimator. A
deterministic native p4870 game reached a true terminal at trainer row 164. The
production geometry has 960 agents, a 15,360-row batch, and horizon 16, which
means exactly one segment per agent per PPO update. Both the current
`0.99/0.95` kernel and proposed terminal-only `1.0/0.99` kernel assigned exactly
zero segmented advantage to both leaders and main picks 1, 10, 25, 40, and 50:
those rows occupied updates 0-6 and were trained and recycled before the
terminal arrived in update 10. In a hypothetical retained trajectory the
proposed advantages were nonzero (about 0.19 at the leaders and 0.53 at pick
50), confirming that segmentation and eviction, rather than the trace
parameters, are the blocker. A second native episode was forcibly truncated
after its 102 draft decisions and produced two truncation flags, no terminal
flags, and an empty win-label map. The machine-readable evidence and readable
report are in `results/draft_terminal_credit_preflight_v1/preflight.json` and
`preflight.md`; the reproducible runner is
`draft_terminal_credit_preflight.py`.

Completed implementation/smoke result: the delayed estimator retains the
leader plus one deterministic main pick per quartile for each live-policy seat,
labels only exact true terminals, drops truncations and incomplete samples, and
uses a fixed 512-row padded/masked replay shape every four updates. Packed
observations remain in host memory while the much smaller action, behavior-log
probability, and recurrent-state tensors remain on the policy device, avoiding
a per-pick GPU-to-host synchronization. Partial replay batches are deferred
until a full batch is available and only the endpoint tail is padded. League
loss telemetry now publishes the current epoch before logging, so the final
flush is observable rather than appearing one row late.

The preregistered 80-update v8 smoke used p4891-p4950 as a 60-update
post-compilation throughput interval. Gradient calibration selected coefficient
`3.59347374043`; the independent production arm's initial delayed-credit
gradient was `1.0113x` the ordinary draft gradient. It trained all `17,445`
labeled records in `17,920` fixed rows, with zero truncations, zero incomplete
episodes, importance means in `[0.99979, 1.00022]`, and zero clipping. Candidate
throughput was `1,290.85` SPS versus `1,353.48` control, a passing `95.373%`
ratio above the absolute `1,235` floor. The earlier v1-v7 campaigns are retained
as failed plumbing/performance evidence and were not used as efficacy results.
The final implementation passes `111` Python tests plus `19` subtests. Evidence
is in `results/draft_terminal_credit_smoke_v8/`.

Completed 15M ladder result: **stop throughput**. The control completed p5847
at `1455.91` steady median SPS and passed the permissive 192-game parent screen
at `46.875%` with zero timeouts. The candidate stopped at p5503, or 64.8% of
the intended continuation, when two non-overlapping 20-update medians were
`1234.56` and `1226.72` against the `1235` floor. Over the same p4871-p5503
rows, candidate SPS was `1310.27` versus `1463.18` control, only `89.55%` of
control against the required 95%.

The stop was not a signal-integrity failure. The candidate captured `150,572`
records, labeled `148,250`, and trained `147,456` examples in the same number
of fixed rows, with zero truncation or incomplete records, zero clipping,
importance means in `[0.999458, 1.000672]`, and median draft gradient norm
`0.1713`. Native shaping remained at the intended resume-safe `0.15` floor.

The retained windows do not justify optimizing around the throughput failure:

- At p5200, candidate-control direct score was 52.60%. Candidate had a paired
  `+5.73` point parent delta and `+3.47` point heldout delta, but its sampled
  decks were slightly more concentrated and 0.51 IKZ more expensive. Control's
  matched main deck had a `+3.75` point sibling-swap advantage while the
  candidate's was exactly zero.
- At p5500, direct score was exactly 50.00%. The paired parent delta was
  `+2.60` points with an 80% interval spanning zero. The heldout delta was
  `-1.39` points with an 80% interval of `-3.82` to `+1.04`; this roughly
  one-point movement is treated as neutral rather than a meaningful regression.
  Candidate decks were less collapsed than control (`18.62` versus `15.50`
  unique cards and 81.2% versus 89.8% four-of-slot share), but control retained
  a `+1.25` point matched-main advantage and candidate again had zero.
- Sibling-gate and leader-conditioned fixed-history KL remained on the order
  of `1e-6` or less in both arms. The candidate changed global draft
  preferences and selected Shao for Water at p5500, but did not learn a strong
  portal- or leader-conditioned main-pick response.
- At p5500, candidate spell selection when any spell was legal was 11.2%
  versus 16.7% control; Water main-spell use per legal window was 10.5% versus
  19.9%. Candidate portals per game were 8.75 versus 9.78. Garden's comparable
  share was numerically higher, but neither arm recorded a Garden play or
  effect offer in that panel, so it is not efficacy evidence.

The machine-readable report, readable summary, per-gate decks, opportunity
traces, and 1,280 fixed-deck hybrid games are in
`results/draft_terminal_credit_ladder15_v1/partial_ladder_report.json`,
`partial_ladder_report.md`, and `partial_trajectory/`. The p4870 atomic model,
trainer state, and league state remain the last accepted parent.

#### Preferred arm: phase-aware terminal trace

The split terminal and shaped value streams currently reuse the same global
`gamma` and `gae_lambda`. Separate their trace settings without changing the
environment reward:

- shaped reward and the battle actor remain on the accepted `gamma=0.99`,
  `lambda=0.95` PPO/GAE path;
- compute a second true-terminal trace with `gamma_terminal=1.0` and
  `lambda_terminal=0.99` for draft actor rows;
- use the long terminal advantage for the leader and all 50 main-pick rows,
  normalized within draft rows, while battle rows keep the existing combined
  advantage;
- replace the existing draft actor advantage rather than summing a second loss,
  so recent draft rows do not receive terminal outcome twice;
- train the terminal baseline from true outcomes, keep it conditioned on gate,
  chosen leader, partial deck, seat, and recurrent history, and exclude every
  truncated episode;
- never multiply this true terminal channel by the shaped-reward anneal.

`lambda_terminal=1.0` is reserved for a later exact Monte Carlo arm if the
`0.99` smoke is stable but still too attenuated. At 150 decisions, `0.99`
retains about 22% of the terminal trace and is a lower-variance first test.

#### Fallback arm: delayed sampled whole-draft estimator

If rollout segmentation cuts the trace, extend the validated terminal-record
mechanism across the draft without storing and replaying every pick. Retain the
leader plus one uniformly sampled main-pick row from each draft quartile for
each live-policy seat. Store the packed observation, sampled action, behavior
log probability, and pre-decision recurrent state. At the exact true terminal,
attach the binary outcome and optimize a PPO-clipped loss using
`outcome - stop_gradient(win_probability_baseline)`.

Use a fixed padded/masked auxiliary batch shape and batch completed records at a
fixed interval so Torch does not compile one graph per completion count. Train
the win-probability baseline identically in both arms through the regular
rollout path; the delayed pass must not add candidate-only baseline gradients.
The stratified five-row sample is an estimator of whole-draft credit and keeps
cost bounded. Do not begin with all 51 draft rows: the leader-only ladder's
3.1% SPS difference and variable-shape compile failure show that performance
must be measured before scaling replay volume.

The phase-aware trace and delayed sampled estimator are alternative candidate
arms. Do not compose them in the first ladder.

#### Matched 15M ladder

Resume the most recent accepted model, trainer state, episode progression, and
league state selected after Step 7. Hash and copy that atomic parent before
launch. Use two arms with the same seed, restart schedule, winning reward stack,
shaping schedule, PFSP state, and promotion disabled:

- `control`: accepted actor-advantage treatment;
- `draft_terminal_long`: exactly one selected long-credit mechanism.

Run a short plumbing/performance smoke first, then approximately 15M learner
rows with checkpoints near 0%, 33%, 67%, and 100% of the continuation. Do not
anneal true terminal credit to zero. If an auxiliary coefficient is required by
the delayed fallback, preregister it after the smoke by matching its initial
draft-gradient scale to control; do not tune it from efficacy results.

Required live telemetry:

- terminal advantage mean, absolute mean, standard deviation, and sign/outcome
  agreement for leader, early, middle, and late main-pick rows;
- labeled, truncated, pending, completed, and trained draft records;
- terminal-baseline BCE/Brier score and prediction mean by draft position;
- draft-only gradient norm, importance ratio, clip fraction, and entropy;
- median and p10 SPS plus auxiliary wall/GPU time;
- the existing reward-scale, timeout, PFSP, and resume-integrity fields.

Evaluation must use windows rather than only the final checkpoint. Retain the
existing paired candidate-control H2H, parent panel, heldout-reference panel,
legal-opportunity traces, deck dumps, and playstyle analysis. Add:

- sibling-gate main-pick KL while holding leader and replayed action history
  fixed, excluding the leader row from the aggregate;
- leader-conditioned main-pick KL while holding gate fixed;
- sampled final-deck Jaccard, unique cards, copy concentration, cost/type mix,
  and per-gate/per-leader composition;
- gate/leader/main hybrid counterfactuals, because higher KL is useful only if
  the matching composition has a causal outcome advantage over a swapped main.

Advance requires all integrity guards, at least 95% of control SPS, no material
paired parent or heldout regression, and either a strength improvement or a
sustained causal drafting improvement. A KL-only change without hybrid
context-fit value is not a pass. Leader distribution is diagnostic rather than
a required target: the mechanism must remain free to retain Zero, select
Kagoro, or vary by Fire gate according to terminal performance.

If long terminal credit reaches early picks but produces unusably high variance,
the next separate experiment is a prefix outcome model. Train it on frozen
accepted-policy league trajectories to predict true terminal outcome from
draft prefixes, validate calibration by gate/leader/seat and heldout episode,
then freeze it for the policy ablation and use prediction differences to
redistribute credit across picks. Do not jointly chase a moving policy and do
not treat an unvalidated predictor as reward. Refresh it only between matched
runs after a new accepted trajectory set is archived.

### Step 6: realized Garden credit redistribution

Use only if opportunity logging shows the relevant cards are drawn and direct
Garden play is legal but systematically declined. Credit the realized useful
effect, not the availability of an On Play clause and not direct Garden play by
itself. Fund it by reducing overlapping generic action bonuses so total shaping
magnitude does not rise.

Completed decision: skip this intervention. The tempo checkpoint supplied 45
exact missed-placement states across 29 source games. Ten common-random suffix
rollouts per arm produced 900 valid rollouts and zero reconstruction errors.
Garden scored 53.0% versus 54.4% for Alley, a -1.4 point paired delta with a
source-game-clustered 80% interval of -6.0 to +3.0 points. Bladebound Ally was
neutral; Sanzu's Envoy was suggestively positive but had only ten states;
Enzo and Koyama Farm Plowman favored Alley in their small samples. Every
tracked useful effect already overlaps leader/board potential shaping or the
adopted temporary-attack realization reward. A generic Garden payment would
therefore duplicate existing credit and push against the terminal
counterfactual. See
`results/strategy_recovery_ladder15_v1/tempo_dedup/garden_counterfactual/report.json`.

## Step 7: long validation

Promote the surviving tempo-dedup recipe to a matched 45,004,800-row comparison
from the Step 5 p4870 control:

- mature nonzero shaping floor;
- shaping held at `0.15`, then annealed to exactly zero early enough to leave a
  substantial pure terminal-only tail.

At 15,360 learner rows per update, the continuation adds 2,930 updates and ends
at p7800. Use the same absolute trainer-update schedule across process resumes:

- p4871-p6000: hold the native `0.15` floor with trainer multiplier `1.0`;
- p6001-p6900: linearly multiply shaped reward from just below `1.0` to
  exactly `0.0` (`0.5` at p6450);
- p6900-p7800: keep the trainer-side multiplier at exactly zero.

This leaves 900 subsequent updates after the zero boundary, or 30.7% of the
continuation, as a pure terminal-only tail. Merely reaching zero at p7800 is
invalid. Evaluate and archive p4870, p6000, p6900, p7400, and p7800 for both
the fixed-floor control and zero-tail treatment. Both arms retain the native
episode-based shaping floor at `0.15`; only the treatment activates this second,
trainer-side multiplier. Fire leader credit remains disabled in both arms.

The trainer-side multiplier uses the restored absolute update, scales only
shaped reward and any enabled draft auxiliary credit, and preserves terminal
win/loss, truncation handling, win labels, and PFSP outcomes. Its schedule and
live multiplier are written to checkpoint metadata. Checkpoints also
fingerprint and restore the complete reward coefficient stack, including the
adopted tempo deduplication, damage mitigation, temporary-charge, and
temporary-attack settings, so process resume cannot silently replay full
shaping or drop a successful reward component. A caller can intentionally
change that stack only through the explicit resume override.

Before the full comparison, run a short matched two-process smoke from p4870
that crosses hold, ramp, and exact-zero boundaries and resumes from an
intermediate checkpoint without re-supplying the schedule variables. Require
checkpoint multiplier telemetry to match the absolute update, terminal labels
to remain nonzero after the multiplier reaches zero, and treatment throughput
to retain at least 95% of the fixed-floor control.

Completed smoke result: pass. Across 40 updates per arm with a forced p4890
restart, fixed-floor median SPS was 1,359.8 and zero-tail median SPS was 1,387.9
(102.1% of control). The treatment checkpoint multipliers were exactly 1.0 at
p4875, 0.5 at p4880, and 0.0 at p4885, p4890, and p4910. The resumed process
received no schedule or reward variables from its command, restored both groups
from p4890 metadata, and remained at 0.0 on p4891. Terminal labels remained
active after zero, reaching 2,106 rows in an update. See
`results/shaped_reward_tail_smoke_v1/report.json`.

The full run uses matched process boundaries at p6000 and p6900 for both arms.
Only the first segment restarts the p4870 cosine schedule; later segments restore
the optimizer and scheduler exactly. This both limits recovery cost and tests
schedule persistence at the start of decay and the start of the terminal-only
tail without changing the registered row budget.

Zero-tail is a hypothesis, not the default. It should win on paired terminal
strength and retain learned mechanics after shaping disappears. Reject it if
mechanics decay, variance rises materially, or the apparent gain exists only in
self-play ecology. Archive checkpoints throughout so strategy trajectories can
be compared rather than judging endpoints alone.

The registered trajectory screen uses 192 paired same-epoch H2H games, paired
p4870-parent panels, 288-game heldout-reference panels, 64 legal-opportunity
self-play games, deterministic and sampled deck dumps, and sibling-gate KL at
p6000, p6900, p7400, and p7800. p6000 is a pre-decay integrity baseline rather
than evidence for zero-tail.

Adopt zero-tail only when all of the following hold:

- p7800 scores at least 52% directly against fixed-floor with an 80% lower
  bound of at least 48%, and the mean direct score across p6900/p7400/p7800 is
  at least 51%;
- its paired p4870-parent and heldout-reference p7800 lower bounds are no worse
  than -3 points, and no late same-epoch direct score is below 47%;
- Water proactive and response selection per legal window lose no more than 3
  and 5 points respectively, heldout Water loses no more than 5 points, and
  portals per game retain at least 90% of fixed-floor;
- deterministic unique-card count loses no more than two cards and four-of slot
  share rises no more than five points;
- schedule, terminal-label, and 95% throughput guards pass.

If all safety gates pass without the clear strength gate, retain fixed-floor and
record zero-tail as neutral. Direct Garden selection is diagnostic rather than
an adoption gate because its exact causal counterfactual was neutral-negative.
The registered evaluator and report are `run_shaped_reward_tail_eval_v1.sh` and
`shaped_reward_tail_report.py`.

Completed result: reject zero-tail on strength and mechanics. Both arms reached
p7800 with valid schedules, no timeouts, active terminal labels, and zero leader
credit. The zero-tail arm retained 98.6% of fixed-floor median SPS, reached an
exact multiplier of zero at p6900, and stayed at zero after the forced p6900
resume, so the implementation and resume invariant passed. Efficacy did not:
zero-tail scored 43.8% directly against fixed-floor at p7800 (80% lower bound
41.1%), with a 43.9% mean across p6900/p7400/p7800. Its paired endpoint delta
was -8.9 points on the p4870-parent panel and -4.5 points on heldout references;
heldout Water fell 9.7 points. Endpoint Water spell slots fell from 17.4% to
7.2%, direct Garden placement fell from 3.1% to 0.3% of comparable choices,
attack rate fell from 26.0% to 22.0%, and mean game length rose from 81.2 to
96.6 decisions. See `results/shaped_reward_tail45_v1/trajectory_report.md` and
`results/shaped_reward_tail45_v1/deck_compositions.md`.

The fixed-floor p7800 control is archived but is not promoted to the next
experimental parent. Its 48.4% direct score against p4870 did not establish an
upgrade, and its deterministic decks were slightly more concentrated. p4870
therefore remains the last model selected by an advancement decision and the
Step 5b atomic parent. This does not reject later or partial annealing; it
rejects this broad 30.7%-of-run terminal-only tail as the next default recipe.

### Annealing interpretation and next design

Treat roughly one-point heldout movements as neutral unless their uncertainty
and independent panels consistently point the same way. This convention does
not change the zero-tail result: its endpoint paired parent delta was `-8.9`
points, heldout delta was `-4.5` points, heldout Water fell `9.7` points, and
the direct candidate-control score was `43.8%`. Those are materially larger
and coherent signals against the schedule that was tested.

Reaching zero shaping can still be a valid long-run objective, but eventual
optimal discovery is not guaranteed by PPO training time alone. The native
impulse preflight proved that draft rows are trained and evicted before the
terminal outcome arrives, so removing shaping does not create the missing
credit path. Sparse terminal feedback also changes exploration and state
visitation; a policy can become less competent before it discovers a better
terminal-only strategy. The zero-tail result is therefore evidence that the
model was not ready on that timetable, not proof that a mature policy must
always retain shaped reward.

If zero annealing is revisited, use a later two-stage matched fork rather than
repeating the p6000-p6900 ramp and 900-update zero plateau:

1. Resume the accepted p4870 atomic parent and train one shared 0.15-floor
   trunk through about 80% of a 45-80 minute continuation. Qualify the fork
   checkpoint against p4870 before spending on branches.
2. Fork the exact model, optimizer, episode count, and league state. Keep the
   control at 0.15; linearly reduce the candidate from 0.15 to zero only over
   the final 20%, with matched checkpoints at the fork, midpoint, and endpoint.
3. Treat an approximately one-point heldout delta as neutral. Require the
   direct, paired-parent, Water-opportunity, portal, attack, and game-length
   panels to remain jointly nonregressive; promotion remains secondary.
4. Only if the zero endpoint passes, extend both arms for a short matched
   5-10% hold to test whether zero is stable. Do not commit to a long zero tail
   before that check, and do not combine this arm with rejected Step 5b credit.

This design directly tests the remaining hypothesis: whether a more mature
policy can shed the final shaping floor late, without conflating it with the
already rejected early removal schedule.

### Late 20.5% zero-tail revisit: viable-neutral

`shaped_reward_latezero20_retry1_v1` executed that narrower design. One shared
p4870-p6900 trunk passed qualification at `48.44%` against p4870, then forked
byte-identical model, optimizer, league, and promotion state. Treatment ramped
from the `0.15` floor at p6900 to exact zero at p7200 and stayed at zero through
p7800: 601 updates, or `20.51%` of the full continuation. The p7200 restart
restored zero correctly, terminal labels remained active, rejected terminal
credit paths remained off, and treatment retained `99.60%` of control SPS.

The direct treatment-control trajectory was `45.3%, 52.1%, 50.5%, 50.5%,
50.5%, 49.5%, 50.0%` from p7200 through p7800. Its final-three mean was
`50.0%`, with Theil-Sen slope `-0.13` points per 100 updates. Endpoint paired
parent and heldout deltas were favorable at `+2.6` and `+3.1` points, but the
parent interval crossed zero and direct H2H ended at parity. This is evidence
of retention and recovery after shaping removal, not a demonstrated ongoing
improvement trend.

Mechanics remained jointly safe: portal rate rose `5.5%`, attacks rose `0.65`
points, games shortened `2.0%`, Water proactive/response selection improved,
and unique-card/four-of metrics were unchanged. The greedy zero-tail Water
deck omitted spells, but sampled Water spell share fell only `1.1` points and
conditional spell use improved, so retain it as a diagnostic rather than call
it a collapse.

Decision: classify the schedule as **zero viable but neutral**, keep p4870 as
the selected atomic parent, and archive p7800 late-zero only as a candidate for
a separate short matched zero-stability extension. Do not make zero shaping
the default and do not combine the extension with Step 5b. See
`late-zero-ablation.md` and
`results/shaped_reward_latezero20_retry1_v1/trajectory_report.md`.
