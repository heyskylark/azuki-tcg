# Next Ablation Plan: Uniform Assignment and Main-Draft Credit

Status: complete through the 45M Stage 3 confirmation; exact retained credit is
a promising experimental base with unresolved mechanics safety, fixed random
prefixing is rejected, and `tailzero60` is the only conditional prefix follow-up

## Execution progress

| Stage | Status | Evidence |
| --- | --- | --- |
| 0: assignment compatibility | Complete | `results/next_ablation_v1/stage0/` |
| 1: uniform assignment | Complete; adopted | `uniform_assignment_confirm45_v1`; every registered gate passes and uniform p7800 is the Stage 2 parent |
| 2: full-episode draft credit | 45M complete; promising, not default | Exact credit reached all 50 picks at `0.9362x` standard SPS and produced large direct, parent, and heldout gains, but broad sampled decks and a persistent Garden/leader ability-rate loss require diagnosis |
| 3: random main prefix | 45M complete; reject fixed recipe | Fixed prefix retained `0.9989x` exact-credit SPS, but its late heldout delta was only `+0.93` points, strength regressed after p9700, Water spells and portals declined, and useful rare-card lines remained zero |
| 4: late shaping removal | Deferred; no accepted terminal-credit parent | Exact-credit p10730 is an experimental checkpoint, not an accepted production parent; do not combine shaping removal with credit or prefix changes |
| 5: promotion redesign | Complete; deployed in shadow mode and qualified | Two deterministic native replays passed context, timing, no-mutation, and semantic-replay gates; uniform p7800 remains the selected final parent |

## Scope

This document defines the experiments to complete before planning a production-scale
training run. The 500M/1B campaign, distributed-training design, hardware funding,
and production sample budget are explicitly deferred until these ablations select a
stable recipe.

The central environment decision is now:

- the environment uniformly assigns the portal/gate card;
- the environment uniformly assigns one of that element's two sibling leaders;
- the policy drafts only the 50-card main deck;
- neither the gate nor the leader is an actor decision during training.

In this document, "gate" means the portal/gate card. "Sibling gates" are the two
gates of one element, while "sibling leaders" are the two leaders of that element.

## Accepted parent and recipe discipline

Start every first-stage matched experiment from the exact accepted `p4870` atomic
snapshot selected in `strategy-recovery-ablation.md`: model, optimizer/trainer
state, completed-episode progression, league state, and promotion state. This is
the selected continuation of the confirmed Task 3/promotion-v2 lineage.

Do not reconstruct the parent by manually combining flags. Hash and archive the
complete parent before launching a branch. In particular:

- retain the accepted temporary-effect realization, early-tempo deduplication,
  damage-mitigation, portal-GP, PFSP, cross-gate replay, and mature `0.15` shaping
  floor exactly as stored by the parent;
- do not enable the rejected leader-credit, delayed sampled draft-credit,
  reference-seat training, entity-damage, generated-IKZ conversion,
  response-availability, privileged-critic, or broad zero-tail candidates;
- keep promotion diagnostic rather than a primary efficacy label until the final
  promotion stage in this plan;
- change one causal factor per matched ladder. A candidate that passes becomes the
  atomic parent for the next stage.

## Permanent assignment contract

For every new episode, assign context in this order:

1. Sample each live-policy seat's gate uniformly from the eight unique gates by
   using the deterministic engine RNG.
2. Apply fixed-reference-seat overrides, when an evaluation explicitly requests
   one. A fixed reference deck keeps its specified gate and leader.
3. Apply the accepted same-element matchup oversampling only when seat 1 is a
   live-policy seat. With probability `0.35`, replace seat 1's gate with seat 0's
   sibling gate. If seat 0 is fixed, use its prescribed gate to derive the sibling;
   if seat 1 is fixed, skip the override. Because the sibling map is a permutation
   over uniformly sampled live gates, this retains a uniform marginal gate
   distribution for ordinary live-versus-live training.
4. Uniformly sample each live-policy seat's leader from the two leaders matching
   its final assigned gate element. Sample the two seats independently.
5. Populate `gate_card_def_id` and `leader_card_def_id` before the first policy
   observation and begin in main-pick mode at pick 1 of 50.

The leader action is removed rather than overridden and trained. There must be no
policy loss, entropy term, behavior log probability, or PPO row for an assigned
leader. All later main-pick and battle decisions are trained normally and remain
conditioned on the assigned gate and leader.

Existing cross-gate replay remains valid: it may swap a completed deck's gate to
the same-element sibling at battle start while keeping the already assigned leader,
which remains legal for that element. Its existing boundary pick masking remains
unchanged.

The current configured deck pool happens to contain two decks for each of the
eight gates, so the existing population-weighted gate sampler is effectively
uniform today. Sampling the unique gate list directly makes uniformity an explicit
invariant that cannot change when the reference pool changes.

### Deployment boundary

Uniform leader assignment is the training contract. If competitive deployment
rules permit deliberate leader selection, handle that as a separate, frozen
meta-selection problem after these ablations. A heldout gate-to-leader lookup can
retain uniform training coverage without randomly choosing a known weaker leader
at deployment. Do not put the long-horizon leader choice back into the 50-card
draft policy.

## What uniform assignment should and should not change

Uniform assignment gives each gate-leader context expected probability
`1/8 * 1/2 = 1/16`. It should:

- prevent leader-selection collapse;
- train battle and main-draft behavior under every legal leader;
- remove learned leader choice as a confound in deck analysis;
- improve statistical support for gate- and leader-conditioned probes;
- make paired gate/leader evaluation and future promotion evaluation cleaner.

It does not repair long-horizon main-draft credit. Removing one leader action per
seat shortens a typical episode by only two trainer rows. The native preflight
previously reached terminal near row 164, while early draft rows were trained and
recycled in updates 0-6 and terminal arrived in update 10. A roughly row-162
terminal still cannot update those rows through the current segmented GAE path.

Uniform coverage can therefore reveal and measure conditional behavior, but a
separate full-draft credit experiment is still required to teach all 50 picks from
the final outcome.

## KL and causal-deck measurements

The old aggregate sibling-gate KL is insufficient after this contract change.
Every checkpoint must report two interventional quantities over the 50 main picks:

1. `gate_KL_given_leader`: same assigned leader, same seed, same replayed main-pick
   history, and only the sibling gate changed.
2. `leader_KL_given_gate`: same assigned gate, same seed, same replayed main-pick
   history, and only the same-element leader changed.

Report both directions, symmetric KL, total variation, pick quartiles, and exact
determinism controls. Aggregate only after reporting all 16 gate-leader contexts.
Marginalizing across leaders can hide opposite changes and is not the primary
metric.

The probe lifecycle must expect exactly 50 main rows, not a leader plus 50 rows.
Because removing the leader row also removes one recurrent transition, old and new
KL magnitudes are not directly comparable. Establish a fresh no-training baseline
and a post-migration baseline before interpreting a trend.

KL is diagnostic, not an advancement target. Prior experiments demonstrated both
stronger models with lower KL and weaker models with higher KL. Any sustained KL
increase must be paired with causal hybrids:

- draft under gate A, then play the same deck under sibling gate B;
- draft under leader A, then play the same deck under leader B;
- compare matched-context and swapped-context outcomes with paired seats and seeds;
- require the context-matched composition to outperform the swapped composition.

Deck reports must dump one greedy and at least 24 stochastic drafts for every one
of the 16 gate-leader contexts. Retain unique-card count, copy concentration,
cost/type mix, card inclusion, Jaccard distance, and opportunity-normalized battle
mechanics.

## Shared experiment protocol

Unless a stage below explicitly overrides it:

- use matched seeds, seats, restart points, LR schedules, league snapshots, reward
  stack, and evaluation opponents;
- run a short integrity/performance smoke before a 15M efficacy ladder;
- evaluate checkpoints near the beginning, one-third, two-thirds, and endpoint;
- make decisions from checkpoint windows rather than the endpoint alone;
- treat an isolated heldout movement near one percentage point as neutral;
- require zero meaningful timeout, truncation, terminal-label, or resume-integrity
  regressions;
- use a 1,200-step cap for heldout reference panels; the earlier 600-step cap
  censored a normally completed Stage 1 game at step 604;
- require candidate median SPS to remain at least 95% of matched control;
- investigate sustained rollout SPS below the historical absolute floor of `1235`;
- retain the historical expectations of roughly 3K SPS early and 1.3K-1.6K SPS
  under a full league snapshot;
- do not use a promotion event or non-event to accept or reject a candidate.

The primary balanced external panel is `16 gate-leader contexts * 24 games = 384`
games per evaluated checkpoint, with paired seats and seeds. Continue reporting the
existing parent panel and heldout-reference panel, but do not allow an old
reference-deck population weighting to replace the uniform-context macro result.

## Required order

| Stage | Experiment | Main question |
| --- | --- | --- |
| 0 | No-learning compatibility panel | Can the accepted policy operate under all assigned contexts, and does removing the leader row disturb its recurrent state? |
| 1 | Uniform-assignment migration | Can training safely adopt the permanent 50-pick contract? |
| 2 | Full-episode main-draft credit | Can true outcomes update all 50 picks without violating the SPS guard? |
| 3 | Random main-card prefix | Does broader card exposure improve drafting and battle rather than merely inject noise? |
| 4 | Late shaping removal | Once terminal draft credit works, can the final training segment remain strong at zero shaping? |
| 5 | Promotion redesign | Can advancement be measured with paired external, gate, leader, and seat evidence? |

Do not compose stages 1-4 in a single first ladder. Complete the registered
decision for one stage, archive its atomic winner, and then branch the next stage
from that winner.

## Stage 0: no-learning assignment compatibility

### Purpose

Separate three effects before spending training samples:

- the current policy's learned leader distribution;
- the performance of both leaders when forced through the existing leader row;
- the recurrent-state effect of starting directly at main pick 1 with a prefilled
  leader.

### Panels

Evaluate the accepted p4870 policy without updating weights under:

1. `policy_leader`: current lifecycle and policy-selected leader;
2. `forced_leader_row`: force each legal leader through the existing leader action
   row, while excluding the forced action from any policy interpretation;
3. `prefilled_no_row`: assign the same leaders before reset output and start at the
   first main pick.

Use identical seeds and cover all 16 gate-leader contexts. Compare draft action
distributions, recurrent-state norms, deck composition, battle strength, seat
effects, and SPS.

### Branch condition

If `forced_leader_row` is healthy but `prefilled_no_row` materially regresses, the
problem is recurrent initialization rather than leader assignment. Test one
non-trainable context burn-in forward that exposes the assigned gate and leader
before main pick 1. It must not create an environment action, reward, PPO row, or
leader loss. Use the simpler direct start if it is already safe.

Stage 0 produces the new 50-row KL baseline and identifies weak leader contexts. It
does not select a new model.

## Stage 1: uniform-assignment migration

### Arms

Resume two matched branches from the exact p4870 atomic parent:

- `control_learned_leader`: current policy-selected leader and 51 learned draft
  decisions per seat;
- `uniform_assigned_leader`: uniform unique gate, uniform compatible leader, and
  exactly 50 learned main picks per seat.

No 25%-forced leader arm is needed. No always-forced action row is trained. The
candidate implements the intended permanent contract directly.

### 15M decision

The candidate must pass all integrity and SPS requirements and show:

- no material uniform-context macro, seat, or element regression across windows;
- usable battle behavior under both leaders of every element;
- no new action, card-type, copy-count, or deck-concentration collapse;
- stable portal, spell, direct-Garden, response, and leader-ability opportunity
  rates;
- a clean new `gate_KL_given_leader` and `leader_KL_given_gate` baseline;
- no reliance on one favorable gate-leader stratum to offset broad losses.

Because this is a chosen environment contract rather than a reward hypothesis, it
does not need to beat the old leader picker to be adopted. A material regression
must nevertheless be diagnosed before building later ablations on it. If the 15M
trajectory is ambiguous or still adapting, extend the matched comparison to 45M.
Otherwise archive the strongest safe candidate window as the new atomic parent.

All subsequent controls use uniform assignment. Old league snapshots may continue
as opponents, but the environment supplies their assigned gate and leader before
their first main pick as it does for the live policy.

## Stage 2: full-episode main-draft credit

### Motivation

The existing segmented rollout path assigns exactly zero same-episode terminal
advantage to early draft rows. The prior delayed sampled estimator proved that
terminal labels and replay plumbing can work, but its additive auxiliary pass fell
to 89.55% of control SPS and did not produce causal context-fit improvement. Do not
rerun or merely enlarge that implementation.

### Preferred design

For each live-policy seat, retain all 50 main-pick records until an exact true
terminal:

- packed observation;
- chosen action;
- behavior log probability;
- pre-decision recurrent state;
- assigned gate, leader, seat, and draft position;
- exact terminal outcome.

Drop truncations and incomplete episodes. Keep packed records in host memory and
never retain autograd graphs across the episode.

Replace the ordinary immediate draft actor loss with a delayed, fixed-shape,
batched Monte Carlo/PPO loss. Do not add a second full actor pass on top of already
trained draft rows. Use a separate terminal-value baseline conditioned on gate,
leader, partial deck, seat, and recurrent history:

`draft_advantage_k = terminal_outcome - stop_gradient(V_terminal(prefix_k))`

Use `gamma_terminal = 1.0` with no lambda attenuation for this draft-only channel.
Battle PPO/GAE and the accepted shaped-reward path remain unchanged. Never multiply
true terminal draft credit by the shaped-reward schedule.

Batch completed drafts in large fixed shapes, report record age and PPO importance
ratios, and amortize learner work. The design target is to replace approximately
the same number of draft rows that the normal learner would have consumed, not to
double draft computation.

### Required sequence

1. Repeat the terminal impulse test under the new 50-pick lifecycle.
2. Run an integrity smoke proving labels, truncation exclusion, all-position
   gradients, fixed shapes, and resume behavior.
3. Measure matched smoke SPS and report both the 95% relative and 1,235 absolute
   diagnostics without using either as a training stop.
4. Only then run a matched 15M efficacy ladder.
5. Require both external safety and either a strength gain or a sustained causal
   gate/leader/deck-fit improvement before a 45M confirmation.

Telemetry must split leader context and draft quartiles and include advantage
mean/variance/sign agreement, terminal-baseline BCE and Brier score, record age,
importance ratios, clipping, draft gradient norm, GPU auxiliary time, and SPS.

### Fallback: frozen prefix outcome model

If objective-consistent retained-row training cannot pass label integrity or is
too noisy to produce useful efficacy, pivot to a frozen outcome predictor rather
than extending the rejected additive estimator. Throughput alone does not cancel
the retained-row efficacy read.

Train a small prefix model on complete self-play episodes from the accepted policy,
historical league snapshots, and deliberately broadened gate/leader/card-prefix
contexts. Human play logs are not required; the target is the exact terminal
outcome. Hold out whole seeds, opponent lineages, and policy generations. Validate
calibration and discrimination separately by gate, leader, seat, and draft
quartile.

Freeze the predictor during each policy ablation. Use return redistribution or
successive prediction differences only after defining the residual correction so
the redistributed draft return does not double-count or replace the battle
terminal objective incorrectly. Refresh the predictor only between accepted
policy generations.

## Stage 3: random main-card prefix

### Purpose

Uniform leader assignment broadens leader contexts but does not directly force
rare main-card exposure. Test ByteRL-style random main prefixes only after a useful
draft credit path exists, so exposed cards can affect both battle learning and
subsequent card-selection learning.

### Arms

From the accepted uniform-assignment and draft-credit parent:

- `control_no_main_prefix`: policy chooses every main card;
- `random_main_prefix`: independently sample
  `n in {0, 1, 2, 4}` with probabilities `{0.5, 0.25, 0.125, 0.125}` for each
  live-policy seat, then uniformly force the first `n` legal main-card picks.

This forces one card per 50-card deck on average. Forced rows receive no actor loss
and are not presented as policy-sampled actions. Every later main pick and battle
decision trains normally. Disable random prefixes for evaluation.

### Evaluation

In addition to the shared panels, report:

- card exposure, draw, legal-play, selected-play, and realized-effect funnels;
- coverage of cards absent or rare in control drafts;
- whether later picks adapt coherently to the forced prefix;
- exact-deck diversity, unique cards, copy concentration, curve, and type mix;
- gate- and leader-conditioned KL after holding the forced prefix fixed;
- fixed-deck and hybrid outcomes showing whether new compositions are useful;
- battle strength on normally policy-drafted decks with prefixing disabled.

Do not advance a candidate merely because it produces more diverse or more
human-looking decks. Require no material external regression and either stronger
outcomes or a repeatable causal deck/battle improvement. If the prefix arm is
neutral on strength and fails to improve causal deck quality, remove it from the
recipe.

If exact full-draft credit fails and the frozen predictor fallback needs broader
training data, random-prefix trajectories may be generated for the offline
predictor dataset before this policy-efficacy stage. Data collection is not itself
evidence that prefixing should be enabled in policy training.

## Stage 4: late shaping removal

Keep the accepted `0.15` shaping floor throughout Stages 0-3. Do not combine a new
assignment, credit path, or prefix mechanism with a shaping-schedule change in its
first matched run.

Only after true terminal credit reaches the 50 main picks should the late-zero
hypothesis be revisited. Fork an exact mature checkpoint into:

- `fixed_floor`: remain at `0.15`;
- `late_zero`: ramp from `0.15` to zero before the final fifth of training, then
  remain at exact zero for the entire final 20%.

The true terminal draft-credit channel remains fully enabled in both arms and is
never annealed. Compare direct H2H, paired parent, uniform 16-context, heldout,
Water mechanics, portal use, attacks, game length, deck composition, and causal
context-fit metrics across multiple points in the exact-zero interval.

Treat another approximately one-point heldout movement as neutral. Adopt zero only
if strength and mechanics are jointly nonregressive and the exact-zero segment is
stable. A neutral result keeps the `0.15` floor; it does not justify another longer
zero tail automatically.

## Stage 5: promotion redesign

Promotion remains deferred until the training interventions above are complete.
Use `promotion-ablation.md` as the detailed starting specification, updated for the
uniform-assignment contract.

The redesigned evaluator should:

- use a panel of recent and historical opponents rather than one incumbent;
- pair seats, seeds, gates, and assigned leaders;
- evaluate all 16 gate-leader contexts or a preregistered balanced subset;
- include an external fixed yardstick and windowed external strength;
- use quorum, matchup-floor, and uncertainty rules rather than one aggregate
  Wilson threshold;
- keep reference decks measurement-only unless a future independent experiment
  reopens reference-seat training;
- record promotion as league management, not the sole label of model improvement;
- run a shadow/replay qualification before it is allowed to mutate the live league;
- measure evaluator wall time separately from rollout SPS and avoid serial gate
  pauses that dominate training time.

Uniform assignment should make promotion less luck-sensitive because every
candidate and opponent can receive identical gate-leader-seat blocks. It does not
remove draw luck, meta cycles, or the need for multiple opponents and windows.

### Stage 5 outcome

The uniform-context implementation is merged in shadow mode and qualified against
the selected uniform p7800 checkpoint. Two independent native replays produced
identical semantic game records. Each replay completed 128 exact-context screen
games, 64 cross-gate confirmation games, and 288 reference games, covering all 16
screen contexts, both candidate seats per context, and both reference seats per
context across two seeds. Every scheduled and realized leader was legal for its
gate, all games completed, and the three-observation external-strength window was
recorded.

The full gate took `372.24s` and `365.13s`, below the registered `480s` budget;
screen time was `80.97s` and `78.92s`, below the `180s` budget. The decision rule
would have admitted this checkpoint, but shadow mode correctly left the quality
archive, production anchor, panel, compatibility pointers, next policy index, and
active PPO pool unchanged. This result qualifies the evaluator and its bookkeeping;
it is not training-efficacy evidence and does not make promotion the primary model
quality signal.

Evidence is sealed in
`results/next_ablation_v1/stage5/uniform_promotion_qualification_v1/summary.json`
and `report.md`.

## 2026-07-23 continuation override

The retained-row hypothesis is reopened by explicit user direction. The two smoke
results are not reclassified: retained rows measured `0.8379x` and `0.8651x` of
their matched controls. For the new efficacy continuation, however, model quality
takes priority over that relative cost. The `0.95` relative and `1,235` absolute
thresholds remain reported but cannot stop the run. Label, gradient,
recurrent-state, completion, timeout, atomic-checkpoint, and model-process
integrity remain hard requirements.

The first continuation cannot be used: every retained label in both its original
and resumed candidate legs was `0.5` because aggregate native `info` supplied a
zero terminal-component array. The candidate lineage through resumed p8400 is
marked invalid. The corrected implementation uses native rewards only on true
terminal rows and records decisive wins and losses separately. Both classes are
required by the new smoke and ladder integrity gates.

The corrected decisive-label smoke and fresh matched 970-update retained-row
ladder completed from uniform p7800, followed by the random main-card-prefix
smoke and matched 970-update ladder with exact retained-row credit in both arms.
This completed the registered test of whether exact terminal credit and broader
early card exposure are complementary.

### Continuation outcome

Corrected full-episode credit is now a valid experiment mechanism. It labeled
46,866 decisive ladder episodes with both win and loss classes, trained every
draft quartile, and ran at 1,310.17 SPS versus 1,454.69 control. Its registered
decision is nevertheless `stop_after_neutral_15m`: the endpoint scored 52.86%
direct, late parent and heldout panels were positive, but no causal
portal/leader deck-fit effect appeared. Endpoint greedy unique-card count rose
from 17.5 to 30.5 and stochastic four-copy slot share fell from 60.15% to
37.42%. The p8770 credit checkpoint is not an accepted parent.

The fixed random-prefix arm also completed without an integrity or performance
failure. It ran at 1,321.49 SPS versus 1,362.56 exact-credit control, sampled all
registered prefix lengths, and correctly excluded forced rows from actor
updates. The late direct mean was 52.69%, but parent-panel delta averaged only
+0.43 points and heldout delta averaged -0.12 points. It reduced stochastic
unique cards by 1.37 and raised four-copy share by 6.08 points, but all p8770
sibling contexts inside an element used the same greedy deck and much of the
concentration came from a universal four-copy weapon preference. Useful
rare-card lines remained zero. The fixed distribution therefore does not
advance to 45M.

### 2026-07-25 45M confirmation addendum

The preceding 15M stop was later overridden by explicit user direction so the
behavioral trends could be measured at 45M. Campaign
`credit_prefix_confirm45_v1` trained standard credit, exact credit/no-prefix,
and exact credit/fixed-prefix from the same p7800 parent to p10730 and completed
both evaluation suites.

Exact credit produced a late direct mean of `65.97%` against standard credit,
a `+17.71` point late parent-panel delta, and a `+14.00` point late heldout
delta at `0.9362x` median SPS. It is retained as an experimental base. It is not
the default because stochastic unique cards reached `38.26`, Water spell slots
declined, and the final three windows carried a persistent roughly `5.6` point
Garden/leader ability-rate deficit.

Fixed prefixing produced a `53.47%` late direct mean but only a `+0.93` point
late heldout delta. It peaked at p9700, regressed at p10730, reduced Water spell
slots and portals, and still produced zero useful rare-card lines. The fixed
recipe remains rejected. If prefixing is tested once more, use one resume-safe
arm whose nonzero mass linearly reaches zero at `0.60H`, followed by a fully
unprefixed final 40%. Do not begin a production-horizon prefix schedule test
unless that 45M rescue passes multiple late windows.

Full analysis:

`results/next_ablation_v1/credit_prefix_45m_behavior_analysis.md`

The current selected configuration remains the hash-locked uniform-assignment
p7800 lineage, the accepted reward stack, and the fixed `0.15` shaping floor,
without exact retained-row credit or random prefixes enabled by default. If
prefixing is reopened, compare the single `tailzero60` rescue against the
sealed 45M no-prefix and fixed-prefix arms before considering a longer run.

## Decision tree

1. If the no-row lifecycle alone is harmful, repair recurrent initialization with
   a non-trainable context burn-in and repeat Stage 0.
2. If uniform migration is safe, make it the permanent baseline and retire learned
   leader selection, leader-prefix probabilities, and leader-credit ablations.
3. If full-episode credit passes integrity but is slow, finish the local efficacy
   read first. Treat the measured SPS cost as engineering debt and an adoption
   tradeoff; optimize or replace the hot path after learning value is known.
4. If full-episode credit reaches early picks but remains too noisy, improve the
   terminal baseline or use return redistribution before inventing card-identity
   rewards.
5. If random main prefixes improve exposure without causal deck or outcome value,
   reject them rather than stacking them for diversity alone.
6. If late zero is neutral or negative, retain the `0.15` floor.
7. After promotion redesign is shadow-qualified, close this ablation sequence and
   create a separate production-scale training plan.

## Explicitly deferred

The following are outside this document and must not be started automatically:

- any 500M, 1B, or larger training run;
- distributed actor/learner implementation or hardware procurement;
- a funding or cloud-cost proposal;
- additional named-card, spell-count, curve, leader-identity, or deck-style
  rewards;
- Suphx-style privileged features in the actor;
- another privileged-critic ladder without a new informativeness result;
- reference-deck league seats or imitation from old human decks;
- a separate learned leader-selection policy.

Those decisions should be revisited only after Stages 0-5 identify the final local
recipe, its measured SPS, its checkpoint-window behavior, and the remaining
strategic gaps.
