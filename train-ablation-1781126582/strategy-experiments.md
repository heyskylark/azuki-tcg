# Strategy-learning experiments

## Purpose

The corrected production run is a healthy systems run and a failed strategy-learning run. This document is the experiment ledger for replacing the current training recipe. It separates attribution experiments from the final combined canary so that a later result can be explained rather than merely observed.

Primary objective: learn high-ceiling, element-, gate-, leader-, deck-, and opponent-conditioned play. Early win rate against cheap face-pressure policies is a safety signal, not the primary short-run optimization target.

The target is not fewer face attacks. The target is to stop face attack from becoming the only robustly learned plan. A correct evaluator must distinguish:

- taking a legal lethal or efficient face attack;
- attacking face because no stronger line is available;
- ignoring a legal entity-control, setup, response, portal, or combo line because immediate face damage has an easier proxy reward; and
- building an entity-heavy common shell that makes strategic cards unavailable in the first place.

## Completed baseline and rejection evidence

The unchanged 1B control completed at p65105 with `1,000,012,800` sampled rows, healthy throughput and learning metrics, zero integrity maxima, and a verified final manifest. The training system worked as configured.

The learned policy did not satisfy the product objective:

- The p53000-p65105 fixed-reference ladder averaged `95.657%` with a slope of only `+0.0418 pp / 1,000 updates`. This panel was saturated.
- p60000 was the strongest final-window checkpoint, but it did not displace p44000 and lost to it directly.
- The final strategy traces remained approximately `75-77%` face attacks.
- All deterministic Lightning contexts at p56000/p60000/p65105 used the same 46-entity, four-Lightning-Orb, zero-weapon deck.
- All deterministic Water contexts used 50 entities and zero spells. Echoed Waves had no interactive recovery offers.
- Lightning sibling-gate deck behavior was effectively identical; aggregate gate sensitivity was overwhelmingly Fire-driven.
- p56000 briefly produced six ordered Defender/Stone lines with four wins, but the line was rare, Earth-only, and did not survive strength qualification or persist to the endpoint.

Decision: the existing recipe is the negative control. Do not extend it or use the endpoint as the next parent merely because it is latest.

Primary evidence:

- `results/corrected_production_1b_lr1500_fresh/gates/gate_1b_summary.json`
- `results/corrected_production_1b_lr1500_fresh/gates/gate_1b_preflight.json`
- `results/corrected_production_1b_lr1500_fresh/gates/strategy_sequences_1b.json`
- `local-production-run.md`, section **Final 1B automated validation outcome**

## Working causal model

The evidence supports a sequence, not one proven root cause:

1. Dense early battle shaping makes common face-tempo and entity-flood behavior easy to discover.
2. Exact full-draft terminal credit broadcasts a shared outcome across all 50 picks. It can reinforce a generally successful common shell without identifying which cards created the win.
3. Cross-gate replay currently contaminates that exact credit on swapped seats: the deck was drafted for one gate, battles under its sibling, and the retained draft rows still receive the swapped-gate result.
4. The league repeatedly trains against one frozen identity for eight updates and retains policies chronologically rather than strategically. This can reinforce a local counter-policy and forget broader play.
5. The native shaping scale reaches its `0.15` floor after only 52 completed episodes per environment. The floor persists for almost the entire run, but the initial full-strength transient is extremely short.
6. Saturated strength panels reward cheap competence and fail to protect an initially weaker strategic trajectory with a higher later ceiling.

Causal ranking differs by training phase:

- **Initial policy discovery:** reward formula, early-tempo bonus, battle/draft opportunity distribution.
- **Late convergence and forgetting:** league concentration, chronological retention, constant broadcast draft credit, and validation/admission rules.
- **Persistent bias:** the `0.15` non-terminal reward floor still favors the current proxy, but it is not sufficient by itself to explain late collapse.

## Non-negotiable experiment rules

- Use the textual gate representation. Keep `gate_id_embedding_enabled=false`; the prior text-only arm passed external parity and separated the compositionally meaningful Lightning/Water sibling pairs.
- Do not add rewards for named cards, prescribed deck lists, or exact scripted combos. Those are evaluator and curriculum concepts, not reward targets.
- Keep the flat portal exploration bonus in the first reward experiments. The prior outcome-only portal reward collapsed portal selection from `0.076` to `0.0038`; removing “pay for trying” before the policy is competent starved exploration.
- Change one causal boundary per matched arm. Combine winners only in the fresh canary.
- Schedule trainer-level anneals by configured sampled rows or update progress, never by per-environment episode counters.
- Balance gate, leader, opponent element, seat, and starting player in every evaluation and retained-credit batch.
- Report window means and slopes. Do not select a recipe from one cyclic checkpoint peak.
- Short experiments prioritize strategy acquisition and coverage, subject to integrity and broad-regression floors. Later experiments must convert that support into strength.

# Experiment order

## 0. Repair evaluation instrumentation and establish baselines

This stage fixes measurement correctness and sensitivity. Stage 5 later replaces the production qualification panel and its decision rules.

### 0.1 Reward-component telemetry

The production logs expose total terminal and shaped returns but not cumulative grants by source. Add fixed-cost per-episode counters for both raw and post-scale values of:

- terminal win/loss and truncation terms;
- potential leader-health term;
- potential Garden-attack term;
- potential untapped-Garden term;
- potential untapped-IKZ term;
- direct leader-edge delta;
- direct board/Garden-attack delta;
- no-op penalty;
- flat portal-GP bonus and portal outcome bonus;
- early-tempo bonus;
- damage-mitigation/Defender bonus;
- temporary Charge realization;
- temporary attack realization;
- entity-damage exchange, generated-IKZ conversion, and response reserve, even when configured to zero; and
- total raw shaping, shaping multiplier, total scaled shaping, and terminal return.

For each component log sum, absolute sum, positive/negative grant count, and maximum single-step magnitude. Add discounted contribution using the trainer gamma so that a large nominal late reward is not compared incorrectly with an early reward. Aggregate by gate, leader, turn bucket, selected action type, winner/loser, and episode length outside the hot path.

Required checks:

- component sums reconstruct the exact shaped reward for every tested step;
- raw × active multiplier reconstructs scaled shaping;
- terminal rewards are never annealed;
- zero-configured components remain exactly zero;
- telemetry disabled and enabled produce identical actions/rewards under the same seed; and
- logging overhead is measured before any long arm.

Status (2026-08-27): implemented behind `env.reward_telemetry`. The native
episode record now carries sparse fixed-cost component statistics and
action/turn slices; Python aggregates gate, leader, outcome, episode-length,
action, and turn dimensions outside `c_step`. Native reconstruction,
terminal-scaling, zero-component, and enabled/disabled trajectory checks pass.
The four-trial 64-environment preflight measured `1.03%` mean throughput
overhead (`8,117.8` control versus `8,033.9` telemetry env-steps/s). Evidence:
`results/reward_telemetry_eval0_preflight.json`.

### 0.2 Strategy event and opportunity telemetry

Raw action counts are insufficient. Record `opportunity -> selected -> resolved -> converted` funnels. At minimum:

- face versus entity attack opportunities, including positions where both were legal;
- lethal attacks available and taken;
- weapon offered/drafted/drawn/legal/attached/recovered/re-equipped;
- spell offered/drafted/drawn/legal/played/replayed;
- portal legal/selected/resolved and whether its gate ability produced a usable result;
- leader and Garden ability legal/selected/resolved;
- IKZ available, held across turns, spent, generated, and later converted;
- response affordable, reserved, selected, and resolved;
- Defender available/declared and damage prevented;
- temporary Charge or attack granted and later realized; and
- card-level drafted/drawn/legal/selected/resolved/converted funnels.

Track ordered semantic sequences for each element. The initial list must include the known lines, but the data model must support new sequences without changing reward code:

- Lightning: weapon attach, discard/recovery or re-equip, portal, relevant leader use, and attack conversion.
- Water: spell play, spell replay, Echoed Waves recovery, Healing Flutter timing, and Shao targeting an actual attacker.
- Earth: lethal Devotion sacrifice, Bobu activation before destruction, Defender, Stone portal, and post-portal attack.
- Fire: Rushfire extra play, prerequisite ordering, Zero before attack when relevant, Kagoro after multi-play, and converted temporary Charge/ATK.

### 0.3 Evaluator sensitivity tests

Before using the metrics to select training arms, prove that they distinguish known policies and fixtures:

- p21000, p29300, p44000, p52000, p56000, p60000, and p65105;
- deterministic and stochastic policy modes;
- curated strategic decks for all eight gates;
- an entity-only/face-pressure deck baseline; and
- scripted positive and negative traces for each ordered sequence.

A metric is invalid if it rewards mere card availability, raw action volume, or an illegal/impossible line. Opportunity-normalized metrics must stay stable when the number of irrelevant no-op decisions changes.

Status (2026-08-27): evaluator contract implemented. Trace schema v2 records
draft offers/picks, exact discard contents, leader attack, post-action state,
and explicit sample/argmax mode without changing policy inputs or rewards.
`strategy_descriptor.py` emits `azuki.strategy_descriptor` schema v1 with
opportunity-normalized attack/lethal, card lifecycle, portal, ability,
response, Defender, resource, and temporary-effect funnels; ten versioned
Lightning/Water/Earth/Fire ordered sequences; context deck/collision
descriptors; and context-keyed payoff vectors. Eight scripted evaluator tests
cover positive/negative sequence traces, irrelevant no-op stability, illegal
actions, resolved-versus-unresolved cards, lethal alternatives, recovery and
re-equip, response reserve, portal usefulness, temporary-effect conversion,
deck collisions, and payoff distance. A real p65105 argmax rollout produced a
complete v2 trace (`draft_events`, `legal_actions`, and `post_action_state`
all true), and the repaired evaluator replayed the 200-game legacy p65105
trace while explicitly marking unavailable resolution/draft capabilities.

### 0.4 Baseline packet

Emit one versioned baseline artifact containing:

- payoff matrix against qualified ancestors and unsaturated opponents;
- all strategy funnels and ordered sequence rates;
- deck descriptors and exact sibling-deck collision flags;
- reward-component distributions reconstructed from representative rollouts; and
- runtime/SPS cost of the evaluators.

The artifact schema becomes part of every later arm. Do not change metric definitions mid-ablation without versioning and recomputing every compared baseline.

Status (2026-08-27): EVAL-0 complete. The versioned baseline packet covers all
seven registered checkpoints and all 21 checkpoint pairs. Every pair has
shared payoff cells and strategy metrics; strategy RMS distance ranges from
`0.0295` to `0.1834`. Reward reconstruction error is exactly zero across the
eight-gate component baseline. The deterministic p65105 curated panel completed
all 32 seat-swapped games across all eight gates against element-matched
entity-only/face-pressure controls (mean strategic score `0.4375`). The packet
contains no pending probe and records evaluator runtime, source hashes,
descriptor IDs, sibling collisions, and the frozen sensitivity contract.
Evidence: `results/strategy_baseline_v1/baseline_packet.json`.

## 1. Reward-shaping updates

### 1.1 What is wrong with the current potential formula

The trainer discounts future reward with `gamma = 0.99`. Proper potential-based reward shaping for a transition is:

```text
F(s_t, s_{t+1}) = gamma * Phi(s_{t+1}) - Phi(s_t)
```

Using the trainer gamma means using the same discount factor as PPO/GAE uses when it values future rewards. A reward `k` actions in the future is weighted by `0.99^k`; for example, a reward 100 actions later has about `0.366` of its undiscounted weight.

With the same gamma, the discounted shaping return telescopes:

```text
sum_t gamma^t * F_t = -Phi(s_0) + gamma^T * Phi(s_T)
```

**Force terminal-state potential to zero** means define `Phi(s_T) = 0` on a natural terminal or truncation transition and include the final correction `-Phi(s_{T-1})` in that transition. It does not change terminal health, cards, or observations. It closes the shaping ledger. With terminal potential zero, the total discounted shaping return differs only by `-Phi(s_0)`, which is fixed for the initial state; the optimal policy is therefore preserved in the tabular/theoretical setting.

The production formula is instead:

```text
time_weight_t * (Phi(s_{t+1}) - Phi(s_t))
time_weight_{t+1} = 0.95 * time_weight_t
```

This does not telescope under the trainer objective. An early gain can be rewarded at high weight while its later reversal is charged at a much lower weight. The policy can profit from temporarily improving the proxy even if the advantage disappears before terminal.

The decay is also much faster than it appears:

| Action index | `0.95^t` |
|---:|---:|
| 10 | 0.599 |
| 20 | 0.358 |
| 50 | 0.077 |
| 100 | 0.0059 |

OpenAI Five scaled non-terminal rewards by approximately `0.6^(game_time / 10 minutes)` to counter late-game reward inflation. That is not policy-invariant PBRS, and it decays on game time rather than every decision. Azuki's `0.95` per action is approximately a `0.6` multiplier every ten actions, so it strongly suppresses delayed card-game setup. Dota's successful use does not establish that this much faster decay is safe for draft and combo learning.

### 1.2 Matched reward arms

Use one parent, seeds, league, draft credit, assignments, optimizer, and evaluation schedule. First run a short smoke for reward correctness, then a 15M screen. Confirm only sustained candidates at 45M.

| ID | Potential transition | Direct leader/board deltas | Early tempo | Purpose |
|---|---|---:|---:|---|
| R0 | Current `0.95^t * delta Phi` | On | On | Exact negative control |
| R1 | Current `0.95^t * delta Phi` | On | Off | Isolate cheap early-tempo credit |
| R2 | Proper `0.99 * Phi(next) - Phi(prev)` | On | Off | Isolate potential correction |
| R3 | Current `0.95^t * delta Phi` | Off | Off | Isolate duplicated edge deltas |
| R4 | Proper PBRS with terminal closure | Off | Off | Clean potential treatment |

The leader-health and Garden-attack edges occur both inside `Phi` and as direct deltas in production. R2/R3/R4 determine whether this duplicated credit matters. R4 is the default candidate only if telemetry, strategy support, and safety panels agree.

Status (2026-08-27): configurable native implementation complete.
`env.pbrs_mode` selects the unchanged `legacy` formula or `discounted`
`gamma * Phi(next) - Phi(prev)`; `env.pbrs_gamma` defaults to `0.99`; and
`env.pbrs_terminal_closure` independently enables the R4 terminal correction
while rejecting legacy-mode misuse. Natural game-over and all truncation paths
apply the scaled `-Phi(previous)` correction exactly once while keeping outcome
and truncation rewards unscaled. Episode telemetry records the selected mode,
gamma, closure flag, and initial potential. Deterministic native tests prove
the discounted potential sum equals `-Phi(initial)` within `2e-5` for both a
forced truncation and a natural game-over; reconstruction remains within
`2e-6`, and six native equivalence/control tests remain green.
Evidence: `python/tests/test_reward_telemetry.py`.

Execution status (2026-08-29): R4 passed a 10-update native preflight with
discounted PBRS and terminal closure enabled: 2,842 SPS, zero timeout/auto-tick/
zero-legal truncations, `7.16e-7` maximum action-component reconstruction
error, unscaled terminal credit preserved, and checkpoint parity at updates 5
and 10. R0-R4 are registered as fresh, seed-42, 15,006,720-row matched screens
and run serially on the same GPU. Cross-gate exact-credit exclusion is
deliberately disabled in every reward arm to preserve the D0 parent boundary;
the later D1-D3 stage owns that change.

After the first screen, test whether combat-state shaping is needed at all:

- **R5 resource-only potential:** remove leader-health and Garden-attack terms; retain only readiness/resource terms under proper PBRS.
- **R6 no state potential:** terminal reward plus separately scheduled exploration/realization aids; no `Phi`.

R5/R6 answer whether the constrained attack space already receives enough terminal signal. Do not add entity-attack or leader-attack rewards during this test. Entity targeting should be learned from outcome; otherwise the experiment cannot determine whether combat shaping was unnecessary.

### 1.3 Exploration rewards are separate from state potential

Potential shaping, exploration aids, and terminal outcome serve different purposes and need separate scales:

- `potential_scale`
- `exploration_scale` for flat portal, no-op, Defender, and temporary-effect realization terms
- `draft_credit_scale`
- terminal scale fixed at `1.0`

Initially preserve the flat portal bonus and existing small mechanic-realization terms while changing the potential formula. Use reward telemetry to determine whether any one component dominates terminal credit or merely creates the first successful examples. Later removal must be an explicit arm.

Do not reward “strategic play” directly. If a mechanic is never discovered, change opportunity/exposure through a balanced curriculum. A named combo reward teaches the evaluator's script and can suppress a different valid strategy.

### 1.4 Reward-arm decision rule

At 15M, strategy support is primary:

- no element is structurally absent from deterministic and stochastic deck/action funnels;
- Lightning weapon and Water spell conversion rise above their matched control opportunities;
- at least three elements improve one resolved ordered sequence without another collapsing;
- sibling decks/context actions become distinguishable where fixed-deck interaction probes show gate-dependent value;
- face attack share is interpreted only conditional on legal alternatives; and
- no integrity, timeout, or catastrophic external-strength regression occurs.

A temporary raw-win deficit against R0 does not reject a strategy arm if strategy coverage and later-window slope improve and the arm remains above the registered safety floor. The 45M confirmation must show that the new support is being converted into stronger play rather than accumulating unproductive mechanics.

## 2. Draft credit and strategic exposure

### 2.1 Current exact-credit behavior

Production uses `azk_draft_episode_credit_coef = 1.0`, one retained batch of up to 80 complete drafts every four updates, and all 50 main picks. Each pick receives the same final win/draw/loss target, with a prefix-state win-probability baseline. The coefficient is constant and is not scaled by the native reward-shaping anneal.

This is a trainer-side auxiliary policy update, not an environment reward. It should be scheduled separately. Prior corrected ablation evidence showed that exact credit:

- produced valid gradients at all 50 pick positions;
- cost approximately ten percent SPS;
- increased unique-card count and reduced four-copy concentration;
- did not produce causal gate/leader deck fit; and
- reduced Water spell slots in the measured endpoint.

The problem is not that terminal outcome is a bad label. The problem is high-variance broadcast assignment: every pick in a winning common shell is reinforced, including unused and anti-synergistic cards.

### 2.2 Cross-gate replay decision

Keep cross-gate replay in the first corrected treatment. The prior `0.15` arm was externally free and was the first intervention to make the critic price gate fit as deck-dependent rather than as a gate constant.

Fix its interaction with exact draft credit:

- tag original draft gate, battle gate, and `gate_swapped` per seat;
- continue masking boundary-crossing ordinary actor rows as today;
- exclude the swapped seat's entire retained draft record from actor and win-baseline exact-credit updates; and
- log excluded records by original gate, battle gate, leader, seat, and outcome.

Battle rows under the swapped textual gate remain valid and retain the critic-contrast benefit. Original-gate draft observations labeled with sibling-gate outcomes are not valid draft credit.

Run one matched `cross_gate_replay_prob = 0` control after the masking fix. Retain replay only if it still improves deck-conditional critic/context sensitivity or strategy acquisition without reducing causal deck fit. Turning it off before this test would discard a previously positive contrast-data mechanism because of a fixable credit leak.

### 2.3 Draft-credit arms

All schedules use global sampled-row progress.

| ID | Exact-credit coefficient | Cross-gate retained rows | Purpose |
|---|---|---|---|
| D0 | Constant `1.0` | Current contaminated behavior | Negative control only |
| D1 | Constant `1.0` | Swapped records excluded | Isolate correctness fix |
| D2 | `1.0 -> 0.1`, then hold | Swapped records excluded | Early assistance with a small late floor |
| D3 | `1.0 -> 0.0`, then hold | Swapped records excluded | Diagnostic: can learned drafting persist without late broadcast pressure? |

Use the first half of the configured arm for the linear ramp in the initial screen; preregister the exact row boundaries in each run config. If D2/D3 are sensitive to that boundary, test schedule length separately rather than silently retuning it.

Add these diagnostics before comparing arms:

- draft-credit gradient norm relative to ordinary PPO actor gradient;
- per-position and per-quartile advantage magnitude/sign agreement;
- gate/leader/seat/outcome balance of retained records;
- card drafted versus later drawn/legal/used/converted contribution groups; and
- deck diversity separated into useful strategic diversity versus singleton noise.

Balance or stratify the retained reservoir by gate/leader context so easier Fire outcomes cannot dominate a supposedly uniform batch through episode-completion or reservoir effects.

Status (2026-08-27): correctness and scheduling implementation complete.
Native episode records now expose each seat's `original_gate`, `battle_gate`,
and `gate_swapped` flag while retaining `gate` as the battle gate. Exact-credit
records carry the same metadata. `AZK_DRAFT_EPISODE_CREDIT_EXCLUDE_XGATE`
defaults on; the swapped seat is removed before both actor and win-baseline
updates, including terminal draws and truncations, while battle rows remain
trainable. Exclusions are logged by original gate, battle gate, leader, seat,
and outcome. The bounded deterministic reservoir now balances observed
gate/leader contexts instead of allowing the easiest context to fill all 80
slots.

`AZK_DRAFT_EPISODE_CREDIT_FINAL_COEF`,
`AZK_DRAFT_EPISODE_CREDIT_ANNEAL_START_ROWS`, and
`AZK_DRAFT_EPISODE_CREDIT_ANNEAL_END_ROWS` define a linear schedule keyed only
to completed rollout count times the registered sampled-row batch size; zero is
a valid final coefficient. Runtime metrics expose the active coefficient,
sampled-row boundary, gate/leader/seat and outcome balance, per-position and
per-quartile advantage diagnostics, and the exact-credit/PPO actor gradient-
norm ratio. The D0 negative-control config must
explicitly set `AZK_DRAFT_EPISODE_CREDIT_EXCLUDE_XGATE=0`; D1-D3 retain the
safe default. Unit/native coverage proves row-boundary interpolation,
pre-label exclusion of all 50 picks, deterministic context balancing, and
one-seat original/battle gate tagging under forced cross-gate replay.

Execution status (2026-08-30): D0 reuses the matched R1 reward control; D1-D3
are fresh seed-42 runs of the same selected reward recipe. D2/D3 ramp over
rows `0-7,503,360`, exactly the first half of each 15,006,720-row screen. A
10-update D3 preflight reached exactly 153,600 sampled rows and the registered
coefficient `0.9795291709`, while trainable actor rows were 153,597; this proves
the schedule is independent of league/trainability masks. It also passed SPS,
integrity, reward-reconstruction, and checkpoint gates. D1 is running.

### 2.4 Strategic battle curriculum

Draft credit cannot identify strategic cards if the battle policy has never learned to use them. Add a separate exposure arm after the credit correctness/schedule screen:

- a small, gate-balanced share of learner episodes starts from curated legal strategic decks;
- cover both gates and leaders for every element, both seats/starters, and diverse opponent elements;
- mask draft actor credit on curated-deck seats;
- train battle actions and value normally from real terminal outcomes;
- retain ordinary drafted episodes as the majority; and
- evaluate with curriculum disabled.

This is exposure, not reward. It lets the model experience successful weapon, spell, portal, Defender, response, IKZ-hold, and combo lines. Once battle value becomes real, exact terminal draft credit has a meaningful signal about cards that enable those lines.

A matched entity-only/face-pressure curriculum control is required. Otherwise improvement could come from easier fixed decks rather than strategic exposure.

## 3. League diversity and retention

### 3.1 Current failure mode

Production settings are:

- `frozen_ratio = 0.40`, which becomes an `0.80` frozen-match probability in a two-player environment;
- `frozen_window_epochs = 8`;
- `max_distinct_frozen = 1`; and
- chronological retention of six recent, four evenly spaced middle, and three oldest policies.

The final pool had 13 active policies, but all frozen games in an eight-update window used one sampled identity. The important quantity is therefore effective opponent diversity per learning window, not the on-disk checkpoint count.

The promotion archive does calculate payoff-vector distance for a `historically_distinct` panel member, but production sets `promotion_archive_affects_training_pool = false`, `promotion_anchor_in_training_pool = false`, and promotion cadence values to 100,000 updates. The archive did not shape the 65,105-update training run.

### 3.2 What “strategy vector” means here

There is no single strategy vector currently fed to the policy or PPO reward.

Two partial representations already exist:

1. an 18-value per-episode behavior record covering coarse action rates and realized mechanics; and
2. a payoff vector containing a policy's scores against common measured opponents, used only for promotion-panel diversity.

For training/evaluation, define a versioned **strategy descriptor** per checkpoint and context from:

- deck composition, curve, element/Normal share, unique/quads, and sibling-deck distances;
- opportunity-normalized action and resolution funnels;
- ordered element-specific sequence rates;
- game length, face/entity targeting conditional on alternatives, and resource-hold/conversion behavior; and
- payoff vector across gates, leaders, opponent elements, seats, starters, and protected policies.

Do not reduce this to one reward scalar. Use it for:

- evaluator reports and checkpoint clustering;
- archive admission and protected strategic roles;
- diversity-aware opponent sampling; and
- detecting regressions/cycles.

Use payoff distance as the primary strategic-retention signal when enough common games exist; it captures non-transitive differences that action-rate novelty can miss. Use behavior/deck distance as a coverage fallback and tie-breaker, not proof of quality.

### 3.3 Pool size and SPS

Thirteen active identities are enough for the first sampler/retention experiment. The current defect is that they are chronological and only one is exposed for eight updates. Enlarging the loaded pool first adds GPU memory and model-management cost without guaranteeing useful diversity.

First change temporal exposure at nearly constant forward-pass structure:

- `frozen_window_epochs = 1`;
- `max_distinct_frozen = 1`.

This still batches frozen inference through one identity per update, but can expose eight different identities over the eight updates that previously shared one opponent. Measure checkpoint-load/refresh cost and SPS rather than assuming it is free.

Only then test `max_distinct_frozen = 2` and `4`. Multiple identities inside one rollout split inference batches and are the likely SPS cost. Advance the smallest setting that materially improves strategy retention and stays inside the preregistered quality-adjusted SPS floor.

If a larger archive is needed, separate:

- a large disk/CPU metadata archive of every qualified or strategically distinct checkpoint; and
- a bounded GPU-resident working set selected at validation boundaries.

Do not load every historical checkpoint merely to make the pool count larger.

### 3.4 League arms

Use the selected reward and draft treatment as a fixed parent recipe.

| ID | Frozen window/distinct | Retention/sampling | Purpose |
|---|---|---|---|
| L0 | `8 / 1` | Current PFSP + chronological 6/4/3 | Negative control |
| L1 | `1 / 1` | Same pool and PFSP | Isolate temporal interleaving at low SPS cost |
| L2 | `1 / 1` | Role quotas over the same 13-policy budget | Isolate strategic retention/sampling |
| L3 | `1 / 2` | Same role quotas | Test within-update diversity/SPS tradeoff |
| L4 | `1 / 4` | Same role quotas | Upper operational comparison, not presumed winner |
| L5 | Selected window | Frozen row ratio `0.25` instead of `0.40` | Compare 50% versus 80% historical matches |

The L2 role budget must include:

- immutable qualified anchors/upper-envelope checkpoints;
- recent policies;
- hard PFSP opponents;
- payoff-distinct strategic policies; and
- old/middle anti-forgetting policies.

Every role gets a nonzero sampling floor. PFSP hardness cannot consume the whole distribution, because an emerging strategic policy may initially lose to face pressure and may also need rehearsal against opponents it already beats.

### 3.5 Strategic incubation and admission

Do not require a novel strategic checkpoint to beat the current cheap-policy champion immediately. Give strategically distinct, integrity-safe checkpoints a protected evidence window of at least two scheduled validations. During that window:

- keep the checkpoint available as an opponent;
- sample it at a small floor;
- measure whether later policies learn to exploit and surpass it;
- retain its payoff/strategy descriptor even if it is not deployment-qualified; and
- prevent it from displacing the immutable strength anchor unless it passes the later strength contract.

This is not permission to preserve every weak checkpoint. Admission requires reproducible strategy novelty, opportunity-normalized use rather than card presence, and no gross integrity or strength failure. Protection is time-bounded and evidence-based.

## 4. Isolate the persistent shaping floor

Production reaches `0.15` after 12 warmup plus 40 ramp episodes per native environment. This is effectively a short initialization transient followed by an almost entire run at `0.15`; it is not “15% of the training run.”

Run this stage only after selecting the potential formula, exploration terms, and draft-credit schedule. Otherwise a floor result cannot be attributed.

Separate the schedules introduced in Stage 1 and compare:

| ID | Potential scale tail | Exploration scale tail | Draft-credit tail | Purpose |
|---|---:|---:|---:|---|
| S0 | Selected baseline | `0.15` | Selected draft schedule | Control |
| S1 | Same | `0.05` | Same | Lower persistent proxy pressure |
| S2 | Same | `0.00` late tail | Same | Test pure terminal conversion late |
| S3 | `0.00` late tail | Selected exploration floor | Same | Isolate potential optimization effect |

Use global sampled-row ramps and retain several checkpoints after the final multiplier is reached. A zero-tail arm is informative only if terminal and draft-credit learning remain active and reward-component telemetry confirms exact zero for the intended terms.

Primary readouts:

- strategic sequence persistence and conversion;
- Lightning/Water structural presence;
- face-versus-entity decisions conditional on alternatives;
- reward-component magnitude relative to discounted terminal return;
- direct strength against cheap face-pressure ancestors over time; and
- late-window forgetting/cycle behavior.

The selected tail is the lowest proxy pressure that preserves exploration and converts strategic support into later strength. Do not retain `0.15` merely because prior validation was neutral; the prior late-zero experiment was conditionally skipped before the 1B failure evidence and did not test this corrected decomposition.

## 5. Replace saturated production validation

Stage 0 supplies trustworthy metrics. This stage changes model selection and automated promotion so the metrics matter.

### 5.1 Strength panel

The old fixed-reference suite remains only a regression floor. Replace its role as the primary ranker with:

- qualified p21000/p44000 upper-envelope checkpoints;
- p60000 as the strongest failed-scheme late checkpoint;
- cheap face-pressure policies and decks;
- payoff-distinct historical policies;
- curated strategic fixed decks/agents for all elements; and
- a rolling holdout of recent candidates not used for training selection.

Run the complete gate × leader × opponent-element × seat × starter schedule. Report pooled score, worst element, worst gate/leader, worst opponent family, and paired deltas. A pooled number cannot hide a failed element.

Retire or strengthen any panel that remains above 90-95% for an extended window and has no useful slope. Saturation is a panel failure, not evidence of continued learning.

### 5.2 Strategy panel

Every checkpoint packet includes:

- exact sibling-deck collisions and Jensen-Shannon/composition distances;
- Normal/element share, card types, cost curve, unique cards, and quad slots;
- full card and mechanic funnels;
- ordered sequence opportunities, completions, wins, and post-sequence turns;
- gate/leader context KL in deterministic and stochastic modes;
- face/entity attack choice when both are legal;
- IKZ hold duration and later conversion; and
- reward-component distributions for training rollouts.

Do not create one pooled “strategy score.” Use a dashboard plus Pareto decision rules. A single scalar can trade complete Water collapse for more Fire actions and recreate the current failure.

### 5.3 Short-run and long-run decisions

For 15M/45M screens:

1. integrity and runtime are hard requirements;
2. broad strength has a non-catastrophic floor, not a champion-beating requirement;
3. strategy acquisition, element coverage, and positive later-window slope are primary;
4. at least one later confirmation window is required; and
5. checkpoint-window evidence overrides one lucky endpoint.

For 100M+ confirmation:

1. strategic support must persist without curriculum/evaluation forcing;
2. the policy must begin converting strategy into payoff against cheap policies;
3. no element may be structurally absent;
4. worst-context strength must stabilize or improve; and
5. archive/league diversity must remain active in actual rollout exposure.

Promotion cadence must be reachable within the run. Promotion and protected archive members must affect the training pool in the treatment; a shadow-only archive is evaluation instrumentation, not a league intervention.

### 5.4 Strategy-first interpretation of short screens

For 15M and 45M ablations, win rate against an established aggressive policy is not a ranking metric or a tiebreaker. Cheap face pressure is expected to mature earlier than sequencing, resource reservation, payoff conversion, and element-specific deck construction. An early strategic learner may therefore lose more while still being the better long-horizon candidate.

Rank short-run arms by element-specific evidence:

1. real eligible opportunities followed by ordered-sequence completion and payoff conversion;
2. presence in both sampled and deterministic traces, with deterministic absence treated as unresolved;
3. multiple learned lines within each element rather than one repeated mechanic;
4. sibling-deck and action differentiation in contexts where their values differ;
5. conditional face-versus-entity attack mix, treated descriptively unless a curated value-labeled probe establishes which action is stronger; and
6. persistence or positive slope across later checkpoints after curriculum, draft credit, and reward pressure decline.

Win rate remains useful only as a long-horizon conversion readout and as a short-run catastrophic-integrity guard. It may veto an arm at 15M only when the regression is broad enough to indicate broken learning and is accompanied by absent or degrading strategy evidence. It must not override positive strategy acquisition merely because a mature aggressive opponent wins the matchup.

Every short-run decision packet must separate `strategy evidence`, `runtime/integrity`, and `deferred strength conversion`. Historical dispositions based mainly on 15M direct or anchor win rates require strategy-first reassessment before they select or reject a longer confirmation arm.

## 6. Fresh combined canary before another long run

Do not use a continuation as the final proof. Continuations answer causal questions from a stable parent; the production candidate must demonstrate that the combined recipe learns from random initialization without inheriting p21000's policy.

### 6.1 Composition

The canary combines only qualified changes from Stages 0-5:

- corrected reward telemetry and proper terminal closure;
- selected potential/exploration formula and schedules;
- selected exact-draft-credit schedule;
- cross-gate exact-credit exclusion, with replay retained only if its control passed;
- selected strategic exposure curriculum;
- selected league window, role sampler, retention, and frozen ratio; and
- unsaturated strength plus strategy validation.

No new architecture, gate-ID channel, element-specialist model, or additional reward term enters this run.

### 6.2 Horizon and gates

Run a fresh 100M sampled-row canary with decision packets at approximately 5M, 15M, 30M, 60M, and 100M. Use dense early checkpoints and preserve the whole late window.

- **5M:** mechanics/opportunity discovery; no strength promotion.
- **15M:** element coverage and reward/draft/league telemetry integrity.
- **30M:** strategy persistence after early curriculum/reward pressure begins to fall.
- **60M:** first skill-ceiling check against cheap face-pressure policies.
- **100M:** late-window strategy, strength conversion, and forgetting decision.

Do not stop merely because strategic play initially loses to R0/p44000. Stop for integrity failure, structural collapse across repeated windows, no strategy acquisition despite real opportunities, or a broad strength regression with no improving strategy trajectory.

### 6.3 Canary acceptance

Advance only if all hold:

- Lightning weapons and Water spells are drafted and converted in opportunity-normalized traces; neither deterministic family is structurally absent.
- Earth and Fire retain multiple valid lines rather than carrying the pooled result alone.
- Sibling decks/actions differ where the fixed-deck interaction probes establish different value.
- Curated value-labeled interaction probes show that face attacks remain efficient when correct and do not dominate stronger legal setup/control alternatives.
- Strategy support persists after reward/draft/curriculum anneals.
- The 60M-100M window improves against cheap-policy ancestors or shows a statistically credible positive conversion slope while respecting the strength floor.
- The actual opponent-exposure log demonstrates role and temporal diversity; configured pool size alone is insufficient.
- Quality-adjusted SPS is acceptable for the intended hardware and no integrity guard regresses.

A passing 100M canary advances to a 450M confirmation with the same recipe. Start another 1B run only after the 450M late window preserves strategy breadth and converts it into strength. Do not modify the recipe between 100M, 450M, and 1B without restarting attribution.

# Experiment ledger

Update this table when an arm is registered, launched, completed, or rejected. Every row must link its config, parent manifest, result artifact, and decision report.

| ID | Status | Parent | Configured rows | Single changed boundary | Primary result | Decision artifact |
|---|---|---|---:|---|---|---|
| BASE-1B | Complete / rejected | Fresh production run | 1,000,000,000 | Current recipe | Healthy run; strategy-learning flunk | `results/corrected_production_1b_lr1500_fresh/gates/gate_1b_summary.json` |
| EVAL-0 | Complete | N/A | N/A | Evaluator/telemetry correctness | Reward telemetry passed at 1.03% overhead; descriptor v1, trace v2, seven-checkpoint/21-pair sensitivity, and 32-game curated panel passed | `results/strategy_baseline_v1/baseline_packet.json` |
| R0-R4 | Complete; strategy-first reassessment keeps R1 and R3 | Same registered fresh seed-42 parent | 15,006,720 each | Reward formula; D0 cross-gate credit held fixed | R1 is the balanced Earth/Fire/Water leader with positive late slopes; R3 is the strongest Fire candidate, while each arm's conditional face-versus-entity mix is descriptive only; no arm repairs Lightning | `results/reward_screens/decision_packet.json` |
| R5-R6 | Conditional | Selected reward parent | 15M then 45M qualifier | Combat-state potential removal | — | — |
| D0-D3 | Complete; strategy-first reassessment keeps D2 and D3 | R1 reward recipe, fresh seed 42 | 15,006,720 each; D0 reuses R1 | Exact draft-credit schedule/correctness | D2 leads broad sampled Water/Earth evidence; D3 leads deterministic Water replay and sibling differentiation; neither repairs Lightning | `results/draft_screens/decision_packet.json` |
| CURRICULUM | Complete; C1 remains a strategy candidate | D2 draft recipe, fresh seed 42 | 15,006,720 each | Fixed learner deck composition: entity-only control vs strategic archetype | C1 improves sampled Fire, opportunity-normalized deterministic Water, and curated interactions; its conditional face-versus-entity mix shifts toward face attacks, and Lightning remains absent | `results/curriculum_screens/decision_packet.json` |
| L0-L5 | Complete; L1/L4 show strategy but fail runtime; L5 is not the strategy leader | C1 curriculum recipe, fresh seed 42 | 15,006,720 each | League sampling/retention | L1 is the most balanced strategy result and L4 the Water specialist; both miss the runtime floor. L5 is fast but loses late element strategy and role exposure remains absent | `results/league_screens/decision_packet.json` |
| S0-S3 | Complete; no single strategy winner | L5 league recipe, fresh seed 42 | 15,006,720 each | Global-row potential/exploration shaping tail | S0 leads Earth/deterministic Water, S1 shows Fire+Water growth, S2 specializes in Water, and S3 has the strongest Fire evidence; conditional face-versus-entity mixes are descriptive only, and all arms retain Lightning absence | `results/shaping_screens/decision_packet.json` |
| S4 | Complete / rejected on strategy evidence | S0 final recipe, fresh seed 42 | 15,006,720 | Discounted PBRS plus terminal closure | Strategy evidence is mixed: curated interactions improved, endpoint Water breadth fell, and Lightning remained absent. Its shift toward face attacks when both were legal is not itself positive or negative evidence; the rejection does not depend on 15M win rate | `results/terminal_closure_screen/decision_packet.json` |
| STRATEGY-REASSESS | Complete | EVAL-0 and R0-S4 | N/A | Remove early win rate as short-run ranker | R1/R3, D2/D3, C1, L1/L4, and the S0-S3 Pareto set show different strategic progress; Lightning is absent everywhere | `results/strategy_first_reassessment.json` |
| CANARY-100M | Held on strategy coverage | Fresh random init | 100M | Combined qualified recipe | Validation contract frozen; launch held because no candidate covers all four elements, especially Lightning—not because it loses to mature aggression at 15M | `results/canary_100m/registration.json` |
| CONFIRM-450M | Blocked on canary | Fresh canary recipe | 450M | Scale confirmation | — | — |
| PRODUCTION-1B-V2 | Blocked on confirmation | Fresh confirmed recipe | 1B | Final scale run | — | — |

# Immediate implementation sequence

1. Implement and verify reward-component reconstruction telemetry.
2. Version the strategy descriptor and repair opportunity/ordered-sequence evaluators.
3. Recompute the baseline packet for p21000/p29300/p44000/p52000/p56000/p60000/p65105.
4. Implement proper PBRS terminal closure behind explicit config controls.
5. Implement cross-gate retained-credit exclusion and a sampled-row draft-credit schedule.
6. Run reward screens before changing league behavior.
7. Run draft and strategic-exposure screens.
8. Run league sampler/retention/SPS screens.
9. Run the shaping-tail experiment with all earlier winners fixed.
10. Freeze the new validation contract, then launch the fresh canary.
