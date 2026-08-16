# Reference-Deck League Ablation

## Objective

Test whether a small stationary reference-deck share improves both drafting
and competitive play without teaching the learner to copy human actions. This
is a matched 15M continuation ladder first. No arm advances to 45M on
training-reference performance alone.

The reference decks are opponents to beat, not action or deck-construction
targets. There is no behavior-cloning loss, draft-pick label, similarity
reward, or reference-action reward.

## Frozen Parent

All arms start from the same promotion-v2 p2930 endpoint:

- model:
  `experiments/azuki_local_promotionv2_archive45_final_resume1500_178433625308/model_azuki_local_002930.pt`
- model SHA-256:
  `7196734b2250ea1901e0e74e7bd693ebe6cf0918a5e51e5d231a3424b5d045d4`
- trainer state:
  `experiments/azuki_local_promotionv2_archive45_final_resume1500_178433625308/trainer_state_002930.pt`
- trainer-state SHA-256:
  `0aa5da87f27801ed008c984ef64373874eec843e17558b1af2710d7e9b2bc5e1`
- league-state SHA-256:
  `7b1c8e5a932694d3370305d542231c63ca28ce8b6ee062563fa80d930675f707`
- promotion-state SHA-256:
  `9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a`

p2930 received a separate frozen full qualification before selection as the
shared parent. It passed with a 71.9% pooled panel score, 68.8% paired lower
bound, 65.6% to 78.1% across the four opponents, and 71.9% from both seats.
Its fixed-reference score was 50.35%; relative to the anchor it was -14.24
points with a -17.36-point paired lower bound, clearing the -15 and -18 point
non-inferiority limits. All games completed without timeout. The immutable
artifact is under
`results/promotion_p2930_qualification_v1/`.

Promotion is shadow-only in this ladder. Neither a promotion outcome nor the
quality archive changes an arm's PPO opponent pool, and promotion is not an
arm-selection signal.

## Matched Arms

| Arm | Fixed reference matchup probability | Purpose |
|---|---:|---|
| `control` | 0% | Matched continuation control |
| `ref05` | 5% | Low stationary-anchor exposure |
| `ref10` | 10% | Upper small-share exposure |

Each arm resumes the p2930 optimizer and starts a fresh cosine schedule at
`3e-4` with entropy fixed at `0.002`. The configured total is 90M cumulative
sampled rows, but the driver pauses atomically at p3900. This adds 970 updates,
or 14,899,200 sampled rows, while preserving a scheduler state that can
continue the selected arm to a full 45M addition without another LR restart.

All arms retain the complete successful parent stack:

- reward shaping at its mature 0.15 floor, with the serialized 12/40 episode
  progression restored rather than restarted;
- portal-GP bonus 0.3 and cross-gate replay mask;
- PFSP and the frozen ratio 0.40 league;
- early tempo 0.10 with cap 4;
- damage mitigation 0.15 with cap 10;
- temporary Charge realization 0.08;
- temporary attack realization 0.025 per effective damage with cap 4;
- pick smoothing 0.02, text-only gates, the 6/4/3 league retention profile,
  seed 42, and the exact frozen league snapshot.

## Opponent-Only Semantics

The C environment samples a fixed reference deck only on episode reset. The
training split is the nine even pool indices
`0,2,4,6,8,10,12,14,16`; the nine odd indices are never sampled during PPO.

On a sampled reference matchup:

1. The fixed-deck seat is detected from the packed reset observation.
2. The learner is assigned to the opposite, drafting seat.
3. The already-selected frozen PFSP policy pilots the fixed deck.
4. Only learner rows are trainable; the fixed-seat pilot supplies no action
   labels or gradients.

This also prevents a reference deck from replacing an ordinary latest-policy
matchup and silently raising the frozen share. With target frozen matchup
fraction `t` and reference probability `p`, ordinary resets use
`q = (t - p) / (1 - p)`. For `t=0.80`, the 5% and 10% arms use 0.78947 and
0.77778 respectively, leaving the expected total frozen fraction at 0.80.
The 0% arm consumes no extra reference RNG draw.

## Validation Before Launch

The 10% integration soak resumed p2930 for 12 updates. It completed 44
reference games, and the learner drafted in all 44. The configured reference
rate was 0.10, the compensated ordinary frozen rate was 0.77778, and total
frozen matchups remained near the 0.80 target. Reference rows entered both
draft and battle training phases.

After compile warmup, median SPS was 1,675. This is within the observed mature
full-league range of roughly 1,300 to 1,600 SPS and above the hard 1,300 SPS
floor. The campaign guard stops an arm only if two consecutive non-overlapping
20-row median windows are both below 1,300 SPS, avoiding false stops on compile
or promotion-pause rows.

The fast native readout was also exercised against the compiled environment.
It reconstructs every 50-card learner deck and retains per-game action,
leader-health, seat, start-player, gate, outcome, and timeout telemetry. A
32-game p2930 self-H2H canary scored exactly 50% with no timeout.

Focused validation before launch: 38 Python tests passed, including native
evaluation controls, promotion schedules, raw reference metrics, opponent-only
seat alignment, frozen-share compensation, and legacy readout metrics. Python
compile checks, `bash -n`, `shellcheck`, and `git diff --check` also passed.

## Readout

Every arm receives identical deterministic schedules:

- 288 games against the nine exposed reference decks, with the p2930 frozen
  parent piloting the fixed seat;
- 288 games against the nine heldout reference decks, also parent-piloted;
- 192 native H2H games against p2930 across six independent, paired-seat,
  paired-gate schedules;
- clean learner-only deck snapshots and training action/reward telemetry;
- SPS distribution, total frozen share, observed reference exposure, and
  completed-game seat-alignment telemetry.

Deck similarity is diagnostic. A rise in similarity to exposed decks is not a
success unless heldout fixed-deck strength and ordinary H2H also improve.
Action-rate changes are descriptive playstyle evidence, never a target to
match human frequencies.

An arm advances to 45M only if:

- reference seat integrity, zero-timeout, and throughput checks pass;
- deck diversity does not collapse;
- heldout performance is not meaningfully worse;
- at least two of exposed-reference score, heldout-reference score, and
  native H2H improve by at least two points over the matched control; and
- the result is not an imitation-only pattern: increased exposed-deck
  similarity with flat or worse heldout and H2H outcomes.

The driver writes a paired-bootstrap JSON and Markdown report after all three
arms. Campaign artifacts live under
`train-ablation-1781126582/results/reference_ladder15_v1/`. The launcher is
`train-ablation-1781126582/run_reference_ladder15_v1.sh`.
