# Local Production Training Run

Status: **NO-GO for the successor 1B run.** The historical control remains rejected. Schema-v3 revalidation preserves R14's reward-contract/integrity pass but withdraws its strategy qualification: corrected sampled Fire breadth falls from three converted lines at p1950 to two at p2925. The user restored GPU operation by restarting the host. The newly authorized strategy-discovery campaign is diagnostic only; it does not authorize a 1B launch.

## Document map

This file is the operational status and runbook for local production training. It now records both the completed historical control and the remaining qualification path for its successor.

- Canonical experiment preregistration, arm definitions, and live ledger: `train-ablation-1781126582/strategy-experiments.md`.
- Historical 1B automated result: `train-ablation-1781126582/results/corrected_production_1b_lr1500_fresh/gates/gate_1b_summary.json`.
- Final R14 reward decision: `train-ablation-1781126582/results/terminal_safe_reward_horizon_followup/final_decision_packet.json`.
- R14 reward config under strategy requalification: `python/config/azuki_reward_r14_tail_030_anneal_100_45m.ini`.
- Current successor validation rollup: `train-ablation-1781126582/results/successor_final_validation/decision_packet.json`; command evidence: `results/successor_final_validation/verification.json`.

The detailed experiment document remains authoritative when a registered arm and this operational summary differ. Update both documents when a recipe boundary is qualified or rejected.

## Current successor qualification status

The completed 1B control is evidence, not the recipe for the next run. It finished with healthy optimizer and environment telemetry but failed the stated strategy-learning objective: Lightning remained effectively gate-insensitive, Water drafted no deterministic spells, and the best late checkpoint did not displace p44000. The complete final-control result is preserved under `Final 1B automated validation outcome` below.

The next candidate must start from random initialization after every recipe boundary is frozen. Current state:

| Workstream | Status | Result | Decision evidence |
|---|---|---|---|
| BASE-1B | Complete / rejected | Healthy 1B execution; strategy-learning objective failed | `results/corrected_production_1b_lr1500_fresh/gates/gate_1b_summary.json` |
| EVAL-0 | Revalidated at descriptor v3 | 218 retained ablation descriptors and seven baseline checkpoints rebuilt; 24 focused CPU regression tests pass; trace-v2 attribution limits are explicit | `results/strategy_baseline_v1/baseline_packet.json` |
| R0-R4 | Screens complete; old rankings superseded | Effect, recovery, timing and conversion semantics changed; R3 removal is already implemented by R14 | `results/reward_screens/decision_packet.json` |
| R5-R14 | Reward math passes; strategy qualification held | R14's corrected sampled Zero line converts once at p1950 and zero times at p2925; registered no-reduction gate is not met | `results/terminal_safe_reward_horizon_followup/final_decision_packet.json` |
| D0-D3 | Longer confirmation blocked | Existing D2/D3 are R1-parent 15M screens, not R14 45M confirmations; replay-off control also absent | `results/draft_screens/decision_packet.json` |
| Strategic exposure | Qualification blocked | Existing C0/C1 use the old D2 recipe; require selected-parent no-exposure, entity-only and strategic comparisons | `results/curriculum_screens/decision_packet.json` |
| L0-L5 | Qualification blocked | L1-L4 miss the runtime floor; L2-L4 have zero completed distinct-role share; configured roles do not prove diversity | `results/league_screens/decision_packet.json` |
| S0-S4 | Interaction tests blocked | S1/S3 remain separate hypotheses on an otherwise frozen recipe, not selected production settings | `results/shaping_screens/decision_packet.json`, `results/terminal_closure_screen/decision_packet.json` |
| Strategy reassessment | Observations rebuilt; rankings withheld | Per-context elemental effects replace hardcoded v2 conclusions; early win rate remains supporting evidence | `results/strategy_first_reassessment.json` |
| Staged successor 1B | Blocked | No final recipe or launch authorization; superseded 100M registration is rejected by the launcher | `strategy-experiments.md`, section 6 |

### Schema-v3 final validation outcome

The v3 results supersede all earlier v1/v2 strategy dispositions in this file. Original decision artifacts are preserved by SHA-256 in `results/strategy_semantics_v3/prior_decision_manifest.json`; training checkpoints, reward settings and trajectories were not changed.

The evaluator now distinguishes attack damage from eventual victory, resource addition from untapping, spell recovery from a second same-code copy, and temporary effects from attacks after expiration. Rushfire uses the real `SELECT_TO_GARDEN` path and requires its destination slot to attack. Echoed Waves tracks immediate deferred recovery in both sequences and card lifecycles. Lightning requires the equipped destination slot to attack with the weapon still attached. Bobu conversion requires Earth loss plus observed healing inside its active window, including opponent-triggered destruction. Alley damage uses the trace producer's leader-offset slot encoding. A spell's own-unit destruction alone is not a positive effect.

Endpoint p2925, 200 games per mode:

| Element | Sampled evidence | Deterministic evidence | Interpretation |
|---|---|---|---|
| Water | 128 resolved spells; 8 observed Echoed replay effects; 30 resource-to-spell lower-bound units | 109 resolved spells; 24 observed Echoed replay effects; Hydromancy readies 281 IKZ but drafts zero spells in all 48 Hydromancy player-games | Pooled Water success hides a gate-specific resource/spell gap |
| Earth | 123 defenders selected; 165 leader HP restored; 9 Bobu loss/heal conversions | 115 defenders selected; 311 leader HP restored; 18 Bobu loss/heal conversions | Defensive/healing support exists, including opponent-triggered destruction; these counts do not prove optimal timing or marginal damage prevention |
| Fire | Rushfire/Kagoro conversions 44/3; Zero 0/49 eligible player-games | Rushfire/Kagoro 43/6; Zero 0/49 | Multi-play/Charge is not evidence of self-damage strategy; sampled Zero converted 1 at p1950 then 0 at p2925 |
| Lightning | 108/108 decks contain weapons; 152 equipped attacks deal observed damage | 108/108 decks contain weapons; 161 equipped attacks deal observed damage | Weapon support is real; damage while equipped is not a causal estimate of weapon-added damage |

R14's registered no-reduction gate fails on the corrected sampled Fire window. One rare conversion disappearing is not statistically conclusive permanent forgetting; it is nevertheless insufficient to claim the preregistered pass. Deterministic Zero conversion is absent in both late windows. Preserve the reward math result, but do not advance a downstream recipe based on the old “stable all-element breadth” statement.

The final provenance audit verifies all 218 ablation descriptor IDs, seven baseline descriptor links, 21 unique baseline pairs, and 17 archived original decisions. All 16 reward-registration entry paths reject superseded parents; the superseded canary launcher also rejects activation before training. Seven historical reward builders emit only superseded observations. The R5/R6 builder instead rejects an existing registration-hash mismatch (`evaluation_index.json` records `3b405a6d…`, current registration hashes to `490041c8…`); that provenance failure is retained, not bypassed or repaired by changing the index.

Historical hardware blocker, now resolved by the user's restart: the earlier GPU query timed out and clients waited on driver locks. A subsequent actual CUDA R14 policy smoke completed normally (one sampled game, 136 battle steps, no truncation). The archived failed-runtime evidence remains historical; it is not a current blocker.

### Completed diagnostic campaign

Managed process `strategy-discovery-v1` completed the authorized queue. Exact registration and terminal execution state:

- `results/strategy_discovery_v1/registration.json`
- `results/strategy_discovery_v1/campaign_status.json`
- Initial paired-deck diagnostic: `results/strategy_deck_diagnostic_v1/registration.json`
- Prelaunch evidence: `results/strategy_discovery_audit_v1/`

Order: 1,536 same-policy games on six CPU workers; then serial GPU training/evaluation for seven 15M screens (shared control, entity exposure, strategic exposure, random prefix, strategic prefix, replay-off, learner gamma 1.0) and matched 45M D2/D3. Total configured training rows: 195,056,640. This is a set of separate diagnostic arms, not a combined recipe or production extension.

Verified before launch: 99 focused tests; two ten-update CUDA training smokes with checkpoint parity; 796 strategic forced picks; actual supplied learner battle-row accounting; six complete paired fixed-deck games; and a successful actual guarded-launcher/checkpoint smoke. Source/config drift, production scope and 1B-sized arms are rejected. The queue has no automatic production promotion and stops for integrity/runtime/provenance failures.

All nine training/evaluation arms and the initial deck diagnostic are complete. The review in `results/strategy_discovery_v1/review.json` grants no production promotion: strategic prefixing retains a useful Water-transfer signal but loses early Fire conversion and trails the strength controls. Other interventions do not establish a robust replacement recipe. All 11,600 self-play and 16,704 H2H games complete; one of 288 curated games is incomplete and excluded from observed-score accounting. Per-arm indexes and the experiment ledger retain temporal and gate/leader evidence. Water and Earth proposals each have one compatible regional source plus a labeled derived variant, not broad archetype coverage.

Discord notifications ran separately in managed process `strategy-discovery-discord`, using `monitor_strategy_discovery.py`. The final notification was delivered and the watcher exited normally. It only read progress artifacts and campaign process identity; it did not modify frozen sources, stop training, launch experiments, or spawn model sessions. Its configured cadence was 60-second polls and 30-minute progress/stall notifications.

Final notification: **CAMPAIGN FINISHED — READY FOR REVIEW**, emitted when the campaign reaches `completed_observations_require_review`. A failure is reported separately, never as successful completion. The user must resume/open an assistant session to review results and authorize subsequent decisions; the 1B prohibition remains unchanged.

The webhook remains in the owner-only file `~/.config/azuki-tcg/discord-webhook-url`; neither its URL nor token is logged. The resumed process uses fresh delivery state in `results/strategy_discovery_v1/discord_monitor_continuation_state.json`, so the prior PID identity and stopped-event deduplication cannot suppress new alerts. The original `discord_monitor_state.json` retains the first attempt. Failed deliveries retry on subsequent polls. Verification of incremental/partial records, completion/failure distinction, stall/recovery, and pending notifications is retained in `discord_monitor_verification.json`; live continuation delivery was accepted.

#### Report recovery and reviewed continuation

The first attempt completed all 1,536 unique games (768 per mode, no incomplete games), then failed at report generation because a relative output path was passed to `relative_to()` with an absolute repository root. No training arm had started. The Discord failure notification was delivered.

`run_deck_diagnostic.py` now resolves the report directory and supports `--report-only`, requiring hashes for exactly the retained trace shards. Missing/changed trace hashes are rejected; normal execution still refuses existing traces. Recovery completed with a relative registration path and left all twelve traces byte-identical. The original registration remains unchanged; `results/strategy_deck_diagnostic_v1/registration_report_recovery.json` records the report-runner-only source revision, original registration hash and retained trace hashes. The failed status, original runner and monitor state are archived under `results/strategy_discovery_audit_v1/report_recovery/`.

Review: strategic decks increased Lightning attacks with observed damage while equipped (learned -> strategic: sampled 102 -> 238, deterministic 87 -> 236; 64 candidate games per element/arm/mode). Water gained deterministic Hydromancy spell access, but 60/60 sampled and 63/64 deterministic strategic spell selections have unmeasured immediate effects; zero recorded positive effects is not proof of useless spells. Learned and strategic Fire still have zero observed Zero self-damage-to-attack conversions in either mode. Earth support is mixed. These observations motivate the registered exposure/prefix comparisons, not a recipe promotion. Supporting scores are not uniformly improved.

The reviewed integrity gate passed and the unchanged training registration subsequently completed under `strategy-discovery-v1`, omitting `--deck-diagnostic` because those games and reports were complete. Launch required `PYTHONPATH=build/python/src:python/src:train-ablation-1781126582`; the trainer received its registered runtime environment from the runner. The continuation review, report hashes, candidate elemental rollup and guard-smoke evidence are in `results/strategy_discovery_audit_v1/report_recovery/review.json`. No training source, config, arm, seed or row budget changed. The completed-results review recommended targeted retained-checkpoint prefix evaluation before any new training; its subsequently authorized launch is recorded below. No 1B run has launched.

#### Retained-checkpoint paired evaluation reviewed

After reviewing the completed campaign, the user authorized evaluations and inspections, not new training. Managed process `prefix-paired-followup-v1` completed `run_prefix_followup.py` with six CPU workers. Registration and execution status are under `results/prefix_paired_followup_v1/`; all 9,216 free-draft games completed across control/random-prefix/strategic-prefix, three checkpoints, both modes, both leaders per gate, two frozen opponents and seat swaps. Separate policy histories persist from draft into battle. Discord completion delivery succeeded.

Eight real smoke games passed; candidate-only summaries and three repeated-task action/outcome matches were verified. The subsequent review checked all 108 trace hashes, registered assignments and engine-terminal outcomes, with 256 actual first/second starts per checkpoint/mode and no incomplete games or draws. `review.json` contains full score/context/mechanism aggregates and the decision; `retained_inspection.json` contains earlier self-play trace inspections, and `verification.json` contains launch proof. Strategic prefixes trail both controls at every checkpoint in both modes; final argmax wins are 102/512 versus CONTROL 163/512 and RANDOM 160/512. Early Zero conversions disappear by p650 despite continued ability use. Final deterministic Water spell use is exclusively Healing Flutter (88 selections across all four Water contexts), not broad Water mastery. Earth/Lightning checks do not establish a compensating gain. Do not promote or launch unchanged independent-seed strategic-prefix replication; preserve the narrow findings for separately registered targeted investigation. The experiment ledger records details and limitations. No new training or 1B launch.

#### Strategy-first replicated recipe campaign reviewed

The user superseded the winrate-influenced no-replication recommendation above: unique strategy emergence, persistence and diversity are the diagnostic objective. Early Zero/Stormchain acquisition and persistent Water healing/replay/resource behavior justify further strategy experiments even without a short-run strength gain.

Managed process `strategy-recipe-v1` completed the immutable registration at `results/strategy_recipe_v1/registration.json` (SHA256 `4e452466ecd57066240ffe37d5a08efc6806defd8dbe0ff74f3a732b9ea6dd67`). Five recipes at seeds 43 and 44: CONTROL, RANDOM20, STRATEGIC20, STRATEGIC50 and STRATEGIC20_HOLD_EXPLORATION. Each has 15,006,720 configured rows, p325/p650/p975 evaluations, both modes and the shared 512-task free-draft candidate-only panel. Totals: 150,067,200 configured training rows and 30,720 paired games; final learner steps were about 9.34–9.38M per run. The exploration arm holds the existing exploration-reward scale at 1.0; it does not change entropy or potential shaping.

Real end-to-end smoke: ten CUDA updates / 153,600 rows, verified checkpoint, eight completed paired games, zero integrity/invalid metrics, and delivered Discord start/evaluation/completion notices. The reporter additionally processed all 9,216 retained paired games, preserved all four Water contexts and Normal Healing Flutter support, and correctly reported missing batches as provisional. The final reporter's conditional converted/eligible field was checked against the known early Zero 10/64 result. Production scope, 1B rows, seed mismatch and source drift were rejected. Evidence and version boundaries are in `results/strategy_recipe_v1/verification.json`.

All 60 evaluation batches completed, the process exited zero, and Discord delivered completion with no notifications pending. `campaign_status.json` retains execution history; `strategy_report.json` and `review.json` contain the completed observations and review. STRATEGIC50 is recommended for longer retention confirmation: final Zero conversions are 31/64 and 23/64 in argmax, 29/64 and 19/64 in sample, with both seeds also converting at p650. Four final batches were independently recomputed from hash-verified traces. This is not an all-strategy winner: deterministic Water spell drafting did not replicate, Stormchain remains inconsistent, and Devotion has no observed conversion. Sampled Water healing/replay persists, including in controls.

Next proposal, not launched: a preregistered 50M-configured-row STRATEGIC50 confirmation with at least two seeds and a RANDOM50 dosage-matched comparator, intermediate retention checks and opportunity/diversity review in both modes. This campaign lacks RANDOM50, so higher prefix dosage is not separated from strategic content at 50%. No short-run winrate gate, automatic production promotion or 1B launch. Full strategy-first comparison is in `strategy-experiments.md`, under the completed recipe review.

#### Authorized retention confirmation launched with independent Discord watcher

The user approved the preceding proposal. `strategy-retention-v1` now runs STRATEGIC50 versus RANDOM50 at fresh matched seeds 43/44, 50,012,160 configured rows each (200,048,640 total), with p325/p650/p975/p1625/p2275/p3250 free-draft paired probes in both modes. Expected 48 batches / 24,576 games. Reward anneal endpoints remain at 15,006,720 rows; the longer fresh horizon changes the learning-rate schedule, so this is not a resumed 15M trajectory. Registration: `results/strategy_retention_v1/registration.json`, SHA256 `1aec81fcb8bd43c70fb10d35ed2af35e0360abac9d771b9a1e5cd3f499d88854`.

Persistent independent watcher `strategy-retention-discord` is attached to the campaign PID; Discord attachment delivery was observed. It polls every 60 seconds and sends 30-minute progress plus stage, inactivity, process-exit and completion alerts. The runner also sends stage/failure/completion messages. Both smoke recipes completed ten CUDA updates and 16 paired games, including successful independent completion notification and zero pending messages. `verification.json` records exact evidence and limits. No 1B run is launched automatically.

Storage at launch: about 951 GiB available, 73% used; prior recipe campaign 108 GiB, projected confirmation 144 GiB with 200 GiB planning allowance. No models or notes were deleted. If cleanup becomes necessary, the user's approval is oldest models first, preserving notes/progress/experiment explanations; do not remove checkpoints referenced by the active registration or its frozen opponents.

#### Completed retention review

The four-arm retention campaign completed all 48 batches / 24,576 games and verified update 3256 in every arm, with zero invalid/integrity maxima. Review: `results/strategy_retention_v1/review.json`; detailed interpretation is in `strategy-experiments.md`. Current registration/result hashes were checked; final Zero and Water totals were independently recomputed from eight hash-verified batches / 4,096 games.

STRATEGIC50's earlier Zero repertoire did not replicate under fresh 50M training (final argmax 0/64 and 4/64; sample 0/64 and 2/64). This is not evidence that the original 15M checkpoints forgot: they were not resumed and the LR horizon changed. RANDOM50 shows replicated late argmax Water healing/replay, largely through Normal Healing Flutter, while both recipes retain sampled Water and several Earth/Fire sequences. Lightning deterministic opportunity/persistence remains uneven; Devotion has no observed conversion. No unique recipe winner or winrate gate.

Recommended next experiment, not launched: continue the original STRATEGIC50 seed43/44 p975 checkpoints under an explicit continuation schedule, recording any state resets and checking the known repertoire immediately around resume and across later checkpoints. Preserve final RANDOM50 Water-repertoire candidates as references. No 1B launch. Review-time storage: 883 GiB available, 75% used; no cleanup needed or performed.

#### RANDOM50 continuation launched with strict resume and live Discord

The user selected the broad-repertoire RANDOM50 route, accepting legitimate card/gate/leader imbalance rather than requiring every sequence equally. The current authorized campaign is `results/random50_continuation_v1/registration.json` (SHA256 `32a045b85ea51d7bc876e8a640642bf93c195f0f3b7a06ad98759b4eb8352633`), superseding STRATEGIC50 Zero continuation as the primary next proposal.

Seeds43/44 resume their hash-locked p3256 model+trainer+league+promotion state in child-only roots to p13021 /200,002,560 cumulative configured rows. Original p3256 parents are evaluated before training; p6511 (~100M) and p13021 (~200M) follow. The original and independently seeded held-out512-task panels remain separate, both sample/argmax:24 batches /12,288 games total. Held-out opponents are unchanged. Reports: `strategy_report.json`, `strategy_report_heldout.json`. No automatic1B.

One new-child cosine restart at3e-5 over9,765 remaining updates is intentional: the saved parent cosine is exhausted. Preserve optimizer moments and all reward schedules; potential/exploration stay at.30/.15, and entropy/temperature/smoothing keep absolute learner-step schedules. Same-child recovery must load its saved scheduler without restarting. Strict compatibility, no critic reset, no model-only fallback. Worker environments/RNG/rollouts are not exactly resumed; see the full experiment notes.

Both-seed ten-update CUDA smoke passed with zero invalid/integrity maxima, exact checkpoint parity and64 real evaluation games. Independent Discord completion delivered. Full cosine and same-run recovery were separately exercised;12 unsafe mutations rejected. Evidence: `results/random50_continuation_v1/verification.json`.

Persistent campaign `random50-continuation-v1` and watcher `random50-continuation-discord` are live; independent attachment delivery and the first parent evaluation were observed. Runner requires that attachment before any run/eval. Watcher polls60s, posts30-minute progress plus stage/stall/exit/completion, and restarts on failure. Storage862GiB available/76% used; no model or note cleanup performed.

#### RANDOM50 continuation completed and reviewed

Both seeds reached p13021 /200,002,560 configured rows;24/24 batches and12,288 games completed across paired/heldout panels. Invalid/integrity maxima are zero; reward floors remain .30/.15. Learner steps are120,662,330 and120,767,974, below the150M learner-step entropy anneal boundary. Watcher completion delivery is recorded with no pending events.

Review: `results/random50_continuation_v1/review.json`, with verified24 result hashes and independent exact recomputation of all11 probes across4,096 final-checkpoint games. Broad sampled behavior persists across both seeds/panels, without clear replicated repertoire expansion. Seed43 argmax loses Echo opportunities between100M and200M because three of four Water contexts stop drafting spells; seed44 retains deterministic Echo/healing and Lightning recovery. Prefer seed44 p13021 as the retained-repertoire endpoint reference if one is needed; preserve seed43 p6511 rather than treating the newest model as automatically better. Normal support spells count; sparse mechanics alone do not reject the recipe. See the completed200M review in `strategy-experiments.md`. No additional training launched; no automatic1B or production qualification.

### Historical R14 reward configuration and v2 results

The following configuration and old tables preserve the experiment history, not a current strategy qualification. The reward reconstruction and training-health measurements remain valid; the v2 sequence counts/dispositions below are superseded by the v3 decision packet.

The state potential for player $i$ is:

$$
\Phi_i(s)=\tanh\left(
2.0\Delta h+
0.35\frac{\Delta\text{GardenAttack}}{10}+
0.15\frac{\Delta\text{UntappedGarden}}{5}+
0.15\frac{\Delta\text{UntappedIKZ}}{10}
\right)
$$

where $h(x)=0.5(x+1-(1-x)^4)$. The per-transition potential term is
$P(n)(0.99\Phi(s')-\Phi(s))$, with terminal closure
$-P(n)\Phi(s_{\mathrm{last}})$. Both sides receive opposite shaping values, so
the reward remains zero-sum.

Frozen reward settings:

- potential scale $P(n)$: `1.0 -> 0.30` over 100% of configured training rows;
- exploration/mechanic scale $E(n)$: `1.0 -> 0.15` over 100% of configured training rows;
- direct leader delta: `0`;
- direct board delta: `0`;
- early-tempo bonus: `0`;
- avoidable no-op penalty: `-0.02`;
- portal GP realization: `0.30 * min(GP, 4) / 4`;
- damage mitigation: `0.15 * min(soak, 10) / 10`;
- temporary Charge realization: `+0.08`;
- temporary ATK realization: `0.025 * min(incremental_damage, 4)`, capped at `0.10`;
- portal-outcome, entity-damage-exchange, generated-IKZ-conversion, and contextual-response-reserve bonuses: `0`;
- terminal win/loss/draw: `+5 / -5 / 0`, followed by potential closure;
- timeout truncation: `-0.35`;
- auto-tick or zero-legal-action truncation: `-0.60`;
- truncation leader and board edges: `1.25` and `0.45`.

There is no fixed episode-total reward because potential and mechanic terms depend on the trajectory and anneal position. In R14's final telemetry window, the mean winner terminal term was `+5` and mean scaled shaping was approximately `-0.0504`, for an approximate combined winner reward of `+4.9496`; the loser received the zero-sum opposite. This is a window mean, not a configured constant.

#### R5-R14 15M screen results

`Active L/W/F/E` is the count of registered completed strategy sequences in Lightning, Water, Fire, and Earth. Sibling distance is multiset-Jaccard deck distance: `0` means identical sibling-gate decks and `1` means disjoint decks.

| Arm | Anchor | Curated | Active L/W/F/E | Conditional face/entity | Sampled/argmax sibling distance | Disposition |
|---|---:|---:|---|---|---|---|
| R5 resource-only Phi | 0.0417 | 0.2813 | 2/3/2/2 | 0.166/0.267 | 0.714/0.010 | Too weak |
| R6 no Phi | 0.0191 | 0.3750 | 2/3/2/2 | 0.158/0.244 | 0.735/0.019 | Too weak |
| R7 half combat Phi | 0.2031 | 0.6875 | 2/3/2/2 | 0.383/0.129 | 0.651/0.000 | Advanced to 45M; failed late breadth |
| R8 full combat Phi | 0.2361 | 0.4063 | 2/3/2/2 | 0.661/0.013 | 0.697/0.000 | Rejected for targeting distortion |
| R9 potential tail 0.30 | 0.2465 | 0.6250 | 2/3/2/2 | 0.360/0.161 | 0.703/0.010 | Advanced to 45M; failed late Lightning |
| R10 potential tail 0.50 | 0.1788 | 0.5000 | 2/3/2/2 | 0.578/0.022 | 0.702/0.000 | Rejected for targeting distortion |
| R11 exploration tail 0.30 | 0.1944 | 0.4688 | 1/3/2/2 | 0.429/0.116 | 0.726/0.010 | Did not repair Lightning |
| R12 exploration tail 0.50 | 0.1250 | 0.5625 | 1/3/2/2 | 0.415/0.093 | 0.755/0.028 | Did not repair Lightning |
| R13 common anneal 75% | 0.2240 | 0.5625 | 1/3/2/2 | 0.387/0.151 | 0.723/0.005 | Advanced to 45M; failed late Lightning |
| R14 common anneal 100% | 0.2344 | 0.4688 | 1/3/2/2 | 0.367/0.137 | 0.727/0.010 | Advanced and qualified at 45M |

Registered direct comparisons agreed with the bracket decisions: R5 beat R6 at `0.7188`, R8 beat R7 at `0.6094`, R9 and R10 tied at `0.4948`, R11 beat R12 at `0.6458`, and R14 beat R13 at `0.5365`.

#### Fresh 45M confirmation results

| Arm | Anchor trajectory at p325/p975/p1950/p2925 | Endpoint active L/W/F/E | Curated | Result |
|---|---|---|---:|---|
| R7 | 0.069 / 0.191 / 0.245 / 0.314 | 1/3/1/2 | 0.4375 | Rejected: late Lightning and Fire loss |
| R9 | 0.099 / 0.148 / 0.319 / 0.408 | 1/3/2/2 | 0.3438 | Rejected: late Lightning loss |
| R13 | 0.057 / 0.160 / 0.276 / 0.280 | 1/3/2/2 | 0.3438, 31/32 complete | Rejected: late Lightning loss and incomplete curated panel |
| R14 | 0.0608 / 0.2240 / 0.3507 / 0.4392 | 2/3/2/2 (v2) | 0.4688, 32/32 complete | V2 pass withdrawn by v3 strategy revalidation |

The old v2 sampled active counts were `2/3/2/2`, `2/3/0/2`, `2/3/2/2`, and `2/3/2/2`. Corrected v3 counts are `2/3/2/2`, `2/3/1/2`, `2/3/3/2`, and `2/3/2/2`: the sampled Zero line appears at p1950 but is not retained at p2925. Do not interpret the old apparent stability as current evidence.

R14 endpoint and runtime evidence:

- sampled conditional face/entity rates: `0.353422 / 0.143879`;
- sampled sibling distance: `0.719567`, exact collisions `0`;
- deterministic sibling distance: `0.141471`, exact collisions `0`;
- steady median SPS: `1,412.979`;
- timeout, auto-tick, zero-legal-action, incomplete-draft, and invalid maxima: `0`;
- raw reward reconstruction maximum absolute error: `1.53e-8`;
- scaled reward reconstruction maximum absolute error: `4.52e-8`;
- PPO component reconstruction maximum absolute error: `2.861e-6`;
- sampled rows: `45,004,800`; final update: p2930; seed: `42`.

### Remaining recipe-finalization sequence

Production recipe boundaries remain blocked behind the corrected R14 strategy decision. The user has separately authorized fresh diagnostic comparisons on an explicitly unqualified R14-derived control; those observations cannot bypass production qualification. Every comparison must be preregistered, fresh, matched on seed and configured rows, and change only the named boundary.

1. **Draft credit:** after the reward hold is resolved, run longer matched D2 and D3 confirmations under immutable qualified rewards. Require exact-credit reconstruction, all-element opportunity-to-effect conversion and late persistence. Run the matched replay-off control after cross-gate credit exclusion. Early win rate is not a ranker.
2. **Strategic exposure:** compare no exposure, entity-only exposure and the retained strategic candidate on the selected draft recipe; evaluate without curriculum. Require actual opportunities/effects, gate/leader fit and no element loss. Conditional face choice alone cannot establish that an alternative was stronger.
3. **League:** settle sampling, retention, frozen ratio, role diversity, and temporal exposure. L1/L4 cannot advance unless implementation first meets the runtime floor and the exposure log proves real opponent-role diversity. Configured pool size is not evidence.
4. **R3 direct-edge boundary — resolved as redundant:** R14 already sets direct leader and board deltas to zero, confirmed by zero observed raw component grants. There is no remaining R3 removal to test. Reintroducing those terms would be a different, unregistered arm.
5. **Final shaping interactions:** test S1 and S3 separately against the frozen winner. Drop an arm if R14 already implements the same effective condition; do not combine two unqualified changes.
6. **Freeze the candidate:** write the final config, source/config hashes, random seed, 1B absolute-row schedules, evaluation registrations, checkpoint cadence, recovery contract, and immutable decision packet before row zero.

Failure at a boundary blocks recipe finalization. It does not authorize an unregistered fallback or a launch with the best-looking partial recipe.

### Staged 1B successor qualification

The successor is one fresh trajectory configured for `1,000,000,000` sampled rows from row zero. The 100M canary is not a separate run. Reward, draft-credit, curriculum, policy-temperature, smoothing, entropy, and league schedules use absolute sampled rows over the full 1B horizon.

Write diagnostic packets at approximately 5M, 15M, and 30M, then stop at durable checkpoints for the formal gates:

| Added rows | Required decision |
|---:|---|
| 5M | Mechanics and opportunity discovery only; no strength promotion |
| 15M | Element coverage plus reward, draft, league, and reconstruction integrity |
| 30M | Strategy persistence after early reward/curriculum pressure begins to fall |
| 50M | Integrity, exact resume, learning health, strategy trajectory, and quick external check |
| 100M | Formal canary: strategy acquisition, strength conversion, forgetting, exposure diversity, and runtime |
| 200M | First post-canary continue/stop decision and schedule-transition check |
| 300M | Full robustness, approximate exploitability, and blind human review |
| 450M | Main persistence and production-funding gate |
| 800M | Late learning, forgetting, polish, and checkpoint-selection assessment |
| 1B | Final production qualification and late-window checkpoint selection |

Every formal gate from 50M onward must run the previous fixed-reference, historical, ancestor, both-seat, robustness, and integrity panels plus:

- sampled and deterministic element funnels;
- corrected Lightning and Water opportunity-to-conversion sequences;
- sibling deck and action differentiation;
- conditional face-versus-entity choices;
- reward-component magnitude and reconstruction;
- anneal persistence;
- opponent role and temporal exposure; and
- change against earlier gates and retained upper-envelope checkpoints.

The 100M canary advances only when all of the following hold:

- Lightning weapons and Water spells are drafted and converted when legal opportunities exist; neither deterministic family is structurally absent.
- Earth and Fire retain multiple valid lines rather than carrying only the pooled score.
- Sibling decks and actions differ where fixed-deck probes establish different values.
- Face attacks remain efficient when correct but do not dominate stronger setup/control alternatives.
- Strategy support persists after reward, draft, and curriculum anneals.
- The 50M-100M window improves against cheap-policy ancestors or has a statistically credible positive conversion slope while respecting the strength floor.
- The exposure log proves role and temporal diversity.
- Quality-adjusted SPS meets the frozen hardware floor and no integrity guard regresses.

Each passing gate releases only the next segment: 50M to 100M, then 200M, 300M, 450M, 800M, and 1B. Resume the exact model, optimizer, scheduler, RNG, league, promotion, environment-progression, and manifest state. Never restart a schedule, rewind episode progression, or mutate rewards, architecture, assignment, opponent semantics, or evaluation panels at a gate. Any recipe change requires a new row-zero candidate.

Final qualification must select from the strength/strategy Pareto frontier across the retained late window, not automatically select the endpoint. Required final outputs are a verified artifact manifest, health report, complete fixed and historical panels, both-seat head-to-head matrix, per-element/gate/leader worst slices, approximate-exploitability evidence, descriptor-v3 elemental reports with explicit attribution limits, blind human review, and a signed decision packet naming the selected checkpoint or rejecting the run.


## Historical control decision

Run a single-GPU, quality-gated continuation from the qualified p10730 exact-retained-credit parent before treating distributed scale as necessary for learning quality.

- Parent: `train-ablation-1781126582/results/production_launch_v1/parent_p10730/`
- Model: `parent_p10730/resume/model_azuki_local_010730.pt`
- Trainer state: `parent_p10730/resume/trainer_state_010730.pt`
- Config: `python/config/azuki_deckbuild_production_3090.ini`
- Current cumulative progress: 164,812,800 configured sampled rows at p10730.
- New schedule envelope: 1,000,000,000 additional configured sampled rows.
- Intended aligned endpoint: 1,164,825,600 configured sampled rows at update 75,835, adding 1,000,012,800 rows because the horizon is rounded up to a complete 15,360-row update.
- Initial operational commitment: 200M additional sampled rows, followed by a formal go/no-go evaluation.
- Main production decision gate: 450M additional sampled rows.
- Continue toward 800M/1B only while external competence or robustness continues to improve.

The run is a continuation, not training from random initialization. It preserves the qualified model, optimizer moments, league state, promotion state, episode progression, deck pool, and source/config fingerprints.

## Initial run outcome

The seeded, clean-log production attempt launched with:

- managed process: `local-prod-1b`;
- run ID: `local_production_1b_20260820T002335Z_178718716658`;
- compact log: `train-ablation-1781126582/results/local_production_1b_v1/20260820T002335Z/logs/production.jsonl`;
- checkpoint directory: `experiments/azuki_local_local_production_1b_20260820T002335Z_178718716658/`;
- configured endpoint: 1,164,825,600 sampled rows / update 75,835;
- process RNGs: Python, NumPy, and Torch explicitly seeded with 42;
- initial LR restart: 65,105 remaining updates at peak `3e-5`, cosine final LR zero.

The LR schedule itself was correct, but the long near-constant `3e-5` peak was not safe for the mature p10730 optimizer/value state. Value loss rose from `0.2389` at update 10,810 to `3.0046` at 10,820, `271.7` at 10,850, and `8.19e8` at 10,970. Policy loss, entropy, approximate KL, SPS, and environment-integrity metrics remained superficially normal, demonstrating why an explicit value-loss collapse monitor is load-bearing.

The trainer was stopped at the initial health gate. Do not resume a collapse-era checkpoint. Evidence is preserved in `results/local_production_1b_v1/20260820T002335Z/monitor/collapse_report.json`; the immutable p10730 parent remains the restart point after selecting and validating a safer continuation schedule.

The update-10,800 checkpoint passed artifact hashes and preceded the exponential phase, but p10730 is still the conservative restart source. Two earlier preproduction launch attempts were also discarded before material progress: one resolved `train.seed_process_rngs=false`; the next reused the append-mode compact-log path. Their logs and decisions are retained as `attempt1.json`, `attempt2.json`, and files under `logs/`.

### Fresh-run alternative also failed its canary

A fresh random-initialization 1B attempt used the modern regional deck pool, exact retained credit, reward curriculum, fresh dynamic league, Muon, and the historically successful fresh peak LR `0.003`. It was stable through update 40 (`value_loss <= 0.092`) but diverged immediately around the first checkpoint/league transition: `12.8@50`, `176@80`, `109k@100`, and `2.76M@110`. The trainer was stopped.

This rules out inherited p10730 weights or optimizer moments as the sole cause. No 1B run should restart until a short controlled isolation separates the shared LR/Muon path, regional deck-pool distribution, exact-credit/value target, and first frozen-opponent transition. The exact failure config is retained as `python/config/azuki_deckbuild_fresh_1b_3090.ini` with a do-not-launch header; evidence is under `results/local_production_fresh_1b_v1/20260820T013338Z/monitor/fresh_collapse_report.json`.

### Autoresearch revert audit

Merge PR47 (`3d49e14`) adds only `python/config/autoresearch_training_tuning.json` and is not loaded by either production recipe, so it cannot cause the critic failure.

PR46 (`f9d7c0e`) changed trainer inference/buffer behavior and policy metadata-table handling and was validated only on short canonical runs that ended before the failure window. A detached pre-PR46 canary was healthy through update 150, while current PR46 code diverged by update 110, so PR46 was reverted as `4f5119a`.

The revert is conservative but not sufficient proof or a complete fix: a post-revert canary with source hashes identical to the healthy detached canary reached `value_loss=7.63` and explained variance `-1.16` at update 110. The two canaries differ only in run paths yet produced different trajectories, so multiprocessing/trajectory nondeterminism or another shared training interaction prevents attributing the failure solely to PR46. Evidence: `results/autoresearch_regression_audit_20260820.json`.

The post-revert update-100 checkpoint was resumed with its optimizer and scheduler to test whether the update-110 spike would recover. It did not: value loss rose `0.20@110`, `0.31@120`, `1,373@130`, `54,535@140`, `5.59M@150`, and `5.44e10@170`. That checkpoint and every later checkpoint in the branch are unsafe. PR46 remains reverted, but the 1B blocker is now classified as a stochastic/shared critic-training instability rather than a verified autoresearch-only regression.

### Qualified runtime resolved the critic instability

Every divergent canary above used `python/.venv-codex` (Python 3.13, Torch 2.9.0, PufferLib 4.0). The historical production launchers and qualified soak use the repository root `.venv` (Python 3.14, Torch 2.11.0, PufferLib 3.0); both environments carry HeavyBall 2.2.1.

A fresh exact-credit 1B canary under the qualified root runtime remained healthy through and beyond update 300: value loss stayed approximately `0.03-0.05`, explained variance reached `0.73-0.75`, environment integrity remained zero, and full-league SPS settled around `1.4-1.7k` as the opponent pool grew. This run is now the active fresh 1B production run:

- managed process: `qualified-runtime-canary`;
- run ID: `qualified_runtime_exact_20260820_178719786506`;
- runtime: `.venv/bin/python`;
- log: `results/critic_isolation_20260820/qualified_runtime_exact/logs/production.jsonl`;
- checkpoints: `experiments/azuki_local_qualified_runtime_exact_20260820_178719786506/`;
- continuous guard: `qualified-prod-health`;
- milestone monitor: `qualified-prod-milestones`.

The operational root cause was a training-runtime mismatch, not saved p10730 weights. PR46 remains reverted conservatively, but the runtime comparison—not the revert alone—is what restored stable critic learning.

### Discord and milestone evaluation workflow

Both detached monitors read an optional webhook from:

`~/.config/azuki-tcg/discord-webhook-url`

The file must be mode `0600`; the URL is never stored in Git or printed. Once present, the health monitor posts one compact summary per hour plus immediate confirmed health violations, and the milestone monitor posts at 50M/100M/200M/300M/450M/800M/1B after verifying the latest checkpoint hashes.

Milestone notification does not automatically pause or run inference. The selected workflow is: Discord posts the checkpoint report, the operator replies in this training thread, and the run is then deliberately paused at a durable checkpoint for the appropriate quick or full evaluation. This avoids unattended SIGTERM/resume and prevents evaluation inference from competing with the single training GPU.

## Matched BPTT league Discord monitoring

`monitor_strategy_discovery.py` also recognizes `azuki.bptt_league_experiment`
registrations in `experiment.json`. It reports each arm's additional learner
steps from `common_parent.json`, not configured raw rows, and counts baseline,
midpoint, and final evaluation traces. Training and evaluation phases remain
nonterminal; a completion notice requires the terminal result and full registered
trace coverage. Partial metric records are deferred until their newline arrives.

The v3 campaign uses persistent watcher `bptt-league-discord-v3`, attached to
`bptt-league-campaign-v3` with process-identity checking. Its Discord attachment
was delivered. It polls every 60 seconds, posts progress every 30 minutes, and
sends stage, failure, process-exit, completion, and 30-minute inactivity alerts.
Delivery state is retained in
`results/bptt_league_15m_v3/discord_monitor_state.json`; failed posts retry.
The watcher only reads campaign artifacts and writes its own state. It does not
modify frozen training inputs, stop training, or start model sessions.

## Why run locally first

The qualified single-GPU path has known learning semantics. A local long-horizon run provides the first trustworthy modern curve beyond a 45M continuation without introducing distributed batch size, policy lag, gradient synchronization, or actor/learner staleness as confounders.

The local result is the semantic control for future distributed work:

- If the policy keeps improving, distributed compute buys faster iteration.
- If the policy cycles or plateaus, distributed compute alone would reach the same failure sooner.
- If the local and distributed trajectories differ, the distributed system must explain the difference before its SPS is treated as useful.

The GPU must be exclusive. A separate Git worktree isolates files, not CPU, RAM, disk, or GPU resources. While training, light design and editing are acceptable; GPU work, large builds, project-wide tests, and heavily parallel compilation should run elsewhere.

## Resume and learning-rate safety

### One-time child-run transition

The production config uses a strict full-state resume:

- `resume.load_optimizer = true`
- `resume.restart_lr_schedule = true`
- `resume.strict = true`
- `resume.auto_reset_critic = false`
- `train.learning_rate = 0.00003`

Loading the parent does not mutate weights. Updates begin only after the model, optimizer, global step, epoch, and league state have restored successfully.

The new child run restarts a cosine schedule once at `3e-5` over the remaining 1B-step envelope. This is conservative relative to prior evidence:

| Continuation | Peak LR | Outcome |
|---|---:|---|
| Invalid model-only continuation | `3e-3` | Entropy and external strength collapsed |
| Successful p2930 continuation | `3e-4` | Broad continued learning |
| Planned p10730 production continuation | `3e-5` | 10x below the successful continuation and 100x below the catastrophic restart |

The p10730 production soak exercised the `3e-5` restart for 300 updates without integrity or numerical failure. It does not prove long-horizon quality, which is why the 200M gate is required.

### Same-run recovery

`restart_lr_schedule` applies only when creating the new child run from p10730.

Every later recovery of that same run must restore the saved scheduler state without restarting the cosine. Repeated schedule restarts would repeatedly raise LR to `3e-5`, invalidate the planned decay, and make milestone comparisons uninterpretable.

Never use a model-only resume. A valid recovery requires the matching model and trainer-state pair plus the league, promotion, and configuration fingerprints.

### 1B cosine reference

Approximate remaining LR fraction under a 1B configured-sample cosine:

| Added sampled rows | LR fraction |
|---:|---:|
| 50M | 99.4% |
| 100M | 97.6% |
| 200M | 90.5% |
| 300M | 79.4% |
| 450M | 57.8% |
| 800M | 9.5% |
| 1B | 0% |

A 45M-sample checkpoint in this run is therefore an early-learning observation, not another zero-LR endpoint.

## Expected RTX 3090 duration

The configured horizon and logged SPS use different counters:

- historical `45M`/`1B` horizons are configured sampled rows (`updates * 15,360`);
- trainer `SPS` counts trainable learner-action rows;
- the production league trains approximately 60% of sampled rows because frozen-opponent rows are not learner updates.

Use the production-relevant 1.28-1.40k learner-action SPS envelope:

- exact-credit 45M median: 1,353.7 SPS;
- exact-credit tail-100 median: 1,337.7 SPS;
- production-soak conservative phase median: 1,277.4 SPS.

Training-only estimates for the configured sampled-row horizon:

| Added sampled rows | 1,277 SPS | 1,338 SPS | 1,400 SPS |
|---:|---:|---:|---:|
| 50M | 6.5 h | 6.2 h | 6.0 h |
| 100M | 13.1 h | 12.5 h | 11.9 h |
| 200M | 26.1 h | 24.9 h | 23.8 h |
| 300M | 39.2 h | 37.4 h | 35.7 h |
| 450M | 58.7 h | 56.1 h | 53.6 h |
| 800M | 104.4 h | 99.7 h | 95.2 h |
| 1B | 130.5 h | 124.6 h | 119.0 h |

Plan on 5.0-5.5 uninterrupted training days and 5.5-7 calendar days after evaluation, checkpoint, recovery, and workstation-contention overhead. A literal 1B trainable learner actions would require about 1.67B configured sampled rows and is not the horizon meant by the historical 45M terminology.

## Historical learning evidence

The modern evidence does not establish a 45M plateau.

The matched p7800-to-p10730 exact-credit continuation showed:

| Added steps | Exact vs standard | Parent-panel delta | Exact heldout score |
|---:|---:|---:|---:|
| 6.1M | 53.39% | +7.29 pp | 68.06% |
| 15.4M | 64.32% | +16.15 pp | 72.92% |
| 29.2M | 68.23% | +19.53 pp | 75.00% |
| 45.0M | 65.36% | +17.45 pp | 76.04% |

Direct strength softened after its p9700 peak, but heldout strength continued rising through p10730. Deck concentration and behavior were also still moving. The August production qualification later showed that raw Garden/leader-rate concerns were not opportunity-normalized use regressions and qualified exact retained credit for production scale with monitoring.

Other 45M results show why the endpoint cannot be privileged:

- S14 external windows moved `39.6 / 51.0 / 35.4 / 78.1 / 52.1 / 52.1 / 49.7%`.
- Its strongest checkpoint was near 30.7M samples, not the endpoint.
- Earlier league recipes sometimes peaked before 15M and then forgot old strategies.
- Specific mechanisms such as sibling-gate draft conditioning remained near zero despite more steps, demonstrating that scale does not repair every credit or representation problem.

The old nominal 200M/500M/1T records are not comparable modern evidence. The available registry shows incomplete jobs, older observation/reward/league stacks, and no current heldout evaluation protocol.

## Milestones and release gates

The run keeps one uninterrupted 1B scheduler. Milestones release more compute; they do not restart or retune the run.

| Added sampled rows | Expected local time | Decision |
|---:|---:|---|
| 50M | 6-7 h | Integrity, resume, learning-health, and quick external check |
| 100M | 12-13 h | First fixed-panel competence gate |
| 200M | 24-27 h | First formal continue/stop decision |
| 300M | 36-40 h | Full robustness, approximate-exploitability, and blind human-review gate |
| 450M | 2.2-2.5 d | Main local-versus-distributed funding decision |
| 800M | 4.0-4.4 d | Late learning and polish assessment |
| 1B | 5.0-5.5 training days | Final late-window checkpoint selection |

Do not declare a plateau before 100M absent integrity failure or clear strategic collapse. The main question at 200M is whether the qualified modern policy continues learning under a schedule that has not already annealed LR to zero.

## Evaluation hierarchy

### External competence

Primary evidence comes from frozen opponents not selected in response to the current checkpoint:

- production anchor;
- historical policy panel;
- regional/top-player deck panel;
- heldout deck signatures;
- both seats, start-player conditions, gates, and leaders.

Report pooled score, paired delta, confidence interval, worst opponent/deck/gate/seat slice, and slope across checkpoint windows.

The existing 384-game suites detect roughly five-point changes near 50%. Major 100M/200M/450M decisions intended to resolve a two-point change should use about 2,500 aggregate paired games per primary panel, subject to variance reduction from the fixed paired schedule.

### Robustness and Nash-style evidence

A single self-play win rate does not establish a Nash equilibrium. At major milestones:

1. Build a payoff matrix against the parent, earlier run checkpoints, old archive policies, strategically distinct branches, and external deck panels.
2. Track the candidate's worst-case score and maximin/Nash-mixture value over that population.
3. Train multiple approximate best-response exploiters against the frozen candidate when infrastructure permits.
4. Calibrate exploiters by verifying that they can exploit known weak historical checkpoints.

A randomized mixture of several qualified, strategically distinct checkpoints may be harder to exploit and closer to a practical mixed equilibrium than a single deterministic endpoint.

### Human-facing competence

The product goal is an impressive opponent for top players. Final qualification must include blind, seat/gate/start-balanced human series and qualitative failure tags such as missed lethal, poor response timing, incoherent drafting, and repeatedly exploitable habits.

Fixed-deck neural evaluation is necessary but does not prove top-human strength.

### 300M blind human review

The 300M gate adds a blind human review to the fixed neural panels. It is diagnostic: do not retune, rewind, or mutate the active production run from review observations. Export from a durable checkpoint, keep training and evaluation artifacts immutable, and use the findings to choose post-run experiments.

Build a seat/gate/start-balanced packet spanning p100, p200, and p300 on matched seeds where possible. Prioritize both Water and Lightning gates, the hardest heldout decks, the Task2 hardest-retained policy, wins, losses, close games, and games where spells or weapons were drafted. Include greedy games and stochastic-policy games only when the existing replay surface can produce both without changing training semantics.

Reviewers must not see checkpoint identity or game outcome while annotating. Record:

- spells and weapons offered, selected, drawn, legal to use, played, delayed, or ignored;
- final deck counts by card type and IKZ cost, archetype coherence, and four-copy concentration;
- missed lethal, response timing, IKZ sequencing, unspent IKZ, dead cards, portal timing, and better legal alternatives;
- whether an off-meta line is coherent, opponent-specific, repeatedly exploitable, or an apparent reward-shaping artifact;
- whether the observed issue is absent opportunity, a correct refusal, or a missed legal opportunity.

Opportunity-normalized use is the required statistic for spells, weapons, leader/Garden abilities, portals, and responses. Raw episode-action percentages alone cannot distinguish an unavailable action from a strategic refusal.

The Next.js client needs a review mode before this gate can be completed. It should consume frozen replay artifacts without accessing the live trainer or GPU, hide checkpoint/result identity, render draft and battle state with the chosen action and available legal alternatives, support the labels above, and export versioned JSON keyed by game ID and reviewer. Filters must cover element, gate, leader, seat, starting player, opponent, result after unblinding, and whether the relevant card was offered, drafted, drawn, or legal.

Interpret the review as follows:

- coherent off-meta play with correct opportunity handling supports continuing the current recipe;
- strong fixed-panel results plus a repeatedly human-exploitable pattern justify a matched league-construction child experiment;
- systematic premature tempo, portal fixation, resource misuse, or refusal to hold value justify a matched reward-shaping child experiment;
- evaluator-only artifacts require fixing the review/evaluation surface, not changing training.

Do not extend the current shaping anneal at 300M. The episode schedule has already reached its final multiplier, so changing it then would reintroduce shaping rather than test a longer anneal. A valid comparison must start fresh or from an early checkpoint whose saved environment progression precedes the schedules being compared.

### Strategic guardrails

Track:

- stochastic and greedy unique-card counts;
- four-copy concentration;
- Water spell slots;
- opportunity-normalized leader, Garden, spell, portal, and response use;
- element/gate/leader slices;
- game length, timeout, truncation, invalid, and incomplete outcomes;
- card/archetype coverage and payoff-vector diversity.

These explain what changed and catch collapse. They do not replace external win rate.

### Optimizer health

Track entropy, approximate KL, clip fraction, explained variance, policy/value losses, draft-credit loss and gradient norm, policy lag, sample reuse, and parameter/optimizer health. These detect broken training but cannot establish competence. Prior strategic regression occurred without a corresponding entropy, KL, clipping, value-loss, LR, SPS, or parameter-norm anomaly.

## Continue, plateau, and stop rules

Continue when at least one primary axis improves with no material guardrail regression:

- heldout external score;
- worst-case archive score;
- approximate exploitability;
- strategically distinct qualified archive admission.

Call a practical plateau only after three consecutive major gates, separated by 25-50M steps, show no external improvement, no worst-case improvement, no exploitability reduction, and no new strategically distinct policy, with uncertainty ruling out more than about one to two points per 100M.

Stop immediately for NaN/Inf, invalid/incomplete/truncated outcomes, checkpoint/accounting failure, entropy collapse, persistent external regression, worst-case opponent/deck/gate collapse, or a growing trained best response.

Select an earlier checkpoint when the late-window mean is more than three points below the preceding window, external and historical robustness decline at two consecutive gates, or the upper envelope stops improving despite continued meta cycling.

## League behavior

### Dynamic ordinary league

The ordinary PPO league is dynamic even though individual opponents are frozen.

- Trainer checkpoints are saved every 50 updates.
- `league.checkpoint_add_interval = 1` admits each eligible saved checkpoint.
- The effective admission cadence is `50 * 15,360 = 768,000` agent steps.
- At about 1.3k SPS, a new immutable checkpoint enters roughly every ten minutes.
- New assignments use the refreshed pool; in-flight games finish against their existing opponent.

### Retention

The active pool keeps approximately:

- 6 recent policies;
- 4 middle-aged policies spread across history;
- 3 oldest policies.

Over 1B steps, approximately 1,302 learner checkpoints could be created while only about 13 remain active. This creates a risk that a strong intermediate peak is retained for evaluation but ceases to exert training pressure.

### Matchup mix and PFSP

`frozen_ratio = 0.40` is a row ratio. In the two-player environment it becomes approximately 80% historical-policy games and 20% latest self-play.

PFSP weights historical opponents by difficulty with an exploration floor. The optimized rollout chooses at most one frozen identity for each eight-update window, then phases it in as games reset. This keeps GPU forward cost independent of pool size but exposes the learner to historical styles sequentially.

### Frozen evaluation archive

The ordinary pool is dynamic, but promotion is initially evaluation-only:

- `promotion_shadow_mode = true`;
- `promotion_archive_affects_training_pool = false`;
- the production anchor is excluded from the training pool;
- the promotion panel is held fixed.

This freezes the yardstick without freezing the sparring partners. Promotion decisions cannot silently alter the training distribution during the initial run.

### Opponent diversity

Opponent diversity is broader than checkpoint count:

- temporal diversity: old, middle, recent, and current policies;
- strategic diversity: distinct decks, gates, play styles, and payoff vectors;
- difficulty diversity: policies the learner barely beats or still loses to;
- adversarial diversity: targeted best-response exploiters;
- external diversity: stationary anchors and top-player decks.

It decomposes into generation, admission, retention, selection, and scheduling. PFSP optimizes selection, but cannot manufacture strategically different policies if every retained checkpoint implements the same style.

Keep the current league unchanged through at least the 200M gate. If external pooled strength is flat while old-opponent performance falls, fork a matched league-diversity child run rather than mutating the active production run. Candidate interventions include protecting qualified payoff-distinct policies, retaining by payoff-vector diversity rather than age alone, interleaving more than one frozen identity per window, or adding calibrated exploiters.

## Monitoring and recovery

Use the production JSONL as the training source of truth and external system monitoring for GPU health.

Immediate alerts:

- non-finite metrics;
- invalid, incomplete, timeout, or truncation events above the configured ceiling;
- checkpoint parity, manifest, or remote-copy failure;
- global-step discontinuity;
- GPU Xid or uncorrectable ECC;
- persistent SPS below the qualified baseline;
- worker or opponent-pool refresh failure.

Retain model, optimizer/trainer state, league state, promotion state, config fingerprint, source hashes, and evaluation results. Copy recovery, evaluation, and milestone bundles off the compute host.

The current 100/250/1,000-update artifact tiers correspond at about 1,277 SPS to roughly 20 minutes, 50 minutes, and 3 hours 20 minutes. Dense evaluation should be queued to a separate GPU where possible rather than repeatedly interrupting the only training GPU.

## Prelaunch requirements

Before the local run begins:

1. The RTX 3090 is no longer used by autoresearch or any other GPU process.
2. The p10730 parent manifest and every required model/trainer/league/promotion artifact verify by hash.
3. The 1B additional-step cumulative endpoint and one-time restart semantics are frozen in a dedicated launch record.
4. Recovery after the initial launch is documented to restore, not restart, the scheduler.
5. Evaluation checkpoints and remote artifact storage have sufficient capacity.
6. The 50M, 100M, 200M, 300M, and 450M evaluation schedules are frozen before training.
7. No other worktree or process will run GPU work or sustained CPU/disk-heavy commands on the host.

## Future distributed run

### First topology

Use one physical four-GPU host before considering multiple networked nodes:

```text
Durable control plane and object storage
                  |
       One four-GPU physical host
       |-- GPU 0: learner rank + local C actors
       |-- GPU 1: learner rank + local C actors
       |-- GPU 2: learner rank + local C actors
       `-- GPU 3: learner rank + local C actors
```

Target 10-12 physical CPU cores and 64-128 GB RAM per GPU, local NVMe, and durable off-host checkpoints. L40/L40S is the preferred price/performance tier; A100 is justified when communication or reliability requires it. H100 is not required by the current model.

The current production entrypoint is not a validated distributed trainer. Generic DDP and Ray-related components do not establish working multi-node production semantics. Distributed qualification must preserve or deliberately revalidate effective global batch, optimizer-step ratio, sample reuse, recurrent state, policy lag, league behavior, checkpoint consistency, and learning quality.

### SPS gates

| Topology | Minimum useful | Goal |
|---|---:|---:|
| One GPU | 1.3k | 1.5-1.7k |
| Four GPUs | 3.0k | 3.6-4.0k |
| Eight GPUs | 5.0k | at least 6.0k |

Quality-adjusted SPS matters more than raw SPS. Asynchronous actor production must remain close to learner consumption; extra actors that create stale samples can reduce competence despite increasing reported throughput.

### Multi-node network floor

The packed observation is 9,416 bytes per agent row. Raw inbound observation traffic alone is about 301 Mbps at 4k SPS and 452 Mbps at 6k SPS, excluding actions, trajectories, framing, checkpoints, and gradient synchronization. A 100 Mbps private network is unsuitable for remote inference at target throughput. Multi-node work should use at least 25-100 Gbps networking, preferably with an appropriate NCCL/RDMA fabric when gradients cross hosts.

### Provider planning rates

Published planning rates previously reviewed:

| Provider shape | Approximate compute rate | Use |
|---|---:|---|
| Hyperstack L40 | $1.00/GPU-h | Initial one-to-four-GPU qualification candidate |
| Runpod Secure L40S | $0.99/GPU-h | Low-friction single-Pod pilot; avoid 100 Mbps cross-Pod training traffic |
| Vultr 8x L40S bare metal | starts at $6.784/h total | Explicit eight-GPU whole host if eight GPUs become justified |
| Lambda 8x A100-40 | $15.92/h total | Stronger interconnect/reliability option at higher cost |

At 4k learner-action SPS and about $4/h, 1B configured sampled rows at a 60% trainable fraction is approximately 41.7 training hours and $167 compute-only. At 6k learner-action SPS on the published Vultr starting rate, it is approximately 27.8 hours and $188. Add evaluation, storage, control-plane, capacity, and contingency overhead.

### Distributed release plan

1. Match the single-GPU production configuration and parent exactly.
2. Prove checkpoint/restart and failure handling at target topology.
3. Run a short semantic parity canary.
4. Measure 1/2/4-GPU scaling and policy lag.
5. Compare distributed learning against the local control at matched agent steps, not wall time.
6. Use four GPUs for independent ablations if one-run scaling is below 3k SPS or changes quality.
7. Move to eight GPUs only after four GPUs sustain 3.6-4.0k quality-preserving SPS and the remaining horizon justifies it.

### Post-local-run league and shaping experiments

Defer league-construction and reward-shaping schedule changes until the current 1B control finishes. Run the resulting matched experiments on a qualified four-GPU distributed host, with an operational target of completing a 1B-step arm in about 2.5 days including normal checkpoint and evaluation overhead. Distributed semantic parity remains a prerequisite; faster wall time does not excuse changes to effective batch, optimizer-step ratio, recurrent replay, policy lag, league state, or checkpoint semantics.

Use the 300M human review and final local-run evidence to select bounded arms:

- league arms may protect payoff-distinct policies, retain by payoff-vector diversity, interleave multiple frozen identities, or admit a validated quality archive into training;
- shaping arms may lengthen the warmup/ramp, change the final multiplier, or separate draft and battle shaping schedules;
- the control must preserve the current recipe, and each treatment should change one decision boundary at a time.

Match seeds, configured agent steps, evaluation schedules, external panels, human-review packet construction, and artifact requirements. Never rewind episode progression on a mature checkpoint to simulate a longer anneal. Compare learning slopes and late-window robustness through 1B steps rather than selecting an arm from an early transient.

#### League regression and recovery ablations

The 450M result makes late-policy retention a first-class future experiment. p21000 improved direct ancestor strength and archive worst-case robustness, while the uninterrupted universal policy later specialized sharply and lost most Lightning, Water, and Fire matchups without an optimizer or integrity failure. The dynamic league is a plausible contributor, but this result does not identify which generation, admission, retention, selection, or scheduling mechanism caused the regression.

Do not modify or resume the paused production run to test these ideas. Preserve p21000 and p29300 as immutable evidence. Run every treatment as a separate matched child with the same model architecture, reward schedule, element/gate/leader assignment, optimizer, configured rows, seeds, evaluation schedule, and qualified runtime. Keep element isolation and league changes in different experiments.

Use a universal-policy continuation from the full p21000 state as the control. Candidate league treatments, one decision boundary per arm:

- **Qualified-checkpoint protection:** permanently retain p18000 and p21000, plus later checkpoints that pass the fixed external, ancestor, and worst-gate floors. Protected policies must remain eligible opponents rather than becoming evaluation-only artifacts.
- **Payoff-diverse retention:** replace some age-based middle/recent slots with checkpoints selected for distinct payoff vectors across elements, gates, leaders, seats, and starting-player conditions.
- **Gate-balanced rehearsal:** reserve a fixed opponent-assignment quota for every opponent element and gate so that PFSP difficulty cannot starve matchups the learner currently wins or has temporarily forgotten.
- **Multiple-opponent interleaving:** expose each rollout window to several frozen identities instead of phasing in at most one identity for an eight-update interval. Measure policy lag and throughput before treating this as viable.
- **Regression-aware admission:** require candidate checkpoints to clear pooled, worst-opponent, worst-gate, and qualified-ancestor floors before they can displace protected training opponents. Admission should use confidence-aware paired results, not one noisy win-rate estimate.
- **Mixture-targeted sampling:** sample from a fixed or solved maximin/Nash-style mixture over the qualified archive, with an exploration floor, rather than allowing the current PFSP score alone to define the training distribution.
- **External rehearsal slice:** allocate a small stationary share to the production anchor, hardest retained policy, and fixed reference-deck contexts. This changes the training distribution and must be tested separately from ordinary archive protection.

Also test operational regression controls independently of league learning:

- run the quick fixed-reference and ancestor panels on durable checkpoint windows during training;
- retain an immutable upper-envelope checkpoint even when training continues;
- trigger a shadow alert when the late-window mean drops more than three points, any gate drops below its floor, or a qualified ancestor dominates the current policy;
- pause for operator review before a regressed checkpoint changes the protected archive or becomes a deployment candidate.

Two recovery questions should not be conflated:

1. **Prevention:** can a p21000 child preserve broad competence while continuing to learn?
2. **Reversibility:** can a diagnostic p29300 child recover forgotten gates when exposed to a corrected opponent mixture?

The p29300 recovery arm is diagnostic only. It must not become the control or deployment candidate merely because it recovers some score. Compare it against the p21000-parent treatments at matched additional rows.

Primary readouts are paired fixed-reference score, direct score against p18000/p21000, historical-panel worst case, per-element and per-gate payoff floors, checkpoint-window slope, and archive maximin value. Strategic guardrails remain opportunity-normalized action use, deck concentration/diversity, game length, and incomplete/timeout/truncation rates. A treatment qualifies only if it prevents broad late regression without hiding a failed gate behind pooled improvement or materially reducing quality-adjusted SPS.

#### Human-evaluation interpretation and final 1B selection plan

Two blind human reviews now add a decision axis that the saturated fixed-reference panels do not fully resolve:

- earlier session `01a035b7-4bac-7893-bff0-9dd9ffe28256`: 32 annotated matches, 31 completed, spanning p6500, p13000, p18000, and p19500;
- later session `01a03a03-80cd-70c3-8731-0bebfe7abfd9`: 48 completed matches, 16 each for the p18000 historical anchor, p21000 upper envelope, and p29300 regression diagnostic.

The later session is the cleaner comparison because it contains the unchanged p18000 anchor in the same reviewer session. p21000 beat the human in `5/16` games versus `1/16` for p18000 and improved all five mean ratings: strength `3.625 vs 3.438`, decision quality `3.750 vs 3.313`, deck coherence `3.125 vs 2.813`, human likeness `3.063 vs 2.750`, and enjoyment `3.625 vs 3.500`. The 16-game outcome difference remains directional rather than conclusive (`Fisher exact two-sided p=0.172`). Raw ratings must not be compared across sessions without the shared anchor: p18000 itself scored lower in the second review even though its aggregate draft profile was effectively unchanged.

p21000 is the current high-floor policy. Its logs show a better mulligan curve, correctly placing Pip in the Alley, repeated low-health Healing Flutter use, efficient board development, and more attacks per game. Its deck is nevertheless a concentrated neutral tempo shell: approximately `13.56` unique main cards, `30.25` commons, `13.69` Beanz, `45/50` slots in four-copy sets, no weapons, and only `15.69/50` matching-element cards. Fire and Water produced most of the human-facing improvement. This policy wins more often, but its direct face-pressure plan is predictable and may have a hard ceiling against stronger humans.

p29300 is the current low-floor/high-ceiling diagnostic. Its pooled human results regressed (`3/16` wins; strength and decision-quality means both `2.938`), consistent with its `70.25%` formal holdout and `20.78%` direct score against p21000. It also broadened behavior sharply: approximately `21.19` unique main cards, `3.56` weapons, `7.00` spells, only `27/50` slots in four-copy sets, more activated abilities, and more defender declarations. Much of this behavior was poorly timed, especially Mocking Dummy targets, nonlethal effects followed by a pass, and losing entity trades. Frequency of a mechanic is not evidence that the mechanic is being used well.

The important exception is later-session match ordinal 8, the strongest human-like multi-turn plan observed so far. The p29300 Earth policy:

1. declared four defenders to stall leader damage;
2. developed two `STT03-013` Stone Masked Ancients in the Alley;
3. maintained the defensive board long enough to preserve those delayed threats;
4. portaled one Ancient into the Garden; and
5. converted that setup into the lethal attack.

This line is strategically more scalable than p21000's generic tempo plan even though p29300 is much less consistent overall. Earth was also the only p29300 element that retained a concentrated deck (`13` unique cards, `48/50` four-copy-set slots, no weapons or spells), and it was p29300's best-rated human element. Lightning, Water, and Fire underwent most of the unstable diversification. The next target is therefore not merely a higher-win-rate p21000 descendant: it is a checkpoint that preserves p29300's defender/delayed-threat planning while recovering p21000's execution floor.

##### Why p29300 became `active: false`, `bucket: "pruned"`

This was automatic ordinary-league retention, not a manual rejection or a configured rule naming p29300. The active recipe admits every eligible saved checkpoint (`league.checkpoint_add_interval = 1`). `LeagueManager._prune()` then calls the age-based `classify_and_prune()` policy, whose current defaults retain 6 recent, 4 evenly spaced middle, and 3 oldest entries. As newer checkpoints entered, p29300 fell outside those approximately 13 active slots and was automatically relabeled `pruned`.

The frozen promotion archive did not protect it because the control deliberately sets:

- `promotion_shadow_mode = true`;
- `promotion_archive_affects_training_pool = false`.

The current learner still descends from p29300's weights, but it is not explicitly rehearsing against the frozen p29300 policy. Continued PPO may refine, forget, or cyclically rediscover the Stone Mask line. Do not mutate the current control run to change this. If final-ladder evidence shows that the line disappeared, a future matched league arm should protect p29300 or a later payoff-distinct strategic checkpoint as an eligible opponent rather than relying on implicit weight inheritance.

##### Final 1B ladder workflow

At the time of this note, the continuation was healthy at p40250 / `618,240,000` added sampled rows, with a configured end at p65105 and zero integrity maxima. Let the control finish unchanged unless an existing health stop rule fires. The rolling entropy decline from the earlier `~0.082` baseline to `~0.056` makes post-run ladder evaluation important: it may represent useful consolidation or loss of strategic diversity, but optimizer telemetry alone cannot distinguish the two.

Evaluate the saved window rather than only the endpoint:

1. Run the lightweight paired panel on every 250-update candidate from p29300 through the endpoint.
2. Run the full mechanics suite every 1,000 updates and at local strength peaks, per-element recoveries, and material deck/action-profile transitions.
3. Include direct both-seat matchups against p21000 and p29300, the frozen external and historical panels, adjacent checkpoints, and per-element/per-gate reporting.
4. Apply strength and integrity floors before style selection. A checkpoint does not qualify merely because it uses defenders or weapons frequently.
5. Select from the strength/strategy Pareto frontier rather than ranking only pooled win rate.

Add outcome-sensitive strategic measurements to the post-run reports:

- defender declarations per legal response opportunity;
- leader damage prevented, attacker value traded, and turns survived after the first defender;
- win conversion after one or multiple defender declarations;
- Stone Masked Ancient copies drafted, drawn, played to the Alley, portaled, stranded, and converted into damage;
- occurrence and outcome of the sequence `defender stall -> two Ancients developed -> Ancient portaled -> Ancient attacks`;
- healing at full, damaged, and critical leader health;
- mulliganed-card cost relative to opening-hand cost;
- Pip Alley-placement rate;
- nonlethal effect followed by no-op;
- entity-clear trade quality;
- Lightning weapon attachment and leader-attack conversion;
- ability targets that materially change the next combat outcome.

The first blind human screen after the ladder should contain four models at eight games each:

1. p21000 as the known efficiency/high-floor anchor;
2. p29300 as the known strategic-origin/low-floor anchor;
3. the strongest qualified late checkpoint; and
4. the strongest strategy-preserving late checkpoint that still clears the strength and worst-element floors.

If entries 3 and 4 are the same checkpoint, use an adjacent local peak or the final checkpoint as the fourth diagnostic. If one late checkpoint appears to combine both styles, confirm it with 16 games each against p21000 and p29300. Balance element, gate, seat, and starting player; keep identities hidden; and interpret raw wins together with timing, conversion, and repeated exploitability. The desired outcome is p21000's reliability plus the p29300 Earth line's multi-turn planning, not either behavior in isolation.

##### 800M validation outcome

The 800M milestone was observed at `800,102,400` added sampled rows and bound to the durable p52000 evaluation/milestone/recovery checkpoint. Training was deliberately paused after the next durable p52750 checkpoint (`810,393,600` added rows) so evaluation did not compete for the RTX 3090. Both p52000 and p52750 five-artifact manifests, artifact sizes, SHA-256 hashes, and config fingerprints verified. The last valid run-specific health window remained healthy: median SPS `1,382.50`, value loss `0.02850`, explained variance `0.69721`, approximate KL `0.000342`, expected LR parity, and zero timeout, auto-tick, zero-legal-action, incomplete-draft, and truncated-draft maxima.

Validation reused the qualified root runtime and the 300M/450M native-paired contracts:

- 32 saved checkpoints from p30000 through p52750 received the 480-game, two-seed fixed-reference screen; the existing matched p29300 result was retained as the starting diagnostic;
- p44000, p51500, and p52000 received the 2,400-game, ten-seed formal holdout;
- p44000 and p52000 each received 640-game both-seat panels against p21000, p29300, the production anchor, Task2 hardest-retained, S13 distinct, and S14 recent-quality policies;
- p52000 received a direct 640-game comparison against p44000; and
- p21000, p29300, p44000, and p52000 each received 200 legal-action-logged, uniform-assignment sample-mode self-play games plus opportunity-normalized, card-funnel, and ordered defender/Stone Mask analysis.

Decision: **pass and continue the unchanged control to 1B**, restoring the complete p52750 state without restarting the scheduler. Preserve p44000 as the strongest qualified late checkpoint and p52000 as the strategy-preserving milestone diagnostic. Do not select p52750 merely because it is the pause endpoint.

The control resumed from the verified p52750 model/trainer/league/promotion set as run `corrected_production_1b_lr1500_fresh_resume_p52750_178783658456`, using a new compact log and no LR-schedule restart. At p52900 the resumed health guard reported median SPS `1,416.69`, value loss `0.02871`, explained variance `0.69454`, approximate KL `0.000396`, expected LR parity, and zero integrity maxima. The dedicated health and final-milestone monitors are active.

The checkpoint ladder shows rapid recovery but no new fixed-reference upper envelope:

- p29300 quick holdout: `68.75%`, worst gate `41.67%`;
- p30000 quick holdout: `94.38%`, showing recovery within 700 updates / 10.75M configured rows;
- p30000-p52750 mean: `95.31%`, with an effectively flat `+0.016 pp / 1,000 updates` fitted slope;
- p50000-p52750 mean: `95.35%`;
- p44000 quick upper envelope: `97.08%`;
- p52000 exact milestone: `95.83%`;
- p52750 pause endpoint: `94.38%`; and
- p21000 retained quick anchor: `97.92%`.

Formal and direct panels separate fixed-reference saturation from policy improvement:

| Checkpoint | Formal holdout | Formal worst gate | Direct vs p21000 | Direct vs p29300 | Historical pooled | Historical worst |
|---|---:|---:|---:|---:|---:|---:|
| p21000 | 96.50% | 94.33% | — | — | 98.87% | 97.97% |
| p44000 | 95.83% | 88.67% | **60.94%** | 88.28% | 98.75% | 97.81% |
| p52000 | 95.29% | 90.33% | **56.72%** | 85.78% | 98.55% | 97.19% |

Against the matched formal fixed-reference schedule, p44000 trailed p21000 by `0.67 pp` (`54` paired improvements, `70` regressions, `2,276` ties, exact two-sided `p=0.178`). p52000 trailed by `1.21 pp` (`52` improvements, `81` regressions, `2,267` ties, `p=0.0149`). This is non-transitive learning rather than a simple monotonic strength curve: both later policies beat p21000 directly while p21000 remains slightly better on the saturated fixed references.

p44000 is the stronger late policy. It beat p52000 `54.53%` overall, remained at least even with p21000 when pooled by element, and was historically robust. Its direct p21000 scores were Lightning `56.25%`, Water `50.00%`, Earth `71.88%`, and Fire `65.63%`. The individual Hydromancy gate remained a counter-matchup at `32.50%`, and the candidate-going-second score was only `46.25%`; pooled selection must retain these caveats.

p52000 is the better strategy diagnostic, not the deployment-strength selection. It beat p21000 in Lightning `56.25%`, Water `53.13%`, and Earth `70.00%`, but Fire pooled at `47.50%`. Against p44000 it was strongest on Stonehaven Earth (`66.25%`) and Hydromancy Water (`67.50%`) while losing the other six gates. Its deterministic deck broadened modestly (`14.31` unique cards, `88%` four-copy-slot share) relative to p44000 (`13.00`, `96%`) without returning to p29300's unstable `21.19` / `54%` profile.

The ordered strategy diagnostic required the same player to draft at least two `STT03-013` Stone Masked Ancients, declare a defender before the portal, play two Ancients to the Alley, portal one, and then attack with it. Across 400 player-games per checkpoint:

| Checkpoint | Ordered lines | Wins after line | Defender selection / legal opportunity | Garden ability selection / legal opportunity |
|---|---:|---:|---:|---:|
| p21000 | 3 | 2 | 98.28% | 44.19% |
| p29300 | **4** | **3** | 67.82% | 65.11% |
| p44000 | 2 | 0 | 74.15% | **77.78%** |
| p52000 | 3 | 2 | 77.01% | 64.00% |

This supports the prior human interpretation with an important qualification: the defender/Stone line exists in multiple checkpoints under sample-mode self-play, but p29300 still produces it most often and p44000 failed to convert either observed line. p52000 recovered two winning conversions while keeping broad external strength. The line remains rare, so human review is still needed to judge timing and repeated exploitability; raw mechanic-use frequency is insufficient.

The provisional four-model blind roster is frozen at `train-ablation-1781126582/human-eval-models-800m.json`: p21000 efficiency anchor, p29300 strategic-origin anchor, p44000 strongest qualified late checkpoint, and p52000 strategy-preserving diagnostic. Human review is not required to resume at 800M. Retain this roster for the final 1B decision unless the final ladder produces a later checkpoint that dominates either late role.

Primary machine-readable evidence:

- `results/corrected_production_1b_lr1500_fresh/gates/gate_800m_preflight.json`;
- `results/corrected_production_1b_lr1500_fresh/gates/gate_800m_summary.json`;
- `results/corrected_production_1b_lr1500_fresh/gates/strategy_sequences_800m.json`;
- `results/corrected_production_1b_lr1500_fresh/gates/holdout_800m_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/h2h_800m_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/mechanics_800m_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/opportunity_800m_*`; and
- `results/corrected_production_1b_lr1500_fresh/gates/card_funnels_800m_*`.


##### Final 1B automated validation outcome

The unchanged control completed at p65105 with `1,000,012,800` added sampled rows. Final health was healthy: median SPS `1,428.77`, value loss `0.02910`, explained variance `0.68248`, entropy `0.05960`, expected and observed learning rate `0`, and zero timeout, auto-tick, zero-legal-action, incomplete-draft, and truncated-draft maxima. The final five-artifact manifest, sizes, hashes, and config fingerprint verified; the model SHA-256 is `c38640c04a1b11f337d900d7f551c4df0e3df6e754c997ba3c1d12f61245d026`.

The final automated suite covered:

- all 26 retained p53000-p65105 checkpoints with the matched 480-game fixed-reference screen (`12,480` games);
- p56000, p60000, and p65105 with the 2,400-game formal holdout (`7,200` games);
- 22 both-seat 640-game direct panels (`14,080` games), including p21000, p29300, p44000, p52000, the four historical policies, and late-checkpoint comparisons;
- every 1,000-update checkpoint from p53000 through p65000 plus p65105 with 200 legal-action-logged sample-mode games (`2,800` games / `5,600` player-games);
- opportunity-normalized, card-funnel, and ordered defender/Stone analysis for all 14 mechanics checkpoints; and
- native gate/leader context replay, 16-context greedy plus stochastic deck dumps, and legacy gate-swap replay for p56000, p60000, and p65105.

The fixed-reference window remained saturated rather than improving materially. Its mean was `95.657%`, its fitted slope was only `+0.0418 pp / 1,000 updates`, p60000 and p62000 shared the `96.458%` quick maximum, and p65105 scored `96.042%`. Formal and direct panels selected p60000 as the best final-window checkpoint but did not displace p44000:

| Checkpoint | Formal | Formal worst gate | Direct vs p21000 | Direct vs p29300 | Direct vs p44000 | Direct vs p52000 | Historical pooled |
|---|---:|---:|---:|---:|---:|---:|---:|
| p56000 | 94.667% | 89.333% | 52.969% | 87.813% | 45.000% | 47.813% | — |
| p60000 | **95.417%** | **92.000%** | **53.750%** | **89.531%** | **46.250%** | **50.781%** | 98.086% |
| p65105 | 94.750% | 90.667% | 53.281% | 89.531% | 43.125% | 48.125% | 98.086% |

Against p21000's identical formal schedule, p56000 trailed by `1.83 pp` (`58` improvements, `102` regressions, exact two-sided `p=0.000629`), p60000 by `1.08 pp` (`56` / `82`, `p=0.03295`), and p65105 by `1.75 pp` (`57` / `99`, `p=0.000966`). p60000 beat p65105 directly `51.094%`. All three late candidates beat p21000 overall while retaining severe opponent- and starting-player-dependent slices; none matched p44000's `60.938%` direct score against p21000 or beat p44000 head to head.

The mechanics ladder produced one signal worth qualifying: p56000 had six ordered defender/Stone lines with four wins, versus p29300's prior `4/3`, p52000's `3/2`, and p44000's `2/0`. Formal and direct qualification rejected it as a replacement: p56000 lost to p44000, p52000, and p60000, and its formal Earth score was only `92.167%`. The late line then declined to `3/2` at p60000 and `2/1` at p65105.

No broad element-strategy recovery accompanied that Earth-only spike:

- all 12 deterministic Lightning contexts across the three probed checkpoints used the same 46-entity / 4-Lightning-Orb deck, with zero weapons and identical sibling-gate decks;
- all 12 deterministic Water contexts used 50 entities and zero spells, no Water spell reached the 200-game card funnels, and Echoed Waves produced zero interactive recovery offers;
- Rushfire engaged every interactive follow-up offer, but Zero was absent from the p60000 and p65105 traces and appeared once without use at p56000; and
- gate sensitivity was overwhelmingly Fire-driven. Aggregate gate-conditioned symmetric KL was `1.1624`, `1.0785`, and `1.2361` at p56000/p60000/p65105, while Fire alone was `4.5953`, `4.2264`, and `4.7687`; Lightning remained approximately `0.0013`.

Decision: **the training scheme flunked the strategy-learning objective**. Preserve p44000 as the strongest qualified checkpoint, p60000 as the best final-window checkpoint, and p56000 only as a narrow late Earth diagnostic. Do not select p65105 merely because it is the endpoint. No final human evaluation is warranted: the only unusual signal was the rare p56000 Earth sequence spike, and the automated qualification showed weaker strength plus continued Lightning/Water structural collapse rather than a new Pareto candidate.

Primary machine-readable evidence:

- `results/corrected_production_1b_lr1500_fresh/gates/gate_1b_preflight.json`;
- `results/corrected_production_1b_lr1500_fresh/gates/gate_1b_summary.json`;
- `results/corrected_production_1b_lr1500_fresh/gates/strategy_sequences_1b.json`;
- `results/corrected_production_1b_lr1500_fresh/gates/holdout_1b_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/h2h_1b_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/mechanics_1b_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/opportunity_1b_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/card_funnels_1b_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/context_kl_1b_*`;
- `results/corrected_production_1b_lr1500_fresh/gates/context_decks_1b_*`; and
- `results/corrected_production_1b_lr1500_fresh/gates/gate_kl_1b_*`.

### Long-run element-isolation ablation

The 450M gate adds a concrete post-control hypothesis: the universal policy may be suffering destructive interference between element-specific drafting and battle strategies. The exact p29300 checkpoint broadened its spell, weapon, ability, and deck-diversity behavior while collapsing on several Lightning, Water, and Fire gates; the retained p21000 upper envelope remained concentrated and robust. This is evidence for testing isolation, not proof that shared parameters caused the regression.

The high-cost reference treatment is four independent full models, one each for Lightning, Water, Earth, and Fire. Each model owns its complete observation encoder, recurrent state, draft policy, battle policy, value function, and optimizer state. The deployed system routes a player to the model for that player's assigned element.

Assignment remains exogenous rather than learned:

- For the learner seat, sample uniformly from the element's two gates and two compatible leaders, covering the full `2 gates x 2 leaders` factorial.
- Balance learner seat and starting player within every gate-leader context.
- Sample opponents across all four elements, all eight gates, and all compatible leaders; an element specialist must learn cross-element matchups rather than train only on mirrors.
- Keep opponent element, gate, leader, seat, and starting-player frequencies matched across specialists and the universal control.
- Preserve legal deck construction: the learner drafts the assigned element plus neutral cards under the same card-pool and copy-limit contract as the control.

The practical continuation experiment should clone the same qualified p21000 parent into all four specialists and a universal-control continuation. Preserve model, optimizer, scheduler, league, promotion, environment progression, source, and config fingerprints before applying only the assignment/isolation treatment. A fresh-from-random element-isolated run is a separate experiment and must not be mixed into the continuation result.

Use the same long-horizon milestones as the universal run: 50M, 100M, 200M, 300M, 450M, and, if still improving, 800M/1B configured sampled rows per specialist. This is approximately four times the learner compute of one universal model. Report two comparisons so the cost is explicit:

1. Equal per-model experience: each specialist receives the same configured rows as the universal control.
2. Equal total system compute: compare the universal control against the specialist system at one quarter of the per-specialist horizon.

Evaluation must treat the four specialists as one routed policy system:

- fixed-reference panels for every own-element gate-leader context against every opponent element;
- a complete specialist-versus-specialist cross-element payoff matrix with both seats and starting-player conditions;
- matched direct games against p18000, p21000, and the universal continuation;
- pooled score, per-element score, worst opponent element, worst gate-leader context, and late-window slope;
- opportunity-normalized spell, weapon, portal, leader/Garden ability, response, and attack-target use;
- deck concentration, singleton/four-copy shares, curve, type mix, element share, and archetype diversity;
- blind human evaluation with model and element-treatment identity hidden.

The treatment qualifies only if the routed specialist system improves paired external or ancestor strength without moving any element's worst matched slice backward by more than three points, and if its late window avoids the broad cross-gate collapse seen after p27000. A pooled gain that hides one failed element is a rejection. Full model isolation is intended as the definitive interference test, not the default production architecture.

Lower-cost treatments should be tested as separate matched arms after the full-isolation result establishes whether parameter sharing is causal:

- **Element-specific actor heads:** share card/observation encoders and the recurrent trunk, but route draft scoring and battle action heads by the learner's assigned element.
- **Element-specific recurrent adapters:** insert small residual adapters around the encoded observation and recurrent output, selected deterministically by element, while retaining shared base weights.
- **Element-specific recurrent cores plus actor heads:** share card encoders but isolate the LSTM and action heads when lightweight adapters are insufficient.
- **Mixture-of-experts routing:** use deterministic own-element routing first; learned routing adds an unnecessary credit-assignment problem and is not the initial ablation.

Element-specific critics may be paired with these treatments if value interference is independently demonstrated, but a privileged or isolated critic alone does not isolate the deployed policy. Change one sharing boundary per arm. Do not combine element isolation with league, reward, entropy, or assignment changes in the same attribution experiment.

This ablation is motivated by the hero-isolation result in ByteDance's Hearthstone study, which reported a 6.5-point gain from replacing one three-hero policy with one full model per hero: `https://arxiv.org/abs/2303.05197`. Their result is precedent for destructive cross-strategy interference, not evidence that Azuki's four elements require the same final architecture.

## Source evidence

Primary repository evidence:

- `python/config/azuki_deckbuild_production_3090.ini`
- `python/src/train.py`
- `python/src/league_manager.py`
- `python/src/league_state.py`
- `python/src/league_training.py`
- `train-ablation-1781126582/final-report.md`
- `train-ablation-1781126582/promotion-ablation.md`
- `train-ablation-1781126582/results/next_ablation_v1/credit_prefix_45m_behavior_analysis.md`
- `train-ablation-1781126582/results/production_launch_v1/parent_p10730/qualification/qualification_summary.json`
- `train-ablation-1781126582/results/production_launch_v1/evaluation_protocol_v1.json`
- `train-ablation-1781126582/results/production_launch_v1/soak/20260817T232829Z/soak_report.json`
- `optimizations/training-speedup-2026-07.md`

## Autoresearch handoff

The fixed production-league benchmark is useful as a semantic and cold-start diagnostic, but its headline `production_league_sps` is not the long-run throughput metric for this plan.

The benchmark runs only eight epochs and computes:

```text
production_league_sps = trained agent steps / complete subprocess wall time
```

That denominator includes Python startup, loading the 90.2M-parameter learner, loading or materializing historical opponents, fresh isolated TorchInductor/Triton compilation, Dynamo recompilation, eight training epochs, checkpointing, and process exit. A result near 190 effective SPS is therefore compatible with cold-start overhead and is not comparable to the warmed 1.28-1.40k trainer windows used for the local-run estimate.

The meaningful diagnostic from that harness is `trainer_tail_sps`, the mean trainer-reported SPS over the last five epochs after warm-up. It should be judged against the 1,235 launch floor and the soak's 1,270.86 95%-baseline requirement.

The interrupted candidate lazily materialized frozen league opponents and reached a diagnostic 2,470 learner-action SPS mean over five warm epochs, but stopped after six of the required eight epochs. The harness emitted no official `trainer_tail_sps` or `production_league_sps`, and the candidate was not qualified for production. Its patch and logs are preserved under `train-ablation-1781126582/results/autoresearch_handoff_20260819/`; the source change was removed before preflight.

Do not delay the local production run for further optimization of the eight-epoch benchmark. The autoresearch controller and orphaned workers must be absent before launch. Do not use a roughly 190 cold full-wall score to revise the 5.5-7-day local estimate.
