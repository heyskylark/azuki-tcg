# Strategy-learning experiments

## Purpose

The corrected production run is a healthy systems run and a failed strategy-learning run. This document is the experiment ledger for replacing the current training recipe. It separates attribution experiments from the final combined canary so that a later result can be explained rather than merely observed.

Primary objective: learn high-ceiling, element-, gate-, leader-, deck-, and opponent-conditioned play. Early win rate against cheap face-pressure policies is a safety signal, not the primary short-run optimization target.

The target is not fewer face attacks. The target is to stop face attack from becoming the only robustly learned plan. A correct evaluator must distinguish:

- taking a legal lethal or efficient face attack;
- attacking face because no stronger line is available;
- ignoring a legal entity-control, setup, response, portal, or combo line because immediate face damage has an easier proxy reward; and
- building an entity-heavy common shell that makes strategic cards unavailable in the first place.

## Current final-validation decision (2026-09-06, descriptor v3)

**NO-GO.** Retained-trace revalidation supersedes the earlier v1/v2 strategic dispositions below. R14 still passes reward reconstruction and training integrity, but its corrected sampled Fire breadth falls from three converted lines at p1950 to two at p2925. The previously hidden difference is Zero self-damage -> target damage/ATK increase -> same-turn attack conversion: `1 -> 0` sampled and `0 -> 0` deterministic. That sparse signal does not prove permanent forgetting, but it does not satisfy the registered no-reduction gate.

The measurement contract explicitly covers:

| Element | Deck and action evidence | What does not qualify |
|---|---|---|
| Water | Gate/leader-fit spells and resource readiness, useful spell effects, recovery/replay, and later spending of readied IKZ | Untapping or including spells without using them; double-counting an untapped resource addition |
| Earth | Defensive opportunities, observed interception, leader HP restoration, and survival/setup conversion | A defender declaration or later entity death alone; healing without checking timing |
| Fire | Self-damage that enables effective same-turn attack, alongside coherent Charge/multi-play lines | Self-damage alone, expired buffs, or using Rushfire/Kagoro as proof that a self-damage strategy was learned |
| Lightning | Weapon draft support, valid recovery/re-equip, and attacks while the destination still has the weapon | Attachment alone, an unrelated later same-code attacker, or inferring damage from eventual victory |

These are minimum interpretable readouts, not named-card rewards or prescribed decks. Coherent new lines remain admissible through trace review and matched payoff probes. Do not rank raw action volume, enforce lower face frequency, or equate a legal alternative with a stronger alternative. Deck difference is not causal gate/leader fit; generic common cards are acceptable when they support the actual plan.

Completed: 218 retained ablation descriptors and all seven baseline checkpoint descriptors rebuilt at schema v3; the 21-pair sensitivity packet, decision packets, strategy-first reassessment and superseded canary contract refreshed; 24 focused CPU regression tests passed. Original decisions are content-hash archived at `results/strategy_semantics_v3/prior_decision_manifest.json`.

Load-bearing corrections include actual target damage instead of eventual-winner credit, the correct alley target offset, separate IKZ addition/untapping, conservative duplicate-copy recovery, real deferred Echoed Waves/Rushfire selection paths, exact Lightning/Rushfire destination slots, opponent-triggered Bobu healing, and same-turn Fire conversion. Spell self-destruction alone is not credited as a positive effect. Trace v2 still lacks physical-copy/effect IDs and complete deferred combat attribution; these remain explicit unmeasured limitations rather than fabricated success.

R14 deterministic p2925 exposes a second gap: all 48 Hydromancy player-games draft zero spells, despite 281 IKZ readied. Echoed Waves supplies all 49 spell-containing Water decks and 109 spell plays. Pooled Water support therefore cannot qualify Hydromancy resource/spell strategy.

R3 direct-edge removal is **redundant**, not an outstanding training arm: both direct deltas are already zero in R14 config and telemetry. D2/D3 long confirmations, replay-off, strategic exposure, qualified league diversity/SPS, and separate S1/S3 are not production-qualified; no 1B launch is authorized. The user resolved the historical GPU-driver stall by restarting the host and authorized the next diagnostic experiments. An actual CUDA R14 policy smoke subsequently completed one sampled game normally.

Current evidence: `results/successor_final_validation/decision_packet.json`, `results/successor_final_validation/verification.json`, `results/terminal_safe_reward_horizon_followup/final_decision_packet.json`, `results/strategy_baseline_v1/baseline_packet.json`, and `results/strategy_first_reassessment.json`. The operational endpoint table is in `local-production-run.md`, **Schema-v3 final validation outcome**.

The rollup verifies 218 ablation descriptor IDs, seven baseline descriptor links, all 21 unique baseline pairs, and all 17 archived original decisions. Launch safety is exercised: 16 reward-registration entry paths reject superseded parents, and the superseded canary launcher rejects activation. Seven historical reward builders emit only superseded observations; the R5/R6 builder rejects its retained registration-hash mismatch. The mismatched historical registration is not silently rewritten to manufacture provenance, and does not invalidate the separately verified R14 registration/config hashes.

### Authorized current-source strategy discovery

The action-probability correction affects training likelihoods and KL, not merely evaluation. Historical R14, D2/D3, random-prefix and replay manifests omit the sampler/policy hashes needed to establish correction provenance. Their pre/post-fix status is **unproven**, not blanket-invalid. Retained weights remain useful observational controls; corrected inference cannot repair historically incorrect gradients. The stale text-feature-cache evaluator defect is separate.

Fresh comparisons use an R14-shaped **diagnostic control**, with swapped-gate draft credit excluded in every arm. This does not resolve R14's production strategy hold. Source fingerprints include the sampler, policy, trainer, runtime bindings, deck proposals and evaluator. All 99 focused probability, prefix, credit, resume and strategy-descriptor tests pass, including strategic card-identity selection, the four-pick boundary and missing-card rejection. Provenance evidence: `results/strategy_discovery_audit_v1/provenance.json`.

The authorized boundaries remain separate:

- Same frozen policy on learned, entity-only and regional-derived strategic decks, with identical battle initialization and paired opponents/seats. Report both policy modes and candidate-seat elemental evidence, not pooled opponent mechanics.
- Three fresh exposure arms: no supplied decks, entity-only supplied decks, strategic supplied decks. The proposed `0.20` lottery means supplied-deck **episodes with the learner assigned that seat**, not 20% of all two-seat rows. Report actual supplied learner battle rows. Forced cards receive no draft actor credit.
- Equal-length random versus strategic four-card enabling prefixes: probability `0.20`, otherwise no prefix; policy chooses the remaining cards. Prefixes are disabled during evaluation.
- Matched 45M D2/D3 credit-tail comparison (`0.1` versus `0.0` after the shared halfway anneal).
- Replay `0.15` versus replay disabled, with swapped-gate credit excluded in both.
- Separately registered learner discount `0.99` versus `1.0`, leaving PBRS gamma at `0.99`. This isolates the learner discount parameter but does not preserve the PBRS policy-invariance guarantee under gamma `1.0`; interpret accordingly.

Regional proposals exclude all duplicate signatures of promotion and holdout panels. Sundering Strike (`AZK01-127`) is Normal and is compatible with every elemental deck. The completed run used an older catalog; its frozen pools and results remain historical evidence, not the current legality baseline. Its Water and Earth panels each retained one enabling-role source and derived a second variant by redistributing one copy among existing legal cards. Those variants are not independent tournament submissions. Its Lightning panel used four source submissions and Fire five. Rebuild new exposure pools against the corrected catalog. No named-card rewards, forced face policy, learned-deck card restriction, or automated production promotion is introduced.

Execution completed under managed process `strategy-discovery-v1`; the final Discord notification was delivered and the watcher exited normally. Registration: `results/strategy_discovery_v1/registration.json`; terminal execution state: `results/strategy_discovery_v1/campaign_status.json`. The initial 1,536-game diagnostic, seven matched 15M screens and two 45M D2/D3 arms are complete: 195,056,640 configured rows total. Every arm started fresh at seed 42. Evaluation used retained checkpoints, 200 games per policy mode/window, and disabled prefixes/exposure. The reviewed disposition is below; no experiment winner was selected automatically.

Prelaunch proof: strategic-prefix and supplied-deck CUDA training each completed 10 updates / 153,600 rows with checkpoint parity. Prefix smoke forced 796 picks. Supplied-deck smoke counted 10,176 supplied learner battle rows / 33,256 learner battle rows; this startup observation is not an estimate of steady-state exposure. Six paired fixed-deck sampled games completed without truncation. The guarded scheduler completed an actual one-update training/checkpoint smoke after correcting directory discovery to exclude the trainer's sibling `.pt` copy. Production scope, a 1B-sized arm, sampler-source drift, changed prefix-pool bytes and missing prefix contexts are rejected. Temporary smoke weights/scripts were removed; command, metric and manifest evidence remains in `results/strategy_discovery_audit_v1/`.

The completed observations have now been reviewed in `results/strategy_discovery_v1/review.json`. All nine training runs passed integrity/reconstruction checks; the review additionally checked retained self-play completion and found one incomplete final curated probe, reported separately below. Formal recipe qualification, S1/S3, league qualification and the successor 1B remain blocked.

Discord monitoring ran as the independent persistent process `strategy-discovery-discord`: 60-second polls, 30-minute progress summaries, stage/failure alerts, and a possible-stall warning after 30 minutes without artifact activity. Its final **CAMPAIGN FINISHED — READY FOR REVIEW** notification was delivered. No model session or unregistered experiment was spawned automatically. Resumed delivery state is `results/strategy_discovery_v1/discord_monitor_continuation_state.json`; the first attempt's state and verification remain in `discord_monitor_state.json` and `discord_monitor_verification.json`. Frozen training sources and qualification boundaries are unchanged.

#### Same-policy diagnostic observations and recovery

All 1,536 unique games completed without truncation (768 sampled, 768 deterministic). Report generation alone failed on relative/absolute path handling. The report directory is now resolved, and `run_deck_diagnostic.py --report-only` validates every retained shard against a separate recovery registration before rebuilding descriptors. No games were rerun and no trace bytes changed. The original diagnostic registration remains immutable; `results/strategy_deck_diagnostic_v1/registration_report_recovery.json` records the report-only revision. The recovered `decision_packet.json` references that recovery registration. Original source/status/monitor evidence and the continuation review are in `results/strategy_discovery_audit_v1/report_recovery/`.

Candidate-seat findings (64 games per element, arm and policy mode; retain the gate/leader breakdown in the packet):

- Lightning: learned -> strategic attacks dealing observed damage while equipped increased from 102 -> 238 sampled and 87 -> 236 deterministic. This demonstrates usable weapon access under supplied decks, not optimal weapon play or weapon-added causal damage.
- Water: strategic decks provide spells in all deterministic Hydromancy games where learned decks provide none. Hydromancy resource-to-spell lower-bound units reach 13 deterministic under strategic decks. However, strategic spell effects are unmeasured for 60/60 sampled and 63/64 deterministic selections; zero observed positive effects and zero converted Echoed replay effects cannot establish zero actual benefit.
- Fire: learned and strategic decks produce zero observed Zero self-damage-to-attack conversions in both modes, despite 32 eligible candidate games per arm/mode. The entity-only control has one sampled conversion; Charge/multi-play support is not a substitute.
- Earth: defense/healing support varies by mode and mechanic, with no uniform strategic-deck improvement. Supporting scores likewise do not establish a strategic-deck winner.

Decision: continue the unchanged registered fresh control, exposure, prefix, replay/discount and D2/D3 comparisons. Supplying cards can expose usable lines (especially Lightning), but access alone has not demonstrated the missing Fire strategy. Water's immediate-effect observation coverage limits usefulness comparisons. No recipe is selected and no production qualification is granted. The existing runner continues without `--deck-diagnostic` only after this completed-game integrity review, preserving the original training registration and all runtime guards.

#### Completed discovery review

**No production promotion. Strategic prefixing is the strongest diagnostic lead, not a selected recipe.** Evidence, input hashes, all 58 mode/window sequence summaries, gate/leader count rollups, achieved intervention telemetry and supporting score trajectories are retained in `results/strategy_discovery_v1/review.json`. The execution status remains an immutable record of observation completion; the separate review records this decision.

Verification: all 29 evaluated checkpoint hashes match their indexes and descriptors; training-status source mappings match registration, and train/evaluation config hashes match. All 1,272 training metric windows contain finite numeric values; four integrity maxima and invalid-metric maxima are zero. Maximum raw/scaled reward reconstruction errors are below `1.6e-8` / `1.2e-7`; PPO component reconstruction maximum is `2.8611e-6`. Steady median SPS ranges from 1,411 to 1,520. All 11,600 retained self-play games terminate without truncation or duplicate game/seed identities within a trace, and all 16,704 H2H games complete normally.

The final curated panels contain **287/288 completed games**. PREFIX_RANDOM's Echoed Waves / Shao probe at seed 198312, strategic seat 1, reached 500 steps. Its recorded score/behavior zeros are not an observed loss or zero action use. Completed-only random-prefix score is 9/31, with censoring still a limitation. Other arms have 32/32 completed curated games. Four games per gate and one leader per gate are too sparse for recipe qualification.

Actual exposure: ENTITY supplied 791,259 / 3,704,649 learner battle rows (21.36%); STRATEGIC supplied 751,895 / 3,613,560 (20.81%). Update-weighted telemetry confirms four-card episode fractions of 20.013% random and 19.982% strategic. These are not percentages of all two-seat training rows.

| Matched comparison | Strategy result | Disposition |
|---|---|---|
| CONTROL / ENTITY / STRATEGIC | All three final deterministic Water policies draft no spells in 107/107 Water player-games. STRATEGIC Surge converted/eligible falls `33/38 -> 32/36 -> 20/26`, while CONTROL reaches `37/42`; final Zero conversion is only `1/49` in either mode. Earth improvements are mixed. | Full-deck strategic exposure has not shown robust free-draft transfer over CONTROL; do not promote either exposure recipe. |
| Random / strategic prefix | With evaluation prefixes disabled, strategic prefix retains Water spell access in all contexts. At p650/p975, Hydromancy has `19/21 -> 15/22` positive observed spell effects and `12 -> 14` readied-resource-to-spell lower-bound units; Echo conversions are `6/13 -> 7/19` eligible player-games. All seven final Echo conversions use Shao (`STT02-001`); none use leader `AZK01-125`. But Zero conversions fall sampled `10/50 -> 1/44 -> 1/49`, deterministic `6/50 -> 0/44 -> 0/49`. | Preserve the Water transfer signal and early Fire acquisition for targeted investigation; persistence and strength are inadequate. Random remains an attribution control, not a selected recipe. |
| Replay on / off | Replay-off Surge conversions fall deterministic `18/19 -> 9/9 -> 0/0`; final sampled `18/20` also trails CONTROL `33/36`. Its mid-window Zero signal disappears deterministically. | Do not select replay-off. |
| Learner gamma .99 / 1 | Gamma1's mid-window deterministic Zero `3/44` returns to `0/49`; final Water has no spells. Lightning evidence is mixed rather than a new persistent strategy. PBRS remains gamma .99. | Deprioritize gamma1; no shared-discount invariance claim. |
| D2 .1 floor / D3 zero tail, both 45M | Neither retains deterministic Zero conversion. Final Stone conversions are `35/46` versus `24/45` eligible player-games; Bobu's apparent advantage reverses between sampled and deterministic modes. Water deferred-effect coverage limits comparison. | Do not replace the .1 floor with a zero tail; neither arm is qualified. Do not infer a credit effect from 45M versus 15M. |

Supporting deterministic score against p21000/p44000/p60000 (576 games/window, not the selection objective):

| 15M arm | p325 | p650 | p975 |
|---|---:|---:|---:|
| CONTROL | 7.47% | 24.13% | 28.13% |
| ENTITY | 4.86% | 21.70% | 20.83% |
| STRATEGIC | 10.59% | 21.01% | 28.30% |
| PREFIX_RANDOM | 14.24% | 20.49% | 27.26% |
| PREFIX_STRATEGIC | 4.17% | 14.93% | 15.63% |
| REPLAY_OFF | 6.94% | 20.14% | 18.92% |
| GAMMA1 | 6.60% | 14.06% | 21.53% |

D2's matched 45M trajectory at p325/p975/p1950/p2925 is `10.76 -> 20.66 -> 18.40 -> 24.31%`; D3 is `10.07 -> 8.51 -> 9.72 -> 7.47%`. One training seed does not establish seed-robust superiority. Self-play sequences are observations with co-evolving opponents, not causal action-quality estimates. Sequence denominators count eligible player-games; legal-decision opportunities are distinct. Water deferred effects remain unmeasured where coverage is absent, not ineffective.

Recommendation at the completed-campaign review, before subsequent authorization: first evaluate retained CONTROL/random-prefix/strategic-prefix early, middle and final checkpoints against a paired, standardized opponent panel, both leaders per gate and both policy modes. Inspect early successful versus later failed Fire paths, preserve Water effect-coverage reporting, and keep Earth/Lightning regression checks. If the signal survives, register independent-seed, equal-horizon prefix replication with explicit persistence and acceptable strength-regression criteria. Do not combine interventions or launch 1B from this screen. The authorized evaluation launch is recorded below.

#### Authorized retained-checkpoint followup

The user authorized the proposed evaluations and inspections. Registered evaluation-only campaign: `results/prefix_paired_followup_v1/registration.json`; live status: `results/prefix_paired_followup_v1/campaign_status.json`; persistent managed process: `prefix-paired-followup-v1`. No new training or production promotion is authorized by this followup.

Panel: CONTROL, PREFIX_RANDOM and PREFIX_STRATEGIC at p325/p650/p975; sampled and deterministic actions; all 16 candidate gate/leader contexts; two frozen opponent checkpoints (p21000/p60000), each spanning eight gate/leader assignments; both candidate seats. The shared 512-task schedule per checkpoint/mode totals **9,216 games**, executed by six CPU workers. Each exact opponent context has one world seed and its seat swap; these are not independent repeated seeds within that full context. Actual starting player is stratified in reports rather than assumed.

Both seats freely draft. Separate policy recurrent histories persist through draft and battle; only the active seat advances, matching native H2H progression. This differs from the previous all-row self-play logger, so raw old/new behavior is not an evaluator-parity claim. Private per-seat sampling streams and paired world seeds prevent candidate sampling from directly consuming the opponent's stream; diverging trajectories can still alter observations and engine RNG consumption. Prefixes, supplied decks, replay and sibling-matchup oversampling are disabled. This panel is not a reset-state fixed-deck probe.

`run_prefix_paired_eval.py` loads both real checkpoints strictly, checks source/config/metadata/checkpoint hashes, verifies full free drafts and forced assignments, and scores from the engine terminal component. Incomplete games retain a null score and stop the shard. `run_prefix_followup.py` aggregates candidate-seat-only contextual effects and supporting scores, preserves trace hashes, and sends progress/completion/failure notifications to the existing private Discord endpoint. Neither script launches training or selects a winner.

Prelaunch proof in `verification.json`: four standalone paired games and four orchestrated games completed; both policies, both modes, both candidate seats and both frozen opponents were exercised across the smokes. Three overlapping tasks reproduced draft actions, battle traces and outcomes exactly across standalone/sharded execution. Each two-game report contained exactly two candidate player-games, not four mixed-seat player-games. Existing trace overwrite was rejected. Discord transport alone was replaced by a no-op during the orchestrator smoke; full-campaign delivery is recorded separately.

Retained inspection in `retained_inspection.json` reproduces original Zero eligible/completed/converted totals across all 18 target traces. Strategic-prefix Zero is selected in every eligible player-game at every window: the collapse is not explained by losing all legal ability access or ceasing to activate it. At p975 deterministic, 88/93 unambiguous target selections hit entities that already attacked that turn and are tapped; 56 targets disappear after selection. At p325, the corresponding already-attacked count is 98/206. Categories overlap, and death-trigger value is not ruled out. Concrete p975 game 2 / seed 80772663 attacks with a target at step 43, then buffs that already-tapped target at step 47; early p325 game 13 / seed 74355222 instead activates, buffs and attacks at steps 53/54/55.

Water transfer is narrower than pooled spell metrics suggest: **all 57 final deterministic spell selections are Healing Flutter (`AZK01-002`, Normal)**. The middle checkpoint also selects only Healing Flutter; early selections include Chilling Water, Commune with Water and Water Orb, whose deferred effects have limited observation coverage. Useful healing/resource/replay observations remain valid but do not establish broad Water spell mastery. Earth/Lightning comparisons and both-mode trajectories are retained for regression checks.

Paired review complete: `results/prefix_paired_followup_v1/review.json` records result hashes, full contextual aggregates, matched outcome differences and mechanism evidence. All 108 trace hashes, 9,216 registered task assignments and engine-terminal candidate outcomes were checked; all games completed, with no draws. Every checkpoint/mode has 512 candidate observations and balanced actual starting order (256 first / 256 second). Discord completion delivery is recorded in campaign status. The runner's `completed_observations_require_review` state is execution history; `review.json` records the subsequent review decision.

| Arm | p325 sample / argmax | p650 sample / argmax | p975 sample / argmax |
|---|---:|---:|---:|
| CONTROL | 3.12% / 6.84% | 7.42% / 26.76% | 12.70% / 31.84% |
| PREFIX_RANDOM | 4.30% / 14.84% | 7.23% / 24.61% | 11.13% / 31.25% |
| PREFIX_STRATEGIC | 1.37% / 4.10% | 5.66% / 16.02% | 9.77% / 19.92% |

These are supporting paired-panel win rates, not promotion thresholds. Strategic prefixes trail both controls at every checkpoint in both modes. Final argmax wins are 102/512 versus CONTROL 163/512 and RANDOM 160/512. Against CONTROL, matched tasks contain 37 strategic-only wins versus 98 control-only wins; the deficit appears against both frozen opponents and both starting-order strata. Final argmax strategic wins are lower in 13/16 candidate gate/leader contexts than CONTROL, with two higher and one tied. Each candidate context has only 32 games; exact full contexts have one world seed plus seat swap, and training has only one seed.

Fire discovery is real but not persistent: strategic Zero completes/converts 10/10 of 64 eligible games at p325 sample and 11/10 of 64 at p325 argmax, then 0/0 of 64 at both p650 and p975 in both modes. CONTROL has zero throughout; RANDOM has at most two conversions per cell. Strategic activation and self-damage still occur in all 64 eligible games at every checkpoint/mode. In final argmax `trace_00.jsonl`, task `c03_o0_e0_r0_s0`, seed 931030001, candidate p0 attacks with STT02-004 at step 46, activates Zero at 47, then targets that now-tapped 1-HP entity at 48; it disappears. This supports a timing/targeting limitation, not a claim about counterfactual death-trigger value.

Water retains a narrow deterministic behavior: all 128 strategic final argmax Water decks contain spells (256 total spell slots), whereas both controls have zero spell slots. All 88 selected/resolved spells are Normal Healing Flutter (`AZK01-002`), distributed 21/23/23/21 across the four Water gate/leader contexts. Effects are observed for all 88 and classified positive for 79; these are action counts, not wins or distinct games. Healing timing converts in 54/64 eligible games, Echo replay in 13/19, and Hydromancy-to-spell resource use has a lower bound of 26 IKZ. Sample behavior remains broader, but strategic final effect coverage is only 40 observed of 196 selections (35 positive); unobserved deferred effects are not failures. Final Water wins are strategic 33/128, CONTROL 44/128, RANDOM 43/128 argmax; sample is 11/128, 11/128, 10/128. Spell access is not broad Water mastery or a strength gain.

Earth/Lightning do not rescue the intervention. Final argmax Stone converted/eligible is strategic 47/52, CONTROL 37/48, RANDOM 58/61; Bobu is 4/64, 2/64, 10/64. Surge is 44/48, 50/52, 0/0 respectively: RANDOM loses eligibility, not execution among eligible games. Strategic early Stormchain conversions (8/9 argmax, 3/5 sample) become 0/0 at the final checkpoint in both modes. Final strategic Earth/Lightning wins are 13/128 and 27/128 versus CONTROL 15/128 and 45/128. Sequence proxies and immediate-effect coverage do not establish causal tactical value.

**Superseded review decision:** the initial paired review recommended against unchanged independent-seed replication, partly on short-run strength. The user explicitly corrected this objective: signs of unique strategy use, not strength or win rate, determine this diagnostic's value. The raw `review.json` remains historical evidence; its no-replication recommendation is superseded by the strategy-first authorization below.

#### Authorized strategy-first recipe search

The paired observations are a qualified strategy-discovery success: early Zero and Stormchain sequences emerge, while deterministic Water spell drafting, healing and gate-supported replay/resource use persist without evaluation prefixes. Normal Healing Flutter does not invalidate Water-specific replay/resource strategy; it limits spell diversity. Loss of late Fire/Stormchain behavior is a retention question, not a win-rate rejection. More training is not assumed to preserve a strategy automatically.

The user authorized replicated recipe experiments toward a later 1B run. The next diagnostic compares five recipes at two fresh matched seeds (43, 44): no prefix, random four-card prefixes at 20%, strategic four-card prefixes at 20%, strategic prefixes at 50%, and strategic prefixes at 20% with the existing exploration-reward scale held at 1.0 instead of decaying to 0.15. The last arm changes reward scheduling, not policy entropy, and leaves potential shaping unchanged. Increased exposure and held exploration are separate arms, not an untested combined recipe. All other training settings remain matched to the diagnostic control.

Each run has 15,006,720 configured rows, with p325/p650/p975 paired free-draft evaluations, prefixes/exposure/replay disabled, both policy modes, all 16 candidate gate/leader contexts, both frozen opponents and seat swaps. Ten runs total 150,067,200 rows and 30,720 evaluation games. The shared evaluation schedule supports matched comparison, not new independent world seeds in each exact cell; the new independence is in training seeds.

Selection evidence: repeated acquisition across training seeds; early-to-middle-to-late persistence; opportunity coverage and realized sequence conversion reported separately; conditional converted/eligible and unconditional converted/player-games; gate/leader and opponent coverage; spell/card diversity and effect-observation coverage. Sample and argmax remain separate. Raw action volume, common-spell identity, lower face frequency and short-run win rate are not admission gates. Descriptive nonzero replication flags are not statistical significance or automatic production qualification.

Registration/output location: `results/strategy_recipe_v1/`. `build_strategy_recipe.py`, `run_strategy_recipe.py` and `report_strategy_recipe.py` own immutable construction, guarded serial training/paired evaluation, and strategy-first evidence reporting. Missing batches remain explicit; no silent zeros, automatic winner or 1B launch. A promising recipe requires reviewed longer-horizon retention confirmation before extrapolation to 1B; this authorization does not claim that a best recipe has already been found.

Launch proof: managed process `strategy-recipe-v1` started from registration SHA256 `4e452466ecd57066240ffe37d5a08efc6806defd8dbe0ff74f3a732b9ea6dd67`. `verification.json` records the ten-update CUDA smoke, eight real paired games, delivered Discord notices, retained-panel reporter exercise, explicit missing-batch handling, matched configuration contrasts and rejection of production/1B/seed/source drift. The initial strategy report is provisional with 60 expected batches; results must not be inferred from process readiness. Monitoring reports stages/progress/failure/completion; no automatic 1B launch.

#### Completed recipe review: STRATEGIC50 leads retention confirmation

All ten runs completed: 60/60 batches and 30,720 paired games. Recorded invalid/integrity maxima are zero. Each run used 15,006,720 configured rows; final learner steps were approximately 9.34–9.38M, not 15M. The completed report and evidence-backed recommendation are `results/strategy_recipe_v1/strategy_report.json` and `review.json`; campaign status retains its execution-history state.

STRATEGIC50 has the clearest replicated distinctive improvement. Zero converted/eligible at p325 → p650 → p975:

| Mode | Seed 43 | Seed 44 |
| --- | --- | --- |
| argmax | 0/64 → 25/64 → 31/64 | 7/64 → 6/64 → 23/64 |
| sample | 6/64 → 22/64 → 29/64 | 11/64 → 11/64 → 19/64 |

The final four batches were independently recomputed from hash-verified traces and matched the report. Conversions cover both Zero gate contexts, both candidate seats and both frozen opponents. Seed43 argmax uses Normal Gurugumi Vanguard and Rei; seed44 also uses Fire Fanatic Kindler and other entities. Representative traces show leader self-damage, a surviving entity gaining attack/losing health, and that entity attacking in the same turn. These are coherent sequences, not activation counts or evidence of causal optimality.

STRATEGIC20 does not replicate the earlier seed42 Zero/Water signature: final Zero is 0/64 and 1/64 in argmax, with no deterministic Water spell slots in either seed. RANDOM20 discovers early Zero but loses it; held exploration retains Zero only in seed44 (argmax 30 → 18 → 14 conversions, seed43 all zero). STRATEGIC50 therefore leads the tested recipes for this retention question, not for every strategy.

Water healing and Echo replay persist in sampled play across recipes, including CONTROL. Final STRATEGIC50 sample healing is 19/19 and 17/18; Echo is 7/36 and 3/22. Both seeds select 18 spell identities across Water contexts, but only two identities have positive observed effects; deferred effects remain incompletely observed. No recipe replicates late deterministic spell drafting across both seeds. Final spell lines appear in CONTROL seed43 and HOLD seed44, not their companion seeds. Normal support cards count; raw spell diversity is not 18 validated strategies.

Lightning remains mixed: STRATEGIC50 seed43 argmax Stormchain falls from 6/6 at p650 to 0/0 at p975, while seed44 goes 5/7 → 3/4. RANDOM20 has stronger replicated late argmax Stormchain coverage (9/10 and 15/19). Surge, Stone, Shao and sparse Bobu lines occur in controls as well; Devotion has no observed conversion in this campaign. These observations prevent describing STRATEGIC50 as a broad strategy solution.

Recommendation, not a launch: preregister a longer (proposed 50M configured rows) STRATEGIC50 retention confirmation across at least two seeds, with a RANDOM50 dosage-matched comparator. RANDOM50 was absent here, so strategic content and increased prefix dosage are not fully separated at 50%. Keep intermediate checkpoints, free-draft sample/argmax evaluation, opportunity denominators, cross-context trace review, and Water/Lightning/Earth breadth checks. Do not combine held exploration with 50% prefixes without testing. No winrate gate, automatic 1B launch, or assumption that short-horizon retention guarantees 1B retention.

#### Authorized 50M retention confirmation running

The user authorized the longer comparison and requested live Discord monitoring plus oldest-model-only cleanup if disk space was insufficient. No cleanup was needed: the training filesystem had about 951 GiB available (`df` 73% used). The prior ten-arm recipe campaign occupied 108 GiB; a conservative linear projection is about 144 GiB for this confirmation, with 200 GiB allowed for planning. All models, notes, reports and experimentation records were preserved.

`results/strategy_retention_v1/registration.json` (SHA256 `1aec81fcb8bd43c70fb10d35ed2af35e0360abac9d771b9a1e5cd3f499d88854`) registers STRATEGIC50 and RANDOM50 at matched fresh seeds 43 and 44. Each run has 50,012,160 configured rows / 3,256 updates; four runs total 200,048,640 configured rows. Both recipes use four-card prefixes with probability 0.5; only strategic versus random content differs within a seed, apart from output identities/paths. No held-exploration combination is introduced.

Reward exploration/potential anneal endpoints stay at 15,006,720 rows, preserving the tested recipe's reward schedule. Fresh training extends the horizon-dependent learning-rate schedule to 50M; this is not an exact continuation or action-parity replay of the earlier 15M models. Checkpoints p325/p650/p975/p1625/p2275/p3250 probe approximately 5/10/15/25/35/50M configured rows. Evaluation follows training for each arm and reuses the frozen 512-task free-draft panel, both modes, all 16 candidate contexts, both frozen opponents and seat swaps: 48 batches / 24,576 games. Evaluation prefixes, supplied decks, replay and same-element oversampling remain disabled.

The existing builder, guarded runner and reporter now support the distinct `strategy_retention_v1` family; matched comparisons use RANDOM50. Before launch, both recipes passed ten-update CUDA smoke runs and 16 real paired games. All four smoke batches completed, both-mode RANDOM50 comparisons were available, and invalid/integrity maxima were zero. Production/1B/seed/missing-comparator/source-drift mutations were rejected. The independent watcher correctly observed 30,720 existing traces and all ten completed prior arms, and its process-exit/stall event paths were exercised. Evidence: `results/strategy_retention_v1/verification.json`; retained smoke: `results/strategy_retention_smoke_v1/`.

Persistent process `strategy-retention-v1` owns serial training/evaluation. Persistent, read-only process `strategy-retention-discord` independently follows its PID and artifacts, polls every 60 seconds and sends 30-minute progress plus stage/stall/process-exit/completion notices. Its attachment was delivered on the actual full campaign; its completion delivery was verified end-to-end on the smoke. The runner also retains its own stage/failure/completion notifier. Both survive closing the assistant session; neither launches a model session or a 1B run. Monitor state is `discord_monitor_state.json`, execution state is `campaign_status.json`, and strategy evidence is `strategy_report.json`.

Review repeated Zero retention without reducing the objective to Zero alone: preserve sampled Water healing/replay and spell-opportunity/effect coverage, Lightning recovery/re-equip behavior, Earth/Fire breadth and unregistered coherent trace lines. Normal support cards count. Sample/argmax remain separate; missing opportunities are not execution failures. No winrate admission gate or automatic 1B launch. Completion requires user-triggered strategy review before the production decision.

#### Completed 50M review: distinct repertoires, no unique recipe winner

All four arms completed update 3256 with zero invalid/integrity maxima; the report contains 48/48 batches and 24,576 games. Current registration and all 48 result hashes match the report. Final Zero and all three Water probes were independently recomputed across eight hash-verified final batches (4,096 unique games), matching every eligible/completed/converted total. Review and full trajectories: `results/strategy_retention_v1/review.json`, bound to the registration and report hashes.

The earlier STRATEGIC50 Zero signature did not reproduce. At p975, the old 15M campaign had argmax 31/64 and 23/64, sample 29/64 and 19/64; the fresh 50M runs at the same update have argmax 0/64 and 1/64, sample 0/64 and 1/64. At p3250 they have argmax 0/64 and 4/64, sample 0/64 and 2/64. These remain genuine opportunities, not 0/0. This is failed replication under the extended horizon, not proof that the original learned policy forgot: the original checkpoint was never continued, and the LR horizon changed. LR causation is not established.

RANDOM50 now supplies the clearer replicated late deterministic Water repertoire. At p2275 → p3250, argmax Echo is 34/44 → 26/35 for seed43 and 21/21 → 39/41 for seed44; Healing Flutter is 81/88 → 64/67 and 47/48 → 86/88. STRATEGIC50 has neither opportunity at p2275 in either seed; at p3250 seed43 still has neither, while seed44 recovers Echo 13/14 and healing 30/34. Sampled Water persists in both recipes: final Echo is 12/30 and 12/33 for strategic, 16/28 and 16/34 for random; healing is 22/25 and 32/33 versus 38/40 and 26/26.

Do not mistake this for many independent deterministic spell strategies: RANDOM50's final positive observed spell-effect identities are only Normal Healing Flutter in both seeds. Final selected argmax identities number two in seed43 (Healing Flutter and Hook Sword Strike), one in seed44. Sampled selected identities number 18/17 for strategic and 16/19 for random; only 3/1 and 2/3 identities respectively have positive observed immediate effects. Normal support counts, deferred effects remain incompletely observed, and identity counts are not a strategy score.

Both recipes retain Stone/Defender/portal, Rushfire, Kagoro, Bobu and Shao probes across both late checkpoints/seeds/modes; these are not strategic-prefix-specific discoveries. No observed Devotion conversion occurs. Argmax Stormchain does not persist in either recipe; final strategic has 0/0 in both seeds, random 0/0 and 0/1. Sampled strategic Stormchain technically persists but only one conversion per seed/checkpoint in the final window, too sparse to call robust. Sampled Surge persists in both recipes; deterministic Surge remains seed/opportunity dependent.

Next recommendation, not launched: continue the actual prior STRATEGIC50 seed43/44 p975 checkpoints with their demonstrated Zero repertoire to a 50M-equivalent horizon, registering the continuation LR schedule and any training-state resets explicitly. Probe immediately around resume and at intermediate checkpoints; keep all-element free-draft sample/argmax evidence rather than optimizing Zero alone. This directly tests persistence of an acquired repertoire. Preserve final RANDOM50 checkpoints as a distinct Water-repertoire candidate/reference, not a scalar winner. No winrate gate, automatic recipe selection or 1B launch. Storage at review: 883 GiB available, 75% used; no models or experiment records deleted.

#### Authorized RANDOM50 continuation toward 200M

The user accepted RANDOM50 as the practical broad-repertoire candidate, not a proven universal winner. Uneven gate/leader/card balance means equal elemental coverage is not required; sparse mechanics do not automatically disqualify a recipe. This supersedes the preceding recommendation to make STRATEGIC50 Zero continuation the primary next experiment. Preserve those alternative checkpoints, but run the RANDOM50 continuation first.

Registered and launched `results/random50_continuation_v1/registration.json`, SHA256 `32a045b85ea51d7bc876e8a640642bf93c195f0f3b7a06ad98759b4eb8352633`. Seeds43/44 continue their actual p3256 full-state endpoints, not the ten-update smoke children. Each reaches p13021 / 200,002,560 cumulative configured rows (+149,990,400 rows / 9,765 updates). Both exact p3256 parents are evaluated before any training; later probes use p6511 / 100,008,960 rows and p13021. Training is serial, with each arm's later checkpoint evaluations following its training.

The original paired512-task panel is retained. A separate held-out512-task panel uses identical gate/leader/seat/opponent assignments with world seeds shifted by1,000,000,000 and distinct task IDs. Both action modes remain separate, with prefixes disabled in evaluation. Expected total: 24 batches / 12,288 games. The held-out panel tests new game seeds, not unseen opponents; reports are `strategy_report.json` and `strategy_report_heldout.json`.

Resume safety: matching model/trainer/league/promotion artifacts and referenced league checkpoints are hash-locked. Mutable league snapshots/checkpoints are copied into child-only paths. Strict actor/critic loading, optimizer moments, coordinator RNG, epoch/global step and saved environment progression are preserved; no critic reset, model-only fallback or compatibility override is permitted. Worker RNG, in-flight native games, rollout/RNN buffers and active frozen-opponent windows are not exactly resumed.

The completed parent cosine is at zero LR. This child deliberately restarts once at conservative3e-5 and decays to zero over9,765 updates, rather than silently reusing the exhausted parent scheduler or returning to the fresh .003 peak. Post-update LR is about2.25e-5 at p6511 and zero at p13021. Any recovery of this same child must restore its saved scheduler without another restart. Potential/exploration endpoints remain15,006,720 configured rows, so restored scales remain .30/.15; draft credit remains1.0. Entropy/temperature/smoothing retain their original150M→225M learner-step schedules, a different clock from configured rows.

Proof: both real parents passed ten CUDA updates to p3266, with exact final checkpoint parity, optimizer/scheduler/coordinator restore, zero invalid/integrity maxima and unchanged reward floors. All16 smoke batches /64 games completed across both panels; independent Discord attachment/stage/completion deliveries succeeded with no pending notifications. Smoke uses a ten-update cosine, so a separate real optimizer/scheduler probe exercised the full9,765-update cosine, moment preservation and same-run recovery continuity. Twelve unsafe config/provenance/panel mutations were rejected. Full registration validation passed. Evidence: `results/random50_continuation_v1/verification.json`.

Persistent processes `random50-continuation-v1` and `random50-continuation-discord` are live. The runner waits for a delivered matching independent watcher attachment before any evaluation/training; full attachment and the first parent evaluation were observed. Watcher polls60s with30-minute progress and stage/stall/exit/completion alerts; it restarts on failure. Both survive closing the assistant session. No automatic1B launch or winrate gate. Post-launch storage:862GiB available,76% used; no old models or notes deleted.

#### Completed 200M review: sampled retention, seed-dependent deterministic narrowing

Both seeds completed p13021 /200,002,560 configured rows, with 120,662,330 and 120,767,974 learner steps respectively. All recorded invalid/integrity maxima are zero. Potential/exploration scales stayed .30/.15 and entropy coefficient .01; the 150M learner-step anneal boundary was not reached. Both panels completed:24 batches /12,288 games, no incomplete games. The independent watcher's `campaign_complete` delivery is recorded with no pending events. All24 result hashes were verified; all11 strategy probes were independently recomputed from the4,096 final-checkpoint candidate traces and exactly matched eligible/completed/converted totals. Evidence, hashes, full trajectories and representative trace events: `results/random50_continuation_v1/review.json`.

The strongest conclusion is persistence, not clear replicated expansion. Sampled Surge, Echo, Healing Flutter, Shao, Bobu, Stone, Rushfire and Kagoro conversions remain nonzero at all three checkpoints in both seeds and panels. These eight named probes are not eight independent strategies or an exhaustive vocabulary. At200M, each also appears against both frozen opponent checkpoints in every sampled seed/panel batch. Earth/Fire core lines remain common under argmax, but deterministic Bobu becomes sparse: seed44 reaches0/64 paired and2/64 heldout. No observed Devotion conversion; Zero is absent except one sampled heldout seed43 conversion at each checkpoint. Those isolated observations do not establish replicated acquisition, and sparse mechanics are not a recipe rejection gate.

Seed43 shows a real deterministic Water opportunity regression. Argmax Echo converted/eligible at50M→100M→200M is26/35→37/42→0/0 paired and32/43→37/44→0/0 heldout. Water decks containing spells fall128/128→128/128→32/128 on both panels; total spell slots fall512→352→96. At200M both Echo gate/leader contexts and Hydromancy+Shao draft no spells; only Hydromancy+Benzai retains them (three Healing Flutter copies in each of32 decks). This is removed drafting support, not evidence that the policy tries and fails replay. Healing falls64/67→79/83→22/24 paired and71/72→75/82→20/20 heldout; sampled Echo/healing and both-mode Shao remain present.

Seed44 retains deterministic Echo:39/41→42/44→38/40 paired and45/46→36/38→43/47 heldout. Its200M healing is85/90 and93/95. Deterministic Lightning is also seed-dependent: seed43 drafts no weapons at any checkpoint; seed44 retains Surge recovery-to-attack (25/27 paired,26/27 heldout at200M), with Stormchain sparse (1/1 and3/4). Sampled Surge remains present in both seeds on both panels.

Do not conflate sampled spell diversity with many proven strategies. At200M, selected spell identities number19/18 for seeds43/44 paired and17/19 heldout; immediate-positive-effect identities number3/3 and2/3. Argmax selects only Normal Healing Flutter in Water contexts. Normal support counts: it supplies the retained Echo/healing line. Candidate-only raw trace examples also show Lotus of Paradise increasing available IKZ and sparse Shao's Perseverance removing an opposing Garden entity followed by face attacks, but do not establish new, broadly replicated strategy families or optimal causal value. Deferred-effect coverage remains incomplete.

Recommendation: retain RANDOM50 as the practical broad-repertoire candidate. If one200M reference is needed, seed44 has the clearer retained deterministic Water/Lightning repertoire; preserve seed43 p6511 as its pre-regression reference and both sampled endpoints. This is a repertoire-based choice, not a winrate ranking, balanced-element requirement, universal recipe winner or production qualification. New game seeds support repeatability against the same opponents, not unseen-opponent generalization. Do not automatically prefer the latest checkpoint or infer that more training monotonically broadens strategy. No1B or other training was launched by this review.

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

Historical v2 execution summary (superseded for strategy decisions): R5/R6
tested readiness/resource-only Phi and zero Phi. R7/R8 restored half/full
combat-state Phi; R9 changed the potential tail; R11/R12 changed exploration
tails. Their original decisions are preserved by content hash. Current
descriptors have been rebuilt, so old rankings are not automatically retained.

R13/R14 changed only the common potential/exploration anneal horizon to
`75%` versus `100%` of sampled rows. R14's external-anchor means remain
`0.0608`, `0.2240`, `0.3507`, and `0.4392`, and the curated panel completed
32/32 games at `0.46875`. Those strength/runtime facts do not repair its
schema-v3 no-reduction failure: sampled Fire has three converted lines at
p1950 and two at p2925, with no deterministic Zero conversion in either late
window. The final reward fallback is held for strategy requalification.

The mathematically validated R14 settings remain unchanged: discounted PBRS
with `gamma=0.99`, terminal closure, leader-health/Garden-attack/untapped-Garden/
untapped-IKZ Phi weights `2.0`/`0.35`/`0.15`/`0.15`, zero direct leader/board
deltas, zero early-tempo bonus, potential tail `0.30`, exploration tail `0.15`,
and both schedules across 100% of training rows. No new reward intervention
was introduced during validation.

Evidence:
`results/terminal_safe_reward_horizon_followup/final_decision_packet.json`.

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
5. reduced unconditional face selection when a stronger setup or control action is legal; and
6. persistence or positive slope across later checkpoints after curriculum, draft credit, and reward pressure decline.

Win rate remains useful only as a long-horizon conversion readout and as a short-run catastrophic-integrity guard. It may veto an arm at 15M only when the regression is broad enough to indicate broken learning and is accompanied by absent or degrading strategy evidence. It must not override positive strategy acquisition merely because a mature aggressive opponent wins the matchup.

Every short-run decision packet must separate `strategy evidence`, `runtime/integrity`, and `deferred strength conversion`. Historical dispositions based mainly on 15M direct or anchor win rates require strategy-first reassessment before they select or reject a longer confirmation arm.

### 5.5 Retained hypotheses and conditional follow-ups

The earlier v2 strategy-first hypotheses are preserved below as candidates, not selections. Corrected v3 observations must resolve the R14 hold before the downstream queue advances.

- **R1 remains a historical control.** The v2 balanced-parent ranking is superseded; do not replace R14 with R1 without a registered decision.
- **R3 removal is redundant under R14.** Both direct combat deltas are already zero. A matched arm with that same condition would test nothing; reintroducing edges would be a different experiment.
- **D2 and D3 both merit longer confirmation.** D2 tests whether a `0.1` exact-credit floor preserves broad sampled drafting; D3 tests whether zero late exact credit preserves more autonomous deterministic deck differentiation. Compare them over a longer late window rather than selecting from 15M strength.
- **L1 advances only behind a performance fix.** Its balanced strategy signal does not justify `61.8%` of L0 throughput. First demonstrate that per-update opponent rotation meets the registered SPS floor through checkpoint caching, refresh amortization, or equivalent inference work reuse. Do not spend a longer training run on unchanged L1.
- **S1 and S3 remain interaction hypotheses, not standalone selections.** S1's lower exploration floor and S3's zero potential tail showed partial Fire/Water signals without stable deterministic breadth. Re-test each as a single boundary on the more complete recipe after reward, draft, curriculum, and acceptable league choices are fixed.

Remaining follow-ups stay one-boundary experiments. Do not combine D3, a league performance rewrite, and an S1/S3 tail; that would erase attribution.

### 5.6 Lightning acquisition sequence

The v1 `lightning.weapon_recovery_portal_attack` readout is invalid for both Lightning gates:

- its `eligible` denominator counts any deck with a legal weapon attachment, including non-Lightning gates;
- for Surge (`STT01-002`), it requires a weapon to be attached, discarded, returned to hand, and manually re-equipped *before* a later portal, although Surge performs discard recovery and equip as part of the portal resolution; and
- it rejects Stormchain (`AZK01-120`) outright even though Stormchain re-equips a Garden weapon as part of its portal resolution.

Consequently, the reported zero does not measure Lightning strategy absence. Existing traces already contain `SELECT_TO_EQUIP` actions sourced by both Lightning gates. Mark every prior Lightning coverage conclusion unresolved until retained traces are reprocessed with corrected gate-specific semantics.

Replace the single chain with two cumulative stage funnels:

1. **Surge recovery:** a legal Surge portal with an eligible discard weapon -> selected/resolved portal -> immediate `SELECT_TO_EQUIP` for that discarded weapon sourced by Surge -> attack/effect by the equipped destination -> realized effective outcome. Prior manual attachment and subsequent discard are an optional provenance branch, because Surge may validly recover a milled or cost-discarded weapon.
2. **Stormchain re-equip:** a legal Stormchain portal while an eligible weapon is equipped in the Garden -> selected/resolved portal -> immediate `SELECT_TO_EQUIP` for that weapon sourced by Stormchain -> attack/effect by the new destination -> realized effective outcome. Prior manual attachment is supporting provenance, not required eligibility.

Preserve gate, weapon code, source target, destination target, and portaled entity at every stage. Bound the equip match to the immediate ability decision so a later portal cannot complete an earlier whiff. Current trace schema v2 identifies card codes but not physical copies, so duplicate-card provenance remains an explicit limitation. Gate-specific legal portal opportunities, not generic weapon legality, define each denominator.

Rebuild every retained descriptor and the strategy-first/canary decision artifacts before choosing an intervention. Only if the corrected funnels still show a real downstream acquisition failure should the intervention queue proceed:

1. **Focused battle exposure:** use legal fixed Lightning decks and scenarios that expose the failed Surge or Stormchain stage. Mask draft actor credit on fixed-deck seats, anneal exposure away, and evaluate with exposure disabled.
2. **Additive portal outcome:** retain a bounded flat discovery reward for a legal GP portal and add a separate smaller kicker when the gate effect observably resolves. Persist portal provenance across the later `SELECT_TO_EQUIP` decision and pay or cancel the kicker when that selection resolves or is skipped; same-`c_step` state differencing cannot observe Lightning's deferred attachment.
3. **Weapon realization:** in a separate arm, reward a recovered/re-equipped weapon only when its attributable effect or effective non-overkill damage is realized. Do not reward raw attachment.
4. **Combination:** combine exposure with one qualified reward only after each independent arm produces repeated downstream Lightning stages without collapsing another element.

At 15M, advance on repeated opportunity-normalized stage progress across two later windows, sampled and deterministic presence, persistence after exposure/reward annealing, safe reward-component magnitude, and no other-element collapse. Direct win rate remains deferred conversion evidence.

Corrected retained-trace result:

| Trace mode | Gate-specific opportunities | Completed through destination attack | Resolved attack | Windows with opportunity / completion |
|---|---:|---:|---:|---:|
| sampled | 789 | 750 (95.1%) | 736 (93.3%) | 63 / 63 |
| deterministic | 358 | 338 (94.4%) | 332 (92.7%) | 35 / 35 |

Surge supplies most evidence: sampled `733/768` completions and deterministic `315/335`. Stormchain is sparse but present: sampled `17/21` and deterministic `23/23`. R1 itself retains corrected Lightning at all sampled windows; at p975 it completes `22/27` sampled opportunities and `6/9` deterministic opportunities. The seven 200-game production-baseline traces create zero corrected gate-specific opportunities, so their zero remains an exposure/state-creation result rather than a failed-chain denominator.

**Decision:** stop the intervention queue before curriculum or reward changes. Corrected Lightning acquisition is already strong whenever a valid portal state exists, so additional shaping would reward an existing behavior and add confounding. Resume at the first failed stage only if the formal recipe repeatedly creates gate-specific opportunities but stops selecting the portal, resolving the immediate equip, or attacking with the destination. Stormchain exposure may justify a measurement-only panel because its opportunity count is low; it does not yet justify reward.

Artifact rebuild order is mandatory after descriptor semantics change: run `regenerate_strategy_descriptors.py`, rebuild `strategy_baseline_v1/baseline_packet.json`, run `refresh_strategy_decision_packets.py`, then rebuild `strategy_first_reassessment.json` and the canary registration. The packet refresher replaces every embedded descriptor ID, sequence summary, stage funnel, endpoint element count, and opportunity-status field from schema-v2 descriptors; prose-only packet edits are invalid.

## 6. Fresh staged 1B production candidate with a 100M canary gate

Operational status, completed-result roll-up, and the gate-by-gate execution checklist are maintained in `train-ablation-1781126582/local-production-run.md` under `Current successor qualification status`. This section remains authoritative for the preregistered experiment contract.

Start one candidate from random initialization with the final 1B horizon and every schedule frozen before row zero. The 100M canary is a stop/go gate on that same trajectory, not a separate short run. A passing candidate resumes from its exact checkpoint, optimizer, RNG, league, and schedule state; it does not restart from an inherited policy or reset any anneal.

### 6.1 Composition

The staged candidate combines only qualified changes from Stages 0-5:

- corrected reward telemetry and proper terminal closure;
- selected potential/exploration formula and schedules;
- selected exact-draft-credit schedule;
- cross-gate exact-credit exclusion, with replay retained only if its control passed;
- selected strategic exposure curriculum;
- selected league window, role sampler, retention, and frozen ratio; and
- unsaturated strength plus strategy validation.

No new architecture, gate-ID channel, element-specialist model, or additional reward term enters the staged run.

### 6.2 Horizon and gates

Configure `1,000,000,000` sampled rows before launch and express reward, draft-credit, curriculum, policy-temperature, smoothing, entropy, and league schedules in absolute sampled rows. No schedule may derive from a temporary 100M horizon. This matters because the current candidate template places policy-temperature, smoothing, and entropy transitions at 150M-225M; a standalone 100M run would stop before them and a restarted 1B run would reset its earlier schedules.

Write diagnostic packets at approximately 5M, 15M, and 30M, then reuse the prior 1B production gates at 50M, 100M, 200M, 300M, 450M, 800M, and 1B. Use dense checkpoints around every gate and preserve every late window.

- **5M:** mechanics and opportunity discovery; no strength promotion.
- **15M:** element coverage and reward/draft/league telemetry integrity.
- **30M:** strategy persistence after early curriculum/reward pressure begins to fall.
- **50M:** integrity, resume, learning-health, strategy trajectory, and quick external check.
- **100M:** formal canary gate for strategy, strength conversion, forgetting, exposure diversity, and runtime.
- **200M:** first formal post-canary continue/stop decision and schedule-transition check.
- **300M:** full robustness, approximate-exploitability, and blind human-review gate.
- **450M:** main persistence and production-funding gate.
- **800M:** late-learning, forgetting, polish, and checkpoint-selection assessment.
- **1B:** final production qualification.

Every formal gate from 50M onward runs the previous production strength and robustness panels plus the descriptor-v3 battery: sampled and deterministic per-context element funnels, Water resource/spell conversion, Earth defense/healing, Fire self-damage conversion, Lightning weapon attacks, sibling deck/action differentiation, conditional face-versus-entity choices, reward-component magnitude, anneal persistence, opponent role/temporal exposure, and change versus earlier gates. Unmeasured causal values and deferred effects must remain explicit.

Do not stop merely because strategic play initially loses to R0/p44000. Stop for integrity failure, structural collapse across repeated windows, no strategy acquisition despite real opportunities, or a broad strength regression with no improving strategy trajectory.

### 6.3 The 100M canary gate

Advance only if all hold:

- Lightning weapons and Water spells are drafted and converted in opportunity-normalized traces; neither deterministic family is structurally absent.
- Earth and Fire retain multiple valid lines rather than carrying the pooled result alone.
- Sibling decks/actions differ where the fixed-deck interaction probes establish different value.
- Face attacks remain efficient when correct but no longer dominate decisions that have stronger legal setup/control alternatives.
- Strategy support persists after reward/draft/curriculum anneals.
- The 50M-100M window improves against cheap-policy ancestors or shows a statistically credible positive conversion slope while respecting the strength floor.
- The actual opponent-exposure log demonstrates role and temporal diversity; configured pool size alone is insufficient.
- Quality-adjusted SPS is acceptable for the intended hardware and no integrity guard regresses.

Each passing formal gate authorizes only the next segment of the already configured trajectory: 50M releases training to the 100M canary gate, which releases 200M, then 300M, 450M, 800M, and finally 1B. Resume the exact checkpoint, optimizer, RNG, league, and schedule state every time. Do not change rewards, schedules, optimizer settings, model architecture, opponent-pool semantics, or selected recipe at any gate; any such change requires a fresh row-zero candidate. Gate evaluation must be side-effect-free with respect to training state.

The 450M gate on the same trajectory proves persistence, not independent-seed reproducibility. If independent training-seed evidence is required, run a separately registered confirmation replicate; never describe the continued trajectory as an independent replicate.

### 6.4 Runtime envelope

For `100,008,960` sampled rows, training-only time on the current RTX 3090 is:

| Throughput basis | SPS | 100M training time | 1B training time |
|---|---:|---:|---:|
| S0 shaping median | 1,752.3 | 15.9 hours | 6.6 days |
| Prior production rolling median | 1,428.8 | 19.4 hours | 8.1 days |
| L0 league median | 1,385.4 | 20.1 hours | 8.4 days |
| Proposed 90%-of-L0 league floor | 1,246.8 | 22.3 hours | 9.3 days |
| Unoptimized L1 | 855.6 | 32.5 hours | 13.5 days |

These figures exclude paused gate evaluations. Budget the 100M gate as approximately 16-22 training hours for a qualified recipe; do not accept unchanged L1 throughput.

# Experiment ledger

Update this table when an arm is registered, launched, completed, or rejected. Every row must link its config, parent manifest, result artifact, and decision report.

| ID | Status | Parent | Configured rows | Single changed boundary | Primary result | Decision artifact |
|---|---|---|---:|---|---|---|
| BASE-1B | Complete / rejected | Fresh production run | 1,000,000,000 | Current recipe | Healthy run; strategy-learning flunk | `results/corrected_production_1b_lr1500_fresh/gates/gate_1b_summary.json` |
| EVAL-0 | Revalidated at descriptor v3; attribution limits explicit | N/A | N/A | Evaluator correctness | 218 ablation + seven baseline descriptors rebuilt; 21-pair sensitivity and 19 CPU regression tests pass | `results/strategy_baseline_v1/baseline_packet.json` |
| R0-R4 | Historical screens complete; rankings superseded | Fresh seed 42 | 15,006,720 each | Reward formula with D0 credit | Recomputed v3 observations; R3 removal already present in R14 | `results/reward_screens/decision_packet.json` |
| R5-R14 | Reward math valid; R14 strategy hold | R1 parent with proper PBRS/closure | 15,006,720 screens; 45,004,800 qualifiers | Registered Phi/tail/horizon brackets | Corrected sampled Fire converted breadth drops 3 -> 2; deterministic Hydromancy has no spells; no downstream advance | `results/terminal_safe_reward_horizon_followup/final_decision_packet.json` |
| D0-D3 | Screens complete; R14 45M confirmations blocked | R1 reward recipe | 15,006,720 each | Draft-credit schedule/correctness | No R14 long confirmation or replay-off evidence; v2 ranking superseded | `results/draft_screens/decision_packet.json` |
| CURRICULUM | Screens complete; qualification blocked | Old D2 recipe | 15,006,720 each | Entity-only versus strategic fixed decks | Must compare selected-parent no exposure/entity-only/strategic exposure with curriculum off in evaluation | `results/curriculum_screens/decision_packet.json` |
| L0-L5 | Screens complete; qualification blocked | Old C1 recipe | 15,006,720 each | League sampling/retention | L1-L4 below SPS floor; L2-L4 completed distinct-role share zero; no qualified R14 league | `results/league_screens/decision_packet.json` |
| S0-S3 | Screens complete; final interactions blocked | Old L5 recipe | 15,006,720 each | Global-row shaping tails | S1/S3 remain separate hypotheses after other boundaries freeze | `results/shaping_screens/decision_packet.json` |
| S4 | Historical screen; strategy disposition superseded | Old S0 recipe | 15,006,720 | PBRS terminal closure | Runtime evidence retained; v3 observations replace old sequence conclusions | `results/terminal_closure_screen/decision_packet.json` |
| STRATEGY-REASSESS | Rebuilt; rankings withheld | EVAL-0 and R0-S4 | N/A | Correct effect/recovery/timing semantics | Per-context elemental evidence; no short-run win-rate ranker | `results/strategy_first_reassessment.json` |
| STAGED-PRODUCTION-1B | NO-GO | Fresh random init only after qualification | 1B planned | Frozen full-horizon trajectory | Recipe hold plus unhealthy GPU runtime; no launch | `local-production-run.md`, current successor status |

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
10. Reprocess all retained descriptors with corrected Surge and Stormchain stage funnels.
11. **Held after v3 revalidation:** R14 reward math passes, but corrected sampled Fire violates the registered no-reduction strategy gate. Resolve that evidence before any downstream arm; do not invent an unregistered fallback.
12. Once the hold is resolved, run longer matched D2/D3 confirmations and the replay-off control under immutable qualified rewards.
13. Settle strategic exposure, then include L1 only if its implementation first recovers acceptable SPS and proves actual opponent-role diversity.
14. **Resolved as redundant:** R14 already removes both direct combat-edge deltas; no R3 removal arm remains.
15. Test S1 and S3 separately as final interaction boundaries; drop redundant arms when the selected reward recipe already implements the same condition.
16. Only after all qualifications pass and the GPU runtime is healthy, freeze a fresh 1B schedule. Early diagnostics and formal production gates must consume current descriptor-v3 elemental evidence. Resume the exact same trajectory only after each gate passes; no 1B launch is authorized by this revalidation.
