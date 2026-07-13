# Azuki TCG — Deck-Building & Training Ablation Research Notes (Part 01)

> ## LIVE STATE (update on every major transition)
> As of 2026-07-06 LATE (post-round-2 phase; goal: 45M confirmation or pivot):
> - CRITIC-SENSITIVITY PROBE (the go/no-go for 45M): probe_critic_gate.py NEW —
>   interventional gate-swap (reuses probe_gate_kl machinery) reading the VALUE
>   head + win-prob aux head instead of pick logits. Per element pair: |dV| at
>   P0 pick steps + battle start, A-A control replay floor (measured EXACTLY 0,
>   bitwise-deterministic on cpu), cross-seed V-std scale reference, sign
>   consistency. RUNNING on all 4 round-2 final ckpts (ep977), 12 eps, 4-way
>   parallel cpu (run_critic_probe.sh; results/critic_probe_<arm>.json).
>   Smoke (ctrl2, LIGHTNING, 2 eps): battle|dV|=0.0007 vs floor 0 — nonzero but
>   tiny; scale interpretation needs the full run (battle V cross-seed std was
>   ~0.001 at n=2 — suspiciously flat, added Vmean/withinEpStd readouts).
> - SIBLING-MATCHUP OVERSAMPLING KNOB SHIPPED (commit 7e316c0):
>   env.draft_same_element_matchup_prob — with prob p, P1's gate is replaced by
>   the same-element partner of P0's. Native C path (per-env field; catalog
>   carries a sibling map) + legacy wrapper (same_element_matchup_prob).
>   prob 0 = bit-identical RNG streams. Tests: test_draft_sibling_oversampling.py
>   (3) + draft parity + deck-building suite all green. NOTE:
>   AZK_DEBUG_FORCE_GATE_DEF_IDS is now re-read per episode (was latched).
> - CRITIC PROBE VERDICT (2026-07-06 23:20): **GO — the critic is NOT gate-blind.**
>   12 eps × 4 elements/arm, control floor EXACTLY 0 everywhere. combo1: battle
>   |dV| sign-consistent 12/12 in ALL FOUR elements (chance ~5e-4/element),
>   |dV|/cross-seed-V-std ratios WATER 0.59, EARTH 0.35, FIRE 0.18, LIGHTNING
>   0.14 (mean 0.30); win-prob head agrees (WATER |dWP| ~45% of spread).
>   Arm ordering: combo1 0.30 ≫ gateid1 0.08 ≈ anneal1 0.09 > ctrl2 0.02 —
>   gateid channel + anneal TOGETHER are what let the value function learn the
>   sibling distinction. Actor KL ≡ 0 at the same ckpts ⇒ value knows, policy
>   hasn't cashed it in ⇒ exactly the longer+sharper-gradient regime.
>   Artifacts: results/critic_probe_<arm>.json; analyze_critic_probe.py.
>   NOTE anneal-arm battle-V cross-seed std is ~10× smaller than ctrl2's
>   (0.002-0.006 vs 0.024) — outcome-dominated training flattens battle-start
>   value spread; ratios computed within-arm.
> - INCIDENT (2026-07-07 00:57-04:15): combo45 hit an ENGINE invalid-action at
>   8.8M steps (battle tick 83, stale-vs-fresh mask desync, 30-card hand state;
>   repro line now logged w/ episode seed+gates) → C abort() → dead worker →
>   vecenv deadlock (2h hang). Fixes shipped: (1) invalid action now TRUNCATES
>   the episode (zero-legal-style) instead of aborting; AZK_INVALID_ACTION_ABORT=1
>   restores abort for debugging. (2) resume unblockers: AZK_RESUME_ALLOW_BINDING_MISMATCH,
>   AZK_RESUME_ALLOW_SOURCE_DRIFT (source_hashes.* only), AZK_RESUME_KEEP_CURRENT_SCHEDULE_ENV
>   (also excuses schedule_env.* fingerprint fields). (3) resume reset-start
>   probe SKIPPED on native deck-building (its mask path sends NOOP into the
>   draft abort — killed the first resume attempt). (4) resume restarts env
>   episode counters (no env progression saved) → resumed leg pins the anneal
>   TAIL via env vars (0.08→0.05 over 5 per-env episodes; measured 0.083 at crash).
>   OPEN WORK ITEM: root-cause the engine mask desync (grep logs for
>   "Invalid-action truncation:" to collect repro seeds) — must be fixed before
>   the production distributed run.
>   Recovery order: anneal45 (fresh, launched 03:36 by the intermediate chain)
>   runs FIRST; run_after_anneal45.sh (chain6) waits for it → draftref →
>   combo45 resume from ep600/8.43M → draftref → both trajectories.
> - **ACHIEVABLE-GAP PROBE, LIGHTNING (2026-07-08 13:45): THE SIBLING GAP IS
>   REAL (~4-5pp)** — mirror decks (identical leader+50 mains, only the gate
>   differs), n=1032/arm: policy-argmax Surge 54.1% (p~.009), portal-forced
>   55.2% (p<.001), portal-blocked 48.1% (CI incl 50 — removing portals
>   equalizes the siblings, as designed). The policy ALREADY exploits the
>   difference in play (~1.4 portals/ep argmax) but not in draft (KL≡0).
>   Clears the pre-registered >=3pp bar → training levers activated.
>   WATER confirms (15:56): Hydromancy vs EchoedWaves policy 55.5% (p<.001),
>   forced 53.7%, blocked 50.6% (~50). Policy > forced here — selective
>   portaling (~2.3/ep) beats portal-spam; the policy is gate-aware IN PLAY.
> - **FULL SIBLING MATRIX (22:57, n=1032/arm each): every pair has a real
>   portal-driven gap; blocked ≈ 50% in all four (clean nulls).**
>   FIRE: policy 55.3 / forced **62.7** / blocked 48.2 — Rushfire crushes
>   Ragefire when portals are maximized; the policy leaves ~7pp unexploited
>   (should portal-spam under Rushfire). EARTH: policy 45.8 / forced 46.6 /
>   blocked 49.3 — sign flips: Stonehaven (defender-grant) > Devotion.
>   OPTIMAL PORTAL STYLE IS GATE-SPECIFIC (spam Rushfire, selective
>   Hydromancy) — exactly the per-gate strategy axis we want drafted+played.
> - **CROSS LADDER partial (neutral all-NORMAL decks vs Hydromancy ref)**:
>   Surge 45.9-47.3, Stormchain 46.7, EchoedWaves 43.0 (blocked ≈ 50.6) —
>   on synergy-starved decks the deck-agnostic ramp gate (Hydromancy) beats
>   deck-dependent gates ⇒ gate-ability value is strongly DECK-DEPENDENT —
>   the very signal a gate-aware drafter should exploit. Ladder completes
>   overnight (~04:00).
> - **portalgp1 smoke @15M: draftref 46.9%** (96 eps) — best 15M arm yet
>   (combo1 43.8, anneal1 44.4), external quality did NOT crater with 2.5×
>   portal exposure (0.076 vs 0.030 action share @3-5M). KL/critic probes on
>   its final ckpt running (results/run15_portalgp1/).
>   Tooling: probe_gate_gap.py (+ cross-element neutral-deck mode: all-NORMAL
>   mirror decks, same leader both sides — engine accepts gate/leader element
>   mismatch); run_gate_gap_all.sh sequencing W/F/E siblings then a 7-gate
>   ability ladder vs Hydromancy ref.
> - **PORTAL-GP SHAPING SHIPPED (c804922)**: AZK_PORTAL_GP_BONUS — on
>   GATE_PORTAL, weight*min(GP,4)/4 of the portaled entity (pre-tick alley
>   lookup via azk_card_def_from_id) joins base_shaped_reward → rides the
>   shaping anneal + zero-sum channel; 0-GP portals earn 0; default off.
>   Differential test green (knob-off bit-identical). portalgp1 smoke arm
>   (15M, recipe + bonus 0.3) launched 13:55 on the idle GPU alongside the
>   CPU probe matrix. Gates a 45M portal-GP run.
> - **45M RESULTS IN (2026-07-08 morning)**: combo45b (fresh full-recipe rerun)
>   COMPLETE 02:45, rc=0, healthy. Draft-vs-ref: **combo45b 41.1%** vs
>   **anneal45 46.4%** (192 eps each; recipe ~1σ below control externally,
>   consistent with round-2 combo1 43.8 vs anneal1 44.4).
> - **E1 ANSWERED AT 45M: actor gate-swap KL ≡ 0 at EVERY checkpoint in BOTH
>   arms** (30 ckpts each, 1e-5 resolution) — 45M steps + id channel + 43%
>   sibling matchups + eps 0.02 do NOT create pick-policy gate conditioning.
> - **BUT the critic trajectory (combo45b) shows persistent, GROWING value-side
>   sensitivity**: battle-start |dV| 0.0016@1.5M → peak 0.0054@32M, |dWP|
>   trend 0.0003→0.0011 by 45M, sign-consistency ~100% late, sens-ratio
>   0.3-1.2. The critic knows siblings differ; the actor never cashes it in.
>   STRUCTURAL CONCLUSION: the sibling residual (~0.002-0.005 V units) is
>   below the pick-head policy-gradient noise floor under PPO+entropy+eps —
>   sharpening exposure (oversampling) grew the critic signal but not the
>   policy. Next lever is NOT more steps: it's (a) portal-EV / achievable-gap
>   analysis — is the intrinsic sibling value gap big enough that a policy
>   SHOULD condition? (game-design question; if gap ≈ 0, E1 was mis-specified
>   and family-level conditioning is the correct optimum), and (b)
>   A-PRIVCRITIC for generally sharper pick baselines. Chain still finishing
>   anneal45 trajectories (~15:30).
> - COMBO45-RESUMED INVALIDATED (19:30): draft-vs-ref cratered to 26.6%
>   (anneal45: 46.4%; combo1@15M: 43.8%). Cause: the model-only resume — fresh
>   optimizer + restarted lr schedule (peak ~3e-3 on converged weights) →
>   POLICY ENTROPY COLLAPSE (losses/entropy 0.88 early-leg → 0.077 end;
>   anneal45 same infra no-resume: 0.74 healthy; ep_len 128→71 degenerate).
>   The 26.6% is attributable to resume damage, NOT the oversampling/eps
>   recipe. LESSON: never train on a model-only resume with a restarted step
>   counter; a30d373 makes the full restore work under the excusal flags.
>   combo45b relaunched FRESH 19:29 (run_combo45b.sh, chain7): same recipe,
>   45M from scratch, crash-truncation live. ETA train ~01:30, draftref
>   ~01:50, then trajectories (combo45b + anneal45 only; resumed-combo45
>   trajectories skipped as invalid).
> - anneal45 @45M draftref: **46.4%** pooled 192 eps (seat0 44.8/seat1 47.9) —
>   vs 44.4% @15M: external quality HOLDS at 3× length (June's decline stays
>   reversed).
> - RECOVERY STATUS (11:20): anneal45 COMPLETE (45M, final ckpt ep2930,
>   draftref done 10:45-11:08). combo45 resumed 11:08 CLEANLY (weights loaded,
>   entropy 1.94, shaping pinned 0.08→0.05, sampler anneal offset 8.43M) BUT
>   a second fingerprint check in the trainer-state restore silently downgraded
>   to MODEL-ONLY resume → optimizer fresh + global_step restarted at 0.
>   CONSEQUENCE: combo45-resumed trains 45M NEW steps on top of the 8.43M
>   checkpoint = 53.4M total experience (ETA ~16:30). Treat as model-carryover
>   run; for matched-experience comparison vs anneal45@45M use the resumed
>   run's ~ep2380 checkpoint (8.43M + 36.6M ≈ 45M). Fixed for future resumes
>   (a30d373: excusal flags now cover the optimizer-restore check too).
>   ANALYSIS CAVEAT: resumed combo45 writes a NEW run dir with epochs from 0 —
>   trajectory JSON names collide with old-dir ep100-600 (trajectory() skips
>   existing files, so new-run ep100-600 will be MISSING from the chain's
>   sweep); rerun trajectories per-dir with distinct prefixes afterwards.
> - 45M CHAIN LAUNCHED 2026-07-06 23:33 (run_45m.sh, detached): combo45
>   (anneal + gateid + pick-eps 0.02 + oversample 0.35 + league 6/4/3, seed 42)
>   → draftref 192 → anneal45 control → draftref → per-ckpt gate-KL + critic
>   trajectories. Logs /tmp/run45m_chain.log, /tmp/train_combo45.log. ETA
>   ~14h. Smoke-validated first: pick-eps 0.02 applied (sampler banner), CLI→
>   env plumbing returns 0.35 float, knob unit tests green.
>
> Previous LIVE STATE (round-2, kept for context):
> As of 2026-07-06 (research resumed on branch skylark/C-train-optimizations):
> - STACK CHANGE: training stack optimized 2026-07-05/06 (native obs path for fixed-deck,
>   vectorized encode, metadata embedding table, GPU ScalarRunningNorm, CUDA-graph rollout,
>   train-side compile). Deck-building runs stay on the legacy PettingZoo path but inherit the
>   policy/trainer speedups: **measured ~1,820-1,920 SPS at pool 0 (vs 515 in June) — 3.5×**.
>   CAVEAT: June checkpoints were trained on the pre-fix slot-scrambled encoder; probing them
>   through the fixed encoder shifts their input distribution. Cross-stack comparisons carry an
>   asterisk; round-2 arms compare only against round-2 arms (fresh ctrl2 control arm).
> - GAMMA1 ANALYSIS DONE (was unfinished): §3.10. Verdict: γ=1.0 = faster early pick commitment
>   (quad spike 7.5 vs 1.2) but same element-level attractor; NO same-element gate divergence;
>   draft-vs-ref ~36% (≤ baseline 45.8% @11.5M, CIs overlap); higher value loss (0.041 vs 0.032);
>   portal/weapon usage collapsed by 12M. NOT the lever alone.
> - CENTRAL NEGATIVE RESULT (§3.10): in BOTH baseline and gamma1, same-element gate pairs draft
>   statistically IDENTICAL decks (L1 divergence at bootstrap noise floor; type shares match to
>   3 decimals) — the model conditions on ELEMENT (availability pool), not on the GATE CARD.
>   Gate mechanics are unused in play (portal ~0.03, weapons ~0.02) and uncorrelated-or-negative
>   with winning → no gradient pressure for gate-conditional strategy. Equilibrium to break.
> - NATIVE PORT LANDED (§6.1, commits a13d66e/c4653eb): deck-building drafts in C,
>   ~3.2-3.9k SPS with league (6× June). Parity-tested. League game-granularity +
>   eval-layout bugs fixed (affected battle-only native too).
> - ROUND 2 (running on NATIVE config): §5. Arms 15M steps each, seed 42, ckpt interval
>   100, own league dirs (keep 2/1/1): ctrl2 → anneal1 (ANNEAL warmup 12 / ramp 40
>   per-env episodes @480 games ≈ fade 1.9M→8.2M) → gateid1
>   (policy.gate_id_embedding_enabled) → combo1 (anneal+gateid+pick-eps 0.05).
>   Chain: run_round2.sh; watchdog monitor active; analyze_arm.sh per arm afterwards.
> - entdeck knobs cherry-picked to this branch (3e4f0f7): policy.deck_pick_smoothing_eps,
>   policy.legal_row_temperature. Functional check: pick-row entropy 0.196→1.381 @ eps 0.5,
>   battle rows unaffected.
> - Old artifacts: baseline checkpoints ep250-1500 (two run dirs), abl-gamma1 ep100-521,
>   snapshots in experiments/abl_snapshots/{base-deckbuild-02,abl-gamma1}. Stage dumps
>   experiments/stage_dumps: only stage250 partially completed (1 pid file) — abandoned.
> - New tools in this dir: traj_compare.py (runlog trajectories with environment/{0,1}
>   agent-averaging), gate_identity_probe.py (same-element L1 vs bootstrap noise floor,
>   behavior→win correlations, leader splits).

Started: 2026-06-10. Machine: RTX 3090 (24GB), 128GB RAM, 24 cores.
Branch: `skylark/model-deck-building` (ablations get their own branches off this one).
Goal: (1) long training runs verifying model-driven deck building + combat improvement, tracked
across the 8 gates; (2) ablations (model size, league configs, input preprocessing, etc.) on how
the model converges to building decks and playing; final report at the end in this directory.

## 1. Current-state survey (2026-06-10)

### Environment / deck building
- `python/src/deck_building.py` — `DeckBuildingParallelEnv` wraps `AzukiTCGParallel`.
  Each episode: each player gets a **random gate** (sampled from the 16-deck pool's gate population,
  so all 8 gates appear, weighted by pool frequency — NOTE: pool has 16 decks, gate frequencies may
  be non-uniform). P0 picks leader (from 2 element-matched leaders), then 30 main cards
  (sequentially, ≤4 copies each, element ∈ {gate element, NORMAL}); then P1; then battle starts via
  `reset_with_decks`. Deck picks are extra env steps with `DECK_PICK_CARD` actions (62 picks/episode).
- Card pool per element: LIGHTNING 29+46 NORMAL, WATER 32+46, EARTH 34+46, FIRE 34+46 candidates
  (entities/spells/weapons; max candidates ~80).
- Tests: `python/tests/test_deck_building.py` — 6 passed (2026-06-10).

### The 8 gates and expected archetypes (from effect text)
| Gate | Element | Effect summary | Expected archetype signal |
|---|---|---|---|
| STT01-002 Surge | LIGHTNING | Portal + replay weapon from discard | weapon-heavy tempo |
| AZK01-120 Stormchain | LIGHTNING | Portal + re-equip weapon | weapon redistribution |
| STT02-002 Hydromancy | WATER | Portal + untap IKZ | ramp; cost-greedy curves |
| AZK01-126 Echoed Waves | WATER | Portal + return spell from discard | spell-heavy control |
| AZK01-122 Rushfire | FIRE | Portal + play extra entity w/ Charge, then sacrifice | cheap entity aggro |
| STT04-002 Ragefire | FIRE | Portal + buff damaged entity's ATK | aggressive combat |
| AZK01-124 Devotion | EARTH | Portal + sacrifice → deal damage | sacrifice/removal |
| STT03-002 Stonehaven | EARTH | Portal + grant Defender | defensive walls |

Leaders (2 per element) also push styles: Raizan/Piko (weapons), Shao (atk debuff response),
Benzai (discard-cost reduction), Kagoro (entity-flood pump), Zero (self-damage combo),
Goro (HP buff), Bobu (death-heal).

### Model (v2, `python/src/policy/v2/tcg_policy.py`)
- Card repr: metadata lookup (type/element/cost/stats/keywords + OpenAI text embeddings of
  name/effect/subtypes) → 48-dim projected metadata embedding. NO learned per-card-ID embedding.
- Zone encoders → 15×64 pooled groups → 960-d LSTM input → LSTM hidden 4096 → legal-action scorer
  head (64-d query · candidate embeddings, max 1024 candidates).
- Deck-building support: `deck_context` obs (mode, gate, leader, partial main deck, candidate list)
  encoded as 15th zone component; candidates scored by the same legal-action scorer
  (`DECK_PICK_CARD` with sub1 = candidate index).
- Critic: `full_lstm_mlp` head; optional privileged critic (opponent hand + both decks via
  transformer deck encoder); win-prob aux head (coef 0.05); optional split value heads
  (terminal vs shaped).
- Key size constants: `UNIT_EMBED_SIZE=64`, `CARD_METADATA_EMBED_SIZE=48`,
  `PROCESS_SET_HIDDEN_SIZE=128`, `LSTM_HIDDEN_SIZE=4096`, `CRITIC_MLP_HIDDEN_SIZE=512`.

### Training stack
- PufferLib v4 fork in-repo (`python/src/azk_puffer/`), PuffeRL PPO + LSTM (bptt_horizon 16),
  multiprocessing vecenv. `MultiagentEpisodeStats` sums per-episode info values at termination;
  trainer averages across episodes → so `X/win` ÷ `X/game` pairs give per-bucket win rates.
- Tuned 3090 config: `python/config/azuki_speed_3090_parallel.ini` — 720 envs, 12 workers,
  direct_parallel, league enabled (frozen_ratio 0.10), lr 3e-3, bf16, 450M steps,
  ent_coef 0.01→0.002 anneal @ 150M-225M, temp 1.2→1.05, smoothing 0.05→0.01.
- League: `LeaguePuffeRL` — row-level frozen opponents, gating/promotion via `league_manager.py`
  (champion + baselines, Wilson bound), state in `experiments/league/main_v1/`.
- Reward: potential-based shaping (leader HP edge w=4.0 dominant, garden atk 0.7, untapped 0.15/0.15)
  + raw deltas (leader 1.25, board 0.35), NOOP penalty 0.02, win ±5, truncation penalties; optional
  shaping anneal via `AZK_REWARD_SHAPING_ANNEAL` env vars. Deck-pick steps give 0 reward (only
  outcome credit flows back through LSTM/value).
- Episode tick curriculum via `AZK_MAX_TICKS_CURRICULUM*` env vars.
- Logging: wandb (`--wandb` flag), checkpoints + meta.json in `experiments/<run>/`.

### Existing deck-building metrics (added in commit 716b37f)
- `deckbuild/*` at battle start: copy-count entropy/histogram, unique count, type/element shares,
  per-ELEMENT gate/leader indicators. **Aggregated per element (4), not per gate card (8).**
- `deckbuild_result/*` at episode end: win rates per gate element, gate pair (ordered/unordered),
  gate match/mismatch. Again element-level only.
- GAP: no per-gate-card (8) metrics, no cost-curve metrics, no weapon/spell-share-by-gate metrics,
  no qualitative deck dumps. → instrumentation work item #1.

### Prior experiments (before this effort)
- `experiments/` has ~150 short runs: SPS autoresearch (3090 tuning → ~3.5-4k SPS), torch profiling,
  `azuki_ab3_auxon/auxoff` (win-prob aux on/off, 2 seeds), retune experiments (combo scale,
  fusionwide, compactdeck), privileged critic ablation dir.
- League state `experiments/league/main_v1/` exists from pre-deck-building era (fixed-deck training).
- No long run with deck_building_enabled=true yet (feature landed 2026-06-02, after last runs 04-21).

## 2. Work log

### 2026-06-10: setup
- Created this directory; tasks tracked in session: survey ✅, research dir ✅(this file), per-gate
  metrics, baseline long run, ablations, archetype analysis, final report.
- Next: download papers, add per-gate instrumentation, smoke-test deck building at scale, launch
  baseline.

### 2026-06-10: literature pass (papers/ + *summary.md files)
Library: OpenAI Five, Hearthstone ByteRL (2303.05197), LOCM ByteRL+OSFP (2303.04096), privileged
critic (2509.26000), Suphx (real id 2003.13590 — the 1912.00126 pdf is a mislabeled math paper),
DouZero, DeepNash, Vieira CCG drafting, Beat-ByteRL exploitability. Key actionable findings:
- **E2E draft+battle in one episode beats alternating training** (ByteRL: 81% vs 68%) — our design
  is right; verify battle reward credit actually reaches deck picks through LSTM/value bootstrap.
- **γ=1.0 for terminal-reward episodes** gave Hearthstone ByteRL +7% alone. We run γ=0.99 with
  heavy shaping — ablation candidate A-GAMMA.
- **Leader-pick collapse warning**: putting hero pick inside the E2E policy collapsed to one hero
  despite entropy. Mitigation candidates: uniform-random leader during training (condition, don't
  choose), or leader-pick entropy boost. Watch `deckbuild_gatecard/*/leader/*` for collapse.
- **Forced random draft picks** (Random-CB): n∈{0,1,2,4} random picks per draft (p=.5/.25/.125/.125)
  → deck diversity at no winrate cost. Ablation A-RANDPICK.
- **IS ratio clipped from below** ([1e-3, 1.007] in ByteRL) prevented vanishing-ratio collapse with
  multiplicative multi-head policies — relevant to our 4-component actions. A-RATIOCLIP.
- **OSFP league**: 60% vs current learner / 40% vs pool by win-stats; add checkpoint only after
  beating all pool members ≥ threshold (0.55-0.7); deploy last iterate. Compare to our
  frozen_ratio=0.10 league. A-LEAGUE.
- **Privileged critic**: keep LSTM history, concat privileged embedding AFTER LSTM; prefer
  reward-relevant aggregates (deck count histograms) over full ordered lists ("noisy TV" risk);
  track speed (AUC) vs final winrate separately. A-PRIVCRITIC.
- **Suphx**: global reward predictor as shaping for delayed rewards; oracle-feature dropout anneal;
  entropy feedback controller α += β(H_target − H̄). A-ENTCTRL.
- **Sample staleness**: ByteRL lost 10% from replay staleness; we're on-policy PPO — keep it.
- **Eval discipline**: argmax at eval (+3%), seat-swapped matchup matrices, fixed held-out opponents.

### 2026-06-10: engine bugs found by random-deck soak (pre-baseline)
Deck-building episodes with random decks/actions crashed the engine — would have killed the long
run. Found via ASAN (build-asan/ + LD_PRELOAD libasan):
1. **UAF in passive observer ctx** (e.g. stt01_009 weapon-discard observer): cleanup freed the
   shared observer ctx immediately while observer deletion is deferred during `ecs_fini` →
   observer fires on freed ctx. FIX: refcounted ctx in passive_runtime (creator ref + one per
   observer released via flecs ctx_free).
2. **Teardown zone access**: observer callbacks + card cleanup hooks (azk01_052, azk01_053, …)
   dereference GameState/zone entities already deleted during world fini → flecs ecs_check abort
   (debug) / UB (release). FIX: `ecs_is_fini || ecs_should_quit` guards in all 14 passive observer
   callbacks; `cleanup_attached_ability_entity` short-circuits to ctx release during quit;
   `azk_cleanup_passive_observer_context` skips buff bookkeeping when quitting.
   Commit 1019d71. ASAN soak: 25 eps seed-42 chain clean; 300-ep seed-777 soak → ANOTHER silent
   abort (different path, investigating).
- NOTE (pre-existing, not mine): `world_tests` segfaults at HEAD 716b37f in
  test_azk01_018_reduces_only_combat_damage_for_equipped_leader → resolve_combat → ecs_get_mut_id.
  Same crash before my changes. Needs separate fix; not blocking training (binding path differs).
  → FIXED by subagent, commit 093b809 on ablation/entdeck-pick-eps: tests wrapped engine calls in
  assert() (compiled out under NDEBUG) and resolve_combat had assert-only guards → release-build
  segfault. Fix follows AGENTS.md call-then-assert pattern; also hardens
  is_card_still_in_owner_battle_zone to fizzle combat when an entity died mid-response-window
  (was UB in release). All tests pass in Debug, Release, and ASan+UBSan builds.
  TODO before ablations: cherry-pick 093b809 onto skylark/model-deck-building + rebuild build/
  (AFTER baseline finishes — no rebuilds while it runs). CAVEAT: the fizzle fix can subtly change
  game outcomes vs the baseline engine (rare path; ASAN soaks never hit it). All ablation arms
  share the new engine so cross-arm comparisons are clean; baseline-vs-ablation deltas carry a
  small engine-version asterisk.

## 3. Ablation backlog (running list; mark ✓ done / ✗ dead end / → in flight)

KEY INSIGHT (sizing the deck-build credit problem): with γ=0.99, λ=0.95 and ~150-300 battle steps,
terminal win reward is discounted to γ^250 ≈ 0.08 by the time it reaches the 62 deck-pick steps,
and GAE credit horizon ≈ 1/(1−γλ) ≈ 17.5 steps. Deck-building learning is therefore carried almost
entirely by the critic's value at battle-start states bootstrapping backwards through pick steps.
The LSTM persists across build→battle (and deck_context shows the decklist during battle), so the
information path exists — but the signal is weak. ByteRL used γ=1.0 (+7% in Hearthstone).

Priority matrix (baseline = azuki_deckbuild_3090.ini recipe, unchanged):
- **A-GAMMA**: γ=1.0 (maybe λ 0.97). Hypothesis: much faster deck-build convergence; risk: critic
  variance from summed shaping. [high priority]
- **A-RANDPICK**: forced random deck picks per episode n∈{0,1,2,4} p=(.5,.25,.125,.125) (ByteRL
  Random-CB). Hypothesis: prevents premature deck collapse, better exploration of card pool.
- **A-LEAGUE**: frozen_ratio 0 (pure self-play) vs 0.10 (baseline) vs 0.25; later OSFP-style
  gating (add checkpoint only when beating pool ≥0.55).
- **A-PRIVCRITIC**: privileged_critic_enabled=true (sees opponent hand + both DECKS — extra
  relevant now that decks are model-built hidden info). Watch value-loss/explained-variance and
  early-AUC vs final.
- **A-SIZE**: LSTM 4096 (base) vs 2048 vs 1024 at fixed wall-clock; does the 4096 LSTM (OpenAI
  Five legacy) earn its FLOPs on a 3090? Smaller may train MORE steps/hour → better final.
- **A-ENTDECK**: entropy treatment of pick steps: (a) baseline, (b) ent_coef 0.02 early→0.002,
  (c) temperature 1.0 for picks (no subaction smoothing on DECK_PICK_CARD candidates).
- **A-GATECOND**: uniform random gate per episode is already env-driven (good). Optionally
  oversample rare gate matchups later.
- **A-PREPROC**: legal_action_scorer_use_references=false (references off was ~7% faster in SPS
  research); scalar normalizer freeze; deck candidate copy-count feature off.
- **A-REWARD-DECKPRIOR**: small shaped bonus at battle start for curve sanity (e.g., penalty for
  >60% same-cost cards). Use ONLY if outcome-only fails; risks reward hacking. [low priority — 
  ByteRL needed no draft shaping]
- **A-SHAPING-ANNEAL**: AZK_REWARD_SHAPING_ANNEAL=1 (dense early → outcome-focused late);
  interacts with A-GAMMA.
- **A-REUSE** (from OpenAI Five digest): update_epochs 1 vs 2 — Five found sample reuse ≥2 cost
  ~2.5× speed; our 2 epochs sits on their bad point. Cheap, high-value ablation.
- OpenAI Five extras: 80/20 current/past opponent sampling with quality weights ≈ our league
  frozen_ratio mechanism (keep); per-param adaptive grad clip; entropy 0.01 confirmed sane;
  win-prob aux head usable as a draft-quality probe (already enabled).

### SPS investigation (2026-06-10)
Smoke run (720 envs, league on, deck building on) steady SPS ≈ 330 vs ~3,500-4,000 for Feb
fixed-deck runs. Trainer perf breakdown over 5 epochs: train=276s (learn=262s!), eval rollout=75s
(env=41s, eval_forward=33s). So the LEARN phase is ~7× the rollout cost — GPU near 21.9/23.5GB;
hypothesis: allocator churn near ceiling + deck-context activations. Probes (no league, 7 epochs):
mb4096, mb2048, and deck-building-OFF control at mb8192 — control distinguishes "deck building
made learn slow" from "current stack (puffer v4/Muon) is slow everywhere".
Deck-build env step cost measured: build steps 826µs vs battle steps 1696µs → deck phase ≈ +25%
steps/episode at half cost; env is NOT the bottleneck.
Probe results so far: mb4096 → 324 SPS (≈ mb8192's 332; minibatch NOT the lever; memory churn
ruled out). Historical SPS check via wandb summaries: March (9mz25s8p, batch 23040) = 3,382 SPS;
April runs = 450-469 BUT at 120 envs/batch 3840 (not comparable).
**Control probe (deck building OFF, same stack): 612 SPS** → mostly a stack/model regression since
March (~5.5×), deck building costs a further 1.85× (612→332).
**bench_forward.py microbenchmark (bf16, deck building obs):**
- forward_eval 1440 rows: 187ms (raw LSTMCell: 11ms → encoder+scorer = ~95% of cost)
- train forward 8192 rows BPTT16: 361ms (raw nn.LSTM: 23ms)
- train fwd+bwd: **5,390ms** — backward:forward ≈ 15:1 (normal ~2:1). Smoking gun.
Diagnosis: the encoder does ~600 slot-encodes/row across 15 zones as hundreds of small
gather/cat/linear kernels → ~3% GPU efficiency, catastrophic in backward (launch-bound).
The 4096 LSTM is NOT the problem (22ms). Model-size ablation won't fix speed; op fusion will.
Levers probing now: update_epochs 2→1 (halves learn + matches OpenAI Five sample-reuse finding);
torch.compile (fuses small ops — the right fix for launch-bound encoders).
- **ue1 probe: 692 SPS steady** (2.1× over 332). Adopted for baseline.
- compile max-autotune-no-cudagraphs: >12 min compiling without finishing an epoch → killed.
  Also would recompile per distinct max-legal-count shape → added power-of-two bucketing to
  `_trim_active_legal_action_candidates` (≤6 shape variants; masking already per-row by count;
  CPU forward sanity-checked).
- compile mode=default + bucketing: **inductor CUDA OOM during compile** + RecursionError →
  compile shelved. Future work: compile only the zone-encoder subgraph, or smaller minibatch
  during compile warmup.
- vec.batch_size 240 (3 async sub-batches): 722 SPS (+4%). Adopted.
- **Final recipe: 332 → 722 SPS (2.2×)**: update_epochs=1 + vec async 240 + text-table precompute
  (the latter also unblocked memory). Config updated (azuki_deckbuild_3090.ini).

### Baseline time-zero fingerprint (epoch 14, ~320k steps, near-random policy)
Deck size is 50 main cards (51 picks/player; 102 picks/episode) — NOT 30 as first assumed.
- copy_entropy_norm 0.976 (near-uniform picks), quad_count 0.355, unique ≈ 37/50
- type shares: ENTITY .731 / SPELL .196 / WEAPON .073 (≈ pool base rates → no preference yet)
- cost shares: 0-1: .246, 2-3: .419, 4-5: .266, 6+: .068 (avg ~2.9 = pool distribution)
- losses: entropy 0.996, explained_variance 0.04, win_prob_aux acc 0.497 (chance)
Archetype emergence = divergence from these numbers, esp. per-gate type shares and quad counts.
NOTE: vec workers don't inherit env vars → baseline run writes NO deck snapshots; per-card
analysis for the baseline uses dump_checkpoint_decks.py on saved checkpoints instead (better
methodology anyway: fixed-policy samples at fixed training stages). Snapshot plumbing for future
runs goes through env config keys (deck_snapshot_dir/every) now.

### 2026-06-10 ~18:20 — baseline v1 DIED at 1.04M steps (epoch ~46): CUDA OOM
OOM during ROLLOUT eval-forward in `_gather_legal_action_refs` (768MB alloc, 23.1/23.6GB used):
once play improved, some state exceeded 512 legal actions → candidate tensors jumped to the 1024
bucket while steady-state allocation sat at ~22.4GB. The setsid wrapper shell kept the tag alive
so the pgrep-based watchdog missed the death (log stalled 1h). No checkpoint yet (interval 250).
FIXES: (1) train.minibatch_size 4096 (halves learn activation peak; probe showed no SPS cost);
(2) relaunched as base-deckbuild-02 (fresh league dir, runlog ..._178113787430.jsonl) — this run
also gets deck snapshots (env-config plumbing fixed); (3) watchdog now alerts on runlog
staleness >12 min instead of pgrep. LESSON for all runs: legal-action-count growth couples play
complexity to memory; keep ≥1.5GB headroom.

### 2026-06-10 ~16:00 — BASELINE LAUNCHED (v1, superseded by v2 above)
`base-deckbuild-01`: 60M steps (~23h @ 722 SPS), league fresh (deckbuild_v1), snapshots every
25th episode → experiments/abl_snapshots/base-deckbuild-01, runlog
experiments/runlogs/base-deckbuild-01_178113228610.jsonl, detached pid (setsid), watchdog armed.
Mid-run analyses planned at ~15M (~6h) and ~30M (~12h): analyze_decks buckets, compare_runs,
draft_vs_reference_eval (CPU, small N during training).
RULE while baseline runs: no branch switches or rebuilds in THIS working tree (workers may
respawn and would import changed code / stale binding). Ablation code prep happens in a separate
git worktree.

### Memory/speed optimization (2026-06-10, required to even run deck building)
Deck-context obs (30 deck slots + 80 candidates/row) OOM'd the 3090 at minibatch 8192: the policy
gathered [batch, slots, 1536]-dim raw text embeddings per zone before projecting. Replaced with a
per-forward **text-feature table**: encode the 194-card vocab once (name/effect/subtype → 56 dims
total), gather 56-dim rows per slot. Exactly equivalent math (per-card function precompute);
verified bit-equal over full vocab on CPU. ~27× less gather memory per zone. Also fixed py3.14
argparse '%' crash in train.py help string.

### Ablation launch commands (fire after baseline analysis; ~15M steps ≈ 7h each at ~590 SPS)
All use `scripts/launch_deckbuild_run.sh TAG STEPS [args...]` (detached; snapshots+jsonl auto).
One at a time (single GPU). Baseline branch unless noted.
- A-GAMMA:    `./scripts/launch_deckbuild_run.sh abl-gamma1 15000000 --train.gamma 1.0 --train.gae_lambda 0.97`
- A-REUSE-2:  (reverse ablation; confirms ue1 was safe) `... abl-ue2 15000000 --train.update_epochs 2`
- A-LEAGUE-0: `./scripts/launch_deckbuild_run.sh abl-league0 15000000 --league.frozen_ratio 0.0`
- A-LEAGUE-25:`./scripts/launch_deckbuild_run.sh abl-league25 15000000 --league.frozen_ratio 0.25`
- A-PRIVCRITIC: `./scripts/launch_deckbuild_run.sh abl-privcritic 15000000 --policy.privileged_critic_enabled true`
- A-ENTDECK (branch ablation/entdeck-pick-eps, run from worktree w/ its own build):
  `... abl-pickeps05 15000000 --policy.deck_pick_smoothing_eps 0.05`
  `... abl-pickeps15 15000000 --policy.deck_pick_smoothing_eps 0.15`
- A-SIZE: needs LSTM_HIDDEN_SIZE config knob (branch ablation/model-size, todo) — 2048 and 1024 arms.
- A-SHAPANNEAL: env AZK_REWARD_SHAPING_ANNEAL=1 variant (edit launcher env or add passthrough).
Comparison: compare_runs.py runlogs + analyze_decks.py snapshots + draft_vs_reference_eval on
final checkpoints; same seed (42) for all arms; deckbuild metrics at matched step counts.
NOTE: league state_path/opponent_dir are SHARED in the config — give each ablation its own
league dir via `--league.state_path experiments/league/<tag>/league_state.json
--league.opponent_dir experiments/league/<tag>/opponents` (REQUIRED to avoid cross-run pollution).
NOTE 2: league candidates are only considered every train.checkpoint_interval epochs (250 in the
baseline → first opponent at ~epoch 250 ≈ 5.8M steps; pure self-play before that). For 15M-step
ablation arms add `--train.checkpoint_interval 100` so the league actually matters
(~6 candidates/run); league-off arms unaffected.
NOTE 3 (memory ceiling): every league pool entry is loaded as a FULL GPU model (~0.8GB each;
measured 15.2GB@pool0 → 16.8GB@pool2). keep totals 6+4+3=13 → projected OOM near pool 9
(~50M steps). Baseline stops at ~30M (pool ~5, ~19GB — safe). Ablation arms use small pools:
`--league.keep_recent 2 --league.keep_mid 1 --league.keep_old 1`. FUTURE WORK: hold opponents on
CPU, move only batch-assigned ones to GPU.
SPS by pool size (league cost): 515 (pool 0) → 400 (pool 1) → 353 (pool 2) — frozen forwards
fragment into small per-policy batches; expect ~330 steady. 30M ETA ≈ tomorrow morning.

Run protocol: short runs 30-50M steps (~3-4h) for triage on 2 seeds where feasible; promote
winners to ≥100M confirmation; decision metrics (in priority order):
1. deckbuild_result winrate trends per gate + overall win0_abs_delta (vs frozen league),
2. deckbuild/main_* composition: unique count ↓ toward 12-20, quad_count ↑, per-gate type
   divergence (weapon share for LIGHTNING, spell share for Echoed Waves),
3. cross-gate L1 divergence from snapshots (analyze_decks.py),
4. losses/explained_variance, entropy trajectory, SPS.

### FINDING (2026-06-10): sampler anneal knobs are dead code on the default path
`tcg_sample_logits` routes legal-action-scorer outputs (the DEFAULT head) to
`_sample_legal_action_rows`, which applies NO temperature and NO smoothing — the configured
`subaction_temperature` / `smoothing_eps` anneals only affect the non-default factorized path.
All recent training explored via raw softmax + entropy bonus only. Implications:
(a) historical anneal settings were no-ops; (b) exploration ablations must patch the row sampler.
Branch `ablation/entdeck-pick-eps` adds policy.deck_pick_smoothing_eps (uniform mix over pick
candidates only, ByteRL Random-CB analog) + policy.legal_row_temperature (global row softmax
temperature, makes the old knobs meaningful). Unit-tested: pick rows entropy 1.54 w/ eps=0.5,
battle rows unaffected.

## 3.5 Baseline launch plan (base-deckbuild-01)
- Config: azuki_deckbuild_3090.ini + `--train.update_epochs 1` (681 SPS vs 332 at ue=2; also
  matches OpenAI Five sample-reuse evidence) + compile if probe green.
- League: fresh `experiments/league/deckbuild_v1` (recreated at launch).
- Snapshots: `experiments/abl_snapshots/base-deckbuild-01`, every 25th episode/env.
- total_timesteps: sized to ~20h wall from measured SPS.
- Mid-run analyses at ~25/50/75%: analyze_decks buckets, compare_runs trends, and
  draft_vs_reference_eval on latest checkpoint (does drafting beat reference decks?).
- Per-gate watch list: deckbuild_gatecard/*/type_share/WEAPON (Lightning gates should rise),
  /SPELL (Echoed Waves), avg_cost (Rushfire should fall), leader split per gate,
  deckbuild_result/gatecard/*/win.
- Privileged critic check (verified): post-LSTM concat fusion — correct per informed-asym paper;
  arms for A-PRIVCRITIC later: off (baseline) / full / hand-only / deck-count-histogram.

## 3.6 Availability priors (essential for honest archetype claims)
Per-element candidate pools (element + NORMAL cards):
| pool | n | ENTITY | SPELL | WEAPON | avg cost | cost 0-1/2-3/4-5/6+ |
|---|---|---|---|---|---|---|
| LIGHTNING | 75 | .69 | .17 | **.13** | 2.79 | .25/.43/.28/.04 |
| WATER | 78 | .71 | **.24** | .05 | 2.86 | .27/.42/.22/.09 |
| FIRE | 80 | .78 | .17 | .05 | 2.88 | .25/.42/.26/.06 |
| EARTH | 80 | .75 | .20 | .05 | **3.20** | .23/.38/.29/.11 |
6/10 weapons are LIGHTNING; 11/38 spells are WATER; EARTH skews expensive. So element-level
type-share differences are partly AVAILABILITY, not strategy. Methodology rules:
1. Compare each gate's shares to its own element pool prior (above), not to other elements.
2. The decisive strategy test is WITHIN-element gate pairs (identical pools):
   Surge vs Stormchain (L), Hydromancy vs EchoedWaves (W), Rushfire vs Ragefire (F),
   Devotion vs Stonehaven (E). Any composition gap there is pure strategy.
3. Same logic for leaders within an element (2 leaders share the gate's pool).

## 3.7 Baseline v2 early dynamics (live observations)
- 1.5M steps: quad_count spiked 0.36→1.16 (early pick fixation) then COLLAPSED to 0.08 by 2.9M;
  unique rose to 39.6 (more spread than random's 37). Early stacking was transient value-noise.
- 2.9M: episode_length 69 ticks, 100% winner terminals, 0 truncations, attack_rate 0.33 —
  short, aggressive, decisive games.
- ~~WATCH ITEM: gate_portal_selected_rate ≈ 0.004 at 2.9M~~ → DOWNGRADED: continuous snapshot
  data shows portal_rate steady at 0.08-0.09 across all buckets; the 0.004 was a single logging
  window artifact (or transient dip). Keep on the dashboard but not alarming.
- Oscillating deck concentration in self-play (pool still empty): unique 37→39.6→22→33.5,
  quads 0.36→1.16→0.08→3.84→0.0 over 0→5.6M steps. Classic non-transitive draft-meta churn;
  league activation (epoch 250+) should damp it. Seat winrate also swung (0.59 p0 → 0.38 p0).

### First snapshot analysis (720 episodes, ≈0→5.5M steps, 3 buckets)
- **Type shares ≈ availability priors everywhere** (L weapons .13-.14 ≈ prior .13; W spells
  .24-.25 ≈ prior .24): no strategy-driven composition yet. Same-element L1 divergence 0.108 →
  0.092 (availability-dominated); cross-element ~0.42.
- **Win ordering emerging** (vs mixed opponents): Lightning gates lead (Surge .58-.61,
  Stormchain .53-.63); Hydromancy lags (.43); Ragefire weakest early (.36→.45 recovering).
- **Leader splits healthy ~50/50 within every gate** — no leader collapse (ByteRL warning not
  manifesting at this stage).
- **Episode lengths growing 84 → 109 → 137 ticks** across buckets: the opening all-aggression
  meta is softening into longer games. attack_rate stays highest for Lightning gates (~0.26).
- Early per-gate flavor in top cards despite similar aggregates: Surge tops = Lightning Shuriken
  (c1 weapon) + Raizan; Devotion concentrated Lone Journeyman x1.44; Stonehaven runs big bodies
  (Sandcoil Python c8, Rock Sloth c6) consistent with EARTH's expensive pool.

## 3.8 MID-RUN ANALYSIS @ ~14M steps (2160 snapshot episodes, 5 buckets; CSV in results/)
**Meta oscillation with a near-death of the differentiating mechanics, then recovery:**
- portal_rate: .09 (b1) → .05 (b2) → **.002 (b3!)** → recovering → .06-.08 (b5)
- SPELL share: .20 → .13 → **.04 (b3)** → .14-.20 (b5); weapons dipped similarly, recovered
- bucket 3 was an entity-only aggro collapse (ENTITY .88-.94, eplen ~70, jaccard .35); league
  activation (pool 1→2) coincides with the recovery — consistent with frozen opponents punishing
  one-trick metas. The earlier 0.004 portal reading was this collapse, not an artifact.
- LIKELY REWARD BIAS: shaping rewards garden attack + untapped counts → entity flooding is
  locally optimal; spells/weapons/portals add no immediate shaped potential. Raises priority of
  A-SHAPANNEAL and A-GAMMA (let terminal signal compete with shaping).
**First genuine archetype signals (bucket 5, ~12-14M):**
- Leader preferences: Rushfire→Zero 111:24 (self-damage combo leader), Devotion→Bobu 58:26;
  other gates ~50/50. Per-gate leader differentiation WITHOUT global collapse.
- Devotion drafts a big-body curve: avg_cost 3.75 vs EARTH availability prior 3.20
  (Osunanami c10 ×1.69/deck, Sandcoil Python c8, Rock Sloth c6) — strategy beyond availability.
- Universal staple: AZK01-014 Trade Guild Cavalry (NORMAL c5) top pick in EVERY gate (1.4-2.0×).
- Win spread widening: Stormchain .675, Hydromancy .628, Surge .608 vs Rushfire .281,
  Devotion .405. Rushfire collapsed .45→.28 (the meta now beats it; its Zero-leader pivot may be
  a compensation attempt).
- Lightning weapon share .10-.11 ≈ prior .13 (still availability-level, NOT yet weapon-leaning).
### 2026-06-11 ~03:00 — crash #2 (league eval) and OOM #3, resumed from ep500
- Crash #2: first REAL league eval crashed training. `league_eval._unwrap_base_env` walked past
  the deck wrapper to the battle env → stale `_active_player_index` during drafts → zeros sent
  to the drafting seat → DeckBuildingParallelEnv ValueError → trainer died (hung shell).
  evaluate_checkpoint.py had the sibling bug (stats wrapper forwards the marker but blocks
  underscore attrs → always seat 0). Both fixed (neither file is resume-fingerprinted);
  inline evaluator now verified over full draft episodes. Training rollouts were unaffected —
  only eval/promotion was broken; explains why no promotion ever happened.
- OOM #3 on resume: the MATURE ep500 policy immediately produces >512-legal-action states →
  whole minibatch pads to the 1024-candidate bucket (4096×1024×310×4B ≈ 5.6GB spike) + 1.6GB
  league opponents resident. Fix: --train.minibatch_size 2048 (CLI; not fingerprinted).
  Memory model: base ~13GB + 0.8GB/opponent + bucket spike ~2.8GB@mb2048 → safe through
  pool ≈ 5-6 (30M stop point), OOM-bound near pool 9. LESSON for ablations: mb2048 + small pools.

- OOM #3 root cause was NOT the learn minibatch: `_rollout_health_snapshot` (resume-only
  diagnostic) forwarded the ENTIRE 23040-row rollout in one pass → [23040,1024,64] deck-candidate
  tensor = 5.62GiB. Fixed by chunking (dim-0 aware for BPTT layout); fingerprints of saved
  checkpoints patched with the new train.py hash (conscious certification, documented here).
- RESUMED from ep750 (~16.2M, pool=3 reloaded) at mb2048: probe passes (entropy 1.64), memory
  10.2GB post-probe, SPS ~300. 30M stop ≈ 13h away. A bonus ep750 checkpoint existed pre-crash.

**Draft-vs-reference eval @ checkpoint ep500 (11.5M): drafter wins 45.8%** (48 eps seat-fair,
0 timeouts; same policy both seats → isolates deck quality). Drafted decks ≈ reference parity
(point estimate slightly under; N small). Track at every checkpoint — the slope is the metric.
Results JSON: results/bdv2_draftref_ep500.json.

## 3.9 BASELINE COMPLETE — base-deckbuild-02 stopped at ~31.7M (epoch 1500, 2026-06-11)
Checkpoints: 250/500/750 (run dir ..._178113787430) + 1000/1250/1500 (..._178117983197).
4,324 snapshot episodes; final-bucket (≈28-32M) findings:
1. **Cheap-aggro attractor won.** avg_cost 2.27-2.36 in EVERY gate (mid-run Devotion's 3.75
   big-body identity died → 2.33). Alley Thug (c1) staple ×2.4-3.0 in all decks. attack_rate
   .27-.34, eplen 71-89, jaccard ~.32 (concentrated decks).
2. **Weapons extinct** (.01-.06 share vs .13 Lightning availability; weapon_rate ≤.02);
   **portal usage fell to .031-.041** (from .09). Gate-defining mechanics largely unused
   at convergence.
3. **Spells survived above availability**: .24-.34 share (Water gates .32-.34 > .24 prior);
   Thunderclap c5 a Surge staple. Spell_rate in play .026-.062.
4. **Same-element divergence COLLAPSED to .057** (from .108 early) — the attractor erased most
   gate-specific identity. Exception: FIRE gates keep the most distinct, element-loyal decks
   (elem_share .46, Cinderwake Seer + Lady Emberheart staples).
5. **Meta rotation in win rates**: early Lightning-dominant (Surge .58-.61) → late Fire-favored
   (Rushfire .28 mid → .64 final; Hydromancy .63 mid → .38 final). Non-transitive churn persists
   at pool 5 but gentler (unique oscillation 30.6-38.5 vs 22-40 pre-league).
6. Leader splits: mostly ~55/45 balanced at the end; the mid-run Rushfire→Zero 82% preference
   RELAXED back to 61/39 — leader specialization was itself a meta phase.
**Draft-vs-reference TRAJECTORY: 45.8% @11.5M → 40.6% @31.7M (96 eps)** — drafted decks got
WORSE against human reference decks as self-play progressed. Self-play overfitting signature:
the cheap-aggro attractor wins internally but loses to balanced human curves. "Train longer"
is NOT the fix; credit/exploration structure is.
Verdict: the model clearly LEARNED to draft (staples, copy concentration, cost discipline,
win-correlated choices) but converged to a low-diversity tempo meta where gate identity barely
matters — consistent with shaped-reward bias toward board-attack tempo (garden-attack potential)
and weak terminal credit to picks (γ^250 ≈ .08). EXACTLY the failure modes A-GAMMA,
A-SHAPANNEAL, and A-ENTDECK target. (Engine-comparability asterisk: ablations run on the
fizzle-fixed engine.)

## 3.10 A-GAMMA COMPLETE — abl-gamma1 analysis (done 2026-07-06; run finished 06-12)
Run: 12M steps (epoch 521), checkpoints 100..500+final, 1,444 snapshot episodes, runlog 46MB.
Matched-step comparison to baseline first 12M (traj_compare.py, 2M bins):
1. **Pick-credit hypothesis CONFIRMED, outcome hypothesis REFUTED.** γ=1.0 produced much
   faster early deck commitment: quad_count spiked to 7.5 (vs baseline 1.2), unique 22 at 2M.
   Stronger terminal credit DID reach the picks. But it converged to the same element-level
   cheap-tempo attractor — faster credit for the same wrong signal.
2. **No gate-conditional drafting in either run.** Same-element gate pairs are numerically
   IDENTICAL through training (Surge/Stormchain weapon share .136/.137; Hydromancy/EchoedWaves
   spell share .246/.247; Devotion/Stonehaven avg_cost 3.26/3.25 — every 2M bin). Snapshot
   bootstrap test (gate_identity_probe.py): same-element L1 excess over noise floor ≈ 0 in all
   4 elements for gamma1 (−0.011..+0.007); baseline at 31.7M has ONE real signal: LIGHTNING
   pair excess +0.043. Element (= candidate pool) drives composition; the gate card does not.
3. **Mechanics usage decayed under γ=1.0**: portal rate 0.042→0.003, weapon attach 0.033→0.010
   by 12M (baseline recovered to 0.040/0.019 on its oscillation). Both runs' oscillations are
   phase-shifted; gamma1 stopped mid-trough (12M) — partial confound, but no sign of γ=1.0
   HELPING mechanics.
4. **Behavior→win correlations (last-window snapshots)**: attack_rate +0.51 pooled (tempo wins);
   portal ≈0 (baseline) / −0.12 (gamma1); weapon/spell ≈0-to-negative. As currently played,
   gate mechanics don't pay. (Correlational + winner-biased, but consistent with shaping bias.)
5. **Training health**: value_loss 0.041 vs 0.032 (expected — undiscounted return variance),
   EV ≈ same (0.74), SPS lower (271 vs 375 — league pool loading earlier due to
   checkpoint_interval 100). Draft-vs-ref final: 36.3% (80 eps) vs baseline 45.8% @11.5M
   (48 eps) — overlapping CIs, no improvement.
VERDICT: γ=1.0 alone ✗ dead end as the primary lever. The bottleneck is not credit strength
but WHAT the battle meta rewards: shaped tempo. The draft can only learn gate identity if
gate mechanics have positive value in play. → attack the reward bias + exploration
(A-SHAPANNEAL, pick-eps, entropy), measure with the noise-floor-corrected divergence test.

## 4.5 Probe results on old checkpoints (2026-07-06; June-checkpoint asterisk applies)
New probes in this dir (all reusable on round-2 checkpoints):
- **probe_gate_kl.py** (interventional gate swap with action replay; control replay KL == 0
  exactly): gamma1@12M pick-distribution sensitivity to the gate card is **KL ≈ 0.00000,
  TV ≤ 0.0002** for every same-element pair. The policy functionally ignores gate identity.
- **Embedding geometry (root cause)**: projected 48-d metadata embeddings of gate cards on
  trained checkpoints: Rushfire/Ragefire cos 0.9995 (L2 0.08 @ norm 2.5), Devotion/Stonehaven
  0.9995, Surge/Stormchain 0.968, Hydromancy/EchoedWaves 0.973. The text-effect features that
  distinguish same-element gates do not survive the learned projection → the policy CANNOT
  condition on gate identity even if rewards demanded it. FIX SHIPPED: flag-gated
  `policy.gate_id_embedding_enabled` (16-d learned per-card-id channel into gate zone encoder
  + deck_context gate slot; commit 7ffca7b). Untrained flag-on already drops FIRE-pair
  combined cos to 0.957.
- **probe_deck_behavior.py** (forced weapon-heavy/spell-heavy/entity-only decks, fixed
  opponent, availability-calibrated by a uniform-legal baseline): smoke-checkpoint
  conditioning ratio ~1.1-1.3 (policy gap ≈ availability gap — no deliberate use yet).
  Metric for round-2 arms: ratio ≫ 1.
- **synergy_lift.py**: co-occurrence lift with sibling-differential mode (same-element sibling
  = identical pool → availability-controlled). Baseline@31.7M shows small dlift pairs
  (e.g., Stonehaven: Sanzu's Envoy+Tenraku dlift +3.6) — needs a permutation null before
  claiming synergy (TODO).
- Checkpoint-load fix: `_materialize_scalar_norm_buffers_from_state_dict` was broken for the
  GPU-resident ScalarRunningNorm (device arg) since the round-2 optimization refactor — every
  checkpoint load (incl. league opponent ingest) crashed. Fixed in 7ffca7b. Smoke checkpoint
  round-trips cleanly (0 missing/unexpected keys).

## 5. ROUND 2 (2026-07-06 →) — new stack, breaking the element-only equilibrium
Hypotheses:
- H1 reward bias: garden-attack potential shaping makes entity-flood tempo locally optimal;
  mechanics (portal/weapon/spell) only pay via terminal outcome → anneal shaping to 0.05.
- H2 exploration: mechanics are rare actions; policy never explores them enough in context to
  learn their value → pick-eps for draft diversity; higher/slower-annealed entropy for battle.
- H3 conditioning: policy may not functionally read the gate id from deck_context → KL probe
  planned on round-2 checkpoints (same-element gate swap in obs → pick-logit KL ≈ 0?).
Protocol: 15M steps/arm, seed 42, mb2048, checkpoint_interval 100, league keep 2/1/1, own
league dirs, snapshots every 25th episode. ~2.3h/arm at 1.8k SPS (slows as pool grows).
Decision metrics (in order):
1. same-element L1 excess over bootstrap floor (gate_identity_probe.py) — the headline signal;
2. mechanics usage rates (portal/weapon/spell) late-run + their win correlation;
3. per-gate playstyle divergence at matched element (weapon_rate for L gates etc.);
4. draft-vs-ref win rate (96+ eps, argmax) at final checkpoint — external validity;
5. quad/unique trajectories + win-prob-aux acc (draft-quality probe).
Arms (chain script: run_round2.sh — serial, ctrl2 → anneal1 → gateid1 → combo1, draft-vs-ref
eval 96 eps argmax after each):
- ctrl2: config unchanged (new-stack control).
- anneal1: AZK_REWARD_SHAPING_ANNEAL=1, INITIAL 1.0, FINAL 0.05, WARMUP_EPISODES 8,
  RAMP_EPISODES 25 (per-env! ≈ fade over 2M→10M agent-steps @720 envs).
- gateid1: policy.gate_id_embedding_enabled=true (representation fix alone).
- combo1: anneal + gateid + policy.deck_pick_smoothing_eps 0.05 (the ceiling arm).
Later (if signal): factor combo1 back out; A-LEAGUE strength; A-PRIVCRITIC (critic sees decks);
mechanic-targeted exploration (legal_row_temperature).
Evidence bar for "model understands per-card strategy" (final report §):
E1 gate-KL probe > 0 by a clear margin on gateid arms at matched steps (ctrl2 ≈ 0), rising
   over checkpoints; E2 same-element L1 excess over noise floor > 0 and growing;
E3 probe-B conditioning ratio ≫ 1 (uses what it has beyond availability);
E4 sibling-differential synergy lifts surviving a permutation null with dwin > 0;
E5 draft-vs-ref ≥ ctrl2 and not degrading with training (external validity).

## 6.1 NATIVE PORT LANDED (2026-07-06, commit a13d66e)
- Draft phase fully in C (c_step_draft/c_reset draft branch); deck_context packed block
  appended to the obs struct (12,604B rows vs 9,416 battle-only; sizes cross-checked at
  import via binding.obs_struct_sizes). Catalog arrays passed from the wrapper's own
  builder → candidate ordering parity by construction.
- PARITY: test_native_deckbuild_equivalence.py — forced gates (AZK_DEBUG_FORCE_GATE_DEF_IDS)
  + identical pick replay: all 102 draft steps bit-equal (modes incl. early-finisher
  BATTLE mode, candidate ids/copies incl. 4-copy reindexing, mask rows, active player),
  battle-transition deck_context equal, privileged decks sanitized; policy packed decode
  equals struct; metric helper == legacy battle-start metrics to 1e-9. Plus existing
  battle-only equivalence + all unit tests + ctest green.
- LEAGUE BUG FIXED (affects battle-only native too): trainer/league grouped rows by worker
  INSTANCE (80 rows) not game; matchups/seat draws/reward decomposition/win-prob labels were
  worker-granular and matchup resampling ~never fired. Now game-granular via
  driver_env.agents_per_match. Inline league evaluator + draftref + evaluate_checkpoint
  force the legacy env path (they drive seats through wrapper internals).
- SMOKE (native-smoke, 2.5M steps, league ckpt-interval 40): steady SPS median 3,215
  (p90 3,916) with league pool active — first promotion at epoch 42 survived (fixed loader,
  game-granular rows). 552 deckbuild metric keys flowing; snapshots written; values sane
  (unique 33.9, cost 3.07, quads 0.94 at ~1M steps). vs 1,836 legacy / 515 June: ~6× June,
  1.75× legacy-optimized. Memory 10GB/24GB at mb 4096.
- Round-2 chain switched to azuki_deckbuild_native_3090.ini; anneal knobs recalibrated for
  480 games (WARMUP 12 / RAMP 40 per-env episodes ≈ fade 1.9M→8.2M of 15M).

## 6. NATIVE DECK-BUILDING PORT (2026-07-06, user directive: land before further ablations)
Round-2 chain was killed ~40 min into ctrl2 (artifacts kept: experiments/runlogs/ctrl2_*,
snapshots) — arms will re-run on the native path so all arms share one env path.
Design (v1):
- C draft phase inside the native vec env (env_binding.h/tcg.h): per-env DraftState
  (gate/leader/main/copy-counts/candidate table per player), catalog computed at vec_init from
  card defs + deck pool (gate population weighted by pool frequency; leader ids per element;
  main candidates element+NORMAL in the wrapper's exact order). Step: validate DECK_PICK_CARD
  (sub1 = candidate index, ≤4 copies, 50 mains), P0→P1, then assemble decks (gate+leader+main
  +IKZ) and reset the engine with decks; auto-reset re-enters draft. Rewards 0 during draft.
- Obs: packed struct extended with a deck_context block (mode, gate/leader def ids,
  main ids[50], counts, candidate ids[1024] + copy counts[1024]) appended to the base
  TrainingObservationData → separate deck-building obs dtype on the Python side
  (AzukiNativeEnv gains deck_building flag + NATIVE_DECKBUILD_OBS_DTYPE); battle-only struct
  unchanged (fixed-deck checkpoints unaffected). Battle steps keep deck_context (mode=BATTLE,
  no candidates) exactly like the wrapper.
- Metrics/snapshots stay in Python: C exports per-episode records (gate, leader, main[50],
  win, behavior rates, eplen) via a drain call; AzukiNativeEnv computes the full
  azk_step_deckbuild/* + gatecard metric surface per log window and writes snapshot JSONLs.
- Parity: struct offset/sizeof cross-check exported by binding; bit-exact draft obs vs
  DeckBuildingParallelEnv under forced gates (AZK_DEBUG_FORCE_GATE_DEF_IDS) + replayed picks
  (methodology of test_native_obs_equivalence.py); metric key-surface equality.
- League: LeaguePuffeRL row mechanics being verified for the 40-envs-per-instance layout
  (mapper agent); cuda_graphs likely incompatible with league row-splitting — v1 runs
  native + train-side compile, cuda_graphs off; revisit after benchmark.

## 7. ROUND-2 RESULTS (filling as arms complete)

### ctrl2 (control, native stack, 15M, epoch 977, pool 4 final) — DONE 07:39
- SPS 3.7k (pool 0) → 1.6k (pool 4); wall ~2.5h incl evals.
- **Negative control airtight**: gate-swap KL = 0.00000 in all 4 elements (306 pick
  steps each, control replay 0); same-element L1 excess ≈ 0 everywhere
  (−0.007..−0.001 vs bootstrap floor). The new stack alone does NOT create gate identity.
- Deck→behavior conditioning ratio: weapons 1.26, spells 1.12 (June baseline: 1.35/0.97).
- Draft-vs-reference: **34.4%** (96 eps argmax) — the bar for the arms.
- Composition dynamics mirror June: quads spike-collapse cycles, unique 32-37,
  avg cost ~2.8; same-element divergence 0.10 (early) → 0.07 (late) ≈ floor.

### anneal1 (shaping→0.05 over ~2-8M, 15M, epoch 977) — DONE 10:25
- **Draft-vs-reference 45.8% vs ctrl2 34.4%** (+11.4pp, 96 eps argmax each, ~1.6σ —
  re-run with more eps at chain end). Shaping anneal = big external-validity win (E5).
- Mid-run playstyle shift while shaping faded: portal rate peaked 0.082 @6M (2-3× ctrl2),
  spells 0.052; settled ~0.03/0.03 as meta equilibrated. H1 (reward bias suppressed
  mechanics) CONFIRMED in play behavior.
- Same-element L1 excess: +0.013 FIRE / +0.007 EARTH / +0.005 WATER / −0.003 LIGHTNING —
  first above-floor divergence (ctrl2 all ≤0), still small (E2 weakly positive).
- Gate-swap KL = 0.0 as expected — no identity channel; anneal alone cannot create
  gate-conditional drafting (representation still collapsed). Factorization confirmed.
- Forced-deck probe: weapon-deck win 0.50 vs spell/entity 0.12 (conditioning ratios
  1.16/1.07 similar to ctrl2).

### gateid1 (gate-id embedding only, 15M, epoch 977) — DONE 13:13
- Gate-swap KL still ≈ 0 (TV ≤ 0.0005, marginally above ctrl2's ≤0.0001 but functionally
  zero); same-element L1 excess at floor; draft-vs-ref 39.6%.
- INTERPRETATION: the identity channel exists (untrained combined cos 0.957 vs 0.9995)
  but training never amplified it — capacity without incentive goes unused. Confirms the
  chicken-egg: gate conditioning needs BOTH a distinguishable representation AND value
  differences that reward using it. combo1 (anneal+gateid+pick-eps) is the both-at-once arm.

### League flat-SPS fix (task 7) — LANDED mid-chain (2a162cd; combo1 unaffected,
its process pre-imported the old module)
- OSFP-style windowed frozen sampling: league.frozen_window_epochs=8,
  max_distinct_frozen=1 in the native deckbuild config; default off elsewhere.
  4 new unit tests. Benchmark after combo1: expect ~flat 3.2-3.5k at pool 4
  (was 1.6-2k). Applies to the NEXT runs, not the current chain.

### combo1 (anneal+gateid+pick-eps, 15M, epoch 977) — DONE 16:00; CHAIN COMPLETE 16:13
- Draft-vs-ref 50.0% @96 eps (chain seed) / 40.6% @192 (seed 555) → pooled 43.8%.
- Synergy: 26 pairs p<0.01 & sup≥20% (vs ctrl2's 6, gateid1's 0); EARTH same-element
  excess +0.018 (largest of any arm); weapon conditioning ratio 1.31; spell 0.50
  (spell aversion under pick-eps — spells diluted, spell_rate 0.004-0.015 in play).
- Gate-swap KL still 0 (TV ≤0.0005) — pair-level conditioning did not ignite at 15M
  even with channel+incentive+exploration. Needs longer runs (recommendation: 45-75M).

### POOLED draft-vs-reference (chain 96 + rerun 192 = 288 eps/arm, argmax):
ctrl2 36.8% | gateid1 39.6% (96 only) | combo1 43.8% | anneal1 44.4%.
anneal−ctrl +7.6pp (~1.9σ); combo−ctrl +7.0pp (~1.7σ). Robust across eval seeds.

### League windowed sampling BENCHMARK (leaguebench, 4M, ckpt interval 40) — task 7 DONE
SPS by pool: 0→3,679 | 1→2,913 | 2→2,486 | 3→2,529 | 4→2,495 — **FLAT from pool 2 on**
(old per-game sampling: 2.9k @2 → 1.6-2.0k @4 and still falling). At pool 4: +25-55%;
pool-size-independent for future bigger pools. Commit 2a162cd.

### Final report: final-report.md (complete — exec summary, evidence scorecard,
distributed-run recommendations). Evidence bar: E3/E4/E5 met, E2 weak-positive,
E1 (sibling-gate KL) open — the explicit target for the long distributed run.

## 4. Key questions to answer
- Does the model build legal-but-coherent decks (curve, type mix) per gate, or collapse to one deck?
- Do per-gate compositions diverge (weapons for LIGHTNING, spells for Echoed Waves, etc.)?
- Does deck quality improve win rate vs fixed reference decks over training?
- How do playstyles differ across gates (aggression metrics, attack frequency, game length)?
- NEW (round 2): does the specific GATE CARD (not just element) causally shift picks and play?
- NEW (round 2): do synergy pairs co-occur above chance (co-occurrence lift), and does the
  policy USE what it drafts (deck→behavior coupling under forced-deck probes)?

## 6.5 portalgp1 probe verdict (2026-07-08 23:50)
- gate-swap KL @15M: 0 in all four elements (TV <= 0.0007) — exposure alone
  does not create draft conditioning at 15M.
- critic sensitivity @15M: mean ratio 0.17, sign 94% (LIGHTNING 0.10/83%,
  WATER 0.19/100%, FIRE 0.24/100%, EARTH 0.16/92%) — comparable band to
  combo1@15M (0.30/100%) given per-checkpoint noise; no clear acceleration.
- Net: portal-GP bonus = external quality + exposure lever (draftref 46.9%,
  2.5x portals), not (yet) a conditioning lever. portalgp45 (45M, launched
  23:45) is the decisive horizon test; trajectories auto-run after.

## 6.6 Complete gate-ability ladder (2026-07-09 06:36, neutral all-NORMAL mirror decks vs Hydromancy)
policy-mode win rate vs REF (n=640/cell; forced/blocked in results/gate_gap/cross_*):
Rushfire 57.7 > [Hydromancy ref ~50] > Devotion 46.7 ~ Stormchain 46.7 ~ Surge 45.9
> Stonehaven 44.1 > EchoedWaves 43.0 > Ragefire 41.2 (forced 37.7 — forcing its
ATK-buff portal is NET-NEGATIVE). 16pp raw ability-power spread on identical
decks; FIRE siblings are the game's strongest AND weakest gates (matches their
12.7pp sibling gap). Blocked ~50-52 everywhere (nulls hold; both sides lose
portals in blocked mirrors). GAME-BALANCE FEEDBACK: gates are far from parity;
Ragefire needs a buff or redesign, Rushfire is dominant deck-independent.

## 6.7 privgp1 (A-PRIVCRITIC + portal-GP recipe) verdict @15M (2026-07-09 13:00)
- draftref 39.6% (96 eps) — weakest 15M arm (portalgp1 46.9, anneal1 44.4).
- gate-swap KL == 0; critic sens ratio 0.08 (< combo1 0.30, portalgp1 0.17).
- Privileged drafted-deck visibility did NOT sharpen sibling value sensitivity
  at 15M and cost external quality. No positive trend to justify a 45M bet.
- TRAINING-LEVER SEARCH CONCLUSION: oversampling / portal exposure /
  privileged critic / 3x horizon — every lever leaves the actor's draft
  distribution sibling-blind (KL == 0 everywhere) while the critic always
  prices the distinction. The remaining question is whether conditioning is
  even OPTIMAL: composition x gate interaction probe running (gate_ix/).

## 6.8 Composition x gate interaction — LIGHTNING (2026-07-09 14:00)
WR(weapon_heavy vs entity_only | same gate both sides, mirror-gate, n=500/cell):
Surge 22.4% vs Stormchain 30.4% -> **interaction -8.0pp (z~2.9, p~0.004)**.
The same composition is worth 8pp more under one sibling than the other ⇒
gate-conditional DRAFTING has real value; the actor's sibling-blindness is an
unexploited margin, not optimal indifference. (Also: naive 24-weapon archetype
loses to entity_only overall at current play skill — the interaction, not the
level, is the finding. Weapon-heavy pairs better with re-equip (Stormchain)
than discard-replay (Surge).)

## 6.9 45M critic-trajectory comparison (2026-07-09 15:00)
Late-run (>30M) sibling critic sens ratio: combo45b 0.71 vs portalgp45 0.21
(both ~95% sign-consistent; KL == 0 across both full runs). Portal-GP bonus
trades value-side sibling sharpness (dense bonus stream occupies value
capacity) for the best external quality (46.9%). Recipe implication: portal-GP
for production play strength; drop it if the objective is critic-side gate
representation research.

## 7. NEXT EXPERIMENT SPEC — draft-time aux objectives (user-approved 2026-07-09 evening)
Rationale: lever search closed with critic gate-knowledge proven but never
reaching the pick head (KL==0 everywhere). These are the first mechanisms that
USE the proven signal instead of hoping exposure/scale transfers it.
- aux1 (bootstrap): pick-step advantages target battle-start V — A_t ≈
  V(s0;g) − V(s_t) for DRAFT steps only (GAE override at the draft/battle
  boundary). Trains picks on the LEVEL of critic-predicted deck value;
  credit path 150 steps → 1. Slow anneal (outcome-anchored, low bias risk).
- aux2 (differential): aux reward λ·max(0, V(s0;g) − V(s0;g→sibling)) at
  draft end — cancels deck-quality level, amplifies ONLY sibling fit. Clip
  at 0 (unclipped is gameable by sabotaging the counterfactual); standard
  shaping anneal. 2 extra forward passes/ep (gate-swap machinery exists).
- Matrix: auxv1 / auxd1 / auxvd1, 15M each on the portal-GP production base;
  portalgp1 is the control. Accept ≤~3pp draftref cost at smoke scale.
  Any KL/composition movement → 45M + long-horizon win comparison vs
  portalgp45. None → distributed scale is the last hypothesis.
- pcritic verdict stands: not used going forward (lost every axis @15M).
- Game rules are FIXED (real TCG) — balance findings are product intel, not
  a training lever; design-amplification avenue is OFF the table.
- Trainer implementation: differential test (aux off = bit-identical) before
  any launch, per convention. Tasks #5/#6 carry the full spec.

## 7.1 Mask-desync fuzz result (2026-07-09 23:15)
32M agent-steps of uniform-random legal play (8 workers x 16 envs x 250k
vec-steps, sibling oversampling on): ZERO desync hits. Plus zero
"Invalid-action truncation" lines across 150M+ trained steps since the
mitigation. Posture for production: non-fatal truncation + automatic repro
logging; root-cause deferred until a seed is captured. (fuzz_mask_consistency.py
kept for regression sweeps after engine changes.)

## 7.2 A-DRAFTAUX post-mortem: the interaction is not in the critic (2026-07-10 05:30)
Per-deck spread of the critic's sibling differential (battle_dv_signed_values,
each seed = a different deck): |mean|/std = 2-10 across all arms/elements
(e.g. auxd1 LIGHTNING mean +0.0043 std 0.0005). The critic prices the gate
MAIN EFFECT but carries almost no gate x composition INTERACTION — while the
game's true interaction is up to 8pp (probe-measured, LIGHTNING).
=> auxd1's injected bonus was a per-gate near-constant: no gradient across
pick choices; the null was structurally guaranteed. GENERAL CONCLUSION: no
critic-derived aux can teach gate-fit drafting until the value function
itself represents the interaction term. The complete bottleneck chain:
game has main effect + interaction -> critic learns main effect only ->
actor exploits main effect in play (state-reactive) -> nothing trainable
carries the interaction -> sibling-conditional drafting unreachable at this
scale REGARDLESS of credit-path engineering. Distributed-scale hypothesis
sharpened: it must buy value-function capacity/data for the interaction term
(bigger critic, more same-gate-different-deck contrast data), not just more
steps of the same.

## 7.3 AUX MATRIX COMPLETE (2026-07-10 06:00) — first actor-side movement
| arm | draftref | gate-swap KL | critic ratio |
|---|---|---|---|
| portalgp1 (ctrl) | 46.9% | 0 exact | 0.17 |
| auxv1 (vboot .05) | 46.9% | 0 exact | 0.32 |
| auxd1 (sibdiff 2.0) | 44.8% | 0 exact | n/a |
| auxvd1 (both) | **50.0%** | **1e-5 FIRE+EARTH — FIRST NONZERO EVER** | **0.86** |
The terms compose: vboot sharpens boundary values; the differential then has
signal to amplify (auxd1 alone had nothing to amplify — the 7.2 post-mortem).
Embryonic but real (control replay is exactly 0). Decision rule satisfied →
auxvd45 (45M) launched 06:07 with full KL/critic trajectory suite. The open
question at 45M: does 1e-5 GROW (emergence curve) or plateau (floor artifact).

## 7.4 auxvd45 verdict (2026-07-10 15:05): 15M spark was a TRANSIENT; Goodhart at length
- draftref 33.3% @45M (was 50.0% @15M; portalgp45 46.9%, anneal45 46.4%).
- Final-ckpt gate-swap KL back to EXACTLY 0 (TV 0.0002-0.0003); the 1e-5
  FIRE/EARTH conditioning at 15M did not grow — it vanished.
- Mechanism: aux coefficients were CONSTANT (the design doc's "ride the
  shaping anneal" was not implemented) → 3x longer exposure to a
  critic-opinion reward → drafts optimize the critic's taste, not winning
  (Goodhart). The critic also lacks the gate x composition interaction
  (§7.2), so what got amplified was main-effect noise.
- If anyone revisits: anneal the aux coefs on the shaping schedule and only
  then rerun the 45M; but the deeper blocker remains the interaction-free
  critic (§7.2). Distributed-run recipe UNCHANGED: portalgp (aux terms NOT
  included).

## 8. S-QUEUE EXECUTION (post-campaign, tasks #7-12)
### 8.1 S1 s1auxann (annealed aux, 15M) — first readout (2026-07-11 01:15)
- **draftref 70.8% (96 eps)** — +20.8pp over control (portalgp1 46.9%),
  +20.8pp over the constant-coef auxvd1 (50.0%). CONFIRMING at 192 eps
  before belief (x192 eval running).
- **Final-ckpt KL NONZERO: 1-2e-6 in 3/4 elements** — first arm to END a
  run above the floor. Early peak 8.1e-5 @1.5M (highest yet), decay through
  4.6M mirrors auxvd45 but does not hit exact zero.
- critic ratio 0.72 at final ckpt.
- Mechanism read: aux as CURRICULUM — dense aligned pick credit while
  shaping is dense, force removed on the same schedule → improvements
  grounded by outcomes are kept, Goodhart drift never starts.
- x192 CONFIRMATION: 60.4% (192 eps; pooled 288 = 63.9%). Even the
  conservative read is +13pp over every prior arm. S1 GATE: PASSED.
  Promotion: s1auxann45 (45M, same knobs) launched; S2 implementation next.
### 8.2 s1auxann45 (45M) — the 15M breakthrough does NOT hold at length
- draftref 37.0% @45M (15M ckpt of same recipe: 60.4-70.8%); KL 0; critic 0.23.
- Aux fully annealed by ~8M ⇒ 8M→45M was effectively portalgp — yet lands
  10pp BELOW portalgp45 (46.9%). Hypotheses: (a) early seeding steers the
  meta into a worse long-run basin; (b) league meta cycle trough vs the
  references at 45M. Draftref trajectory over ep500-2500 running to locate
  the peak/decay shape. Verdict for the production recipe pends that curve —
  15M-horizon distributed runs would still favor s1auxann; 45M+ would not.
### 8.3 Draftref oscillation (2026-07-11): the meta cycle dominates single-point evals
45M draftref trajectories (96 eps/pt):
  s1auxann45: 8M=44.8 15M=47.9 23M=41.7 31M=54.2 38M=47.9 45M=37.0 (mean 45.6, peak 54.2)
  portalgp45: 8M=56.2 15M=40.6 23M=40.6 31M=60.4 38M=49.0 45M=46.9 (mean 49.0, peak 60.4)
Both oscillate over a ~20pp range with the league meta cycle. CONSEQUENCES:
(1) every single-checkpoint arm comparison in this campaign carries ±10pp
cycle noise on top of sampling CI — arm deltas under ~10pp are not
individually load-bearing (round-2 anneal-vs-ctrl survives via pooling +
mechanism evidence; the 45M portalgp/anneal/combo ordering does not).
(2) s1auxann's 63.9% pooled at 15M-total reflects cycle peak + end-of-schedule
lr polish — the recipe wins only at short horizons; at 45M its window mean is
BELOW portalgp's. (3) production protocol must include CHECKPOINT SELECTION
by external eval over a late window (peaks 54-60% are deployable in every
recipe) and end-of-run lr polish at the target horizon.
### 8.4 S2 s2outcome verdict (2026-07-11 15:45): NEGATIVE — keep the flat GP bonus
draftref 40.6% (cycle band), KL ~0, critic 0.16, and the tell:
**portal usage collapsed to 0.0038** (flat-GP arm: 0.076; no bonus: 0.030).
Outcome-grading pays ~27% of attempts under unskilled play → whiffs are pure
tempo cost → the policy portals LESS. "Pay for trying" was the load-bearing
exploration property of the flat bonus; grading strictness inverted the
incentive. Production recipe keeps AZK_PORTAL_GP_BONUS (flat).
### 8.5 S3 s3xgate verdict (2026-07-11 17:00): FIRST interaction-spread mover — PASS
- **Critic differential spread |m|/std 3.98 vs 6.4-6.5 (portalgp1/s1auxann)**;
  FIRE 1.72, WATER 2.81 — cross-gate replay measurably taught the critic
  deck-DEPENDENT gate values (the §7.2 blocker, moved for the first time).
- draftref 54.2% (best non-aux single point; cycle caveat), critic ratio
  0.48, masking healthy (~40 steps/epoch), KL ~0 alone (expected — no seed).
- GATE: PASS as the data lever. Composition s13combo (S1 annealed aux +
  S3 replay 0.15) launched — the design-session prediction: the seed only
  grows if the critic has the interaction term to keep feeding it.
### 8.6 QUEUED AS FINAL EXPERIMENT (user directive, 2026-07-11): gate-id drop test
Sequencing: run at the VERY END, after the full S-queue resolves and a best
composition passes the 45M bar — then rerun THAT final recipe with
policy.gate_id_embedding_enabled=false. Rationale: first get the model to
its best state via the successful experiment combination; the id-drop then
answers whether that state is reachable with generalizable inputs only.
Motivation: generality to NEW cards — the 16-d learned gate-id channel is
tied to card ids (untrained for unseen gates); the metadata/TEXT pathway
transfers. The June embedding collapse (sibling cos 0.99) was a SYMPTOM of
missing gradient pressure, not a cause — contrast-data pressure (S3) may let
the text projection differentiate on its own. Readouts: embedding geometry
(probe_embedding_geometry.py — do sibling cosines drop?), interaction
spread, KL, draftref vs the id-enabled twin. If parity holds without the id
channel, the production model prices UNSEEN gate cards from effect text
alone.
### 8.7 s13combo verdict (2026-07-11 20:45): interaction learning compounds; retention still unsolved
- **Interaction spread 2.14 — best yet** (S3 alone 3.98; baselines 6.4-6.5;
  WATER 0.86, LIGHTNING 1.66): aux seeding + contrast data COMPOUND on the
  critic side; the differential is now genuinely deck-dependent.
- Main-effect ratio fell to 0.07 — consistent with the constant main effect
  deflating as the critic redistributes it into the interaction term.
- Actor: KL ~0 at final (1e-6 WATER only) — seed decay unchanged from S1;
  retention remains THE open problem. draftref 42.7% (cycle band).
- S7 = this recipe at 45M (s7combo45), judged by window means + KL/critic/
  spread TRAJECTORIES per the oscillation protocol. S4/S5 deferred as
  optional builds; S8 (gate-id drop) terminal after S7.
### 8.8 S7 s7combo45 draftref window (2026-07-12 08:50)
8M=42.7 15M=56.2 23M=52.1 31M=46.9 38M=49.0 45M=47.9 →
**window mean 49.1%, peak 56.2% — parity with portalgp45 (49.0/60.4)**.
The conditioning levers (annealed aux + cross-gate replay) are externally
FREE at 45M. Verdict pends the KL + interaction-spread trajectories
(sweeps at ep1600/2930).
### 8.9 S7 s7combo45 FULL VERDICT (2026-07-12 14:40)
- External: window mean 49.1 / peak 56.2 — PARITY with portalgp45. The
  conditioning levers are free, not additive, at 45M.
- Seed: KL peak 1.9e-5 @3M, persists to ~9M (vs ~5-8M in S1/auxvd) — the
  contrast data extends the seed's life marginally; still 0 from ~10M on.
- Interaction spread TRAJECTORY: oscillates 2.5-18.7 with the meta cycle —
  the 15M spread readings (s13combo 2.14, s3xgate 3.98) were favorable
  samples of a noisy quantity, NOT a stable regime change. The spread metric
  needs windowed averaging like draftref.
- S7 CONCLUSION: single-box 45M cannot make sibling-draft conditioning
  durable. Distributed recipe: portalgp base + cross-gate replay (free,
  gives the critic contrast at scale) + annealed aux for short-horizon
  polish phases. S8 (gate-id drop, text-only generalization) is the last
  gate before the production spec is final.
### 8.10 S8 s8textonly FINAL VERDICT (2026-07-12 18:00) — text-only representation is production-viable
- draftref 46.9% (id-on twin 42.7%) — external PARITY without the id channel.
- **Embedding geometry: the text pathway differentiated exactly the pairs
  whose text differences carry compositional meaning** — LIGHTNING cos 0.972,
  WATER 0.983 (vs 0.988/0.990 id-on) — while FIRE 0.9993 / EARTH 0.9997 stay
  collapsed (their siblings differ in portal magnitude, not deck-fit; the
  interaction probes measured FIRE interaction = exact null). The collapse-
  as-pressure-symptom theory CONFIRMED: given contrast data (S3) the
  projection separates where separation pays.
- KL 0 / critic 0.15 / spread 5.38 — no conditioning change (as expected).
- PRODUCTION SPEC: gate-id embedding OPTIONAL. Text-only is viable and
  preferred for new-card generalization.
### 8.11 S9 live telemetry + arm semantics (2026-07-14 fork note)
- **frozen_ratio is ROW-level, not game-level**: 0.4 rows ⇒ **77.6% frozen
  GAMES** measured (frozen_matchup_fraction 0.776) ⇒ s9pfsp is the
  AlphaStar-extreme arm; queued s9pfsp02 (0.2 rows) ⇒ ~40% games = the
  moderate arm; an OpenAI-Five-equivalent (20% games) would be rows ≈ 0.1
  + PFSP — candidate third arm if the two don't separate.
- PFSP telemetry live (league/pfsp_picked_winrate logging); SPS 1.9k at the
  78%-league extreme (vs ~2.2-2.5k typical) — wall-clock cost mild as
  designed; trainable-row share ≈ 61% (vs 95% at rows 0.10).
- **USER DIRECTIVE (2026-07-14): take the BEST of the 0.4-row and 0.2-row
  PFSP arms** (SPS impact acceptable). Verdict metrics: H2H ladder
  monotonicity (primary), promotion-gate acceptances, draftref window,
  steady-state SPS. Winner's config goes into the production spec §20 and
  the final report gets a S9 verdict section like all other experiments.
### 8.12 S9 arm 1 (s9pfsp: 78% league games, PFSP-with-wipe-bug) H2H ladder
mid(7.7M) beats early(1.5M) **62.0%** — FIRST monotone segment ever measured
(portalgp's same comparison: 47.4%). But final(15M) still loses to mid
(34.4%) and early (31.8%): improvement now happens, then late-run regression
undoes it. Suspects for the remaining regression: (a) arm-1 PFSP stats wiped
every pool refresh (fixed 54ec27f — arm 2 tests this), (b) old-bucket
retention only 3/13 slots — pruned styles can't be prioritized, (c)
post-anneal sparse phase. draftref 46.9% final.
