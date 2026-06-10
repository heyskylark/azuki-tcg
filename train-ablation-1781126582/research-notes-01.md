# Azuki TCG — Deck-Building & Training Ablation Research Notes (Part 01)

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

### Memory/speed optimization (2026-06-10, required to even run deck building)
Deck-context obs (30 deck slots + 80 candidates/row) OOM'd the 3090 at minibatch 8192: the policy
gathered [batch, slots, 1536]-dim raw text embeddings per zone before projecting. Replaced with a
per-forward **text-feature table**: encode the 194-card vocab once (name/effect/subtype → 56 dims
total), gather 56-dim rows per slot. Exactly equivalent math (per-card function precompute);
verified bit-equal over full vocab on CPU. ~27× less gather memory per zone. Also fixed py3.14
argparse '%' crash in train.py help string.

Run protocol: short runs 30-50M steps (~3-4h) for triage on 2 seeds where feasible; promote
winners to ≥100M confirmation; decision metrics (in priority order):
1. deckbuild_result winrate trends per gate + overall win0_abs_delta (vs frozen league),
2. deckbuild/main_* composition: unique count ↓ toward 12-20, quad_count ↑, per-gate type
   divergence (weapon share for LIGHTNING, spell share for Echoed Waves),
3. cross-gate L1 divergence from snapshots (analyze_decks.py),
4. losses/explained_variance, entropy trajectory, SPS.

## 4. Key questions to answer
- Does the model build legal-but-coherent decks (curve, type mix) per gate, or collapse to one deck?
- Do per-gate compositions diverge (weapons for LIGHTNING, spells for Echoed Waves, etc.)?
- Does deck quality improve win rate vs fixed reference decks over training?
- How do playstyles differ across gates (aggression metrics, attack frequency, game length)?
