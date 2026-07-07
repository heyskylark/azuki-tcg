# Azuki TCG — Deck-Building Training Research: Final Report

> STATUS: Round-2 complete (2026-07-06). Running work log: research-notes-01.md.
> All round-2 artifacts under results/round2_*; probes reusable on any checkpoint.

## 1. Executive summary

**Question**: does the model learn per-card strategies — building decks with synergy and
playing differently depending on what it drafted — and what unlocks that learning?

**Answer**: yes at the card / card-type / gate-family level, once the reward-shaping bias is
removed; not yet at the sibling-gate level (same-element gate pairs), for which we found and
fixed a representation root cause but 15M-step arms were not long enough to exploit it.

Headline numbers (round-2 arms, 15M steps each on the native stack, seed 42; draft-vs-ref
pooled over 288 argmax episodes/arm vs human reference decks):

| arm | intervention | draft-vs-ref | gate-swap KL | same-elem L1 excess | synergy pairs (p<.01, sup≥20%) |
|---|---|---|---|---|---|
| ctrl2 | none (control) | 36.8% | 0.0 | ≈0 (all 4 elements) | 6 |
| gateid1 | gate-id embedding | 39.6%* | 0.0 | ≈0 | 0 |
| anneal1 | shaping→0.05 by ~8M | **44.4%** | 0.0 | +0.005..+0.013 (3/4 elems) | **33** |
| combo1 | anneal+gateid+pick-eps | **43.8%** | 0.0 | EARTH +0.018, rest ≈0 | **26** |

*96 episodes only. anneal−ctrl = +7.6pp (~1.9σ); combo−ctrl = +7.0pp (~1.7σ); robust across
two eval seeds.

Evidence the model understands cards (details §3-§6):
1. **It builds decks that win against human references** when trained outcome-dominated —
   +7.6pp over control, reversing the June finding that longer self-play made external
   quality WORSE (45.8%→40.6%).
2. **It drafts synergies**: 26-33 high-confidence card pairs (permutation p<0.01, in ≥20%
   of that gate's decks) that co-occur specifically under one gate vs its same-element
   sibling — e.g. under weapon gates: Hidden Dagger + Tenraku (two weapons, p=0.003,
   sup 33%) and Piko (the weapon-synergy leader) + Black Jade Pawnbroker; under Hydromancy:
   Mocking Dummy + Young Shao with Δwin +0.15. Control shows 6 such pairs; gateid-only 0.
3. **It plays what it drafts**: weapon-gate games show 14-15% drafted weapon share (vs 13%
   availability) and in-play weapon-attach rates 3× every other gate (0.059-0.063 vs
   0.017-0.023), plus the highest attack rates (0.32); Water gates lead spell usage. In the
   forced-deck probe (same policy, controlled decks, availability-calibrated by a
   uniform-legal baseline) the policy exceeds availability by ~26-35% on weapon usage and
   wins 50% with a weapon-heavy deck vs 12% with spell/entity decks (anneal1).
4. **Behavioral confirmation of the reward-bias hypothesis**: as shaping faded in anneal
   arms, portal usage briefly ran at 2-3× control (0.082 at 6M) before the meta
   equilibrated — the mechanics ARE learnable; dense shaping had suppressed them.

The honest negative: the interventional gate-swap probe (replay the identical draft with
only the gate identity changed) still measures KL ≈ 0 in every arm — composition strategy is
conditioned on the ELEMENT/gate family, not the specific gate card, at 15M steps. gateid1
proves capacity alone is not sufficient (the channel went unused without value pressure);
anneal1 proves value pressure alone cannot create conditioning without training long enough
for second-order distinctions to pay. The recommendation for the distributed run is the
combo recipe with 3-5× the steps and the now-flat league cost.

## 2. Setup
- Environment: integrated draft+battle episodes (gate assigned → leader pick → 50 main picks,
  strictly alternating seats → battle), 8 gates / 8 leaders / 193-card metadata pool,
  18-deck reference pool (gate population weighted), 2-player zero-sum, PPO+LSTM
  (PufferLib v4 fork, Muon, bf16), league self-play (frozen pool 2/1/1, promotion gating).
- Hardware: single RTX 3090, 128GB RAM, 12 cores.
- Round-1 stack (June): legacy PettingZoo path, 332→722 SPS. Round-2 stack (this report):
  **deck-building draft ported into the native C vec path** (commit a13d66e) — packed obs
  with deck_context block, catalog-parity by construction, per-episode deck export, full
  legacy metric surface rebuilt Python-side; parity-tested bit-exact over all 102 draft
  steps. 3.7k SPS at pool 0, ~1.6-2.9k under league (now flattened, see §8).
- June checkpoints carry an encoder asterisk (trained on pre-fix slot-scrambled inputs);
  all round-2 comparisons are within-stack.

## 3. How the model learns deck building (trajectories)
- Random-start fingerprint: unique ≈ 38/50, copy-entropy ~0.97, cost ≈ 2.9 (availability).
- All arms: early quad-count spike-collapse oscillations (draft-meta churn), settling to
  unique 30-35, quads 0.9-1.3, avg cost 2.8-3.1 — decks are deliberately concentrated
  versus random but not degenerate.
- June's 31.7M baseline collapsed to a cheap-aggro attractor (cost 2.27-2.36 everywhere,
  weapons extinct) and its external quality DECLINED with training. Round-2 anneal arms do
  not show that collapse at 15M: weapon-gate identity strengthens instead.

## 4. Deck compositions per gate
- Element-level identity is strong everywhere (cross-element L1 ≈ 0.37-0.42).
- Gate-FAMILY identity (the decisive same-element test, vs a bootstrap noise floor):
  control sits at the floor in all four elements; anneal1 shows the first above-floor
  excesses (FIRE +0.013, EARTH +0.007, WATER +0.005); combo1's EARTH pair (Stonehaven vs
  Devotion) reaches +0.018 — small but real gate-pair composition divergence.
- Weapon gates draft weapons at/above availability (14-15% vs 13% prior) while every
  non-weapon gate suppresses them to 5-6% — a per-card-TYPE strategy driven by the gate's
  mechanic, recovered from June's extinction.

## 5. Playstyle differences across gates
anneal1 final bucket (behavior rates per gate):
- Stormchain/Surge (weapon gates): attack 0.322-0.323, weapon 0.059-0.063 — 3× others.
- Ragefire/Rushfire: lowest attack (0.233-0.249) at this stage of the meta cycle.
- Water gates: spell rate 0.040-0.047 (highest), matching their spell-rich pool identity.
- Forced-deck probe (E3): same checkpoint, controlled decks — the policy's weapon-usage gap
  between weapon-heavy and spell-heavy decks exceeds the uniform-legal availability gap by
  1.26-1.35×; spell conditioning is weaker (0.5-1.15×). The policy adapts play to its deck
  beyond what card availability forces.

## 6. What improved external battle performance
Draft-vs-reference (pooled 288 eps): control 36.8% → anneal 44.4% / combo 43.8%.
The single load-bearing intervention is the reward-shaping anneal (dense early → 0.05 by
~8M steps; per-env episode calibration matters — defaults would have been a silent no-op).
Mechanism (probes): shaping's garden-attack potential made entity-flood tempo locally
optimal; with it annealed, mechanics (portals/weapons/spells) briefly surged, the meta
re-equilibrated at higher external quality, and synergy structure appeared in drafts.

## 7. Ablation results (both rounds)
| Ablation | Arm | Steps | Outcome | Verdict |
|---|---|---|---|---|
| Baseline (June) | base-deckbuild-02 | 31.7M | Learned to draft; cheap-aggro attractor; draft-vs-ref DECLINED 45.8→40.6% | reference |
| A-GAMMA (June) | abl-gamma1 γ=1.0 | 12M | Faster pick commitment, same attractor, no gate identity | ✗ |
| Round-2 control | ctrl2 | 15M | 36.8% draft-vs-ref; KL 0; excess ≈0; 6 synergy pairs | reference |
| A-SHAPANNEAL | anneal1 | 15M | **44.4%**; 33 synergy pairs; weapon-gate playstyle 3×; first excess>floor | **✓ adopt** |
| A-GATEID | gateid1 | 15M | 39.6%; KL still 0 — channel unused without incentive | ✓ keep (needed later), insufficient alone |
| A-COMBO | combo1 | 15M | 43.8%; 26 pairs; EARTH excess +0.018; KL 0 | ✓ adopt as recipe; needs longer runs for pair-level identity |
| League window (perf) | leaguebench | 4M | see §8 | ✓ |

### Root causes found (before round 2)
1. **Representation collapse**: projected metadata embeddings of same-element gate pairs at
   cos 0.97-0.9995 on trained checkpoints; gate-swap KL ≡ 0. Fixed by a flag-gated 16-d
   learned gate-id channel (7ffca7b).
2. **Reward bias**: garden-attack shaping suppressed gate mechanics (portal/weapon/spell
   win-correlations ≈ 0/negative as played). Fixed by the shaping anneal.
3. Also fixed en route: checkpoint-loader ScalarRunningNorm crash; league row-granularity
   bug (worker-instance vs game) affecting matchups/reward decomposition/win-prob labels on
   native; league evaluator layout mismatch (native-trained policies on legacy eval envs).

### Dead ends
- γ=1.0 alone; torch.compile whole-policy (June, inductor OOM); gate-id embedding alone.

## 8. League cost fix (post-chain)
Per-distinct-frozen-policy batch splitting made SPS sag with pool size (3.7k→1.6k at
pool 4). OSFP-style windowed sampling (one frozen policy per 8-epoch window, 2a162cd)
makes rollout cost pool-size-independent. Benchmark: results in research-notes-01.md §7
(leaguebench run).

## 9. Evidence-bar scorecard (per the research directive)
- E1 gate-swap KL > 0: **not yet** (0 in all arms at 15M) — the one open item.
- E2 same-element composition excess > noise floor: **weak-positive** (anneal/combo).
- E3 deck→behavior conditioning beyond availability: **yes** (1.26-1.35× weapons; wins with
  weapon decks).
- E4 synergy pairs surviving permutation null with positive Δwin: **yes** (33/26 vs 6).
- E5 external validity non-declining and above control: **yes** (+7.6pp, reversal of June).

## 10. Recommendations for the distributed run
1. Recipe: anneal + gate-id embedding + pick-eps 0.05 (combo), γ=0.99, ue1, native path,
   league windowed sampling on.
2. Length: 45-75M steps/seed (3-5× round-2) — June showed gate-level signals emerge as
   transients ~12-14M; pair-level identity needs the longer horizon plus the id channel.
3. Anneal calibration is per-env-episodes: recompute WARMUP/RAMP for the distributed env
   count (rule: warmup ≈ 12% of expected per-env episodes, ramp ≈ 40%).
4. Track the evidence bar live: gate-swap KL at every checkpoint (probe_kl_trajectory.sh),
   noise-floor excess, synergy-pair counts, draft-vs-ref every 250 epochs (192+ episodes).
5. Consider oversampling same-element gate matchups (A-GATECOND) to sharpen the pair-level
   value differences the KL probe needs.

## 11. Literature applied
- ByteRL LOCM (2303.04096): E2E draft+battle ✓ (kept); forced random picks → pick-eps ✓;
  γ=1 ✗ (didn't transfer).
- ByteRL Hearthstone (2303.05197): eval discipline (argmax, seat-fair) ✓; leader-collapse
  warning (monitored; never manifested beyond transients).
- OpenAI Five: sample reuse (ue1) ✓; league frozen-ratio ✓ (now windowed).
- Suphx: reward-signal annealing philosophy → shaping anneal ✓ (the round-2 winner).
- Informed asymmetric critic: deferred (A-PRIVCRITIC still queued).
