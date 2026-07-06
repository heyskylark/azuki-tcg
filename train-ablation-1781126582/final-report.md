# Azuki TCG — Deck-Building Training Research: Final Report

> STATUS: SKELETON — being filled as experiments complete. Will be finalized at the end of the
> research program. See research-notes-01.md for the running work log.

## 1. Executive summary
(TBD: 1 page — did the model learn to build decks; which interventions mattered; headline numbers.)

## 2. Setup
- Environment: integrated draft+battle episodes (gate assigned → leader pick → 30 main picks →
  battle), 8 gates / 8 leaders / ~190 card pool, 2-player zero-sum, PPO+LSTM (PufferLib v4 fork).
- Hardware: single RTX 3090, 128GB RAM, 24 cores.
- Instrumentation added for this study: per-gate-card composition/result/playstyle metrics,
  cost-curve metrics, JSONL deck snapshots, JSONL run logs, draft-vs-reference eval,
  snapshot/runlog analyzers. Engine fixes that unblocked training: observer-ctx UAF,
  teardown zone access, IKZ token name collision (random decks exercised paths reference decks
  never hit).
- Speed work that sized the program: 332 → 722 SPS (text-feature table precompute,
  update_epochs 1, async vec batches). torch.compile attempted and shelved (inductor OOM).

## 3. How the model learned deck building
(TBD: timeline from baseline run — composition entropy/unique-count trajectories, when win rates
per gate diverge, copy-count histograms, element share, cost curves. Include analyze_decks bucket
tables at 4-6 time points.)

## 4. Deck compositions per gate
(TBD: per-gate top cards, type shares, costs at end of baseline + best ablation; whether
LIGHTNING→weapons, EchoedWaves→spells, Rushfire→cheap aggro, Stonehaven→defensive emerged;
within-gate deck similarity (Jaccard) and cross-gate divergence (L1) trends.)

## 5. Playstyle differences across gates
(TBD: per-gate behavioral rates (attack/spell/weapon/portal/noop), episode lengths, leader health
margins; correlation with gate effects; examples from playback.)

## 6. What improved battle performance after decks were built
(TBD: which changes raised deckbuild_result win rates and draft-vs-reference win rate.)

## 7. Ablation results
| Ablation | Arm | Steps | Outcome vs baseline | Verdict |
|---|---|---|---|---|
| A-GAMMA | abl-gamma1 (γ=1.0, λ=.97) | 12M | Faster early pick commitment (quad 7.5 vs 1.2) but same element-level attractor; no gate divergence; draft-vs-ref 36% (≈baseline, CIs overlap); value_loss +28%; mechanics usage decayed | ✗ not the lever alone |
| Round 2 | ctrl2 (new-stack control) | 15M | (running 2026-07-06) | |
| Round 2 | anneal1 (shaping→0.05 by ~9M) | 15M | (queued) | |
| Round 2 | gateid1 (gate-id embedding) | 15M | (queued) | |
| Round 2 | combo1 (anneal+gateid+pick-eps .05) | 15M | (queued) | |

### Root cause found before round 2 (2026-07-06)
1. **Representation collapse**: projected metadata embeddings of same-element gate pairs are
   near-identical on trained checkpoints (F/E pairs cos 0.9995) — the effect-text features that
   distinguish gates don't survive the learned projection. Interventional replay probe: pick
   distributions have KL ≈ 0 to a same-element gate swap. The policy cannot condition on the
   gate card. Fix: flag-gated 16-d learned gate-id embedding (gateid1/combo1 arms).
2. **Reward bias**: garden-attack potential shaping makes entity-flood tempo locally optimal;
   portal/weapon/spell usage uncorrelated-or-negative with winning as played (probe on
   snapshots). Fix: shaping anneal to 0.05 (anneal1/combo1 arms). NOTE: anneal counts PER-ENV
   episodes (~46/env per 12M steps @720 envs) — June defaults would have been a silent no-op;
   canary-verified env-var plumbing to workers (scale 1.0→0.525 on schedule).

### Evidence bar for "understands per-card strategy" (probes in this dir)
- E1 probe_gate_kl: pick-distribution KL to a same-element gate swap ≫ 0 (ctrl2 ≈ 0), growing
  over checkpoints. E2 gate_identity_probe: same-element L1 excess over bootstrap noise floor
  > 0, growing. E3 probe_deck_behavior: forced-deck conditioning ratio ≫ 1 vs uniform-legal
  availability baseline. E4 synergy_lift sibling-differential pairs with permutation p < .05
  and dwin > 0. E5 draft-vs-ref ≥ control and non-decreasing over training.

### Helpful and why
(TBD)
### Dead ends and why
- γ=1.0 alone (see table): stronger pick credit amplifies convergence to whatever the battle
  meta rewards — with biased shaping, that's the same cheap-aggro attractor, reached faster.
- torch.compile whole-policy (June): inductor OOM/recursion; shelved. Round-2 stack compiles
  train-side subgraphs only (fixed-deck native config), not used for deck-building arms yet.

## 8. Literature applied
- ByteRL LOCM (2303.04096): end-to-end draft+battle; OSFP; forced random picks; γ=1.
- ByteRL Hearthstone (2303.05197): ratio low-clip; eval discipline; leader-collapse warning.
- OpenAI Five (1912.06680): sample reuse; 80/20 league; surgery; entropy/lr anneals.
- Informed asymmetric critic (2509.26000): privileged critic conditioning rules; signal selection.
- Suphx (2003.13590): oracle annealing; global reward predictor; entropy controller.
(TBD: which transferred, which didn't, with evidence.)

## 9. Future work
(TBD: compile subgraphs; OSFP-style league; oracle deck evaluator for critic; 50-card decks;
cross-set generalization (the text-embedding card repr was built for this).)
