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
| (TBD) | | | | |

### Helpful and why
(TBD)
### Dead ends and why
(TBD — include: torch.compile attempts; anything that destabilized training.)

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
