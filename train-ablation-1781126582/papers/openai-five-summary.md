# Summary: Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five)

**Paper:** OpenAI (Berner et al.), arXiv:1912.06680v1 (Dec 2019, rev. Mar 2021). 66 pp.; read in full, including all appendices A-Q.

## TL;DR

OpenAI Five beat the Dota 2 world champions with *scaled-up but otherwise standard* PPO+GAE over a single-layer 4096-unit LSTM (159M params, LSTM = 84% of params), trained 10 months via asynchronous self-play (80% current / 20% quality-weighted past agents) on batches of ~1-3M timesteps across up to 1536 GPUs. The paper's enduring lessons are mostly *not* the scale: (1) data quality dominates — keep staleness <1 policy version and sample reuse ~1; (2) hand-shaped dense rewards + zero-sum symmetrization + "team spirit" annealing (individual→shared credit) work and were barely changed in 10 months; (3) hyperparameter changes should be few, gradual, and scheduled (Rerun used only 4: LR, entropy, team spirit, GAE horizon); (4) "surgery" (function-preserving model/obs/action changes) enables continual training across env changes at ~20% of from-scratch cost, at the price of a slightly lower final plateau; (5) their draft phase was solved *without* RL actions — a win-probability head evaluated lineups and a minimax/DP drafter picked heroes.

## Architecture (Figure 1, 17, 18; Appendix H)

- Per-hero replica networks with **tied weights**; obs nearly identical per hero, plus a controlled-hero embedding. (For our 2-player TCG: one shared policy with seat/role features is the analog.)
- Obs processing by data type: continuous → running mean/std normalization, clipped to (-5,5), **no learned processing**; categorical → embeddings; spatial → 2-layer conv; unordered sets (units/items/modifiers) → "Process Set" = shared 2×FC → max-pool. Weights shared across allied/enemy sets.
- **Single-layer 4096-unit LSTM** core (grown 2048→4096 by surgery mid-run). Final model 158,502,815 params.
- Action space factorized: primary action (≤30, avg 8.1 *available* after rule-based filters) + parameters: Delay(4), Unit Selection(189), Offset(81). Up to ~1.8M combinations; ~8k-80k legal at a step.
- **Pointer-attention targeting:** unit-selection head = dot product of FC(LSTM state) against per-unit embeddings carried alongside the LSTM → softmax. Available-action IDs are embedded and dot-producted similarly (only legal actions scored).
- **Masking:** unavailable actions filtered before softmax; parameter heads not read by the chosen primary action are masked out of the loss "since their gradients would be pure noise." (We already do per-head masking — keep it.)
- **Rare-action heads split:** Teleport targeting got its own head because a shared unit-selection head drowned out the rare action's learning signal. Same for ward-placement offset. (Analog: separate heads/embeddings for rare card mechanics.)
- Value function = a linear projection of the same LSTM state (shared trunk, shared gradients). Value loss weight applied **after reward normalization** (running std estimate).
- Auxiliary supervised heads from LSTM state: **win probability** (passes a very small gradient into the trunk), net-worth rank, team objectives (stop-gradient). Trained with bootstrapped labels: last segment gets ground truth, earlier segments use the model's own end-of-segment prediction (Eq. 10).

## Exact PPO/LSTM hyperparameters (Appendix C, Table 2)

| Param | Rerun | OpenAI Five | small-Baseline |
|---|---|---|---|
| Frameskip | 4 | 4 | 4 |
| LSTM unroll (BPTT) | 16 | 16 | 16 |
| Samples/segment sent to optimizer | 16 (=256 steps) | 16 | 16 |
| Optimizer GPUs | 512 | 480↔1536 | 64 |
| Batch/GPU (samples of 16 steps) | 120 | 120↔128 | 120 |
| **Total batch (timesteps)** | 983,040 | 983k↔3,146k | 122,880 |
| LSTM size | 4096 | 2048→4096 | 4096 |
| **Sample reuse** | 1.0↔1.1 | 0.8↔2.7 | 1.0↔1.1 |
| **Team spirit** | 0.3→0.8 | 0.3→1.0 | 0.3 |
| **GAE horizon** | 180s→360s | 60s→840s | 180s |
| GAE λ | 0.95 | 0.95 | 0.95 |
| PPO clip | 0.2 | 0.2 | 0.2 |
| Value loss weight | 1.0 | 0.25↔1.0 | 1.0 |
| **Entropy coef** | 0.01→0.001 | 0.01→0.001 | 0.01 |
| **Learning rate (Adam)** | 5e-5→5e-6 | 5e-5↔1e-6 | 5e-5 |
| Adam β1/β2 | 0.9/0.999 | same | same |
| **Past-opponent fraction** | 20% | 20% | 20% |
| Past-opponent quality LR η | 0.01 | 0.01 | 0.01 |

- Optimizer detail: Adam + truncated BPTT over 16 steps; per-parameter gradient clip to ±5√v (v = running 2nd moment of the *unclipped* gradient); NCCL allreduce; 32 gradient steps per published parameter version.
- GAE horizon ↔ gamma via H = T/(1−γ), T=0.133s/step: 180s→γ=0.99926, 360s→0.99963, 840s→0.99984. Resuming a trained agent with longer horizon kept improving win rate up to 6-12 min horizons (Fig. 6) — long-horizon credit assignment worked, with diminishing returns.
- **Why these values (their stated reasoning):** many were "set for historical reasons or on preliminary investigations without full ablations" — they explicitly disclaim optimality. λ=0.95 forces reward smoothing over ≫20 steps, hence ≥256-step segments. Unroll 16 was an engineering tradeoff, not ablated. They appeared to ablate only what mattered: batch size, staleness, sample reuse, entropy, team spirit, horizon.
- **Rerun's entire schedule (Fig. 7)** — only 4 changes over 42 days, each applied gradually over 1-2 days: team spirit 0.3→0.8 @ iter ~15k (TS 210); horizon 180→360s @ ~23k (TS 232); entropy 1e-2→1e-3 @ ~43k (TS 245); LR 5e-5→5e-6 @ ~54k (TS 258). Pre-planned moves to horizon 840s / spirit 1.0 / LR 1e-6 were never needed.

## Reward shaping + team spirit (Appendix G, Table 6)

Reward designed **once at project start from domain intuition, barely changed for 10 months** — "our initial choice of what to reward worked fairly well."

- Key weights: Win **5** (team); hero death **-1** (solo); XP +0.002, gold gained +0.006 (not revoked when spent), health change ±2×fraction-of-max (quartic ramp: (x+1−(1−x)⁴)/2 — low health matters more), mana ±0.75, last-hit **-0.16** and kill **-0.6** as *counterweights* because the gold/XP rewards for those events are already very high (net kill reward ~0.4); buildings 2.25-6 (team), 2/3 paid linearly with building damage, 1/3 lump on destruction; ancient HP ±5 (team).
- **Zero-sum enforcement:** subtract the enemy team's mean reward from each hero's reward each tick.
- **Game-time renormalization:** all non-win/loss rewards scaled by 0.6^(T/10min) to counter end-game reward inflation (analog: late TCG turns have bigger swings — decay shaped rewards by turn number).
- **Team spirit τ:** rᵢ = (1−τ)ρᵢ + τ·ρ̄. τ=0 selfish → low gradient variance, clean individual credit; τ=1 = true team objective. **Annealed 0.3→1.0 (Five) / 0.3→0.8 (Rerun).** Ablation (Fig. 29): very early (TS<125) τ=0 trains fastest; by TS150-175 τ=0.5 is best (~1.35× speedup vs ~0.55× for τ=0); hypothesized τ=1.0 best late. General principle: **start with low-variance proxy credit, anneal toward the true objective.**
- **Sparse-reward ablation (Fig. 16):** win/loss only (1-hour horizon, γ=0.99996) still learns to TS ~155 vs ~200 for shaped baseline (scripted bot = 100). Shaping is a large sample-efficiency win, not strictly required for competence.

## Self-play opponent mixing (Appendix N)

- **80% games vs current self, 20% vs past versions** — the 20% exists to avoid *strategy collapse* (forgetting how to beat older/diverse strategies, cyclic counters).
- Past-opponent manager: each past agent i has quality qᵢ; sampled p_i ∝ e^{qᵢ} (softmax). Current agent snapshot added **every 10 iterations**, initialized at the max existing quality. After each rollout: if past opponent wins → no update; if current agent wins → qᵢ ← qᵢ − η/(N·pᵢ), η=0.01. So beaten opponents decay until rarely sampled; the spread of the distribution self-tunes to learning speed (fast progress → narrow recent pool; plateau → broad pool).
- This is essentially a cheap prioritized fictitious-self-play; no league of distinct exploiter agents (contrast AlphaStar). They note AlphaStar's privileged-information value function as a promising direction they didn't use (we already do privileged critic).

## Exploration (Appendix O)

- **Entropy bonus (Fig. 28):** 0.01 best in early training; 1e-3 comparable; 0 still learns but slower (~0.2-0.4× speedup); **0.1 catastrophic** (near-zero speedup). Annealed 0.01→0.001 mid-run in both Five and Rerun.
- **Team spirit** is treated as an exploration/credit knob (above).
- **Environment randomization** (three stated goals: shorten lucky-sequence discovery, break repetitive local minima, robustness to diverse human strategies):
  - Initial-state perturbations (level/XP/gold/armor/speed/regen/stats randomized in rollouts).
  - **Roshan health randomized 0..full** — an explicit curriculum making a hard subtask sometimes-easy so the agent learns to attempt it at all.
  - Hero lineup randomly sampled per game; item builds randomly perturbed around scripted builds (the only place human data enters the system).
  - Cautionary result (Fig. 30): their hand-added "lane assignment" randomization + penalty ablated to ~no benefit — randomizations should be ablated, they accumulate as superstition.
- **Hero pool size (Appendix P):** training on 80 heroes vs 17 only ~20% slower in early training (speedup ~0.8) — diversity of "characters" costs surprisingly little. (Analog: training across many decks/archetypes simultaneously is cheap; don't over-narrow the deck pool.)

## Batch size / staleness / sample reuse (Sections 4.3-4.4, Appendix M)

Run-to-run TrueSkill noise at small scale: ~±2 TS (4 identical baseline runs) — they gate all speedup claims on this.

- **Batch size (Fig. 5a/21):** speedup vs 123k-step baseline measured at fixed TS thresholds. 983k batch → ~2.5× speedup at TS175; 1966k → ~3.2×. **Sublinear in compute** at early-training thresholds, and benefit grows for later thresholds (TS175 > TS125 > TS100). They did *not* retune LR per batch size (acknowledged caveat). Verdict: bigger batch = faster wall-clock learning but worse compute-efficiency per sample, in the regime they measured.
- **Staleness (Fig. 5b/22):** staleness = optimizer version − behavior version. A few versions of staleness is fine; **~8 versions ≈ 0.5× speed; ~32 versions can prevent learning**. Final system targeted **staleness 0-1** by shipping 30s data chunks (256 steps) and refreshing rollout params ~every minute. Gradients from old params "were often useless or destructive."
- **Sample reuse (Fig. 5c/24):** reuse = optimizer consumption rate / rollout production rate. **Reuse 2-3 → ~2.5× slowdown (speedup ~0.4); reuse ~6-8 → may fail outright (converged < TS 75)**. Final target ~1. Reuse **0.5** (2× rollout production, sampling buffer means some samples never used) was *slightly better* than 1.0 after ~5k iterations — over-producing data helps a bit because random buffer sampling reuses some samples even at reuse 1.
- **Async vs sync (Fig. 26):** fully synchronous (staleness 0, reuse ≤1) matches async **per iteration**; async is ~3× faster in wall time purely from hardware utilization. So async is an engineering optimization, not an algorithmic one.
- Their summary: **"high quality data matters even more than compute consumed; small degradations in data quality have severe effects on learning."**

## Surgery (Section 3.3, 4.2; Appendix B)

Tools for continuing one long training run across model/obs/action/env changes (~one surgery per 2 weeks; >20 successful over 10 months; Table 1 lists all, incl. LSTM 2048→4096 at iter ~91k).

- **Widening an FC layer:** Ŵ₁=[W₁;R()], B̂₁=[B₁;R()], Ŵ₂=[W₂ 0] — new *outgoing* weights zero (function preserved), new *incoming* weights random (symmetry broken). Zero-init only the minimal set.
- **Growing the LSTM (recurrent, can't be exact):** new weights random at a *much smaller magnitude* than existing ones; scale chosen empirically as the largest that didn't drop TrueSkill.
- **New observations:** Ŵ=[W 0] — zero columns for new inputs; exact function preservation w.r.t. the old encoder.
- **New actions / env changes:** **anneal in**: 0% → 100% of rollout games use the new action/env version; if TrueSkill drops, revert and anneal slower. Un-annealed buyback control caused a skill drop requiring "repeating" large compute.
- **Removing parts: effectively impossible** in this framework — deprecated obs stay as constants forever.
- **Post-surgery restart: LR=0 for the first several hours** so Adam moments and the rollout distribution re-equilibrate before real updates.
- **Past-opponent pool must be converted with the same surgery**, else frozen opponents degrade and poison the pool.
- **Verification (Rerun):** from-scratch retrain in the final env took 2 months / 150 PFLOPs/s-days = **20% of Five's compute**, and surpassed Five (>98% winrate vs it). Surgery saved ~10× vs always-restarting, but the surgered model **plateaued below** what from-scratch achieved. Use surgery to keep iterating; budget one final from-scratch run with the lessons learned.

## Bloopers worth internalizing (Appendix Q)

- **Q.1:** Frantic manual hyperparameter tuning under deadline ("designing skyscrapers") was counterproductive; prefer few, scheduled, gradual changes.
- **Q.2 Zero team-spirit embedding:** zeroing a vestigial 128-param learned embedding raised winrate ~55% while leaving *shaped* reward unchanged — the optimizer couldn't find the improvement because **the shaped reward was blind to it**. Proxy-reward/true-objective gaps are real; periodically evaluate on pure win rate.
- **Q.3 Path dependency (Divine Rapier):** one high-variance-reward game element (transferable high-value item) put Rerun into a negative feedback loop (value function couldn't stay reliable); banning it fixed training. Order of feature introduction and reward-variance spikes matter. (Analog: cards with huge swingy reward variance may need delayed introduction or reward clamping.)

## Scale-dependent vs scale-independent

**Likely transfers to single-GPU (scale-independent):**
- Staleness <1-2 policy versions; sample reuse ≈1-2 max (their reuse curve is about *algorithmic* off-policy degradation, not hardware). Our `update_epochs=2` is exactly their reuse-2 point (~0.4× speed): worth ablating 1 vs 2.
- Reward-shaping structure: zero-sum symmetrization, counterweight rewards, time-decay of shaped terms, win bonus dominant; sparse-only works but ~4× less efficiently.
- Team-spirit-style anneal from low-variance proxy credit → true objective.
- Entropy ~0.01 → 0.001 late anneal; LR drop late; few gradual scheduled hyperparameter changes.
- 80/20 self-play with softmax-quality past sampling (η=0.01) — costs nothing at our scale.
- Architecture patterns: pointer/attention action targeting over entity embeddings, masked heads, separate heads for rare actions, set-pooling encoders, shared value head with normalized rewards, win-prob auxiliary head with bootstrapped segment labels.
- Surgery recipes (zero-out new outgoing weights, LR=0 warmup, anneal-in new actions/obs, convert frozen opponents) — directly usable when we change obs/action layouts mid-run.
- Long-horizon finding: raise γ only after basic competence (resume-with-longer-horizon worked; from-scratch long horizon was not what they did).
- Curriculum-by-randomization (Roshan-health analog) and initial-state randomization; ablate every randomization you add.
- Diversity (hero pool analog: deck pool) is cheap: only ~20% early slowdown for ~5× pool.

**Likely does NOT transfer (scale-dependent):**
- Absolute batch size (1-3M timesteps) and "bigger batch → faster": at fixed compute (one 3090) bigger batches trade against iterations; their speedup was sublinear *even with* compute scaled proportionally, and they note it may matter more later in training. On one GPU, batch size should be tuned for gradient-noise/throughput, not copied.
- LR 5e-5 is tied to their enormous batch + 159M model; our LR must be tuned independently.
- Asynchronous rollout/optimizer split, forward-pass GPU pools, 0.5 sample reuse via doubled rollout workers — pure hardware-utilization wins; with synchronous PufferLib vec-envs we already have staleness ≈ 0, which their Fig. 26 shows is per-iteration optimal anyway.
- 4096-unit LSTM: at their scale the LSTM was 84% of 159M params trained on ~10¹¹ frames; on ~10⁸-10⁹ frames a 4096 hidden size is likely over-parameterized — capacity should be ablated, not inherited (their own 1v1 predecessor and the surgery history show 2048 worked for months).
- 10-month continual surgery cadence, TrueSkill infra with 83 reference agents (we can run a 5-10 checkpoint ladder instead).
- 20,000-step episodes / 840s horizons: our episodes are 10²-10³ steps; γ=0.99-0.999 already covers them. Their horizon *schedule* idea transfers; the values don't.

## Top 10 ablation/training ideas for the Azuki single-GPU setup

1. **Sample reuse ablation: `update_epochs` 1 vs 2 (vs 3).** Their cleanest scale-independent curve says reuse 2-3 ≈ 2.5× slower per sample consumed; PPO clipping does not save you. If 1 epoch matches 2 at equal wall-clock SPS, take the free win. (Scale-independent.)
2. **Team-spirit-style annealed credit for the draft phase.** Treat draft-pick shaped rewards (e.g., curve/synergy heuristics or per-pick ΔwinProb from the aux head) as the "solo reward" and final win as "team reward": r = (1−τ)·shaped + τ·terminal, anneal τ 0.3→1.0 over training. Mirrors their variance-reduction-then-true-objective recipe. (Scale-independent.)
3. **Adopt their opponent-manager exactly:** enable league with 20% past-opponent games (ours: `frozen_ratio=0.075` — ablate 7.5% vs 20%), snapshot every N updates, sample p∝e^q, q ← q − η/(Np) on losses only, η=0.01, new snapshots start at max q. Cheap insurance against strategy collapse / draft-meta cycling. (Scale-independent.)
4. **Reward time-decay and counterweights:** decay all shaped battle rewards by ~0.6^(turn/k) so late-game swings don't dominate gradients, keep win bonus undecayed and ≥5× any shaped term, and add explicit negative counterweights where an event already pays out indirectly (their kill −0.6 pattern; e.g., if destroying a unit grants tempo features, tax the event reward). (Scale-independent.)
5. **Entropy + LR schedule, not constants:** our config holds ent 0.01 flat; replicate their 0.01→0.001 anneal (start ~60-70% of run) and a late LR /10 step. Also ablate entropy {0, 1e-3, 1e-2, 1e-1} once — expect 1e-2 ≈ 1e-3 > 0 ≫ 1e-1. (Scale-independent.)
6. **Gamma/horizon schedule:** start γ=0.99 for fast early credit, resume-anneal to 0.997-0.999 once win rate vs scripted/early checkpoints plateaus, so draft picks (early steps) receive terminal credit across the full episode. Their Fig. 6 shows lengthening horizon on a trained agent keeps paying. (Scale-independent; their absolute γ values are not.)
7. **Win-prob-head drafter baseline (their Appendix D.2 trick):** before investing in RL draft actions, evaluate draft strength by querying our existing `win_prob_aux` head at battle start over candidate decks/picks with greedy/minimax selection. It both validates the head's calibration and gives a strong non-RL draft baseline to beat. (Scale-independent; they solved drafting this way at zero RL cost.)
8. **LSTM capacity ablation: 4096 vs 2048 vs 1024.** Their 4096 served 159M params and 180 days of data; on one 3090, halving the LSTM likely buys 2-3× SPS at no skill cost in our data regime. If a later upgrade is needed, use their grow-surgery (small-random new weights, LR=0 warmup) instead of restarting. (Their 4096 is scale-dependent; the surgery recipe transfers.)
9. **Batch-size sweep with LR retune per point** (e.g., 16k/32k/64k/128k timesteps): expect sublinear returns; pick the knee that keeps the 3090 saturated rather than copying big batches. Flag: their "bigger is better" finding is the most scale-dependent result in the paper — they added GPUs per batch doubling; we can't. (Scale-dependent.)
10. **Randomization-as-curriculum for hard subtasks + ablate every randomization:** randomize starting life/hand/IKZ or deck-pool matchups per rollout (their initial-state + hero-lineup randomization; pool diversity cost them only ~20% early speed), add a Roshan-style easy-mode randomization for rarely-explored mechanics (e.g., gate/ability usage), and re-ablate each randomization later — their lane-assignment hack turned out useless. Watch for a Rapier-style high-variance card destabilizing the value head; if value loss spikes track one mechanic, gate or delay it. (Scale-independent.)
