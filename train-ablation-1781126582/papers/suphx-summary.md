# Summary: Suphx: Mastering Mahjong with Deep Reinforcement Learning

**Paper:** Li, Koyamada, Ye, Liu, Wang, Yang, Zhao, Qin, Liu, Hon (MSR Asia) — arXiv:2003.13590v2 (2020).

> **File provenance warning:** the local file `suphx-mahjong-oracle-1912.00126.pdf` is **not** the Suphx paper. arXiv 1912.00126 is "Contradictory Predictions" (Burdzy & Pal, math.PR — sharp bounds on disagreement of two conditional probabilities; unrelated to RL). The real Suphx paper (arXiv 2003.13590) was downloaded to `suphx-real-2003.13590.pdf` (28 pp.) and read in full; this summary is based on it.

## TL;DR

Suphx reaches superhuman level in 4-player Riichi Mahjong (10 dan on Tenhou, stable rank 8.74, above 99.99% of humans) via SL warm-start → self-play policy-gradient RL with three additions: (1) **global reward prediction** (a learned GRU credit-assigner that converts sparse game-level reward into per-round reward), (2) **oracle guiding** (train with perfect information in the *policy* input, then anneal it away with feature dropout), and (3) **run-time policy adaptation** (finetune on rollouts with sampled hidden information). Each component is ablated and adds measurable strength; the paper also reports that naive oracle distillation fails and explicitly recommends an oracle/privileged *critic* as a promising alternative.

## Core method and exact implementation details

### Models & features
- Five CNN decision models — discard (34-way), Riichi/Chow/Pong/Kong (binary) — plus a rule-based win-declaration model. Only the **discard model** is RL-trained; the other four stay at their SL weights.
- Input encoding: everything as **34×1 binary channels** (34 tile types). Private hand = 4 channels (channel n, column m = "hand has n copies of tile m"); categorical features = all-0/all-1 channels; **integer features bucketed** into channels. Discard model input 34×838; others 34×958. No pooling anywhere ("every column has semantic meaning").
- Architecture: 3×1 conv (256) → **50 residual blocks** (two 3×1 conv-256 each) → 1×1 conv head (Riichi/Chow/Pong/Kong add FC 1024→256).
- **Look-ahead features:** 100+ hand-crafted 34-dim feature planes from depth-first search over *own* discard/draw sequences (opponents ignored), encoding e.g. "discarding tile X can lead to a winning hand of score 12,000 after replacing 3 tiles." Hand-coded search knowledge injected as features, not as a planner.

### Training pipeline
1. **Supervised learning** from Tenhou expert logs: discard 15M samples → 76.7% top-1 accuracy; Riichi 85.7% (5M); Chow 95.0% (10M); Pong 91.9% (10M); Kong 94.0% (4M).
2. **Distributed self-play RL:** policy gradient with **importance sampling** to correct trajectory staleness (`π_θ/π_θ'` ratio × advantage, Eq. 1); parameter server + CPU Mahjong simulators + GPU inference engines; inference engines pull fresh weights regularly (~every 1000 mini-batches). 1.5M self-play games per ablation agent ≈ 44 GPUs × 2 days (final online Suphx ≈ 2.5M games).
3. **Entropy regularization with dynamic coefficient (Eq. 2–3):** gradient `∇J + α∇H(π)` with **α ← α + β(H_target − H̄(π))** — α is *increased when entropy falls below target and decreased above it*. Motivation: entropy too small → RL converges quickly and self-play stops improving; too large → unstable, high-variance policy.

### Global reward prediction (credit assignment across rounds)
- A game = 8–12 rounds; game reward is rank-based at the end; round scores are misleading (top players deliberately lose late rounds to protect rank-1).
- Reward predictor Φ = **2-layer GRU → 2 FC layers**, trained on top-human game logs by MSE to predict final game reward from the *sequence of per-round features* (round score, accumulated scores, dealer position, repeat-dealer/Riichi-bet counters).
- RL reward for round k = **Φ(x¹..x^k) − Φ(x¹..x^{k−1})** — i.e., a learned potential-difference shaping that distributes the terminal reward over rounds.

### Oracle guiding (the oracle-weight-decay scheme)
- **Oracle agent input** = normal features (own tiles, all open/discarded tiles, public scores/bets) **plus perfect features**: the three opponents' private tiles and the wall tiles.
- Procedure: train the oracle by RL on the full input, then **anneal the perfect features with element-wise Bernoulli dropout**: perfect features `x_o(s)` are multiplied by mask δ_t with `P(δ_t(i,j)=1) = γ_t`, and **γ_t decays from 1 → 0** during training (Eq. 5). At γ = 0 the oracle has *become* a normal agent.
- **Continual training after γ=0** needs two tricks or it is unstable and gains nothing: (a) **decay learning rate to 1/10**, (b) **reject state-action pairs whose importance weight exceeds a threshold**.
- **Failed alternative:** vanilla knowledge distillation oracle→normal "does not work well" — a limited-information student cannot mimic a perfect-information teacher.
- Alternatives flagged in the conclusion: (a) jointly train oracle + normal agent with distillation **while constraining the distance between the two policies** (preliminary experiments "work quite well"); (b) **"designing an oracle critic, which provides state-level instant feedback (instead of round-level feedback)... based on the perfect information"** — i.e., exactly a privileged critic, proposed as future work.

### Run-time policy adaptation (pMCPA)
- At round start (own hand known): sample opponents' hands + wall from the remaining tiles, roll out **K trajectories** with the offline policy (100K in their eval), **finetune the policy by basic policy gradient on those rollouts**, play the round with the adapted policy, reset to the offline policy next round. Parametric, so it generalizes beyond the simulated states; K need not be large.

## Main results and ablations

- **Offline ablation ladder** (each agent: 1.5M games training; eval: 1M games vs three under-trained SL agents; stable-rank with 1000× bootstrap of 800K games), read from Fig. 8:
  - SL ≈ **7.65** → RL-basic (round score as reward + entropy reg) ≈ **8.05** → RL-1 (+ global reward predictor) ≈ **8.25** → RL-2 (+ oracle guiding) ≈ **8.3–8.35**.
  - So: RL itself ≈ +0.4 dan, reward predictor ≈ +0.2 dan, oracle guiding ≈ +0.1 dan. All three stack; oracle guiding is the smallest but consistent increment.
- **pMCPA:** adapted RL-2 beats non-adapted RL-2 with **66% win rate** (hundreds of initial rounds tested). Qualitatively enables hand-conditioned risk-taking (e.g., choosing lower-probability/higher-score wins when needed to escape 4th place). Not deployed online due to rollout latency.
- **Online (Tenhou expert room, 5,760+ games):** record rank **10 dan** (first AI ever), **stable rank 8.74** vs Bakuuchi 6.59, NAGA 6.64, top-human macro-player 7.46 — ≈2 dan above prior AIs, above 99.99% of ranked humans.
- Style statistics: **lowest deal-in rate 10.06%** and **lowest 4th-place rate 18.7%** (4th place carries the big rank penalty in Tenhou) — the global reward predictor demonstrably induces rank-aware risk modulation (defensive play with a big lead in the last round, Fig. 9).

## Failure modes / cautions

1. **Naive distillation from a perfect-information teacher fails** — the information gap makes the teacher's behavior unmatchable; guidance must be gradual (feature decay) or constrained (bounded policy distance), or moved to the critic.
2. **The transition off perfect features is fragile:** continuing training after γ→0 without lr×0.1 + importance-weight rejection is "not stable and does not lead to further improvements."
3. **Entropy mis-tuning breaks self-play:** too low stalls improvement, too high destabilizes — hence the feedback controller on α rather than a fixed coefficient/schedule.
4. **Naive reward signals misattribute credit** in multi-segment episodes: per-round score punishes correct tactical sacrifices; game-level reward cannot differentiate well/poorly played rounds. A learned global-reward predictor is their fix.
5. Oracle guiding gives the *smallest* of the three component gains — perfect-information tricks are worth ablating but should not be expected to dominate reward-design fixes.
6. (Noted limitation) Their reward predictor ignores luck/difficulty of the deal; they propose conditioning the reward predictor on perfect information (initial hands of all players) to normalize for round difficulty — an unimplemented idea worth borrowing.

## Actionable ideas for Azuki TCG (most promising first)

1. **Global reward predictor for deck-building credit assignment.** Our episode = build + battle with one terminal reward. Train a small GRU Φ on (frozen-policy) self-play episodes to predict final outcome from the prefix of build picks (and battle phase summaries); use **ΔΦ per pick/turn as dense shaping reward**. This is Suphx's biggest single RL gain (≈ +0.2 dan) and maps directly onto our worst credit-assignment problem.
2. **Oracle-feature annealing as the head-to-head competitor to our privileged critic.** Arm A: current privileged critic. Arm B: feed opponent-hand/deck features to *policy and critic*, anneal with per-element Bernoulli dropout γ: 1→0 over a fixed fraction of training; at γ=0 drop lr ×0.1 and reject/clip high-IS-ratio samples (PPO's clipping partially covers this; add an explicit ratio-rejection threshold as an ablation knob). Suphx found this beats no-oracle; the privileged-critic paper (and Suphx's own conclusion) suggests the critic-only route is cleaner — measure both.
3. **Dynamic entropy targeting instead of a fixed/linear entropy schedule:** α ← α + β(H_target − H̄) per update, with per-head entropy targets for our 4 action heads (computed over masked-legal actions). Directly ablatable against our current entropy coefficient.
4. **Luck normalization:** condition a *training-only* reward predictor / advantage baseline on privileged draw information (e.g., both initial decks/draw order) so wins from lucky draws are rewarded less than wins from hard positions — Suphx's proposed-but-unimplemented extension, and a natural use of our existing privileged inputs that complements the privileged critic.
5. **Constrained oracle distillation (third arm):** jointly train an oracle policy and the normal policy, distill oracle→normal with a KL distance constraint between them (Suphx reports preliminary success). Cheaper than a league; useful if the privileged critic plateaus.
6. **Eval/inference-time adaptation (pMCPA analog):** at battle start, sample opponent hand/deck consistent with observations, roll out with the current policy, finetune a copy for that game. 66% head-to-head win rate in Suphx; expensive, so eval-only or for generating stronger league opponents.
7. **Reward-prediction features as observations:** Suphx's hand-crafted look-ahead features (win probability/score per candidate action) substantially boost the network. Analog: cheap C-engine rollout/heuristic scores per legal action (e.g., lethal-check, damage forecast) appended to observations — orthogonal to critic asymmetry.
8. **IS-based staleness handling** if we scale to asynchronous vec-envs: importance-weight the PPO loss against the behavior snapshot and *reject* extreme ratios rather than only clipping.
