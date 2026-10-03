# Summary: Mastering Strategy Card Game (Legends of Code and Magic) via End-to-End Policy and Optimistic Smooth Fictitious Play

**Xi, Zhang, Xiao, Huang, Deng, Liang, Chen, Sun (ByteDance), arXiv:2303.04096v1, Mar 2023 ("ByteRL")**

Won 1st place in BOTH tracks of the COG2022 LoCM competition: 84.41% average winrate on LoCM 1.5 (official track), 94.56% on LoCM 1.2 (bonus track). Internal re-evaluation: 0.842 avg head-to-head winrate vs all COG2022 1.5 submissions, 0.943 vs all COG2021 1.2 submissions.

---

## 1. Problem setup and draft/battle integration

LoCM is a two-player zero-sum imperfect-information strategy card game with **two stages in one game**:

- **CB (Card-deck Building) stage.**
  - *LoCM 1.2*: card pool of 160 fixed cards; 90 cards are randomly selected per game and split into 30 rounds; each round the player picks 1 card to add to the deck (arena/draft mode — explicitly compared to Hearthstone Arena). Opponent's CB observations and actions are hidden.
  - *LoCM 1.5*: 120 **randomly generated** cards per game (procedural attributes, no fixed card identities); the player must select 30 of them. The authors deliberately **serialize this into 30 sequential steps, picking one card at a time**, so that one unified architecture handles both 1.2 and 1.5 — only the card-selection action mask differs by version.
- **BT (Battle) stage.** Standard SCG battle: mana, creatures on two lanes, spells, keep own HP positive. AOE effects added in 1.5.

**Draft state/action representation (the key part):**
- CB observation (information state) = **all candidate cards that can currently be selected + all cards already selected into the deck**, at every CB step.
- CB action space = **one categorical distribution over the candidate cards** (pick exactly one card per step), with a legality mask.
- BT observation = all self hand cards, **all self deck cards**, cards on both lanes, scalar features for both players (HP, mana, number of remaining deck cards). Obs/action design borrowed from Ronaldo Vieira's `gym-locm`.
- BT action space = a single flat categorical over all possible actions, with a 0/1 action mask zeroing illegal actions each step.

**Integration:** the whole game (CB steps + BT steps) is **one POMDP episode / one trajectory**, trained end-to-end ("E2E"). A stage indicator δ (1 at CB, 0 at BT) gates which sub-policy produces the action distribution:

```
π_θ(·|s) = δ · π_θCB(·|s) + (1−δ) · π_θBT(·|s)
```

This replaces the previous standard of **Alternating Training (AT)** (deck built by a separate evolutionary/Bayesian/scoring method; battle by MCTS or RL; the two trained alternately), which the paper shows to be markedly worse (Sec. 5 below).

## 2. Network architecture (Fig. 3)

- **Shared per-card embedding (the "cards embedding")**: a small fc stack (2 fc layers in the figure) that embeds each card's attribute vector. **The same embedding layer is shared by the CB policy and the BT policy** so card representations are consistent across stages. (Important for 1.5, where cards are procedurally generated, so the encoder must work from attributes, not card IDs.)
- **CB branch**:
  1. Embed all candidate cards in the pool with the shared card embedding.
  2. Multiply by the **"card selected mask"** (1 = already in deck) and **mean-pool** → a "deck-cards-so-far" embedding (the "selected cards feature"). This is **updated recursively at each CB step** as picks accumulate.
  3. Concatenate per-card embeddings with the deck-so-far feature → fc → elementwise-mask with **"card can be selected mask"** → fc → **CB action head** (per-candidate-card scoring/logits).
- **BT branch**:
  1. BT observation split into sub-groups: Hand, Deck, My Lane, Oppo Lane, Player0 scalars, Player1 scalars. Hand and Deck cards go through the **shared card embedding + mean pooling**; the other groups through fcs.
  2. Concatenate all group features **plus the (now frozen) deck-so-far "selected cards feature"** as context → fc → fc → **LSTM (256 hidden states)** → fc → **BT action head** (flat masked categorical).
- **Single value head for both stages**: input = concat of the CB-extracted feature (selected cards feature) and the BT-extracted feature → fc → V(s). Value is predicted at every time-step in both stages.
- **Credit assignment across stages**: because of LSTM BPTT and value bootstrapping, **the BT win/loss reward "penetrates backwards" and updates the CB parameters θ_CB** — that is the mechanism by which deck-building learns from battle outcomes.

Reported sizes: LSTM 256; fc sizes not given. No attention/transformer — just embeddings + mean pooling + fcs + LSTM.

## 3. Training algorithm: OSFP + DRL

### Theory (Sec. 2.1)
- Goal: Nash equilibrium of the 2-player zero-sum game. Smooth Best Response (SBR) = best response with a strongly-convex regularizer ψ (negative entropy, applied per information state, weighted by reach probability — Eq. 22).
- **Smooth Fictitious Play (SFP)**: best-respond to the mixture of all historical policies F̄_k = Σ_t F_t. Only *average-iterate* converges (must maintain an average policy — awkward with neural nets).
- **Optimistic SFP (OSFP)**: best-respond to **η(F̄_k + F_k)** — the historical mixture **plus the latest payoff vector counted one extra time** ("optimistic prediction of opponent's future policy"). Equivalent to Optimistic Mirror Descent; inherits **last-iterate convergence** → you can deploy the latest checkpoint directly, no policy averaging.
- Practical consequence (Eq. 16–18): the SBR objective is a weighted sum over opponents with weights α_i = 2 for i = k (latest) and 1 for i < k, estimated by **sampling the opponent from a multinomial** and doing standard RL rollouts against it. So OSFP ≈ "fictitious self-play where the current policy is over-weighted as opponent".

### Practical implementation (Algorithm 1, Appendix B) — the checkpoint-pool loop
```
H = []                       # historical model pool
for each Learning Period LP (fixed number of samples):
    reset per-opponent stats G[i] (cumulative ±1 scores), C[i] (counts)
    while LP not finished, for each actor:
        if H empty or Unif(0,1) < p:        # p = 0.6
            opponent = CURRENT learner      # self-play (the "optimistic" over-weighting)
        else:
            sample i ~ f(..., (G[i], C[i]), ...)   # prioritized by stats vs each historical model
            opponent = H[i]; play; G[i] += g (±1); C[i] += 1
    if (G[i]/C[i] > ξ for ALL i) or (count > c):    # ξ = 0.7, c = 6
        add current learner snapshot to H; count = 0
    else:
        count += 1
```
- ξ = 0.7 on average score g∈{+1,−1} ⇒ the learner must beat **every** pool member with ≥ 85% winrate before a new checkpoint is added (or after at most c = 6 LPs as a fallback).
- f is a probability function over historical opponents based on their (G, C) stats (prioritized opponent sampling; exact form not given).

### RL sub-solver
Policy-gradient with **V-Trace** value estimation + **UPGO** auxiliary loss (AlphaStar-style), negative-entropy regularization per information state. Trained on rollout segments with value bootstrapping (not whole episodes).

### All reported hyperparameters
| RL (Table 6) | Value |
|---|---|
| V-Trace policy-gradient weight | 1.0 |
| UPGO policy-gradient weight | 1.0 |
| Value loss weight | 1.0 |
| Entropy penalty weight | 0.01 |
| Learning rate | 5e-5 |
| Batch size | 4e4 × num GPUs |
| Discount γ | 0.99 |
| LSTM states | 256 |
| Sample reuse | 2 |
| V-Trace c clip / ρ clip | 1.0 / 1.0 |

| OSFP (Table 7) | Value |
|---|---|
| Self-play probability p | 0.6 |
| Add-to-pool threshold ξ | 0.7 (≈85% winrate vs every pool member) |
| Max LPs before forced add, c | 6 |
| Samples per Learning Period | 8e8 |

Compute: TLeague-like actor-learner framework; "1 Unit" = 1 V100 + 600 CPU cores. LoCM 1.5 winner: 24 Units, ~72 h, ~500K obs/s throughput. LoCM 1.2 winner: 1 Unit × 3 days, then 8 Units × 6 days.

## 4. Reward design

- **Terminal-only: +1 win / −1 loss at the end of the game. No shaping whatsoever.** No intermediate reward for the draft stage; deck quality is learned purely through the propagated battle outcome (via the shared value function and BPTT).
- γ = 0.99 (the follow-up Hearthstone paper shows γ = 1.0 is better for this terminal-only setting).
- Exploration comes from the entropy regularizer (0.01) — which is also the "smooth" in Smooth Best Response, so exploration and the game-theoretic smoothing are the same knob.

## 5. Ablations and findings

1. **Card-embedding sharing (Sec. 4.2)** — 2500-match head-to-head, with-vs-without:
   - Sharing the **selected-cards (deck-so-far) feature** into the BT branch: **53%** winrate vs not sharing.
   - Sharing the **per-card embedding** between CB and BT: **55%** winrate vs not sharing.
   - Both kept in the final system. Small but consistent gains.
2. **E2E vs Alternating Training (Sec. 4.3, Table 3)** — the headline ablation, same resource budget:
   - LoCM 1.2 (winrate vs DrainPower baseline): Evo-AT 64.7%, Neural-AT 68%, **E2E 81%**.
   - LoCM 1.5 (winrate vs Evo-AT): Neural-AT 56.5%, **E2E 65.5%**.
   - Evo-AT = evolutionary deck builder with expert-prior cost-efficiency curves; Neural-AT = RL on CB stage, alternating with BT training (Algorithm 2: stage flag flips each time a checkpoint is added). E2E wins decisively.
3. **Evaluation temperature (Sec. 4.4)**: argmax (τ → 0+) beats sampling (τ = 1.0) at evaluation: **53%**. They always evaluate with argmax.
4. **One-Turn-Kill post-processing**: adding a rule-based OTK pass did NOT significantly help their model (it already computes lethal precisely); removing OTK from rival NeteaseOPD dropped that bot 2%.
5. **MCTS (Sec. 4.5, Tables 4–5)**: IS-MCTS on the ground-truth world state, used as behavior policy during training. With MCTS params (expand-prob p, n expansions, m successive states): under the same compute budget, frequent shallow search (0.1, 40, 1) is much worse (36%); rare deep search (0.00025, 400, 40) gives 56% at 40 h — but the advantage **decays with wall time: 56% → 52% (80 h) → 51% (144 h)**. Verdict: not worth it; dropped from the final system. (MCTS internals: n=400, PUCT c=5.0, prior temperature τ=10.0, Dirichlet α=0.03, prior mix p=0.75.)

## 6. Actionable ideas for Azuki TCG (most promising first)

Our setup: PPO+LSTM, one episode = draft (1 leader + 30 cards from ~80 candidates, ≤4 copies) + battle; 4-head action space; PufferLib.

1. **Keep/strengthen true end-to-end training across draft+battle — it is the single biggest win here (+13–16% over alternating schemes).** Make sure value bootstrapping and LSTM BPTT actually carry battle outcomes back into draft steps: check that `bptt_horizon` segments straddling the draft→battle boundary bootstrap correctly, and that draft steps are not zero-advantage padding. Ablation: E2E vs freezing draft policy while training battle (and vice versa).
2. **Share one card encoder between the draft head and the battle heads, and feed a mean-pooled "deck-so-far" embedding into the battle trunk + value head.** Both sharing ablations were positive (53%/55%). Concretely: embed the 80 candidates once; draft logits = score(card_emb, deck_pool_emb); battle obs gets the frozen deck-pool embedding as extra context. Ablations: (a) shared vs separate card encoders, (b) with vs without deck-context vector in battle.
3. **Represent the draft pick as per-candidate scoring with masks, not a fixed-index categorical.** CB obs = candidate set + already-selected multiset; "can-be-selected" mask implements our 4-copy limit exactly like their mask. This generalizes better than treating pick-index as an opaque discrete head.
4. **Adopt the OSFP opponent-pool recipe on top of PPO**: 60% of games vs the current learner, 40% vs a checkpoint pool sampled by per-opponent win statistics; add a checkpoint only when the learner beats *every* pool member at ≥85% winrate (ξ=0.7) or after a max-patience of 6 learning periods. Last-iterate convergence means we deploy the latest checkpoint, no policy averaging — a natural fit for PPO. Ablation: pure self-play vs SFP (uniform pool) vs OSFP (p=0.6 over-weighting).
5. **Keep reward terminal-only ±1; do not shape the draft.** If credit assignment to draft picks is too slow, prefer architectural fixes (deck context, shared value) over shaping. Add a UPGO-style auxiliary policy loss (weight 1.0) — cheap to implement next to PPO.
6. **Evaluate with argmax, train with sampling + entropy 0.01.** Their τ→0 eval gave +3%; entropy is the exploration/smoothing knob (and theoretically the SBR regularizer).
7. **Use attribute-based card features (cost/stats/keywords/element) in the encoder even though our pool is fixed** — LoCM 1.5's procedurally-generated cards prove attribute encoders learn transferable card evaluations; this also future-proofs against card-pool changes and supports ablations that swap ID-embedding vs attributes vs both.
8. **Skip MCTS for training data generation.** Under equal compute it lost or broke even, and its advantage shrank as training progressed; model-free PG with masks was their championship recipe.
9. **Hyperparameter anchors for our scale**: lr 5e-5, entropy 0.01, LSTM 256, sample reuse 2, V-Trace/PPO clip near 1.0, γ 0.99 (but see Hearthstone paper: γ=1.0 better for terminal-only reward).
