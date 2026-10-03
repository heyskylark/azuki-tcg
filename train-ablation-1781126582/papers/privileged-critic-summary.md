# Summary: Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access

**Paper:** Ebi, Ernst, Böhm, Lambrechts — arXiv:2509.26000v3, ICML 2026.
**Source file:** `privileged-critic-2509.26000.pdf` (28 pp., read in full).
**Code:** https://github.com/EbiDa/informed-asymmetric-a2c

## TL;DR

Generalizes asymmetric actor-critic: the critic may condition on **any** state-derived privileged signal `i_t` (not just the full state) and the policy gradient stays **unbiased**, provided the critic also keeps the observable history `h_t`. The practical question becomes *which* signal to give the critic; they provide two statistical tests for that and show that carefully chosen partial signals **match or beat full-state critics** — full state is often not the best choice because reward-irrelevant features act as structured noise.

## Core method and exact implementation details

### Theory / conditioning rules
- **Informed POMDP**: information variable `i_t ~ I(·|s_t)` with the assumption that the observation is conditionally independent of the state given `i_t` (non-restrictive; any auxiliary observation `o⁺` works via `ĩ_t = (o_t, o⁺_t)`).
- **Informed history critic** `V(h_t, i_t)`: critic input = encoded full action-observation history **plus** the privileged signal. Theorem 3.1: for *any* state-conditioned `i_t`, the informed asymmetric policy gradient equals the standard policy gradient (unbiased; `E_{i|h}[V(h,i)] = V(h)`). Special case `i_t = s_t` recovers Baisero & Amato's history-state critic.
- **Critical caveat:** a critic conditioned on privileged state **alone** (`V(s)`, no history) is generally ill-defined/biased in POMDPs (needs state-decodability). The history must stay in the critic input.
- `i_t` may even be an **expert/oracle action** `a* ~ π*(·|s_t)`: the critic exploits oracle info for value estimation without the bias of direct imitation (which is suboptimal in POMDPs, Warrington et al. 2021).
- Why it helps: `E_{i|h}[H(s|h,i)] ≤ H(s|h)` and, by law of total variance, `E_{i|h}[Var(G|h,i)] ≤ Var(G|h)` → lower-variance value targets, mainly in environments with **value aliasing** (same history, different returns — exactly the situation with hidden opponent hands).
- Algorithm: A2C-style; TD-trained informed critic; advantage = TD error `Â = r + γV(h',i') − V(h,i)`.

### Signal-selection tests (the paper's main practical contribution)
1. **α-residual informativeness (pre-training, works on random-policy data):** encode history `z = f_RNN(h)` (RNN width 64, trained on 100 episodes); cross-fitted regressions (random forest, 100 trees, K=5 folds) of `G` and `i` on `(z, a)`; compute residuals; measure dependence with **HSIC** (Gaussian RBF, median-heuristic bandwidth, Nyström approx. with 512 landmarks); episode-level permutation test (B=1000) → p-value; signal is informative if `p < α`. Data: 250 episodes × 25 steps under a random policy.
2. **(ε,δ)-prediction informativeness (post hoc):** train symmetric `Q̂(h,a)` and informed `Q̂(h,i,a)` by TD on the same data (2,500 episodes × 25 steps, 5-fold CV); per-episode squared-error gain `L_τ`; one-sided t-test (or bootstrap for small N) of `E[L] > ε` (they use ε = 0).
- Both tests rank feature **subsets**; their effect sizes correlate with downstream policy-return gains even when computed from random-policy episodes (Fig. 3).

### Architecture & feature-scale details (Appendix E)
- **Navigation tasks:** 128-dim single-layer GRU history encoder; actor and critic have identical architectures but **separate parameters**; 64-dim embeddings for state/action/observation; 2-layer MLP heads 512→256 ReLU (Memory-Four-Rooms: 8-dim categorical embeddings → 144-dim obs vector; 3-layer CNN over state grid; 512-unit head).
- **Privileged signal handling:** embedded *analogously to observations*, then **concatenated with the latent history representation** before the critic's feedforward head. (Never fed to the actor.)
- **POPGym tasks:** observations projected to a **128-dim zero-mean unit-variance representation (linear → LayerNorm → LeakyReLU)** before a 256-dim GRU; actor/critic heads = 2×128 LeakyReLU; lr 5e-4; **BPTT truncation 1024**; entropy weight 0; γ = 0.99; privileged-signal embedding size **64** (card games) / **128** (cart pole).
- **Navigation hyperparameters:** γ = 0.99; episodes capped at 100 steps; 2 episodes per gradient update; **frozen target network for the critic, updated every 10,000 steps**; lr 1e-3 (some 3e-4); initial entropy weight λ0 per env in **{0.03 … 3.0}**, decaying **linearly over 2M steps to λ0/10**.
- Synthetic POMDPs: |S| = 20, |A| = 4, state features `s ∈ R^5` Gaussian; reward linear with weights `w_r = [0.0001, 0.0001, −0.0001, −1.0, 1.0]` (s⁴, s⁵ dominate reward); signals = masked feature subsets `i_t = W_i(x_i ⊙ s_t)`; observations = noisy masked subsets of `i_t`.

## Main results and ablations

- **12 benchmarks** (6 navigation + 6 POPGym), 20 seeds, 2M steps. `informed-asym-A2C` consistently improves sample efficiency/stability over symmetric A2C and **matches or beats** full-state asymmetric critics (`asym-A2C-s`, `asym-A2C-hs`) while using strictly less state information.
- **Car-Flag:** privileged signal = the agent's *own velocity* (a tiny partial signal) → beats *all* baselines in both convergence speed and asymptotic return.
- **Position Cart Pole (AUC over 2M steps):** angle-velocity alone **8.68e5** > both-velocities 8.48e5 > full state 8.26e5 ≫ x-velocity 2.57e5 > none 1.53e5. The *right* partial signal beats full state; the *wrong* partial signal is nearly useless.
- **Synthetic POMDPs (Table 1):** signals containing the reward-dominant features s⁴,s⁵ are flagged informative by both tests and give the best AUC — `[s¹,s²,s⁴,s⁵]`: **1.23e5 vs 1.06e5 symmetric baseline (~+16%)**; full state 1.19e5 (worse than the best subset); a signal identical to the observation `[s¹,s²]` gives no gain (1.07e5).
- **Repeat First (Table 6, final return / AUC):** none **0.83** / 6.6e5; dealt-stats 0.88 / 1.7e5; full state 0.90 / **2.0e4**; hand 0.55 / 5.4e5; **first-card 0.35 / 2.9e5**. I.e., privileged info traded early speed for final performance (full state), and a single highly aliased feature (first-card) was catastrophic. Aggregated statistics ("dealt" counts) were the benign form.
- **Count Recall:** most privileged signals mildly hurt (freq-cards-queried 0.72, full state 0.75 vs none 0.85); only the **expert signal** helped (0.88, best AUC 1.82e6 vs 1.76e6).
- **Concentration:** the history+full-state critic (`asym-A2C-hs`) **degrades over training** (return falls to ≈ −0.5 to −0.75 while others stay ≈ −0.25) — "likely due to the high-dimensional state representation with many potentially irrelevant features."
- **Cleaner:** symmetric A2C suffers a performance collapse after ~2.5M steps; the informed variant converges similarly but **more stably**.
- **Compute:** informed critic costs only **+5–14% wall-clock** vs symmetric A2C; full history+state critics cost up to **+58–66%** on memory tasks.
- Informativeness rankings are robust to observation noise (β_o ∈ {0.1, 0.5}) and to changing which features are observable.

## Failure modes / when privileged information hurts

1. **Reward-irrelevant or weakly reward-related features** in the critic input act as structured noise ("noisy-TV" effect), especially rapidly varying components → worse value estimation, slower learning, sometimes outright divergence (Concentration). Full state is frequently suboptimal for exactly this reason.
2. **State-only critic without history** is theoretically unsound in POMDPs and empirically weaker/unstable on several tasks.
3. **Highly aliased single features that the actor itself must memorize** (Repeat First: first-card, hand) slowed learning drastically and hurt final return even though they are perfectly reward-relevant — the informed baseline then varies with information the actor cannot act on, making early advantages noisy from the actor's perspective. Aggregates (counts/frequencies) and richer contexts were much safer.
4. **Speed/asymptote trade-off:** privileged critics sometimes reduce AUC (early learning) while improving final performance — evaluating only one of the two is misleading.
5. Performance gains depend on *what* the signal encodes, **not how much**: "learning performance depends primarily on whether the input contains reward-relevant information, rather than on the amount of information provided."

## Actionable ideas for Azuki TCG (most promising first)

1. **Keep the LSTM history in the privileged critic and concatenate privileged features after the LSTM.** Embed opponent-hand/deck features with the same encoders + normalization as regular observations, concat with the LSTM hidden state, then the value MLP. Never let privileged features *replace* the history (bias) — verify our current implementation does exactly this.
2. **Ablate privileged-signal subsets, not all-or-nothing.** Candidate arms: (a) opponent hand only; (b) + opponent deck *composition as counts/histogram*; (c) + own remaining-deck counts; (d) full (current). Per Repeat-First/Concentration: prefer **aggregated counts over ordered card lists** — exact deck order is a noisy-TV risk with near-zero reward relevance.
3. **Run the cheap post-hoc prediction-informativeness test before long ablations:** on replay from a frozen checkpoint, TD-train a symmetric and a privileged critic on identical data, compare per-episode value MSE gain with a one-sided t-test. Only signals with significant gain earn a full training run. The HSIC residual test can rank signals even from random-policy episodes (useful pre-training, e.g., for deck-building-phase features).
4. **Measure both AUC and final performance** in ablations; privileged critics may slow early learning while improving the asymptote (and vice versa). Our eval harness should log area-under-winrate-curve, not just final winrate.
5. **Try an expert/oracle-derived signal for the critic:** any strong scripted evaluator or oracle search result (e.g., heuristic board-strength given full information) can be fed to the critic as `i_t` — unbiased by Theorem 3.1, and it was the *only* signal that helped in Count Recall.
6. **Phase-gated privileged features:** returns in deck-building are aliased mostly on the opponent's concurrent picks; in battle, on the opponent's hand. Ablate masking privileged features per phase (build vs battle vs response windows) to cut irrelevant-feature noise.
7. **Stabilizers worth copying:** frozen critic target network (update every ~10k steps), per-env-tuned entropy weight with **linear decay to 10% of initial over training**, observation pre-projection to zero-mean/unit-variance (linear+LayerNorm) before the LSTM.
8. **Budget check:** expect only ~5–15% wall-clock overhead from a privileged critic head; if ours costs much more, the encoder is too heavy.
