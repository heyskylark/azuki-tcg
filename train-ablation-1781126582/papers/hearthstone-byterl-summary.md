# Summary: Mastering Strategy Card Game (Hearthstone) with Improved Techniques

**Xiao, Zhang, Huang, Huang, Chen, Sun (ByteDance), arXiv:2303.05197v2, May 2023 ("ByteRL on Hearthstone")**

Direct follow-up to the LoCM OSFP paper (arXiv:2303.04096): same E2E-policy + OSFP framework applied to a much bigger commercial-grade game, plus a sequence of measured improvements. Final model beats the LoCM-champion baseline 73.6% (80.2% with deck-peeking "cheat"), and **defeats a top-10-of-China-region-league Hearthstone streamer in all four Best-of-5 Conquest tournaments** (3:0, 3:0 normal version; 3:1, 3:2 cheat version) — claimed first AI to beat top humans at full-game Hearthstone (deck building + battle).

---

## 1. Problem setup and draft/battle integration

Environment: modified open-source **Hearthbreaker** (≈ commercial Hearthstone of April 2015, Blackrock Mountain). 3 heroes supported (Mage, Warrior, Hunter); card pool 350+ developed (~270 usable: 240 common + 30+ hero-specific per hero).

**Three stages per game**, vs LoCM's two:
1. **PH (Pick Hero)**: choose 1 hero; each hero has a unique Hero Power; opponent's hero hidden until BT. *(Analogue of our leader pick.)*
2. **CB (deck building)**: draft a 30-card deck from the full pool visible at once (constructed-style, not arena); deck hidden from opponent.
3. **BT (battle)**: mana ramps each turn, multiple cards/actions per turn, pre-end turn allowed, win by reducing opponent hero HP to 0.

**State/action representation:**
- CB observation (Table V): `hero` (one of 3), `card set` (all cards incl. hero-specific), `card selected mask` (1 if already in deck), `card can be selected mask`. CB action: **categorical pick of one card per step** (30 sequential picks), exactly the serialized-draft formulation from the LoCM paper.
- BT observation (Table V): my hero, oppo hero, **my deck (cards remaining)**, decision type (construct / select / minion battlecry / spell card / minion-hero attack / hero power / end turn), my board + oppo board (minions with scalar features), my hand, my+oppo graveyard, my/oppo player scalars (hand count, minions, mana, weapon), BT action mask.
- BT action space (Table VI): **auto-regressive 2-step decomposition (type, target)**: step 1 pick *type* ∈ {my hand card, my board card, opponent's board card, my hero power card, end turn card} (hero power and end-turn are modeled as pseudo-cards); step 2 pick *target* ∈ {my hero, opponent's hero, my board card, opponent's board card}, conditioned on the chosen type. ≤ a few tens of actions per step; 0/1 action masks zero unavailable options inside each softmax.
- E2E policy with stage indicators: π_θ = δ_CB·π_θCB + δ_BT·π_θBT (PH deliberately excluded — see ablation B). Whole game = one episode; reward only at the end.

## 2. Network architecture (Fig. 2, appendix)

- **Shared card embedding and shared hero embedding** used by both θ_CB and θ_BT (pink/yellow blocks in their diagram) — carried over from LoCM where sharing was ablated positive.
- CB branch: card-set embeddings + selected/selectable masks → extracted CB feature → CB head scoring candidate cards.
- BT branch: per-group encoders over hand / my board / oppo board / deck / graveyards / player scalars (cards via the shared card embedding; heroes via the hero embedding) → concat → fc trunk → **LSTM (256 states)** → auto-regressive heads: *type* head, then *target* head conditioned on the selected type; masks applied at every softmax.
- **Value function estimated from the intermediate features of both π_θCB and π_θBT** (single critic spanning stages, as in LoCM).
- Policy gradient: combination of **V-Trace** (off-policy corrected value targets) and **UPGO**; later replaced the policy-gradient part by a PPO surrogate (improvement F below).

## 3. Training algorithm: OSFP + improvements

OSFP exactly as in the LoCM paper (Algorithm 1 reproduced): best response against a mixture of historical checkpoints with the **latest policy over-weighted** (last-iterate convergence ⇒ deploy the newest checkpoint, no averaging).

Checkpoint-pool mechanics:
- With probability **p = 0.6** an actor plays vs the **current learner** (self-play); otherwise vs a historical checkpoint sampled from pool H by probability function f over running stats (G[i] cumulative ±1 score, C[i] games).
- After each Learning Period (**3.2e8 samples**), the learner is added to H if its average score vs **every** pool member exceeds **ξ = 0.55** (⇒ ≥ 77.5% winrate; relaxed from LoCM's 0.7/85%) or after **c = 6** LPs of patience.

### Hyperparameters (Table III; blue = changed vs LoCM paper)
| Parameter | Value |
|---|---|
| Weight of policy gradient from **PPO** (replaces V-Trace PG) | 1.0 |
| Weight of policy gradient from UPGO | 1.0 |
| Value loss weight | 1.0 |
| Entropy penalty weight | 0.01 |
| Learning rate | **7e-5** |
| Batch size | **1e4 × 8 GPUs** |
| Discount γ | **1.0** |
| LSTM states | 256 |
| Sample reuse | 2 |
| V-Trace c clip | **[0.001, 1.007]** |
| V-Trace ρ clip | **[0.001, 1.007]** |

OSFP (Table IV): p = 0.6, ξ = **0.55**, max LP c = 6, samples per LP = **3.2e8**.

Compute: actor-learner (IMPALA-style); baselines trained on 8 V100s each; best non-cheat model (b4) used 24 GPUs + 5856 CPU cores (3 hero-isolated models × 8 GPUs); models compared after 2 days unless noted; human-beating versions trained 16–23 days.

## 4. Reward design

- **Terminal-only ±1 win/loss, no shaping** (unchanged from LoCM).
- Key change: **γ = 1.0** instead of 0.99. Rationale: episodes are short (~100 steps) and reward is purely terminal, so γ=1.0 recovers the true win/loss return with bounded variance, while γ=0.99 systematically distorts it. **+7% winrate by itself.**
- Exploration of the *deck space* is handled by Random-CB (below), not by reward shaping.

## 5. Improvements / ablations ladder (the core of the paper)

Evaluation protocol: A-vs-B = 2 (first/second player) × 3×3 (both sides play all 3 heroes) × 2500 matches; "A increases a% w.r.t. B" means winrate (50+a)%. Each baseline = previous + exactly one change (Table I winrate matrix):

| Step | Change | Gain |
|---|---|---|
| b0 | LoCM recipe ported (uniform hero sampling, γ=0.99, E2E+OSFP) | reference |
| **B. PH exclusion** | Including pick-hero in the E2E policy **fails**: hero distribution collapses to a single hero (or stays uniform with heavy entropy tuning); policy becomes good only at hunter-vs-hunter. **Excluded PH; sample hero uniformly.** | negative result, design choice |
| **C. γ = 1.0** (b1) | discount 0.99 → 1.0 for terminal-only reward | **+7%** vs b0 |
| **D. Random-CB** (b1.5) | Per player per game, sample n ∈ {0,1,2,4} with probs (0.5, 0.25, 0.125, 0.125); the first n draft picks are forced uniform-random (implemented by zeroing CB logits); disabled at evaluation. Decks become diverse and more "human-like" — the model independently rediscovered the 2015 community "Flamewaker Mage" deck. | ≈0% winrate, kept for deck diversity/exploration |
| **E. On-policyness via balancing** (b2) | Asynchronous system produced data ~2× faster than consumed (c = s_p/s_c > 2). Use a **queue (FIFO) buffer, not a ring buffer**, and **halve the actors** so c ≈ 1 (data consumed once, near on-policy). | **+10.0%** vs b1.5 |
| **F. V-Trace clipping + PPO surrogate** (b3) | Convergence only needs c̄ ≤ ρ̄, not ≤ 1: raise upper clip to 1.007; many ρ's collapsed to ~1e-4 killing the trace, so **clip ρ, c from below at 0.001** → clip(x, 0.001, 1.007). Replace the clipped-IS policy gradient with the **PPO clipped surrogate (ε = 0.2) while keeping V-Trace value targets** v_{t+1}. | **+15.1%** vs b2 (biggest single gain) |
| **G. Model isolation by hero** (b4) | Scaling batch/GPUs on one shared model: no gain. Instead **one model per hero** (3× compute, each hero has a distinct play style). | **+6.5%** vs b3 |
| **H. Cheat** (c5) | Peek hidden info: include the opponent's **first n chosen deck cards** in CB+BT observations (opponent hand still hidden). **Asymmetric training** to keep data consistent: sample n1, n2 ~ U{0..30}, set n2 = min(n1,n2); the learner-side policy observes n1 cards, the opponent policy observes n2 ≤ n1; eval samples n ~ U{0..30}. Learns counter-deck-building (e.g., 2× Antique Healbot specifically vs Freeze Mage). | **+5.5%** vs b4 |

Cumulative: b4 = 73.6% vs b0; c5 = 80.2% vs b0.

**Human evaluation findings:** b4-23day beat the top-10-region human 3:0, 3:0; c5-16day won 3:1, 3:2. Notably **c5-2day beats b4-2day 55% machine-vs-machine, but c5-16day played worse than b4-23day vs the human** (e.g., wasted Antique Healbot at full HP) — training time matters more than the cheat advantage, and machine-vs-machine winrate does not perfectly transfer to humans. Human's review: AI knows when to clear minions, computes exact lethal/OTK, builds human-like decks with zero domain knowledge, and cheat-mode produces deck diversity + targeted counter-strategies.

## 6. Actionable ideas for Azuki TCG (most promising first)

Our setup: PPO+LSTM, one episode = pick 1 leader + draft 30 cards from ~80 (≤4 copies) + battle; 4 action heads; terminal win/loss.

1. **Set γ = 1.0** (we have short episodes and terminal ±1 reward — identical setting; their +7%). Cheapest ablation we can run; check `train.gamma` in `python/config/azuki.ini`.
2. **Do NOT learn the leader pick inside the same E2E policy naively — expect collapse to one leader.** Their PH-in-E2E experiment collapsed even with entropy tuning. For us: (a) uniform-sample the leader during training and condition policy/value on a leader embedding; (b) optionally pick the leader at deployment via measured per-leader winrates; (c) if compute allows, try per-leader specialization (their hero isolation gave +6.5%) — e.g., per-leader fine-tunes or leader-conditioned experts rather than 3 full models.
3. **Fix IS-ratio pathologies: clip importance ratios from below as well as above** (they used [0.001, 1.007] for both ρ and c) and use the PPO surrogate with off-policy-corrected (V-Trace) value targets. With our 4 multiplicative action heads, joint ratios can vanish exactly the way their ρ ≈ 1e-4 did. Biggest single gain in the paper (+15.1%). Ablation: per-head ratio clipping vs joint, lower-clip on/off.
4. **Random-CB-style draft exploration schedule**: with per-player probability (0.5, 0.25, 0.125, 0.125) force n ∈ {0,1,2,4} uniformly random draft picks at episode start; off at eval. Diversifies decks the battle policy sees (better battle generalization, less deck collapse) at ~zero winrate cost. Easy to implement by overriding draft logits/masks for the first n picks. Ablation knobs: n distribution, anneal over training.
5. **Privileged/cheat training on opponent deck info**: we already have a privileged critic — extend it to see the opponent's drafted deck (or first n cards). If we want a *cheating policy* (not just critic) as a league exploiter, copy their **asymmetric n1 ≥ n2 sampling** trick to keep the data distribution consistent. Their +5.5%; also produced counter-drafting behavior — useful as a robustness sparring partner even if the deployed agent stays non-cheating.
6. **Stay near on-policy: monitor the produce/consume ratio.** In PufferLib sync PPO this maps to sample reuse and stale-opponent checkpoints; they got +10% just by ensuring each sample is consumed ~once (queue semantics, c→1). Keep sample reuse ≤ 2.
7. **OSFP pool with relaxed gate**: ξ = 0.55 (≥77.5% winrate vs every pool member) + 6-LP patience worked at Hearthstone scale; the stricter LoCM gate (0.7) may stall pool growth in harder games. Ablate gate threshold and pool-sampling function f (uniform vs winrate-prioritized).
8. **Auto-regressive action decomposition with masks at every softmax** — validates our multi-head design; condition later heads (target/aux) on earlier head choices (type/card) rather than predicting heads independently; treat "end turn" and leader/gate abilities as pseudo-cards for a uniform card-scoring interface.
9. **Evaluation discipline**: full matchup matrices (both sides × both seats × 2500 games), name checkpoints by recipe+train-days, and remember their lesson that machine-vs-machine deltas (c5 > b4) can invert against humans/out-of-distribution opponents — keep a held-out opponent suite, and prefer longer training before judging a recipe.
10. **Deck-quality probe**: they validated draft learning qualitatively by checking decks against known-good human archetypes (Flamewaker Mage). For Azuki: track drafted-deck statistics (copy counts, curve, element mix, overlap with hand-built decks) as a cheap diagnostic of whether battle reward is actually shaping the draft.
