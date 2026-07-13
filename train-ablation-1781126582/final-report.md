# Azuki TCG — Deck-Building Training Research: Final Report

> STATUS: Round-2 + 45M campaign complete (2026-07-09). Part II (§12-14) has the
> 45M matrix, causal gate-value probes, lever-search conclusion, and the
> production recipe. Running work log: research-notes-01.md.
> All round-2 artifacts under results/round2_*; probes reusable on any checkpoint.

## 1. Executive summary

**Question**: does the model learn per-card strategies — building decks with synergy and
playing differently depending on what it drafted — and what unlocks that learning?

**Answer**: yes at the card / card-type / gate-family level, once the reward-shaping bias is
removed; and gate-aware PORTAL PLAY emerges (Part II). Sibling-gate DRAFT conditioning does
not emerge under any tested lever at up to 45M steps — the causal probes (Part II §13) show
the value is real (4-13pp play gaps, +8pp LIGHTNING composition interaction) but
second-order for the pick head; see §14 for the production recipe and design memo.

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

---

# PART II — 45M campaign & causal gate-value probes (2026-07-07 → 07-09)

## 12. Arms and headline outcomes

| arm | steps | recipe | draft-vs-ref | sibling KL | critic ratio (late) |
|---|---|---|---|---|---|
| anneal45 | 45M | anneal only, league 6/4/3 | **46.4%** (192) | 0 at all ckpts | (sweep, see run45_anneal45) |
| combo45b | 45M | anneal+gateid+eps.02+oversample.35 | 41.1% (192) | 0 at all 30 ckpts | 0.71, sign 95% |
| portalgp45 | 45M | combo + AZK_PORTAL_GP_BONUS 0.3 | **46.9%** (192) | 0 at all ckpts | 0.21, sign 96% |
| portalgp1 | 15M | same as portalgp45 | **46.9%** (96) | 0 | 0.17 |
| privgp1 | 15M | portalgp + privileged critic (drafted decks) | 39.6% (96) | 0 | 0.08 |
| combo45-resumed | — | INVALIDATED (model-only resume → entropy collapse 0.077, 26.6%) | — | — | — |

Portal-GP bonus: +2.5× portal usage at matched steps (0.076 vs 0.030 action
share @3-5M), best external quality at BOTH horizons, no win-rate damage —
but trades away critic-side sibling sharpness (0.21 vs combo45b's 0.71).

## 13. Causal value of gate identity (probe matrix, combo45b checkpoint)

**Sibling mirror decks** (identical deck+leader, only the gate differs; n=1032/arm):

| pair | policy | portal-forced | portal-blocked |
|---|---|---|---|
| Surge > Stormchain | 54.1% | 55.2% | 48.1% |
| Hydromancy > EchoedWaves | 55.5% | 53.7% | 50.6% |
| Rushfire > Ragefire | 55.3% | **62.7%** | 48.2% |
| Stonehaven > Devotion | 54.2% | 53.4% | 50.7% |

Every sibling gap is real and lives ENTIRELY in the portal abilities (blocked
≈ 50% everywhere). The trained policy already extracts the gaps in play
(1.4-2.3 portals/ep argmax) — and optimal portal STYLE is gate-specific:
forcing portals gains +7pp under Rushfire but loses under Hydromancy/water.

**Ability ladder** (all-NORMAL mirror decks vs Hydromancy, same leader both
sides): Rushfire 57.7 ≫ ref ~50 > Devotion 46.7 ≈ Stormchain 46.7 ≈ Surge
45.9 > Stonehaven 44.1 > EchoedWaves 43.0 > Ragefire 41.2 (forced 37.7 —
net-negative portal). **16pp raw ability-power spread on identical decks.**

**Composition × gate interaction** (mirror-gate, archetype vs entity-only,
n=500/cell): LIGHTNING **+8.0pp ± 5.6 (p≈.004)** — weapon-heavy is worth 8pp
more under Stormchain (re-equip) than Surge (discard-replay); WATER −3.0 ± 6.2
n.s.; EARTH +2.4 ± 6.3 n.s.; FIRE +0.0 ± 6.3 (exact null — cheap-aggro 50.0%
under both FIRE gates). Gate-conditional drafting has real value for at least
the LIGHTNING pair — the drafter's sibling-blindness is unexploited margin
there — while FIRE siblings differ in HOW MUCH to portal, not what to draft.

## 14. Conclusions

1. **The model has per-gate strategy where the game rewards it first-order**:
   element/family drafting, weapon-gate specialization, synergy pairs,
   deck-conditional playstyles, and gate-aware PORTAL PLAY (it beats
   portal-blocked baselines and modulates portal usage by gate).
2. **Sibling-gate DRAFT conditioning is not reachable with PPO pick-gradients
   at this scale**: KL ≡ 0 across 45M × {control, oversampling, portal
   exposure} and 15M privileged critic. The critic prices siblings from 1.5M
   steps (sign ~95-100%) — the signal exists; the pick-head gradient can't
   clear its noise floor. Levers exhausted: representation (id channel),
   experience (43% sibling matchups), exposure (2.5× portals), baselines
   (privileged critic), horizon (3×).
3. **Production recipe**: anneal (warmup 12/ramp 40) + gate-id embedding +
   pick-eps 0.02 + oversample 0.35 + portal-GP bonus 0.3, league 6/4/3
   windowed, native path — 46.9% draft-vs-ref at 15M and 45M, no long-run
   decline, richest mechanic usage. Track KL/critic per checkpoint at scale:
   June showed second-order signals emerge transiently near scale boundaries;
   the distributed run is itself the next (and only remaining) training test
   of sibling conditioning.
4. **Game-design memo**: gates are far from balanced (16pp ladder spread;
   Ragefire's portal net-negative; Rushfire dominant deck-independent).
   Sibling identity is portal-mediated only, second-order for drafting except
   LIGHTNING. To make gate identity a first-order draft consideration:
   scale abilities with deck composition (e.g. Surge with weapon count),
   buff Ragefire, temper Rushfire.
5. **Future training work** (post-production candidates): draft-specific
   auxiliary objectives (pick-step advantage from battle-start V deltas),
   pick-head-targeted credit, distributed-scale seeds × longer horizons.
6. **Infra shipped en route**: engine invalid-action now truncates (abort
   deadlocked a 45M run); resume fingerprint excusal flags + reset-probe skip
   (model-only resume collapses entropy — never train on one); sibling
   oversampling knob; portal-GP shaping; privileged drafted-deck exposure;
   probe suite (gap / ladder / interaction / critic-sensitivity / KL
   trajectory). OPEN: engine stale-mask desync root cause (pre-production
   blocker; repro seeds logged).

---

# PART III — Draft-time auxiliary objectives (A-DRAFTAUX, 2026-07-09 → 07-10)

## 15. Motivation and mechanism
Part II closed the conventional lever search with a paradox: the critic prices
sibling gates (sign-consistent from 1.5M steps) but the actor never drafts on
them (KL ≡ 0 everywhere). A-DRAFTAUX wires the proven critic signal directly
into pick credit at the draft→battle boundary (commit 39c1fc2, both trainer
paths, off by default):
- **aux1 / vboot** (`AZK_DRAFT_VBOOT_COEF`): the battle-start value V(s₀;g)
  added as reward at the last pick step — shortens the pick credit path from
  ~150 steps to ~1.
- **aux2 / sibdiff** (`AZK_DRAFT_SIBDIFF_COEF`): clipped counterfactual
  differential max(0, V(s₀;g) − V(s₀;g→sibling)) — the partial derivative of
  deck value w.r.t. gate identity; a generic good deck scores zero, only
  gate-FIT survives. Clip prevents "make it worse under the sibling" gaming;
  computed by a gate-swapped forward with the exact pre-forward LSTM state.

## 16. Results

| arm | steps | draftref | gate-swap KL | critic ratio |
|---|---|---|---|---|
| portalgp1 (control) | 15M | 46.9% | 0 exact | 0.17 |
| auxv1 (vboot .05) | 15M | 46.9% | 0 exact | 0.32 |
| auxd1 (sibdiff 2.0) | 15M | 44.8% | 0 exact | — |
| auxvd1 (both) | 15M | **50.0%** (campaign best) | **1e-5 (first nonzero ever)** | 0.86 |
| **auxvd45** (both) | 45M | **33.3%** (campaign worst) | 0 at final ckpt | 0.95 early / 0.59 late |

**The KL trajectory is the finding.** auxvd45 per-checkpoint sweep (30 ckpts):
mean KL peaks at **5.8e-5 at 1.5M steps** — nonzero in all four elements,
~6× anything measured in any prior arm, against a control floor of exactly 0 —
then decays to the measurement floor by **~8M** (isolated 1-2e-6
single-element blips through 18M; structurally zero from 19M to the end). Actor-side gate conditioning is **creatable but not
retainable** under the current optimization.

The critic-side trajectory completes the picture: sensitivity peaks with
the actor's KL (ratio 1.65 at 3.1M, |dV| ~0.006 — the strongest critic
readings of the campaign) and then *also* decays (early ≤9M mean 0.95 →
late ≥37M mean 0.59). The aux and the conditioning it created faded
together: the actor's KL died first (~8M), the critic's edge eroded after.
Under constant aux pressure the VALUE side stayed nonzero — so the erasure
is not the critic forgetting first; the pick head loses the distinction
while the critic still prices it, then the critic's sharpness drifts down
as the (Goodharted) meta stops exercising the difference.

Two failure mechanisms, both now characterized:
1. **Erasure**: the conditioning decays inside the shaping-anneal window
   (1.9M→8.2M) as the outcome-dominated meta equilibrates; PPO + entropy +
   pick smoothing pull the pick head back to the element-conditioned optimum.
2. **Goodhart at length**: the aux coefficients were constant (the design's
   "ride the shaping anneal" was not implemented), so after shaping faded the
   aux became the loudest dense signal; 45M of optimizing the critic's
   *opinion* produced critic-pleasing, non-winning decks (33.3% vs 46.9% for
   the identical recipe without aux). At 15M the same recipe was net-positive
   (50.0%) because dense shaping still dominated.
3. Root blocker (from the §7.2 post-mortem, notes): the critic prices the
   gate MAIN effect but carries almost no gate×composition INTERACTION
   (per-deck differential spread |mean|/std 2–10) — while the game's true
   interaction is up to 8pp (LIGHTNING probe). No critic-derived signal can
   teach gate-FIT drafting until the value function represents fit itself.

## 16.5 Post-campaign S-queue addendum (2026-07-11)
S1 (annealed aux): 15M-total run hit **63.9% pooled draftref** (best ever) but
the same recipe at 45M-total landed at 37.0% — and per-checkpoint draftref
trajectories revealed why: **draft-vs-ref oscillates over a ~20pp range with
the league meta cycle in every 45M arm** (portalgp45: 56.2→40.6→40.6→60.4→
49.0→46.9; s1auxann45: 44.8→47.9→41.7→54.2→47.9→37.0). Methodological
correction: single-checkpoint arm deltas under ~10pp are within cycle noise;
window means favor portalgp (49.0 vs 45.6). Production protocol additions:
(a) periodic external evals + CHECKPOINT SELECTION over a late window —
every recipe has deployable 54-60% peaks; (b) lr-polish at the target
horizon (S1's 63.9% = cycle peak locked in by end-of-schedule lr decay —
the annealed-aux recipe is the SHORT-HORIZON (<=15M) choice); (c) portalgp
remains the long-horizon base.

**S2 (outcome-graded portal bonus): NEGATIVE.** Portal usage collapsed to
0.0038 (flat bonus: 0.076; none: 0.030) — grading pays ~27% of attempts under
unskilled play, so whiffs became pure tempo cost and the policy learned to
portal LESS. "Pay for trying" is the load-bearing exploration property; the
flat GP bonus stays in the recipe. (draftref 40.6% in-band; KL ~0.)

**S3 (cross-gate replay 0.15 + boundary pick-masking): PASS — the first
lever to move the critic-interaction blocker.** Critic differential spread
|mean|/std fell to **3.98 vs 6.4-6.5** in both baselines (FIRE 1.72, WATER
2.81): with same-deck-both-gates outcome labels in its data, the value
function began pricing gate-fit as deck-DEPENDENT rather than a per-gate
constant. draftref 54.2% (best non-aux point, cycle caveat), critic ratio
0.48, KL ~0 alone (expected without a seeding term). Composition s13combo
(S1 annealed aux seed + S3 contrast data) is the direct test of the
seed-plus-sustenance hypothesis; verdict pending.

## 17. Final conclusions (whole campaign)

1. **Production recipe (unchanged, final)**: anneal (12/40 per-env episodes)
   + gate-id embedding + pick-eps 0.02 + sibling oversampling 0.35 +
   portal-GP bonus 0.3, league 6/4/3 windowed, native path — 46.9%
   draft-vs-ref at 15M and 45M, best external quality, no length decline,
   2.5× richer portal usage. **No aux terms.**
2. **Strategy emergence, demonstrated**: element/family-conditional drafting,
   synergy pairs, deck-conditional playstyles, gate-aware portal play
   (extracts 4-6pp sibling gaps in play; modulates portal style per gate).
3. **Sibling-gate DRAFT conditioning**: real value exists (up to 8pp
   composition interaction, LIGHTNING), the critic sees the main effect, the
   actor can be made to condition briefly — but nothing at single-box scale
   RETAINS it. The bottleneck is structural: the critic lacks the
   gate×composition interaction term.
4. **Single-box follow-ups** (documented, not run; production run takes
   priority):
   - **Composition-contrast exploration** (new, highest-leverage): the
     missing interaction term is most plausibly a DATA problem — the trained
     drafter builds ~one deck style per element, so the critic never sees
     contrasting compositions under the same gate and cannot learn gate-FIT.
     Occasionally forcing archetype-style drafts during training (the
     probes' forced-deck machinery already exists) would put the
     gate×composition contrast into the critic's training distribution,
     attacking the root directly. No capacity increase can substitute for
     contrast the data lacks.
   - **Annealed-aux retry**: auxvd with coefficients on the shaping
     schedule — cleanly answers the retention question and should remove the
     Goodhart collapse, but the prior for a qualitative payoff is low while
     the critic still lacks the interaction term. Highest expected value is
     BOTH combined: seed the differentiation and feed the critic the
     contrast data needed to sustain it.
   - **Contrast-data mechanisms** (design menu, 2026-07-10 discussion):
     (a) cross-gate replay — ~3-5% of episodes silently swap the drafted
     gate to its sibling at battle start (sibling map exists), producing
     same-deck-both-gates outcome labels; pick steps of those episodes must
     be masked from actor training (league trainability masking reusable);
     (b) scripted-contrast episodes (probe archetype builders as fixed decks;
     needs native fixed-deck injection); (c) pick-mutation exploration.
     Success metric: the critic's per-deck differential SPREAD rises.
   - **Grounded draft-phase rewards** (no critic proxy → no Goodhart):
     (1) hindsight pick credit — annealed, win-gated bonus routed back to
     the pick steps of cards actually USED in battle (portaled with GP,
     weapon attached); differentiates picks WITHIN a draft; needs per-card
     play-event export from the env (the biggest build, strongest single
     idea — its own arm once plumbing exists);
     (2) draft-novelty bonus vs per-gate running composition (fights the
     mono-deck attractor AND generates contrast data);
     (3) per-gate pick baselines (variance reduction on the right axis);
     (4) league seats for forced off-meta archetype drafters
     (AlphaStar-exploiter analog).
     Minimal well-controlled next 45M if appetite exists: annealed aux +
     cross-gate replay (one seeding lever + one data lever).
   - **Converged smoke queue** (2026-07-10 design session; 15M smokes first,
     winners compose into the next 45M; readouts add the critic-differential
     SPREAD across decks as the direct interaction-learning metric):
     S1 annealed aux (retention test; tiny code) →
     S2 cross-gate replay (critic contrast; small: post-draft sibling swap +
        pick-step masking via league trainability machinery) →
     S3 portal-outcome-graded bonus (moderate: pre/post effect diff within
        the portal step; whiffed portals pay 0; upgrade of the GP bonus —
        rare case where the Goodhart exploit IS the desired behavior) →
     S4 human-deck league seats (moderate: web-DB deck export + native
        fixed-deck episodes; best product alignment — trains against the
        distribution the model faces online; the same C mechanism unlocks
        S6 scripted contrast for free) →
     S5 hindsight pick credit (largest: per-card play-event export + trainer
        credit routing; strongest single idea, own arm once plumbing exists).
     Diversity bonus held in reserve (S2/S4 generate contrast as a side
     effect). S1 alone likely proves retention-of-a-seed, not visible
     strategy — pair with S2 (later S4) for qualitative change.
   Beyond single-box: the distributed run remains the scale test (capacity +
   data volume for the interaction term), with per-checkpoint KL/critic
   tracking to catch emergence live.
5. **For the product**: gate balance numbers (16pp ladder spread, Ragefire's
   net-negative portal, Rushfire dominance) and the meta expectations they
   imply are competitive intel independent of any training outcome.

## 18. Part II/III artifact index
- results/run45_{combo45b,anneal45,portalgp45,auxvd45}/ — per-ckpt KL+critic
  sweeps; results/gate_gap/ — sibling mirrors + cross ladder;
  results/gate_ix/ — composition×gate interaction; results/run15_* — 15M arm
  final-ckpt probes; *_draftref.json — external evals.
- Probes: probe_gate_gap.py, probe_deck_gate_interaction.py,
  probe_critic_gate.py, probe_embedding_geometry.py, probe_gate_kl.py,
  fuzz_mask_consistency.py. Drivers: run_gate_gap_all.sh,
  run_interaction_probe.sh, run_aux_matrix.sh, run_auxvd45.sh.
- Trainer: A-DRAFTAUX knobs (AZK_DRAFT_VBOOT_COEF / SIBDIFF_COEF /
  SIBDIFF_CAP), default off. Env: draft_same_element_matchup_prob,
  deck_building_privileged_decks, AZK_PORTAL_GP_BONUS, invalid-action
  truncation (+AZK_INVALID_ACTION_ABORT), resume excusal flags.

---

# PART IV — Next-experiment roadmap (converged with user, 2026-07-10 evening)

> Written for the main thread: full specs so each task can be executed
> without re-deriving context. Ordering is simplest → most complex. Protocol:
> every lever gets a 15M smoke (~2-2.5h) with the standard readout suite
> before any 45M. Standard readouts per arm: draftref 96 argmax (192 at 45M),
> gate-swap KL (probe_gate_kl.py), critic ratio AND **critic-differential
> spread across decks** (std of battle_dv_signed_values in
> probe_critic_gate.py output — the direct interaction-learning metric),
> plus portal usage where relevant. Compare against portalgp1 (46.9%) at 15M
> and portalgp45 (46.9%) at 45M.

## S1 — Annealed aux (retention test) [tiny: trainer schedule]
Rerun the auxvd recipe with BOTH aux coefficients annealed linearly
1.0→0.0 over global_step ∈ [0, AZK_DRAFT_AUX_ANNEAL_END_STEP] (default
8_200_000, matching the shaping fade). Implementation: scale
self._draftaux_vboot/_sibdiff contributions by the schedule inside
_draftaux_step / _draftaux_league_stash using self.global_step; add the env
var; keep base coefs 0.05 / 2.0.
**15M IS meaningful as a gate** (user question answered): in auxvd45 the
conditioning peaked at 1.5M and decayed to ~0 by 11M with the aux STILL ON —
so a 15M annealed run (aux off after ~8M) directly answers "does the seed
survive without support?" If KL at the 15M final checkpoint > 0 → promote to
45M (S7). If already dead at 15M → the 45M is pointless; skip.
Also expect: no Goodhart (draftref should be ≥ ~46% at 15M).

## S2 — Portal-outcome-graded bonus [small-moderate: env-only C]
Upgrade AZK_PORTAL_GP_BONUS from "portaled with GP" to "portal that actually
resolved an effect": in c_step, diff pre/post state within the step around
the tick loop for the acting player — weapon-count delta on the portaled
entity (Surge/Stormchain), untapped-IKZ delta (Hydromancy), opponent leader
HP delta (Devotion/damage), garden occupancy delta (Rushfire). Whiffed
portal ⇒ bonus 0; resolved ⇒ scale bonus by min(GP,4)/4 as now. New env var
AZK_PORTAL_OUTCOME_BONUS (replaces the flat GP bonus in this arm; keep both
knobs independent). Goodhart watch: "engineering resolvable portals" is the
DESIRED behavior (e.g. weapons in discard for Surge) — but track draftref
at 15M and 45M for late drift like auxvd45's.

## S3 — Cross-gate replay (critic interaction data) [moderate: C + trainer masking]
With prob AZK_CROSS_GATE_REPLAY_PROB (~0.05), after the draft completes swap
ONE player's gate to its sibling before battle init (sibling map exists in
g_draft_catalog). Purpose: outcome labels for "same deck, other gate" — the
exact data the critic lacks (§7.2). REQUIRED: mask that episode's PICK steps
out of ACTOR training (picks were made under the pre-swap gate; unmasked,
5% sibling-averaged pick labels actively FIGHT conditioning). Mechanism:
post-rollout in the trainer, detect rows whose deck_context.gate_card_def_id
differs between draft steps and the battle boundary (both readable from the
stored obs buffer via the byte offsets in _draftaux_init_layout) and zero
those pick steps' contribution to the policy loss (values/critic stay).
Success metric: critic-differential SPREAD across decks rises (|mean|/std
drops below ~2) — that's the interaction being learned.

## S4 — Reference-deck league seats [moderate-large: native fixed-deck episodes + seat wiring]
Real-human-deck opponents in the league (user will export a large sample
from the production DB later; FOR NOW use the existing 18-deck reference
pool). **EVAL-LEAK CAVEAT (user-flagged, mandatory)**: those 18 decks ARE
the draftref benchmark. Split them: 9 for training seats / 9 HELD OUT for
draftref, and report draftref against the held-out 9 only, clearly labeled
(numbers not comparable to earlier 18-deck draftrefs — rerun portalgp1's
draftref on the same held-out 9 as the control). Implementation: native
fixed-deck episodes — with prob (or for designated league seats), a player
skips the draft and plays an assigned deck from a table passed like the
draft catalog (mirror the legacy wrapper's _fixed_state_from_deck semantics
in draft_begin_episode/c_step_draft). This same mechanism unlocks S6 free.

## S5 — Hindsight pick credit [largest: engine/env event export + trainer routing]
At episode end, small annealed bonus to the PICK steps of cards that were
actually used in battle, gated on winning (or zero-sum symmetric): card
played / weapon attached / portaled-with-GP ⇒ its pick step earns credit.
Needs per-card play-event export from the env (per-episode card-usage
bitmap alongside the deck record) and trainer-side routing from card →
pick step (pick order is in the deck record). Differentiates picks WITHIN
a draft — the thing the flat critic-aux never did — and is grounded in real
outcomes (no critic proxy). Do NOT stack with S2's bonus in the same arm.

## S6 — Scripted contrast episodes [free after S4 — backup]
Same fixed-deck mechanism, decks from build_archetype_decks instead of the
reference pool. Run only if S3/S4 fail to move the critic-differential
spread.

## S7 — 45M confirmation of winners
Compose the levers that passed their 15M gates (draftref within 3pp of
control AND either KL > 0 retained or critic-differential spread improved)
into one 45M with the full trajectory suite. Only promote S1 to 45M if its
15M gate passed (see S1).

## Notes for the main thread
- auxvd45 critic trajectory sweep still running (~22:30); append its numbers
  to Part III when done (does critic ratio also decay with the KL?).
- All aux arms MUST NOT ship in the production recipe (Part II §14 stands:
  portalgp, no aux) unless S7 changes the picture.
- Keep arms single-lever at 15M; combinations only at S7.

---

# PART IV — S-queue verdicts and the final production specification (2026-07-11 → 07-12)

## 19. S-queue results

| exp | lever | verdict |
|---|---|---|
| S1 annealed aux | aux coefs ride the shaping anneal | **Short-horizon win**: 63.9% pooled draftref at 15M-total (campaign best; cycle peak + lr polish); safe at 45M but window mean 45.6 < portalgp 49.0. Adopted for ≤15M runs / polish phases. |
| S2 outcome-graded portal bonus | pay only on ability resolution | **Negative**: portal usage collapsed to 0.0038 (whiffs = pure cost). "Pay for trying" is load-bearing; flat GP bonus stays. |
| S3 cross-gate replay 0.15 | same-deck-both-gates labels + boundary pick-masking | **Pass**: first mover on the critic-interaction blocker (spread 3.98 vs 6.4); externally free. Adopted. |
| s13combo (S1+S3) | seed + sustenance | Interaction learning compounds at 15M (spread 2.14); retention unchanged. |
| S7 = s13combo @45M | the composition at length | External parity with portalgp (window 49.1/56.2); seed persists to ~9M (vs ~5-8M) then decays; spread metric revealed as cycle-noisy. Single-box scale cannot make conditioning durable — final. |
| S8 gate-id drop | S7 recipe, text-only gates | **Pass**: external parity (46.9 vs 42.7 twin) AND the text projection separates exactly the compositionally-distinct sibling pairs (LIGHTNING 0.972, WATER 0.983 vs id-on 0.988/0.990) while FIRE/EARTH (interaction ≈ null) stay collapsed. Collapse-as-pressure-symptom confirmed. |

S4 (reference-deck league seats) and S5 (hindsight pick credit) remain
specced but unbuilt — the two highest-value follow-ups beyond this campaign
(S4 additionally attacks the meta-cycle oscillation at its root).

## 20. FINAL PRODUCTION SPECIFICATION

Recipe (distributed run):
- Base: shaping anneal (warmup 12 / ramp 40 per-env episodes, recalibrated
  per env count) + pick-eps 0.02 + sibling-matchup oversampling 0.35 +
  flat AZK_PORTAL_GP_BONUS 0.3 + league 6/4/3 windowed, native path.
- + cross-gate replay 0.15 with boundary pick-masking (free; feeds the
  critic the gate×composition contrast at scale).
- Gate representation: TEXT-ONLY (gate-id channel off) — parity today,
  generalizes to unseen gate cards.
- Short-horizon phases (≤15M) or a final polish phase: enable annealed aux
  (AZK_DRAFT_VBOOT_COEF 0.05, AZK_DRAFT_SIBDIFF_COEF 2.0, AZK_DRAFT_AUX_ANNEAL=1).
- Protocol: periodic draftref evals; CHECKPOINT SELECTION over a late window
  (every recipe's deployable peaks are 54-60%+); all trajectory metrics
  (draftref, KL, critic ratio, interaction spread) evaluated as WINDOWS,
  never single checkpoints (±10pp / large cycle noise).
- Engine posture: invalid-action truncation + repro logging (root cause
  outstanding, fuzz-clean over 32M random steps).

## 21. Final thoughts

The model demonstrably learns strategy at every level the game pays
first-order: element/family drafting, synergy pairs, deck-conditional
playstyle, and gate-aware portal play that extracts real (probe-verified)
sibling ability gaps. Sibling-conditional DRAFTING — the last strategic
axis — has now been shown to be: (a) genuinely valuable (up to 8pp
composition interaction, LIGHTNING), (b) creatable (aux seeding produces
the only actor-side conditioning ever measured), (c) sustainable on the
critic side (cross-gate contrast data teaches deck-dependent gate values,
and the text pathway differentiates where differentiation pays), but
(d) not yet retainable in the actor at single-box scale — the seed decays
as its source anneals away, and nothing measured at 45M holds it. The
remaining hypotheses are distributed scale (capacity + data volume) and the
two unbuilt levers (human-deck league seats anchoring the meta; hindsight
pick credit grounding draft rewards in realized card usage). The
infrastructure, probe suite, and windowed-evaluation protocol built here
are what the production run needs to answer the question definitively —
and to catch the answer live if it emerges.

## 22. League-health autopsy (2026-07-13, post-campaign question)
All five 45M-class runs: **zero champion promotions** (champion = ep100
p000001 throughout; 8-9 candidates/run all Wilson-rejected on
winrate_vs_champion_too_low), ELO inert (0-10 rated games/policy). The league
functioned as a recency-bucket DIVERSITY POOL (30 added → 13 active, 10%
frozen matchups) — which worked and preserved deck diversity — but not as a
strength ladder. Diagnosis: intransitive cycling meta (the same disease as
the ±10pp draftref oscillation) + conservative Wilson gate on tiny inline
samples + a stale ep100 yardstick. Production fixes, in order: S4
reference-deck league seats (external meta anchor — now doubly justified);
promotion by external draftref instead of champion head-to-heads; real
rating accumulation if internal gating is kept. Matches the user's historical
observation of promotion stalls in old 100M runs — config-independent,
structural.

## 23. H2H ladder: the league gate was RIGHT — self-play strength regresses (2026-07-13)
Offline seat-fair 192-ep matchups (portalgp45): final45M loses to the ep100
champion **40.1%** (matches the inline gate's 40.6% — the gate measured
truth) and loses to its own 15M checkpoint **32.3%**; 15M vs ep100 = 47.4%.
Policy-vs-policy strength peaks before ~15M then REGRESSES while draftref
oscillates sideways: strategy cycling with forgetting. ROOT CAUSE is the
matchup distribution: frozen_ratio is ROW-level and doubles to game-level
(×agents/(agents−1)), so 0.10 = ~18% league GAMES (measured 0.178-0.180) —
faithfully OpenAI Five's 80/20 with UNIFORM past-selves — split over 13
opponents via one-per-window sampling ≈ 1.4% of games per old style: no
pressure to remain robust to past strategies. I.e., the OpenAI Five recipe
itself, correctly replicated, fails under this game's strategy cycling. (The windowed
sampler's sequential one-style-at-a-time adaptation likely aggravates it.)
FIX PRIORITY for production (supersedes §22's ordering):
1./2. Ratio × selection ablation (S9, running): historical 0.10-rows
   (18% games, uniform — broken baseline) vs s9pfsp 0.4-rows (~78% games,
   PFSP, trainable rows 61%) vs s9pfsp02 0.2-rows (~36% games, PFSP) vs
   s9pfsp01 0.1-rows (18% games, PFSP — "was prioritization alone the
   missing piece?"). Judged by H2H ladder monotonicity, promotion
   acceptances, steady SPS, draftref window.
3. S4 reference-deck seats as the stationary anchor; the existing Wilson
   promotion gate should then function unchanged.
CONSEQUENCE: all 45M window means in this report sit on top of cycling —
recipe ceilings are likely UNDERESTIMATED; fixing the matchup mix is the
highest-leverage single change identified by the campaign.

## 24. S9 verdict: league volume creates improvement; the anneal knee destroys it (2026-07-13)
Ratio × selection matrix (15M arms, within-run H2H ladders, 192 eps/matchup):
18%-uniform mid>early 47.4 / 18%-PFSP 40.6 / 36%-PFSP 38.0 / **78%-PFSP 62.0**
— only ~78% league games produced absolute improvement (first monotone
segment of the project). PFSP without volume does not help. EVERY arm
regresses after ~8M — the window tracks the shaping-anneal knee, not the
matchup mix; the anneal-floor experiment (shaping floor 0.15) is the next
isolation. Deployment caveat discovered: external (draftref 61.5%) and
internal (final loses 80% to early) strength can diverge in the same
checkpoint — production selection needs BOTH yardsticks. ADOPTED into spec
§20: league.frozen_ratio 0.4 (≈78% league games) + AZK_PFSP=1 with
persistence (54ec27f); SPS cost mild (windowed two-batch design), trainable
rows 61%.
