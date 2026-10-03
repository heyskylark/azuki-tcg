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

## 24.5 S12 early-tempo bonus (user-designed): cap-4 adopted (2026-07-14)
Flat 0.1/action for development actions in each player's first two turns.
Cap 4/turn: draftref 54.2% (+7pp), cost 3.44→2.59 (intended shift, no
collapse), KL retained 4/4, ladder healthy → ADOPTED (spec §20). Uncapped:
draftref 29.2%, cost 2.41 = the cheap-aggro collapse — the Goodhart boundary
located empirically between cap-4 and unbounded. FIRE-sibling prediction not
confirmed (decks identical across the pair; the reward-mechanical asymmetry
is too marginal vs the common component).

## 24.6 S13-DMG damage-mitigation bonus (user-designed): ADOPTED (2026-07-14)
0.15 × min(soak,10)/10 to the intercepting player at combat resolution,
target-agnostic, opponent-gated. Externally neutral (46.9%), uniquely
endpoint-stable (the ONLY 15M arm whose final checkpoint is its strongest:
>mid 57.8, >early 56.2), zero turtling (ep_len/timeouts/attack unchanged),
Foamback-class defensive drafting +40%. With S12-cap4, the reward system now
prices development tempo AND damage mitigation — the offense-only bias
identified in June is fully closed out.

## 25. S10: the anneal floor completes the recipe — first net-improving run (2026-07-13)
s10annfloor (= S9 winner + shaping floor 0.15): mid>early **71.9%**,
**final>early 56.8% — the first end-to-end net improvement of the project**
(floor 0.05: 31.8%). final-vs-mid 40.1% — oscillation persists but around a
RISING trend. Bonus: end-of-run gate-swap KL nonzero in ALL FOUR elements
(EARTH 3.2e-5) with NO aux seeding — the denser floor also retains
conditioning. The post-anneal sparse phase was the regression driver.
SPEC §20 UPDATED: AZK_REWARD_SHAPING_ANNEAL_FINAL=0.15. Terminal validation
launched: s11final45 — the complete production recipe (portalgp + S3 replay
0.15 + text-only gates + 78% PFSP league + floor 0.15) at 45M with the full
trajectory + ladder suite.

## 26. S14: the combined recipe at 45M — validation, a campaign-best peak, and the promotion-gate diagnosis (2026-07-15)

S14 (`s14prod45`) ran the FULL production specification of §20 — including both
user-designed rewards (S12 early-tempo cap-4, S13 damage-mitigation) — for 45M
steps against control s11final45 (same spec/seed minus the two bonuses).

**Verdict: ADOPTED.** The combined recipe is confirmed at 45M, composing with
text-only gates, and produced the strongest external checkpoint of the entire
campaign.

### 26.1 Strength results
- Final checkpoint draftref **49.7%** (n=384, timeouts 0) vs control 46.9%.
- Windowed draftref (ep500→2930): 39.6 / 51.0 / 35.4 / 78.1 / 52.1 / 52.1 /
  49.7. Meta-cycle oscillation persists at ±20pp — windowed evaluation is not
  optional.
- **ep2000 = 76.5% pooled (n=384): the best external result at any horizon in
  the campaign** (prior best: 63.9–70.8% at 15M with S1-aux; prior 45M best:
  46.9%). Confirmed at n=288 after the n=96 window flagged it, and it beats
  the final checkpoint head-to-head (55.7%) — internal and external yardsticks
  agree on the peak.
- Ladder shows early(ep500) > mid ≈ final internally, but ep500's draftref is
  only 39.6%: this is internal/external divergence (in-meta specialization),
  not forgetting. **Checkpoint-selection protocol finalized: windowed draftref
  primary, H2H agreement secondary.** It picks ep2000 unambiguously.
- Probes: sibling gate-KL retained on LIGHTNING (4.6e-6) and WATER (3.8e-6) —
  precisely the pairs whose text embeddings separate (S8); FIRE/EARTH at noise.
  Critic sign-consistency 12/12 (L).

### 26.2 What S12+S13 change at 45M (final-bucket meta vs control)
avg cost 2.57 vs 3.01; portal rate 0.094 vs 0.074 (+27%); attack rate 0.203
vs 0.270; episode length 112 vs 87 with ZERO eval timeouts — the tempo shift
and the defensive action budget both persist at 3× the validation horizon
without turtling. In-meta consequences: Water and Devotion win rates rose
(EchoedWaves .542 vs .438, Devotion .509 vs .366), Fire fell (Ragefire .366
vs .497) — the defensive unlock re-prices aggro.

### 26.3 Per-gate strategy atlas (new deliverable)
`profile_gate_matchups.py` (8-gate round-robin, 36 pairings × 16 eps,
seat-fair, sampled, forced gates, zero timeouts) + per-gate draftref splits:

- **Field ranking (internal, win-vs-field):** Surge .661, Stormchain .634,
  EchoedWaves .598, Hydromancy .536, Devotion .500, Stonehaven .411,
  Rushfire .384, Ragefire .277. External draftref ordering agrees at the
  extremes (Surge 79.7%, Ragefire 20.5%).
- **Element archetypes are behaviorally legible in the action traces:**
  LIGHTNING = tempo-attack (highest attack rates ~.245, weapon-heaviest decks
  16%, short-mid games); WATER = spell-control (41% spells, EchoedWaves uses
  SELECT_FROM_SELECTION at .061 — the draw/filter identity, longest-but-one
  games); FIRE = burn-aggro (shortest games, highest SELECT_EFFECT_TARGET
  .154–.176, declare-defender ≈.002 — never blocks); EARTH = attrition
  (longest games 150–163 steps, highest ability rates, **Stonehaven
  declare-defender .031 = 5–11× every other gate** — the S13 archetype
  unlock made visible).
- **Siblings draft nearly identical decks** (within-gate Jaccard .70–.72,
  same top cards) **but play differently** — within-pair splits in portal,
  declare, and selection rates. Draft-time sibling indifference (the known
  limit) coexists with real play-time conditioning.
- **Gate power is meta-relative:** Rushfire, dominant on the June
  neutral-deck ladder (57.7%), is weak (38.4%) against policy-drafted
  low-curve decks. Balance memo: Ragefire is worst on every axis in both
  eras (its net-negative portal stands); the LIGHTNING pair is overtuned in
  the current meta.

### 26.4 League: the promotion gate measures the wrong axis
League mechanics ran exactly as configured (frozen-matchup fraction 0.795,
PFSP live, pool 6/4/3). But `promotion_accepted = 0` for the entire run while
candidate-vs-champion winrate DECLINED 0.59 → 0.34 — during the same span in
which external strength rose to the campaign-best 76.5%. The Wilson gate
anchors promotion to head-to-head vs an ep100 champion, and in this game that
internal axis anticorrelates with external strength (§26.1's divergence, now
measured inside the league itself). This is the core issue behind "the model
never promotes in long runs" — the gate is honest but aimed at the wrong
target. Fix: give promotion an external anchor. **S4 (reference-deck league
seats) is upgraded from optional to the highest-leverage next build**; the
cheap interim alternative is re-anchoring the gate on windowed draftref.

### 26.5 Spec deltas to §20
1. S12 cap-4 + S13 damage-mitigation: CONFIRMED at 45M with text-only gates —
   keep both (AZK_EARLY_TEMPO_BONUS=0.1/CAP=4, AZK_DMG_MITIGATION_BONUS=0.15/CAP=10).
2. Checkpoint selection: windowed draftref (96+ eps per checkpoint across the
   final third of the run, confirm peaks at n≥288) + H2H agreement check.
   Never ship the endpoint blindly.
3. Promotion gate: re-anchor on an external yardstick (S4 reference seats or
   draftref-based gate) before the distributed run.

## 27. S4/S5 handoff state for future sessions (2026-07-15)

**Both are ON HOLD by explicit user instruction ("please dont build or run s4
or s5 yet"). Do not launch either without a fresh user go-ahead.**

### S4 — reference-deck seats: BUILT (inert), never validated
The core mechanism was implemented and committed (4f7e79a, 24d22da) before
the hold arrived; it is dormant unless its env vars are set (off-parity
proven by test_draft_ref_seat.py):
- `AZK_DRAFT_REF_SEAT_PROB` — prob one seat/episode skips the draft and plays
  a deck from the pool (battle uses the spec verbatim; S3 gate-swap skips the
  ref seat; drafter seat unaffected).
- `AZK_DRAFT_REF_DECK_INDICES` — csv restricting which pool decks serve as
  ref seats. Split: evens 0-16 = train, odds 1-17 = holdout (18-deck pool).
- Telemetry: episode records carry ref_seat/ref_deck_index; metrics emit
  `ref_anchor_winrate` (drafter winrate vs fixed decks — the external
  promotion yardstick per §26.4) and `ref_seat_rate`; snapshots tag ref
  episodes and analyze_decks excludes them from per-gate draft stats.
- Eval support: `draft_vs_reference_eval.py --deck-indices` /
  `AZK_FIXED_SEAT_DECK_INDICES` (legacy wrapper) for leak-free holdout evals.
- Driver ready: `run_s4.sh` (15M smoke, prob 0.20, full prod spec, holdout
  readout incl. s14prod45 ep2000/final holdout controls). Its first launch
  was killed at ~2M steps on the hold; partial artifacts remain under
  experiments/*s4ref15*. Known open item: observed ref_seat_rate ran ~0.34
  vs 0.20 configured (ref episodes cycle faster — half-length drafts);
  check the settled value before tuning prob.

### S5 — hindsight pick credit: NOT BUILT
Spec unchanged (Part IV): engine/env export of per-card play events routed
back to the originating pick steps as annealed, win-gated credit. Largest
build in the queue; nothing started.

### Promotion redesign discussed with the user (not yet implemented)
Motivated by §26.4 (lone-champion Wilson gate anticorrelated with external
strength). Agreed direction from discussion:
1. **Panel-based gate**: evaluate candidates vs K=4-6 seats — 2-3 recent
   pool members (live meta), 1-2 past champions/old members (retention),
   optionally ONE external reference-deck seat (measurement-only). Promote
   iff Wilson-LB of pooled winrate > ~0.52 AND quorum (>=break-even vs >=3
   of 4 seats) AND no matchup < ~0.35. Panel ages with the pool — removes
   the stale-champion trap by construction.
2. **Gate-paired evals**: promotion games in mirrored gate pairs (swap gate
   assignment across each pair) to cancel gate-power luck (Surge ~.66-.80
   vs Ragefire ~.21-.28 makes n=16 unpaired evals unacceptably noisy).
   Mirror-gate evals are the stronger variant (both seats same gate).
3. **Training matchups stay randomized** (+ existing 0.35 sibling
   oversampling); at most ~10-15% mirrors if ever tried — heavy mirror
   training would under-train cross-gate matchup skill.

**Deferred until after the current reward-shaping ablations and qualifying
45M confirmations.** Do not change the league gate inside matched runs. Before
implementation, discuss and fix the panel membership/aging policy, mirrored
gate protocol, sample size/confidence rule, quorum and matchup floor, and
whether an external reference seat is measurement-only or part of the gate.
Until then, legacy promotion outcomes are contextual matchup/cycle telemetry,
not a model-improvement criterion.

**Future promotion TODO (ordered dependency):**
1. **Redesign promotion evaluation for validity first.** Make promotion track
   improvement in this TCG by settling the opponent panel, paired seats/gates,
   retention checks, sample/confidence rule, and matchup quorum/floor. Validate
   that the resulting decisions agree with broader external and strategic
   evidence before treating promotion as a model-quality signal.
2. **Optimize the validated evaluator second.** Once the game-aligned protocol
   is fixed, replace/profile the legacy serial one-environment path and use an
   appropriate native, batched, and/or CPU-parallel implementation. Preserve
   identical matchup semantics and raw per-opponent/seat telemetry while
   reducing gate wall time and verifying that training SPS is not regressed.

Do not optimize the legacy decision rule into permanence: evaluation speed is
subordinate to first establishing that promotion measures the right thing.

### User's design stance (respect in any future S4 work)
Reference decks are an ANCHOR (beat them), never a TARGET (resemble them):
no imitation terms; keep ref-seat share small or measurement-only; the
9/9 train/holdout split exists to detect overfitting to the training refs;
the goal remains decks humans would not build (the current meta's cheap
near-singleton piles are exactly that — see s14prod45_ep2000_decks.md).

## 28. Competitive-play reward shaping build (2026-07-15)

Four article-derived reward signals were implemented as default-off native
knobs for matched smoke tests. The control is **S14 (`s14prod45`)**, not an
older reward baseline. Every arm retains the full adopted stack: shaping
anneal 1.0 -> 0.15 (12/40 episodes), portal-GP 0.3, S12 early-tempo 0.1 cap
4, S13 damage mitigation 0.15/cap 10, text-only gates, pick smoothing 0.02,
same-element oversampling 0.35, cross-gate replay/mask 0.15, frozen ratio
0.4, PFSP, league retention 6/4/3, and seed 42. Arms start from scratch so
the experiment measures early-learning guidance; matched S14 checkpoints
at ep100/300/1000 are the controls.

### 28.1 Signals and overlap controls

1. **Entity-damage exchange ledger**
   (`AZK_ENTITY_DAMAGE_EXCHANGE_PER_HP=0.025`, step cap 6). Counts effective
   non-leader damage with overkill removed, then rewards opponent damage
   minus own damage. It overlaps the existing garden-attack potential and
   board-delta reward. To limit double counting, there is no separate kill
   bonus, the maximum new step reward is 0.15 before annealing, and it is a
   standalone arm. Nondamage destruction is deliberately excluded.
2. **Generated/recovered IKZ conversion**
   (`AZK_GENERATED_IKZ_CONVERSION_BONUS=0.05`, step cap 4). Hydromancy-style
   untaps and effect-created IKZ are marked at the source; credit is paid
   only when that exact source is later tapped for a cost. Unspent credit is
   cleared at natural turn refresh. This arm sets
   `AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0`, replacing rather than stacking with
   the unconditional untapped-IKZ potential.
3. **Temporary-effect realization**
   (`AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08`, temporary attack damage
   0.025/HP, cap 4). Temporary Charge earns credit only when its attack deals
   positive effective damage. Positive end-of-turn attack buffs earn only
   their incremental effective damage, with overkill removed; innate or
   permanent Charge earns nothing.
4. **Contextual paid-response reserve**
   (`AZK_CONTEXTUAL_RESPONSE_RESERVE_BONUS=0.08`). After an opposing attack,
   the defender earns credit at most once per opposing turn only when the
   already-built response mask contains a currently legal action with a
   positive IKZ cost. Free Defender declarations do not qualify. This arm
   also sets `AZK_REWARD_UNTAPPED_IKZ_WEIGHT=0`.

All four signals ride the existing shaping anneal and zero-sum channel.
Engine counters for signals 1 and 2 are gated behind their active reward
variables, so default-off runs do not add hot-path ECS lookups. Signal 4
scans the observation mask already generated for the defender; it does not
rebuild or revalidate the action mask.

### 28.2 Validation and smoke protocol

- Full C engine test binary passed.
- Existing portal-GP, early-tempo, and damage-mitigation differential tests
  passed (4 tests total).
- New deterministic differential tests passed for all four signals: action
  trajectories remain identical, reward deltas are zero-sum, each signal
  fires, and coefficient/cap quantization is exact.
- Matched trainer SPS probes compare S14-off vs each active arm before long
  runs. Hard guardrail: investigate any repeatable >5% SPS loss; reject or
  optimize at >10%. The 15M runs must also remain near historical S14 rates
  both before league saturation (~3K SPS) and with the full pool
  (~1.3-1.6K observed historically).
- The first 153,600-step isolated matrix (league disabled identically for all
  arms) cleared that gate. Tail SPS was 4,747.7 for S14 control; entity damage
  4,787.0 (+0.8%); generated-IKZ conversion 4,552.6 (-4.1%); temporary-effect
  realization 4,670.1 (-1.6%); contextual reserve 4,847.6 (+2.1%). Total
  wall-clock runtimes were 82.7-84.2 seconds, so the tail variation did not
  translate into a material end-to-end slowdown. Entity-damage and temporary
  effect telemetry fired in sampled games; the rarer IKZ/reserve paths are
  covered by deterministic differential tests. Artifacts:
  `results/reward_shaping_sps_v2/`.
- Because generated-IKZ conversion landed closest to the 5% review line, it
  received a longer paired 307,200-step repeat. It measured 4,843.8 SPS versus
  4,695.6 control (+3.2%), with 117.0 versus 128.3 seconds wall time and 228.7
  versus 234.2 microseconds/profiled environment step. The longer sample also
  observed the rare path: about 1.2-1.4 generated sources and 0.7-0.9 exact
  conversions per completed episode across seats. Artifact:
  `results/reward_shaping_sps_ikz_repeat/`. No isolated arm has a measurable
  SPS regression requiring optimization before full-league training.
- One 15M trajectory per arm supplies ep100 (~1.54M), ep300 (~4.61M), and
  final (~15M) checkpoints without changing the production checkpoint/league
  cadence. Readout is draft-vs-reference, training action/reward telemetry,
  and sibling-gate KL. Expensive deck-composition, strategy, and Defender
  probes are not launched automatically.
- Matched S14 control draftref is 45/96 = 46.9% at ep100, 37/96 = 38.5% at
  ep300, and 49/96 = 51.0% at ep1000; all three have zero timeouts. The
  ep1000 result reuses the exact existing S14 argmax artifact. These controls
  make the known early/mid/final oscillation explicit instead of comparing
  every candidate checkpoint only with the S14 endpoint.
- `draft_vs_reference_eval.py` now evaluates the two independent fixed-seat
  halves concurrently by default using spawn workers (with
  `--no-parallel-seats` as the deterministic serial fallback). A live schema
  smoke passed. On the ep300 control this used two CPU cores and 3.3 GB VRAM,
  raised GPU utilization from about 13% to 35-45%, and reduced the 96-game
  wall time from roughly 12 minutes to roughly 6 minutes without changing
  per-seat seeds or aggregation.
- Promotion is fixed before reading the remaining endpoints: advance an arm
  when its 15M draft-vs-reference score is above matched S14, its aggregate
  across ep100/ep300/final is nonnegative versus S14, and there is no timeout,
  action-collapse, or sustained SPS regression. A tied endpoint with only
  early gains is borderline and does not automatically consume a 45M run.
  Because 96 games have wide sampling error, the three-checkpoint aggregate
  is supporting trajectory evidence rather than a substitute endpoint.

### 28.3 Learned turn-boundary potential (task 5, held)

If the hand-designed arms leave a clear gap, train the turn-boundary
predictor from completed S14/candidate trajectories as a separate supervised
model, validate calibration and ranking out of sample, then freeze it before
using `gamma * Phi(s') - Phi(s)` for PPO shaping. Training it jointly with the
policy would make the reward target move underneath PPO and would weaken the
policy-invariance argument. A frozen predictor can later be refreshed only
between explicit training stages, never continuously inside one run.

### 28.4 Smoke results

#### Task 1: entity-damage exchange (15M complete; 45M qualified)

Run `azuki_local_rs1entity15_178416708169` completed 977 epochs / 15.006M
sampled steps with no timeouts. Draft-vs-reference improved at every matched
checkpoint: 53/96 = 55.2% at ep100 versus S14's 45/96 = 46.9% (+8.3 pp),
44/96 = 45.8% at ep300 versus 37/96 = 38.5% (+7.3 pp), and 50/96 = 52.1%
at the endpoint versus S14 ep1000's 49/96 = 51.0% (+1.0 pp). Aggregated over
the matched trajectory, Task 1 scored 147/288 = 51.0% versus 131/288 = 45.5%
(+5.6 pp). This is a **45M qualifier**: the consistent trajectory
is encouraging, while the endpoint alone is only one game better and should
not be read as a precise 1 pp effect.

The gain was not yet seat-robust. With the drafter in seat 1, Task 1 was +8,
+6, and +2 wins versus S14 at ep100/ep300/final; with the drafter in seat 0,
it was flat, +1, and -1. Both halves contain 48 games and the reported totals
are seat-balanced, so this does not invalidate the aggregate, but the 45M
confirmation must show that strength broadens or at least does not become
more dependent on one seat.

The early/mid benefit was broadest in Earth and Water. At ep300, Earth moved
from 8/22 to 12/22 and Water from 7/25 to 10/25; Lightning gained one game and
Fire lost one. At the endpoint, Lightning and Earth were each +1 game, Water
was unchanged, and Fire was -1. This fits the intended board-combat signal but
also flags that direct-damage Fire may receive less benefit.

The learned behavior was active rather than merely defensive. In the final
100-epoch window versus matched S14, attacks were 20.76% versus 22.61%, portal
actions 12.58% versus 8.98%, plays 35.07% versus 31.93%, abilities 4.72%
versus 3.43%, no-ops 14.97% versus 15.54%, and mean episode length 100.90
versus 98.56. Mean leader health was 0.250 versus 0.224, with about 7.44
effective entity damage dealt per episode. The extra portal/board activity is
consistent with useful board control, but it also confirms partial behavioral
overlap with the existing portal-GP and board-potential rewards.

The internal league gates oscillated despite the positive external trajectory.
At ep300, Task 1 was 62.5% versus the champion and 59.4% against its worst
baseline. At ep600 those fell to 37.5% and 31.25%; by ep900 they recovered to
53.1% and 43.8%. For context, matched S14's ep900 values were 43.8% and 40.6%.
No gate cleared the Wilson promotion rule. These values are contextual cycle
and matchup telemetry only: the current promotion rule is known to miss large
external improvements and does not qualify or veto this arm. The 45M decision
uses the fixed external/action/stability rule above.

Full-league performance stayed inside the 5% guardrail. Endpoint SPS was
1,520.3 versus S14's 1,503.8 (+1.1%); the final 100-epoch mean was 1,472.1
versus 1,512.0 (-2.6%). Total time through ep977 was 2h19m55s versus S14's
2h16m25s (+2.6%), including candidate-side evaluation overlap and somewhat
longer games. Sibling-gate mean KL remained numerically tiny but above S14:
1.09e-5 versus 3.73e-7 at ep100, 4.23e-7 versus 2.14e-7 at ep300, and
1.53e-7 versus 6.56e-8 at the endpoint. The endpoint difference was driven
mainly by Lightning and Water rather than uniform gate divergence.

Artifacts: `results/reward_smokes/rs1entity15/`,
`experiments/runlogs/rs1entity15_178416708169.jsonl`, and
`experiments/azuki_local_rs1entity15_178416708169/`.

**Verdict: promote to a fresh matched 45M confirmation.** It clears the fixed
external/action/stability rule. The confirmation must show that the gain
broadens or at least does not become more dependent on drafter seat 1.

#### Task 2: generated/recovered IKZ conversion (15M complete; 45M qualified)

The ep100 and ep300 checkpoints are complete. The trajectory starts worse
than S14, then reverses sharply: ep100 scored 37/96 = 38.5% versus 45/96 =
46.9% (-8.3 pp), while ep300 scored 49/96 = 51.0% versus 37/96 = 38.5%
(+12.5 pp). Both evaluations had zero timeouts. At ep100, the drafter was +2
wins versus S14 in seat 1 but -10 in seat 0. At ep300 it was +6 in each seat,
so the mid-checkpoint gain is seat-balanced rather than one favorable half.

The element split points to delayed discovery of the intended resource chain.
At ep100, Lightning was +1 game and Fire -1, but Earth was -3 and Water -5.
At ep300, every element was positive: Lightning +2, Fire +2, Earth +3, and
Water +5. Removing the easy untapped-IKZ potential appears to hurt Water/Earth
before the policy learns to create and spend the marked IKZ, after which the
same elements show the largest gains. Across the first ~718 completed episode
batches, about 84% of marked sources were eventually spent, averaging about
1.5 conversions across both seats per game.

Through the matched ep110-200 league window, SPS was 1,832.2 versus S14's
1,833.5 (-0.07%), attacks were 23.86% versus 22.77%, portal actions 11.11%
versus 9.06%, plays 31.48% versus 30.82%, abilities 6.80% versus 9.68%, and
mean episode length 107.5 versus 116.9. This is consistent with converting
available IKZ into attacks, portals, and board actions rather than preserving
it for the removed raw potential.

The 15M endpoint scored 53/96 = 55.2% versus S14's 49/96 = 51.0% (+4.2 pp),
again with zero timeouts. Across ep100/ep300/final, Task 2 scored 139/288 =
48.3% versus 131/288 = 45.5% (+2.8 pp). At the endpoint, the drafter was -1
game versus S14 in seat 1 and +5 in seat 0; over all three checkpoints, the
two seat aggregates were +7 and +1, respectively. The trajectory is therefore
positive in both seats overall despite its severe early seat-0 deficit.

The final element effect is concentrated and strategically plausible:
Lightning was +2 games, Earth +2, and Water +5, while Fire was -5. The Water
gain matches the intended Hydromancy/resource-conversion behavior. The Fire
loss is the main 45M risk and may reflect replacing a generic resource-reserve
signal with one whose marked sources are concentrated in Water/Earth cards.

Mean sibling-gate KL was 1.05e-6 versus S14's 3.73e-7 at ep100, 1.09e-6
versus 2.14e-7 at ep300, and 8.10e-8 versus 6.56e-8 at the endpoint. Values
remain small, but differentiation did not collapse and the residual endpoint
difference is concentrated in Lightning and Water. The internal ep300 gate beat the
ep100 champion 65.6%, but its worst retained-baseline matchup was only 31.25%,
so promotion was rejected. This is useful evidence of a cyclic or specialized
matchup state, but the current league gate is not a reliable improvement signal
and has no veto over the balanced external gain. At ep600, champion H2H was
59.4% and the worst baseline improved to 37.5%; the legacy rule still rejected
it. At ep900 those values cycled to 46.9% and 34.4%. These diagnostics do not
alter the external verdict.

In the final 100-epoch floor window, SPS was 1,489.5 versus S14's 1,512.0
(-1.5%), attacks 21.84% versus 22.61%, no-ops 15.87% versus 15.54%, portal
actions 12.76% versus 8.98%, plays 33.59% versus 31.93%, abilities 6.19%
versus 5.84%, mean episode length 93.22 versus 98.56, and mean surviving
leader health 0.232 versus 0.224. The active conversion reward was only about
0.006 per player/game at the 0.15 floor, so the retained portal/play shift was
not dominated by dense return. Total time through ep977 was 2h20m16s versus
S14's 2h16m25s; the 3m51s difference is almost exactly the extra concurrent
checkpoint evaluation time at ep300, while matched rollout SPS stayed within
the guardrail.

**Verdict: promote to a fresh matched 45M confirmation.** It clears the fixed
external/action/stability rule, but the confirmation must track whether the
Water gain persists without sacrificing Fire and whether both seats remain
nonnegative. Artifacts:
`results/reward_smokes/rs2ikzconv15/`,
`experiments/runlogs/rs2ikzconv15_178417646702.jsonl`, and
`experiments/azuki_local_rs2ikzconv15_178417646702/`.

If the endpoint remains positive but the ep100 deficit matters, do not restore
the conflicting unconditional untapped-IKZ potential. A cleaner follow-up is
reward redistribution with the same total return: for example, +0.01 when a
source is marked, +0.04 when it is spent, and -0.01 when it expires unused.
A converted source still nets 0.05 and an unspent source nets zero, while the
immediate 0.01 supplies an easier discovery cue. This is a held hypothesis,
not part of the current matched arm.

#### Task 3: temporary-effect realization (15M complete; 45M qualified)

The matched arm rewards only realized value: +0.08 when an attack enabled by
temporary Charge resolves for positive effective damage, and +0.025 per point
of positive effective attack damage caused by a temporary attack modifier,
capped at four damage. Merely granting Charge or temporary attack produces no
reward, and overkill is excluded. Both paths remained live. Across ep1-99,
the two seats averaged 0.116 temporary-Charge realizations and 0.708
temporary-attack damage per game; through ep300-599 those values were 0.091
and 1.199.

The first two external checkpoints were modestly positive with zero timeouts.
Ep100 scored 47/96 = 49.0% versus S14's 45/96 = 46.9% (+2.1 pp), and ep300
scored 39/96 = 40.6% versus 37/96 = 38.5% (+2.1 pp). At ep100, seat 1 gained
three wins while seat 0 lost one; at ep300, each seat gained one.

The endpoint strengthened to 53/96 = 55.2% versus S14's 49/96 = 51.0%
(+4.2 pp), again with zero timeouts. Across ep100/ep300/final, Task 3 scored
139/288 = 48.3% versus 131/288 = 45.5% (+2.8 pp). Its final seat deltas were
-1 and +5 games, while the three-checkpoint seat aggregates were +3 and +5.
The positive trajectory is therefore present in both seat assignments rather
than being carried by one fixed-seat half.

Element movement was mixed early but broadened by the endpoint. Ep100 deltas
were Lightning +3, Water +2, Earth -1, and Fire -2; ep300 moved Earth +3 and
Lightning +1, held Fire flat, and lost two Water games. At the endpoint,
Lightning was +2, Water +3, Earth tied, and Fire -1. Across all three
checkpoints the element deltas were Lightning +6, Water +3, Earth +2, and
Fire -3. Fire remains the 45M risk, but the gain is not an element-specific
shortcut.

Sibling-gate differentiation did not collapse. Mean KL was 2.77e-6 versus
S14's 3.73e-7 at ep100, driven mainly by Water; 2.83e-6 versus 2.14e-7 at
ep300, driven mainly by Lightning; and 9.31e-8 versus 6.56e-8 at the endpoint,
again driven by Water. These are supporting directional evidence only because
the absolute divergences remain very small.

Performance stayed inside the fixed 5% guardrail. Early ep20-40 SPS was
3,531.6 versus S14's 3,628.2 (-2.7%), ep110-200 was effectively identical at
1,833.4 versus 1,833.5, and ep500-599 was 1,504.1 versus 1,514.4 (-0.7%). In
the final 100 epochs, SPS was 1,504.9 versus 1,512.0 (-0.5%), attacks 22.13%
versus 22.61%, no-ops 14.66% versus 15.54%, portal actions 11.51% versus
8.98%, plays 32.21% versus 31.93%, abilities 6.78% versus 5.84%, mean episode
length 93.13 versus 98.56, and surviving leader health 0.195 versus 0.224.
There were no timeouts or action collapse.

At the 0.15 floor, the final window averaged 0.108 temporary-Charge
realizations and 1.505 temporary-attack damage per game across both seats.
That is only about 0.00694 gross owner-side credit per game summed over both
players (about 0.00347 per player/game before the opposite-seat zero-sum
transfer), so the retained action shift is not dominated by a large dense
return. Total uptime through ep977 was 2h30m17s versus S14's 2h16m25s. The
13m52s excess is accounted for by the required ep100/ep300 probes competing
with the serial
epoch-600 league evaluator; matched rollout windows and endpoint SPS did not
show a persistent regression.

The legacy internal gates cycled from 59.4% champion / 62.5% worst-baseline
at ep300 to 43.8% / 43.8% at ep600 and 46.9% / 40.6% at ep900. None promoted.
As with the other arms, this is contextual matchup telemetry and neither
qualifies nor vetoes Task 3.

**Verdict: promote to a fresh matched 45M confirmation.** It clears the fixed
external/action/stability rule. The confirmation should test whether the
broad Lightning/Water/Earth gain persists at the shaping floor without
deepening the smaller Fire regression. Artifacts:
`results/reward_smokes/rs3tempreal15/`,
`experiments/runlogs/rs3tempreal15_178418553588.jsonl`, and
`experiments/azuki_local_rs3tempreal15_178418553588/`.

#### Task 4: contextual paid-response reserve (15M complete; rejected)

This arm replaced raw untapped-IKZ potential with +0.08 at most once per
opposing turn when an attack opened a response window containing a currently
legal positive-cost action. Free Defender declarations did not qualify. The
signal was easy to discover: ep20-99 averaged about 1.51 qualifying
opportunities per game across both seats, and ep110-200 averaged 1.12. At the
0.15 floor, the final window still averaged 0.717 per game, only about 0.00430
gross owner-side credit per player/game before the zero-sum transfer.

External performance did not improve. Ep100 scored 41/96 = 42.7% versus
S14's 45/96 = 46.9% (-4.2 pp), ep300 recovered narrowly to 38/96 = 39.6%
versus 37/96 = 38.5% (+1.0 pp), and the endpoint fell to 39/96 = 40.6%
versus 49/96 = 51.0% (-10.4 pp). All evaluations had zero timeouts. Across
the trajectory, Task 4 scored 118/288 = 41.0% versus 131/288 = 45.5%
(-4.5 pp), well below the fixed qualification rule.

The failure is strongly seat-asymmetric. Task 4's three-checkpoint aggregate
was 48/144 in drafter seat 1 versus S14's 66/144 (-18 games), while seat 0
was 70/144 versus 65/144 (+5). The endpoint alone was -11/+1 by seat. This is
not a small sampling wobble that the positive seat can safely offset.

Element results also reject a narrow matchup explanation. Ep100 deltas were
Earth -2, Fire -1, Lightning +1, and Water -2; ep300 was Earth +2, Fire -2,
Lightning +1, and Water tied. At the endpoint every element was negative:
Earth -4, Fire -3, Lightning -1, and Water -2. Across all checkpoints only
Lightning remained positive (+1), while Earth was -4, Fire -6, and Water -4.

Sibling-gate differentiation did not collapse: mean KL was 7.16e-6 versus
S14's 3.73e-7 at ep100, 3.85e-7 versus 2.14e-7 at ep300, and 1.29e-7 versus
6.56e-8 at the endpoint, mainly from Lightning and Water. This is useful
negative evidence: greater gate-conditioned draft divergence alone did not
produce competitive decks or play.

Performance and action stability were clean, so rejection is about strategy,
not implementation cost. Ep20-40 SPS was 3,633.9 versus 3,628.2 (+0.2%),
ep110-200 was 1,805.7 versus 1,833.5 (-1.5%), and the final 100 epochs were
1,511.0 versus 1,512.0 (-0.1%). In that final window, attacks were 24.77%
versus 22.61%, no-ops 14.49% versus 15.54%, portals 10.55% versus 8.98%,
plays 31.22% versus 31.93%, abilities 7.36% versus 5.84%, episode length
89.92 versus 98.56, and surviving leader health 0.194 versus 0.224. Total
uptime through ep977 was 2h16m09s versus S14's 2h16m25s.

The internal gates were also weak (31.25% champion / 50.0% worst-baseline at
ep300, 31.25% / 34.4% at ep600, and 34.4% / 34.4% at ep900), but they do not
drive this verdict. The balanced external endpoint and aggregate already do.

**Verdict: no 45M confirmation.** Rewarding response availability can pay
for holding IKZ through a response window even when the paid action is not
used or would not improve the outcome. If this family is revisited after the
current confirmations, prefer realized credit for a paid response that
actually mitigates effective damage or changes combat, with the generic
untapped-IKZ term still disabled. Do not tune the coefficient on this failed
availability target first. Artifacts:
`results/reward_smokes/rs4reserve15/`,
`experiments/runlogs/rs4reserve15_178419523079.jsonl`, and
`experiments/azuki_local_rs4reserve15_178419523079/`.

### 28.5 Fresh 45M confirmation protocol (fixed before the runs)

Every qualifying arm starts from scratch on the exact S14 stack and seed 42;
the 15M checkpoint and league state are not resumed. This preserves matched
anneal counters, opponent-pool formation, and early trajectory. The qualified
set is fixed at Tasks 1-3; Task 4 failed its 15M endpoint and aggregate.

The external readout uses the same S14 meta-cycle checkpoints and seeds:
ep1000, 1500, 2000, 2500, and 2900 receive 96 balanced argmax games each, and
the final ep2930 checkpoint receives 384 games. Existing S14 controls are
49/96, 34/96, 75/96, 50/96, and 50/96 for the five windows (258/480 total),
plus 191/384 at the endpoint. The multiple windows are required because S14
itself ranges from 35.4% to 78.1%; a single checkpoint would mostly measure
meta-cycle phase.

A 45M arm is confirmed when either (a) its 384-game endpoint is strictly
above 191/384 and its five-window aggregate is at least 258/480, or (b) its
endpoint is at least 191/384 and its five-window aggregate is strictly above
258/480. It must also have zero meaningful timeout regression, no action or
seat collapse, no new catastrophic element matchup, and no sustained >5% SPS
loss. Results that trade a material endpoint loss for a window gain, or vice
versa, remain mixed rather than being rescued by internal promotion.

Sibling-gate KL is measured at the same six checkpoints. Training action,
reward-signal, league-cycle, and SPS telemetry is compared with S14 across the
trajectory and at the shaping floor. Legacy promotion remains diagnostic only.
The held strategy/deck-composition/Defender probes and reference-seat training
are not part of this stage. CPU-only KL probes may overlap post-training
draftref evaluation, but no probe overlaps training. Driver:
`run_competitive_reward_45m.sh`.

### 28.6 Task 1 fresh 45M confirmation (complete; rejected)

The entity-damage exchange arm ran from scratch on the exact S14 stack as
`rs1entity45_178420486524`. It did not confirm the positive 15M smoke. Its
five external windows were 49, 35, 46, 36, and 43 wins out of 96, compared
with S14's 49, 34, 75, 50, and 50. The resulting aggregate was **209/480 =
43.5% versus 258/480 = 53.8%** (-49 wins). The curve was effectively tied
through ep1500 (84/192 versus 83/192), then missed S14's strong ep2000 state
by 29 wins and remained below control at ep2500 and ep2900.

The larger endpoint independently rejected the arm: **168/384 = 43.8%
versus 191/384 = 49.7%** (-23 wins). The loss was present in both seats:
drafter seat 1 scored 74/192 versus 87/192 (-13), and drafter seat 0 scored
94/192 versus 104/192 (-10). All six external evaluations had zero timeouts.
Endpoint element results were mixed rather than uniformly worse: Lightning
was 53/109 versus 73/109 (-20), Water 45/106 versus 59/106 (-14), Fire 36/85
versus 25/85 (+11), and Earth tied at 34/84. The Fire improvement does not
offset the broad Lightning/Water and both-seat regressions.

The late training phenotype was coherent but not competitively better. Over
the final quarter (the last five fixed comparison buckets), mean SPS was
1,441 versus 1,452 (-0.7%), safely inside the performance guardrail. Relative
to S14, attacks rose from 20.77% to 23.67%, no-ops from 13.72% to 15.31%, and
portal actions from 9.42% to 10.60%. Plays were nearly flat (31.07% versus
30.80%), while abilities fell from 8.63% to 6.89%, mean episode length fell
from 110.72 to 87.67, and surviving leader health fell from 0.211 to 0.186.
This is an aggressive, shorter-game specialization, not an action collapse or
an implementation-speed failure, but the fixed reference decks exploit it.

Sibling-gate differentiation was unstable and was not a quality proxy. Mean
KL for Task 1 versus S14 was 6.71e-7 versus 6.56e-8 at ep1000, 1.67e-7 versus
4.80e-8 at ep1500, 2.65e-9 versus 6.90e-7 at ep2000, 1.03e-7 versus 1.89e-7
at ep2500, 4.12e-8 versus 8.26e-8 at ep2900, and 2.36e-6 versus 5.51e-8 at
the endpoint. The near-zero ep2000 differentiation coincided with the largest
external deficit, but the much larger final differentiation still produced a
weak endpoint. Draft KL remains supporting telemetry, not an optimization
target.

Legacy promotion rejected every scheduled candidate and is recorded only as
cycle/matchup context. In particular, ep2400 scored 14/32 against the
champion and 16/32 against the worst retained baseline; ep2700 scored 13/32
and 11/32. Those outcomes neither cause nor rescue this verdict. Task 1 is
rejected by the preregistered balanced external endpoint and aggregate.

**Verdict: do not add the full-schedule entity-damage exchange reward to the
S14 stack.** The 15M result was a false positive for 45M retention. A future
revisit, if any, should be a separately preregistered short-lived curriculum
or earlier shutoff rather than a coefficient-only retry; the present data do
not justify that follow-up ahead of Tasks 2 and 3. Artifacts:
`results/reward_45m/rs1entity45/`,
`experiments/runlogs/rs1entity45_178420486524.jsonl`, and
`experiments/azuki_local_rs1entity45_178420486524/`.

### 28.7 Task 2 fresh 45M confirmation (complete; rejected)

The generated/recovered-IKZ conversion arm ran from scratch on the exact S14
stack as `rs2ikzconv45_178423488709`. It did not confirm the positive 15M
smoke. Its five external windows scored 39, 44, 48, 49, and 46 wins out of 96,
compared with S14's 49, 34, 75, 50, and 50. The aggregate was **226/480 =
47.1% versus 258/480 = 53.8%** (-32 wins). Ep1000 and ep1500 exchanged equal
10-win deficits/gains, leaving the first two windows tied at 83/192. The arm
then missed S14's strong ep2000 state by 27 wins and never recovered the
aggregate deficit.

The larger endpoint independently rejected the arm: **169/384 = 44.0%
versus 191/384 = 49.7%** (-22 wins). The failure was strongly seat-specific
at the endpoint. Drafter seat 1 scored 88/192 versus 87/192 (+1), while
drafter seat 0 scored 81/192 versus 104/192 (-23). Across the five smaller
windows, however, both seats were exactly 16 wins below S14 (104 versus 120
in seat 0 and 122 versus 138 in seat 1). This is not a globally dead seat,
but the learned policy became particularly exploitable in seat 0 at the
endpoint. All six evaluations had zero timeouts.

The element trade was the opposite of the intended Hydromancy benefit. Across
the five windows, Earth gained six wins and Fire gained one, while Lightning
lost 14 and Water lost 25. At the endpoint, Earth was 36/84 versus 34/84 (+2)
and Fire was 36/85 versus 25/85 (+11), but Lightning was 55/109 versus 73/109
(-18) and Water was 42/106 versus 59/106 (-17). The largest gate-level losses
were Surge and Hydromancy, both -14 wins; Ragefire gained nine. Task 1 showed
almost the same endpoint trade (Fire +11 with large Lightning/Water losses),
which points to a shared rush/value specialization rather than a uniquely
successful resource-conversion strategy.

The play-style shift was large, symmetric across training seats, and stable
well before the endpoint. From the trailing 100 epochs at ep1000 through
ep2930, attack rate stayed in a narrow 25.98%-26.96% band and mean training
episode length stayed at 82.05-84.48 steps. Marked conversions rose from 0.92
per game near ep1000 to 1.56 near ep2000, then plateaued at 1.53-1.59. There
is no late discovery trend hidden by the external readout.

Over the final quarter, Task 2 attacked 26.27% versus S14's 20.77%, portaled
11.26% versus 9.42%, and no-oped 14.96% versus 13.72%. Ordinary play actions
were nearly identical (30.98% versus 31.07%), while ability use fell to 6.28%
from 8.63%. Mean training games shortened to 82.86 from 110.76 steps and
surviving leader health fell to 0.174 from 0.211. The shorter phenotype also
survived outside the league: endpoint reference games averaged 141.63 steps
versus S14's 157.42.

This was not failure to find or use the shaped path. The final quarter
averaged 1.90 marked IKZ sources created and 1.51 spent per game, about 79%
conversion. At the 0.15 shaping floor this pays only about 0.0113 gross
owner-side reward per game across both players, or 0.0057 per player before
the opposite-seat zero-sum transfer. Despite the small magnitude, paying for
any marked spend appears to reinforce faster expenditure and attacks instead
of teaching when resource conversion creates a winning value line.

The gate-specific late metrics support that interpretation. Hydromancy
attacked 24.30% versus S14's 22.13%, portaled only 10.41% versus 9.66%, used
abilities 5.10% versus 6.33%, and shortened from 119.71 to 93.03 steps. Echoed
Waves shortened from 126.57 to 91.52 steps. The policy learned a broad tempo
shift, not a narrow increase in high-value Hydromancy sequencing.

Sibling-gate differentiation again failed as a quality proxy. Mean KL for
Task 2 versus S14 was 9.63e-8 versus 6.56e-8 at ep1000, 1.27e-8 versus
4.80e-8 at ep1500, 3.26e-9 versus 6.90e-7 at ep2000, 3.60e-7 versus 1.89e-7
at ep2500, 6.65e-7 versus 8.26e-8 at ep2900, and 2.67e-8 versus 5.51e-8 at
the endpoint. The near-zero ep2000 KL coincided with the largest external
deficit, but the much larger ep2900 KL did not restore external strength.

Performance remained safe. Final-quarter SPS was 1,441 versus 1,452 (-0.8%)
with no truncation regression. Total training uptime was 7h37m44s, including
the legacy serial league gates; the ep1800 gate alone cost 23.8 minutes, and
the ep2100/2400/2700 gates cost about 15-16 minutes each. These are evaluator
wall-time costs, not rollout SPS regressions, and are additional evidence for
the deferred promotion-evaluator redesign.

Legacy promotion rejected every scheduled candidate and remains diagnostic
only. The clearest calibration example was ep2400: it won 20/32 against the
incumbent and 17/32 against the weakest retained baseline, yet failed the
current Wilson thresholds. Ep2700 then cycled to 14/32 against both. Neither
result determines this verdict; the preregistered balanced external endpoint
and aggregate do.

**Verdict: do not continue or adopt this exact full-schedule conversion
reward.** Its behavior plateaued for roughly the last 30M steps without an
external recovery, so simply training it longer is not supported. This does
not establish that conversion credit can never help. A future matched salvage
test, only after Task 3 and a protocol discussion, should anneal the conversion
bonus fully to zero and leave the conflicting raw untapped-IKZ reward off.
A separate raw-off/no-conversion arm is needed to distinguish harmful generic
spend credit from the effect of removing resource-reserve credit. Do not use
the held mark/spend/expiry redistribution first: discovery was already solved,
and making the same target easier is unlikely to correct the rush bias.
Artifacts: `results/reward_45m/rs2ikzconv45/`,
`experiments/runlogs/rs2ikzconv45_178423488709.jsonl`, and
`experiments/azuki_local_rs2ikzconv45_178423488709/`.

### 28.8 Task 3 fresh 45M confirmation (complete; confirmed)

The temporary-effect realization arm ran from scratch on the exact S14 stack
as `rs3tempreal45_178426575499`. It decisively confirmed the positive 15M
smoke. Its five external windows scored 45, 46, 71, 74, and 70 wins out of 96,
compared with S14's 49, 34, 75, 50, and 50. The aggregate was **306/480 =
63.8% versus 258/480 = 53.8%** (+48 wins). It was only four wins below S14 at
each of ep1000 and S14's unusually strong ep2000 state, while gaining 12, 24,
and 20 wins at ep1500, ep2500, and ep2900.

The larger endpoint was stronger still: **259/384 = 67.4% versus 191/384 =
49.7%** (+68 wins). Both seats improved. Drafter seat 1 scored 136/192 versus
87/192 (+49), and drafter seat 0 scored 123/192 versus 104/192 (+19). Across
the five smaller windows, seat 1 was 157/240 versus 138/240 (+19), and seat 0
was 149/240 versus 120/240 (+29). All six evaluations had zero timeouts. The
arm therefore satisfies the preregistered confirmation rule by a wide margin
without relying on either seat assignment.

The gain was also broad by element and gate. Across the five windows, Earth
gained 22 wins, Fire eight, Lightning 15, and Water three. At the endpoint,
Earth was 52/84 versus 34/84 (+18), Fire 49/85 versus 25/85 (+24), Lightning
87/109 versus 73/109 (+14), and Water 71/106 versus 59/106 (+12). Every one
of the eight gates improved: the smallest gains were Hydromancy +2 and Surge
+3, while Rushfire gained 13 and Stormchain/Ragefire gained 11 each. This is
not a narrow Fire shortcut or one favorable temporary-effect package.

Task 3 did learn a faster style, but unlike Tasks 1 and 2 it retained the
valuable states needed to beat the external panel. Over the final quarter,
it attacked 23.38% versus S14's 20.77%, portaled 11.18% versus 9.42%, and
no-oped 14.53% versus 13.72%. Ordinary play rate was effectively unchanged
(30.96% versus 31.07%), while ability use fell to 7.23% from 8.63%. Mean
training games shortened to 88.74 from 110.76 steps and surviving leader
health fell to 0.179 from 0.211. Against the fixed reference decks, however,
endpoint games shortened only to 148.01 from 157.42 steps while win rate rose
17.7 percentage points.

The trajectory explains why shorter games alone were not a sufficient
diagnosis. Task 3's style moved with the league: trailing games near ep1000
were 92.0 versus S14's 98.9 steps, while ep1500 and ep2000 widened to 88.6
versus 119.1 and 87.0 versus 113.2. Despite that aggressive phase, Task 3
still scored 71/96 at ep2000, where Tasks 1 and 2 scored only 46 and 48.
Temporary-effect realization therefore taught an externally useful timing
policy, whereas broad entity damage and unconditional marked-IKZ spending
settled into exploitable rush policies.

The late gate-specific profile was not uniformly aggressive. Relative to
S14, Gate of Devotion attacked slightly less (17.79% versus 18.85%) while
portaling more (11.59% versus 9.44%) and still gained nine endpoint wins.
Hydromancy attack rate was nearly unchanged (22.39% versus 22.13%) while its
portal rate rose modestly. Fire's two gates showed the largest attack-rate
increases, consistent with their archetype, but every element improved
externally. The reward preserved strategic variation instead of forcing all
gates into one action mix.

Both realization paths remained active at the shaping floor. The final
quarter averaged 0.073 temporary-Charge realizations and 0.958 effective
temporary-buff damage per game across both seats. After applying the 0.08 and
0.025 coefficients and the 0.15 floor, that is only about 0.00447 gross
owner-side reward per game, or 0.00224 per player before the opposite-seat
zero-sum transfer. The confirmed gain is not explained by a dense reward
overwhelming terminal win/loss.

Sibling-gate KL was neither required for nor predictive of this gain. Mean KL
for Task 3 versus S14 was 1.20e-8 versus 6.56e-8 at ep1000, 9.06e-9 versus
4.80e-8 at ep1500, 1.04e-7 versus 6.90e-7 at ep2000, 3.87e-8 versus 1.89e-7
at ep2500, 1.22e-7 versus 8.26e-8 at ep2900, and 7.99e-9 versus 5.51e-8 at
the endpoint. Task 3 was usually less differentiated between sibling gates
than S14 while being much stronger externally. Draft KL remains diagnostic
telemetry only.

Performance was clean. Final-quarter SPS was 1,483 versus S14's 1,452
(+2.1%), with no truncation regression. Total training uptime was 7h27m57s,
including the serial league gates. The long ep1800 gate cost 22.8 minutes;
the ep1500/2100/2400/2700 gates each cost about 15-16 minutes. Those pauses
are evaluator overhead rather than rollout slowdown.

Legacy promotion again did not drive the decision. Task 3 promoted at ep300,
then was only 8/32 against that champion at ep900; it later promoted at
ep2700 after a series of rejections. The late promotion happens to agree with
the external endpoint, but the oscillating intermediate decisions miss the
smoothly positive external trajectory and remain an unsuitable model-quality
label.

**Verdict: add temporary-effect realization to the next S14-derived model
stack.** It clears the fixed endpoint, aggregate, seat, element, stability,
and SPS criteria. The current leading checkpoint is
`experiments/azuki_local_rs3tempreal45_178426575499/model_azuki_local_002930.pt`.
Its stack is exactly S14 plus `AZK_TEMP_CHARGE_REALIZATION_BONUS=0.08`,
`AZK_TEMP_ATTACK_REALIZATION_PER_DAMAGE=0.025`, and
`AZK_TEMP_ATTACK_REALIZATION_DAMAGE_CAP=4`; do not also enable the rejected
Task 1, Task 2, or Task 4 rewards. Artifacts:
`results/reward_45m/rs3tempreal45/`,
`experiments/runlogs/rs3tempreal45_178426575499.jsonl`, and
`experiments/azuki_local_rs3tempreal45_178426575499/`.

### 28.9 Competitive reward campaign synthesis

| Signal | 15M smoke | 45M confirmation | Decision |
| --- | ---: | ---: | --- |
| Entity-damage exchange | 147/288 vs 131/288 | 209/480 and 168/384 | Reject |
| Generated-IKZ conversion | 139/288 vs 131/288 | 226/480 and 169/384 | Reject |
| Temporary-effect realization | 139/288 vs 131/288 | **306/480 and 259/384** | **Adopt** |
| Paid-response availability | 118/288 vs 131/288 | Did not qualify | Reject |

The central reward-design result is about credit specificity, not aggression
itself. Task 4 rewarded opportunity availability and failed. Task 2 rewarded
resource expenditure regardless of what the expenditure accomplished and
failed. Task 1 rewarded a broad intermediate combat proxy and failed to
retain. Task 3 paid only when a transient opportunity was converted into
positive effective damage, excluding mere grants, innate/permanent Charge,
and overkill; that tighter causal link produced a broad external gain.

For future shaping proposals, prefer **realized, counterfactual-like credit**:
reward the useful portion of a temporary or expiring opportunity only after
its intended outcome occurs. Avoid paying for readiness, generic spending, or
broad activity counts. A realized paid-response signal that measures actual
effective damage prevented is more consistent with the evidence than Task 4's
response-availability reward. Any such proposal still needs its own smoke and
fresh confirmation rather than being stacked into the confirmed model.

The play-style evidence also answers whether the failed arms merely needed
more training. Task 2's attacks, game length, and conversion behavior had
plateaued by ep1000 and stayed there through ep2930 while external strength
did not recover; extending the same objective is not supported. Task 3's
faster play persisted too, but it retained S14's strong ep2000 state and
improved every element and gate. Faster play is therefore a phenotype to
validate, not a defect by definition. The relevant warning signs are loss of
high-value checkpoints, element/seat concentration, falling ability use and
leader health without external compensation, and a stable proxy behavior
whose fixed-panel score does not improve.

Do not launch a longer production run, the learned turn-boundary potential,
or reference-seat training automatically from this result. The next model
baseline should be the confirmed Task 3 stack/checkpoint. Before additional
training, discuss the deferred promotion redesign from section 27 so future
league management does not discard large improvements or waste substantial
wall time on a poor signal. The learned potential is now lower priority: it
should be attempted only if a specific remaining strategic gap is identified
after the confirmed Task 3 model is evaluated with the agreed broader panel.

## 29. LAST: final-stage anneal-to-zero validation (COMPLETED)

The final-stage experiment kept the adopted early curriculum and compared a
`0.15` fixed-floor control with exact-zero treatment tails from the accepted
p4870 atomic model and league state. Reaching zero only at the final update was
never considered a valid test.

The first broad schedule reached zero at p6900 and spent 900 of 2,930
continuation updates (`30.7%`) at zero. It was rejected: endpoint direct H2H
was `43.8%`, paired p4870 performance fell `8.9` points, heldout performance
fell `4.5` points, heldout Water fell `9.7` points, attacks and spells fell,
and games lengthened. Schedule integrity and SPS passed, so this was an
efficacy failure rather than an implementation failure.

The narrower retry first trained one shared floor-shaped trunk through p6900,
qualified it, then ramped treatment to exact zero at p7200. It remained at
zero through p7800 for 601 updates, or `20.51%` of the full continuation. The
restart at p7200 preserved zero, terminal labels stayed active, and treatment
retained `99.60%` of control SPS.

This later treatment ended at `50.0%` direct H2H with a `50.0%` final-three
mean. The robust late slope was slightly negative at `-0.13` points per 100
updates, while OLS was slightly positive. Endpoint parent and heldout deltas
were favorable (`+2.6` and `+3.1` points), and all registered deck and battle
mechanics safety gates passed. The result demonstrates that a mature policy
can retain strength for the final fifth of training without shaped reward; it
does not demonstrate a superior endpoint or a continuing upward trend.

Decision: **zero viable but neutral**. Keep p4870 as the selected parent and
do not make zero shaping the production default yet. The late-zero p7800
checkpoint is archived for a separate short matched stability extension at
zero. See `late-zero-ablation.md` and
`results/shaped_reward_latezero20_retry1_v1/trajectory_report.md`.

## 30. Uniform-assignment and main-draft-credit sequence

### 30.1 Stage 0: assignment compatibility (COMPLETED)

The permanent lifecycle now has an opt-in implementation in both the Python
and native environments. It samples each live seat's gate uniformly from the
eight unique gates, applies the accepted same-element seat-1 override, samples
one of the two compatible leaders uniformly, and starts the actor at main pick
1. A live seat emits exactly 50 draft rows. Fixed reference seats retain their
prescribed gate and leader. Scheduled native evaluation can independently pin
both gates and leaders without affecting ordinary training resets.

The implementation passed 27 focused lifecycle, native parity, promotion
schedule, and evaluation-control tests after rebuilding the release native
module. The uniform native/Python equivalence test produced exactly 100 total
draft actions, compatible leaders, and identical seeded decks. The evaluation
export now includes direct-Garden and alley play rates, split ability rates,
response opportunities, temporary-effect realization, generated-IKZ, and
entity-damage counters. These fields are populated only at episode export and
do not add work to the rollout hot path.

The frozen p4870 panel compared a forced leader action row with the same leader
prefilled and no row. Across 384 paired games covering all 16 gate-leader
contexts, all 384 outcomes were identical. Aggregate score was `0.5703125` in
both arms, every game completed normally, deck and battle metrics were exactly
equal, and the no-row evaluator retained `97.84%` of forced-row throughput.
This clears the registered 95% guard.

The stochastic replay probe used four histories in each of the 16 contexts.
Removing the recurrent transition changed probability calibration: mean
symmetric main-pick KL was `0.255544`, mean TV was `0.238832`, and Earth was
the most sensitive at approximately `0.50-0.53` KL. This is not a determinism
failure: the repeated-row control KL was exactly zero, mean hidden-state cosine
was `0.999014`, candidate histories replayed exactly, and forced/prefilled deck
summaries matched in every context. The learned-leader diagnostic also showed
that p4870 used both legal leaders for every gate but with material skews, such
as `19/24` versus `5/24` on Stormchain and Stonehaven.

Decision: use the direct prefilled lifecycle. A synthetic context burn-in would
preserve an otherwise unnecessary recurrent transition and is not supported by
the frozen outcome evidence. The nonzero stochastic KL becomes the registered
pretraining baseline; old 51-row KL values are not compared directly with the
new 50-row lifecycle.

Artifacts are
`results/next_ablation_v1/stage0/assignment_compatibility.json`,
`assignment_compatibility.md`, `assignment_panel.json`, and
`assignment_panel.md`. The reusable exact-context evaluator is
`uniform_context_eval.py`; its 32-game self-control smoke covered all 16
contexts, scored exactly `0.5000`, and had zero timeouts.

### 30.2 Stage 1: uniform-assignment migration (COMPLETE; ADOPTED)

The matched launcher is `run_uniform_assignment_ladder15_v1.sh`. Both arms are
hash-pinned to the accepted p4870 model, trainer state, league state, and
promotion state. The control retains learned leader selection; the candidate
uses uniform assignment and 50 policy picks. Both retain the complete accepted
temporary-realization, early-tempo, mitigation, portal-GP, PFSP, cross-gate,
and `0.15` shaping-floor recipe, with rejected credit and reward candidates
explicitly zeroed. The launcher enforces resume progression, target epoch,
lifecycle fingerprint, a sustained absolute `1235` SPS floor, and a final 95%
candidate/control median SPS gate.

The live matched namespace is `uniform_assignment_ladder15_v2` in tmux session
`uniform-assignment-v2`. The control restored `global_step=45504363`, epoch
`4870`, completed-episode progression `404`, optimizer state, and the requested
970-update restarted LR schedule before its startup marker was written. Its
authoritative live log is
`results/next_ablation_v1/stage1/uniform_assignment_ladder15_v2/control/train.live.log`;
the candidate writes the corresponding `uniform_assignment/train.live.log`.
The aggregate outer `runner.log` was not opened due
to a directory-creation race in the tmux wrapper, but this is outside the
launcher: both per-arm logs, JSONL metrics, guards, checkpoints, and sequential
arm control remain intact.

The control subsequently completed all 970 expected updates at p5840. Its 969
valid interval measurements have median SPS `1466.57`; the final-100 median is
`1442.82`. The uniform candidate then restored the same p4870 model, optimizer,
global step, and completed-episode counter, started its own 970-update cosine
schedule, reported `uniform_assignment=True` and 50 policy picks per seat, and
cleared the startup marker. Runtime source remains frozen through its endpoint.

Stage 1 decision tooling now includes symmetric mechanics for both policies in
`uniform_context_eval.py`, interventional `gate_KL_given_leader` and
`leader_KL_given_gate` in `probe_context_kl.py`, and one greedy plus 24
stochastic deck drafts for every gate-leader context in
`dump_context_decks.py`. These are evaluation-only files and were added without
changing the source loaded by either training arm.

The final three Stage 1 windows also receive causal fixed-deck evaluation.
Within each target gate-leader context, the evaluator holds policy, opponent,
seed, seat, gate, and leader fixed while replacing only the main with one
drafted for the sibling gate, sibling leader, or both. Each policy/window uses
384 paired games. The migration now requires its matched-main advantage not to
regress by five points at the endpoint or on average. These CPU-only games start
after every GPU panel and cannot perturb rollout SPS. The idle evaluation
watcher was restarted after this addition so it cannot retain the older script
inode; candidate training was not interrupted.

Review also found that the shared heldout-reference helper still left the old
leader action unspecified. That would have tested retained p4870 leader-row
behavior rather than the permanent 50-pick lifecycle. The evaluation-only
helper now has an explicit uniform-assignment mode: both arms receive the same
forced gate, sibling leaders swap between seats on the second seed, and all 32
gate-leader-seat coordinates are covered without changing the 288-game count.
The four Stage 1-4 evaluators require this assignment fingerprint. Three focused
tests, Bash syntax, and ShellCheck pass; the waiting Stage 1 evaluator was
restarted again so it loaded the corrected function definitions.

The first live heldout invocation exposed one remaining schedule-validator
edge: the candidate seat had its forced leader, while the fixed-reference seat
still carried legacy sentinel `-1`. The engine correctly rejected a partially
forced leader pair before applying the reference-deck override. The helper now
assigns a deterministic compatible placeholder leader to that scheduled seat;
the engine still replaces it with the reference deck's actual gate and leader,
so candidate contexts and reference gameplay are unchanged. Three focused
tests and a real 16-game CUDA smoke pass with zero timeouts. The evaluator was
restarted idempotently and retained the three already-complete p5000 uniform
panels.

Both matched p4870-to-p5840 arms are now complete. The uniform-assignment arm
emitted all 970 expected metric rows, finished with zero timeout truncation,
and recorded median/tail-100 interval SPS of `1446.95`/`1443.29` under the
sealed report's steady-window definition. The matched control median was
`1466.57`, giving a candidate/control ratio of `0.9866`
and clearing both the 95% relative and 1,235 absolute guards. The endpoint
model, trainer state, metadata, league state, and promotion state were written
before `LADDER_TRAIN_DONE`; the automatic evaluator then became the sole GPU
process and began the four registered windows.

At p5000, the direct uniform-context panel was effectively tied: the candidate
scored `0.4922` against matched control, and both candidate and control scored
`0.5130` against the p4870 parent (384 games per panel, zero timeouts). After
the placeholder-leader repair, the heldout-reference panels completed all 288
games with zero timeouts and the required 8 leaders / 16 gate-leader contexts.
The candidate scored `0.6354` versus control's `0.6458`, a `-1.04 pp` delta
inside the plan's approximately one-point neutral band. This is an early
window only; adoption remains gated on the three late windows, per-context
losses, mechanics, deck concentration, interventional KL, and causal hybrids.

The first legacy p5000 context-deck dump then exposed a separate evaluator
defect. `EpisodeRunner` warmed the policy before loading checkpoint weights,
while `_load_model_weights` left the derived text-feature table cached from
random initialization. This invalidates legacy `EpisodeRunner` deck dumps,
context KL, and fixed-deck hybrids, but not the completed native direct or
heldout panels, whose policies load before their first forward. The candidate
dump and interrupted control log are preserved under
`evaluation/invalid_stale_text_cache/` and excluded from reports.

The shared loader now invalidates the derived cache after `load_state_dict` and
has a regression test. A full 100-decision uniform draft then produced exact
native/legacy agreement for raw observations, canonical tensors, encoded
features, legal logits, and argmax actions (`max_logit_delta=0`). Replacement
diagnostics `native-batched-draft-v1` and
`native-batched-context-replay-v1` passed byte-identical CPU and CUDA repeats.
At registered sizes, 384 deck drafts took about 22 seconds and 6,400 KL rows
took about 15 seconds, replacing the approximately 31-minute-per-checkpoint
serial dump while preserving the policy/engine contract.

The cache defect did not affect either training arm. `train.py` constructs the
policy and loads resume weights before the first policy forward; CUDA-graph
warmup happens only after trainer construction. The failing legacy evaluator
did the opposite explicitly. The 15M instability is therefore training
evidence, not an artifact of stale random text features.

The first late window at p5200 is materially negative. Uniform assignment
scored `0.4427` versus matched control, `0.4036` versus the p4870 parent, while
control scored `0.4948` versus parent. The resulting parent-panel delta is
`-9.11 pp`. Heldout reference scores were `0.5938` candidate and `0.6319`
control (`-3.82 pp`). All panels completed with zero timeouts. The direct loss
was broad across Lightning, Fire, and Water while Earth remained positive, so
it cannot be dismissed as one adverse seat or context. This is one late window;
p5500 and p5840 determine whether it is an adaptation dip or a sustained
migration regression.

The next window at p5500 shows recovery rather than a monotonic decline. The
candidate scored `0.5078` directly against control. Candidate and control
scored `0.4349` and `0.4401` against p4870, respectively, leaving only a
`-0.52 pp` paired parent-panel delta. On the heldout-reference panel the
candidate scored `0.6076` versus control's `0.5903`, a `+1.74 pp` delta. All
five p5500 panels again completed without timeouts. Fire, Lightning, and Water
all recovered to at least `0.479` in the direct element split while Earth was
`0.458`; the p5200 loss therefore currently looks transient, but p5840 and the
three registered causal-hybrid windows still control the Stage 1 decision.

The p5840 endpoint regressed again. Uniform assignment scored `0.4583`
directly against control and `0.3984` against p4870; control scored `0.4635`
against p4870, for a `-6.51 pp` parent-panel delta. Heldout scores were
`0.5521` candidate and `0.6111` control (`-5.90 pp`). Lightning (`0.3646`) and
Earth (`0.4271`) carried most of the direct loss, and candidate seat 0 scored
`0.4010` versus seat 1's `0.5156`; this is not one isolated gate offsetting
otherwise broad improvement. All endpoint panels completed without timeouts.

Across p5200, p5500, and p5840, the direct mean is `0.4696`, the mean paired
parent-panel delta is `-5.38 pp`, and the mean heldout delta is `-2.66 pp`.
The first two values narrowly miss their registered `0.47` and `-5 pp` safety
floors. This is not a structural deck-diversity collapse: at p5840 candidate
and control greedy unique-card means were `18.50` and `18.25`, stochastic
unique-card means were `27.72` and `27.39`, and four-copy slot shares were
`0.555` and `0.563`. Both context KL quantities remain approximately
`1e-6`. With the causal panels also clean, the preregistered outcome is
**ambiguous/unstable and requires a fresh matched 45M adaptation test**, not
adoption of p5840. The fresh run restarts both arms from the exact
p4870 atomic parent with one 2,930-update cosine schedule; it does not continue
from the exhausted 970-update schedules.

All three causal windows are safe but uninformative. At p5200 the uniform
candidate's matched-main advantage over the sibling-both swap was exactly
`0.00 pp`, versus control's `-1.04 pp`, for a `+1.04 pp` relative delta. At
p5500 candidate was again exactly `0.00 pp`, versus control's `+2.08 pp`, for
a `-2.08 pp` relative delta. At p5840 both candidate and control were exactly
`0.00 pp`, so the relative delta was also zero. Gate-only and leader-only swaps
tell the same near-tie story. The three-window candidate-minus-control mean is
`-0.35 pp`, safely above the `-5 pp` causal-regression floor, but no window
shows that uniform coverage alone taught context-specific deck construction.

The sealed report is
`results/next_ablation_v1/stage1/uniform_assignment_ladder15_v2/ladder_report.json`
with the human-readable companion `ladder_report.md`. Its verdict is
**diagnose before adoption**: throughput, timeout integrity, and causal safety
pass, while external-strength safety narrowly fails. No p5840 artifact is an
accepted parent. Per the registered branch condition, Stage 1 proceeds with a
fresh matched p4870-to-p7800 comparison: both arms receive independent
2,930-update schedules and freshly cloned parent league state. The four full
windows are p5000, p5800, p6800, and p7800; the last three also receive causal
hybrids. Stage 2 remains isolated until that confirmation selects a parent.

Before the confirmation produced evaluation outcomes, the Stage 1 report was
aligned with the plan's existing no-collapse clause. Adoption now also requires
endpoint portal use to remain at least `90%` of control; attack, spell,
direct-Garden play, and Garden/leader-ability deltas to stay within the same
tolerances used by Stage 2; stochastic unique-card count not to fall by more
than three; and four-copy slot share not to rise by more than `0.10`. Replaying
the sealed 15M evidence passes this mechanics/deck gate and leaves its
external-safety-based verdict unchanged.

The confirmation launched at `2026-07-22T08:48:03-07:00` in tmux session
`uniform_assignment_confirm45_v1`. The control arm restored p4870 at
`global_step=45504363`, completed episode `404`, all 13 frozen opponents, and
the optimizer before restarting a 2,930-update cosine schedule. Its startup
contract reports `total_epochs=7800`, `remaining_epochs=2930`, learned-leader
lifecycle with 51 policy picks per seat, and has written
`STARTUP_INVARIANTS_OK`. The first ten measured intervals had median SPS about
`1728`; this is only a startup health observation, not the final throughput
estimate. At the first sustained sentinel, p5103, the 233 logged post-parent
epochs had median SPS `1426.38`, tail-30 median SPS `1441.46`, and maximum
timeout-truncation rate `0`. This clears both registered performance guards but
remains an interim control-arm measurement. The tmux chain starts the uniform
arm only after control seals, then runs the registered
p5000/p5800/p6800/p7800 evaluation automatically.

The p5200 atomic checkpoint also sealed successfully. Through p5204, all 334
post-parent rows had median SPS `1438.97`, tail-30 median SPS `1443.36`, normal
game-over terminal rate `1.0`, and timeout-truncation rate `0`. The p5200 model,
trainer, and metadata hashes are respectively `e284069e428f8ec8`,
`c7638cd88d588d81`, and `4440857c16d5bde8` (prefixes shown).

The longer p5500 control sentinel is also healthy. Through p5501, 631 logged
post-parent rows give median SPS `1438.97` and tail-100 median SPS `1431.11`;
maximum timeout-truncation remains `0` and minimum game-over terminal rate
remains `1.0`. Numbered p5500 model, trainer, and metadata artifacts all exist.
This remains control-arm integrity evidence rather than an efficacy result.

At p5800, 930 post-parent rows give median SPS `1432.34` and tail-100 median
SPS `1471.67`; timeout-truncation is still `0` and the minimum game-over
terminal rate is still `1.0`. The atomic p5800 snapshot passed model, trainer,
metadata, league, and promotion checksums. Its model/trainer/metadata hash
prefixes are `2c28fc826ce783df`, `f7ed6765c94d84c5`, and `4ce1350be446fb18`.
Metadata records update `5800`, completed episode `481`, and
`draft_uniform_assignment=false`; the copied league points to p000061 created
at epoch 5800. Resume resolution selects `trainer_state_005800.pt` beside the
numbered model.

The p6800 control window remains stable. Through p6801, 1,931 post-parent rows
give median SPS `1428.69` and tail-100 median SPS `1422.13`, with timeout rate
`0` and minimum game-over terminal rate `1.0`. All five atomic snapshot hashes
pass; the model/trainer/metadata prefixes are `a7f2f4ab2aed09e6`,
`455bbde6311cda15`, and `881eaa4a70b26c96`. Metadata records completed episode
`563`, while the league copy points to p000071 created exactly at epoch 6800.
Resume resolution selects the adjacent `trainer_state_006800.pt`.

The control then sealed exactly 2,930 continuation rows at p7800. Its final
median SPS is `1427.52`, tail-100 median SPS is `1371.21`, maximum timeout rate
is `0`, and minimum game-over terminal rate is `1.0`. The endpoint model,
trainer, metadata, league, and promotion checksums all pass. Model/trainer/
metadata hash prefixes are `a2e6db9c9d2bc4aa`, `7a3844eaeeeaef43`, and
`48538de1173e7183`; metadata records completed episode `645`, and the copied
league points to p000081 created exactly at epoch 7800.

The sequential launcher then restarted the uniform-assignment candidate from
the original p4870 parent, not the control endpoint. Startup restored
`global_step=45504363`, epoch `4870`, completed episode `404`, the optimizer,
and all 13 opponents; it restarted the same 2,930-update LR schedule. Runtime
source hashes still match the campaign record. The lifecycle banner reports
`uniform_assignment=True` and exactly 50 policy picks per seat. In the first
20 logged candidate updates, every observed completed draft has `picks=50`
and `main_count=50`; per-element gate and assigned-leader frequencies are
identical, timeout rate is `0`, and game-over terminal rate is `1.0`.
Startup-amortized SPS is not used for the sustained throughput decision.

The first sustained candidate sentinel is healthy. Through p5002, 132
post-parent rows give median SPS `1434.96` and tail-30 median SPS `1460.38`;
all observed drafts have exactly 50 picks, maximum timeout rate is `0`, and
minimum game-over terminal rate is `1.0`. The candidate median at this point is
about `100.5%` of the sealed control's full-run median, but the registered
relative SPS gate will use both complete 2,930-update arms. Numbered p5000
model, metadata, and trainer-state artifacts all exist.

At candidate p5200, all 330 post-parent rows remain valid. Median SPS is
`1444.53`, tail-50 median SPS is `1452.05`, every observed draft has exactly
50 picks, timeout rate is `0`, and minimum game-over terminal rate is `1.0`.
The first candidate atomic snapshot passes all five checksums; model/trainer/
metadata hash prefixes are `4d4a8bd6dd07608e`, `2e0a3e2f0ffb8ae0`, and
`ed43adb54cdb6695`. Metadata records completed episode `432` and uniform
assignment enabled, while the copied league points to p000055 created exactly
at epoch 5200. Resume resolution selects `trainer_state_005200.pt`.

The candidate remains stable through p5501. Across 631 post-parent rows its
median SPS is `1447.02` and tail-100 median SPS is `1460.58`, or `101.37%` of
the sealed control's full-run median. Every observed draft still has exactly
50 picks, timeout rate remains `0`, and minimum game-over terminal rate remains
`1.0`. Uniform assignment therefore shows no sustained throughput penalty in
this matched run so far.

At candidate p5800, 933 post-parent rows give median SPS `1446.01` and
tail-100 median SPS `1445.55`, or `101.30%` of the sealed control median.
Draft count remains exactly 50, timeout rate remains `0`, and minimum game-over
terminal rate remains `1.0`. All five atomic snapshot checksums pass; the
model/trainer/metadata hash prefixes are `9a8db85dd1431088`,
`a5af14c64c5a69ed`, and `22f4000a0253ce39`. Metadata records completed episode
`482` and uniform assignment enabled, while the league copy points to p000061
created exactly at epoch 5800. Resume resolution selects the numbered trainer
state beside the model.

The longer candidate p6303 sentinel is also stable. Across 1,433 post-parent
rows, median SPS is `1442.98` and tail-100 median SPS is `1448.83`, or
`101.08%` of the complete control median. Draft count remains exactly 50,
timeout rate remains `0`, and minimum game-over terminal rate remains `1.0`.

At candidate p6800, 1,933 post-parent rows give median SPS `1446.48` and
tail-100 median SPS `1455.01`, or `101.33%` of the complete control median.
Every observed draft remains exactly 50 picks, timeout rate is `0`, and minimum
game-over terminal rate is `1.0`. The atomic model, trainer, metadata, league,
and promotion hashes all pass; model/trainer/metadata prefixes are
`517cc28528af7b82`, `e03a9fe40bbaafa3`, and `612e64972c172a1d`. Metadata
records completed episode `565` and uniform assignment enabled, while the
league copy points to p000071 created exactly at epoch 6800. Resume resolution
selects the numbered trainer state beside the model.

The final pre-endpoint sentinel at candidate p7303 remains stable. Across 2,433
post-parent rows, median SPS is `1444.60` and tail-100 median SPS is `1467.04`,
or `101.20%` of the complete control median. Draft count remains exactly 50,
timeout rate remains `0`, and minimum game-over terminal rate remains `1.0`.

The candidate then sealed exactly 2,930 continuation rows at p7800. Final
median SPS is `1443.45`, tail-100 median SPS is `1443.17`, and the matched
candidate/control ratio is `1.01116`, passing the 95% relative and 1,235
absolute guards. Every observed draft has exactly 50 picks; maximum timeout
rate is `0` and minimum game-over terminal rate is `1.0`. The endpoint model,
trainer, metadata, league, and promotion hashes all pass. Model/trainer/
metadata hash prefixes are `6f9df1ff11045a6e`, `2a2b9299ed5cb170`, and
`284ed70ce0ace032`. Metadata records completed episode `648`, uniform
assignment enabled, and the copied league points to p000081 created exactly at
epoch 7800. Resume resolution selects the numbered endpoint trainer state.

Both matched 45M arms are therefore training-complete and throughput-safe. The
chained evaluation started with the registered 384-game p5000 uniform-versus-
control panel; efficacy and adoption remain undecided until all four windows,
heldout panels, deck/KL diagnostics, and last-three causal hybrids complete.

The complete p5000 evaluation window is safe and near-neutral. Uniform versus
control scores `0.4792` with paired 80% lower bound `0.4688`; its parent-panel
delta is `-2.60 pp` and heldout delta is `-0.35 pp`, with no timeout. Candidate
versus control stochastic unique-card means are `27.74` versus `28.09`, and
four-copy slot shares are `0.553` versus `0.541`, so there is no deck collapse.
Candidate `gate_KL_given_leader` and `leader_KL_given_gate` are only
`6.5e-7` and `8.5e-7`, respectively, similar to control and with determinism
KL exactly zero. This early window shows neither material regression nor
meaningful context-conditioned drafting; later windows and causal hybrids are
still required.

The complete p5800 window remains inside the registered safety envelope.
Uniform versus control scores `0.4844` with paired 80% lower bound `0.4661`.
The candidate and control score `0.4635` and `0.5026` against the p4870
parent, a `-3.91 pp` delta; heldout-reference scores are `0.5486` and
`0.5764`, a `-2.78 pp` delta. All five panels have zero timeouts. Candidate
versus control stochastic unique-card means are `28.36` versus `27.77`, and
four-copy slot shares are `0.534` versus `0.554`, again ruling out a deck
concentration collapse. Candidate `gate_KL_given_leader` and
`leader_KL_given_gate` remain approximately zero at `6.6e-7` and `9.0e-7`,
with determinism KL exactly zero. Its heldout attack, spell, portal, and
ability rates are `0.2241`, `0.0081`, `0.1152`, and `0.0863`, versus control's
`0.2255`, `0.0078`, `0.1134`, and `0.0722`; no mechanics sentinel has failed.
The p6800 and p7800 windows plus all three causal hybrids remain outstanding.

The complete p6800 window is also safe. Uniform assignment scores `0.5078`
against control with paired 80% lower bound `0.4896`. Candidate and control
score `0.4427` and `0.4635` against the p4870 parent, a `-2.08 pp` delta;
heldout-reference scores are `0.5556` and `0.5833`, a `-2.78 pp` delta. Every
panel again has zero timeouts. Candidate and control stochastic unique-card
means are effectively identical at `27.82` and `27.80`, as are four-copy slot
shares at `0.554` and `0.555`. Candidate heldout portal use is slightly higher
(`0.1152` versus `0.1095`), attacks are also higher (`0.2298` versus `0.2203`),
and spell use is within `0.14 pp`; no mechanics sentinel fails. Both candidate
context KL quantities remain approximately zero (`6.2e-7` gate given leader,
`8.8e-7` leader given gate) with exact determinism. The p7800 endpoint and the
three causal-hybrid windows remain outstanding.

The first p7800 heldout control pass produced one censored game at the
registered 600-step cap (`1/288`, `0.35%`), while the candidate completed all
288 games. The evaluator stopped before diagnostics rather than treating the
censoring as valid evidence. A symmetric full-panel protocol extension to
1,200 steps changed no candidate game and no other control game after removing
wall-time fields. The one censored control game ended normally at step `604`,
only four steps past the original cap. Original max-600 JSON/log pairs are
retained as `heldout.max600.*`; the authoritative p7800 pair records
`max_steps=1200` and zero timeouts. The shared evaluator now uses this explicit
1,200-step heldout cap so a restart reproduces the repaired protocol.

With the censoring resolved, the p7800 endpoint passes every noncausal Stage 1
gate. Uniform assignment scores `0.4740` directly against control with paired
80% lower bound `0.4531`. Candidate and control score `0.4688` and `0.4583`
against the p4870 parent, a `+1.04 pp` delta; heldout-reference scores are
`0.5833` and `0.5799`, a `+0.35 pp` delta. Candidate versus control stochastic
unique-card means are `27.35` versus `27.52`, and four-copy slot shares are
`0.569` versus `0.560`. On the paired parent panels, candidate portal rate is
`104.9%` of control; attack, spell, direct-Garden, and Garden/leader-ability
deltas are `+0.51`, `+0.10`, `-0.56`, and `+2.30 pp`, all inside their
registered thresholds. Candidate context KL remains approximately zero
(`6.1e-7` and `9.3e-7`) with exact determinism. Across p5800/p6800/p7800,
direct score averages `0.4887`, parent-panel delta averages `-1.65 pp`, and
heldout delta averages `-1.74 pp`.

All six causal panels then completed: 2,304 games across the three late
checkpoints and two arms, with no truncations. Gate-only, leader-only, and
combined sibling swaps have exactly `0.00 pp` matched advantage for both
policies at every window, so candidate-minus-control causal delta is also
exactly zero. This clears the registered causal safety floor but confirms that
uniform assignment alone did not create within-element deck specialization.
The sealed report verdict is **adopt uniform assignment**. Throughput,
integrity, external strength, mechanics, and causal safety all pass, and p7800
uniform assignment becomes the atomic Stage 2 parent.

Because the active league state changes whenever a 100-update checkpoint joins
the recent pool, model files from intermediate windows are not atomic parents
by themselves. `capture_uniform_assignment_state_snapshots_v1.sh` now watches
both sequential arms and atomically preserves model, trainer, metadata, league,
and promotion state at the registered later windows. Its first p5200 control
snapshot passed all five checksum checks and records current candidate p000055
at created epoch 5200. Snapshots retain the numbered checkpoint and trainer
basenames; a direct `_resolve_resume_artifacts` check finds
`trainer_state_005200.pt`, preventing a model-only resume. Future large files
are hard-linked with a copy fallback, while mutable league state is copied. The
watcher is read-only with respect to training.

### 30.3 Stage 2: full-episode main-draft credit (COMPLETE; NO CREDIT PATH ADOPTED)

The Stage 2 implementation was developed in `/tmp/azuki-stage2-prototype` so
the running Stage 1 processes continued using their hash-recorded trainer
source. Before the Stage 1 decision, the isolated prototype passed 49 focused
tests and syntax compilation without replacing the live trainer.

After Stage 1 adoption, the isolated implementation and tests were applied to
the live tree byte-for-byte. The resume fingerprint now includes all seven
`AZK_DRAFT_EPISODE_CREDIT_*` controls. Compilation, Bash syntax, ShellCheck,
and 61 focused league-training/resume tests pass. The Stage 2 parent manifest
locks uniform p7800 at completed episode `648`: model hash
`6f9df1ff11045a6e`, trainer hash `2a2b9299ed5cb170`, league hash
`b87ba9b6d1e176f9`, and promotion hash `9a1557ee90dbaa85` (prefixes shown),
plus the accepted Stage 1 decision hash.

The first live smoke, `full_episode_credit_smoke_v1`, passed every integrity
check but correctly failed performance. Its last-24-update median was
`1221.96` SPS versus control's `1458.30`, a ratio of `0.8379`; this missed both
the 95% relative gate and the absolute `1235` floor. It nevertheless retained
120,887 CPU records, labeled 101,600 rows from 2,032 exact terminal episodes,
trained 36,350 examples across every quartile, produced nonzero gradients, and
had zero truncations or incomplete drafts. Importance means stayed within
`0.9994`-`1.0004` and clip fraction remained zero.

Profiling separated `10.61s` of delayed training and `6.03s` of record capture
from avoidable per-minibatch accelerator synchronizations used only for mask
telemetry and Python branching. The actor normalization now computes masked
mean, unbiased standard deviation, and row counts entirely on-device, with one
telemetry synchronization after the minibatch loop. This does not change
actions, labels, sampling, masks, losses, fixed batch shape, or update cadence.
A focused empty/nonempty statistics test raises the live total to 62 passing
tests. A fresh matched `full_episode_credit_smoke_v2` is running; failure still
routes to the registered frozen-predictor fallback rather than relaxing SPS.

Smoke v2 retained full integrity and improved absolute throughput, but still
failed the relative gate. Candidate median SPS rose to `1327.90` and cleared
the absolute floor, while its matched control reached `1534.99`; the ratio was
only `0.8651`. It labeled 101,000 rows from 2,020 complete episodes, trained
37,150 examples at all positions, had zero incomplete/truncated records,
nonzero gradients, importance means within `0.9998`-`1.0003`, and zero
clipping. The optimization therefore repaired a real implementation cost but
did not make retained-row training performance-safe. Per the registered
branch, no 15M ladder starts and the preferred path is rejected on throughput.
Stage 2 now pivots to the frozen prefix-outcome predictor fallback; the 95%
threshold is unchanged.

The fallback dataset contains 3,072 paired policy-prefix and randomized-prefix
complete trajectories: 156,672 exact prefix rows across four policy generations,
two opponent lineages, all 16 uniform gate-leader contexts, both seats, and 142
observed card ids. Random prefixes changed 686 of 1,536 paired decks and 157
outcomes. The original argmax-only dataset exposed only 40 cards and was rejected
as a coverage failure. The first combined model had seed-heldout discrimination
but an anti-predictive final-prefix increment, so it was not deployed.

Predictor v4 adds matched counterfactual difference loss to exact-outcome BCE and
keeps both variants of a coordinate in the same calibration split. Its selected
artifact is
`results/next_ablation_v1/stage2/prefix_outcome_model_v4/frozen_prefix_outcome_v4.npz`
with SHA-256
`0b1fb15627e935be5bba3c04c3f2caeb82084b065b27b3a61b95c79f2f891fa8`.
It has 4,569 parameters and NumPy/Torch parity error below `9e-8`. Whole-seed
heldout AUC is `0.6107` and final-versus-empty incremental AUC is `0.5241`;
heldout-generation values are `0.6048` and `0.5502`. The opponent-lineage split
is weak (`0.4955` outcome AUC and `0.4851` incremental AUC), so this is a modest
fallback candidate, not evidence that the credit problem is solved.

The frozen NumPy runtime uses incremental card-embedding sums. Each live-policy
seat receives `coef * (Q(prefix_k) - Q(prefix_{k-1}))` on an exact main pick.
At true terminal or truncation it receives the residual
`coef * (Q(empty) - Q(final))`; therefore the added return sums to exactly zero
over every episode and neither duplicates nor replaces the true battle outcome.
The channel is independent of shaped-reward annealing. Runtime telemetry covers
all draft quartiles, completed and truncated episodes, synchronization, prediction
spread, inference time, and maximum telescope error. The disabled arm performs no
extra tensor transfer or rollout allocation. Sixty-nine focused tests pass,
including artifact hashing, unseen catalog cards, terminal and truncation closure,
live packed-observation decoding, and resume fingerprinting.

An 8-update integration-only comparison, `prefix_outcome_smoke8_v0`, passed. Its
steady interval median was `1,949.01` SPS versus `1,736.95` control (`1.1221x`).
All four quartiles emitted nonzero deltas, 28,800 pick deltas required `0.0838s`
of predictor inference, and there were no synchronization errors. Eight updates
do not span a complete fresh draft-plus-battle lifecycle, so terminal residual and
telescope integrity remained requirements of the registered 48-update
`prefix_outcome_smoke_v1`. That gate passed: candidate median SPS was `1,625.35`
versus `1,444.71` control (`1.1250x`), with 2,025 completed episodes, 120,429
prefix deltas, zero truncations or synchronization faults, all four quartiles,
and exact `0.0` maximum telescope error. Predictor inference consumed `0.708s`
across the whole candidate arm.

The fresh matched `prefix_outcome_ladder15_v1` is complete from the original
p7800 parent. Candidate median SPS was `1431.71` versus control `1440.38`, a
`0.9940` ratio; both performance guards pass. Across 970 updates it emitted
2,329,644 pick deltas and 46,151 terminal residuals, covered all quartiles, used
`16.36s` of predictor inference, and had zero truncations, synchronization
faults, or telescope error.

The sealed verdict is `stop_after_neutral_15m`. Over p8100/p8400/p8770, direct
score averaged `0.5087`, parent-panel delta averaged `+0.69 pp`, and heldout
delta averaged `+3.01 pp`. The endpoint was `0.4792` direct, `-0.26 pp` against
the parent panel, and exactly tied on heldout. Integrity, throughput, external
safety, and every mechanics sentinel pass, but the endpoint strength rule does
not. More decisively, candidate and control both had exactly `0.000` matched
advantage for sibling-gate, sibling-leader, and combined main-deck swaps at all
three causal windows. Stochastic unique-card count was also lower for the
candidate at every measured window, by `0.53` to `1.89` cards. The early p8100
strength peak therefore does not establish useful phase-aware draft credit.
There is no 45M confirmation, and uniform p7800 remains the selected parent.

The rejected retained-row path was distinct from the earlier five-row additive
estimator. For
each live-policy seat it retains every legal main pick from 1 through 50,
including the packed observation, actual action, behavior log probability,
and actual pre-decision LSTM hidden and cell state. Inactive alternating-seat
observations are rejected by the native legal-action count. Complete records
receive only an exact terminal target; truncations and incomplete drafts are
dropped, while true terminal draws receive target `0.5`.

Ordinary PPO keeps all of its existing recurrent minibatches, value learning,
and battle rows. Main-draft rows are removed only from the ordinary actor and
entropy terms, then receive the delayed Monte Carlo/PPO actor loss with
`outcome - stop_gradient(win_probability)` advantage. The delayed pass calls
the eager differentiable forward explicitly rather than the rollout CUDA-graph
replay. A test with one independently indexed actor parameter per pick proves
nonzero gradients at all 50 positions.

To bound memory and avoid a short-game completion bias, complete drafts enter a
deterministic hash-priority reservoir of 80 episodes. One fixed 4,000-row batch
is trained every four updates; padding is masked, and the reservoir spans the
whole four-update window. Telemetry includes capture/completion/drop counts,
quartile advantage and calibration statistics, draw rate, record age,
importance ratios, clipping, gradient norm, auxiliary wall/GPU time, standard
actor versus masked-draft rows, and gate/leader outcome coverage. The feature
allocates no rollout mask when disabled.

`run_full_episode_credit_smoke_v1.sh` is the registered 48-update matched
integrity/performance gate. It consumes a hash-locked parent generated by
`write_atomic_parent_manifest.py`, repeats the terminal impulse test under the
uniform no-leader-row lifecycle, and requires at least 95% of control SPS plus
the absolute `1,235` floor before any 15M efficacy ladder. The impulse utility
now understands the 100-action uniform draft and expects only the ten requested
main-pick rows, not obsolete learned-leader rows.

The matched 15M launcher and evaluation path are now prepared as
`run_full_episode_credit_ladder15_v1.sh` and
`run_full_episode_credit_eval_v1.sh`. The ladder requires the smoke marker,
restores both arms from the same atomic parent, checks all 970 expected update
rows, and refuses efficacy evaluation if either the absolute or relative SPS
gate fails. Four matched checkpoints receive direct sibling, parent-panel,
heldout-reference, 16-context deck, and bidirectional context-KL evaluation.
The last three checkpoints additionally receive paired fixed-deck causal
tests: the target gate and leader stay fixed while the main deck is replaced
with one drafted for the sibling gate, sibling leader, or both.

Before the Stage 2 outcomes were inspected, its heldout evaluator was aligned
with the already-authoritative Stage 1 censoring repair: reference panels use
`max_steps=1200`, not the obsolete 600-step cap that stopped one normal game at
step 604. The same prospective correction is present in the conditional Stage
3 and Stage 4 evaluators. Uniform direct panels remain at 600 because they have
not exhibited censoring.

The causal runner now assigns complete four-arm groups to 12 physical-core
shards using a deterministic SHA-256 ordering. This preserves every global
game index and paired comparison while spreading gate/reference contexts more
evenly than the prior contiguous schedule. A coverage test proves all 384
tasks appear exactly once, every four-arm group stays on one shard, every
shard receives 32 games, and repeated scheduling is identical. Stage 2-4
evaluators derive expected line counts from their configured shard count.

`full_episode_credit_ladder_report.py` preregisters separate integrity,
throughput, external-strength, mechanics, greater-than-noise strength, and
sustained causal-fit decisions. A complete synthetic fixture exercised all
inputs and the advance branch successfully; the launcher and report also pass
ShellCheck, Python compilation, and diff validation. These files are
evaluation-only until the Stage 1 training boundary is reached.

A positive 15M decision gates
`run_full_episode_credit_confirm45_v1.sh`. Confirmation is a fresh 2,930-update
control/candidate comparison from the same accepted Stage 1 parent, not a
continuation of either 15M arm, so early trajectory, opponent-pool formation,
and the restarted LR schedule remain matched. The corresponding evaluation
wrapper reuses the registered four-window, causal-hybrid, mechanics, and SPS
panels, then emits `accept_full_episode_credit` only under final confirmation
semantics. Both wrappers pass Bash syntax and ShellCheck validation.

### 30.4 Stage 3: random main-card prefix (COMPLETE; CONDITIONALLY SKIPPED)

The conditional Stage 3 implementation is isolated in
`/tmp/azuki-stage3-prototype` and is not present in the Stage 1 or Stage 2 live
source. It passes 52 focused tests. The registered distribution is exactly
`n in {0,1,2,4}` with probabilities `{0.5,0.25,0.125,0.125}`, giving one
forced card per live-policy deck on average.

Prefix lengths and legal candidate indices are selected by stable 64-bit
episode/seat/pick hashes. Only learner and latest-policy rows can be
overridden; frozen league opponents and reference seats are untouched. The
implementation inspects only active rows in the first four main picks, so the
disabled path is constant-time and the enabled overhead does not scan the
remaining 46 picks.

The policy still processes each forced observation and carries its recurrent
state forward, but the stored action is the actual uniformly forced legal
action. Its policy log probability is recomputed for rollout bookkeeping.
Full-episode records carry a per-pick actor-valid mask: forced rows receive no
ordinary or delayed actor/entropy loss, while the terminal baseline may still
learn from their prefix states. A parameter-indexed test shows zero actor
gradient on four forced rows and nonzero gradient on every unforced row.
Runtime telemetry reports sampled prefix lengths, forced rows, actor versus
baseline example counts, and the existing full-credit diagnostics. Deployment
and any ladder remain conditional on Stage 2 clearing integrity, SPS, and
efficacy gates.

The conditional smoke and 15M launchers are now prepared as
`run_random_main_prefix_smoke_v1.sh` and
`run_random_main_prefix_ladder15_v1.sh`. Both arms retain exact full-episode
credit; only `random_main_prefix` receives the registered distribution. The
launchers require all four prefix buckets, a sampled mean between `0.7` and
`1.3`, nonzero forced rows, baseline training on those rows, no delayed actor
examples on them, no prefix telemetry in control, and the shared 95%/1,235 SPS
guards. Both pass Bash syntax, ShellCheck, and diff validation.

`run_random_main_prefix_eval_v1.sh` disables prefixes during evaluation. It
uses the same four uniform-context, parent, heldout, deck, interventional-KL,
and last-three causal-hybrid windows as Stage 2. At the endpoint it additionally
runs 96 matched, uniform-assignment self-play traces per arm across 12 CPU
shards. `analyze_card_funnels.py` separates opening-hand exposure from cards
first observed after the opening snapshot, then reports per-card draft,
observation, legal-play, selected-play, and realized-use funnels. Realized use
means a spell or weapon was played, or a card later attacked, defended,
portaled, activated, or drove an ability follow-up. This adds no rollout-hot-path
work.

`random_main_prefix_ladder_report.py` treats exposure as diagnostic rather than
efficacy. Advancement still requires external and mechanics safety plus either
a greater-than-noise strength gain or a sustained causal context-fit gain in
at least two of the last three windows including the endpoint. A complete
synthetic fixture exercised the acceptance branch; report compilation and the
evaluation scripts pass static validation.

Stage 3 is complete as a registered conditional skip. The accepted Stage 2
prerequisite was not met: retained rows were too slow and frozen redistribution
was safe-neutral with zero causal deck-fit effect. Running forced-prefix policy
training without a useful credit path would test exposure alone, contrary to the
plan. The offline randomized prefixes already used to broaden the frozen
predictor dataset are data collection, not evidence for enabling this mechanism.

### 30.5 Stage 4: late shaping removal (COMPLETE; CONDITIONALLY SKIPPED)

The new Stage 4 path will restore the selected Stage 2/3 parent's reward
fingerprint rather than reconstructing it from the old p4870 recipe. It will
override only the trainer-side shaping schedule. `fixed_floor` remains at an
effective `0.15`; `late_zero` ramps its trainer multiplier from one to zero and
then logs exactly 194 zero-multiplier updates, the final 20% of a 970-update
matched continuation. Exact terminal draft credit and the selected prefix
setting are held identical between arms and are never annealed.

The schedule gate will validate every logged multiplier against the absolute
update formula, require effective shaping to be exactly zero throughout the
registered tail, and require nonzero terminal-credit training within that same
tail. Multiple exact-zero checkpoints will receive direct, parent, heldout,
Water, Garden, deck, and causal-context panels. Approximately one point of
heldout movement remains neutral; a neutral or unstable zero tail keeps the
`0.15` floor.

The 48-update smoke, matched 970-update launcher, six-window evaluation, and
decision report are prepared as `run_late_zero_smoke_v1.sh`,
`run_late_zero_ladder15_v1.sh`, `run_late_zero_eval_v1.sh`, and
`late_zero_ladder_report.py`. Static validation passes, and synthetic clear-win
and neutral fixtures exercise both adoption and fixed-floor retention branches.

Stage 4 is complete as a registered conditional skip. Its prerequisite was exact
true-terminal credit at all 50 main picks; the retained-row implementation failed
the SPS gate, while the frozen fallback only redistributes a zero-sum learned
potential and is not that terminal-credit channel. The selected recipe therefore
keeps the accepted `0.15` shaping floor. No additional late-zero ladder is run.

### 30.6 Stage 5: uniform-context promotion (DEPLOYED; SHADOW-QUALIFIED)

The existing promotion-v2 implementation already supplies the protected
quality archive, four-policy panel, anchor-relative comparison, opponent
quorum and floors, paired bootstrap uncertainty, external reference yardstick,
shadow mode, immutable records, and separately timed native evaluation. Its
remaining mismatch with the proposed lifecycle is that production
`paired-v1` schedules leave both leaders unspecified.

An isolated `/tmp/azuki-stage5-prototype` first supplied the missing
uniform-context contract without changing the then-running Stage 1 source. When
uniform assignment is active, the screen expands from eight gate blocks to 16
gate-leader blocks per opponent. Each context uses the same seed, gate, and
leader with the candidate in both seats. The existing eight cross-gate blocks
then assign compatible leaders so the candidate covers every gate-leader
context once during confirmation. This produces 192 panel games for four
opponents: 128 exact context-paired screen games and 64 cross-gate confirmation
games. The two-seed reference schedule remains 288 games; the second seed swaps
sibling leaders between seats, so every one of the 16 candidate contexts is
measured from both seats without adding games.

Evaluation records now retain actual candidate and opponent leaders, and panel
summaries/comparisons expose leader- and gate-leader-conditioned scores and
deltas. Legacy learned-leader schedules remain behavior-compatible and keep
`paired-v1`; uniform schedules use a distinct version so cache keys cannot
reuse old anchor results. The manager selects the new schedule only when
`env.draft_uniform_assignment` is true.

Full fixed-yardstick evaluations also append a schedule-hash-matched
three-observation external-strength window. It reports score mean/min/max,
anchor-relative delta, paired lower bounds, and score slope per 100 updates.
This is production-selection telemetry only: it is written to immutable run
metadata and audit history but is deliberately not another archive-admission
veto.

Before deployment, the isolated prototype passed 34 promotion, archive, and reference-evaluation tests,
plus syntax compilation and a real-catalog schedule check. The real catalog
contains exactly 16 screen contexts; the generated four-opponent schedule has
192 games, and a one-seed nine-deck reference schedule remains 144 games.
`qualify_uniform_promotion.py` was prepared to run two independent native
replays from cloned league/archive state, compare every semantic game field,
verify all context and seat coordinates, enforce the three/eight-minute timing
budgets, and prove that shadow mode leaves the quality archive, production
anchor, panel, compatibility pointer, and active PPO pool unchanged. Live
deployment and this GPU qualification were deferred until the training
interventions select their final parent.

The qualification clone now relaxes only its anchored screen thresholds so it
always exercises screen, confirmation, and reference scheduling even when the
selected model would legitimately fail the production screen. Production
thresholds remain unchanged. Its before/after shadow snapshot also includes the
current-candidate pointer and next policy index, preventing those compatibility
fields from changing unnoticed during replay.

The Stage 1 heldout validator failure exposed the same latent sentinel issue in
this isolated Stage 5 reference schedule: its candidate seat had an assigned
leader while the fixed-reference seat still used `-1`. The prototype now gives
the reference seat the other compatible sibling leader as a validation
placeholder; native reset still replaces it with the fixed deck's real leader.
All focused tests pass after the repair. A real-catalog check covers 144
games, all 16 candidate contexts, and verifies that both scheduled placeholder
leaders match their gate element.

A later cache audit found and repaired a second predeployment issue. Immutable
promotion artifacts already serialized the new candidate/opponent leader ids,
but the cache loader did not restore those fields. The first uniform evaluation
would work, while a later candidate could fail to pair against an anchor panel
loaded from cache. The isolated prototype loader now round-trips both leader
ids, with a regression test proving semantic record equality after reload. The
isolated promotion/archive/reference suite passes 34 tests after this fix. The
deployment merge retained the repository's newer reference-evaluator tests
rather than replacing that file with the prototype's older snapshot.

After Stages 2-4 closed, the prototype was merged into the live promotion path
while retaining the repository's newer reference-evaluator override hook. The
promotion evaluation cap is `1,200` steps, matching the censoring fix used by the
heldout experiment panels. The merged implementation passes 35 focused
promotion/archive/native-reference tests and six additional native-control,
promotion-ablation, and draft-reference tests. A real-catalog audit produces
exactly 192 panel games, 128 screen games, 64 confirmation games, and 288
reference games with all assigned leaders legal.

The first qualification preflight exposed an invalid test-only
`min_candidate_epoch_gap=0`; it stopped before model construction or GPU games.
The qualification helper was corrected to the minimum legal value of one, and
the failed preflight was retained separately for audit. No production threshold
was changed.

Two fresh native CUDA replays then passed the registered qualification. Both used
uniform p7800 with model SHA-256
`6f9df1ff11045a6ef20c32a1bd84df8139df9ded63a49f578ed433b19f75a39d`.
Their semantic game records were identical, every game completed, all 16 screen
gate-leader contexts and 32 context-seat coordinates were present, and the
two-seed reference schedule covered 32 context-seat coordinates. The full gates
took `372.24s` and `365.13s`, and their screens took `80.97s` and `78.92s`,
passing the `480s` and `180s` budgets. Both recorded the external-strength
window and had zero timeout rate.

The rule-level decision was admission on the standard route, with panel score
`0.7604`, anchor-relative delta `+0.2760`, and paired lower bound `0.2448`.
Reference score was `0.5312`, reference delta `-0.0521`, and paired lower bound
`-0.0799`; this telemetry is intentionally not a separate archive veto. Live
admission remained zero because shadow mode was enabled. Both replays proved
that the quality archive, production anchor, panel, compatibility/current-candidate
pointers, next policy index, and active PPO pool were unchanged.

Stage 5 therefore qualifies the new evaluator and promotion bookkeeping for
continued shadow observation. It does not establish that this checkpoint is a
better training parent, nor does it turn promotion into the primary efficacy
label. The sealed evidence is in
`results/next_ablation_v1/stage5/uniform_promotion_qualification_v1/summary.json`
and `report.md`.

## 31. Final decision for the next-ablation sequence

The sequence is complete. Uniform assignment is the sole adopted training
intervention: uniformly assign one of eight gates and one of its two legal sibling
leaders, remove the leader action from PPO, and train only the 50-card main draft
plus battle. The selected atomic parent is uniform p7800 at completed episode 648,
with model, trainer, league, and promotion state locked by
`results/next_ablation_v1/stage2/parent_manifest.json`.

The adopted 45M uniform arm preserved rollout performance: median SPS was
`1443.45` versus `1427.52` for control (`1.0112x`), and its final-100-update
median was `1443.17`. Promotion evaluation is timed separately and adds no work
to the rollout hot path between gates. The non-adopted frozen-credit arm also met
the SPS guard (`1431.71` versus `1440.38`, or `0.9940x`), while both retained-row
credit attempts were rejected before a ladder at `0.8379x` and `0.8651x`.

Keep the accepted reward stack and mature `0.15` shaping floor. Do not enable the
retained-row terminal-credit path, frozen prefix-outcome redistribution, random
main-card prefixes, or late-zero schedule. Retained-row credit failed the SPS
guard; the frozen fallback was performance-safe but safe-neutral, showed exactly
zero causal context-fit advantage, and reduced stochastic unique-card count at
every measured window. Those results do not justify a 45M confirmation.

Keep the redesigned promotion path in shadow mode while it accumulates multiple
real candidate windows. Its paired gate-leader-seat evidence is now suitable for
league management, but external panels, mechanics, causal deck-fit probes, and
checkpoint-window trends remain the model-quality decision set. A production-scale
sample budget and distributed-training plan remain explicitly deferred to a
separate document, as requested.

### 31.1 Closure validation

The final live tree passes 108 focused tests covering promotion schedules and
caches, archive behavior, native reference and evaluation controls, frozen
draft-credit runtime, league training utilities, and resume fingerprints. Python
compilation, Bash syntax, ShellCheck for the Stage 0-4 wrappers, and `git diff
--check` pass. Both ladder completion markers and the Stage 5 qualification report
are present. The model, trainer, metadata, league, promotion, and source-decision
files all reproduce the SHA-256 values in the atomic parent manifest. No C engine
file changed, so an engine/native-module rebuild was not required.

## 32. Retained-row and random-prefix continuation (2026-07-23; COMPLETE)

The user reopened retained-row credit after reviewing its cost and explicitly
prioritized learned play quality over the measured SPS penalty. Both the `0.95`
relative threshold and absolute `1,235` threshold are diagnostic for these local
efficacy experiments. Candidate/control SPS and both raw threshold results remain
reported. Label, gradient, recurrent-state, draft-completion, timeout, checkpoint,
and model-process integrity remain mandatory.

Campaign `retained_rows_ladder15_userwaiver_v1` is a fresh 970-update matched
control/retained-row comparison from the hash-locked uniform p7800 atomic parent.
It retains the selected reward recipe and changes only exact full-episode credit.
Four windows will receive direct, parent, heldout, 16-context deck, context-KL,
mechanics, and last-three causal hybrid evaluation.

The matched control completed all 970 updates with median SPS `1407.07`,
tail-100 median `1398.71`, exactly 50 main picks, and zero timeouts. The first
candidate leg reached `p8330`, where its legacy rolling absolute guard fired.
The process shut down cleanly, but `p8330` had not been checkpointed. The atomic
recovery boundary is therefore `p8300`; the logged but uncheckpointed
`p8301-p8330` rows remain retained as audit evidence and are excluded from the
final stitched ladder.

Recovery `retained_rows_ladder15_userwaiver_v1_full_episode_credit_resume8300`
restored the exact `p8300` model, optimizer, cosine scheduler, completed-episode
counter, and cloned league state. The first resumed update reproduced the expected
learning rate exactly (`1.4223046562112473e-05`). The process was then stopped at
`p8404` after a label-integrity audit invalidated both candidate legs.

Every retained target was `0.5`. In the atomic original range `p7801-p8300`, all
23,685 completed records were counted as draws; all 123 auxiliary-update rows had
target mean `0.5` and Q1 draw fraction `1.0`. The resumed `p8301-p8404` segment
repeated this for all 4,721 completed records and all 24 auxiliary-update rows.
Native games had decisive outcomes. The exact-credit caller had passed the
per-agent terminal component decoded from aggregate native `info`, which is zero,
instead of the native reward on true terminal rows. Therefore the SPS measurements
remain valid implementation-cost evidence, but neither candidate checkpoint is
efficacy evidence. The campaign is marked
`INVALID_TERMINAL_LABELS.md`; `p8300` and resumed `p8400` must not be evaluated,
promoted, or resumed.

The corrected caller masks native rewards to true terminal rows. Telemetry now
records decisive, win, loss, and draw episodes separately; training fails after
four completed-label epochs with no decisive outcome. Smoke and ladder gates
require both win and loss labels. The replacement smoke and matched 970-update
campaign restarted from the hash-locked uniform `p7800` parent.

Random main-card prefixing was also reopened. Both Stage 3 arms used exact
retained-row credit; only the candidate forced the
registered `n in {0,1,2,4}` prefix with probabilities
`{0.5,0.25,0.125,0.125}`. Unlike the earlier conditional rule, a strategically
neutral but integrity-safe retained-row result does not cancel Stage 3. Prefix
training stopped only for integrity failure or model-process collapse. Relative and
absolute SPS remain visible adoption and optimization evidence, but are not
efficacy-run kill switches. This directly tests the proposed interaction between
terminal draft credit and broader card exposure.

### 32.1 Corrected retained-row result

The corrected `full_episode_credit_smoke_labelfix_v1` passed the decisive-label
gate. It recorded 2,019 completed decisive episodes, 1,019 wins, 1,000 losses,
zero draws, and nonzero gradients in all four draft quartiles. Candidate median
SPS was `1,354.28` versus `1,430.85` control, or `0.9465x`; this remained
diagnostic under the user-approved policy.

The fresh 970-update
`full_episode_credit_ladder15_labelfix_v1` then completed from the atomic p7800
parent. The candidate labeled 46,866 decisive episodes containing 24,113 wins
and 22,753 losses, with zero draws, incomplete episodes, or timeouts. All
retained quartiles trained. Median SPS was `1,310.17` versus `1,454.69`
control, or `0.9007x`; the candidate still cleared the absolute `1,235` floor.
This confirms that the approximately ten-percent retained-row cost is
engineering debt rather than a model-process blocker.

The registered strategy verdict is `stop_after_neutral_15m`. Direct scores at
p7900, p8100, p8400, and p8770 were `53.65%`, `48.44%`, `53.91%`, and
`52.86%`. The last-three parent-panel delta averaged `+2.08` points and heldout
delta averaged `+4.40` points, but the endpoint direct score narrowly missed
the registered 53% threshold and none of the three causal deck-swap windows
showed matched portal/leader value.

The lack of causal deck fit is material. At p8770, retained credit increased
greedy unique-card count from `17.5` to `30.5`, stochastic unique count from
`26.31` to `32.67`, and reduced four-copy slot share from `60.15%` to
`37.42%`. It also reduced Water spell slots by `1.34`. Battle mechanics and
external safety remained inside their registered limits, but exact outcome
credit changed global draft preferences toward substantially more singleton
variety without producing portal- or leader-conditioned decks. The corrected
implementation is valid and may remain available for future credit research;
this p8770 checkpoint is not adopted.

Evidence is in
`results/next_ablation_v1/stage2/full_episode_credit_ladder15_labelfix_v1/ladder_report.json`
and `ladder_report.md`.

### 32.2 Random main-prefix result

Stage 3 isolates prefixing with exact retained-row credit enabled in both arms.
The candidate forces only legal deterministic random cards, carries those
actions through recurrent state and the outcome baseline, and masks forced rows
from actor and entropy gradients. Evaluation disables prefixing completely.

The smoke passed with 2,025 candidate decisive episodes, every prefix bucket
exercised, 2,604 forced rows, 756 delayed forced rows correctly actor-masked,
and no lifecycle error. Candidate SPS was `1,332.75` versus `1,255.65`
control. The full 970-update ladder also passed every integrity check. It
recorded 47,004 decisive candidate episodes, 47,581 prefix episodes, 47,712
forced rows, mean prefix length `0.9958`, and zero draws, incompletes, or
timeouts. Candidate median SPS was `1,321.49` versus `1,362.56` control, or
`0.9699x`; both raw SPS gates pass.

The registered verdict is `diagnose_mechanics_or_deck_regression`, with no
45-minute continuation:

| Epoch | Prefix vs control | Parent-panel delta | Heldout delta |
| ---: | ---: | ---: | ---: |
| 7900 | 49.74% | +0.26 pp | +4.86 pp |
| 8100 | 53.13% | +1.82 pp | -2.08 pp |
| 8400 | 53.13% | -0.78 pp | -1.04 pp |
| 8770 | 51.82% | +0.26 pp | +2.78 pp |

The last-three direct mean is `52.69%`, but the parent-panel delta averages only
`+0.43` points and heldout delta averages `-0.12` points. Strength improvement
therefore fails. At p8100, the only leader-specific greedy deck change performed
`3.13` points worse than the sibling leader's deck. At p8400 and p8770, every
sibling portal/leader context inside an element again uses the exact same greedy
main deck, so causal context fit also fails.

Prefixing consistently reduces singleton-heavy construction: endpoint
stochastic unique cards fall by `1.37` and four-copy slot share rises by `6.08`
points. However, the shift is global rather than contextual. Every element
moves Monk Staff of Warding from one copy to four, weapon slots rise in all four
greedy element decks, and the p8770 prefix decks use no Lightning or Fire spells
just like control. Normal evaluation already drafts all 175 cards in both arms.
Selected-card coverage rises by three and realized-effect coverage by two, but
there are zero useful rare-card lines under the registered rule.

The endpoint raw garden/leader ability rate falls `2.12` points, narrowly outside
the `-2.0` point safety limit, while weapon rate rises `1.32` points and episodes
shorten by `6.47` actions. Opportunity-normalized evidence shows that legal
leader-ability selection is effectively unchanged (`17.03%` versus `17.04%`),
comparable Garden selection changes only `-0.16` points, and Water spell
selection improves. The raw failure is therefore a shorter, more weapon-heavy
style rather than refusal to use a legal ability. It still does not establish
efficacy or justify overriding the registered stop.

Do not promote p8770 or repeat this fixed prefix distribution for 45 minutes.
Keep prefixing experiment-gated. If the hypothesis is reopened, the next
isolated test should anneal prefix probability to zero before the final 40% of a
new 15-minute run, leaving that entire tail prefix-free; compare it with both
no-prefix and fixed-prefix controls. This tests whether early exposure retains
copy consistency without preserving the late global weapon bias.

The detailed human-readable comparison is
`results/next_ablation_v1/stage3/random_main_prefix_ladder15_v1/strategy_comparison.md`;
the registered report is `ladder_report.json`.

### 32.3 Closure validation

Both corrected ladders contain `LADDER_TRAIN_DONE` and `EVALUATION_DONE`
markers. The focused retained-credit, prefix-lifecycle, and resume suite passes
72 tests. Python compilation passes for the changed training and report modules;
ShellCheck passes for all six Stage 2/3 smoke, ladder, and evaluation launchers;
`git diff --check` passes. This continuation changed no C engine source, so it
did not require an engine or native-module rebuild.
