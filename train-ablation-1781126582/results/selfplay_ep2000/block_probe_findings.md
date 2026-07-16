# Counterfactual Block Probe — ep2000 (findings)

**Question:** the 300-game self-play logs showed only 5.4% of attacks intercepted, and a
first analysis suggested the model declined ~53% of "available" blocks. Is defending a
blind spot?

**Answer: no. The model blocks essentially whenever blocking is legal, and its blocks are
mildly EV-positive. The apparent passivity was a measurement artifact plus a rules
constraint.**

## Method

`probe_block_counterfactual.py` (extract → run → report):

1. From the 300 logged games, sample response-window decision points after a face attack
   where the defender had ≥1 untapped garden entity: 18 where the model blocked, 18 where
   it "declined" (one point max per game; pools were 108 blocked / 123 declined).
2. **Exact replay** to the branch point: draft picks force-reconstructed from the logged
   decks (stored in pick order), battle prefix force-replayed from logged actions. This is
   required — re-sampling from the seed is *not* reproducible across processes (CPU
   float nondeterminism under different thread counts/load flips `torch.multinomial`
   picks; verified empirically). Env transitions are deterministic given seed + applied
   actions, so the branch state is exact; verified via actor/combat-context checks.
3. At the branch: force `DECLARE_DEFENDER` (block arm) or `NOOP` (pass arm) from the
   actual legal-action mask, then roll out to terminal 10× per arm with per-rollout
   sampling seeds. 558 rollouts total.

Data: `block_probe/arm{0..5}.jsonl`, points in `block_probe_points.json`.

## Results

1. **All 18 "declined" points had no legal `DECLARE_DEFENDER` in the mask.** The engine
   requires the Defender keyword (natural — Penny, Foamback Crab, Raimaru, Sloth
   Scarecrow, … — or granted by Stonehaven's rider) on an untapped entity, and the
   attacker must lack Infiltrate (`src/validation/action_validation.c:978`). The
   original "availability" heuristic (any untapped garden entity) was wrong. At those
   points the model was doing other legitimate response things (leader ability, response
   spells). Extrapolating 0/18 to the 123-window declined pool: virtually all real block
   opportunities in the logs were taken (~108/~108).
2. **Forced-block vs forced-pass at the 18 points where the model blocked:**
   E[defender wins | block] = **0.583** vs E[win | pass] = **0.517** (Δ = **+6.7 pts**,
   n = 18 points × 10 rollouts/arm; rough SE ≈ 5 pts). Per point: 5 clearly favored
   blocking (Δ > +5 pts), 2 clearly favored passing, 11 ~neutral. On the 7 clearly-signed
   points the model's actual choice matched the better arm 5/7.

## Interpretation

- Block-decision quality is **not** a training gap: the policy blocks when legal and the
  blocks are (weakly) positive-EV. Most blocks are low-stakes (11/18 neutral), consistent
  with a race meta where chip damage matters less than tempo.
- Interception is rare because **Defender-keyword bodies are rare and the race taps
  everything** — a deck-construction/rules fact, not a play-skill fact. It also explains
  Stonehaven's 70% rider usage (Defender grants are the only way to manufacture blocks)
  and Penny's status as the most-declared blocker (114).
- The genuinely open question moved upstream: would *creating* more block opportunities
  (more natural Defenders drafted, holding attackers untapped, Stonehaven grant timing)
  outperform the all-in race? That is a deck-policy/board-position question, not a
  response-window question, and would need a different probe (e.g., counterfactual at
  attack declarations: force "hold this attacker home" vs "attack", or draft-level
  ablations forcing extra Defender density).

## Caveats

- n = 18 branch points for the EV estimate; the +6.7 pt delta is directional, not tight.
- Rollout continuations use the same policy for both seats (self-play EV under the
  current policy), which is the right notion for "should *this* policy have blocked" but
  not for exploitability questions.
- The declined-pool extrapolation (0/18 → ~0/123) is a sample; a full mask-logged re-run
  of the 300 games would nail the exact legal-block counts (the logger now also records a
  `D` flag on board snapshots for future runs).
