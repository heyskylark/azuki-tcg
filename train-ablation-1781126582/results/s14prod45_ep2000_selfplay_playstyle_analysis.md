# S14prod45 ep2000 — Self-Play Game Logs: Strategy & Playstyle Analysis

**Question this answers:** does the ep2000 checkpoint *play* strategically — stringing cards together into short-term lines and long-term plans — or does it just take plausible-looking random actions?

**Verdict up front: the play is unambiguously strategic.** The model executes the exact synergy loops its drafts implied (weapon recursion through Surge, spell recovery through Echoed Waves, sacrifice conversion through Devotion), spends its mana with 92% efficiency in the early game, enables Charge before attacking with fresh bodies, uses its response windows with element-appropriate cards, and wins 88% of its games with a face attack after a HP race it deliberately paces. It also has clear learned *habits and blind spots* (always mulligans, never attacks the Alley, barely uses the lightning leaders' abilities) — the interesting failures are listed at the end.

---

## Method / provenance

- **Data:** 300 full games (draft + battle), both seats played by `model_azuki_local_002000.pt` under eval sampling (temperature 1.0, no smoothing). Gates randomized per seat by the env's own draft sampler with sibling-matchup oversampling disabled; leader and 50-card deck drafted live by the policy each game.
- **Logging:** every battle decision is recorded with actor, phase, derived turn, raw 4-head action, a semantic decode (card codes for plays / attacks / portals / equips / ability sources / selection-zone picks), leader HPs, IKZ, hand, both boards, and terminal rewards.
- **Dataset:** `results/selfplay_ep2000/games_shard{0..3}.jsonl` (300 games, ~15 MB), aggregates in `aggregate_stats.json` + `pattern_stats.json` in the same directory.
- **Tooling (new, in `train-ablation-1781126582/`):** `play_selfplay_games.py` (runner/logger), `analyze_selfplay_games.py` (aggregates), `query_selfplay_patterns.py` (pattern queries), `render_game.py` (turn-by-turn narrative of any logged game).

Reproduce / inspect:
```bash
PYTHONPATH=build/python/src:python/src:train-ablation-1781126582 .venv/bin/python \
  train-ablation-1781126582/play_selfplay_games.py --checkpoint CKPT --games 75 --out games.jsonl
python3 train-ablation-1781126582/render_game.py results/selfplay_ep2000/games_shard*.jsonl --game 125
```

Headline game stats: mean 15.6 turns / 120 decision steps per game; seat balance 153–147 (no first-player artifact); 0 draws/truncations.

---

## 1. Macro outcomes: the gate you draw decides a lot

Win rates by element (both seats pooled; recall each element plays an identical 50-card archetype, so differences within an element isolate the *gate ability* and how well the model uses it):

| Element | Games | Win rate | Gate split |
|---|---|---|---|
| LIGHTNING | 155 | **.677** | Surge .711 (97g) / Stormchain .621 (58g) |
| WATER | 168 | .524 | Hydromancy .588 (102g) / Echoed Waves .424 (66g) |
| FIRE | 132 | .402 | Rushfire .479 (73g) / Ragefire .305 (59g) |
| EARTH | 145 | .372 | Devotion .375 (72g) / Stonehaven .370 (73g) |

- Lightning is the apex predator of this meta: it beats Earth 83% and Fire 80% head-to-head (Water puts up the best resistance at ~46%).
- The within-element gate gaps are a clean measurement of gate-ability value in practice: Hydromancy's IKZ ramp is worth ~16 points over Echoed Waves' spell recursion on the same deck; Rushfire is worth ~17 points over Ragefire. Ragefire (.305) is the worst gate in the game as played — its rider (buff an entity that took damage this turn) almost never lines up (used after only 12% of its portals).

## 2. The universal engine: Alley as a staging zone, mana as a metronome

- **Play-to-alley → portal out the same turn happened 2,412 times — ~8 times per game.** The dominant motor across all elements: cheap entity lands in the Alley (triggering its on-play tutor/loot from safety), then immediately portals into the Garden for the gate rider. The Alley is not a parking lot; it's a conveyor belt.
- **Mana discipline is near-perfect early:** in turns 1–4, 92% of turn-passes happen with **zero** unspent IKZ (74% in turns 5–10, 37% late — where floating mana for response spells is often correct anyway).
- **The game plan is a paced face race.** Mean first attack: turn 2.7. Attack targets: 100% face in the early third; board attacks only appear mid (3%) and late (11%) game — the model trades only when blockers actually block. Winner/loser HP curves separate at turn 4–5 and the gap widens monotonically (turn 10: 14.1 vs 11.2; turn 16: 9.8 vs 7.0); winners stabilize around 9–10 HP while the loser slides to lethal. 265/300 wins end with a face attack, 26 more with targeted burn ("select effect target" lethal), 5 with a gate rider as the killing blow.
- **It always mulligans.** 600 mulligan decisions, 600 shuffles, 0 keeps. Whatever hand it's dealt, it shuffles — a degenerate learned reflex (keep is a legal NOOP) worth its own probe.

## 3. Gate-specific execution — the riders are used, and used differently

Rider usage measured as ability-window actions attributable to the own gate immediately after each portal:

| Gate | Portals | Rider used | Rate | What it does with it |
|---|---|---|---|---|
| Stonehaven (E) | 477 | 334 | **70%** | grants Defender nearly every portal — the wall plan |
| Echoed Waves (W) | 391 | 224 | 57% | recovers Chilling Water (61), Healing Flutter (36), Red Bean (31), Tidal Insight (24)… |
| Surge (L) | 508 | 182 | 36% | replays weapons from discard: BJ Dagger 59, L. Shuriken 49, Hidden Dagger 47… |
| Devotion (E) | 408 | 124 | 30% | executes the sacrifice→damage conversion |
| Rushfire (F) | 286 | ~80 | 28% | free-plays entities with Charge (BJ Recruit, Fanatic Kindler, Penny…) |
| Ragefire (F) | 233 | 29 | 12% | +attack rider almost never lines up |
| Stormchain (L) | 338 | 11 | 3% | re-equips are nearly never taken (portal value only) |
| Hydromancy (W) | 603 | implicit | — | rider is an automatic IKZ untap; most-portaled gate in the game |

Two rendered examples (`render_game.py`, abridged):

**The Surge loop, executed every turn** (game 125: Surge/Piko-of-a-Thousand-Blades beats Stonehaven, 20 turns). Turns 4, 6, and 8 are literally the same engine crank:
```
P1: play Elder Hoshin -> alley            (tutors Black Jade Brawler)
P1: PORTAL Elder Hoshin alley->garden
P1: select_to_equip Black Jade Dagger     [resolving Surge]   <- same dagger, from discard,
P1: confirm_ability                        [resolving BJD]        every single turn
P1: attack MY_LEADER -> OPP_LEADER
```
The *same Black Jade Dagger* is recycled out of the discard pile onto the leader turn after turn — the weapon-recursion engine the deck analysis predicted from card text, observed as an actual play pattern. Weapons overall go to the leader 405 times vs 198 to entities, and leaders attacked face 803 times — "weaponized leader" is the lightning win condition in practice.

**Devotion's sacrifice conversion** (game 250, Earth mirror, 27 turns): the winner runs `PORTAL Pip → confirm → select_cost_target → select_effect_target [resolving Gate of Devotion]` three separate times — portal, feed a body to the gate, throw its health at an opponent entity.

## 4. Short-term card stringing (the model knows what enables what)

- **Charge awareness is precise.** 148 same-turn portal→attack events, and the attacker distribution is a Charge roll-call: Rooftop Hunter 80 (native Charge), Black Jade Brawler 21 (native Charge), Reckless Tinkerer, Indra… Meanwhile **The Red Bean was cast in 26 of the turns where a freshly-arrived entity attacked** — the model plays the Charge granter, then attacks with the granted body. Fresh entities without a Charge source almost never attempt to attack.
- **Response windows are used with the right cards.** 1,900+ response-window decisions. The response leaders: leader abilities 298 (mostly Shao's −1 attack, a Response-tagged ability, plus Bobu's shield), Water Orb 47, Aquatic Veil 43, Lotus of Paradise 37 (untapping IKZ mid-combat), Wrong Step 33 (destroying the attacker), Sundering Strike 31, Commune with Water 30 (bouncing the attacker), Bubble Adept sacrifice 27, Lightning Orb 27, Oathstone 25, Raizan's Riposte 11 (instant-speed weapon from discard). Each element leans on exactly the reactive suite its deck drafted.
- **Defender declarations are chosen, not sprayed:** Penny (the 0/1 whose only job is Defender) is the most-declared blocker at 114, followed by Foamback Crab 26 and Gus 25 (who bounces himself when attacked — a free intercept). Only 5.4% of the 7,051 attacks get intercepted though — in a face-race meta with Defender-poor boards, most damage is simply accepted (or the race is preferred).
- **Leader abilities are an element-defining habit:** Earth uses its leader ability in 145/145 games (Bobu 648 total activations, Goro 430), Water 162/168, Fire 130/132 — but **Lightning only 37/155** (Raizan 15 uses, Piko 37). See blind spots below.

## 5. Long-term arcs

- **Ramp discipline into finishers:** expensive spells are cast on schedule — Good Enough Replica mean play turn 10.1, Power of Friendship 10.9, Forging Tricks 53 casts, Wrath of Sinder 31, Fire Orb 23 — the cheap engine cards all sit at mean play turns 6–8, the payoff cards 9–12. The curve the drafts implied is the curve the games actually follow.
- **The portal economy is the tempo backbone:** 10.8 portals per game (both seats), mean portal turn 8–11 per gate. Hydromancy portals the most (603) because each portal is a mana refund; Stonehaven portals late (mean turn 11) because its value is defensive.
- **Winners pull ahead on HP by turn 4–5 and never give it back** — consistent with a tempo game decided by early race positioning rather than by late topdecks.

## 6. Per-card usage highlights

Workhorses (most-played): Healing Flutter 220, Red Bean 216, Gus 199, Sleight of Hand 197, Penny 180 — the neutral staple core from the draft analysis is also the in-play core. The vanilla 1-drops are the primary attackers (Rei: 376 attacks, 335 at face; Frida 270; Black Jade Courier 267) — cheap bodies portal in, then peck the leader every turn for the rest of the game.

Biggest win-rate lift when played vs. merely drafted (correlational — cards playable when you're already winning will show lift, so read as "associated with winning lines", min 20 games): Cinderwake Ritualist +.26, Wrong Step +.18, Black Jade Brawler +.18, Ruby +.14, Koyama Farm Caretaker +.14 (the IKZ-body — ramp correlates with winning). Biggest negative lifts: Mizuryuu's Torrent −.19, Lightning Kanabo −.17, Serene Fist Misaki −.17, Mirage Frog −.13 — the expensive water tempo spells fire mostly from losing positions (or losing positions force them).

## 7. Blind spots and habits (the "still random-ish" parts)

1. **The Alley is never attacked.** 0 of 7,051 attacks target an alley slot. The lightning deck drafts a whole Riftwalk package (Riven Flashborne, Stormcaller Tenkichi, Stormglass Daggers/Katana) whose *only* differentiated text is alley attacking — in play those cards made 31 attacks, all at the leader. The action space supports alley targets (enumerated and validated in the engine), but two forces make it rare: opposing alleys are usually empty (entities portal out the same turn they land), and the policy has plainly never valued the option. Draft-play mismatch worth a targeted probe.
2. **Lightning wins despite ignoring its leaders.** Raizan's Charge-grant used 15 times in 155 games; Piko's +attack pump 37. The decks win on the weapon-recursion race alone. Either the abilities are genuinely weak (Raizan's requires a weapon on an *entity*, but the model puts weapons on the *leader* 2:1) or there's free win-rate being left on the table.
3. **Always-mulligan reflex** (600/600) — the keep/shuffle head has collapsed to one action.
4. **Stormchain's rider is dead weight in practice** (3% usage) — the model treats it as a plain portal gate, which is consistent with Stormchain's 9-point win-rate deficit vs Surge on the same 50 cards.
5. **Blocking is near-maximal, and correctly so** (resolved by the counterfactual block probe — see `selfplay_ep2000/block_probe_findings.md`). Raw interception is 5.4% of all attacks, and a first log pass suggested the model declined ~half its block opportunities — but that used a wrong availability heuristic (any untapped garden entity). The engine requires the **Defender keyword** — natural or Stonehaven-granted — on an untapped entity vs. a non-Infiltrate attacker (`action_validation.c:978`). Replaying 18 sampled "declined" windows against the actual action mask found **zero** had a legal block: the model wasn't declining anything. Combined with 108 blocked windows in the same pool, it blocks essentially whenever blocking is legal. Counterfactual rollouts at 18 points where it blocked put the block arm at 58.3% win vs 51.7% for forced pass (+6.7 ± ~5 pts; 5 points clearly favored blocking, 2 the opposite, 11 neutral) — the blocks are mildly EV-positive, not reflexive. The scarcity of interception is a deck/rules property (few Defender-keyword cards, and the race taps everything), which also explains why Stonehaven's Defender-grant is the most-used gate rider (70%). The open question is upstream: whether *creating* more block opportunities (drafting more natural Defenders, holding bodies untapped, Stonehaven timing) would beat the pure race.

## 8. Caveats

- Self-play only: both seats share one policy, so "win rate when played" conflates card effect with game-state selection; matchup numbers describe this checkpoint's meta, not an external ladder.
- Turn indices are derived from MAIN-phase ownership changes (the engine's internal turn counter isn't exposed in observations); response-window steps are attributed to the turn they interrupt.
- Rider-usage rates come from ability-window steps whose source card is the actor's own gate immediately after a portal; riders that resolve with no decision (Hydromancy's untap) are invisible in action logs and counted as implicit.
- Semantic decodes are validated by inspection (plays decode to hand cards of the right type, attacks to occupied slots), but selection-zone picks occasionally show as `sel[i]?` when the zone shrank mid-resolution; raw actions are always preserved alongside.
