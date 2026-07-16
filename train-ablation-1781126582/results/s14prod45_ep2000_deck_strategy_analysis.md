# S14prod45 ep2000 — Drafted-Deck Strategy & Synergy Analysis

**Source decks:** `results/s14prod45_ep2000_decks.json` (checkpoint `model_azuki_local_002000.pt`, the campaign-best 76.5% checkpoint), produced by `dump_gate_decks.py`: per gate, one greedy draft (`argmax_deck`, temperature 1e-6) plus 24 stochastic drafts aggregated as mean copies (`sampled`, top-30 shown).
**Card text source:** effect text from the web-service seed migrations (`packages/backend-core/drizzle/0001_seed-cards.sql`, `0016_card-batch.sql`, with cost fixes from `0017`/`0019`); stats/gate-points cross-checked against `scripts/azuki-card-defs.jsonl`.

Draft rules relevant here (from `python/src/deck_building.py`): pick pool at every step = all cards of the gate's element **plus all NORMAL cards**, up to `MAX_MAIN_COPIES = 4` per card, 50-card main deck, leader chosen by the policy from the element's two leaders.

---

## Headline findings

1. **The policy builds element decks, not gate decks.** The argmax deck is *byte-identical* between the two gates of each element (Surge ≡ Stormchain, Hydromancy ≡ Echoed Waves, Rushfire ≡ Ragefire, Devotion ≡ Stonehaven), and the sampled aggregates differ only by noise. Composition is conditioned on the leader/element pool, not on gate identity — consistent with the gate-conditioning weakness tracked all campaign. The mitigating observation: each element deck contains enablers for *both* of its sibling gates (details per section below), so gate-blindness at draft time is cheap; the gate-specific adaptation can happen at play time.

2. **Strict singleton decks by choice, not by rule.** Up to 4 copies are legal and duplicates stay in the candidate set at every pick, yet all four argmax decks are 50 distinct cards, and no card in any sampled top-30 exceeds mean 1.0. Greedy argmax picking a fresh card 50 times in a row means the pick head genuinely ranks any new card above any duplicate — the policy learned breadth (toolbox access) over redundancy (consistency).

3. **A hard tempo prior.** Average cost 2.12–2.24 across all four decks. Every element cut essentially its entire ≥5-cost entity slate (Wu Cha 8, Soryu no Rin 8, Zero 8, Teb Fea 8, Sandcoil Python 8, Lord of Sands 10, all the 5–6 cost midrange normals: Ed, Mo, Trade Guild Cavalry, Bladebound Ally, Gurugumi Mentor, Young Shao, Scorchland Raven were cut from **all four** decks). The only expensive cards kept are *spell finishers* the element's plan actually needs: Thunderclap (L, 5), Pulled Under / Shao's Perseverance (W, 6), Firestorm (F, 5), Crushing Weight / Quicksand (E, 5).

4. **Response-window investment tracks the element's identity.** Response-legal cards per argmax deck: Water 8, Lightning 8, Earth 7, **Fire 3**. Water/Lightning play the AEC response window as a core mechanic; Fire almost opts out of it and just races.

5. **Leader choice is nearly indifferent.** Every gate shows a 13/11 leader split over 24 sampled drafts. Fire even flips: argmax drafts Kagoro, but the sampled majority is Zero (13/11). Both leaders' abilities are supported by the same 50 cards in each element (see per-element notes), which is arguably why the head can stay soft there without cost.

---

## The 19-card neutral core (in all four decks)

Penny, Healing Flutter, Black Jade Courier, Alley Thug, Gus, Johnny, The Red Bean, Sleight of Hand, Hook Sword Strike, Monk Staff of Warding, Jay, Power of Friendship, Pip, Link, Beanz Mentor, Gurugumi Vanguard, Good Enough Replica, Invigorating Concoction, Alley Guy.

What this core actually is:

- **The Beanz micro-tribe as a card-advantage engine.** Link (`[On Play] look 5, tutor a Beanz card`) fetches Johnny / Gus / Pip / Penny / Jay / Beanz Mentor; Pip loots when played to the Alley; Beanz Mentor pumps other Beanz on attack. Fire/Lightning/Earth also carry Drunken Brewmaster (sacrifice 2 Beanz: +2/+2), and Lightning adds Fermented Beanz, Water adds Top Beanz — the tribe scales with the element's needs but the kernel travels everywhere.
- **Cheap interaction that works in any deck:** Hook Sword Strike (split ping that reaches the leader), Power of Friendship (Main pump / Response health trick), Invigorating Concoction (+2 attack, Main *or* Response — a combat-math blowout in the response window), Mocking Dummy (attack-shrink tap ability; in 3 of 4 argmax decks and all sampled lists).
- **The Red Bean in all four decks is the clearest learned gate synergy.** Gates portal entities from the Alley into the Garden, where they arrive without Charge; The Red Bean (1 IKZ, grant Charge to a ≤4-cost entity) converts any portal — or any freshly played body — into immediate attack tempo. It's the cheapest universal "portal → attack now" enabler in the pool.
- **Good Enough Replica** (return a NORMAL entity ≤6 from discard) is a universal recursion staple precisely *because* the model made NORMAL bodies the backbone of every deck — it always has targets.
- **Defensive glue:** Penny (Defender), Healing Flutter, Monk Staff of Warding (leader takes 1 less combat damage) — the same minimal survival package in every list.

The model also learned a consistent *reject list*: every 4–6 cost NORMAL entity outside this core (Ed, Mo, Trade Guild Cavalry, Bladebound Ally, Gurugumi Mentor, Young Shao, Scorchland Raven) is cut from all four decks. Big vanilla-ish bodies are dead weight under its tempo prior.

---

## Lightning — Raizan / Piko of Thousand Blades (Surge + Stormchain gates)

50 cards, avg cost 2.22 — 30 entities / 13 spells / **7 weapons (14%, the highest weapon share of any deck)**.

**Gate texts.** Surge: *portal from Alley → play a Weapon from your discard pile with cost ≤ the portaled entity's Gate Power.* Stormchain: *portal → re-equip a weapon in your Garden (cost ≤ Gate Power) to a different leader/entity.*

**Leader texts.** Raizan: *1 IKZ — give a weapon-equipped entity Charge.* Piko of Thousand Blades: *3 IKZ — a weapon-equipped entity gets +1 attack per Weapon in your discard pile, cap +3.*

This is the most clearly "built" deck of the four — a closed loop around **weapons in the discard pile**:

1. **Fill the discard with weapons:** Crate Rat Kurobo (self-mill 3, or 5 if no weapons yet in discard), Black Jade Pawnbroker (mill 5, may retrieve a weapon), Lightning Shuriken (mills on every attack), plus generic loot — Sleight of Hand, Pip, Alley Guy, Alpine Prowler (Alley sac: draw 3 discard 2).
2. **Cash weapons out of the discard:** Surge itself, Raizan's Riposte (`[Response] play a Weapon ≤2 from discard` — instant-speed stats mid-combat), Black Jade Recruit (discard a weapon → tutor a weapon), and Forging Tricks (bottom-deck up to 5 weapons from discard: leader gets +1 attack each — a mill-fueled burst finisher). Tenraku (+1 attack at 15+ cards in discard) is the passive mill payoff. Piko-the-leader's ability reads the same resource. The deck's weapon curve is deliberately bottom-heavy (Hidden Dagger 1, Lightning Shuriken 1, Stormglass Daggers 2, Lightning Kanabo 3) so that Surge portals off gate-power-1/2 bodies still hit.
3. **Convert equips into attacks now:** Raizan's Charge grant, The Red Bean, Piko-the-entity (`[When Equipped] gain Charge`), Arms Dealer Kin (from the Alley, tap: play a weapon ≤3 from hand).

Two supporting packages show real positional understanding:

- **Alley warfare / anti-gate tech:** Riven Flashborne and Stormcaller Tenkichi (0.96 mean in sampled decks; oddly absent from the argmax list) can attack entities *in the opponent's Alley*, and Stormglass Daggers/Katana give the **leader** that reach. Since every gate in the format is powered by untapped Alley entities, Alley-sniping is direct gate denial. The model drafted the entire Riftwalk suite.
- **Defensive Alley micro-game:** Kira (Alley: swap herself in as the new attack target), Ninpo: Thunderstep (Response: Garden/Alley swap, retargets the attack), Raimaru the Stolen (a Defender you can play *as a Response*), Denmu (shocks whatever attacks it — attacker skips its next untap).

Cheap interaction fills the rest: Sundering Strike, Lightning Orb, Hook Sword Strike, Silver Current Haruhi's attack-trigger ping, and Thunderclap (5) as the one board-wipe-shaped finisher.

**Telling cuts:** Weapon Master Yamada (+2 attack with 6+ weapons in discard) and Raizan's Zanbato both *fit the theme* but are 4-cost stat-poor cards — the model values curve over theme when they conflict. It also cut the generic Black Jade Dagger/Tenshin that every other deck plays, because its native weapon slots are better.

**Gate fit:** the deck serves Surge directly (weapons in discard + cheap portals) and Stormchain adequately (7 weapons and many bodies to re-equip across), which is exactly why one list can cover both gates.

---

## Water — Shao / Benzai the Sly (Hydromancy + Echoed Waves gates)

50 cards, avg cost 2.16 — 28 entities / **18 spells (36%, the spell-densest deck)** / 4 weapons. Only deck with a 0-cost card (Aquatic Veil) and the only one keeping 6-cost cards (Pulled Under, Shao's Perseverance).

**Gate texts.** Hydromancy: *portal → untap IKZ up to the portaled entity's Gate Power* (ramp). Echoed Waves: *portal → return a spell (cost ≤ Gate Power) from discard to hand* (spell recursion).

**Leader texts.** Shao: *1 IKZ Response — reduce an attacker's attack by 1.* Benzai the Sly: *1 IKZ — if you discarded this turn, your next card costs 2 less.*

The deck is a response-window control deck whose two gates feed the same resource — **spell mana**:

1. **Attack-shrink wall:** Shao's ability stacks with Aquatic Veil (discard 2: −3 attack, at 0 cost), Water Orb (discard 1: −2), Bubble Adept (sac: −1), Mocking Dummy (−1), and Power of Friendship's response mode. Combat math against this deck is unknowable for the opponent; damage-based removal also bounces off Foamback Crab / Serene Fist Misaki (**Effect Immune**) and Bubblemancer's immunity grant — which reads as deliberate tech against the Fire/Lightning ping decks that make up the rest of the league.
2. **Bounce tempo compounding into card advantage:** Mirage Frog, Commune with Water (Response), Rippling Recall, Mizuryuu's Torrent (2 entities, combined cost ≤5), Pulled Under, and Shao's Perseverance (bounce the *entire* opposing Garden at ≤6 cost) — with **Selis of the Shore** turning every single bounce into a card. Fumiko re-buys her own side's On-Play triggers (Benzai the Merchant's draw, Mizuto/Hayabusa Itto's tutors, Mirage Frog's bounce) by self-bouncing; Gus bounces himself when attacked.
3. **IKZ economy that pays for double-spelling:** Hydromancy's IKZ refund per portal, Lotus of Paradise (Response: untap 2 IKZ), and Benzai-the-Sly's discount, which is live almost every turn because the deck's costs *want* discards (Aquatic Veil, Water Orb, Sleight of Hand, Pip). This is how a 2.16-curve deck realistically casts its 6-drops.
4. **Echoed Waves recursion fuel:** 13 of the 18 spells cost ≤2, exactly the band a gate-power-1/2 portal can return. Water is also the only element that kept both 3-cost/gate-power-2 NORMAL bodies (Top Beanz, Stalking Assassin), raising portal payoffs for both gates — the ramp gate and the recursion gate reward the same stat.
5. **Card selection:** Tidal Insight, Lotus of Reflection (dig 5 *and* free-cast a water spell ≤2), Hayabusa Itto and Mizuto as subtype tutors.

**Telling cuts:** every water bomb (Mizuki, Kaiya Mizumi, Naiyara the Tideweaver, Soryu no Rin, Mizuryuu Fist Master) and every ≥4 entity, plus Forging Tricks (no weapons to fuel it — the only deck to cut it). Control here means spells, never big bodies.

---

## Fire — Kagoro / Zero (Rushfire + Ragefire gates)

50 cards, avg cost 2.12 (lowest) — **33 entities (66%, the most creature-dense deck)** / 14 spells / 3 weapons. Zero entities with gate-power ≥2, and only 3 response cards — this deck mostly declines the response-window game and races.

**Gate texts.** Rushfire: *portal → you may play an entity (cost ≤ Gate Power) from hand with Charge, sacrificing it at end of turn.* Ragefire: *portal → an entity that took damage this turn gets +attack equal to Gate Power.*

**Leader texts.** Kagoro: *1 IKZ — +1 attack on the leader per entity played this turn, cap +2* (go-wide payoff). Zero: *deal 1 to Zero — deal 1 to one of your entities, it gets +1 attack* (self-damage enabler). Argmax drafts Kagoro; the sampled mode narrowly prefers Zero (13/11). Notably the deck supports both: it is simultaneously the widest deck (Kagoro) and the self-damage deck (Zero).

Three interlocking engines:

1. **The enrage engine (Ragefire's condition is generated in-house).** Damage sources the deck controls: Cinderwake Seer (pings itself every turn), Lounge Siren Saeko (1 to your entity + 1 to theirs every turn), Reckless Tinkerer (Charge, pings itself on play), Collateral Burst (1 to yours → up to 2 to theirs), Zero's leader ability, Fire Orb / Detonation Pact / Ignition Pact (leader self-damage). Payoffs that read "when damaged": Spice (+1 attack to another entity), Spiteful Raider (1 damage anywhere), and **Firebrand Renji** (3 distinct damage sources in one turn → 3 damage anywhere — Seer + Saeko + Zero can trigger him without the opponent's help), plus Ragefire itself pumping any entity that took damage this turn. The model drafted the complete tribe: sources, sinks, and the gate as the top-end payoff.
2. **Rushfire burst with sacrifice value.** Rushfire's drawback (sacrifice the free Charge entity at end of turn) is converted into value with **Crazed Arsonist** — `[When Sacrificed] deal 1 damage to all leaders` — making him the ideal Rushfire target: attack with Charge, then his forced sacrifice burns face again. Black Jade Warlord sacrifices *itself* after attacking for a +2 pump; Wrath of Sinder (sac N entities → N damage anywhere) and Fanatic Kindler give the deck manual sacrifice outlets for the same bodies.
3. **Face-damage reach to close races:** Detonation Pact (1 self → 2 face), Fire Orb (3 self → up to 5 anywhere), Firestorm (2 AoE), Crazed Arsonist / Saeko chip, Omen Peddler (2 damage when played to the Alley — an Alley card that generates value *before* it portals). Healing Flutter and Monk Staff of Warding are the only concessions to all the self-inflicted damage.

Charge density is also the highest here (Reckless Tinkerer, Cinderwake Pursuer's conditional Charge, Rushfire itself, The Red Bean, Ignition Pact ≤5), matching a deck whose entities all want to convert to damage immediately — and whose Kagoro payoff counts entities *played this turn* (Cinderwake Pursuer's "played 2 other cards" trigger reads the same game state).

**Telling cuts:** Tenraku (the only element to cut it — weapons are not the plan), all ≥4 entities including on-theme ones (Kurai the Volcano, Spiteful-adjacent Suzuka, Enzo), and both extra NORMAL 1-drops other decks keep (Rei, Black Jade Recruit) in favor of fire 1-drops that ping.

---

## Earth — Goro Graveloth / Bobu (Devotion + Stonehaven gates)

50 cards, avg cost 2.24 (highest) — 30 entities / 16 spells / 4 weapons.

**Gate texts.** Devotion: *portal → you may sacrifice another untapped entity (cost ≤ Gate Power); deal damage equal to the sacrificed entity's health to an entity in the opponent's Garden.* Stonehaven: *portal → give an entity with base health ≤ Gate Power Defender until your next turn.*

**Leader texts.** Goro: *1 IKZ — +1 health to an entity until end of turn.* Bobu: *1 IKZ — until your next turn, first Earth entity destroyed/sacrificed heals your leader 1* (directly subsidizes Devotion's sacrifice cost).

The deck's organizing idea is **health as a convertible resource** — the same stat serves defense, healing, and burn:

1. **Health pumps and healing:** Goro's ability, Sand Stands Still (+3 health to an Earth entity, Main or Response), Oathstone (Carapace 1), Jar of Beans (heal 3, or 5 at 7+ IKZ), Shiko the Priestess (heal on attack), Sloth Scarecrow (Defender, sac: heal 1), Shroom Tender (heal 2), Healing Flutter.
2. **Health→damage converters:** Crushing Weight (deal damage equal to one of your Earth entities' health, cap 5), Prickly Tumbleweed (sac: deal its own health — 3 base — as damage), and **Devotion itself** (damage = sacrificed entity's health). All three scale with the pumps above: Sand Stands Still or Goro's +1 turn a defensive trick into extra reach the same turn. This is the most "combo-shaped" synergy in any of the four decks.
3. **Devotion ammo:** cheap high-health bodies that are happy to be sacrificed — Mocking Dummy (0/2), Rei (1/2), Alley Guy (1/2), Wolf Cub (1/2) — plus death-triggered value from Wobbly Cabbage Cart (on destroy: kill a 1-health entity) and Cactus Farmer (on destroy: loot). *Caveat:* Bobu's text distinguishes "destroyed or sacrificed", so `[When Destroyed]` triggers may not fire on Devotion's sacrifice in the engine; the draft pattern is suggestive either way.
4. **The inevitability clock + ramp:** Mina the Geomancer (permanently tapped, pings the enemy leader every turn — a win condition that never attacks), Koyama Farm Caretaker (counts as an IKZ card while in the Garden), Offering to Stillstone (0-cost temporary IKZ) ramping into the 5-drops: Quicksand (destroy all ≤2-health entities — a wipe the deck's own high-health bodies largely dodge) and Crushing Weight. Jar of Beans' 7-IKZ kicker reads the same ramp.
5. **Stonehaven wall:** the gate grants Defender by *base health*, layering with natural Defenders (Penny, Sloth Scarecrow), Wrong Step (Response: destroy an attacker with ≤2 health), Earth Orb (ping + attack-shrink), and the pump suite to make the granted Defender actually survive.

**Telling cuts:** the entire expensive Earth top-end (Rock Sloth, Stone Masked Ancient, Sandcoil Python, Lord of Sands at 10, Warding Totem, Shroommancer) — even in the element whose identity is big health, the model refuses to pay midrange prices for it, preferring to *manufacture* big health on cheap bodies.

---

## What this says about the policy

- **Element-aware, gate-blind at draft time.** Composition keys entirely on the leader-element pool. But the four element decks are each *dual-gate viable by construction* — lightning's weapon-mill serves Surge and its equip density serves Stormchain; water's cheap spells serve Echoed Waves and its gate-point bodies serve Hydromancy; fire's self-damage serves Ragefire and its sac-value bodies serve Rushfire; earth's fat-health fodder serves Devotion and its base-health bodies serve Stonehaven. The remaining gate adaptation is a play-time problem (which the per-gate strategy atlas covers), not necessarily a draft-time failure — though a stronger gate signal should still show *some* compositional divergence, and none exists.
- **Real cross-card reasoning, not stat drafting.** The clearest evidence: Crazed Arsonist valued in the element whose gate force-sacrifices (Rushfire); Selis of the Shore alongside nine bounce effects; Forging Tricks kept in exactly the decks that mill weapons and cut in the one that doesn't; Riftwalk Alley-attackers as gate denial; Top Beanz/Stalking Assassin (gate-power 2) kept only where gate payoffs scale with gate power and IKZ. These are second-order interactions between draft picks and rules text.
- **One strong global prior: tempo.** Curve ceiling ~5, singleton toolbox breadth, response-window density where the element supports it. Whether the singleton habit is optimal or a pick-head artifact (novelty always outranking a duplicate) is untested — an ablation forcing 2-of cores against the singleton baseline would answer it.

## Caveats

- The argmax deck is a single greedy draft (seed 90001); the sampled aggregate shows only the top-30 of 50 slots, so the tail of the sampled distribution is unobserved. Mean copies never exceeding 1.0 bounds duplicates as rare but doesn't strictly rule them out in the unshown tail.
- Effect text is taken from the web-service seed data; the C engine implements these effects, but two rule interactions cited above are inferred from text and flagged inline (sacrifice vs. destroyed triggers; portaled entities lacking Charge by default).
- Opponent seat during these dumps is a fixed constant gate and ignored, so no claims here about matchup-conditional drafting.
