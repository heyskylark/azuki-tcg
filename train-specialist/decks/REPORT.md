# Curated specialist deck pool — report (2026-09-30, Earth additions 2026-10-05)

Files
- `curated_decks.json` — authoritative manifest: 22 unique lists with tier, evidence text, source URLs, event, date, placement/record, Rushfire status, matching Garden Arena submission numbers.
- `curated_deck_pool.json` — training pool built by `build_curated_pool.py` (rerun: `.venv/bin/python train-specialist/decks/build_curated_pool.py`).
- `raw/thegate_supabase_2026-09-30.json`, `raw/thegate_supabase_2026-10-04.json`, `raw/egman_azuki_2026-09-30.json` — API snapshots; `raw/evidence/*.jpg` — official/article decklist and rank images used as evidence.
- `earth_candidates_2026-10-04.json` — Earth research: candidate lists, evidence, and 9-deck panel simulations (including Gate of Devotion swap controls).

Rushfire Gate (AZK01-122) stays legal, as the user asked. Pre-ban lists are included. All 22 lists pass these checks: 1 leader + 1 gate of the same element, 50 main cards that are Neutral or the gate's element, at most 4 copies of any card, IKZ-001 x10, every card in the official card list, and every card either in engine metadata or one of the 5 pending cards. Only `pasadena_dkb_goro_36th` (x3) and `pasadena_yqqq_goro_regional` (x2) use a pending card: AZK01-079 Gin and Tonika.

### Earth additions (2026-10-05, user-approved)
These three decks were added on the user's judgement. They do not meet the strict A/B evidence bar and are tier B (sampling weight 1).
- `pasadena_yqqq_goro_regional`: yqbird (YQ) posted this Goro/Stonehaven list in "Post Regional Goro Deck Notes". It exactly matches Garden Arena submission #60. Its Pasadena placement was not published.
- `online_yqbird_goro_devotion`: the same player's Goro/Gate of Devotion list, 2nd (3-1) at a 9-player online event (2026-03-03). It is the only Devotion list with any recorded placement.
- `constructed_bobu_devotion_case1`: **constructed**. It is the `case1_bobu_3rd` main deck on Gate of Devotion. No real Bobu/Devotion list is public, and the public Gate lists scored 12–29% in simulation. Replace it when a real list appears.

## How lists were verified
- **Pasadena Garden Arena Top 8** (315 players, 2026-08-15). Placements come from the official recap (https://tcg.azuki.com/blog/01a017f6-5ed3-7990-a97e-4540e4c0b3d5). Lists come from the official X decklist images (Top 4: https://x.com/AzukiTCG/status/2089407723720818814; places 8–5: https://x.com/AzukiTCG/status/2089407720902238468) and the MarredPlatypus Top-8 video (https://www.youtube.com/watch?v=I00VDjQ0k2E). Each list was transcribed and then matched against the 315 anonymous submissions (`.codex/docs/azuki_garden_arena_2026-08-15_deck_submissions.json`). All 8 match a submission exactly (L1 distance 0), so the card codes are the registered ones.
- Two pairs of players registered identical lists: ducky (1st) = drewwskiee (2nd), and mrmexicant (3rd) = jbae (4th). Each pair is one deck entry with both placements recorded.
- **2026 Invitational** (64 invited players, 2026-01-11). Lists come from The Gate's official-tagged `2026 Invitational` decks. Placements and records come from The Gate's `tournament_placements` table. Two more pairs are identical: xell = toraoy and dylhu = jackie.
- **CoreTCG Case Tournament** (2026-06-30, official "Small Official" event with a Top-8 cut). Results come from the official X post (https://x.com/AzukiTCG/status/2072107915411132798). Lists come from the official X decklist images (https://x.com/AzukiTCG/status/2072119400073929202) and the Egman deck DB (https://deckbuilder.egmanevents.com/azuki/tournaments).
- **Article-backed lists.** Each deck was matched exactly to the deck embedded in, or pictured in, a Gate article, and where possible to a Garden Arena submission.

## Per-deck table (pool index, tier, split)
| idx | id | element | leader / gate | tier | split | date | placements | Rushfire status |
|---|---|---|---|---|---|---|---|---|
| 5 | `pasadena_ducky_drewwskiee` | FIRE | Zero / Rushfire Gate | A | train | 2026-08-15 | ducky 1 (7-1 swiss, champion (3-0 top cut)); drewwskiee 2 (7-1 swiss, finalist) | pre-ban |
| 6 | `pasadena_mrmexicant_jbae` | FIRE | Zero / Rushfire Gate | A | train | 2026-08-15 | mrmexicant 3 (7-1 swiss, top 4); jbae 4 (7-1 swiss, top 4) | pre-ban |
| 7 | `pasadena_szerro` | FIRE | Zero / Rushfire Gate | A | train | 2026-08-15 | szerro 8 (7-1 swiss, top 8) | pre-ban |
| 0 | `pasadena_orion` | FIRE | Zero / Rushfire Gate | A | held-out | 2026-08-15 | orion 7 (7-1 swiss, top 8) | pre-ban |
| 1 | `case1_brayan` | FIRE | Zero / Rushfire Gate | A | held-out | 2026-06-30 | brayan 2 (4-1) | pre-ban |
| 19 | `inv_xell_toraoy` | FIRE | Zero / Ragefire Gate | A | train | 2026-01-11 | xell 2 (5-1); toraoy 5 (5-1) | pre-Rushfire (Jan) |
| 2 | `inv_dylhu_jackie` | FIRE | Zero / Ragefire Gate | A | held-out | 2026-01-11 | dylhu 4 (5-1); jackie 8 (5-1) | pre-Rushfire (Jan) |
| 20 | `inv_bfg` | FIRE | Zero / Ragefire Gate | A | train | 2026-01-11 | bfg 6 (4-2) | pre-Rushfire (Jan) |
| 21 | `inv_drewwskiee` | FIRE | Zero / Ragefire Gate | A | train | 2026-01-11 | drewwskiee 7 (5-1) | pre-Rushfire (Jan) |
| 8 | `pasadena_bjmcs_19th` | FIRE | Zero / Rushfire Gate | B | train | 2026-08-15 | (The Gate author of "High Stakes Fire") 19 (19th of 315 (top 6%); 2 losses to eventual top-8 players) | pre-ban |
| 11 | `pasadena_baconman` | LIGHTNING | Raizan / Surge Gate | A | train | 2026-08-15 | BaconMan 5 (8-0 swiss (only undefeated), top 8) | pre-ban |
| 4 | `case1_banchan` | LIGHTNING | Raizan / Stormchain Gate | A | train | 2026-06-30 | notshanenam (Egman: banchan) 1 (5-0) | pre-ban |
| 12 | `inv_ducky` | LIGHTNING | Raizan / Surge Gate | A | train | 2026-01-11 | ducky 1 (5-1) | pre-Rushfire (Jan) |
| 3 | `inv_jerry` | LIGHTNING | Raizan / Surge Gate | A | held-out | 2026-01-11 | jerry 3 (6-0) | pre-Rushfire (Jan) |
| 17 | `pasadena_cat` | EARTH | Bobu / Stonehaven Gate | A | train | 2026-08-15 | cat 6 (7-1 swiss, top 8) | pre-ban |
| 18 | `case1_bobu_3rd` | EARTH | Bobu / Stonehaven Gate | B | train | 2026-06-30 | unattributed The Gate user 3 (unknown) | pre-ban |
| 15 | `pasadena_dkb_goro_36th` | EARTH | Goro Graveloth / Stonehaven Gate | B | train | 2026-08-15 | dkb 36 (6-2 (36th of 315, top 11.4%)) | pre-ban |
| 16 | `pasadena_yqqq_goro_regional` | EARTH | Goro Graveloth / Stonehaven Gate | B (user-approved) | train | 2026-08-15 | yqbird Pasadena participant, placement unpublished (GA #60) | pre-ban |
| 9 | `online_yqbird_goro_devotion` | EARTH | Goro Graveloth / Gate of Devotion | B (user-approved) | train | 2026-03-03 | yqbird 2 (3-1, 9 players) | pre-Rushfire (Mar) |
| 10 | `constructed_bobu_devotion_case1` | EARTH | Bobu / Gate of Devotion | B (constructed) | train | 2026-10-05 | none; Case 3rd main deck on Devotion | n/a |
| 13 | `inv_aez_shao_v1` | WATER | Shao / Hydromancy Gate | B | train | 2026-01-11 | aez 15 (4-2) | pre-Rushfire (Jan) |
| 14 | `ladder1_aez_shao_v2` | WATER | Shao / Hydromancy Gate | B | train | 2026-09-15 | aez #1 ladder (n/a) | pre-ban |

## Element balance

| element | tier A | tier B | training decks | held-out | contexts trained (gate/leader) |
|---|---|---|---|---|---|
| FIRE | 9 | 1 | 7 (6A, 1B) | 3A | Zero/Rushfire (4), Zero/Ragefire (3) |
| LIGHTNING | 4 | 0 | 3 (3A) | 1A | Raizan/Surge (2), Raizan/Stormchain (1) |
| EARTH | 1 | 5 | 6 (1A, 5B) | 0 | Bobu/Stonehaven (2), Goro/Stonehaven (2), Goro/Devotion (1), Bobu/Devotion (1, constructed) |
| WATER | 0 | 2 | 2 (2B) | 0 | Shao/Hydromancy (2) |

Held-out decks are about one third of each element's tier-A decks, rounded: Fire 3 of 9, Lightning 1 of 4, Earth 0 of 1, Water 0 of 0. They sit at pool indices 0–3 (`summary.holdout_reference_deck_indices`) and are never training decks.
- Fire held-out: orion (Pasadena 7th, Rushfire), brayan (Case 2nd, Black Jade go-wide Rushfire), and dylhu/jackie (Invitational 4th/8th, Ragefire). Together they cover both Fire gates and a non-Cinderwake build.
- Lightning held-out: jerry (Invitational 3rd, 6-0, Ikazuchi build).
- The champion lists (ducky Pasadena, ducky Invitational, BaconMan 8-0, banchan Case 1st, cat) stay in training.

### Archetypes covered
- **Fire:** Zero Rushfire Cinderwake burn (4 Pasadena Top-8 variants, plus the 19th-place High Stakes list); Zero Rushfire Black Jade go-wide (Case 2nd); Zero Ragefire burn/tempo from the January Invitational (Healing Flutter, Tenmoku Daiki, and Ignition Pact variants). The Ragefire lists are the only proven Fire lists that stay legal after the ban.
- **Lightning:** Raizan weapons/Black Jade tempo (BaconMan), the 16-weapon Mo/Rooftop build on Stormchain (Case 1st), and January Raizan weapons (Elder Hoshin/Vault Master; Ikazuchi/Pawnbroker).
- **Earth:** Bobu Stonehaven ramp-control (cat), Bobu Stonehaven Kale/Shiko aggro-control (Case 3rd), Goro Beanz defender/heal (dkb; yqqq's Shiko/Mo variant), Goro Devotion removal-control (yqbird), and a constructed Bobu Devotion stand-in.
- **Water:** Shao Hydromancy tempo (Aez, January) and Shao Hydromancy midrange (Aez, #1 ladder).

### Gaps (stated honestly)
- **Water** has no tier-A list and no held-out deck. The only proven Shao lists are Aez's (Invitational 15th; #1 ladder). Pasadena had 5 Shao in the Top 32 and 9 in the Top 64 (MarredPlatypus video description), but none of those lists are attributable to a placement. Water has a single context, Shao/Hydromancy, so Benzai and Gate of Echoed Waves are unproven.
- **Earth** has only one tier-A list (cat), so nothing is held out. `pasadena_dkb_goro_36th` is a borderline B: 36th of 315 at 6-2 is the top 11.4%, just outside the top-10%/top-32 guideline. It was kept because it is the only documented Goro finish (Top-64 prize tier, official rank screenshot). Gate of Devotion evidence is thin: yqbird's list placed 2nd of 9, and the Bobu/Devotion deck is constructed. Post-ban Devotion results exist (shozen 3rd at the Kaleido three-case on 2026-08-30; Doc_Camel 1st 4-0 online on 2026-08-23), but their lists are unpublished.
- **Leaders with no proven lists:** Kagoro (Fire), Piko (Lightning), Benzai (Water). No gate/leader context of theirs is in the pool.
- **Post-ban evidence:** there are no published post-ban (after 2026-08-18) tournament decklists from a sizeable event. Ragefire Fire is represented only by January 2026 lists, which predate the AZK01 full-set release.
- **Single-deck contexts:** Raizan/Stormchain (idx 4), Goro/Devotion (idx 9) and Bobu/Devotion (idx 10) each have only one deck. Under context-uniform sampling, each Devotion context gets a quarter of Earth's prebuilt mass. Per-deck `sampling_weight` (A=2, B=1) is provided.
- **Case 1st gate conflict:** the official decklist image shows Stormchain Gate, while the result-post text and Egman (which transcribed that text) say Surge Gate. I used the image, since it is the actual list. The main 50 cards are identical in both sources.

## Rejected notable candidates
| candidate | source | reason |
|---|---|---|
| "Water is Wet" Shao (my-deck-mt69ryhr, Garden Arena submission 30) | https://thegateikz.com/articles/water-is-wet | 5-3, 101st of 315 (top 32%), below the B bar |
| Baby Platypus Goro/Stonehaven Beanz | https://www.youtube.com/watch?v=I00VDjQ0k2E (02:30–06:30) | 55th of 315 (top 17.5%), below the B bar |
| Szerro NYC list (Zero/Rushfire, Dagger/Sinder) | https://thegateikz.com/articles/szerro-sligh, https://thegateikz.com/articles/azuki-tcg-tournament-recap-nyc | 1st of a 10-player local, too small. Szerro's Pasadena Top-8 list is included instead |
| NYC 2nd navindrenhodges (Raizan/Surge), 3rd jathinganesh (Bobu/Devotion), 4th V1Zhual (Zero/Rushfire) | https://thegateikz.com/articles/azuki-tcg-tournament-recap-nyc | no decklists published; 10-player field |
| CoreTCG Case 4th (Raizan/Surge) | https://x.com/AzukiTCG/status/2072107915411132798 | no decklist published |
| CoreTCG double-case tourney Top 4 (2026-07-16: Raizan, Bobu, Zero, Raizan) | https://x.com/AzukiTCG/status/2077883977747333620 | only a group photo; no decklists |
| Azuki Online Pre-Season (Full Set 1): Tiny 1st 4-0 "Raizenv2" | The Gate `tournament_placements` | 9-player online event, too small (yqbird's 2nd-place Earth list was later added on user approval) |
| Azuki Online Pre-Season #3: Orpheus 2-0, NAN, JUN ("⚡ 3/1 Genki 🥉") | The Gate `tournament_placements` | 6-player online event, too small |
| "Zero Top 8" (my-deck-mkbix0mz) | https://thegateikz.com/decks/my-deck-mkbix0mz | unattributed variant of bfg's official Invitational 6th list (4 Collateral Burst instead of 4 Ignition Pact); the official-tagged list is used |
| "Invitational 16th" Zero/Ragefire (invitational-16th-mkd3yy9k) | https://thegateikz.com/decks/invitational-16th-mkd3yy9k | placement only in the deck title; no rank, record, or author evidence |
| Bobu's Rocky Invitational (Bobu, 4-2 by the article's matchup tallies) | https://thegateikz.com/articles/bobu-s-rocky-invitational | no final rank documented; list only in images and not matchable |
| Post-Regional Goro deck notes | https://thegateikz.com/articles/postregional-goro-deck-notes | its list is Goro/Stonehaven, not Devotion; now included as `pasadena_yqqq_goro_regional` on user approval |
| Zero Midrange (my-deck-mq1qo54w; also Garden Arena submission 258) | https://thegateikz.com/articles/every-point-of-damage-matters | deck tech with no result |
| Miss the Rage (post-ban Ragefire) | https://thegateikz.com/articles/miss-the-rage | theorycraft with no result |
| "PKMNs 1st PL - HQ" Zero/Rushfire (also Garden Arena submission 48) | https://thegateikz.com/decks/wetpotatos-1st-hq-msqe6m1e | weekly HQ local win, attendance unknown; not a store championship; regional placement unknown |
| "Alpha Local winner" Raizan/Surge (2026-09-10, post-ban) | https://thegateikz.com/decks/alpha-local-winner-mtv3uh6r | local win, attendance unknown |
| "Genki- 5th or 6th?" Zero (Dec 2025) | https://thegateikz.com/decks/saka-mjbq8lc7 | uncertain 5th/6th at a local of unknown size |
| "3rd place coretcg" later copies (mrfviiv1, mstiause) | The Gate | byte-identical copies of the included 2026-07-01 list |
| AEZ RushFire, Regionals- Ex, Current List, PapaBear "Competitive - PB" series, starter decks, other untitled "My Deck" brews | The Gate decks | no placement evidence (casual/untested) |
| The remaining 299 Garden Arena submissions (16 of the 315 exactly match included lists) | official submissions JSON | anonymous, with no placements; used only to verify the card codes of named lists |
| Szerro "Azuki TCG Meta Report #1" ("Proven Sample Decklists", including Tempo Shao) | https://metafy.gg/guides/view/azuki-tcg-meta-report-number-1-rGpotILbzV7 | paywalled; lists and results not readable, so not used |

## Loader verification (snapshot)
- `load_training_deck_pool('train-specialist/decks/curated_deck_pool.json')` returns 22 decks, all of 62 cards. `load_training_deck_labels` returns deck slugs.
- `prebuilt_deck_pool.load_specialist_deck_groups` returns groups `((4,), (5,6,7,8), (9,), (10,), (11,12), (13,14), (15,16), (17,18), (19,20,21))`.
- The legacy `load_prebuilt_deck_groups`, which requires an 18-deck reference panel and all 16 contexts, is not used for this pool.
