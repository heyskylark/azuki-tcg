# Specialist behavior profile vs u8223 (paired natural-game traces)

Output dir: `train-specialist/analysis/behavior/` (snapshot runtime). No training, no source edits; tools are patched copies in `tools/`.

## Design
- **Arms (candidate seat):** element specialist, u8223, u9305 (deterministic only). **Opponent:** always u8223. Every arm plays the *identical* task list:
  same world seed, decks, seat and opponent sampling stream (`tools/run_traces.py::build_tasks`, seeds = sha256 of block id). Trajectories diverge after the first different action.
- **Fixed suite:** candidate plays every curated deck of its element (Fire capped at 6: 0,1,5 Rushfire + 2,16,17 Ragefire; Lightning 3,4,9,10; Water 11,12; Earth 13,14,15) × opponent panel {0,2,3,4,11,12,13,14} (both Fire gates, Surge, Stormchain, 2× Hydromancy, Stonehaven Goro/Bobu) × both seats × 4 replicates.
- **Free-draft suite:** candidate drafts 50 cards under every (gate, leader) context its element can draft in the curated catalog (Fire 4, Lightning 4, Water 2, Earth 2; Devotion/Echoed Waves gates are not in the pool) × same opponent panel × both seats × 3 replicates.
- 7,680 complete games (0 invalid, 0 truncated): per element and arm-mode 576 Fire / 448 Lightning / 224 Water / 288 Earth games. Both modes: argmax and sampled (T=1, no smoothing, private per-seat/phase CPU RNG streams). Inference ran on GPU; sampling on CPU so the streams decide.
- **Metrics:** every rate is (uses / legal opportunities), ratio-of-sums. Opportunity unit = candidate turn (legal at least once that turn), response window, or upstream funnel stage. Paired Δ = arm − u8223; 95% CI = 2,000-rep bootstrap over seed blocks (the two seat-swapped games that share a seed). **Bold** = CI excludes 0. n = opportunity count.

## Per-element results (all suites pooled)

#### FIRE (all suites pooled; u8223 → fire)

| metric | u8223 argmax (n) | spec argmax (n) | Δ argmax [95% CI] | Δ sampled [95% CI] (n u8223/spec) | u9305−u8223 argmax |
|---|---|---|---|---|---|
| outcome/win | 70.8 (576) | 74.0 (576) | +3.1 [-0.3,+6.4] | **+3.6** [+0.3,+6.9] (576/576) | -1.2 [-4.3,+1.6] |
| strategy/favorable_trade_taken_per_available_turn | 51.9 (1000) | 58.5 (1015) | **+6.6** [+3.2,+9.9] | **+8.9** [+5.5,+12.3] (1107/1158) | +2.4 [-0.7,+5.5] |
| strategy/face_target_share_when_both_legal | 68.4 (4095) | 64.0 (4042) | **-4.4** [-6.6,-2.4] | **-6.1** [-8.4,-3.9] (4259/4251) | -0.5 [-2.1,+1.1] |
| response/spell_played_when_legal | 47.3 (55) | 64.0 (50) | +16.7 [+0.0,+34.7] | **+20.4** [+5.6,+36.5] (63/54) | -2.0 [-15.8,+11.0] |
| response/any_nonblock_response_when_legal | 72.0 (107) | 83.0 (112) | +11.1 [-0.4,+22.9] | **+13.3** [+4.3,+22.4] (133/131) | -1.0 [-11.5,+8.5] |
| gate/portal_per_legal_turn | 98.8 (2317) | 98.2 (2217) | -0.6 [-1.2,+0.1] | **-0.7** [-1.4,-0.1] (2324/2238) | -0.5 [-1.1,+0.1] |
| gate/Rushfire/portal_then_payload | 72.6 (1392) | 74.5 (1390) | +2.0 [-0.4,+4.5] | -0.1 [-3.0,+2.8] (1421/1415) | -1.0 [-3.2,+1.2] |
| gate/Rushfire/payload_then_attack_same_turn | 98.2 (1010) | 94.6 (1036) | **-3.6** [-5.0,-2.2] | **-2.7** [-4.1,-1.4] (982/977) | -0.8 [-1.9,+0.3] |
| gate/Ragefire/portal_then_buff | 29.2 (897) | 31.5 (788) | +2.3 [-1.1,+5.7] | +0.1 [-3.4,+3.5] (879/784) | +2.0 [-0.8,+4.8] |
| gate/Ragefire/buffed_then_attacks_same_turn | 47.7 (262) | 54.8 (248) | **+7.1** [+0.5,+13.8] | +7.2 [-0.7,+14.8] (227/203) | -5.7 [-11.1,+0.0] |
| leader/use_per_legal_turn | 5.8 (3169) | 5.1 (3192) | -0.7 [-1.6,+0.2] | **-1.1** [-2.2,-0.1] (3198/3206) | -0.2 [-1.1,+0.7] |
| fire/burn_cast_per_legal_turn | 28.5 (1724) | 27.0 (1740) | -1.6 [-3.2,+0.0] | **-2.8** [-4.5,-1.1] (1969/1985) | **-1.4** [-2.7,-0.1] |
| fire/burn_cast_then_opp_damage | 92.0 (654) | 93.2 (616) | +1.1 [-1.7,+4.0] | -1.3 [-3.7,+1.1] (715/623) | -0.1 [-2.5,+2.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 31.4 (315) | **+5.8** [+0.5,+11.0] | **+7.9** [+2.5,+13.4] (345/295) | -0.8 [-4.9,+3.9] |
| ikz/held_then_spent_in_opp_turn | 3.5 (751) | 4.5 (708) | +1.1 [-0.2,+2.4] | +1.0 [-0.3,+2.4] (805/756) | +0.2 [-0.5,+0.9] |
| outcome/own_turns | 6.10 (576) | 6.11 (576) | +0.0 [-0.1,+0.1] | +0.0 [-0.1,+0.1] (576/576) | +0.0 [-0.0,+0.1] |

#### LIGHTNING (all suites pooled; u8223 → lightning)

| metric | u8223 argmax (n) | spec argmax (n) | Δ argmax [95% CI] | Δ sampled [95% CI] (n u8223/spec) | u9305−u8223 argmax |
|---|---|---|---|---|---|
| outcome/win | 51.6 (448) | 57.8 (448) | **+6.2** [+1.6,+10.7] | **+4.9** [+0.4,+9.6] (448/448) | +0.4 [-3.3,+4.7] |
| strategy/favorable_trade_taken_per_available_turn | 42.4 (874) | 37.0 (738) | **-5.5** [-9.0,-1.9] | -3.0 [-7.5,+1.6] (844/813) | **+7.1** [+3.3,+10.8] |
| strategy/face_target_share_when_both_legal | 65.9 (3204) | 70.7 (3093) | **+4.9** [+2.6,+7.1] | +1.0 [-1.2,+3.2] (3183/3209) | **-3.6** [-5.6,-1.7] |
| strategy/attacks_by_equipped_attacker | 22.3 (5166) | 25.3 (5374) | **+3.1** [+1.9,+4.2] | **+1.5** [+0.3,+2.7] (5011/5187) | -0.4 [-1.3,+0.6] |
| gate/portal_per_legal_turn | 99.8 (1764) | 99.6 (1683) | -0.2 [-0.6,+0.2] | -0.1 [-0.4,+0.1] (1776/1672) | +0.1 [-0.2,+0.3] |
| gate/Surge/portal_with_eligible_weapon | 53.3 (1350) | 65.5 (1400) | **+12.2** [+8.4,+16.1] | **+5.7** [+2.1,+9.4] (1350/1397) | -2.1 [-4.9,+0.6] |
| gate/Surge/equip_then_destination_attacks | 97.5 (719) | 97.4 (917) | -0.1 [-1.5,+1.4] | **-1.6** [-2.9,-0.2] (694/798) | +0.1 [-1.2,+1.4] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.8 (215) | 93.0 (243) | **+4.2** [+0.5,+8.3] | +2.7 [-1.5,+6.7] (206/215) | +1.8 [-1.5,+5.5] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (410) | 0.0 (276) | +0.0 [+0.0,+0.0] | +0.0 [+0.0,+0.0] (423/270) | +0.0 [+0.0,+0.0] |
| lightning/weapon_attach_per_legal_turn | 26.4 (2385) | 27.5 (2430) | +1.1 [-0.6,+2.9] | +0.7 [-1.0,+2.4] (2280/2359) | -0.6 [-2.1,+1.1] |
| lightning/weapon_attach_to_entity_share | 0.3 (1035) | 0.0 (1084) | -0.3 [-0.7,+0.0] | -0.2 [-0.6,+0.2] (1028/1082) | -0.2 [-0.5,+0.0] |
| response/any_nonblock_response_when_legal | 79.5 (337) | 86.7 (361) | **+7.2** [+1.9,+12.6] | +6.3 [-0.4,+12.7] (337/366) | -0.5 [-6.3,+5.6] |
| ikz/held_then_spent_in_opp_turn | 33.6 (745) | 38.6 (731) | **+5.0** [+1.3,+8.6] | +1.9 [-1.3,+5.2] (759/820) | +0.6 [-2.1,+3.3] |
| outcome/own_turns | 6.36 (448) | 6.35 (448) | -0.0 [-0.1,+0.1] | +0.1 [-0.1,+0.2] (448/448) | -0.0 [-0.1,+0.1] |

#### WATER (all suites pooled; u8223 → water)

| metric | u8223 argmax (n) | spec argmax (n) | Δ argmax [95% CI] | Δ sampled [95% CI] (n u8223/spec) | u9305−u8223 argmax |
|---|---|---|---|---|---|
| outcome/win | 57.1 (224) | 63.8 (224) | **+6.7** [+0.4,+13.4] | **+9.8** [+3.1,+16.1] (224/224) | +0.0 [-5.8,+5.8] |
| gate/portal_per_legal_turn | 97.3 (1176) | 97.5 (1182) | +0.3 [-1.1,+1.7] | +0.2 [-0.8,+1.2] (1319/1207) | +0.2 [-0.9,+1.2] |
| gate/Hydromancy/portal_readied_ikz | 90.7 (1144) | 92.5 (1153) | +1.7 [-0.1,+3.6] | +1.4 [-0.4,+3.3] (1295/1187) | +1.4 [-0.1,+3.1] |
| gate/Hydromancy/readied_then_spent_same_turn | 78.4 (1038) | 77.0 (1066) | -1.4 [-4.3,+1.6] | +3.3 [-3.3,+11.0] (1176/1095) | -1.7 [-4.3,+0.8] |
| leader/use_per_legal_turn | 87.1 (434) | 87.0 (463) | -0.1 [-2.9,+2.9] | -2.1 [-5.7,+1.5] (525/515) | -1.4 [-4.3,+1.4] |
| leader/Shao/target_is_current_attacker | 45.9 (377) | 51.7 (400) | +5.9 [-0.9,+12.2] | +3.3 [-3.0,+9.7] (418/439) | +0.6 [-5.0,+6.3] |
| response/spell_played_when_legal | 19.3 (367) | 22.1 (344) | +2.7 [-2.3,+8.4] | +2.6 [-3.1,+8.6] (382/394) | +0.5 [-2.9,+3.9] |
| response/any_nonblock_response_when_legal | 67.3 (615) | 73.5 (618) | **+6.1** [+0.5,+12.2] | +4.1 [-3.3,+11.3] (677/686) | -1.3 [-5.4,+2.8] |
| ikz/held_at_end_of_own_turn | 42.0 (1582) | 44.3 (1605) | +2.3 [-0.6,+5.0] | +0.6 [-3.9,+5.2] (1790/1648) | **+3.1** [+0.9,+5.4] |
| ikz/held_then_spent_in_opp_turn | 57.7 (665) | 58.6 (711) | +0.9 [-3.3,+4.9] | +7.4 [-0.9,+15.3] (889/829) | -2.9 [-6.3,+0.5] |
| water/bounce_play_per_legal_turn | 70.6 (722) | 65.3 (862) | **-5.3** [-9.3,-1.2] | -2.5 [-8.1,+3.3] (954/791) | -0.4 [-3.4,+2.6] |
| water/bounce_play_removes_opp_entity | 47.9 (541) | 45.5 (593) | -2.3 [-6.5,+1.7] | **-8.6** [-16.9,-0.1] (709/566) | +0.9 [-2.4,+4.3] |
| strategy/favorable_trade_taken_per_available_turn | 55.5 (584) | 57.0 (584) | +1.5 [-3.1,+6.7] | +2.8 [-1.9,+7.5] (611/620) | +1.5 [-2.9,+5.9] |
| strategy/face_target_share_when_both_legal | 59.5 (2053) | 61.1 (2058) | +1.7 [-2.2,+5.4] | +3.3 [-0.3,+6.9] (1970/2056) | -1.5 [-4.5,+1.5] |
| outcome/own_turns | 7.06 (224) | 7.17 (224) | +0.1 [-0.1,+0.3] | **-0.6** [-1.4,-0.1] (224/224) | +0.1 [-0.1,+0.2] |

#### EARTH (all suites pooled; u8223 → earth)

| metric | u8223 argmax (n) | spec argmax (n) | Δ argmax [95% CI] | Δ sampled [95% CI] (n u8223/spec) | u9305−u8223 argmax |
|---|---|---|---|---|---|
| outcome/win | 54.9 (288) | 59.0 (288) | +4.2 [-1.4,+10.1] | +3.5 [-2.1,+9.0] (288/288) | -1.7 [-6.6,+2.8] |
| strategy/favorable_trade_taken_per_available_turn | 55.1 (670) | 66.3 (725) | **+11.3** [+6.7,+15.5] | **+11.0** [+5.9,+16.1] (696/710) | +2.1 [-1.8,+6.1] |
| strategy/face_target_share_when_both_legal | 50.8 (1532) | 43.1 (1643) | **-7.8** [-12.0,-3.1] | **-8.4** [-12.6,-3.9] (1665/1636) | -0.5 [-4.3,+3.2] |
| gate/portal_per_legal_turn | 99.6 (1348) | 98.2 (1495) | **-1.4** [-2.2,-0.6] | **-1.2** [-1.9,-0.5] (1308/1509) | -0.3 [-0.8,+0.1] |
| gate/Stonehaven/portal_then_grant | 75.2 (1342) | 73.8 (1468) | -1.4 [-4.6,+1.8] | +1.7 [-1.2,+4.8] (1297/1478) | **-4.8** [-7.8,-2.0] |
| gate/Stonehaven/grant_then_block | 40.7 (1009) | 43.9 (1083) | +3.1 [-0.4,+6.8] | **+7.1** [+3.5,+10.6] (906/1058) | -1.0 [-4.1,+2.0] |
| response/defender_declared_when_legal | 55.2 (982) | 49.0 (1189) | **-6.2** [-10.3,-1.7] | **-5.6** [-9.8,-1.5] (937/1357) | +0.3 [-3.2,+3.8] |
| block/attacker_destroyed | 29.5 (543) | 32.2 (583) | +2.8 [-2.1,+7.7] | -1.1 [-5.5,+3.3] (475/612) | **+5.1** [+0.7,+9.6] |
| block/blocker_survives | 41.6 (543) | 36.2 (583) | **-5.4** [-10.4,-0.8] | -4.5 [-9.5,+0.7] (475/612) | -1.1 [-5.2,+2.7] |
| leader/use_per_legal_turn | 5.4 (2193) | 5.2 (2260) | -0.2 [-1.9,+1.7] | +1.5 [-0.7,+4.1] (2165/2263) | **-1.6** [-2.6,-0.4] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 16.0 (25) | +1.2 [-13.5,+15.5] | +14.7 [-6.1,+36.8] (41/26) | +3.0 [-11.7,+20.5] |
| earth/quicksand_cast_per_legal_turn | 47.5 (354) | 45.9 (270) | -1.5 [-7.7,+4.8] | **-8.9** [-15.3,-2.2] (250/291) | **-7.2** [-12.9,-1.8] |
| earth/quicksand_cast_multi_removal | 86.9 (168) | 84.7 (124) | -2.2 [-9.3,+4.7] | -1.3 [-9.8,+6.4] (134/130) | -1.3 [-7.2,+4.8] |
| heal/heal_spell_per_legal_turn_below_max_hp | 28.9 (757) | 27.3 (699) | -1.6 [-4.6,+1.3] | **-6.4** [-9.5,-3.1] (625/762) | -2.3 [-5.2,+0.6] |
| outcome/own_turns | 7.99 (288) | 8.16 (288) | +0.2 [-0.0,+0.4] | **+0.4** [+0.1,+0.6] (288/288) | -0.0 [-0.2,+0.2] |

### Per gate/leader context (leader use, portal use, main gate funnel)
| element | context | leader use / legal turn: u8223→spec argmax (n) | Δ argmax [CI] | Δ sampled [CI] | portal / legal turn Δ argmax | key gate/leader funnel (argmax u8223→spec, n) |
|---|---|---|---|---|---|---|
| FIRE | Ragefire/Kagoro | 24.1→24.4 (162/180) | +0.4 [-8.4,+8.9] | +9.0 [-0.7,+18.4] | -0.7 [-2.1,+0.0] | Ragefire/buffed_then_attacks_same_turn: 69.6→59.6 (46/52) |
| FIRE | Ragefire/Zero | 2.6→1.9 (1542/1570) | -0.7 [-1.6,+0.3] | **-1.5** [-2.7,-0.2] | **-1.1** [-2.0,-0.2] | Ragefire/buffed_then_attacks_same_turn: 43.1→53.6 (216/196) |
| FIRE | Rushfire/Kagoro | 87.5→81.6 (96/103) | -5.9 [-14.2,+2.6] | -2.4 [-11.8,+7.9] | -0.0 [-3.0,+2.9] | Rushfire/payload_then_attack_same_turn: 96.0→91.1 (175/179) |
| FIRE | Rushfire/Zero | 1.5→0.3 (1369/1339) | **-1.2** [-2.0,-0.4] | **-1.6** [-2.8,-0.7] | -0.2 [-1.1,+0.6] | Rushfire/payload_then_attack_same_turn: 98.7→95.3 (835/857) |
| LIGHTNING | Stormchain/Piko | never legal (0 rows) | – | – | -0.4 [-4.1,+3.2] | Stormchain/portal_with_eligible_weapon: 0.0→0.0 (90/66) |
| LIGHTNING | Stormchain/Raizan | never legal (0 rows) | – | – | +0.0 [+0.0,+0.0] | Stormchain/portal_with_eligible_weapon: 0.0→0.0 (320/210) |
| LIGHTNING | Surge/Piko | never legal (0 rows) | – | – | +0.1 [-1.6,+1.8] | Surge/portal_with_eligible_weapon: 12.4→33.3 (218/246) |
| LIGHTNING | Surge/Raizan | never legal (0 rows) | – | – | -0.3 [-0.8,+0.1] | Surge/portal_with_eligible_weapon: 61.1→72.4 (1132/1154) |
| WATER | Hydromancy/Benzai | 7.1→18.8 (14/16) | +11.6 [-10.9,+33.3] | **-48.0** [-65.9,-11.0] | **+2.5** [+0.4,+4.8] | Hydromancy/readied_then_spent_same_turn: 83.4→83.1 (235/237) |
| WATER | Hydromancy/Shao | 89.8→89.5 (420/447) | -0.3 [-2.9,+2.4] | +0.1 [-2.9,+2.9] | -0.4 [-2.0,+1.3] | Hydromancy/readied_then_spent_same_turn: 77.0→75.3 (803/829) |
| EARTH | Stonehaven/Bobu | 3.7→1.7 (1470/1502) | **-2.0** [-3.3,-0.7] | **-1.1** [-2.2,-0.0] | **-1.5** [-2.7,-0.4] | Stonehaven/grant_then_block: 38.7→43.3 (646/688) |
| EARTH | Stonehaven/Goro | 9.0→12.1 (723/758) | +3.1 [-1.1,+8.4] | **+6.1** [+0.7,+12.6] | **-1.2** [-2.2,-0.2] | Stonehaven/grant_then_block: 44.4→44.8 (363/395) |

Full tables (every metric × suite × context × mode, plus u9305): `tables.md`. Raw descriptors: `descriptors/<element>__<arm>__<mode>.json`; paired deltas: `descriptors/paired_comparisons.json`.

### Free-draft composition (u8223 → specialist, Δ)
| element | mode | spell % | weapon % | Normal % | unique cards | mean cost | max wJaccard to curated (same ctx/gate) |
|---|---|---|---|---|---|---|---|
| FIRE | argmax | 0.0→1.0 (**+1.0**) | 6.5→4.0 (**-2.5**) | 44.5→46.5 (**+2.0**) | 18.0→18.8 (**+0.8**) | 2.54→2.31 (**-0.23**) | 0.250→0.256 (+0.006) |
| FIRE | sample | 5.2→7.3 (**+2.1**) | 6.1→4.9 (**-1.2**) | 52.5→53.4 (**+0.9**) | 34.4→34.7 (**+0.3**) | 2.60→2.62 (**+0.02**) | 0.186→0.179 (**-0.007**) |
| LIGHTNING | argmax | 0.0→0.0 (+0.0) | 20.0→23.0 (**+3.0**) | 65.5→60.0 (**-5.5**) | 20.2→18.0 (**-2.2**) | 2.89→2.38 (**-0.51**) | 0.189→0.250 (**+0.061**) |
| LIGHTNING | sample | 5.1→4.3 (**-0.9**) | 17.3→18.6 (**+1.3**) | 65.1→62.8 (**-2.3**) | 33.1→32.5 (**-0.6**) | 2.73→2.70 (**-0.03**) | 0.132→0.154 (**+0.022**) |
| WATER | argmax | 0.0→0.0 (+0.0) | 7.0→12.0 (**+5.0**) | 83.0→82.0 (-1.0) | 23.5→18.0 (**-5.5**) | 2.31→2.06 (**-0.25**) | 0.152→0.134 (**-0.018**) |
| WATER | sample | 7.6→6.2 (**-1.4**) | 8.8→10.1 (**+1.3**) | 75.8→77.8 (**+2.0**) | 33.6→32.1 (**-1.5**) | 2.58→2.61 (**+0.03**) | 0.133→0.132 (-0.001) |
| EARTH | argmax | 15.0→5.0 (**-10.0**) | 10.0→0.0 (**-10.0**) | 58.0→55.0 (**-3.0**) | 17.0→20.5 (**+3.5**) | 3.36→3.42 (**+0.06**) | 0.254→0.191 (**-0.063**) |
| EARTH | sample | 9.3→8.8 (-0.5) | 10.5→5.4 (**-5.1**) | 77.0→70.0 (**-6.9**) | 33.1→34.5 (**+1.4**) | 2.98→3.29 (**+0.30**) | 0.122→0.127 (+0.005) |

- **Argmax drafts are deterministic.** Within-context similarity is 1.00 for every arm, so argmax draft results describe one deck per context. Their bootstrap CIs show only how the seed blocks are mixed. Sampled drafts (within-context weighted Jaccard about 0.3) are the distributional evidence.
- **Sibling-gate differentiation.** Under argmax, the Fire specialist differentiates Rushfire from Ragefire more than u8223 does: between-gate weighted Jaccard is 0.49–0.56, against 0.54–0.67 for u8223. Lightning Surge and Stormchain decks stay nearly identical for both (0.96 spec, 0.92–1.0 u8223). Leaders within a gate barely differ (0.89–1.00). Under sampling, between-context similarity equals within-context similarity for every arm (about 0.29–0.36), so sampled drafts show no context differentiation.
- **Absence check (deck lists).** Water argmax decks contain 0 spells for both u8223 and the specialist. The Earth specialist's argmax decks contain 0 weapons and 5% spells, against u8223's 10% weapons and 15% spells.

## Top behavioral changes (with example traces, `examples.txt`)
**Earth**
1. **More trading, less face.** The specialist takes a favorable trade on more of the turns where one is available: **+11.3pp** argmax and **+11.0pp** sampled. Its face share falls **−7.8/−8.4pp**. u9305 shows no such shift (+2.1, n.s.), so this comes from specialization rather than more training. Example: `fixed:EARTH:r0:cand13:opp0:seat0` step 40, AZK01-019 2/3 attacks STT04-003 2/1 although face was legal; the attacker survives and the target dies.
2. **Stonehaven grant → block.** **+7.1pp** sampled; +3.1 argmax, CI includes 0; Goro sampled **+12.6**. Example (same game, steps 16→23): portal AZK01-019, Stonehaven grants Defender at step 17, and the entity blocks an attack aimed at the leader at step 23.
3. **Fewer blocks overall and worse block outcomes.** Defender declared when legal **−6.2/−5.6pp**; blocker survival **−5.4pp** argmax; favorable block outcome −6.1 sampled. Games are longer: **+0.4** own turns sampled and +4.7 decisions per game.

**Fire**
1. **More trading, less face.** Favorable trade **+6.6/+8.9pp**; face share **−4.4/−6.1pp**; u9305 +2.4 (n.s.).
2. **More responses.** Response spell played when legal: **+20.4pp** sampled, +16.7 argmax (borderline). Any non-block response: **+13.3** sampled. Example: `fixed:FIRE:r0:cand0:opp11:seat1` step 74, Sundering Strike (AZK01-127) cast in response to an attack.
3. **Ragefire buff → attack.** **+7.1pp** argmax, +7.2 sampled. Heal spell use when below max HP: **+5.8/+7.9pp**. Example: `fixed:FIRE:r1:cand2:opp0:seat0` steps 24–26, Ragefire buffs STT04-003 from 2 to 3 attack and it attacks.

**Lightning**
1. **More Surge set-up.** Share of portals with a recoverable discard weapon: **+12.2pp** argmax (Surge/Piko 12→33%), +5.7 sampled. Game-level Surge recovery→attack conversion: **+4.2pp** argmax. Example: `fixed:LIGHTNING:r0:cand3:opp0:seat0` steps 15–17, Surge re-equips Lightning Shuriken to the leader and the leader attacks face.
2. **More aggressive.** Face share **+4.9pp**, attacks with an equipped attacker **+3.1pp**, favorable trades taken **−5.5pp** (argmax). This is the opposite direction from u9305, which takes **+7.1** more trades.
3. **More responses and reserved IKZ.** Non-block response when legal **+7.2pp**; IKZ held at end of turn then spent on the opponent's turn **+5.0pp** (argmax).

**Water**
1. **More responses.** Non-block response when legal **+6.1pp** argmax. Shao targets the current attacker +5.9pp (CI −0.9…+12.2). Example: `fixed:WATER:r0:cand11:opp11:seat0` steps 39–40, Shao reduces the attacking STT02-003 to 0 attack.
2. **Win rate and game length.** Win rate rises (**+6.7/+9.8pp**), most against Water (+14.3 argmax). Games are shorter sampled (**−0.6** turns). Hydromancy readied→spent is unchanged (78→77%).
3. **Draft.** In argmax drafts, weapons go 7→12%, unique cards 23.5→18, spells stay 0.

## What got worse
- **Fire, Rushfire payload → same-turn attack:** −3.6/−2.7pp (98→95%). **Engine caveat:** `src/abilities/cards/azk01_122.c` grants Charge but never sacrifices the payload at end of turn. In the traces, the payload is still on the board next turn in about 90% of cases. Holding the payload may therefore be rational in this engine. [INFERENCE] The card text says it must be sacrificed, so this behavior would not transfer to the real rules.
- **Zero leader use, already rare, fell further.** Per legal turn: Rushfire/Zero 1.5→0.3%, Ragefire/Zero sampled −1.5pp. Zero activations total 60 for the specialist against 124 for u8223 over 960 Zero-leader games per arm (both modes). The ping mostly feeds on-damage synergies (Spice, Fanatic Kindler, Pekiro/Cinderwake) rather than attacks. The `zero_before_attack` sequence stays ≤0.8% for both. Fire burn spells cast per legal turn: −2.8 sampled.
- **Earth:** fewer and worse blocks (above). Quicksand cast per legal turn **−8.9** sampled. Heal spells **−6.4** sampled. Bobu use per legal turn 3.7→1.7%.
- **Water:** bounce cards played per legal turn **−5.3** argmax; bounce removes an opposing entity **−8.6** sampled. Benzai leader use (sampled) 86→37% of legal turns (n=59/27).
- **Lightning:** fewer favorable trades (above); Surge destination attack −1.6 sampled.

## Unchanged structural gaps (validated by direct trace counts, `absence_checks.json`)
- **Lightning weapons almost always go to the leader.** Weapon-to-entity was legal on 3,000+ decisions per arm, yet entity attaches were 0/1,084 for the specialist and 3/1,035 for u8223 (argmax). As a result:
  - Raizan's and Piko's abilities had **0 legal decisions** for both the specialist and u8223 (u9305: 1).
  - Stormchain was never eligible: no Stormchain portal had an entity-equipped Garden weapon (0/410 u8223, 0/276 specialist).
  - These are true zeros, not telemetry gaps: they come from legal-row counts, not native counters.
- **Water argmax drafts contain 0 spells**, as in prior campaigns.
- **Earth and Fire leaders are used on about 5% of legal turns**, except Kagoro: 82–88% on Rushfire, 24% on Ragefire.

## Caveats and telemetry
- **Known false negatives avoided:**
  - Raizan's native effect counter is never used; Raizan is measured from legal rows and traces.
  - Surge/Stormchain use the corrected v3 helpers (immediate SELECT_TO_EQUIP sourced by the gate).
  - Devotion does not apply (its gate is not in the pool).
- **New bug fixed in my copy:** the frozen `annotate_turns` starts a new "turn" whenever the non-active seat answers a selection during MAIN. This split turns and broke same-turn funnels. `tools/analyze.py` ignores selection steps when it assigns turns.
- **Approximations:**
  - Shao target matching is by card code (duplicate codes are ambiguous).
  - Block outcome is read at the first post-combat state.
  - Favorable trade means the attacker kills the target and survives under simultaneous damage; Defender redirects are ignored.
- **Coverage limits:**
  - Sparse cells: Bobu (n≈20–50), Benzai (n≤59) and response windows for Fire/Lightning (n≈50–360).
  - Only u8223 was the opponent.
  - Fixed-suite Water and Earth decks cover a single gate (Hydromancy/Shao for Water; Stonehaven with Goro or Bobu for Earth).

## Commands (from the snapshot root)
```
WORKERS=12 SHARDS=4 bash train-specialist/analysis/behavior/tools/run_all.sh      # 80 shards → traces/*.jsonl (+ .done)
#   each shard: .venv/bin/python …/tools/run_traces.py --element E --arm {E|u8223|u9305} --mode {argmax|sample} --shards 4 --shard-index S --out …/traces/E__ARM__MODE__sS.jsonl
#   env: OMP_NUM_THREADS=1 PYTHONPATH=build/python/src:python/src:…/tools LD_LIBRARY_PATH=build/_deps/flecs_src-build; u9305 argmax only
.venv/bin/python train-specialist/analysis/behavior/tools/analyze.py --reps 2000   # descriptors/*.json, paired_comparisons.json
.venv/bin/python train-specialist/analysis/behavior/tools/report_tables.py         # tables.md
.venv/bin/python train-specialist/analysis/behavior/tools/absence_checks.py > train-specialist/analysis/behavior/absence_checks.json
.venv/bin/python train-specialist/analysis/behavior/tools/show_example.py FILE TASK_ID STEP --n N   # examples.txt
```
Config: `behavior_eval.ini`, a copy of the Water specialist ini with these changes: native=false, learner_element=none, prebuilt off, same-element matchup 0, league off.
