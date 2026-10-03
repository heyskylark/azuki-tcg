# Paired behavior tables (auto-generated)
Rates in %, (n) = opportunities (denominator) for that arm; delta = arm − u8223 in pp (own_turns/unique_cards/mean_cost in raw units); 95% block-bootstrap CI over seed blocks; ** = CI excludes 0.

## FIRE
### fire vs u8223 — argmax — all|ALL

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (20) | 0.0 (23) | +0.0 | [+0.0, +0.0] |
| draft/max_wjaccard_to_curated | 25.0 (192) | 25.6 (192) | +0.6 | [-0.2, +1.4] |
| draft/mean_cost | 2.54 (9600) | 2.31 (9600) | **-0.2** | [-0.3, -0.2] |
| draft/normal_share | 44.5 (9600) | 46.5 (9600) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (9600) | 1.0 (9600) | **+1.0** | [+0.8, +1.2] |
| draft/unique_cards | 18.00 (192) | 18.75 (192) | **+0.8** | [+0.4, +1.1] |
| draft/weapon_share | 6.5 (9600) | 4.0 (9600) | **-2.5** | [-2.7, -2.3] |
| fire/burn_cast_per_legal_turn | 28.5 (1724) | 27.0 (1740) | -1.6 | [-3.2, +0.0] |
| fire/burn_cast_then_opp_damage | 92.0 (654) | 93.2 (616) | +1.1 | [-1.7, +4.0] |
| fire/zero_ping_redirect_target_then_opp_damage | 100.0 (8) | 100.0 (6) | +0.0 | [+0.0, +0.0] |
| fire/zero_ping_target_is_redirect_card | 14.3 (56) | 19.4 (31) | +5.1 | [-10.7, +22.1] |
| gate/Ragefire/buffed_then_attacks_same_turn | 47.7 (262) | 54.8 (248) | **+7.1** | [+0.5, +13.8] |
| gate/Ragefire/portal_then_buff | 29.2 (897) | 31.5 (788) | +2.3 | [-1.1, +5.7] |
| gate/Rushfire/payload_then_attack_same_turn | 98.2 (1010) | 94.6 (1036) | **-3.6** | [-5.0, -2.2] |
| gate/Rushfire/portal_then_payload | 72.6 (1392) | 74.5 (1390) | +2.0 | [-0.4, +4.5] |
| gate/portal_per_legal_turn | 98.8 (2317) | 98.2 (2217) | -0.6 | [-1.2, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 31.4 (315) | **+5.8** | [+0.5, +11.0] |
| ikz/held_at_end_of_own_turn | 21.4 (3515) | 20.1 (3521) | **-1.3** | [-2.6, -0.0] |
| ikz/held_then_spent_in_opp_turn | 3.5 (751) | 4.5 (708) | +1.1 | [-0.2, +2.4] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.9 (123) | 96.9 (128) | +0.9 | [-3.7, +5.9] |
| leader/Zero/target_then_attacks_same_turn | 5.4 (56) | 12.5 (32) | +7.1 | [-4.4, +21.4] |
| leader/Zero/use_with_target | 93.3 (60) | 94.1 (34) | +0.8 | [-8.2, +9.4] |
| leader/use_per_legal_turn | 5.8 (3169) | 5.1 (3192) | -0.7 | [-1.6, +0.2] |
| outcome/own_turns | 6.10 (576) | 6.11 (576) | +0.0 | [-0.1, +0.1] |
| outcome/win | 70.8 (576) | 74.0 (576) | +3.1 | [-0.3, +6.4] |
| outcome/win_vs_EARTH | 66.7 (144) | 72.2 (144) | +5.6 | [-0.8, +12.1] |
| outcome/win_vs_FIRE | 59.7 (144) | 57.6 (144) | -2.1 | [-10.1, +6.0] |
| outcome/win_vs_LIGHTNING | 74.3 (144) | 84.7 (144) | **+10.4** | [+4.5, +16.7] |
| outcome/win_vs_WATER | 82.6 (144) | 81.2 (144) | -1.4 | [-6.5, +3.7] |
| response/any_nonblock_response_when_legal | 72.0 (107) | 83.0 (112) | +11.1 | [-0.4, +22.9] |
| response/defender_declared_when_legal | 28.6 (70) | 27.4 (84) | -1.2 | [-13.5, +12.4] |
| response/spell_played_when_legal | 47.3 (55) | 64.0 (50) | +16.7 | [+0.0, +34.7] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 15.8 (95) | 21.9 (96) | +6.1 | [-4.6, +17.2] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 10.5 (95) | 16.7 (96) | +6.1 | [-3.9, +16.2] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (288) | 100.0 (288) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 64.2 (288) | 64.2 (288) | +0.0 | [-2.1, +2.1] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.2 (480) | 0.2 (480) | +0.0 | [-0.6, +0.6] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (480) | 0.2 (480) | +0.2 | [+0.0, +0.6] |
| strategy/attacks_by_equipped_attacker | 1.2 (7490) | 0.7 (7591) | **-0.5** | [-0.8, -0.2] |
| strategy/face_target_share_when_both_legal | 68.4 (4095) | 64.0 (4042) | **-4.4** | [-6.6, -2.4] |
| strategy/favorable_trade_taken_per_available_turn | 51.9 (1000) | 58.5 (1015) | **+6.6** | [+3.2, +9.9] |
| strategy/spell_cast_per_legal_main_turn | 28.9 (1904) | 26.8 (2018) | **-2.1** | [-3.7, -0.5] |

### fire vs u8223 — argmax — all|Ragefire/Kagoro

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 23.6 (48) | 20.9 (48) | **-2.7** | [-2.7, -2.7] |
| draft/mean_cost | 2.68 (2400) | 2.62 (2400) | **-0.1** | [-0.1, -0.1] |
| draft/normal_share | 48.0 (2400) | 50.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 2.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/unique_cards | 17.00 (48) | 19.00 (48) | **+2.0** | [+2.0, +2.0] |
| draft/weapon_share | 10.0 (2400) | 8.0 (2400) | **-2.0** | [-2.0, -2.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 69.6 (46) | 59.6 (52) | -9.9 | [-30.5, +11.8] |
| gate/Ragefire/portal_then_buff | 33.6 (137) | 38.0 (137) | +4.4 | [-7.6, +15.6] |
| gate/portal_per_legal_turn | 100.0 (137) | 99.3 (138) | -0.7 | [-2.1, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | – (0) | 23.1 (39) | – | |
| ikz/held_at_end_of_own_turn | 9.6 (314) | 8.0 (327) | -1.6 | [-4.4, +0.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (30) | 0.0 (26) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 97.4 (39) | 90.9 (44) | -6.5 | [-15.7, +4.4] |
| leader/use_per_legal_turn | 24.1 (162) | 24.4 (180) | +0.4 | [-8.4, +8.9] |
| outcome/own_turns | 6.54 (48) | 6.81 (48) | +0.3 | [-0.0, +0.6] |
| outcome/win | 77.1 (48) | 68.8 (48) | -8.3 | [-22.9, +8.3] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 25.5 (47) | 37.5 (48) | +12.0 | [-7.5, +29.2] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 17.0 (47) | 27.1 (48) | +10.1 | [-7.3, +27.1] |
| strategy/attacks_by_equipped_attacker | 3.1 (650) | 3.2 (698) | +0.1 | [-1.4, +1.5] |
| strategy/face_target_share_when_both_legal | 69.5 (367) | 58.4 (375) | **-11.1** | [-18.2, -4.1] |
| strategy/favorable_trade_taken_per_available_turn | 46.5 (86) | 60.2 (88) | **+13.7** | [+1.2, +25.1] |
| strategy/spell_cast_per_legal_main_turn | – (0) | 19.1 (47) | – | |

### fire vs u8223 — argmax — all|Ragefire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (20) | 0.0 (23) | +0.0 | [+0.0, +0.0] |
| draft/max_wjaccard_to_curated | 23.6 (48) | 19.6 (48) | **-4.0** | [-4.0, -4.0] |
| draft/mean_cost | 2.68 (2400) | 2.64 (2400) | **-0.0** | [-0.0, -0.0] |
| draft/normal_share | 48.0 (2400) | 50.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 2.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/unique_cards | 17.00 (48) | 20.00 (48) | **+3.0** | [+3.0, +3.0] |
| draft/weapon_share | 10.0 (2400) | 8.0 (2400) | **-2.0** | [-2.0, -2.0] |
| fire/burn_cast_per_legal_turn | 27.4 (909) | 25.6 (926) | -1.8 | [-4.0, +0.5] |
| fire/burn_cast_then_opp_damage | 93.5 (352) | 92.7 (328) | -0.8 | [-5.0, +3.2] |
| fire/zero_ping_target_is_redirect_card | 11.1 (36) | 17.9 (28) | +6.7 | [-10.7, +25.4] |
| gate/Ragefire/buffed_then_attacks_same_turn | 43.1 (216) | 53.6 (196) | **+10.5** | [+3.5, +17.4] |
| gate/Ragefire/portal_then_buff | 28.4 (760) | 30.1 (651) | +1.7 | [-1.6, +5.0] |
| gate/portal_per_legal_turn | 99.7 (762) | 98.6 (660) | **-1.1** | [-2.0, -0.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 32.6 (276) | **+7.0** | [+1.6, +12.3] |
| ikz/held_at_end_of_own_turn | 21.4 (1559) | 20.7 (1582) | -0.8 | [-2.9, +1.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (334) | 0.0 (327) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 8.3 (36) | 7.1 (28) | -1.2 | [-12.2, +11.2] |
| leader/Zero/use_with_target | 90.0 (40) | 93.3 (30) | +3.3 | [-7.1, +14.5] |
| leader/use_per_legal_turn | 2.6 (1542) | 1.9 (1570) | -0.7 | [-1.6, +0.3] |
| outcome/own_turns | 6.50 (240) | 6.59 (240) | +0.1 | [-0.0, +0.2] |
| outcome/win | 49.2 (240) | 55.8 (240) | **+6.7** | [+0.8, +12.1] |
| response/any_nonblock_response_when_legal | 94.4 (18) | 100.0 (23) | +5.6 | [+0.0, +16.7] |
| response/defender_declared_when_legal | 28.6 (70) | 27.4 (84) | -1.2 | [-13.1, +12.5] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.4 (240) | 0.0 (240) | -0.4 | [-1.2, +0.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (240) | 0.0 (240) | +0.0 | [+0.0, +0.0] |
| strategy/attacks_by_equipped_attacker | 1.0 (2729) | 0.3 (2750) | **-0.7** | [-1.1, -0.3] |
| strategy/face_target_share_when_both_legal | 63.5 (1809) | 60.1 (1800) | **-3.4** | [-6.0, -1.1] |
| strategy/favorable_trade_taken_per_available_turn | 54.8 (496) | 61.0 (525) | **+6.1** | [+1.9, +10.5] |
| strategy/spell_cast_per_legal_main_turn | 28.3 (1089) | 25.9 (1157) | **-2.4** | [-4.5, -0.3] |

### fire vs u8223 — argmax — all|Rushfire/Kagoro

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 27.9 (48) | 32.5 (48) | **+4.6** | [+4.6, +4.6] |
| draft/mean_cost | 2.32 (2400) | 1.90 (2400) | **-0.4** | [-0.4, -0.4] |
| draft/normal_share | 42.0 (2400) | 44.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 2.0 (2400) | 0.0 (2400) | **-2.0** | [-2.0, -2.0] |
| gate/Rushfire/payload_then_attack_same_turn | 96.0 (175) | 91.1 (179) | **-4.9** | [-8.3, -1.4] |
| gate/Rushfire/portal_then_payload | 75.1 (233) | 77.8 (230) | +2.7 | [-4.0, +9.0] |
| gate/portal_per_legal_turn | 97.9 (238) | 97.9 (235) | -0.0 | [-3.0, +2.9] |
| ikz/held_at_end_of_own_turn | 14.2 (253) | 13.2 (250) | -1.0 | [-5.3, +3.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (36) | 0.0 (33) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.2 (84) | 100.0 (84) | **+4.8** | [+1.1, +9.2] |
| leader/use_per_legal_turn | 87.5 (96) | 81.6 (103) | -5.9 | [-14.2, +2.6] |
| outcome/own_turns | 5.27 (48) | 5.21 (48) | -0.1 | [-0.3, +0.2] |
| outcome/win | 95.8 (48) | 97.9 (48) | +2.1 | [-4.2, +8.3] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 6.2 (48) | 6.2 (48) | +0.0 | [-12.5, +10.4] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 4.2 (48) | 6.2 (48) | +2.1 | [-10.4, +12.5] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 100.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 62.5 (48) | 66.7 (48) | +4.2 | [-4.2, +12.5] |
| strategy/attacks_by_equipped_attacker | 1.0 (720) | 0.0 (716) | **-1.0** | [-1.6, -0.4] |
| strategy/face_target_share_when_both_legal | 84.0 (300) | 76.4 (271) | **-7.6** | [-14.0, -1.9] |
| strategy/favorable_trade_taken_per_available_turn | 34.5 (55) | 44.9 (49) | +10.4 | [-6.0, +26.2] |

### fire vs u8223 — argmax — all|Rushfire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 25.0 (48) | 29.4 (48) | **+4.4** | [+4.4, +4.4] |
| draft/mean_cost | 2.48 (2400) | 2.10 (2400) | **-0.4** | [-0.4, -0.4] |
| draft/normal_share | 40.0 (2400) | 42.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 18.00 (48) | 18.00 (48) | +0.0 | [+0.0, +0.0] |
| draft/weapon_share | 4.0 (2400) | 0.0 (2400) | **-4.0** | [-4.0, -4.0] |
| fire/burn_cast_per_legal_turn | 29.8 (815) | 28.5 (814) | -1.3 | [-3.5, +0.9] |
| fire/burn_cast_then_opp_damage | 90.4 (302) | 93.8 (288) | +3.4 | [-0.4, +7.1] |
| fire/zero_ping_target_is_redirect_card | 20.0 (20) | 33.3 (3) | +13.3 | [-31.6, +81.8] |
| gate/Rushfire/payload_then_attack_same_turn | 98.7 (835) | 95.3 (857) | **-3.4** | [-4.9, -1.9] |
| gate/Rushfire/portal_then_payload | 72.0 (1159) | 73.9 (1160) | +1.8 | [-0.7, +4.4] |
| gate/portal_per_legal_turn | 98.2 (1180) | 98.0 (1184) | -0.2 | [-1.1, +0.6] |
| ikz/held_at_end_of_own_turn | 25.3 (1389) | 23.6 (1362) | -1.6 | [-3.7, +0.3] |
| ikz/held_then_spent_in_opp_turn | 7.4 (351) | 9.9 (322) | +2.5 | [-0.2, +5.4] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (20) | 50.0 (4) | +50.0 | [+0.0, +100.0] |
| leader/Zero/use_with_target | 100.0 (20) | 100.0 (4) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 1.5 (1369) | 0.3 (1339) | **-1.2** | [-2.0, -0.4] |
| outcome/own_turns | 5.79 (240) | 5.67 (240) | **-0.1** | [-0.2, -0.0] |
| outcome/win | 86.2 (240) | 88.3 (240) | +2.1 | [-2.1, +6.7] |
| response/any_nonblock_response_when_legal | 67.0 (88) | 78.4 (88) | +11.4 | [-1.2, +25.4] |
| response/spell_played_when_legal | 47.3 (55) | 64.0 (50) | **+16.7** | [+1.1, +35.1] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (240) | 100.0 (240) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 64.6 (240) | 63.7 (240) | -0.8 | [-2.9, +0.8] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (240) | 0.4 (240) | +0.4 | [+0.0, +1.2] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (240) | 0.4 (240) | +0.4 | [+0.0, +1.2] |
| strategy/attacks_by_equipped_attacker | 1.0 (3391) | 0.6 (3427) | **-0.4** | [-0.7, -0.0] |
| strategy/face_target_share_when_both_legal | 70.9 (1619) | 67.7 (1596) | -3.2 | [-6.8, +0.5] |
| strategy/favorable_trade_taken_per_available_turn | 51.8 (363) | 56.4 (353) | +4.6 | [-1.2, +10.3] |
| strategy/spell_cast_per_legal_main_turn | 29.8 (815) | 28.5 (814) | -1.3 | [-3.5, +0.9] |

### fire vs u8223 — argmax — fixed|ALL

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (20) | 0.0 (23) | +0.0 | [+0.0, +0.0] |
| fire/burn_cast_per_legal_turn | 28.5 (1724) | 27.0 (1740) | -1.6 | [-3.1, +0.1] |
| fire/burn_cast_then_opp_damage | 92.0 (654) | 93.2 (616) | +1.1 | [-1.7, +4.0] |
| fire/zero_ping_redirect_target_then_opp_damage | 100.0 (8) | 100.0 (6) | +0.0 | [+0.0, +0.0] |
| fire/zero_ping_target_is_redirect_card | 23.5 (34) | 23.1 (26) | -0.5 | [-22.4, +23.2] |
| gate/Ragefire/buffed_then_attacks_same_turn | 39.5 (167) | 50.0 (148) | **+10.5** | [+2.8, +18.4] |
| gate/Ragefire/portal_then_buff | 28.3 (591) | 29.0 (510) | +0.8 | [-2.7, +4.1] |
| gate/Rushfire/payload_then_attack_same_turn | 98.5 (677) | 95.9 (683) | **-2.6** | [-4.3, -1.0] |
| gate/Rushfire/portal_then_payload | 75.4 (898) | 75.7 (902) | +0.3 | [-2.3, +3.0] |
| gate/portal_per_legal_turn | 98.9 (1506) | 98.3 (1436) | -0.5 | [-1.3, +0.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 35.5 (231) | **+9.9** | [+4.6, +15.3] |
| ikz/held_at_end_of_own_turn | 23.3 (2325) | 22.2 (2326) | -1.1 | [-2.5, +0.4] |
| ikz/held_then_spent_in_opp_turn | 4.8 (542) | 6.2 (517) | +1.4 | [-0.3, +3.2] |
| leader/Zero/target_then_attacks_same_turn | 8.8 (34) | 14.8 (27) | +6.0 | [-8.8, +25.3] |
| leader/Zero/use_with_target | 94.4 (36) | 93.1 (29) | -1.3 | [-10.7, +7.4] |
| leader/use_per_legal_turn | 1.6 (2291) | 1.3 (2291) | -0.3 | [-1.0, +0.3] |
| outcome/own_turns | 6.05 (384) | 6.06 (384) | +0.0 | [-0.1, +0.1] |
| outcome/win | 66.1 (384) | 70.1 (384) | +3.9 | [-0.3, +8.1] |
| outcome/win_vs_EARTH | 57.3 (96) | 64.6 (96) | +7.3 | [-1.0, +15.6] |
| outcome/win_vs_FIRE | 53.1 (96) | 56.2 (96) | +3.1 | [-7.3, +13.5] |
| outcome/win_vs_LIGHTNING | 74.0 (96) | 81.2 (96) | +7.3 | [+0.0, +14.6] |
| outcome/win_vs_WATER | 80.2 (96) | 78.1 (96) | -2.1 | [-8.7, +4.5] |
| response/any_nonblock_response_when_legal | 71.7 (106) | 82.9 (111) | **+11.2** | [+0.0, +23.1] |
| response/defender_declared_when_legal | 28.6 (70) | 27.4 (84) | -1.2 | [-12.6, +12.6] |
| response/spell_played_when_legal | 47.3 (55) | 64.0 (50) | +16.7 | [-0.0, +34.9] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (192) | 100.0 (192) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 65.6 (192) | 65.1 (192) | -0.5 | [-2.5, +1.1] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.3 (384) | 0.3 (384) | +0.0 | [-0.8, +0.8] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (384) | 0.3 (384) | +0.3 | [+0.0, +0.8] |
| strategy/attacks_by_equipped_attacker | 0.5 (4788) | 0.5 (4844) | -0.0 | [-0.1, +0.1] |
| strategy/face_target_share_when_both_legal | 66.3 (2794) | 62.6 (2778) | **-3.7** | [-6.1, -1.3] |
| strategy/favorable_trade_taken_per_available_turn | 57.3 (688) | 62.5 (715) | **+5.3** | [+1.7, +8.9] |
| strategy/spell_cast_per_legal_main_turn | 28.9 (1904) | 27.4 (1909) | -1.5 | [-3.0, +0.0] |

### fire vs u8223 — argmax — fixed|Ragefire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (20) | 0.0 (23) | +0.0 | [+0.0, +0.0] |
| fire/burn_cast_per_legal_turn | 27.4 (909) | 25.6 (926) | -1.8 | [-4.2, +0.5] |
| fire/burn_cast_then_opp_damage | 93.5 (352) | 92.7 (328) | -0.8 | [-4.8, +3.4] |
| fire/zero_ping_target_is_redirect_card | 16.7 (24) | 21.7 (23) | +5.1 | [-19.8, +30.4] |
| gate/Ragefire/buffed_then_attacks_same_turn | 39.5 (167) | 50.0 (148) | **+10.5** | [+3.0, +18.3] |
| gate/Ragefire/portal_then_buff | 28.3 (591) | 29.0 (510) | +0.8 | [-2.7, +4.3] |
| gate/portal_per_legal_turn | 99.7 (593) | 98.8 (516) | -0.8 | [-1.9, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 35.5 (231) | **+9.9** | [+4.4, +15.1] |
| ikz/held_at_end_of_own_turn | 22.1 (1231) | 21.1 (1243) | -1.0 | [-3.1, +1.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (272) | 0.0 (262) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 12.5 (24) | 8.7 (23) | -3.8 | [-17.7, +13.1] |
| leader/Zero/use_with_target | 92.3 (26) | 92.0 (25) | -0.3 | [-11.1, +11.8] |
| leader/use_per_legal_turn | 2.1 (1217) | 2.0 (1231) | -0.1 | [-1.2, +1.1] |
| outcome/own_turns | 6.41 (192) | 6.47 (192) | +0.1 | [-0.0, +0.2] |
| outcome/win | 46.4 (192) | 53.6 (192) | **+7.3** | [+1.0, +13.5] |
| response/any_nonblock_response_when_legal | 94.4 (18) | 100.0 (23) | +5.6 | [+0.0, +17.6] |
| response/defender_declared_when_legal | 28.6 (70) | 27.4 (84) | -1.2 | [-13.4, +12.1] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.5 (192) | 0.0 (192) | -0.5 | [-1.6, +0.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (192) | 0.0 (192) | +0.0 | [+0.0, +0.0] |
| strategy/attacks_by_equipped_attacker | 0.0 (2135) | 0.0 (2168) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 62.3 (1484) | 58.8 (1449) | **-3.5** | [-6.3, -0.7] |
| strategy/favorable_trade_taken_per_available_turn | 57.4 (404) | 64.5 (422) | **+7.0** | [+2.4, +11.6] |
| strategy/spell_cast_per_legal_main_turn | 28.3 (1089) | 26.6 (1095) | -1.7 | [-3.9, +0.4] |

### fire vs u8223 — argmax — fixed|Rushfire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| fire/burn_cast_per_legal_turn | 29.8 (815) | 28.5 (814) | -1.3 | [-3.4, +0.9] |
| fire/burn_cast_then_opp_damage | 90.4 (302) | 93.8 (288) | +3.4 | [-0.3, +6.9] |
| fire/zero_ping_target_is_redirect_card | 40.0 (10) | 33.3 (3) | -6.7 | [-60.0, +55.6] |
| gate/Rushfire/payload_then_attack_same_turn | 98.5 (677) | 95.9 (683) | **-2.6** | [-4.3, -1.0] |
| gate/Rushfire/portal_then_payload | 75.4 (898) | 75.7 (902) | +0.3 | [-2.3, +3.0] |
| gate/portal_per_legal_turn | 98.4 (913) | 98.0 (920) | -0.3 | [-1.3, +0.6] |
| ikz/held_at_end_of_own_turn | 24.7 (1094) | 23.5 (1083) | -1.1 | [-3.3, +1.0] |
| ikz/held_then_spent_in_opp_turn | 9.6 (270) | 12.5 (255) | +2.9 | [-0.6, +6.7] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (10) | 50.0 (4) | +50.0 | [+0.0, +100.0] |
| leader/Zero/use_with_target | 100.0 (10) | 100.0 (4) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 0.9 (1074) | 0.4 (1060) | -0.6 | [-1.1, +0.0] |
| outcome/own_turns | 5.70 (192) | 5.64 (192) | -0.1 | [-0.2, +0.0] |
| outcome/win | 85.9 (192) | 86.5 (192) | +0.5 | [-4.7, +5.7] |
| response/any_nonblock_response_when_legal | 67.0 (88) | 78.4 (88) | +11.4 | [-1.4, +25.0] |
| response/spell_played_when_legal | 47.3 (55) | 64.0 (50) | **+16.7** | [+0.8, +34.8] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (192) | 100.0 (192) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 65.6 (192) | 65.1 (192) | -0.5 | [-2.6, +1.0] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (192) | 0.5 (192) | +0.5 | [+0.0, +1.6] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (192) | 0.5 (192) | +0.5 | [+0.0, +1.6] |
| strategy/attacks_by_equipped_attacker | 0.8 (2653) | 0.8 (2676) | -0.0 | [-0.3, +0.2] |
| strategy/face_target_share_when_both_legal | 70.8 (1310) | 66.7 (1329) | **-4.1** | [-8.0, -0.2] |
| strategy/favorable_trade_taken_per_available_turn | 57.0 (284) | 59.7 (293) | +2.7 | [-3.4, +8.4] |
| strategy/spell_cast_per_legal_main_turn | 29.8 (815) | 28.5 (814) | -1.3 | [-3.4, +0.9] |

### fire vs u8223 — argmax — free_draft|ALL

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 25.0 (192) | 25.6 (192) | +0.6 | [-0.2, +1.4] |
| draft/mean_cost | 2.54 (9600) | 2.31 (9600) | **-0.2** | [-0.3, -0.2] |
| draft/normal_share | 44.5 (9600) | 46.5 (9600) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (9600) | 1.0 (9600) | **+1.0** | [+0.8, +1.2] |
| draft/unique_cards | 18.00 (192) | 18.75 (192) | **+0.8** | [+0.4, +1.2] |
| draft/weapon_share | 6.5 (9600) | 4.0 (9600) | **-2.5** | [-2.7, -2.3] |
| fire/zero_ping_target_is_redirect_card | 0.0 (22) | 0.0 (5) | +0.0 | [+0.0, +0.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 62.1 (95) | 62.0 (100) | -0.1 | [-14.6, +13.4] |
| gate/Ragefire/portal_then_buff | 31.0 (306) | 36.0 (278) | +4.9 | [-2.1, +12.2] |
| gate/Rushfire/payload_then_attack_same_turn | 97.6 (333) | 92.1 (353) | **-5.5** | [-8.0, -3.1] |
| gate/Rushfire/portal_then_payload | 67.4 (494) | 72.3 (488) | **+4.9** | [+0.3, +9.6] |
| gate/portal_per_legal_turn | 98.6 (811) | 98.1 (781) | -0.6 | [-1.8, +0.6] |
| heal/heal_spell_per_legal_turn_below_max_hp | – (0) | 20.2 (84) | – | |
| ikz/held_at_end_of_own_turn | 17.6 (1190) | 16.0 (1195) | -1.6 | [-3.9, +0.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (209) | 0.0 (191) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.9 (123) | 96.9 (128) | +0.9 | [-3.8, +5.9] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (22) | 0.0 (5) | +0.0 | [+0.0, +0.0] |
| leader/Zero/use_with_target | 91.7 (24) | 100.0 (5) | +8.3 | [+0.0, +23.8] |
| leader/use_per_legal_turn | 16.7 (878) | 14.8 (901) | -2.0 | [-4.5, +0.4] |
| outcome/own_turns | 6.20 (192) | 6.22 (192) | +0.0 | [-0.1, +0.2] |
| outcome/win | 80.2 (192) | 81.8 (192) | +1.6 | [-4.7, +7.8] |
| outcome/win_vs_EARTH | 85.4 (48) | 87.5 (48) | +2.1 | [-9.1, +14.3] |
| outcome/win_vs_FIRE | 72.9 (48) | 60.4 (48) | -12.5 | [-26.5, +2.2] |
| outcome/win_vs_LIGHTNING | 75.0 (48) | 91.7 (48) | **+16.7** | [+6.5, +28.3] |
| outcome/win_vs_WATER | 87.5 (48) | 87.5 (48) | +0.0 | [-8.7, +8.3] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 15.8 (95) | 21.9 (96) | +6.1 | [-5.6, +17.6] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 10.5 (95) | 16.7 (96) | +6.1 | [-4.8, +16.3] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (96) | 100.0 (96) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 61.5 (96) | 62.5 (96) | +1.0 | [-4.3, +6.5] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (96) | 0.0 (96) | +0.0 | [+0.0, +0.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (96) | 0.0 (96) | +0.0 | [+0.0, +0.0] |
| strategy/attacks_by_equipped_attacker | 2.5 (2702) | 1.1 (2747) | **-1.4** | [-2.0, -0.7] |
| strategy/face_target_share_when_both_legal | 73.1 (1301) | 67.1 (1264) | **-6.0** | [-9.8, -1.7] |
| strategy/favorable_trade_taken_per_available_turn | 40.1 (312) | 49.0 (300) | **+8.9** | [+1.6, +15.9] |
| strategy/spell_cast_per_legal_main_turn | – (0) | 16.5 (109) | – | |

### fire vs u8223 — argmax — free_draft|Ragefire/Kagoro

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 23.6 (48) | 20.9 (48) | **-2.7** | [-2.7, -2.7] |
| draft/mean_cost | 2.68 (2400) | 2.62 (2400) | **-0.1** | [-0.1, -0.1] |
| draft/normal_share | 48.0 (2400) | 50.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 2.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/unique_cards | 17.00 (48) | 19.00 (48) | **+2.0** | [+2.0, +2.0] |
| draft/weapon_share | 10.0 (2400) | 8.0 (2400) | **-2.0** | [-2.0, -2.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 69.6 (46) | 59.6 (52) | -9.9 | [-30.5, +11.8] |
| gate/Ragefire/portal_then_buff | 33.6 (137) | 38.0 (137) | +4.4 | [-7.6, +15.6] |
| gate/portal_per_legal_turn | 100.0 (137) | 99.3 (138) | -0.7 | [-2.1, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | – (0) | 23.1 (39) | – | |
| ikz/held_at_end_of_own_turn | 9.6 (314) | 8.0 (327) | -1.6 | [-4.4, +0.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (30) | 0.0 (26) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 97.4 (39) | 90.9 (44) | -6.5 | [-15.7, +4.4] |
| leader/use_per_legal_turn | 24.1 (162) | 24.4 (180) | +0.4 | [-8.4, +8.9] |
| outcome/own_turns | 6.54 (48) | 6.81 (48) | +0.3 | [-0.0, +0.6] |
| outcome/win | 77.1 (48) | 68.8 (48) | -8.3 | [-22.9, +8.3] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 25.5 (47) | 37.5 (48) | +12.0 | [-7.5, +29.2] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 17.0 (47) | 27.1 (48) | +10.1 | [-7.3, +27.1] |
| strategy/attacks_by_equipped_attacker | 3.1 (650) | 3.2 (698) | +0.1 | [-1.4, +1.5] |
| strategy/face_target_share_when_both_legal | 69.5 (367) | 58.4 (375) | **-11.1** | [-18.2, -4.1] |
| strategy/favorable_trade_taken_per_available_turn | 46.5 (86) | 60.2 (88) | **+13.7** | [+1.2, +25.1] |
| strategy/spell_cast_per_legal_main_turn | – (0) | 19.1 (47) | – | |

### fire vs u8223 — argmax — free_draft|Ragefire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 23.6 (48) | 19.6 (48) | **-4.0** | [-4.0, -4.0] |
| draft/mean_cost | 2.68 (2400) | 2.64 (2400) | **-0.0** | [-0.0, -0.0] |
| draft/normal_share | 48.0 (2400) | 50.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 2.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/unique_cards | 17.00 (48) | 20.00 (48) | **+3.0** | [+3.0, +3.0] |
| draft/weapon_share | 10.0 (2400) | 8.0 (2400) | **-2.0** | [-2.0, -2.0] |
| fire/zero_ping_target_is_redirect_card | 0.0 (12) | 0.0 (5) | +0.0 | [+0.0, +0.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 55.1 (49) | 64.6 (48) | +9.5 | [-9.0, +25.5] |
| gate/Ragefire/portal_then_buff | 29.0 (169) | 34.0 (141) | +5.0 | [-2.6, +13.6] |
| gate/portal_per_legal_turn | 100.0 (169) | 97.9 (144) | -2.1 | [-4.4, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | – (0) | 17.8 (45) | – | |
| ikz/held_at_end_of_own_turn | 18.9 (328) | 19.2 (339) | +0.3 | [-5.3, +5.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (62) | 0.0 (65) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (12) | 0.0 (5) | +0.0 | [+0.0, +0.0] |
| leader/Zero/use_with_target | 85.7 (14) | 100.0 (5) | +14.3 | [+0.0, +33.3] |
| leader/use_per_legal_turn | 4.3 (325) | 1.5 (339) | **-2.8** | [-4.8, -0.7] |
| outcome/own_turns | 6.83 (48) | 7.06 (48) | +0.2 | [-0.1, +0.5] |
| outcome/win | 60.4 (48) | 64.6 (48) | +4.2 | [-10.4, +16.7] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (48) | 0.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (48) | 0.0 (48) | +0.0 | [+0.0, +0.0] |
| strategy/attacks_by_equipped_attacker | 4.7 (594) | 1.5 (582) | **-3.2** | [-4.7, -1.8] |
| strategy/face_target_share_when_both_legal | 68.9 (325) | 65.2 (351) | -3.7 | [-10.1, +3.3] |
| strategy/favorable_trade_taken_per_available_turn | 43.5 (92) | 46.6 (103) | +3.1 | [-9.8, +16.5] |
| strategy/spell_cast_per_legal_main_turn | – (0) | 14.5 (62) | – | |

### fire vs u8223 — argmax — free_draft|Rushfire/Kagoro

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 27.9 (48) | 32.5 (48) | **+4.6** | [+4.6, +4.6] |
| draft/mean_cost | 2.32 (2400) | 1.90 (2400) | **-0.4** | [-0.4, -0.4] |
| draft/normal_share | 42.0 (2400) | 44.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 2.0 (2400) | 0.0 (2400) | **-2.0** | [-2.0, -2.0] |
| gate/Rushfire/payload_then_attack_same_turn | 96.0 (175) | 91.1 (179) | **-4.9** | [-8.3, -1.4] |
| gate/Rushfire/portal_then_payload | 75.1 (233) | 77.8 (230) | +2.7 | [-4.0, +9.0] |
| gate/portal_per_legal_turn | 97.9 (238) | 97.9 (235) | -0.0 | [-3.0, +2.9] |
| ikz/held_at_end_of_own_turn | 14.2 (253) | 13.2 (250) | -1.0 | [-5.3, +3.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (36) | 0.0 (33) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.2 (84) | 100.0 (84) | **+4.8** | [+1.1, +9.2] |
| leader/use_per_legal_turn | 87.5 (96) | 81.6 (103) | -5.9 | [-14.2, +2.6] |
| outcome/own_turns | 5.27 (48) | 5.21 (48) | -0.1 | [-0.3, +0.2] |
| outcome/win | 95.8 (48) | 97.9 (48) | +2.1 | [-4.2, +8.3] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 6.2 (48) | 6.2 (48) | +0.0 | [-12.5, +10.4] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 4.2 (48) | 6.2 (48) | +2.1 | [-10.4, +12.5] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 100.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 62.5 (48) | 66.7 (48) | +4.2 | [-4.2, +12.5] |
| strategy/attacks_by_equipped_attacker | 1.0 (720) | 0.0 (716) | **-1.0** | [-1.6, -0.4] |
| strategy/face_target_share_when_both_legal | 84.0 (300) | 76.4 (271) | **-7.6** | [-14.0, -1.9] |
| strategy/favorable_trade_taken_per_available_turn | 34.5 (55) | 44.9 (49) | +10.4 | [-6.0, +26.2] |

### fire vs u8223 — argmax — free_draft|Rushfire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 25.0 (48) | 29.4 (48) | **+4.4** | [+4.4, +4.4] |
| draft/mean_cost | 2.48 (2400) | 2.10 (2400) | **-0.4** | [-0.4, -0.4] |
| draft/normal_share | 40.0 (2400) | 42.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 18.00 (48) | 18.00 (48) | +0.0 | [+0.0, +0.0] |
| draft/weapon_share | 4.0 (2400) | 0.0 (2400) | **-4.0** | [-4.0, -4.0] |
| fire/zero_ping_target_is_redirect_card | 0.0 (10) | – (0) | – | |
| gate/Rushfire/payload_then_attack_same_turn | 99.4 (158) | 93.1 (174) | **-6.3** | [-9.6, -2.7] |
| gate/Rushfire/portal_then_payload | 60.5 (261) | 67.4 (258) | **+6.9** | [+0.1, +13.2] |
| gate/portal_per_legal_turn | 97.8 (267) | 97.7 (264) | -0.0 | [-2.1, +2.1] |
| ikz/held_at_end_of_own_turn | 27.5 (295) | 24.0 (279) | -3.4 | [-8.3, +1.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (81) | 0.0 (67) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (10) | – (0) | – | |
| leader/Zero/use_with_target | 100.0 (10) | – (0) | – | |
| leader/use_per_legal_turn | 3.4 (295) | 0.0 (279) | **-3.4** | [-6.9, -0.7] |
| outcome/own_turns | 6.15 (48) | 5.81 (48) | **-0.3** | [-0.6, -0.0] |
| outcome/win | 87.5 (48) | 95.8 (48) | **+8.3** | [+2.1, +16.7] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 100.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 60.4 (48) | 58.3 (48) | -2.1 | [-10.4, +4.2] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (48) | 0.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (48) | 0.0 (48) | +0.0 | [+0.0, +0.0] |
| strategy/attacks_by_equipped_attacker | 1.6 (738) | 0.0 (751) | **-1.6** | [-2.9, -0.7] |
| strategy/face_target_share_when_both_legal | 71.2 (309) | 72.3 (267) | +1.1 | [-7.4, +11.0] |
| strategy/favorable_trade_taken_per_available_turn | 32.9 (79) | 40.0 (60) | +7.1 | [-7.9, +22.4] |

### fire vs u8223 — sample — all|ALL

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 12.9 (31) | 7.7 (39) | -5.2 | [-22.1, +8.2] |
| draft/max_wjaccard_to_curated | 18.6 (192) | 17.9 (192) | **-0.7** | [-1.0, -0.4] |
| draft/mean_cost | 2.60 (9600) | 2.62 (9600) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 52.5 (9600) | 53.4 (9600) | **+0.9** | [+0.4, +1.4] |
| draft/spell_share | 5.2 (9600) | 7.3 (9600) | **+2.1** | [+1.7, +2.5] |
| draft/unique_cards | 34.44 (192) | 34.72 (192) | **+0.3** | [+0.1, +0.5] |
| draft/weapon_share | 6.1 (9600) | 4.9 (9600) | **-1.2** | [-1.4, -1.0] |
| fire/burn_cast_per_legal_turn | 28.1 (1969) | 25.2 (1985) | **-2.8** | [-4.5, -1.1] |
| fire/burn_cast_then_opp_damage | 94.5 (715) | 93.3 (623) | -1.3 | [-3.7, +1.1] |
| fire/zero_ping_target_is_redirect_card | 9.0 (67) | 11.1 (27) | +2.2 | [-12.5, +17.2] |
| gate/Ragefire/buffed_then_attacks_same_turn | 44.1 (227) | 51.2 (203) | +7.2 | [-0.7, +14.8] |
| gate/Ragefire/portal_then_buff | 25.8 (879) | 25.9 (784) | +0.1 | [-3.4, +3.5] |
| gate/Rushfire/payload_then_attack_same_turn | 97.9 (982) | 95.2 (977) | **-2.7** | [-4.1, -1.4] |
| gate/Rushfire/portal_then_payload | 69.1 (1421) | 69.0 (1415) | -0.1 | [-3.0, +2.8] |
| gate/portal_per_legal_turn | 99.0 (2324) | 98.3 (2238) | **-0.7** | [-1.4, -0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 26.7 (345) | 34.6 (295) | **+7.9** | [+2.5, +13.4] |
| ikz/held_at_end_of_own_turn | 22.5 (3581) | 21.1 (3587) | **-1.4** | [-2.8, -0.0] |
| ikz/held_then_spent_in_opp_turn | 3.5 (805) | 4.5 (756) | +1.0 | [-0.3, +2.4] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 97.1 (138) | 98.6 (147) | +1.5 | [-1.1, +4.4] |
| leader/Zero/target_then_attacks_same_turn | 1.5 (68) | 10.7 (28) | +9.2 | [-1.6, +21.8] |
| leader/Zero/use_with_target | 91.9 (74) | 96.6 (29) | +4.7 | [-4.4, +12.5] |
| leader/use_per_legal_turn | 6.6 (3198) | 5.5 (3206) | **-1.1** | [-2.2, -0.1] |
| outcome/own_turns | 6.22 (576) | 6.23 (576) | +0.0 | [-0.1, +0.1] |
| outcome/win | 62.8 (576) | 66.5 (576) | **+3.6** | [+0.3, +6.9] |
| outcome/win_vs_EARTH | 58.3 (144) | 63.2 (144) | +4.9 | [-2.2, +11.7] |
| outcome/win_vs_FIRE | 50.7 (144) | 54.9 (144) | +4.2 | [-1.6, +10.5] |
| outcome/win_vs_LIGHTNING | 69.4 (144) | 72.2 (144) | +2.8 | [-3.9, +9.2] |
| outcome/win_vs_WATER | 72.9 (144) | 75.7 (144) | +2.8 | [-3.9, +9.4] |
| response/any_nonblock_response_when_legal | 69.9 (133) | 83.2 (131) | **+13.3** | [+4.3, +22.4] |
| response/defender_declared_when_legal | 22.0 (141) | 24.5 (159) | +2.5 | [-6.3, +9.9] |
| response/spell_played_when_legal | 44.4 (63) | 64.8 (54) | **+20.4** | [+5.6, +36.5] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 14.3 (91) | 22.0 (91) | +7.7 | [-2.7, +18.8] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 8.8 (91) | 16.5 (91) | +7.7 | [-2.1, +18.0] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (288) | 98.6 (288) | **-1.4** | [-2.9, -0.3] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 69.1 (288) | 68.1 (288) | -1.0 | [-3.6, +1.4] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.2 (480) | 0.6 (480) | +0.4 | [-0.4, +1.3] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.2 (480) | 0.6 (480) | +0.4 | [-0.4, +1.3] |
| strategy/attacks_by_equipped_attacker | 1.1 (7310) | 0.8 (7565) | **-0.3** | [-0.6, -0.0] |
| strategy/face_target_share_when_both_legal | 67.4 (4259) | 61.3 (4251) | **-6.1** | [-8.4, -3.9] |
| strategy/favorable_trade_taken_per_available_turn | 52.8 (1107) | 61.7 (1158) | **+8.9** | [+5.5, +12.3] |
| strategy/spell_cast_per_legal_main_turn | 27.1 (2339) | 24.9 (2364) | **-2.2** | [-3.8, -0.6] |

### fire vs u8223 — sample — all|Ragefire/Kagoro

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.2 (48) | 17.2 (48) | **-1.0** | [-1.7, -0.4] |
| draft/mean_cost | 2.66 (2400) | 2.75 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 51.5 (2400) | 52.3 (2400) | +0.8 | [-0.3, +2.0] |
| draft/spell_share | 6.3 (2400) | 9.2 (2400) | **+2.9** | [+2.1, +3.8] |
| draft/unique_cards | 34.62 (48) | 35.12 (48) | **+0.5** | [+0.1, +0.9] |
| draft/weapon_share | 7.2 (2400) | 6.2 (2400) | **-1.0** | [-1.4, -0.7] |
| fire/burn_cast_per_legal_turn | 13.8 (80) | 9.1 (99) | -4.7 | [-10.2, +1.4] |
| fire/burn_cast_then_opp_damage | 81.8 (11) | 66.7 (9) | -15.2 | [-55.0, +22.2] |
| gate/Ragefire/buffed_then_attacks_same_turn | 37.5 (32) | 48.5 (33) | +11.0 | [-13.8, +33.6] |
| gate/Ragefire/portal_then_buff | 23.2 (138) | 27.5 (120) | +4.3 | [-6.1, +14.6] |
| gate/portal_per_legal_turn | 100.0 (138) | 98.4 (122) | -1.6 | [-4.1, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 22.2 (9) | 22.2 (18) | +0.0 | [-72.7, +37.1] |
| ikz/held_at_end_of_own_turn | 10.4 (307) | 10.4 (307) | +0.0 | [-3.0, +3.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (32) | 3.1 (32) | +3.1 | [+0.0, +9.1] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 98.4 (64) | 98.7 (76) | +0.2 | [-3.5, +5.1] |
| leader/use_per_legal_turn | 39.8 (161) | 48.7 (156) | +9.0 | [-0.7, +18.4] |
| outcome/own_turns | 6.40 (48) | 6.40 (48) | +0.0 | [-0.3, +0.3] |
| outcome/win | 45.8 (48) | 50.0 (48) | +4.2 | [-8.3, +18.7] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 22.9 (48) | 37.5 (48) | +14.6 | [-4.2, +33.3] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 14.6 (48) | 27.1 (48) | +12.5 | [-4.2, +29.2] |
| strategy/attacks_by_equipped_attacker | 3.2 (592) | 2.3 (616) | -0.9 | [-2.6, +0.7] |
| strategy/face_target_share_when_both_legal | 68.7 (380) | 61.2 (420) | -7.5 | [-15.5, +1.4] |
| strategy/favorable_trade_taken_per_available_turn | 46.8 (94) | 58.6 (116) | +11.8 | [-1.7, +24.2] |
| strategy/spell_cast_per_legal_main_turn | 10.7 (122) | 9.1 (154) | -1.6 | [-7.1, +3.2] |

### fire vs u8223 — sample — all|Ragefire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 7.4 (27) | 6.1 (33) | -1.3 | [-16.7, +10.8] |
| draft/max_wjaccard_to_curated | 17.5 (48) | 17.0 (48) | -0.5 | [-1.2, +0.1] |
| draft/mean_cost | 2.69 (2400) | 2.79 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 53.5 (2400) | 53.8 (2400) | +0.3 | [-0.7, +1.4] |
| draft/spell_share | 6.7 (2400) | 9.0 (2400) | **+2.3** | [+1.3, +3.4] |
| draft/unique_cards | 34.98 (48) | 35.56 (48) | **+0.6** | [+0.2, +1.0] |
| draft/weapon_share | 7.9 (2400) | 6.1 (2400) | **-1.8** | [-2.4, -1.1] |
| fire/burn_cast_per_legal_turn | 27.2 (992) | 25.3 (998) | -2.0 | [-4.5, +0.4] |
| fire/burn_cast_then_opp_damage | 95.9 (364) | 93.1 (321) | -2.7 | [-5.9, +0.5] |
| fire/zero_ping_target_is_redirect_card | 7.7 (39) | 10.0 (20) | +2.3 | [-14.3, +18.6] |
| gate/Ragefire/buffed_then_attacks_same_turn | 45.1 (195) | 51.8 (170) | +6.6 | [-1.7, +14.7] |
| gate/Ragefire/portal_then_buff | 26.3 (741) | 25.6 (664) | -0.7 | [-4.3, +2.7] |
| gate/portal_per_legal_turn | 98.9 (749) | 98.7 (673) | -0.3 | [-1.6, +0.9] |
| heal/heal_spell_per_legal_turn_below_max_hp | 26.7 (307) | 35.1 (259) | **+8.4** | [+3.1, +13.8] |
| ikz/held_at_end_of_own_turn | 23.9 (1564) | 22.1 (1584) | -1.8 | [-3.8, +0.1] |
| ikz/held_then_spent_in_opp_turn | 0.3 (374) | 0.0 (350) | -0.3 | [-0.8, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 2.6 (39) | 9.5 (21) | +7.0 | [-4.4, +23.1] |
| leader/Zero/use_with_target | 88.6 (44) | 95.5 (22) | +6.8 | [-5.5, +18.2] |
| leader/use_per_legal_turn | 2.9 (1537) | 1.4 (1564) | **-1.5** | [-2.7, -0.2] |
| outcome/own_turns | 6.52 (240) | 6.60 (240) | +0.1 | [-0.1, +0.2] |
| outcome/win | 41.7 (240) | 48.8 (240) | **+7.1** | [+1.3, +13.3] |
| response/any_nonblock_response_when_legal | 96.3 (27) | 100.0 (29) | +3.7 | [+0.0, +13.0] |
| response/defender_declared_when_legal | 22.0 (123) | 26.8 (123) | +4.9 | [-4.2, +12.9] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.4 (240) | 0.8 (240) | +0.4 | [-0.8, +1.7] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.4 (240) | 0.8 (240) | +0.4 | [-0.8, +1.7] |
| strategy/attacks_by_equipped_attacker | 0.8 (2597) | 0.2 (2752) | **-0.6** | [-1.1, -0.2] |
| strategy/face_target_share_when_both_legal | 62.5 (1769) | 57.9 (1794) | **-4.5** | [-7.5, -1.6] |
| strategy/favorable_trade_taken_per_available_turn | 56.1 (512) | 63.1 (545) | **+7.1** | [+2.3, +11.8] |
| strategy/spell_cast_per_legal_main_turn | 27.5 (1231) | 26.6 (1213) | -0.8 | [-3.0, +1.3] |

### fire vs u8223 — sample — all|Rushfire/Kagoro

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.3 (48) | 18.8 (48) | -0.5 | [-1.1, +0.1] |
| draft/mean_cost | 2.49 (2400) | 2.44 (2400) | **-0.1** | [-0.1, -0.0] |
| draft/normal_share | 52.8 (2400) | 53.8 (2400) | **+1.0** | [+0.0, +2.0] |
| draft/spell_share | 4.2 (2400) | 5.8 (2400) | **+1.6** | [+1.0, +2.2] |
| draft/unique_cards | 34.04 (48) | 34.12 (48) | +0.1 | [-0.3, +0.5] |
| draft/weapon_share | 4.9 (2400) | 4.1 (2400) | **-0.8** | [-1.3, -0.3] |
| fire/burn_cast_per_legal_turn | 8.6 (35) | 6.7 (30) | -1.9 | [-14.3, +13.0] |
| gate/Rushfire/payload_then_attack_same_turn | 96.8 (155) | 95.7 (140) | -1.1 | [-4.9, +3.0] |
| gate/Rushfire/portal_then_payload | 64.9 (239) | 61.4 (228) | -3.5 | [-9.5, +3.0] |
| gate/portal_per_legal_turn | 99.2 (241) | 97.9 (233) | -1.3 | [-3.1, +0.4] |
| ikz/held_at_end_of_own_turn | 12.8 (282) | 11.8 (280) | -1.0 | [-5.0, +2.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (36) | 0.0 (33) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.9 (74) | 98.6 (71) | +2.6 | [-0.0, +6.0] |
| leader/use_per_legal_turn | 80.4 (92) | 78.0 (91) | -2.4 | [-11.8, +7.9] |
| outcome/own_turns | 5.88 (48) | 5.83 (48) | -0.0 | [-0.3, +0.2] |
| outcome/win | 83.3 (48) | 83.3 (48) | +0.0 | [-8.3, +8.3] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 4.7 (43) | 4.7 (43) | +0.0 | [-9.3, +9.3] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 2.3 (43) | 4.7 (43) | +2.3 | [-5.1, +10.3] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 100.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 79.2 (48) | 77.1 (48) | -2.1 | [-12.5, +8.3] |
| strategy/attacks_by_equipped_attacker | 1.3 (716) | 1.5 (719) | +0.3 | [-0.6, +1.1] |
| strategy/face_target_share_when_both_legal | 72.6 (351) | 66.9 (332) | -5.8 | [-14.1, +2.1] |
| strategy/favorable_trade_taken_per_available_turn | 37.6 (93) | 54.3 (92) | **+16.7** | [+5.7, +28.4] |
| strategy/spell_cast_per_legal_main_turn | 7.8 (64) | 10.4 (67) | +2.6 | [-5.5, +10.5] |

### fire vs u8223 — sample — all|Rushfire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.3 (48) | 18.5 (48) | **-0.8** | [-1.4, -0.2] |
| draft/mean_cost | 2.55 (2400) | 2.50 (2400) | **-0.1** | [-0.1, -0.0] |
| draft/normal_share | 52.3 (2400) | 53.6 (2400) | **+1.3** | [+0.4, +2.1] |
| draft/spell_share | 3.7 (2400) | 5.2 (2400) | **+1.5** | [+1.1, +2.0] |
| draft/unique_cards | 34.12 (48) | 34.06 (48) | -0.1 | [-0.5, +0.4] |
| draft/weapon_share | 4.5 (2400) | 3.3 (2400) | **-1.2** | [-1.5, -0.8] |
| fire/burn_cast_per_legal_turn | 31.2 (862) | 27.7 (858) | **-3.5** | [-5.7, -1.2] |
| fire/burn_cast_then_opp_damage | 93.5 (336) | 94.2 (291) | +0.7 | [-2.9, +4.4] |
| fire/zero_ping_target_is_redirect_card | 10.7 (28) | 14.3 (7) | +3.6 | [-23.1, +50.0] |
| gate/Rushfire/payload_then_attack_same_turn | 98.1 (827) | 95.1 (837) | **-3.0** | [-4.3, -1.5] |
| gate/Rushfire/portal_then_payload | 70.0 (1182) | 70.5 (1187) | +0.5 | [-2.5, +3.5] |
| gate/portal_per_legal_turn | 98.8 (1196) | 98.1 (1210) | -0.7 | [-1.5, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 30.4 (23) | 33.3 (12) | +2.9 | [-36.7, +46.0] |
| ikz/held_at_end_of_own_turn | 25.4 (1428) | 24.1 (1416) | -1.3 | [-3.7, +1.0] |
| ikz/held_then_spent_in_opp_turn | 7.4 (363) | 9.7 (341) | +2.2 | [-0.5, +5.4] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (29) | 14.3 (7) | +14.3 | [+0.0, +40.0] |
| leader/Zero/use_with_target | 96.7 (30) | 100.0 (7) | +3.3 | [+0.0, +9.5] |
| leader/use_per_legal_turn | 2.1 (1408) | 0.5 (1395) | **-1.6** | [-2.8, -0.7] |
| outcome/own_turns | 5.95 (240) | 5.90 (240) | -0.0 | [-0.2, +0.1] |
| outcome/win | 83.3 (240) | 84.2 (240) | +0.8 | [-3.3, +5.0] |
| response/any_nonblock_response_when_legal | 62.4 (101) | 78.0 (100) | **+15.6** | [+4.4, +27.6] |
| response/defender_declared_when_legal | 18.2 (11) | 14.3 (35) | -3.9 | [-88.9, +9.2] |
| response/spell_played_when_legal | 43.5 (62) | 64.2 (53) | **+20.6** | [+5.9, +37.2] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (240) | 98.3 (240) | **-1.7** | [-3.3, -0.4] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 67.1 (240) | 66.2 (240) | -0.8 | [-2.9, +1.2] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (240) | 0.4 (240) | +0.4 | [+0.0, +1.2] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (240) | 0.4 (240) | +0.4 | [+0.0, +1.2] |
| strategy/attacks_by_equipped_attacker | 1.0 (3405) | 0.9 (3478) | -0.0 | [-0.4, +0.2] |
| strategy/face_target_share_when_both_legal | 70.9 (1759) | 63.7 (1705) | **-7.3** | [-10.6, -3.6] |
| strategy/favorable_trade_taken_per_available_turn | 53.4 (408) | 62.2 (405) | **+8.8** | [+3.3, +14.0] |
| strategy/spell_cast_per_legal_main_turn | 30.2 (922) | 26.3 (930) | **-3.8** | [-6.1, -1.4] |

### fire vs u8223 — sample — fixed|ALL

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (21) | 7.1 (28) | +7.1 | [+0.0, +16.7] |
| fire/burn_cast_per_legal_turn | 29.6 (1741) | 27.2 (1747) | **-2.4** | [-4.2, -0.7] |
| fire/burn_cast_then_opp_damage | 94.8 (677) | 93.5 (597) | -1.4 | [-3.8, +1.0] |
| fire/zero_ping_target_is_redirect_card | 12.5 (40) | 14.3 (21) | +1.8 | [-18.3, +20.8] |
| gate/Ragefire/buffed_then_attacks_same_turn | 42.0 (157) | 47.8 (138) | +5.8 | [-2.0, +14.2] |
| gate/Ragefire/portal_then_buff | 26.4 (595) | 26.1 (529) | -0.3 | [-4.0, +3.4] |
| gate/Rushfire/payload_then_attack_same_turn | 98.5 (672) | 95.5 (686) | **-3.0** | [-4.7, -1.5] |
| gate/Rushfire/portal_then_payload | 74.1 (907) | 74.6 (920) | +0.5 | [-2.6, +3.5] |
| gate/portal_per_legal_turn | 98.9 (1519) | 98.9 (1465) | +0.0 | [-0.5, +0.6] |
| heal/heal_spell_per_legal_turn_below_max_hp | 26.9 (294) | 35.7 (241) | **+8.8** | [+3.5, +14.3] |
| ikz/held_at_end_of_own_turn | 23.6 (2347) | 21.7 (2353) | **-1.9** | [-3.7, -0.3] |
| ikz/held_then_spent_in_opp_turn | 4.9 (555) | 6.5 (511) | +1.6 | [-0.3, +3.7] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (40) | 4.5 (22) | +4.5 | [+0.0, +15.2] |
| leader/Zero/use_with_target | 90.9 (44) | 95.7 (23) | +4.7 | [-7.0, +14.5] |
| leader/use_per_legal_turn | 1.9 (2306) | 1.0 (2313) | **-0.9** | [-1.8, -0.2] |
| outcome/own_turns | 6.11 (384) | 6.13 (384) | +0.0 | [-0.1, +0.1] |
| outcome/win | 65.1 (384) | 70.3 (384) | **+5.2** | [+1.6, +8.9] |
| outcome/win_vs_EARTH | 56.2 (96) | 64.6 (96) | +8.3 | [-1.1, +17.0] |
| outcome/win_vs_FIRE | 53.1 (96) | 55.2 (96) | +2.1 | [-4.5, +8.3] |
| outcome/win_vs_LIGHTNING | 75.0 (96) | 78.1 (96) | +3.1 | [-5.2, +11.7] |
| outcome/win_vs_WATER | 76.0 (96) | 83.3 (96) | **+7.3** | [+2.8, +12.5] |
| response/any_nonblock_response_when_legal | 68.0 (122) | 81.5 (119) | **+13.5** | [+3.7, +23.6] |
| response/defender_declared_when_legal | 22.3 (94) | 28.9 (97) | +6.5 | [-2.6, +14.7] |
| response/spell_played_when_legal | 43.5 (62) | 64.2 (53) | **+20.6** | [+6.1, +37.2] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (192) | 100.0 (192) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 66.1 (192) | 65.1 (192) | -1.0 | [-2.8, +0.0] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (384) | 0.3 (384) | +0.3 | [+0.0, +0.8] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (384) | 0.3 (384) | +0.3 | [+0.0, +0.8] |
| strategy/attacks_by_equipped_attacker | 0.4 (4759) | 0.5 (4956) | +0.1 | [-0.1, +0.3] |
| strategy/face_target_share_when_both_legal | 67.5 (2786) | 61.8 (2793) | **-5.6** | [-8.4, -2.9] |
| strategy/favorable_trade_taken_per_available_turn | 56.3 (691) | 63.7 (725) | **+7.4** | [+3.1, +11.3] |
| strategy/spell_cast_per_legal_main_turn | 30.1 (1919) | 27.9 (1924) | **-2.2** | [-3.8, -0.5] |

### fire vs u8223 — sample — fixed|Ragefire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (21) | 7.1 (28) | +7.1 | [+0.0, +16.1] |
| fire/burn_cast_per_legal_turn | 27.6 (923) | 26.2 (929) | -1.5 | [-4.1, +1.1] |
| fire/burn_cast_then_opp_damage | 96.0 (349) | 92.9 (312) | -3.0 | [-6.4, +0.2] |
| fire/zero_ping_target_is_redirect_card | 11.1 (27) | 11.1 (18) | +0.0 | [-21.9, +20.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 42.0 (157) | 47.8 (138) | +5.8 | [-2.8, +14.2] |
| gate/Ragefire/portal_then_buff | 26.4 (595) | 26.1 (529) | -0.3 | [-3.8, +3.4] |
| gate/portal_per_legal_turn | 98.8 (602) | 99.2 (533) | +0.4 | [-0.3, +1.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 26.9 (294) | 35.7 (241) | **+8.8** | [+3.3, +14.2] |
| ikz/held_at_end_of_own_turn | 23.4 (1246) | 21.1 (1258) | **-2.4** | [-4.6, -0.1] |
| ikz/held_then_spent_in_opp_turn | 0.0 (292) | 0.0 (265) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (27) | 5.3 (19) | +5.3 | [+0.0, +18.8] |
| leader/Zero/use_with_target | 90.0 (30) | 95.0 (20) | +5.0 | [-8.9, +19.2] |
| leader/use_per_legal_turn | 2.5 (1222) | 1.6 (1239) | -0.8 | [-2.0, +0.3] |
| outcome/own_turns | 6.49 (192) | 6.55 (192) | +0.1 | [-0.1, +0.2] |
| outcome/win | 44.8 (192) | 52.6 (192) | **+7.8** | [+1.6, +14.1] |
| response/any_nonblock_response_when_legal | 95.2 (21) | 100.0 (24) | +4.8 | [+0.0, +16.7] |
| response/defender_declared_when_legal | 22.3 (94) | 28.9 (97) | +6.5 | [-3.4, +14.5] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (192) | 0.5 (192) | +0.5 | [+0.0, +1.6] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (192) | 0.5 (192) | +0.5 | [+0.0, +1.6] |
| strategy/attacks_by_equipped_attacker | 0.0 (2098) | 0.0 (2229) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 62.8 (1431) | 58.9 (1468) | **-4.0** | [-7.6, -0.5] |
| strategy/favorable_trade_taken_per_available_turn | 57.8 (396) | 64.0 (433) | **+6.1** | [+0.7, +11.6] |
| strategy/spell_cast_per_legal_main_turn | 28.7 (1101) | 27.6 (1106) | -1.1 | [-3.3, +1.1] |

### fire vs u8223 — sample — fixed|Rushfire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| fire/burn_cast_per_legal_turn | 31.9 (818) | 28.4 (818) | **-3.5** | [-5.8, -1.3] |
| fire/burn_cast_then_opp_damage | 93.6 (328) | 94.0 (285) | +0.4 | [-3.2, +4.1] |
| fire/zero_ping_target_is_redirect_card | 15.4 (13) | 33.3 (3) | +17.9 | [-37.5, +100.0] |
| gate/Rushfire/payload_then_attack_same_turn | 98.5 (672) | 95.5 (686) | **-3.0** | [-4.6, -1.5] |
| gate/Rushfire/portal_then_payload | 74.1 (907) | 74.6 (920) | +0.5 | [-2.6, +3.5] |
| gate/portal_per_legal_turn | 98.9 (917) | 98.7 (932) | -0.2 | [-1.0, +0.6] |
| ikz/held_at_end_of_own_turn | 23.9 (1101) | 22.5 (1095) | -1.4 | [-3.8, +0.9] |
| ikz/held_then_spent_in_opp_turn | 10.3 (263) | 13.4 (246) | +3.1 | [-0.7, +7.2] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (13) | 0.0 (3) | +0.0 | [+0.0, +0.0] |
| leader/Zero/use_with_target | 92.9 (14) | 100.0 (3) | +7.1 | [+0.0, +14.3] |
| leader/use_per_legal_turn | 1.3 (1084) | 0.3 (1074) | **-1.0** | [-2.2, -0.1] |
| outcome/own_turns | 5.73 (192) | 5.70 (192) | -0.0 | [-0.1, +0.1] |
| outcome/win | 85.4 (192) | 88.0 (192) | +2.6 | [-0.5, +6.2] |
| response/any_nonblock_response_when_legal | 62.4 (101) | 76.8 (95) | **+14.5** | [+3.6, +26.3] |
| response/spell_played_when_legal | 43.5 (62) | 64.2 (53) | **+20.6** | [+6.1, +36.8] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (192) | 100.0 (192) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 66.1 (192) | 65.1 (192) | -1.0 | [-2.6, +0.0] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (192) | 0.0 (192) | +0.0 | [+0.0, +0.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (192) | 0.0 (192) | +0.0 | [+0.0, +0.0] |
| strategy/attacks_by_equipped_attacker | 0.8 (2661) | 1.0 (2727) | +0.2 | [-0.1, +0.5] |
| strategy/face_target_share_when_both_legal | 72.4 (1355) | 65.1 (1325) | **-7.3** | [-11.3, -3.0] |
| strategy/favorable_trade_taken_per_available_turn | 54.2 (295) | 63.4 (292) | **+9.1** | [+2.4, +15.4] |
| strategy/spell_cast_per_legal_main_turn | 31.9 (818) | 28.4 (818) | **-3.5** | [-5.8, -1.3] |

### fire vs u8223 — sample — free_draft|ALL

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 40.0 (10) | 9.1 (11) | -30.9 | [-66.7, +6.7] |
| draft/max_wjaccard_to_curated | 18.6 (192) | 17.9 (192) | **-0.7** | [-1.0, -0.4] |
| draft/mean_cost | 2.60 (9600) | 2.62 (9600) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 52.5 (9600) | 53.4 (9600) | **+0.9** | [+0.4, +1.4] |
| draft/spell_share | 5.2 (9600) | 7.3 (9600) | **+2.1** | [+1.7, +2.5] |
| draft/unique_cards | 34.44 (192) | 34.72 (192) | **+0.3** | [+0.1, +0.5] |
| draft/weapon_share | 6.1 (9600) | 4.9 (9600) | **-1.2** | [-1.5, -0.9] |
| fire/burn_cast_per_legal_turn | 16.2 (228) | 10.9 (238) | **-5.3** | [-9.3, -1.2] |
| fire/burn_cast_then_opp_damage | 89.5 (38) | 88.5 (26) | -1.0 | [-17.0, +12.8] |
| fire/zero_ping_target_is_redirect_card | 3.7 (27) | 0.0 (6) | -3.7 | [-14.3, +0.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 48.6 (70) | 58.5 (65) | +9.9 | [-7.0, +24.2] |
| gate/Ragefire/portal_then_buff | 24.6 (284) | 25.5 (255) | +0.8 | [-6.7, +8.2] |
| gate/Rushfire/payload_then_attack_same_turn | 96.5 (310) | 94.5 (291) | -1.9 | [-4.6, +0.5] |
| gate/Rushfire/portal_then_payload | 60.3 (514) | 58.8 (495) | -1.5 | [-6.9, +3.8] |
| gate/portal_per_legal_turn | 99.1 (805) | 97.0 (773) | **-2.1** | [-3.5, -0.8] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.5 (51) | 29.6 (54) | +4.1 | [-10.9, +19.6] |
| ikz/held_at_end_of_own_turn | 20.3 (1234) | 19.9 (1234) | -0.4 | [-2.9, +2.1] |
| ikz/held_then_spent_in_opp_turn | 0.4 (250) | 0.4 (245) | +0.0 | [-1.2, +1.2] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 97.1 (138) | 98.6 (147) | +1.5 | [-1.1, +4.3] |
| leader/Zero/target_then_attacks_same_turn | 3.6 (28) | 33.3 (6) | +29.8 | [-8.3, +90.6] |
| leader/Zero/use_with_target | 93.3 (30) | 100.0 (6) | +6.7 | [+0.0, +16.7] |
| leader/use_per_legal_turn | 18.8 (892) | 17.1 (893) | -1.7 | [-4.8, +1.5] |
| outcome/own_turns | 6.43 (192) | 6.43 (192) | +0.0 | [-0.2, +0.2] |
| outcome/win | 58.3 (192) | 58.9 (192) | +0.5 | [-5.7, +6.8] |
| outcome/win_vs_EARTH | 62.5 (48) | 60.4 (48) | -2.1 | [-13.2, +8.7] |
| outcome/win_vs_FIRE | 45.8 (48) | 54.2 (48) | +8.3 | [-4.3, +21.7] |
| outcome/win_vs_LIGHTNING | 58.3 (48) | 60.4 (48) | +2.1 | [-9.1, +13.5] |
| outcome/win_vs_WATER | 66.7 (48) | 60.4 (48) | -6.2 | [-22.7, +11.4] |
| response/any_nonblock_response_when_legal | 90.9 (11) | 100.0 (12) | +9.1 | [+0.0, +33.3] |
| response/defender_declared_when_legal | 21.3 (47) | 17.7 (62) | -3.5 | [-23.2, +11.1] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 14.3 (91) | 22.0 (91) | +7.7 | [-3.1, +20.1] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 8.8 (91) | 16.5 (91) | +7.7 | [-1.6, +17.9] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (96) | 95.8 (96) | **-4.2** | [-8.5, -1.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 75.0 (96) | 74.0 (96) | -1.0 | [-8.1, +5.7] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 1.0 (96) | 2.1 (96) | +1.0 | [-2.3, +4.9] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 1.0 (96) | 2.1 (96) | +1.0 | [-2.3, +4.9] |
| strategy/attacks_by_equipped_attacker | 2.5 (2551) | 1.5 (2609) | **-1.0** | [-1.7, -0.4] |
| strategy/face_target_share_when_both_legal | 67.1 (1473) | 60.2 (1458) | **-7.0** | [-10.7, -3.4] |
| strategy/favorable_trade_taken_per_available_turn | 46.9 (416) | 58.2 (433) | **+11.3** | [+6.3, +16.7] |
| strategy/spell_cast_per_legal_main_turn | 13.6 (420) | 11.8 (440) | -1.8 | [-5.0, +1.7] |

### fire vs u8223 — sample — free_draft|Ragefire/Kagoro

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.2 (48) | 17.2 (48) | **-1.0** | [-1.7, -0.4] |
| draft/mean_cost | 2.66 (2400) | 2.75 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 51.5 (2400) | 52.3 (2400) | +0.8 | [-0.3, +2.0] |
| draft/spell_share | 6.3 (2400) | 9.2 (2400) | **+2.9** | [+2.1, +3.8] |
| draft/unique_cards | 34.62 (48) | 35.12 (48) | **+0.5** | [+0.1, +0.9] |
| draft/weapon_share | 7.2 (2400) | 6.2 (2400) | **-1.0** | [-1.4, -0.7] |
| fire/burn_cast_per_legal_turn | 13.8 (80) | 9.1 (99) | -4.7 | [-10.2, +1.4] |
| fire/burn_cast_then_opp_damage | 81.8 (11) | 66.7 (9) | -15.2 | [-55.0, +22.2] |
| gate/Ragefire/buffed_then_attacks_same_turn | 37.5 (32) | 48.5 (33) | +11.0 | [-13.8, +33.6] |
| gate/Ragefire/portal_then_buff | 23.2 (138) | 27.5 (120) | +4.3 | [-6.1, +14.6] |
| gate/portal_per_legal_turn | 100.0 (138) | 98.4 (122) | -1.6 | [-4.1, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 22.2 (9) | 22.2 (18) | +0.0 | [-72.7, +37.1] |
| ikz/held_at_end_of_own_turn | 10.4 (307) | 10.4 (307) | +0.0 | [-3.0, +3.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (32) | 3.1 (32) | +3.1 | [+0.0, +9.1] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 98.4 (64) | 98.7 (76) | +0.2 | [-3.5, +5.1] |
| leader/use_per_legal_turn | 39.8 (161) | 48.7 (156) | +9.0 | [-0.7, +18.4] |
| outcome/own_turns | 6.40 (48) | 6.40 (48) | +0.0 | [-0.3, +0.3] |
| outcome/win | 45.8 (48) | 50.0 (48) | +4.2 | [-8.3, +18.7] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 22.9 (48) | 37.5 (48) | +14.6 | [-4.2, +33.3] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 14.6 (48) | 27.1 (48) | +12.5 | [-4.2, +29.2] |
| strategy/attacks_by_equipped_attacker | 3.2 (592) | 2.3 (616) | -0.9 | [-2.6, +0.7] |
| strategy/face_target_share_when_both_legal | 68.7 (380) | 61.2 (420) | -7.5 | [-15.5, +1.4] |
| strategy/favorable_trade_taken_per_available_turn | 46.8 (94) | 58.6 (116) | +11.8 | [-1.7, +24.2] |
| strategy/spell_cast_per_legal_main_turn | 10.7 (122) | 9.1 (154) | -1.6 | [-7.1, +3.2] |

### fire vs u8223 — sample — free_draft|Ragefire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 17.5 (48) | 17.0 (48) | -0.5 | [-1.2, +0.1] |
| draft/mean_cost | 2.69 (2400) | 2.79 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 53.5 (2400) | 53.8 (2400) | +0.3 | [-0.7, +1.3] |
| draft/spell_share | 6.7 (2400) | 9.0 (2400) | **+2.3** | [+1.4, +3.4] |
| draft/unique_cards | 34.98 (48) | 35.56 (48) | **+0.6** | [+0.2, +1.0] |
| draft/weapon_share | 7.9 (2400) | 6.1 (2400) | **-1.8** | [-2.4, -1.2] |
| fire/burn_cast_per_legal_turn | 21.7 (69) | 13.0 (69) | **-8.7** | [-16.5, -0.6] |
| fire/burn_cast_then_opp_damage | 93.3 (15) | 100.0 (9) | +6.7 | [+0.0, +23.1] |
| fire/zero_ping_target_is_redirect_card | 0.0 (12) | 0.0 (2) | +0.0 | [+0.0, +0.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 57.9 (38) | 68.8 (32) | +10.9 | [-12.1, +29.0] |
| gate/Ragefire/portal_then_buff | 26.0 (146) | 23.7 (135) | -2.3 | [-12.5, +7.9] |
| gate/portal_per_legal_turn | 99.3 (147) | 96.4 (140) | -2.9 | [-8.7, +1.3] |
| heal/heal_spell_per_legal_turn_below_max_hp | 23.1 (13) | 27.8 (18) | +4.7 | [-12.1, +21.4] |
| ikz/held_at_end_of_own_turn | 25.8 (318) | 26.1 (326) | +0.3 | [-4.7, +5.0] |
| ikz/held_then_spent_in_opp_turn | 1.2 (82) | 0.0 (85) | -1.2 | [-3.9, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 8.3 (12) | 50.0 (2) | +41.7 | [-18.2, +100.0] |
| leader/Zero/use_with_target | 85.7 (14) | 100.0 (2) | +14.3 | [+0.0, +36.4] |
| leader/use_per_legal_turn | 4.4 (315) | 0.6 (325) | **-3.8** | [-7.7, -0.6] |
| outcome/own_turns | 6.62 (48) | 6.79 (48) | +0.2 | [-0.2, +0.5] |
| outcome/win | 29.2 (48) | 33.3 (48) | +4.2 | [-10.4, +18.8] |
| response/defender_declared_when_legal | 20.7 (29) | 19.2 (26) | -1.5 | [-28.0, +43.5] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 2.1 (48) | 2.1 (48) | +0.0 | [-6.2, +6.2] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 2.1 (48) | 2.1 (48) | +0.0 | [-6.2, +6.2] |
| strategy/attacks_by_equipped_attacker | 4.2 (499) | 1.1 (523) | **-3.1** | [-5.3, -0.9] |
| strategy/face_target_share_when_both_legal | 60.9 (338) | 53.7 (326) | **-7.3** | [-12.1, -1.8] |
| strategy/favorable_trade_taken_per_available_turn | 50.0 (116) | 59.8 (112) | +9.8 | [-0.2, +19.2] |
| strategy/spell_cast_per_legal_main_turn | 16.9 (130) | 16.8 (107) | -0.1 | [-6.6, +7.6] |

### fire vs u8223 — sample — free_draft|Rushfire/Kagoro

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.3 (48) | 18.8 (48) | -0.5 | [-1.1, +0.1] |
| draft/mean_cost | 2.49 (2400) | 2.44 (2400) | **-0.1** | [-0.1, -0.0] |
| draft/normal_share | 52.8 (2400) | 53.8 (2400) | **+1.0** | [+0.0, +2.0] |
| draft/spell_share | 4.2 (2400) | 5.8 (2400) | **+1.6** | [+1.0, +2.2] |
| draft/unique_cards | 34.04 (48) | 34.12 (48) | +0.1 | [-0.3, +0.5] |
| draft/weapon_share | 4.9 (2400) | 4.1 (2400) | **-0.8** | [-1.3, -0.3] |
| fire/burn_cast_per_legal_turn | 8.6 (35) | 6.7 (30) | -1.9 | [-14.3, +13.0] |
| gate/Rushfire/payload_then_attack_same_turn | 96.8 (155) | 95.7 (140) | -1.1 | [-4.9, +3.0] |
| gate/Rushfire/portal_then_payload | 64.9 (239) | 61.4 (228) | -3.5 | [-9.5, +3.0] |
| gate/portal_per_legal_turn | 99.2 (241) | 97.9 (233) | -1.3 | [-3.1, +0.4] |
| ikz/held_at_end_of_own_turn | 12.8 (282) | 11.8 (280) | -1.0 | [-5.0, +2.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (36) | 0.0 (33) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.9 (74) | 98.6 (71) | +2.6 | [-0.0, +6.0] |
| leader/use_per_legal_turn | 80.4 (92) | 78.0 (91) | -2.4 | [-11.8, +7.9] |
| outcome/own_turns | 5.88 (48) | 5.83 (48) | -0.0 | [-0.3, +0.2] |
| outcome/win | 83.3 (48) | 83.3 (48) | +0.0 | [-8.3, +8.3] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 4.7 (43) | 4.7 (43) | +0.0 | [-9.3, +9.3] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 2.3 (43) | 4.7 (43) | +2.3 | [-5.1, +10.3] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 100.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 79.2 (48) | 77.1 (48) | -2.1 | [-12.5, +8.3] |
| strategy/attacks_by_equipped_attacker | 1.3 (716) | 1.5 (719) | +0.3 | [-0.6, +1.1] |
| strategy/face_target_share_when_both_legal | 72.6 (351) | 66.9 (332) | -5.8 | [-14.1, +2.1] |
| strategy/favorable_trade_taken_per_available_turn | 37.6 (93) | 54.3 (92) | **+16.7** | [+5.7, +28.4] |
| strategy/spell_cast_per_legal_main_turn | 7.8 (64) | 10.4 (67) | +2.6 | [-5.5, +10.5] |

### fire vs u8223 — sample — free_draft|Rushfire/Zero

| metric | u8223 | fire | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.3 (48) | 18.5 (48) | **-0.8** | [-1.4, -0.2] |
| draft/mean_cost | 2.55 (2400) | 2.50 (2400) | **-0.1** | [-0.1, -0.0] |
| draft/normal_share | 52.3 (2400) | 53.6 (2400) | **+1.3** | [+0.5, +2.1] |
| draft/spell_share | 3.7 (2400) | 5.2 (2400) | **+1.5** | [+1.1, +2.0] |
| draft/unique_cards | 34.12 (48) | 34.06 (48) | -0.1 | [-0.5, +0.4] |
| draft/weapon_share | 4.5 (2400) | 3.3 (2400) | **-1.2** | [-1.5, -0.8] |
| fire/burn_cast_per_legal_turn | 18.2 (44) | 15.0 (40) | -3.2 | [-13.4, +6.7] |
| fire/burn_cast_then_opp_damage | 87.5 (8) | 100.0 (6) | +12.5 | [+0.0, +33.3] |
| fire/zero_ping_target_is_redirect_card | 6.7 (15) | 0.0 (4) | -6.7 | [-28.6, +0.0] |
| gate/Rushfire/payload_then_attack_same_turn | 96.1 (155) | 93.4 (151) | -2.8 | [-6.5, +0.7] |
| gate/Rushfire/portal_then_payload | 56.4 (275) | 56.6 (267) | +0.2 | [-7.9, +7.6] |
| gate/portal_per_legal_turn | 98.6 (279) | 96.0 (278) | **-2.5** | [-4.5, -0.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 30.4 (23) | 33.3 (12) | +2.9 | [-36.2, +44.4] |
| ikz/held_at_end_of_own_turn | 30.6 (327) | 29.6 (321) | -1.0 | [-8.1, +5.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (100) | 0.0 (95) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (16) | 25.0 (4) | +25.0 | [+0.0, +50.0] |
| leader/Zero/use_with_target | 100.0 (16) | 100.0 (4) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 4.9 (324) | 1.2 (321) | **-3.7** | [-7.3, -1.0] |
| outcome/own_turns | 6.81 (48) | 6.69 (48) | -0.1 | [-0.6, +0.2] |
| outcome/win | 75.0 (48) | 68.8 (48) | -6.2 | [-20.8, +10.4] |
| response/defender_declared_when_legal | 18.2 (11) | 14.3 (35) | -3.9 | [-88.9, +9.4] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 91.7 (48) | **-8.3** | [-16.7, -2.1] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 70.8 (48) | 70.8 (48) | +0.0 | [-8.3, +8.3] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (48) | 2.1 (48) | +2.1 | [+0.0, +6.2] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (48) | 2.1 (48) | +2.1 | [+0.0, +6.2] |
| strategy/attacks_by_equipped_attacker | 1.9 (744) | 0.9 (751) | **-0.9** | [-1.8, -0.3] |
| strategy/face_target_share_when_both_legal | 66.1 (404) | 58.7 (380) | **-7.4** | [-14.7, -0.9] |
| strategy/favorable_trade_taken_per_available_turn | 51.3 (113) | 59.3 (113) | +8.0 | [-1.5, +17.1] |
| strategy/spell_cast_per_legal_main_turn | 16.3 (104) | 11.6 (112) | -4.7 | [-13.5, +4.1] |

### u9305 vs u8223 — argmax — all|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (20) | 13.0 (23) | +13.0 | [+0.0, +29.2] |
| draft/max_wjaccard_to_curated | 25.0 (192) | 25.8 (192) | **+0.7** | [+0.6, +0.9] |
| draft/mean_cost | 2.54 (9600) | 2.54 (9600) | +0.0 | [-0.0, +0.0] |
| draft/normal_share | 44.5 (9600) | 48.0 (9600) | **+3.5** | [+3.2, +3.8] |
| draft/spell_share | 0.0 (9600) | 0.0 (9600) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 18.00 (192) | 17.75 (192) | **-0.2** | [-0.3, -0.2] |
| draft/weapon_share | 6.5 (9600) | 7.0 (9600) | **+0.5** | [+0.3, +0.7] |
| fire/burn_cast_per_legal_turn | 28.5 (1724) | 27.1 (1781) | **-1.4** | [-2.7, -0.1] |
| fire/burn_cast_then_opp_damage | 92.0 (654) | 91.9 (620) | -0.1 | [-2.5, +2.2] |
| fire/zero_ping_redirect_target_then_opp_damage | 100.0 (8) | 100.0 (2) | +0.0 | [+0.0, +0.0] |
| fire/zero_ping_target_is_redirect_card | 14.3 (56) | 5.7 (35) | -8.6 | [-18.8, +0.9] |
| gate/Ragefire/buffed_then_attacks_same_turn | 47.7 (262) | 42.0 (295) | -5.7 | [-11.1, +0.0] |
| gate/Ragefire/portal_then_buff | 29.2 (897) | 31.2 (944) | +2.0 | [-0.8, +4.8] |
| gate/Rushfire/payload_then_attack_same_turn | 98.2 (1010) | 97.4 (995) | -0.8 | [-1.9, +0.3] |
| gate/Rushfire/portal_then_payload | 72.6 (1392) | 71.6 (1390) | -1.0 | [-3.2, +1.2] |
| gate/portal_per_legal_turn | 98.8 (2317) | 98.3 (2374) | -0.5 | [-1.1, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 24.8 (262) | -0.8 | [-4.9, +3.9] |
| ikz/held_at_end_of_own_turn | 21.4 (3515) | 22.2 (3535) | +0.9 | [-0.3, +2.1] |
| ikz/held_then_spent_in_opp_turn | 3.5 (751) | 3.7 (786) | +0.2 | [-0.5, +0.9] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.9 (123) | 99.3 (140) | +3.4 | [-0.0, +7.4] |
| leader/Zero/target_then_attacks_same_turn | 5.4 (56) | 5.6 (36) | +0.2 | [-9.7, +12.0] |
| leader/Zero/use_with_target | 93.3 (60) | 92.3 (39) | -1.0 | [-10.8, +7.7] |
| leader/use_per_legal_turn | 5.8 (3169) | 5.6 (3207) | -0.2 | [-1.1, +0.7] |
| outcome/own_turns | 6.10 (576) | 6.14 (576) | +0.0 | [-0.0, +0.1] |
| outcome/win | 70.8 (576) | 69.6 (576) | -1.2 | [-4.3, +1.6] |
| outcome/win_vs_EARTH | 66.7 (144) | 67.4 (144) | +0.7 | [-6.0, +7.0] |
| outcome/win_vs_FIRE | 59.7 (144) | 53.5 (144) | -6.2 | [-13.6, +0.7] |
| outcome/win_vs_LIGHTNING | 74.3 (144) | 79.9 (144) | **+5.6** | [+0.7, +10.6] |
| outcome/win_vs_WATER | 82.6 (144) | 77.8 (144) | **-4.9** | [-9.3, -0.6] |
| response/any_nonblock_response_when_legal | 72.0 (107) | 71.0 (124) | -1.0 | [-11.5, +8.5] |
| response/defender_declared_when_legal | 28.6 (70) | 26.7 (86) | -1.8 | [-13.3, +6.6] |
| response/spell_played_when_legal | 47.3 (55) | 45.3 (64) | -2.0 | [-15.8, +11.0] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 15.8 (95) | 15.8 (95) | +0.0 | [-9.1, +9.2] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 10.5 (95) | 11.6 (95) | +1.1 | [-7.5, +9.4] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (288) | 100.0 (288) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 64.2 (288) | 66.7 (288) | **+2.4** | [+0.7, +4.3] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.2 (480) | 0.4 (480) | +0.2 | [-0.4, +1.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (480) | 0.4 (480) | +0.4 | [+0.0, +1.1] |
| strategy/attacks_by_equipped_attacker | 1.2 (7490) | 1.1 (7562) | -0.1 | [-0.3, +0.2] |
| strategy/face_target_share_when_both_legal | 68.4 (4095) | 68.0 (4130) | -0.5 | [-2.1, +1.1] |
| strategy/favorable_trade_taken_per_available_turn | 51.9 (1000) | 54.3 (1014) | +2.4 | [-0.7, +5.5] |
| strategy/spell_cast_per_legal_main_turn | 28.9 (1904) | 27.9 (1944) | -1.0 | [-2.3, +0.2] |

### u9305 vs u8223 — argmax — all|Ragefire/Kagoro

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 23.6 (48) | 23.6 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 2.68 (2400) | 2.78 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 48.0 (2400) | 50.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 17.00 (48) | 17.00 (48) | +0.0 | [+0.0, +0.0] |
| draft/weapon_share | 10.0 (2400) | 10.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 69.6 (46) | 45.0 (60) | **-24.6** | [-39.5, -8.4] |
| gate/Ragefire/portal_then_buff | 33.6 (137) | 40.0 (150) | +6.4 | [-3.8, +16.8] |
| gate/portal_per_legal_turn | 100.0 (137) | 99.3 (151) | -0.7 | [-2.1, +0.0] |
| ikz/held_at_end_of_own_turn | 9.6 (314) | 8.9 (313) | -0.6 | [-3.7, +1.9] |
| ikz/held_then_spent_in_opp_turn | 0.0 (30) | 0.0 (28) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 97.4 (39) | 100.0 (47) | +2.6 | [+0.0, +9.7] |
| leader/use_per_legal_turn | 24.1 (162) | 29.0 (162) | +4.9 | [-4.0, +14.0] |
| outcome/own_turns | 6.54 (48) | 6.52 (48) | -0.0 | [-0.3, +0.2] |
| outcome/win | 77.1 (48) | 66.7 (48) | -10.4 | [-27.1, +4.2] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 25.5 (47) | 27.1 (48) | +1.6 | [-14.6, +18.4] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 17.0 (47) | 18.8 (48) | +1.7 | [-12.5, +16.4] |
| strategy/attacks_by_equipped_attacker | 3.1 (650) | 2.7 (662) | -0.4 | [-1.9, +1.0] |
| strategy/face_target_share_when_both_legal | 69.5 (367) | 64.7 (368) | -4.8 | [-11.9, +1.3] |
| strategy/favorable_trade_taken_per_available_turn | 46.5 (86) | 54.0 (100) | +7.5 | [-6.4, +22.0] |

### u9305 vs u8223 — argmax — all|Ragefire/Zero

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (20) | 13.0 (23) | +13.0 | [+0.0, +28.6] |
| draft/max_wjaccard_to_curated | 23.6 (48) | 23.6 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 2.68 (2400) | 2.80 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 48.0 (2400) | 50.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 17.00 (48) | 16.00 (48) | **-1.0** | [-1.0, -1.0] |
| draft/weapon_share | 10.0 (2400) | 10.0 (2400) | +0.0 | [+0.0, +0.0] |
| fire/burn_cast_per_legal_turn | 27.4 (909) | 25.2 (942) | **-2.2** | [-4.0, -0.4] |
| fire/burn_cast_then_opp_damage | 93.5 (352) | 92.2 (322) | -1.2 | [-4.5, +2.1] |
| fire/zero_ping_target_is_redirect_card | 11.1 (36) | 4.5 (22) | -6.6 | [-18.5, +3.9] |
| gate/Ragefire/buffed_then_attacks_same_turn | 43.1 (216) | 41.3 (235) | -1.8 | [-7.7, +4.2] |
| gate/Ragefire/portal_then_buff | 28.4 (760) | 29.6 (794) | +1.2 | [-1.4, +4.0] |
| gate/portal_per_legal_turn | 99.7 (762) | 99.0 (802) | -0.7 | [-1.9, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 24.8 (262) | -0.8 | [-4.7, +3.5] |
| ikz/held_at_end_of_own_turn | 21.4 (1559) | 22.2 (1575) | +0.7 | [-1.2, +2.6] |
| ikz/held_then_spent_in_opp_turn | 0.0 (334) | 0.0 (349) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 8.3 (36) | 4.5 (22) | -3.8 | [-16.3, +12.1] |
| leader/Zero/use_with_target | 90.0 (40) | 88.0 (25) | -2.0 | [-18.0, +12.2] |
| leader/use_per_legal_turn | 2.6 (1542) | 1.6 (1565) | **-1.0** | [-2.0, -0.0] |
| outcome/own_turns | 6.50 (240) | 6.56 (240) | +0.1 | [-0.0, +0.1] |
| outcome/win | 49.2 (240) | 49.6 (240) | +0.4 | [-4.6, +5.4] |
| response/any_nonblock_response_when_legal | 94.4 (18) | 100.0 (19) | +5.6 | [+0.0, +16.7] |
| response/defender_declared_when_legal | 28.6 (70) | 26.7 (86) | -1.8 | [-12.2, +7.3] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.4 (240) | 0.4 (240) | +0.0 | [-1.2, +1.2] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (240) | 0.4 (240) | +0.4 | [+0.0, +1.2] |
| strategy/attacks_by_equipped_attacker | 1.0 (2729) | 0.6 (2754) | **-0.4** | [-0.8, -0.0] |
| strategy/face_target_share_when_both_legal | 63.5 (1809) | 62.2 (1843) | -1.3 | [-4.1, +1.2] |
| strategy/favorable_trade_taken_per_available_turn | 54.8 (496) | 56.7 (513) | +1.9 | [-1.7, +5.9] |
| strategy/spell_cast_per_legal_main_turn | 28.3 (1089) | 26.9 (1105) | -1.4 | [-3.0, +0.1] |

### u9305 vs u8223 — argmax — all|Rushfire/Kagoro

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 27.9 (48) | 29.4 (48) | **+1.5** | [+1.5, +1.5] |
| draft/mean_cost | 2.32 (2400) | 2.12 (2400) | **-0.2** | [-0.2, -0.2] |
| draft/normal_share | 42.0 (2400) | 46.0 (2400) | **+4.0** | [+4.0, +4.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 20.00 (48) | +0.0 | [+0.0, +0.0] |
| draft/weapon_share | 2.0 (2400) | 4.0 (2400) | **+2.0** | [+2.0, +2.0] |
| gate/Rushfire/payload_then_attack_same_turn | 96.0 (175) | 97.0 (165) | +1.0 | [-2.7, +4.3] |
| gate/Rushfire/portal_then_payload | 75.1 (233) | 72.7 (227) | -2.4 | [-7.1, +2.1] |
| gate/portal_per_legal_turn | 97.9 (238) | 97.0 (234) | -0.9 | [-3.5, +2.0] |
| ikz/held_at_end_of_own_turn | 14.2 (253) | 15.0 (253) | +0.8 | [-4.6, +6.5] |
| ikz/held_then_spent_in_opp_turn | 0.0 (36) | 0.0 (38) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.2 (84) | 98.9 (93) | +3.7 | [-0.9, +8.6] |
| leader/use_per_legal_turn | 87.5 (96) | 88.6 (105) | +1.1 | [-6.4, +8.8] |
| outcome/own_turns | 5.27 (48) | 5.27 (48) | +0.0 | [-0.2, +0.2] |
| outcome/win | 95.8 (48) | 91.7 (48) | -4.2 | [-12.5, +4.2] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 6.2 (48) | 4.3 (47) | -2.0 | [-12.5, +6.2] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 4.2 (48) | 4.3 (47) | +0.1 | [-10.4, +10.4] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 100.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 62.5 (48) | 64.6 (48) | +2.1 | [+0.0, +6.2] |
| strategy/attacks_by_equipped_attacker | 1.0 (720) | 1.9 (740) | **+0.9** | [+0.2, +1.8] |
| strategy/face_target_share_when_both_legal | 84.0 (300) | 82.0 (311) | -2.0 | [-9.4, +4.2] |
| strategy/favorable_trade_taken_per_available_turn | 34.5 (55) | 30.8 (52) | -3.8 | [-19.6, +13.3] |

### u9305 vs u8223 — argmax — all|Rushfire/Zero

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 25.0 (48) | 26.4 (48) | **+1.4** | [+1.4, +1.4] |
| draft/mean_cost | 2.48 (2400) | 2.46 (2400) | **-0.0** | [-0.0, -0.0] |
| draft/normal_share | 40.0 (2400) | 46.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 18.00 (48) | 18.00 (48) | +0.0 | [+0.0, +0.0] |
| draft/weapon_share | 4.0 (2400) | 4.0 (2400) | +0.0 | [+0.0, +0.0] |
| fire/burn_cast_per_legal_turn | 29.8 (815) | 29.3 (839) | -0.5 | [-2.2, +1.2] |
| fire/burn_cast_then_opp_damage | 90.4 (302) | 91.6 (298) | +1.2 | [-2.3, +4.4] |
| fire/zero_ping_target_is_redirect_card | 20.0 (20) | 7.7 (13) | -12.3 | [-36.4, +4.6] |
| gate/Rushfire/payload_then_attack_same_turn | 98.7 (835) | 97.5 (830) | **-1.2** | [-2.3, -0.1] |
| gate/Rushfire/portal_then_payload | 72.0 (1159) | 71.4 (1163) | -0.7 | [-3.1, +1.9] |
| gate/portal_per_legal_turn | 98.2 (1180) | 98.0 (1187) | -0.2 | [-1.1, +0.6] |
| ikz/held_at_end_of_own_turn | 25.3 (1389) | 26.6 (1394) | +1.3 | [-0.6, +3.4] |
| ikz/held_then_spent_in_opp_turn | 7.4 (351) | 7.8 (371) | +0.4 | [-0.9, +1.9] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (20) | 7.1 (14) | +7.1 | [+0.0, +26.3] |
| leader/Zero/use_with_target | 100.0 (20) | 100.0 (14) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 1.5 (1369) | 1.0 (1375) | -0.4 | [-1.1, +0.1] |
| outcome/own_turns | 5.79 (240) | 5.81 (240) | +0.0 | [-0.1, +0.1] |
| outcome/win | 86.2 (240) | 85.8 (240) | -0.4 | [-3.8, +2.9] |
| response/any_nonblock_response_when_legal | 67.0 (88) | 65.0 (103) | -2.0 | [-13.5, +8.0] |
| response/spell_played_when_legal | 47.3 (55) | 45.3 (64) | -2.0 | [-15.5, +10.7] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (240) | 100.0 (240) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 64.6 (240) | 67.1 (240) | **+2.5** | [+0.8, +4.6] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (240) | 0.4 (240) | +0.4 | [+0.0, +1.2] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (240) | 0.4 (240) | +0.4 | [+0.0, +1.2] |
| strategy/attacks_by_equipped_attacker | 1.0 (3391) | 1.1 (3406) | +0.1 | [-0.3, +0.4] |
| strategy/face_target_share_when_both_legal | 70.9 (1619) | 72.6 (1608) | +1.7 | [-0.2, +3.7] |
| strategy/favorable_trade_taken_per_available_turn | 51.8 (363) | 54.4 (349) | +2.7 | [-1.8, +7.1] |
| strategy/spell_cast_per_legal_main_turn | 29.8 (815) | 29.3 (839) | -0.5 | [-2.2, +1.2] |

### u9305 vs u8223 — argmax — fixed|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (20) | 13.0 (23) | +13.0 | [+0.0, +28.6] |
| fire/burn_cast_per_legal_turn | 28.5 (1724) | 27.1 (1781) | **-1.4** | [-2.6, -0.2] |
| fire/burn_cast_then_opp_damage | 92.0 (654) | 91.9 (620) | -0.1 | [-2.4, +2.3] |
| fire/zero_ping_redirect_target_then_opp_damage | 100.0 (8) | 100.0 (2) | +0.0 | [+0.0, +0.0] |
| fire/zero_ping_target_is_redirect_card | 23.5 (34) | 11.1 (18) | -12.4 | [-30.8, +2.9] |
| gate/Ragefire/buffed_then_attacks_same_turn | 39.5 (167) | 42.0 (169) | +2.5 | [-2.4, +7.5] |
| gate/Ragefire/portal_then_buff | 28.3 (591) | 27.4 (616) | -0.8 | [-3.3, +1.7] |
| gate/Rushfire/payload_then_attack_same_turn | 98.5 (677) | 98.4 (668) | -0.2 | [-1.1, +0.8] |
| gate/Rushfire/portal_then_payload | 75.4 (898) | 73.2 (912) | -2.1 | [-4.5, +0.1] |
| gate/portal_per_legal_turn | 98.9 (1506) | 98.2 (1556) | -0.7 | [-1.4, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 24.8 (262) | -0.8 | [-4.8, +3.8] |
| ikz/held_at_end_of_own_turn | 23.3 (2325) | 24.2 (2340) | +0.9 | [-0.4, +2.2] |
| ikz/held_then_spent_in_opp_turn | 4.8 (542) | 5.1 (566) | +0.3 | [-0.6, +1.2] |
| leader/Zero/target_then_attacks_same_turn | 8.8 (34) | 5.3 (19) | -3.6 | [-16.7, +12.9] |
| leader/Zero/use_with_target | 94.4 (36) | 90.5 (21) | -4.0 | [-16.7, +6.6] |
| leader/use_per_legal_turn | 1.6 (2291) | 0.9 (2313) | **-0.7** | [-1.3, -0.2] |
| outcome/own_turns | 6.05 (384) | 6.09 (384) | +0.0 | [-0.0, +0.1] |
| outcome/win | 66.1 (384) | 65.6 (384) | -0.5 | [-3.9, +2.9] |
| outcome/win_vs_EARTH | 57.3 (96) | 59.4 (96) | +2.1 | [-6.5, +10.0] |
| outcome/win_vs_FIRE | 53.1 (96) | 52.1 (96) | -1.0 | [-7.7, +5.8] |
| outcome/win_vs_LIGHTNING | 74.0 (96) | 76.0 (96) | +2.1 | [-3.4, +8.1] |
| outcome/win_vs_WATER | 80.2 (96) | 75.0 (96) | -5.2 | [-11.0, +0.0] |
| response/any_nonblock_response_when_legal | 71.7 (106) | 70.5 (122) | -1.2 | [-11.9, +8.2] |
| response/defender_declared_when_legal | 28.6 (70) | 26.7 (86) | -1.8 | [-11.8, +7.1] |
| response/spell_played_when_legal | 47.3 (55) | 45.3 (64) | -2.0 | [-15.5, +11.4] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (192) | 100.0 (192) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 65.6 (192) | 66.7 (192) | +1.0 | [+0.0, +2.7] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.3 (384) | 0.3 (384) | +0.0 | [-0.8, +0.8] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (384) | 0.3 (384) | +0.3 | [+0.0, +0.8] |
| strategy/attacks_by_equipped_attacker | 0.5 (4788) | 0.4 (4770) | -0.0 | [-0.2, +0.1] |
| strategy/face_target_share_when_both_legal | 66.3 (2794) | 67.2 (2784) | +0.9 | [-0.5, +2.4] |
| strategy/favorable_trade_taken_per_available_turn | 57.3 (688) | 58.0 (684) | +0.8 | [-1.8, +3.2] |
| strategy/spell_cast_per_legal_main_turn | 28.9 (1904) | 27.9 (1944) | -1.0 | [-2.1, +0.2] |

### u9305 vs u8223 — argmax — fixed|Ragefire/Zero

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (20) | 13.0 (23) | +13.0 | [+0.0, +28.6] |
| fire/burn_cast_per_legal_turn | 27.4 (909) | 25.2 (942) | **-2.2** | [-3.9, -0.5] |
| fire/burn_cast_then_opp_damage | 93.5 (352) | 92.2 (322) | -1.2 | [-4.5, +2.1] |
| fire/zero_ping_target_is_redirect_card | 16.7 (24) | 8.3 (12) | -8.3 | [-28.3, +10.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 39.5 (167) | 42.0 (169) | +2.5 | [-2.2, +7.5] |
| gate/Ragefire/portal_then_buff | 28.3 (591) | 27.4 (616) | -0.8 | [-3.3, +1.7] |
| gate/portal_per_legal_turn | 99.7 (593) | 98.7 (624) | -0.9 | [-2.4, +0.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.6 (285) | 24.8 (262) | -0.8 | [-4.8, +3.5] |
| ikz/held_at_end_of_own_turn | 22.1 (1231) | 23.4 (1237) | +1.3 | [-0.5, +3.1] |
| ikz/held_then_spent_in_opp_turn | 0.0 (272) | 0.0 (289) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 12.5 (24) | 0.0 (12) | -12.5 | [-25.9, +0.0] |
| leader/Zero/use_with_target | 92.3 (26) | 85.7 (14) | -6.6 | [-26.0, +10.5] |
| leader/use_per_legal_turn | 2.1 (1217) | 1.1 (1228) | **-1.0** | [-2.0, -0.2] |
| outcome/own_turns | 6.41 (192) | 6.44 (192) | +0.0 | [-0.0, +0.1] |
| outcome/win | 46.4 (192) | 46.4 (192) | +0.0 | [-5.2, +5.7] |
| response/any_nonblock_response_when_legal | 94.4 (18) | 100.0 (19) | +5.6 | [+0.0, +17.6] |
| response/defender_declared_when_legal | 28.6 (70) | 26.7 (86) | -1.8 | [-12.6, +7.2] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.5 (192) | 0.0 (192) | -0.5 | [-1.6, +0.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (192) | 0.0 (192) | +0.0 | [+0.0, +0.0] |
| strategy/attacks_by_equipped_attacker | 0.0 (2135) | 0.0 (2120) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 62.3 (1484) | 62.8 (1466) | +0.5 | [-1.6, +2.6] |
| strategy/favorable_trade_taken_per_available_turn | 57.4 (404) | 58.8 (398) | +1.4 | [-2.3, +4.7] |
| strategy/spell_cast_per_legal_main_turn | 28.3 (1089) | 26.9 (1105) | -1.4 | [-3.0, +0.1] |

### u9305 vs u8223 — argmax — fixed|Rushfire/Zero

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| fire/burn_cast_per_legal_turn | 29.8 (815) | 29.3 (839) | -0.5 | [-2.2, +1.3] |
| fire/burn_cast_then_opp_damage | 90.4 (302) | 91.6 (298) | +1.2 | [-2.1, +4.5] |
| fire/zero_ping_target_is_redirect_card | 40.0 (10) | 16.7 (6) | -23.3 | [-63.3, +11.1] |
| gate/Rushfire/payload_then_attack_same_turn | 98.5 (677) | 98.4 (668) | -0.2 | [-1.2, +0.8] |
| gate/Rushfire/portal_then_payload | 75.4 (898) | 73.2 (912) | -2.1 | [-4.4, +0.1] |
| gate/portal_per_legal_turn | 98.4 (913) | 97.9 (932) | -0.5 | [-1.4, +0.4] |
| ikz/held_at_end_of_own_turn | 24.7 (1094) | 25.1 (1103) | +0.4 | [-1.4, +2.5] |
| ikz/held_then_spent_in_opp_turn | 9.6 (270) | 10.5 (277) | +0.8 | [-1.0, +2.6] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (10) | 14.3 (7) | +14.3 | [+0.0, +50.0] |
| leader/Zero/use_with_target | 100.0 (10) | 100.0 (7) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 0.9 (1074) | 0.6 (1085) | -0.3 | [-0.7, +0.1] |
| outcome/own_turns | 5.70 (192) | 5.74 (192) | +0.0 | [-0.1, +0.2] |
| outcome/win | 85.9 (192) | 84.9 (192) | -1.0 | [-5.2, +2.6] |
| response/any_nonblock_response_when_legal | 67.0 (88) | 65.0 (103) | -2.0 | [-13.4, +7.7] |
| response/spell_played_when_legal | 47.3 (55) | 45.3 (64) | -2.0 | [-15.3, +10.9] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (192) | 100.0 (192) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 65.6 (192) | 66.7 (192) | +1.0 | [+0.0, +2.6] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (192) | 0.5 (192) | +0.5 | [+0.0, +1.6] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (192) | 0.5 (192) | +0.5 | [+0.0, +1.6] |
| strategy/attacks_by_equipped_attacker | 0.8 (2653) | 0.8 (2650) | -0.1 | [-0.4, +0.2] |
| strategy/face_target_share_when_both_legal | 70.8 (1310) | 72.2 (1318) | +1.3 | [-0.7, +3.1] |
| strategy/favorable_trade_taken_per_available_turn | 57.0 (284) | 57.0 (286) | -0.0 | [-4.1, +3.7] |
| strategy/spell_cast_per_legal_main_turn | 29.8 (815) | 29.3 (839) | -0.5 | [-2.2, +1.3] |

### u9305 vs u8223 — argmax — free_draft|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 25.0 (192) | 25.8 (192) | **+0.7** | [+0.6, +0.9] |
| draft/mean_cost | 2.54 (9600) | 2.54 (9600) | +0.0 | [-0.0, +0.0] |
| draft/normal_share | 44.5 (9600) | 48.0 (9600) | **+3.5** | [+3.2, +3.8] |
| draft/spell_share | 0.0 (9600) | 0.0 (9600) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 18.00 (192) | 17.75 (192) | **-0.2** | [-0.3, -0.2] |
| draft/weapon_share | 6.5 (9600) | 7.0 (9600) | **+0.5** | [+0.3, +0.7] |
| fire/zero_ping_target_is_redirect_card | 0.0 (22) | 0.0 (17) | +0.0 | [+0.0, +0.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 62.1 (95) | 42.1 (126) | **-20.0** | [-32.1, -7.3] |
| gate/Ragefire/portal_then_buff | 31.0 (306) | 38.4 (328) | **+7.4** | [+1.1, +13.7] |
| gate/Rushfire/payload_then_attack_same_turn | 97.6 (333) | 95.4 (327) | -2.2 | [-4.9, +0.4] |
| gate/Rushfire/portal_then_payload | 67.4 (494) | 68.4 (478) | +1.0 | [-3.7, +5.6] |
| gate/portal_per_legal_turn | 98.6 (811) | 98.5 (818) | -0.1 | [-1.3, +1.0] |
| ikz/held_at_end_of_own_turn | 17.6 (1190) | 18.4 (1195) | +0.8 | [-1.7, +3.4] |
| ikz/held_then_spent_in_opp_turn | 0.0 (209) | 0.0 (220) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.9 (123) | 99.3 (140) | +3.4 | [-0.1, +7.6] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (22) | 5.9 (17) | +5.9 | [+0.0, +25.0] |
| leader/Zero/use_with_target | 91.7 (24) | 94.4 (18) | +2.8 | [-13.3, +20.1] |
| leader/use_per_legal_turn | 16.7 (878) | 17.7 (894) | +0.9 | [-1.9, +3.7] |
| outcome/own_turns | 6.20 (192) | 6.22 (192) | +0.0 | [-0.1, +0.2] |
| outcome/win | 80.2 (192) | 77.6 (192) | -2.6 | [-8.3, +3.6] |
| outcome/win_vs_EARTH | 85.4 (48) | 83.3 (48) | -2.1 | [-11.5, +7.9] |
| outcome/win_vs_FIRE | 72.9 (48) | 56.2 (48) | -16.7 | [-32.5, +0.0] |
| outcome/win_vs_LIGHTNING | 75.0 (48) | 87.5 (48) | **+12.5** | [+4.0, +21.9] |
| outcome/win_vs_WATER | 87.5 (48) | 83.3 (48) | -4.2 | [-12.5, +3.7] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 15.8 (95) | 15.8 (95) | +0.0 | [-9.6, +9.2] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 10.5 (95) | 11.6 (95) | +1.1 | [-7.6, +9.0] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (96) | 100.0 (96) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 61.5 (96) | 66.7 (96) | **+5.2** | [+1.1, +10.0] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (96) | 1.0 (96) | +1.0 | [+0.0, +3.8] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (96) | 1.0 (96) | +1.0 | [+0.0, +3.8] |
| strategy/attacks_by_equipped_attacker | 2.5 (2702) | 2.3 (2792) | -0.2 | [-0.8, +0.5] |
| strategy/face_target_share_when_both_legal | 73.1 (1301) | 69.5 (1346) | -3.6 | [-7.6, +0.1] |
| strategy/favorable_trade_taken_per_available_turn | 40.1 (312) | 46.7 (330) | +6.6 | [-0.6, +14.2] |

### u9305 vs u8223 — argmax — free_draft|Ragefire/Kagoro

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 23.6 (48) | 23.6 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 2.68 (2400) | 2.78 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 48.0 (2400) | 50.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 17.00 (48) | 17.00 (48) | +0.0 | [+0.0, +0.0] |
| draft/weapon_share | 10.0 (2400) | 10.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 69.6 (46) | 45.0 (60) | **-24.6** | [-39.5, -8.4] |
| gate/Ragefire/portal_then_buff | 33.6 (137) | 40.0 (150) | +6.4 | [-3.8, +16.8] |
| gate/portal_per_legal_turn | 100.0 (137) | 99.3 (151) | -0.7 | [-2.1, +0.0] |
| ikz/held_at_end_of_own_turn | 9.6 (314) | 8.9 (313) | -0.6 | [-3.7, +1.9] |
| ikz/held_then_spent_in_opp_turn | 0.0 (30) | 0.0 (28) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 97.4 (39) | 100.0 (47) | +2.6 | [+0.0, +9.7] |
| leader/use_per_legal_turn | 24.1 (162) | 29.0 (162) | +4.9 | [-4.0, +14.0] |
| outcome/own_turns | 6.54 (48) | 6.52 (48) | -0.0 | [-0.3, +0.2] |
| outcome/win | 77.1 (48) | 66.7 (48) | -10.4 | [-27.1, +4.2] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 25.5 (47) | 27.1 (48) | +1.6 | [-14.6, +18.4] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 17.0 (47) | 18.8 (48) | +1.7 | [-12.5, +16.4] |
| strategy/attacks_by_equipped_attacker | 3.1 (650) | 2.7 (662) | -0.4 | [-1.9, +1.0] |
| strategy/face_target_share_when_both_legal | 69.5 (367) | 64.7 (368) | -4.8 | [-11.9, +1.3] |
| strategy/favorable_trade_taken_per_available_turn | 46.5 (86) | 54.0 (100) | +7.5 | [-6.4, +22.0] |

### u9305 vs u8223 — argmax — free_draft|Ragefire/Zero

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 23.6 (48) | 23.6 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 2.68 (2400) | 2.80 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 48.0 (2400) | 50.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 17.00 (48) | 16.00 (48) | **-1.0** | [-1.0, -1.0] |
| draft/weapon_share | 10.0 (2400) | 10.0 (2400) | +0.0 | [+0.0, +0.0] |
| fire/zero_ping_target_is_redirect_card | 0.0 (12) | 0.0 (10) | +0.0 | [+0.0, +0.0] |
| gate/Ragefire/buffed_then_attacks_same_turn | 55.1 (49) | 39.4 (66) | -15.7 | [-34.1, +2.0] |
| gate/Ragefire/portal_then_buff | 29.0 (169) | 37.1 (178) | **+8.1** | [+0.6, +16.5] |
| gate/portal_per_legal_turn | 100.0 (169) | 100.0 (178) | +0.0 | [+0.0, +0.0] |
| ikz/held_at_end_of_own_turn | 18.9 (328) | 17.8 (338) | -1.2 | [-6.3, +4.6] |
| ikz/held_then_spent_in_opp_turn | 0.0 (62) | 0.0 (60) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (12) | 10.0 (10) | +10.0 | [+0.0, +50.0] |
| leader/Zero/use_with_target | 85.7 (14) | 90.9 (11) | +5.2 | [-25.0, +28.6] |
| leader/use_per_legal_turn | 4.3 (325) | 3.3 (337) | -1.0 | [-3.8, +2.1] |
| outcome/own_turns | 6.83 (48) | 7.04 (48) | +0.2 | [-0.0, +0.5] |
| outcome/win | 60.4 (48) | 62.5 (48) | +2.1 | [-12.5, +14.6] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (48) | 2.1 (48) | +2.1 | [+0.0, +6.2] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (48) | 2.1 (48) | +2.1 | [+0.0, +6.2] |
| strategy/attacks_by_equipped_attacker | 4.7 (594) | 2.7 (634) | **-2.0** | [-3.6, -0.4] |
| strategy/face_target_share_when_both_legal | 68.9 (325) | 59.9 (377) | -9.0 | [-19.5, +1.3] |
| strategy/favorable_trade_taken_per_available_turn | 43.5 (92) | 49.6 (115) | +6.1 | [-6.7, +20.5] |

### u9305 vs u8223 — argmax — free_draft|Rushfire/Kagoro

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 27.9 (48) | 29.4 (48) | **+1.5** | [+1.5, +1.5] |
| draft/mean_cost | 2.32 (2400) | 2.12 (2400) | **-0.2** | [-0.2, -0.2] |
| draft/normal_share | 42.0 (2400) | 46.0 (2400) | **+4.0** | [+4.0, +4.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 20.00 (48) | +0.0 | [+0.0, +0.0] |
| draft/weapon_share | 2.0 (2400) | 4.0 (2400) | **+2.0** | [+2.0, +2.0] |
| gate/Rushfire/payload_then_attack_same_turn | 96.0 (175) | 97.0 (165) | +1.0 | [-2.7, +4.3] |
| gate/Rushfire/portal_then_payload | 75.1 (233) | 72.7 (227) | -2.4 | [-7.1, +2.1] |
| gate/portal_per_legal_turn | 97.9 (238) | 97.0 (234) | -0.9 | [-3.5, +2.0] |
| ikz/held_at_end_of_own_turn | 14.2 (253) | 15.0 (253) | +0.8 | [-4.6, +6.5] |
| ikz/held_then_spent_in_opp_turn | 0.0 (36) | 0.0 (38) | +0.0 | [+0.0, +0.0] |
| leader/Kagoro/use_then_leader_attacks_same_turn | 95.2 (84) | 98.9 (93) | +3.7 | [-0.9, +8.6] |
| leader/use_per_legal_turn | 87.5 (96) | 88.6 (105) | +1.1 | [-6.4, +8.8] |
| outcome/own_turns | 5.27 (48) | 5.27 (48) | +0.0 | [-0.2, +0.2] |
| outcome/win | 95.8 (48) | 91.7 (48) | -4.2 | [-12.5, +4.2] |
| sequence/fire.kagoro_after_multi_play/completed_per_eligible_game | 6.2 (48) | 4.3 (47) | -2.0 | [-12.5, +6.2] |
| sequence/fire.kagoro_after_multi_play/converted_per_eligible_game | 4.2 (48) | 4.3 (47) | +0.1 | [-10.4, +10.4] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 100.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 62.5 (48) | 64.6 (48) | +2.1 | [+0.0, +6.2] |
| strategy/attacks_by_equipped_attacker | 1.0 (720) | 1.9 (740) | **+0.9** | [+0.2, +1.8] |
| strategy/face_target_share_when_both_legal | 84.0 (300) | 82.0 (311) | -2.0 | [-9.4, +4.2] |
| strategy/favorable_trade_taken_per_available_turn | 34.5 (55) | 30.8 (52) | -3.8 | [-19.6, +13.3] |

### u9305 vs u8223 — argmax — free_draft|Rushfire/Zero

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 25.0 (48) | 26.4 (48) | **+1.4** | [+1.4, +1.4] |
| draft/mean_cost | 2.48 (2400) | 2.46 (2400) | **-0.0** | [-0.0, -0.0] |
| draft/normal_share | 40.0 (2400) | 46.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 18.00 (48) | 18.00 (48) | +0.0 | [+0.0, +0.0] |
| draft/weapon_share | 4.0 (2400) | 4.0 (2400) | +0.0 | [+0.0, +0.0] |
| fire/zero_ping_target_is_redirect_card | 0.0 (10) | 0.0 (7) | +0.0 | [+0.0, +0.0] |
| gate/Rushfire/payload_then_attack_same_turn | 99.4 (158) | 93.8 (162) | **-5.5** | [-9.2, -2.0] |
| gate/Rushfire/portal_then_payload | 60.5 (261) | 64.5 (251) | +4.0 | [-3.3, +11.2] |
| gate/portal_per_legal_turn | 97.8 (267) | 98.4 (255) | +0.7 | [-1.9, +2.9] |
| ikz/held_at_end_of_own_turn | 27.5 (295) | 32.3 (291) | +4.8 | [-0.6, +10.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (81) | 0.0 (94) | +0.0 | [+0.0, +0.0] |
| leader/Zero/target_then_attacks_same_turn | 0.0 (10) | 0.0 (7) | +0.0 | [+0.0, +0.0] |
| leader/Zero/use_with_target | 100.0 (10) | 100.0 (7) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 3.4 (295) | 2.4 (290) | -1.0 | [-3.8, +1.4] |
| outcome/own_turns | 6.15 (48) | 6.06 (48) | -0.1 | [-0.4, +0.2] |
| outcome/win | 87.5 (48) | 89.6 (48) | +2.1 | [-4.2, +8.3] |
| sequence/fire.rushfire_charge_conversion/completed_per_eligible_game | 100.0 (48) | 100.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.rushfire_charge_conversion/converted_per_eligible_game | 60.4 (48) | 68.8 (48) | **+8.3** | [+2.1, +16.7] |
| sequence/fire.zero_before_attack/completed_per_eligible_game | 0.0 (48) | 0.0 (48) | +0.0 | [+0.0, +0.0] |
| sequence/fire.zero_before_attack/converted_per_eligible_game | 0.0 (48) | 0.0 (48) | +0.0 | [+0.0, +0.0] |
| strategy/attacks_by_equipped_attacker | 1.6 (738) | 2.1 (756) | +0.5 | [-0.8, +1.6] |
| strategy/face_target_share_when_both_legal | 71.2 (309) | 74.8 (290) | +3.6 | [-2.4, +10.0] |
| strategy/favorable_trade_taken_per_available_turn | 32.9 (79) | 42.9 (63) | +9.9 | [-5.8, +24.8] |

## LIGHTNING
### lightning vs u8223 — argmax — all|ALL

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 36.7 (30) | 41.2 (34) | +4.5 | [-16.1, +27.3] |
| draft/max_wjaccard_to_curated | 18.9 (192) | 25.0 (192) | **+6.1** | [+6.0, +6.2] |
| draft/mean_cost | 2.89 (9600) | 2.38 (9600) | **-0.5** | [-0.5, -0.5] |
| draft/normal_share | 65.5 (9600) | 60.0 (9600) | **-5.5** | [-5.7, -5.3] |
| draft/spell_share | 0.0 (9600) | 0.0 (9600) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.25 (192) | 18.00 (192) | **-2.2** | [-2.3, -2.2] |
| draft/weapon_share | 20.0 (9600) | 23.0 (9600) | **+3.0** | [+2.8, +3.2] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (410) | 0.0 (276) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (719) | 100.0 (917) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 97.5 (719) | 97.4 (917) | -0.1 | [-1.5, +1.4] |
| gate/Surge/portal_with_eligible_weapon | 53.3 (1350) | 65.5 (1400) | **+12.2** | [+8.4, +16.1] |
| gate/portal_per_legal_turn | 99.8 (1764) | 99.6 (1683) | -0.2 | [-0.6, +0.2] |
| ikz/held_at_end_of_own_turn | 26.2 (2848) | 25.7 (2845) | -0.5 | [-2.2, +1.3] |
| ikz/held_then_spent_in_opp_turn | 33.6 (745) | 38.6 (731) | **+5.0** | [+1.3, +8.6] |
| lightning/weapon_attach_per_legal_turn | 26.4 (2385) | 27.5 (2430) | +1.1 | [-0.6, +2.9] |
| lightning/weapon_attach_to_entity_share | 0.3 (1035) | 0.0 (1084) | -0.3 | [-0.7, +0.0] |
| outcome/own_turns | 6.36 (448) | 6.35 (448) | -0.0 | [-0.1, +0.1] |
| outcome/win | 51.6 (448) | 57.8 (448) | **+6.2** | [+1.6, +10.7] |
| outcome/win_vs_EARTH | 55.4 (112) | 58.0 (112) | +2.7 | [-7.0, +12.5] |
| outcome/win_vs_FIRE | 29.5 (112) | 38.4 (112) | +8.9 | [+0.0, +18.0] |
| outcome/win_vs_LIGHTNING | 63.4 (112) | 69.6 (112) | +6.2 | [-2.0, +15.2] |
| outcome/win_vs_WATER | 58.0 (112) | 65.2 (112) | +7.1 | [-1.2, +15.3] |
| response/any_nonblock_response_when_legal | 79.5 (337) | 86.7 (361) | **+7.2** | [+1.9, +12.6] |
| response/defender_declared_when_legal | 50.0 (62) | 61.0 (59) | +11.0 | [-6.0, +33.1] |
| response/spell_played_when_legal | 86.7 (263) | 88.8 (278) | +2.2 | [-1.5, +6.6] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 98.6 (215) | 99.6 (243) | +1.0 | [-0.7, +3.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.8 (215) | 93.0 (243) | **+4.2** | [+0.5, +8.3] |
| strategy/attacks_by_equipped_attacker | 22.3 (5166) | 25.3 (5374) | **+3.1** | [+1.9, +4.2] |
| strategy/face_target_share_when_both_legal | 65.9 (3204) | 70.7 (3093) | **+4.9** | [+2.6, +7.1] |
| strategy/favorable_trade_taken_per_available_turn | 42.4 (874) | 37.0 (738) | **-5.5** | [-9.0, -1.9] |
| strategy/spell_cast_per_legal_main_turn | 44.8 (29) | 45.5 (22) | +0.6 | [-17.6, +22.9] |

### lightning vs u8223 — argmax — all|Stormchain/Piko

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.6 (48) | 25.0 (48) | **+5.4** | [+5.4, +5.4] |
| draft/mean_cost | 2.90 (2400) | 2.40 (2400) | **-0.5** | [-0.5, -0.5] |
| draft/normal_share | 64.0 (2400) | 60.0 (2400) | **-4.0** | [-4.0, -4.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 20.0 (2400) | 22.0 (2400) | **+2.0** | [+2.0, +2.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (90) | 0.0 (66) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 98.9 (91) | 98.5 (67) | -0.4 | [-4.1, +3.2] |
| ikz/held_at_end_of_own_turn | 22.8 (320) | 15.1 (317) | **-7.7** | [-13.8, -2.0] |
| ikz/held_then_spent_in_opp_turn | 2.7 (73) | 10.4 (48) | +7.7 | [-2.3, +20.5] |
| lightning/weapon_attach_per_legal_turn | 18.0 (250) | 24.0 (262) | **+6.0** | [+1.0, +11.6] |
| lightning/weapon_attach_to_entity_share | 0.0 (60) | 0.0 (87) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.67 (48) | 6.60 (48) | -0.1 | [-0.3, +0.2] |
| outcome/win | 66.7 (48) | 68.8 (48) | +2.1 | [-10.4, +14.6] |
| response/any_nonblock_response_when_legal | 40.0 (5) | 62.5 (8) | +22.5 | [-60.0, +75.0] |
| strategy/attacks_by_equipped_attacker | 7.1 (560) | 9.4 (594) | +2.3 | [-0.3, +4.7] |
| strategy/face_target_share_when_both_legal | 64.5 (383) | 73.7 (388) | **+9.2** | [+4.1, +14.9] |
| strategy/favorable_trade_taken_per_available_turn | 43.0 (142) | 30.4 (125) | **-12.6** | [-20.8, -4.8] |

### lightning vs u8223 — argmax — all|Stormchain/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (15) | 10.0 (20) | +10.0 | [+0.0, +33.3] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 25.0 (48) | **+6.7** | [+6.7, +6.7] |
| draft/mean_cost | 2.92 (2400) | 2.40 (2400) | **-0.5** | [-0.5, -0.5] |
| draft/normal_share | 66.0 (2400) | 60.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 20.0 (2400) | 22.0 (2400) | **+2.0** | [+2.0, +2.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (320) | 0.0 (210) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (320) | 100.0 (210) | +0.0 | [+0.0, +0.0] |
| ikz/held_at_end_of_own_turn | 33.2 (751) | 33.2 (781) | +0.0 | [-3.7, +3.4] |
| ikz/held_then_spent_in_opp_turn | 40.6 (249) | 40.2 (259) | -0.4 | [-6.0, +5.8] |
| lightning/weapon_attach_per_legal_turn | 27.0 (623) | 31.5 (628) | **+4.6** | [+1.2, +7.6] |
| lightning/weapon_attach_to_entity_share | 0.0 (259) | 0.0 (296) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.71 (112) | 6.97 (112) | **+0.3** | [+0.1, +0.5] |
| outcome/win | 36.6 (112) | 42.9 (112) | +6.2 | [-2.7, +15.2] |
| response/any_nonblock_response_when_legal | 87.9 (124) | 93.4 (122) | +5.5 | [-0.6, +12.4] |
| response/defender_declared_when_legal | 32.6 (46) | 46.5 (43) | +13.9 | [-3.2, +39.4] |
| response/spell_played_when_legal | 94.9 (99) | 93.2 (103) | -1.7 | [-4.6, +0.3] |
| strategy/attacks_by_equipped_attacker | 13.8 (1146) | 15.3 (1240) | +1.5 | [-0.3, +3.3] |
| strategy/face_target_share_when_both_legal | 67.3 (715) | 71.7 (769) | +4.4 | [-0.4, +9.1] |
| strategy/favorable_trade_taken_per_available_turn | 41.2 (211) | 40.0 (190) | -1.2 | [-7.2, +4.9] |

### lightning vs u8223 — argmax — all|Surge/Piko

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.3 (48) | 25.0 (48) | **+6.7** | [+6.7, +6.7] |
| draft/mean_cost | 2.92 (2400) | 2.36 (2400) | **-0.6** | [-0.6, -0.6] |
| draft/normal_share | 66.0 (2400) | 60.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 20.0 (2400) | 24.0 (2400) | **+4.0** | [+4.0, +4.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (27) | 100.0 (82) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.3 (27) | 92.7 (82) | -3.6 | [-10.4, +3.3] |
| gate/Surge/portal_with_eligible_weapon | 12.4 (218) | 33.3 (246) | **+20.9** | [+9.9, +32.4] |
| gate/portal_per_legal_turn | 99.1 (220) | 99.2 (248) | +0.1 | [-1.6, +1.8] |
| ikz/held_at_end_of_own_turn | 16.2 (314) | 16.9 (308) | +0.6 | [-5.3, +6.6] |
| ikz/held_then_spent_in_opp_turn | 0.0 (51) | 11.5 (52) | **+11.5** | [+3.8, +17.7] |
| lightning/weapon_attach_per_legal_turn | 20.2 (248) | 22.6 (243) | +2.5 | [-4.1, +8.7] |
| lightning/weapon_attach_to_entity_share | 0.0 (69) | 0.0 (87) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.54 (48) | 6.42 (48) | -0.1 | [-0.5, +0.2] |
| outcome/win | 50.0 (48) | 68.8 (48) | +18.8 | [+0.0, +37.5] |
| response/any_nonblock_response_when_legal | 0.0 (1) | 75.0 (8) | **+75.0** | [+40.0, +100.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 88.2 (17) | 96.7 (30) | +8.4 | [-6.9, +24.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.2 (17) | 96.7 (30) | +8.4 | [-6.9, +24.0] |
| strategy/attacks_by_equipped_attacker | 12.4 (555) | 19.3 (600) | **+6.9** | [+3.2, +10.6] |
| strategy/face_target_share_when_both_legal | 60.8 (380) | 68.4 (361) | **+7.6** | [+1.0, +14.5] |
| strategy/favorable_trade_taken_per_available_turn | 37.4 (123) | 40.9 (93) | +3.5 | [-4.1, +12.1] |

### lightning vs u8223 — argmax — all|Surge/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 73.3 (15) | 85.7 (14) | +12.4 | [-12.3, +41.7] |
| draft/max_wjaccard_to_curated | 19.6 (48) | 25.0 (48) | **+5.4** | [+5.4, +5.4] |
| draft/mean_cost | 2.82 (2400) | 2.36 (2400) | **-0.5** | [-0.5, -0.5] |
| draft/normal_share | 66.0 (2400) | 60.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 21.00 (48) | 18.00 (48) | **-3.0** | [-3.0, -3.0] |
| draft/weapon_share | 20.0 (2400) | 24.0 (2400) | **+4.0** | [+4.0, +4.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (692) | 100.0 (835) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 97.5 (692) | 97.8 (835) | +0.3 | [-1.1, +1.7] |
| gate/Surge/portal_with_eligible_weapon | 61.1 (1132) | 72.4 (1154) | **+11.2** | [+7.4, +15.1] |
| gate/portal_per_legal_turn | 99.9 (1133) | 99.7 (1158) | -0.3 | [-0.8, +0.1] |
| ikz/held_at_end_of_own_turn | 25.4 (1463) | 25.9 (1439) | +0.4 | [-1.7, +2.7] |
| ikz/held_then_spent_in_opp_turn | 39.5 (372) | 44.9 (372) | **+5.4** | [+0.3, +10.6] |
| lightning/weapon_attach_per_legal_turn | 29.0 (1264) | 27.2 (1297) | -1.8 | [-3.9, +0.3] |
| lightning/weapon_attach_to_entity_share | 0.5 (647) | 0.0 (614) | -0.5 | [-1.1, +0.0] |
| outcome/own_turns | 6.10 (240) | 6.00 (240) | -0.1 | [-0.2, +0.0] |
| outcome/win | 55.8 (240) | 60.4 (240) | +4.6 | [-0.8, +10.0] |
| response/any_nonblock_response_when_legal | 75.8 (207) | 84.3 (223) | **+8.5** | [+1.5, +15.7] |
| response/defender_declared_when_legal | 100.0 (16) | 100.0 (16) | +0.0 | [+0.0, +0.0] |
| response/spell_played_when_legal | 81.7 (164) | 86.3 (175) | +4.6 | [-0.8, +10.7] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.5 (198) | 100.0 (213) | +0.5 | [+0.0, +1.6] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.9 (198) | 92.5 (213) | +3.6 | [-0.3, +7.7] |
| strategy/attacks_by_equipped_attacker | 30.4 (2905) | 34.0 (2940) | **+3.6** | [+2.0, +5.1] |
| strategy/face_target_share_when_both_legal | 66.7 (1726) | 70.0 (1575) | **+3.3** | [+0.1, +6.6] |
| strategy/favorable_trade_taken_per_available_turn | 44.5 (398) | 36.7 (330) | **-7.8** | [-14.2, -1.7] |
| strategy/spell_cast_per_legal_main_turn | 44.8 (29) | 45.5 (22) | +0.6 | [-19.3, +19.4] |

### lightning vs u8223 — argmax — fixed|ALL

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 36.7 (30) | 41.2 (34) | +4.5 | [-16.7, +27.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (222) | 0.0 (149) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (638) | 100.0 (722) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 97.6 (638) | 98.5 (722) | +0.8 | [-0.4, +2.2] |
| gate/Surge/portal_with_eligible_weapon | 70.0 (911) | 78.5 (920) | **+8.4** | [+5.2, +11.6] |
| gate/portal_per_legal_turn | 99.9 (1134) | 99.9 (1070) | -0.0 | [-0.3, +0.3] |
| ikz/held_at_end_of_own_turn | 32.6 (1581) | 33.2 (1591) | +0.6 | [-1.5, +2.7] |
| ikz/held_then_spent_in_opp_turn | 47.8 (515) | 48.1 (528) | +0.3 | [-3.5, +4.6] |
| lightning/weapon_attach_per_legal_turn | 30.2 (1409) | 31.1 (1424) | +0.9 | [-1.0, +2.8] |
| lightning/weapon_attach_to_entity_share | 0.4 (768) | 0.0 (754) | -0.4 | [-0.9, +0.0] |
| outcome/own_turns | 6.18 (256) | 6.21 (256) | +0.0 | [-0.1, +0.2] |
| outcome/win | 46.5 (256) | 50.0 (256) | +3.5 | [-1.6, +9.0] |
| outcome/win_vs_EARTH | 43.8 (64) | 50.0 (64) | +6.2 | [-6.2, +18.6] |
| outcome/win_vs_FIRE | 25.0 (64) | 34.4 (64) | **+9.4** | [+1.5, +17.9] |
| outcome/win_vs_LIGHTNING | 68.8 (64) | 68.8 (64) | +0.0 | [-10.9, +11.1] |
| outcome/win_vs_WATER | 48.4 (64) | 46.9 (64) | -1.6 | [-11.7, +8.8] |
| response/any_nonblock_response_when_legal | 82.8 (319) | 87.2 (327) | +4.4 | [-0.3, +9.3] |
| response/defender_declared_when_legal | 50.0 (62) | 61.0 (59) | +11.0 | [-6.4, +30.9] |
| response/spell_played_when_legal | 86.7 (263) | 88.8 (278) | +2.2 | [-1.4, +6.4] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.4 (172) | 100.0 (177) | +0.6 | [+0.0, +1.9] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 89.0 (172) | 93.8 (177) | **+4.8** | [+1.1, +8.8] |
| strategy/attacks_by_equipped_attacker | 31.4 (2870) | 33.8 (2951) | **+2.4** | [+1.2, +3.6] |
| strategy/face_target_share_when_both_legal | 70.2 (1700) | 72.4 (1639) | +2.1 | [-0.8, +4.9] |
| strategy/favorable_trade_taken_per_available_turn | 38.8 (353) | 35.1 (319) | -3.7 | [-9.1, +1.8] |
| strategy/spell_cast_per_legal_main_turn | 44.8 (29) | 45.5 (22) | +0.6 | [-17.3, +21.5] |

### lightning vs u8223 — argmax — fixed|Stormchain/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (15) | 10.0 (20) | +10.0 | [+0.0, +31.6] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (222) | 0.0 (149) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (222) | 100.0 (149) | +0.0 | [+0.0, +0.0] |
| ikz/held_at_end_of_own_turn | 43.0 (433) | 43.2 (449) | +0.3 | [-3.5, +3.7] |
| ikz/held_then_spent_in_opp_turn | 53.2 (186) | 50.5 (194) | -2.7 | [-8.7, +3.1] |
| lightning/weapon_attach_per_legal_turn | 32.3 (378) | 37.2 (384) | **+5.0** | [+1.6, +8.1] |
| lightning/weapon_attach_to_entity_share | 0.0 (202) | 0.0 (221) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.77 (64) | 7.02 (64) | **+0.2** | [+0.0, +0.5] |
| outcome/win | 21.9 (64) | 28.1 (64) | +6.2 | [-3.1, +17.2] |
| response/any_nonblock_response_when_legal | 89.2 (120) | 93.9 (115) | +4.7 | [-0.4, +10.9] |
| response/defender_declared_when_legal | 32.6 (46) | 46.5 (43) | +13.9 | [-3.6, +38.1] |
| response/spell_played_when_legal | 94.9 (99) | 93.2 (103) | -1.7 | [-4.6, +0.3] |
| strategy/attacks_by_equipped_attacker | 20.6 (579) | 22.3 (628) | +1.7 | [-0.3, +3.7] |
| strategy/face_target_share_when_both_legal | 77.5 (360) | 76.2 (383) | -1.3 | [-7.0, +4.0] |
| strategy/favorable_trade_taken_per_available_turn | 33.3 (81) | 35.0 (80) | +1.7 | [-6.1, +10.0] |

### lightning vs u8223 — argmax — fixed|Surge/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 73.3 (15) | 85.7 (14) | +12.4 | [-11.9, +42.9] |
| gate/Surge/eligible_portal_then_equip | 100.0 (638) | 100.0 (722) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 97.6 (638) | 98.5 (722) | +0.8 | [-0.4, +2.2] |
| gate/Surge/portal_with_eligible_weapon | 70.0 (911) | 78.5 (920) | **+8.4** | [+5.3, +11.6] |
| gate/portal_per_legal_turn | 99.9 (912) | 99.9 (921) | +0.0 | [-0.3, +0.3] |
| ikz/held_at_end_of_own_turn | 28.7 (1148) | 29.2 (1142) | +0.6 | [-1.9, +3.3] |
| ikz/held_then_spent_in_opp_turn | 44.7 (329) | 46.7 (334) | +2.0 | [-3.1, +7.4] |
| lightning/weapon_attach_per_legal_turn | 29.5 (1031) | 28.8 (1040) | -0.6 | [-2.8, +1.6] |
| lightning/weapon_attach_to_entity_share | 0.5 (566) | 0.0 (533) | -0.5 | [-1.2, +0.0] |
| outcome/own_turns | 5.98 (192) | 5.95 (192) | -0.0 | [-0.2, +0.1] |
| outcome/win | 54.7 (192) | 57.3 (192) | +2.6 | [-3.6, +8.3] |
| response/any_nonblock_response_when_legal | 78.9 (199) | 83.5 (212) | +4.6 | [-1.5, +11.2] |
| response/defender_declared_when_legal | 100.0 (16) | 100.0 (16) | +0.0 | [+0.0, +0.0] |
| response/spell_played_when_legal | 81.7 (164) | 86.3 (175) | +4.6 | [-1.1, +10.4] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.4 (172) | 100.0 (177) | +0.6 | [+0.0, +1.8] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 89.0 (172) | 93.8 (177) | **+4.8** | [+1.2, +8.8] |
| strategy/attacks_by_equipped_attacker | 34.1 (2291) | 36.9 (2323) | **+2.8** | [+1.3, +4.2] |
| strategy/face_target_share_when_both_legal | 68.3 (1340) | 71.2 (1256) | +2.9 | [-0.3, +6.4] |
| strategy/favorable_trade_taken_per_available_turn | 40.4 (272) | 35.1 (239) | -5.3 | [-12.0, +1.3] |
| strategy/spell_cast_per_legal_main_turn | 44.8 (29) | 45.5 (22) | +0.6 | [-17.4, +21.0] |

### lightning vs u8223 — argmax — free_draft|ALL

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.9 (192) | 25.0 (192) | **+6.1** | [+6.0, +6.2] |
| draft/mean_cost | 2.89 (9600) | 2.38 (9600) | **-0.5** | [-0.5, -0.5] |
| draft/normal_share | 65.5 (9600) | 60.0 (9600) | **-5.5** | [-5.7, -5.3] |
| draft/spell_share | 0.0 (9600) | 0.0 (9600) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.25 (192) | 18.00 (192) | **-2.2** | [-2.3, -2.2] |
| draft/weapon_share | 20.0 (9600) | 23.0 (9600) | **+3.0** | [+2.8, +3.2] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (188) | 0.0 (127) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (81) | 100.0 (195) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.3 (81) | 93.3 (195) | -3.0 | [-7.4, +1.1] |
| gate/Surge/portal_with_eligible_weapon | 18.5 (439) | 40.6 (480) | **+22.2** | [+13.8, +31.0] |
| gate/portal_per_legal_turn | 99.5 (630) | 99.0 (613) | -0.5 | [-1.5, +0.5] |
| ikz/held_at_end_of_own_turn | 18.2 (1267) | 16.2 (1254) | -2.0 | [-5.0, +1.1] |
| ikz/held_then_spent_in_opp_turn | 1.7 (230) | 13.8 (203) | **+12.1** | [+7.3, +17.2] |
| lightning/weapon_attach_per_legal_turn | 20.9 (976) | 22.5 (1006) | +1.6 | [-1.5, +4.8] |
| lightning/weapon_attach_to_entity_share | 0.0 (267) | 0.0 (330) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.60 (192) | 6.53 (192) | -0.1 | [-0.2, +0.1] |
| outcome/win | 58.3 (192) | 68.2 (192) | **+9.9** | [+2.1, +17.7] |
| outcome/win_vs_EARTH | 70.8 (48) | 68.8 (48) | -2.1 | [-18.3, +13.8] |
| outcome/win_vs_FIRE | 35.4 (48) | 43.8 (48) | +8.3 | [-10.4, +28.3] |
| outcome/win_vs_LIGHTNING | 56.2 (48) | 70.8 (48) | **+14.6** | [+1.7, +28.6] |
| outcome/win_vs_WATER | 70.8 (48) | 89.6 (48) | **+18.8** | [+6.7, +31.6] |
| response/any_nonblock_response_when_legal | 22.2 (18) | 82.4 (34) | **+60.1** | [+33.9, +81.4] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 95.3 (43) | 98.5 (66) | +3.1 | [-3.2, +10.6] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.4 (43) | 90.9 (66) | +2.5 | [-9.4, +12.8] |
| strategy/attacks_by_equipped_attacker | 10.9 (2296) | 15.1 (2423) | **+4.2** | [+2.3, +6.1] |
| strategy/face_target_share_when_both_legal | 60.9 (1504) | 68.8 (1454) | **+7.9** | [+4.5, +11.2] |
| strategy/favorable_trade_taken_per_available_turn | 44.9 (521) | 38.4 (419) | **-6.5** | [-11.0, -1.8] |

### lightning vs u8223 — argmax — free_draft|Stormchain/Piko

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.6 (48) | 25.0 (48) | **+5.4** | [+5.4, +5.4] |
| draft/mean_cost | 2.90 (2400) | 2.40 (2400) | **-0.5** | [-0.5, -0.5] |
| draft/normal_share | 64.0 (2400) | 60.0 (2400) | **-4.0** | [-4.0, -4.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 20.0 (2400) | 22.0 (2400) | **+2.0** | [+2.0, +2.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (90) | 0.0 (66) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 98.9 (91) | 98.5 (67) | -0.4 | [-4.1, +3.2] |
| ikz/held_at_end_of_own_turn | 22.8 (320) | 15.1 (317) | **-7.7** | [-13.8, -2.0] |
| ikz/held_then_spent_in_opp_turn | 2.7 (73) | 10.4 (48) | +7.7 | [-2.3, +20.5] |
| lightning/weapon_attach_per_legal_turn | 18.0 (250) | 24.0 (262) | **+6.0** | [+1.0, +11.6] |
| lightning/weapon_attach_to_entity_share | 0.0 (60) | 0.0 (87) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.67 (48) | 6.60 (48) | -0.1 | [-0.3, +0.2] |
| outcome/win | 66.7 (48) | 68.8 (48) | +2.1 | [-10.4, +14.6] |
| response/any_nonblock_response_when_legal | 40.0 (5) | 62.5 (8) | +22.5 | [-60.0, +75.0] |
| strategy/attacks_by_equipped_attacker | 7.1 (560) | 9.4 (594) | +2.3 | [-0.3, +4.7] |
| strategy/face_target_share_when_both_legal | 64.5 (383) | 73.7 (388) | **+9.2** | [+4.1, +14.9] |
| strategy/favorable_trade_taken_per_available_turn | 43.0 (142) | 30.4 (125) | **-12.6** | [-20.8, -4.8] |

### lightning vs u8223 — argmax — free_draft|Stormchain/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.3 (48) | 25.0 (48) | **+6.7** | [+6.7, +6.7] |
| draft/mean_cost | 2.92 (2400) | 2.40 (2400) | **-0.5** | [-0.5, -0.5] |
| draft/normal_share | 66.0 (2400) | 60.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 20.0 (2400) | 22.0 (2400) | **+2.0** | [+2.0, +2.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (98) | 0.0 (61) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (98) | 100.0 (61) | +0.0 | [+0.0, +0.0] |
| ikz/held_at_end_of_own_turn | 19.8 (318) | 19.6 (332) | -0.2 | [-6.8, +6.2] |
| ikz/held_then_spent_in_opp_turn | 3.2 (63) | 9.2 (65) | +6.1 | [-2.1, +14.9] |
| lightning/weapon_attach_per_legal_turn | 18.8 (245) | 22.5 (244) | +3.8 | [-2.8, +10.3] |
| lightning/weapon_attach_to_entity_share | 0.0 (57) | 0.0 (75) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.62 (48) | 6.92 (48) | +0.3 | [+0.0, +0.6] |
| outcome/win | 56.2 (48) | 62.5 (48) | +6.2 | [-10.4, +22.9] |
| strategy/attacks_by_equipped_attacker | 6.9 (567) | 8.2 (612) | +1.3 | [-1.5, +4.3] |
| strategy/face_target_share_when_both_legal | 56.9 (355) | 67.1 (386) | **+10.2** | [+2.3, +17.3] |
| strategy/favorable_trade_taken_per_available_turn | 46.2 (130) | 43.6 (110) | -2.5 | [-11.0, +6.0] |

### lightning vs u8223 — argmax — free_draft|Surge/Piko

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.3 (48) | 25.0 (48) | **+6.7** | [+6.7, +6.7] |
| draft/mean_cost | 2.92 (2400) | 2.36 (2400) | **-0.6** | [-0.6, -0.6] |
| draft/normal_share | 66.0 (2400) | 60.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 20.0 (2400) | 24.0 (2400) | **+4.0** | [+4.0, +4.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (27) | 100.0 (82) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.3 (27) | 92.7 (82) | -3.6 | [-10.4, +3.3] |
| gate/Surge/portal_with_eligible_weapon | 12.4 (218) | 33.3 (246) | **+20.9** | [+9.9, +32.4] |
| gate/portal_per_legal_turn | 99.1 (220) | 99.2 (248) | +0.1 | [-1.6, +1.8] |
| ikz/held_at_end_of_own_turn | 16.2 (314) | 16.9 (308) | +0.6 | [-5.3, +6.6] |
| ikz/held_then_spent_in_opp_turn | 0.0 (51) | 11.5 (52) | **+11.5** | [+3.8, +17.7] |
| lightning/weapon_attach_per_legal_turn | 20.2 (248) | 22.6 (243) | +2.5 | [-4.1, +8.7] |
| lightning/weapon_attach_to_entity_share | 0.0 (69) | 0.0 (87) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.54 (48) | 6.42 (48) | -0.1 | [-0.5, +0.2] |
| outcome/win | 50.0 (48) | 68.8 (48) | +18.8 | [+0.0, +37.5] |
| response/any_nonblock_response_when_legal | 0.0 (1) | 75.0 (8) | **+75.0** | [+40.0, +100.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 88.2 (17) | 96.7 (30) | +8.4 | [-6.9, +24.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.2 (17) | 96.7 (30) | +8.4 | [-6.9, +24.0] |
| strategy/attacks_by_equipped_attacker | 12.4 (555) | 19.3 (600) | **+6.9** | [+3.2, +10.6] |
| strategy/face_target_share_when_both_legal | 60.8 (380) | 68.4 (361) | **+7.6** | [+1.0, +14.5] |
| strategy/favorable_trade_taken_per_available_turn | 37.4 (123) | 40.9 (93) | +3.5 | [-4.1, +12.1] |

### lightning vs u8223 — argmax — free_draft|Surge/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.6 (48) | 25.0 (48) | **+5.4** | [+5.4, +5.4] |
| draft/mean_cost | 2.82 (2400) | 2.36 (2400) | **-0.5** | [-0.5, -0.5] |
| draft/normal_share | 66.0 (2400) | 60.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 21.00 (48) | 18.00 (48) | **-3.0** | [-3.0, -3.0] |
| draft/weapon_share | 20.0 (2400) | 24.0 (2400) | **+4.0** | [+4.0, +4.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (54) | 100.0 (113) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.3 (54) | 93.8 (113) | -2.5 | [-8.6, +3.0] |
| gate/Surge/portal_with_eligible_weapon | 24.4 (221) | 48.3 (234) | **+23.9** | [+11.8, +35.4] |
| gate/portal_per_legal_turn | 100.0 (221) | 98.7 (237) | -1.3 | [-3.2, +0.0] |
| ikz/held_at_end_of_own_turn | 13.7 (315) | 12.8 (297) | -0.9 | [-4.6, +3.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (43) | 28.9 (38) | **+28.9** | [+17.9, +40.5] |
| lightning/weapon_attach_per_legal_turn | 27.0 (233) | 20.6 (257) | **-6.4** | [-12.6, -0.6] |
| lightning/weapon_attach_to_entity_share | 0.0 (81) | 0.0 (81) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.56 (48) | 6.19 (48) | -0.4 | [-0.8, +0.0] |
| outcome/win | 60.4 (48) | 72.9 (48) | +12.5 | [+0.0, +27.1] |
| response/any_nonblock_response_when_legal | 0.0 (8) | 100.0 (11) | **+100.0** | [+100.0, +100.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 100.0 (26) | 100.0 (36) | +0.0 | [+0.0, +0.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.5 (26) | 86.1 (36) | -2.4 | [-16.6, +10.9] |
| strategy/attacks_by_equipped_attacker | 16.6 (614) | 23.2 (617) | **+6.6** | [+1.3, +11.1] |
| strategy/face_target_share_when_both_legal | 61.1 (386) | 65.5 (319) | +4.4 | [-3.4, +12.6] |
| strategy/favorable_trade_taken_per_available_turn | 53.2 (126) | 40.7 (91) | -12.5 | [-26.4, +0.1] |

### lightning vs u8223 — sample — all|ALL

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 24.4 (41) | 27.7 (47) | +3.3 | [-14.8, +20.0] |
| draft/max_wjaccard_to_curated | 13.2 (192) | 15.4 (192) | **+2.2** | [+1.9, +2.5] |
| draft/mean_cost | 2.73 (9600) | 2.70 (9600) | **-0.0** | [-0.1, -0.0] |
| draft/normal_share | 65.1 (9600) | 62.8 (9600) | **-2.3** | [-2.8, -1.8] |
| draft/spell_share | 5.1 (9600) | 4.3 (9600) | **-0.9** | [-1.2, -0.5] |
| draft/unique_cards | 33.06 (192) | 32.48 (192) | **-0.6** | [-0.9, -0.3] |
| draft/weapon_share | 17.3 (9600) | 18.6 (9600) | **+1.3** | [+0.8, +1.8] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (423) | 0.0 (270) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (694) | 100.0 (798) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 98.8 (694) | 97.2 (798) | **-1.6** | [-2.9, -0.2] |
| gate/Surge/portal_with_eligible_weapon | 51.4 (1350) | 57.1 (1397) | **+5.7** | [+2.1, +9.4] |
| gate/portal_per_legal_turn | 99.8 (1776) | 99.7 (1672) | -0.1 | [-0.4, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.3 (44) | 19.6 (46) | -7.7 | [-23.1, +9.0] |
| ikz/held_at_end_of_own_turn | 26.5 (2864) | 28.4 (2889) | +1.9 | [-0.1, +3.7] |
| ikz/held_then_spent_in_opp_turn | 32.4 (759) | 34.3 (820) | +1.9 | [-1.3, +5.2] |
| lightning/weapon_attach_per_legal_turn | 27.4 (2280) | 28.1 (2359) | +0.7 | [-1.0, +2.4] |
| lightning/weapon_attach_to_entity_share | 0.4 (1028) | 0.2 (1082) | -0.2 | [-0.6, +0.2] |
| outcome/own_turns | 6.39 (448) | 6.45 (448) | +0.1 | [-0.1, +0.2] |
| outcome/win | 43.5 (448) | 48.4 (448) | **+4.9** | [+0.4, +9.6] |
| outcome/win_vs_EARTH | 44.6 (112) | 46.4 (112) | +1.8 | [-7.5, +11.5] |
| outcome/win_vs_FIRE | 26.8 (112) | 30.4 (112) | +3.6 | [-4.4, +11.5] |
| outcome/win_vs_LIGHTNING | 60.7 (112) | 61.6 (112) | +0.9 | [-8.5, +9.6] |
| outcome/win_vs_WATER | 42.0 (112) | 55.4 (112) | **+13.4** | [+3.3, +23.6] |
| response/any_nonblock_response_when_legal | 79.5 (337) | 85.8 (366) | +6.3 | [-0.4, +12.7] |
| response/defender_declared_when_legal | 39.0 (105) | 50.5 (97) | +11.5 | [-5.9, +27.3] |
| response/spell_played_when_legal | 87.5 (264) | 90.8 (295) | +3.3 | [-0.5, +8.2] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.5 (206) | 99.1 (215) | -0.4 | [-2.2, +1.1] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 90.3 (206) | 93.0 (215) | +2.7 | [-1.5, +6.7] |
| strategy/attacks_by_equipped_attacker | 22.7 (5011) | 24.1 (5187) | **+1.5** | [+0.3, +2.7] |
| strategy/face_target_share_when_both_legal | 66.1 (3183) | 67.1 (3209) | +1.0 | [-1.2, +3.2] |
| strategy/favorable_trade_taken_per_available_turn | 44.1 (844) | 41.1 (813) | -3.0 | [-7.5, +1.6] |
| strategy/spell_cast_per_legal_main_turn | 13.3 (278) | 18.4 (174) | +5.1 | [-0.2, +11.1] |

### lightning vs u8223 — sample — all|Stormchain/Piko

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 11.3 (48) | 13.5 (48) | **+2.2** | [+1.7, +2.7] |
| draft/mean_cost | 2.75 (2400) | 2.73 (2400) | -0.0 | [-0.1, +0.0] |
| draft/normal_share | 64.6 (2400) | 62.6 (2400) | **-2.0** | [-2.8, -1.2] |
| draft/spell_share | 4.9 (2400) | 4.2 (2400) | **-0.7** | [-1.3, -0.1] |
| draft/unique_cards | 33.77 (48) | 32.52 (48) | **-1.2** | [-1.8, -0.7] |
| draft/weapon_share | 18.1 (2400) | 18.6 (2400) | +0.5 | [-0.3, +1.3] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (106) | 0.0 (72) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 99.1 (107) | 100.0 (72) | +0.9 | [+0.0, +2.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 26.3 (19) | 11.1 (9) | -15.2 | [-35.7, +26.5] |
| ikz/held_at_end_of_own_turn | 21.5 (331) | 22.8 (324) | +1.4 | [-7.0, +9.3] |
| ikz/held_then_spent_in_opp_turn | 5.6 (71) | 8.1 (74) | +2.5 | [-5.1, +11.4] |
| lightning/weapon_attach_per_legal_turn | 23.1 (242) | 21.8 (257) | -1.4 | [-8.3, +5.8] |
| lightning/weapon_attach_to_entity_share | 0.0 (82) | 0.0 (79) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.90 (48) | 6.75 (48) | -0.1 | [-0.6, +0.3] |
| outcome/win | 43.8 (48) | 50.0 (48) | +6.2 | [-12.5, +27.1] |
| response/any_nonblock_response_when_legal | 71.4 (7) | 72.7 (11) | +1.3 | [-40.0, +66.7] |
| response/defender_declared_when_legal | 20.0 (5) | 25.0 (16) | +5.0 | [-10.0, +60.0] |
| strategy/attacks_by_equipped_attacker | 9.3 (592) | 8.8 (548) | -0.5 | [-4.0, +2.8] |
| strategy/face_target_share_when_both_legal | 61.3 (406) | 60.5 (370) | -0.8 | [-8.6, +6.9] |
| strategy/favorable_trade_taken_per_available_turn | 52.9 (138) | 42.8 (138) | -10.1 | [-24.0, +2.4] |
| strategy/spell_cast_per_legal_main_turn | 11.5 (61) | 13.0 (23) | +1.6 | [-13.6, +27.6] |

### lightning vs u8223 — sample — all|Stormchain/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 4.0 (25) | 16.0 (25) | +12.0 | [-4.0, +25.9] |
| draft/max_wjaccard_to_curated | 11.2 (48) | 13.4 (48) | **+2.2** | [+1.6, +2.8] |
| draft/mean_cost | 2.71 (2400) | 2.68 (2400) | -0.0 | [-0.1, +0.0] |
| draft/normal_share | 66.1 (2400) | 63.2 (2400) | **-2.8** | [-3.8, -1.9] |
| draft/spell_share | 5.1 (2400) | 4.2 (2400) | **-0.8** | [-1.4, -0.2] |
| draft/unique_cards | 32.69 (48) | 32.40 (48) | -0.3 | [-0.8, +0.2] |
| draft/weapon_share | 16.0 (2400) | 18.4 (2400) | **+2.4** | [+1.4, +3.4] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (317) | 0.0 (198) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (317) | 100.0 (198) | +0.0 | [+0.0, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 11.1 (9) | 22.2 (18) | +11.1 | [-80.0, +50.0] |
| ikz/held_at_end_of_own_turn | 30.4 (753) | 34.7 (761) | **+4.3** | [+1.4, +7.3] |
| ikz/held_then_spent_in_opp_turn | 39.7 (229) | 40.2 (264) | +0.4 | [-5.0, +6.4] |
| lightning/weapon_attach_per_legal_turn | 28.4 (595) | 31.2 (612) | +2.8 | [-0.2, +5.6] |
| lightning/weapon_attach_to_entity_share | 0.0 (270) | 0.0 (299) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.72 (112) | 6.79 (112) | +0.1 | [-0.1, +0.3] |
| outcome/win | 29.5 (112) | 27.7 (112) | -1.8 | [-8.9, +5.4] |
| response/any_nonblock_response_when_legal | 84.9 (119) | 91.3 (126) | +6.4 | [-3.9, +15.7] |
| response/defender_declared_when_legal | 39.1 (64) | 46.3 (54) | +7.2 | [-8.1, +23.6] |
| response/spell_played_when_legal | 95.8 (96) | 94.4 (107) | -1.4 | [-4.2, +0.6] |
| strategy/attacks_by_equipped_attacker | 14.0 (1119) | 16.6 (1106) | **+2.6** | [+0.8, +4.4] |
| strategy/face_target_share_when_both_legal | 66.3 (716) | 64.3 (722) | -2.1 | [-7.0, +2.9] |
| strategy/favorable_trade_taken_per_available_turn | 41.6 (202) | 44.7 (190) | +3.2 | [-7.1, +13.2] |
| strategy/spell_cast_per_legal_main_turn | 5.7 (70) | 12.0 (50) | +6.3 | [-1.0, +15.3] |

### lightning vs u8223 — sample — all|Surge/Piko

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 14.5 (48) | 17.0 (48) | **+2.5** | [+1.8, +3.1] |
| draft/mean_cost | 2.76 (2400) | 2.71 (2400) | **-0.0** | [-0.1, -0.0] |
| draft/normal_share | 65.0 (2400) | 62.4 (2400) | **-2.5** | [-3.5, -1.6] |
| draft/spell_share | 5.6 (2400) | 4.4 (2400) | **-1.3** | [-1.9, -0.5] |
| draft/unique_cards | 33.19 (48) | 32.71 (48) | -0.5 | [-1.1, +0.2] |
| draft/weapon_share | 17.9 (2400) | 19.2 (2400) | **+1.3** | [+0.4, +2.4] |
| gate/Surge/eligible_portal_then_equip | 100.0 (24) | 100.0 (52) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 95.8 (24) | 92.3 (52) | -3.5 | [-14.7, +7.0] |
| gate/Surge/portal_with_eligible_weapon | 10.4 (230) | 22.0 (236) | **+11.6** | [+0.3, +23.1] |
| gate/portal_per_legal_turn | 100.0 (230) | 99.6 (237) | -0.4 | [-1.3, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 36.4 (11) | 7.7 (13) | **-28.7** | [-88.9, -10.7] |
| ikz/held_at_end_of_own_turn | 21.0 (305) | 20.4 (319) | -0.6 | [-6.2, +5.2] |
| ikz/held_then_spent_in_opp_turn | 7.8 (64) | 9.2 (65) | +1.4 | [-9.7, +12.1] |
| lightning/weapon_attach_per_legal_turn | 24.2 (207) | 26.8 (205) | +2.7 | [-4.6, +9.4] |
| lightning/weapon_attach_to_entity_share | 0.0 (64) | 0.0 (73) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.35 (48) | 6.65 (48) | +0.3 | [-0.1, +0.7] |
| outcome/win | 39.6 (48) | 52.1 (48) | +12.5 | [-4.2, +29.2] |
| response/any_nonblock_response_when_legal | 55.6 (9) | 53.8 (13) | -1.7 | [-42.9, +30.6] |
| response/defender_declared_when_legal | 0.0 (1) | 41.7 (12) | **+41.7** | [+25.0, +66.7] |
| response/spell_played_when_legal | 55.6 (9) | 100.0 (5) | +44.4 | [+0.0, +80.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 100.0 (13) | 94.7 (19) | -5.3 | [-16.7, +0.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 92.3 (13) | 89.5 (19) | -2.8 | [-25.0, +15.5] |
| strategy/attacks_by_equipped_attacker | 12.8 (507) | 15.7 (573) | +2.9 | [-2.7, +8.2] |
| strategy/face_target_share_when_both_legal | 61.7 (376) | 61.1 (404) | -0.6 | [-7.5, +5.4] |
| strategy/favorable_trade_taken_per_available_turn | 38.5 (122) | 49.5 (109) | +11.0 | [+0.0, +22.0] |
| strategy/spell_cast_per_legal_main_turn | 8.0 (75) | 2.6 (38) | -5.4 | [-13.1, +2.3] |

### lightning vs u8223 — sample — all|Surge/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 60.0 (15) | 57.1 (14) | -2.9 | [-41.1, +34.5] |
| draft/max_wjaccard_to_curated | 15.8 (48) | 17.7 (48) | **+2.0** | [+1.3, +2.6] |
| draft/mean_cost | 2.70 (2400) | 2.67 (2400) | -0.0 | [-0.1, +0.0] |
| draft/normal_share | 64.6 (2400) | 62.8 (2400) | **-1.8** | [-2.8, -0.8] |
| draft/spell_share | 5.0 (2400) | 4.2 (2400) | -0.7 | [-1.4, +0.0] |
| draft/unique_cards | 32.60 (48) | 32.31 (48) | -0.3 | [-0.9, +0.3] |
| draft/weapon_share | 17.4 (2400) | 18.4 (2400) | **+1.0** | [+0.1, +1.9] |
| gate/Surge/eligible_portal_then_equip | 100.0 (670) | 100.0 (746) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 99.0 (670) | 97.6 (746) | **-1.4** | [-2.6, -0.1] |
| gate/Surge/portal_with_eligible_weapon | 59.8 (1120) | 64.3 (1161) | **+4.4** | [+1.1, +7.7] |
| gate/portal_per_legal_turn | 99.8 (1122) | 99.7 (1165) | -0.2 | [-0.5, +0.2] |
| ikz/held_at_end_of_own_turn | 26.8 (1475) | 28.1 (1485) | +1.3 | [-1.3, +3.9] |
| ikz/held_then_spent_in_opp_turn | 37.0 (395) | 39.1 (417) | +2.1 | [-1.7, +6.1] |
| lightning/weapon_attach_per_legal_turn | 28.3 (1236) | 28.1 (1285) | -0.2 | [-2.3, +1.9] |
| lightning/weapon_attach_to_entity_share | 0.7 (612) | 0.3 (631) | -0.3 | [-1.0, +0.3] |
| outcome/own_turns | 6.15 (240) | 6.19 (240) | +0.0 | [-0.1, +0.2] |
| outcome/win | 50.8 (240) | 57.1 (240) | +6.2 | [+0.0, +12.5] |
| response/any_nonblock_response_when_legal | 77.7 (202) | 85.2 (216) | +7.5 | [-0.9, +16.2] |
| response/defender_declared_when_legal | 42.9 (35) | 100.0 (15) | **+57.1** | [+5.9, +76.6] |
| response/spell_played_when_legal | 84.5 (155) | 88.3 (179) | +3.8 | [-1.3, +9.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.5 (193) | 99.5 (196) | +0.0 | [-1.5, +1.5] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 90.2 (193) | 93.4 (196) | +3.2 | [-0.7, +7.2] |
| strategy/attacks_by_equipped_attacker | 30.8 (2793) | 31.4 (2960) | +0.7 | [-0.8, +2.2] |
| strategy/face_target_share_when_both_legal | 68.1 (1685) | 71.1 (1713) | **+3.0** | [+0.2, +5.9] |
| strategy/favorable_trade_taken_per_available_turn | 44.0 (382) | 36.2 (376) | **-7.8** | [-13.9, -1.8] |
| strategy/spell_cast_per_legal_main_turn | 27.8 (72) | 34.9 (63) | +7.1 | [-6.6, +18.2] |

### lightning vs u8223 — sample — fixed|ALL

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 29.0 (31) | 30.3 (33) | +1.3 | [-20.6, +21.2] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (209) | 0.0 (149) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (638) | 100.0 (706) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 99.1 (638) | 97.7 (706) | **-1.3** | [-2.6, -0.0] |
| gate/Surge/portal_with_eligible_weapon | 70.2 (909) | 76.7 (921) | **+6.5** | [+3.2, +9.5] |
| gate/portal_per_legal_turn | 99.8 (1120) | 99.8 (1072) | -0.0 | [-0.3, +0.2] |
| ikz/held_at_end_of_own_turn | 31.0 (1590) | 33.4 (1601) | **+2.3** | [+0.0, +4.8] |
| ikz/held_then_spent_in_opp_turn | 46.7 (493) | 47.9 (534) | +1.3 | [-2.5, +5.2] |
| lightning/weapon_attach_per_legal_turn | 30.7 (1409) | 31.2 (1437) | +0.5 | [-1.2, +2.4] |
| lightning/weapon_attach_to_entity_share | 0.5 (764) | 0.3 (794) | -0.3 | [-0.8, +0.2] |
| outcome/own_turns | 6.21 (256) | 6.25 (256) | +0.0 | [-0.1, +0.2] |
| outcome/win | 44.5 (256) | 47.7 (256) | +3.1 | [-2.7, +8.6] |
| outcome/win_vs_EARTH | 48.4 (64) | 46.9 (64) | -1.6 | [-13.5, +10.6] |
| outcome/win_vs_FIRE | 25.0 (64) | 26.6 (64) | +1.6 | [-7.8, +10.9] |
| outcome/win_vs_LIGHTNING | 64.1 (64) | 67.2 (64) | +3.1 | [-8.1, +13.3] |
| outcome/win_vs_WATER | 40.6 (64) | 50.0 (64) | +9.4 | [-3.4, +21.7] |
| response/any_nonblock_response_when_legal | 80.8 (307) | 88.3 (324) | **+7.5** | [+1.5, +14.0] |
| response/defender_declared_when_legal | 51.7 (60) | 61.8 (55) | +10.2 | [-8.0, +25.5] |
| response/spell_played_when_legal | 88.6 (245) | 90.1 (274) | +1.6 | [-1.9, +5.3] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.4 (176) | 100.0 (178) | +0.6 | [+0.0, +1.8] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 90.9 (176) | 94.4 (178) | +3.5 | [-0.0, +7.4] |
| strategy/attacks_by_equipped_attacker | 32.0 (2833) | 33.5 (2944) | **+1.5** | [+0.3, +2.7] |
| strategy/face_target_share_when_both_legal | 70.9 (1698) | 71.9 (1658) | +1.0 | [-1.8, +3.8] |
| strategy/favorable_trade_taken_per_available_turn | 39.7 (350) | 34.0 (324) | **-5.8** | [-11.5, -0.1] |
| strategy/spell_cast_per_legal_main_turn | 60.0 (25) | 57.1 (28) | -2.9 | [-23.6, +16.7] |

### lightning vs u8223 — sample — fixed|Stormchain/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (19) | 10.0 (20) | +10.0 | [+0.0, +20.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (209) | 0.0 (149) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (209) | 100.0 (149) | +0.0 | [+0.0, +0.0] |
| ikz/held_at_end_of_own_turn | 39.5 (430) | 44.3 (447) | **+4.8** | [+1.5, +8.2] |
| ikz/held_then_spent_in_opp_turn | 52.4 (170) | 51.5 (198) | -0.8 | [-7.1, +5.6] |
| lightning/weapon_attach_per_legal_turn | 33.6 (375) | 37.4 (382) | **+3.8** | [+0.4, +7.1] |
| lightning/weapon_attach_to_entity_share | 0.0 (216) | 0.0 (235) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.72 (64) | 6.98 (64) | **+0.3** | [+0.0, +0.5] |
| outcome/win | 20.3 (64) | 17.2 (64) | -3.1 | [-12.5, +6.2] |
| response/any_nonblock_response_when_legal | 84.3 (115) | 92.5 (120) | +8.2 | [-1.8, +17.8] |
| response/defender_declared_when_legal | 39.6 (48) | 48.8 (41) | +9.2 | [-10.1, +26.6] |
| response/spell_played_when_legal | 95.7 (94) | 94.2 (103) | -1.6 | [-4.8, +0.5] |
| strategy/attacks_by_equipped_attacker | 20.8 (577) | 23.2 (599) | **+2.4** | [+0.4, +4.4] |
| strategy/face_target_share_when_both_legal | 76.5 (374) | 71.9 (360) | -4.5 | [-9.4, +0.2] |
| strategy/favorable_trade_taken_per_available_turn | 29.6 (81) | 32.4 (71) | +2.8 | [-4.5, +11.2] |

### lightning vs u8223 — sample — fixed|Surge/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 75.0 (12) | 61.5 (13) | -13.5 | [-47.7, +21.4] |
| gate/Surge/eligible_portal_then_equip | 100.0 (638) | 100.0 (706) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 99.1 (638) | 97.7 (706) | **-1.3** | [-2.5, -0.2] |
| gate/Surge/portal_with_eligible_weapon | 70.2 (909) | 76.7 (921) | **+6.5** | [+3.4, +9.5] |
| gate/portal_per_legal_turn | 99.8 (911) | 99.8 (923) | +0.0 | [-0.3, +0.3] |
| ikz/held_at_end_of_own_turn | 27.8 (1160) | 29.1 (1154) | +1.3 | [-1.6, +4.2] |
| ikz/held_then_spent_in_opp_turn | 43.7 (323) | 45.8 (336) | +2.2 | [-2.2, +6.8] |
| lightning/weapon_attach_per_legal_turn | 29.6 (1034) | 28.9 (1055) | -0.7 | [-2.7, +1.4] |
| lightning/weapon_attach_to_entity_share | 0.7 (548) | 0.4 (559) | -0.4 | [-1.2, +0.3] |
| outcome/own_turns | 6.04 (192) | 6.01 (192) | -0.0 | [-0.2, +0.1] |
| outcome/win | 52.6 (192) | 57.8 (192) | +5.2 | [-1.6, +12.0] |
| response/any_nonblock_response_when_legal | 78.6 (192) | 85.8 (204) | +7.1 | [-0.7, +15.6] |
| response/defender_declared_when_legal | 100.0 (12) | 100.0 (14) | +0.0 | [+0.0, +0.0] |
| response/spell_played_when_legal | 84.1 (151) | 87.7 (171) | +3.6 | [-1.6, +8.8] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.4 (176) | 100.0 (178) | +0.6 | [+0.0, +1.8] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 90.9 (176) | 94.4 (178) | +3.5 | [+0.0, +7.4] |
| strategy/attacks_by_equipped_attacker | 34.8 (2256) | 36.1 (2345) | +1.2 | [-0.1, +2.6] |
| strategy/face_target_share_when_both_legal | 69.3 (1324) | 71.9 (1298) | +2.5 | [-0.6, +5.7] |
| strategy/favorable_trade_taken_per_available_turn | 42.8 (269) | 34.4 (253) | **-8.4** | [-15.4, -1.4] |
| strategy/spell_cast_per_legal_main_turn | 60.0 (25) | 57.1 (28) | -2.9 | [-23.3, +16.2] |

### lightning vs u8223 — sample — free_draft|ALL

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 10.0 (10) | 21.4 (14) | +11.4 | [-13.9, +40.0] |
| draft/max_wjaccard_to_curated | 13.2 (192) | 15.4 (192) | **+2.2** | [+1.9, +2.5] |
| draft/mean_cost | 2.73 (9600) | 2.70 (9600) | **-0.0** | [-0.1, -0.0] |
| draft/normal_share | 65.1 (9600) | 62.8 (9600) | **-2.3** | [-2.7, -1.8] |
| draft/spell_share | 5.1 (9600) | 4.3 (9600) | **-0.9** | [-1.2, -0.5] |
| draft/unique_cards | 33.06 (192) | 32.48 (192) | **-0.6** | [-0.9, -0.3] |
| draft/weapon_share | 17.3 (9600) | 18.6 (9600) | **+1.3** | [+0.8, +1.8] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (214) | 0.0 (121) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (56) | 100.0 (92) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.4 (56) | 93.5 (92) | -3.0 | [-9.4, +3.9] |
| gate/Surge/portal_with_eligible_weapon | 12.7 (441) | 19.3 (476) | +6.6 | [-1.3, +14.2] |
| gate/portal_per_legal_turn | 99.8 (656) | 99.5 (600) | -0.3 | [-1.0, +0.3] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.3 (44) | 19.6 (46) | -7.7 | [-23.7, +8.8] |
| ikz/held_at_end_of_own_turn | 20.9 (1274) | 22.2 (1288) | +1.3 | [-1.8, +4.4] |
| ikz/held_then_spent_in_opp_turn | 6.0 (266) | 8.7 (286) | +2.7 | [-1.3, +7.0] |
| lightning/weapon_attach_per_legal_turn | 22.2 (871) | 23.3 (922) | +1.2 | [-2.0, +4.4] |
| lightning/weapon_attach_to_entity_share | 0.0 (264) | 0.0 (288) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.64 (192) | 6.71 (192) | +0.1 | [-0.1, +0.3] |
| outcome/win | 42.2 (192) | 49.5 (192) | +7.3 | [+0.0, +14.6] |
| outcome/win_vs_EARTH | 39.6 (48) | 45.8 (48) | +6.2 | [-9.3, +21.7] |
| outcome/win_vs_FIRE | 29.2 (48) | 35.4 (48) | +6.2 | [-8.1, +19.6] |
| outcome/win_vs_LIGHTNING | 56.2 (48) | 54.2 (48) | -2.1 | [-17.6, +14.3] |
| outcome/win_vs_WATER | 43.8 (48) | 62.5 (48) | **+18.8** | [+2.6, +34.8] |
| response/any_nonblock_response_when_legal | 66.7 (30) | 66.7 (42) | +0.0 | [-27.9, +25.4] |
| response/defender_declared_when_legal | 22.2 (45) | 35.7 (42) | +13.5 | [-8.8, +42.3] |
| response/spell_played_when_legal | 73.7 (19) | 100.0 (21) | +26.3 | [+0.0, +57.1] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 100.0 (30) | 94.6 (37) | -5.4 | [-13.2, +0.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 86.7 (30) | 86.5 (37) | -0.2 | [-16.7, +16.9] |
| strategy/attacks_by_equipped_attacker | 10.6 (2178) | 11.9 (2243) | +1.3 | [-0.7, +3.5] |
| strategy/face_target_share_when_both_legal | 60.5 (1485) | 62.0 (1551) | +1.4 | [-2.2, +5.1] |
| strategy/favorable_trade_taken_per_available_turn | 47.2 (494) | 45.8 (489) | -1.4 | [-7.8, +5.4] |
| strategy/spell_cast_per_legal_main_turn | 8.7 (253) | 11.0 (146) | +2.3 | [-2.8, +7.6] |

### lightning vs u8223 — sample — free_draft|Stormchain/Piko

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 11.3 (48) | 13.5 (48) | **+2.2** | [+1.7, +2.7] |
| draft/mean_cost | 2.75 (2400) | 2.73 (2400) | -0.0 | [-0.1, +0.0] |
| draft/normal_share | 64.6 (2400) | 62.6 (2400) | **-2.0** | [-2.8, -1.2] |
| draft/spell_share | 4.9 (2400) | 4.2 (2400) | **-0.7** | [-1.3, -0.1] |
| draft/unique_cards | 33.77 (48) | 32.52 (48) | **-1.2** | [-1.8, -0.7] |
| draft/weapon_share | 18.1 (2400) | 18.6 (2400) | +0.5 | [-0.3, +1.3] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (106) | 0.0 (72) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 99.1 (107) | 100.0 (72) | +0.9 | [+0.0, +2.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 26.3 (19) | 11.1 (9) | -15.2 | [-35.7, +26.5] |
| ikz/held_at_end_of_own_turn | 21.5 (331) | 22.8 (324) | +1.4 | [-7.0, +9.3] |
| ikz/held_then_spent_in_opp_turn | 5.6 (71) | 8.1 (74) | +2.5 | [-5.1, +11.4] |
| lightning/weapon_attach_per_legal_turn | 23.1 (242) | 21.8 (257) | -1.4 | [-8.3, +5.8] |
| lightning/weapon_attach_to_entity_share | 0.0 (82) | 0.0 (79) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.90 (48) | 6.75 (48) | -0.1 | [-0.6, +0.3] |
| outcome/win | 43.8 (48) | 50.0 (48) | +6.2 | [-12.5, +27.1] |
| response/any_nonblock_response_when_legal | 71.4 (7) | 72.7 (11) | +1.3 | [-40.0, +66.7] |
| response/defender_declared_when_legal | 20.0 (5) | 25.0 (16) | +5.0 | [-10.0, +60.0] |
| strategy/attacks_by_equipped_attacker | 9.3 (592) | 8.8 (548) | -0.5 | [-4.0, +2.8] |
| strategy/face_target_share_when_both_legal | 61.3 (406) | 60.5 (370) | -0.8 | [-8.6, +6.9] |
| strategy/favorable_trade_taken_per_available_turn | 52.9 (138) | 42.8 (138) | -10.1 | [-24.0, +2.4] |
| strategy/spell_cast_per_legal_main_turn | 11.5 (61) | 13.0 (23) | +1.6 | [-13.6, +27.6] |

### lightning vs u8223 — sample — free_draft|Stormchain/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 11.2 (48) | 13.4 (48) | **+2.2** | [+1.7, +2.8] |
| draft/mean_cost | 2.71 (2400) | 2.68 (2400) | -0.0 | [-0.1, +0.0] |
| draft/normal_share | 66.1 (2400) | 63.2 (2400) | **-2.8** | [-3.8, -1.9] |
| draft/spell_share | 5.1 (2400) | 4.2 (2400) | **-0.8** | [-1.4, -0.2] |
| draft/unique_cards | 32.69 (48) | 32.40 (48) | -0.3 | [-0.8, +0.2] |
| draft/weapon_share | 16.0 (2400) | 18.4 (2400) | **+2.4** | [+1.4, +3.3] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (108) | 0.0 (49) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (108) | 100.0 (49) | +0.0 | [+0.0, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 11.1 (9) | 22.2 (18) | +11.1 | [-80.0, +50.0] |
| ikz/held_at_end_of_own_turn | 18.3 (323) | 21.0 (314) | +2.8 | [-2.1, +7.4] |
| ikz/held_then_spent_in_opp_turn | 3.4 (59) | 6.1 (66) | +2.7 | [-4.3, +11.3] |
| lightning/weapon_attach_per_legal_turn | 19.5 (220) | 20.9 (230) | +1.3 | [-4.1, +6.3] |
| lightning/weapon_attach_to_entity_share | 0.0 (54) | 0.0 (64) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.73 (48) | 6.54 (48) | -0.2 | [-0.5, +0.1] |
| outcome/win | 41.7 (48) | 41.7 (48) | +0.0 | [-10.4, +10.4] |
| response/defender_declared_when_legal | 37.5 (16) | 38.5 (13) | +1.0 | [-26.7, +66.7] |
| strategy/attacks_by_equipped_attacker | 6.8 (542) | 8.9 (507) | +2.0 | [-0.9, +4.6] |
| strategy/face_target_share_when_both_legal | 55.3 (342) | 56.6 (362) | +1.4 | [-7.2, +10.6] |
| strategy/favorable_trade_taken_per_available_turn | 49.6 (121) | 52.1 (119) | +2.5 | [-11.6, +17.2] |
| strategy/spell_cast_per_legal_main_turn | 5.7 (70) | 12.0 (50) | +6.3 | [-0.6, +15.2] |

### lightning vs u8223 — sample — free_draft|Surge/Piko

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 14.5 (48) | 17.0 (48) | **+2.5** | [+1.8, +3.1] |
| draft/mean_cost | 2.76 (2400) | 2.71 (2400) | **-0.0** | [-0.1, -0.0] |
| draft/normal_share | 65.0 (2400) | 62.4 (2400) | **-2.5** | [-3.5, -1.6] |
| draft/spell_share | 5.6 (2400) | 4.4 (2400) | **-1.3** | [-1.9, -0.5] |
| draft/unique_cards | 33.19 (48) | 32.71 (48) | -0.5 | [-1.1, +0.2] |
| draft/weapon_share | 17.9 (2400) | 19.2 (2400) | **+1.3** | [+0.4, +2.4] |
| gate/Surge/eligible_portal_then_equip | 100.0 (24) | 100.0 (52) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 95.8 (24) | 92.3 (52) | -3.5 | [-14.7, +7.0] |
| gate/Surge/portal_with_eligible_weapon | 10.4 (230) | 22.0 (236) | **+11.6** | [+0.3, +23.1] |
| gate/portal_per_legal_turn | 100.0 (230) | 99.6 (237) | -0.4 | [-1.3, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 36.4 (11) | 7.7 (13) | **-28.7** | [-88.9, -10.7] |
| ikz/held_at_end_of_own_turn | 21.0 (305) | 20.4 (319) | -0.6 | [-6.2, +5.2] |
| ikz/held_then_spent_in_opp_turn | 7.8 (64) | 9.2 (65) | +1.4 | [-9.7, +12.1] |
| lightning/weapon_attach_per_legal_turn | 24.2 (207) | 26.8 (205) | +2.7 | [-4.6, +9.4] |
| lightning/weapon_attach_to_entity_share | 0.0 (64) | 0.0 (73) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.35 (48) | 6.65 (48) | +0.3 | [-0.1, +0.7] |
| outcome/win | 39.6 (48) | 52.1 (48) | +12.5 | [-4.2, +29.2] |
| response/any_nonblock_response_when_legal | 55.6 (9) | 53.8 (13) | -1.7 | [-42.9, +30.6] |
| response/defender_declared_when_legal | 0.0 (1) | 41.7 (12) | **+41.7** | [+25.0, +66.7] |
| response/spell_played_when_legal | 55.6 (9) | 100.0 (5) | +44.4 | [+0.0, +80.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 100.0 (13) | 94.7 (19) | -5.3 | [-16.7, +0.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 92.3 (13) | 89.5 (19) | -2.8 | [-25.0, +15.5] |
| strategy/attacks_by_equipped_attacker | 12.8 (507) | 15.7 (573) | +2.9 | [-2.7, +8.2] |
| strategy/face_target_share_when_both_legal | 61.7 (376) | 61.1 (404) | -0.6 | [-7.5, +5.4] |
| strategy/favorable_trade_taken_per_available_turn | 38.5 (122) | 49.5 (109) | +11.0 | [+0.0, +22.0] |
| strategy/spell_cast_per_legal_main_turn | 8.0 (75) | 2.6 (38) | -5.4 | [-13.1, +2.3] |

### lightning vs u8223 — sample — free_draft|Surge/Raizan

| metric | u8223 | lightning | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 15.8 (48) | 17.7 (48) | **+2.0** | [+1.3, +2.6] |
| draft/mean_cost | 2.70 (2400) | 2.67 (2400) | -0.0 | [-0.1, +0.0] |
| draft/normal_share | 64.6 (2400) | 62.8 (2400) | **-1.8** | [-2.8, -0.8] |
| draft/spell_share | 5.0 (2400) | 4.2 (2400) | **-0.7** | [-1.4, -0.0] |
| draft/unique_cards | 32.60 (48) | 32.31 (48) | -0.3 | [-0.9, +0.2] |
| draft/weapon_share | 17.4 (2400) | 18.4 (2400) | **+1.0** | [+0.2, +1.8] |
| gate/Surge/eligible_portal_then_equip | 100.0 (32) | 100.0 (40) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.9 (32) | 95.0 (40) | -1.9 | [-9.1, +8.3] |
| gate/Surge/portal_with_eligible_weapon | 15.2 (211) | 16.7 (240) | +1.5 | [-8.0, +10.2] |
| gate/portal_per_legal_turn | 100.0 (211) | 99.2 (242) | -0.8 | [-2.0, +0.0] |
| ikz/held_at_end_of_own_turn | 22.9 (315) | 24.5 (331) | +1.6 | [-4.7, +7.5] |
| ikz/held_then_spent_in_opp_turn | 6.9 (72) | 11.1 (81) | +4.2 | [-2.1, +9.6] |
| lightning/weapon_attach_per_legal_turn | 21.8 (202) | 24.3 (230) | +2.6 | [-4.4, +9.4] |
| lightning/weapon_attach_to_entity_share | 0.0 (64) | 0.0 (72) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.56 (48) | 6.90 (48) | **+0.3** | [+0.0, +0.6] |
| outcome/win | 43.8 (48) | 54.2 (48) | +10.4 | [-4.2, +25.0] |
| response/any_nonblock_response_when_legal | 60.0 (10) | 75.0 (12) | +15.0 | [-50.0, +61.5] |
| response/defender_declared_when_legal | 13.0 (23) | 100.0 (1) | **+87.0** | [+50.0, +100.0] |
| response/spell_played_when_legal | 100.0 (4) | 100.0 (8) | +0.0 | [+0.0, +0.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 100.0 (17) | 94.4 (18) | -5.6 | [-17.4, +0.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 82.4 (17) | 83.3 (18) | +1.0 | [-23.5, +26.8] |
| strategy/attacks_by_equipped_attacker | 13.6 (537) | 13.7 (615) | +0.1 | [-4.7, +4.6] |
| strategy/face_target_share_when_both_legal | 63.4 (361) | 68.7 (415) | +5.2 | [-0.4, +11.1] |
| strategy/favorable_trade_taken_per_available_turn | 46.9 (113) | 39.8 (123) | -7.1 | [-19.0, +5.2] |
| strategy/spell_cast_per_legal_main_turn | 10.6 (47) | 17.1 (35) | +6.5 | [-6.7, +18.9] |

### u9305 vs u8223 — argmax — all|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 36.7 (30) | 31.2 (32) | -5.4 | [-22.4, +12.0] |
| draft/max_wjaccard_to_curated | 18.9 (192) | 18.3 (192) | **-0.6** | [-0.9, -0.3] |
| draft/mean_cost | 2.89 (9600) | 3.02 (9600) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 65.5 (9600) | 72.0 (9600) | **+6.5** | [+6.3, +6.7] |
| draft/spell_share | 0.0 (9600) | 0.0 (9600) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.25 (192) | 17.50 (192) | **-2.8** | [-2.8, -2.7] |
| draft/weapon_share | 20.0 (9600) | 20.0 (9600) | +0.0 | [+0.0, +0.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (410) | 0.0 (450) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (719) | 100.0 (700) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 97.5 (719) | 97.6 (700) | +0.1 | [-1.2, +1.4] |
| gate/Surge/portal_with_eligible_weapon | 53.3 (1350) | 51.1 (1369) | -2.1 | [-4.9, +0.6] |
| gate/portal_per_legal_turn | 99.8 (1764) | 99.8 (1822) | +0.1 | [-0.2, +0.3] |
| ikz/held_at_end_of_own_turn | 26.2 (2848) | 27.7 (2846) | +1.5 | [-0.1, +3.1] |
| ikz/held_then_spent_in_opp_turn | 33.6 (745) | 34.2 (787) | +0.6 | [-2.1, +3.3] |
| lightning/weapon_attach_per_legal_turn | 26.4 (2385) | 25.9 (2363) | -0.6 | [-2.1, +1.1] |
| lightning/weapon_attach_to_entity_share | 0.3 (1035) | 0.1 (1021) | -0.2 | [-0.5, +0.0] |
| outcome/own_turns | 6.36 (448) | 6.35 (448) | -0.0 | [-0.1, +0.1] |
| outcome/win | 51.6 (448) | 52.0 (448) | +0.4 | [-3.3, +4.7] |
| outcome/win_vs_EARTH | 55.4 (112) | 58.0 (112) | +2.7 | [-5.9, +11.6] |
| outcome/win_vs_FIRE | 29.5 (112) | 30.4 (112) | +0.9 | [-7.8, +9.6] |
| outcome/win_vs_LIGHTNING | 63.4 (112) | 63.4 (112) | +0.0 | [-6.9, +7.3] |
| outcome/win_vs_WATER | 58.0 (112) | 56.2 (112) | -1.8 | [-7.8, +4.3] |
| response/any_nonblock_response_when_legal | 79.5 (337) | 79.0 (372) | -0.5 | [-6.3, +5.6] |
| response/defender_declared_when_legal | 50.0 (62) | 62.7 (51) | **+12.7** | [+2.2, +24.9] |
| response/spell_played_when_legal | 86.7 (263) | 88.8 (276) | +2.1 | [-2.4, +6.6] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 98.6 (215) | 99.0 (204) | +0.4 | [-1.6, +2.6] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.8 (215) | 90.7 (204) | +1.8 | [-1.5, +5.5] |
| strategy/attacks_by_equipped_attacker | 22.3 (5166) | 21.9 (5065) | -0.4 | [-1.3, +0.6] |
| strategy/face_target_share_when_both_legal | 65.9 (3204) | 62.3 (3036) | **-3.6** | [-5.6, -1.7] |
| strategy/favorable_trade_taken_per_available_turn | 42.4 (874) | 49.5 (848) | **+7.1** | [+3.3, +10.8] |
| strategy/spell_cast_per_legal_main_turn | 44.8 (29) | 36.7 (30) | -8.2 | [-25.1, +13.5] |

### u9305 vs u8223 — argmax — all|Stormchain/Piko

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.6 (48) | 19.6 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 2.90 (2400) | 3.04 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 64.0 (2400) | 72.0 (2400) | **+8.0** | [+8.0, +8.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 17.00 (48) | **-3.0** | [-3.0, -3.0] |
| draft/weapon_share | 20.0 (2400) | 20.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (90) | 0.0 (123) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 98.9 (91) | 100.0 (123) | +1.1 | [+0.0, +3.4] |
| ikz/held_at_end_of_own_turn | 22.8 (320) | 25.9 (332) | +3.1 | [-2.6, +8.9] |
| ikz/held_then_spent_in_opp_turn | 2.7 (73) | 0.0 (86) | -2.7 | [-6.8, +0.0] |
| lightning/weapon_attach_per_legal_turn | 18.0 (250) | 21.8 (239) | +3.8 | [-0.8, +8.4] |
| lightning/weapon_attach_to_entity_share | 0.0 (60) | 0.0 (66) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.67 (48) | 6.92 (48) | **+0.2** | [+0.0, +0.5] |
| outcome/win | 66.7 (48) | 60.4 (48) | -6.2 | [-18.8, +6.2] |
| strategy/attacks_by_equipped_attacker | 7.1 (560) | 7.7 (581) | +0.6 | [-1.5, +2.9] |
| strategy/face_target_share_when_both_legal | 64.5 (383) | 57.0 (374) | **-7.5** | [-14.4, -1.1] |
| strategy/favorable_trade_taken_per_available_turn | 43.0 (142) | 50.3 (153) | +7.4 | [-2.2, +17.6] |

### u9305 vs u8223 — argmax — all|Stormchain/Raizan

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (15) | 5.9 (17) | +5.9 | [+0.0, +20.0] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 19.6 (48) | **+1.3** | [+1.3, +1.3] |
| draft/mean_cost | 2.92 (2400) | 3.00 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 66.0 (2400) | 72.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 20.0 (2400) | 20.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (320) | 0.0 (327) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (320) | 100.0 (327) | +0.0 | [+0.0, +0.0] |
| ikz/held_at_end_of_own_turn | 33.2 (751) | 32.7 (752) | -0.4 | [-3.4, +2.7] |
| ikz/held_then_spent_in_opp_turn | 40.6 (249) | 41.9 (246) | +1.3 | [-3.6, +6.3] |
| lightning/weapon_attach_per_legal_turn | 27.0 (623) | 26.5 (616) | -0.5 | [-4.2, +3.3] |
| lightning/weapon_attach_to_entity_share | 0.0 (259) | 0.0 (266) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.71 (112) | 6.71 (112) | +0.0 | [-0.1, +0.1] |
| outcome/win | 36.6 (112) | 39.3 (112) | +2.7 | [-3.6, +8.9] |
| response/any_nonblock_response_when_legal | 87.9 (124) | 77.1 (144) | -10.8 | [-21.5, +1.6] |
| response/defender_declared_when_legal | 32.6 (46) | 47.2 (36) | **+14.6** | [+5.0, +29.5] |
| response/spell_played_when_legal | 94.9 (99) | 92.6 (108) | -2.4 | [-10.4, +2.8] |
| strategy/attacks_by_equipped_attacker | 13.8 (1146) | 13.6 (1105) | -0.2 | [-2.3, +1.8] |
| strategy/face_target_share_when_both_legal | 67.3 (715) | 65.4 (693) | -1.9 | [-5.5, +1.6] |
| strategy/favorable_trade_taken_per_available_turn | 41.2 (211) | 46.0 (211) | +4.7 | [-1.7, +11.5] |

### u9305 vs u8223 — argmax — all|Surge/Piko

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.3 (48) | 17.0 (48) | **-1.3** | [-1.3, -1.3] |
| draft/mean_cost | 2.92 (2400) | 3.04 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 66.0 (2400) | 72.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 17.00 (48) | **-3.0** | [-3.0, -3.0] |
| draft/weapon_share | 20.0 (2400) | 20.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (27) | 100.0 (19) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.3 (27) | 89.5 (19) | -6.8 | [-27.8, +7.5] |
| gate/Surge/portal_with_eligible_weapon | 12.4 (218) | 8.6 (221) | -3.8 | [-12.1, +5.5] |
| gate/portal_per_legal_turn | 99.1 (220) | 100.0 (221) | +0.9 | [+0.0, +2.2] |
| ikz/held_at_end_of_own_turn | 16.2 (314) | 19.7 (314) | +3.5 | [-2.3, +9.4] |
| ikz/held_then_spent_in_opp_turn | 0.0 (51) | 1.6 (62) | +1.6 | [+0.0, +5.6] |
| lightning/weapon_attach_per_legal_turn | 20.2 (248) | 22.9 (218) | +2.8 | [-3.4, +9.0] |
| lightning/weapon_attach_to_entity_share | 0.0 (69) | 0.0 (60) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.54 (48) | 6.54 (48) | +0.0 | [-0.3, +0.3] |
| outcome/win | 50.0 (48) | 54.2 (48) | +4.2 | [-8.3, +16.7] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 88.2 (17) | 90.9 (11) | +2.7 | [-23.3, +23.5] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.2 (17) | 81.8 (11) | -6.4 | [-30.8, +11.1] |
| strategy/attacks_by_equipped_attacker | 12.4 (555) | 11.0 (543) | -1.4 | [-6.1, +3.7] |
| strategy/face_target_share_when_both_legal | 60.8 (380) | 52.3 (342) | **-8.5** | [-13.8, -3.3] |
| strategy/favorable_trade_taken_per_available_turn | 37.4 (123) | 52.5 (118) | **+15.1** | [+5.6, +26.8] |

### u9305 vs u8223 — argmax — all|Surge/Raizan

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 73.3 (15) | 60.0 (15) | -13.3 | [-35.2, +11.5] |
| draft/max_wjaccard_to_curated | 19.6 (48) | 17.0 (48) | **-2.5** | [-2.5, -2.5] |
| draft/mean_cost | 2.82 (2400) | 3.00 (2400) | **+0.2** | [+0.2, +0.2] |
| draft/normal_share | 66.0 (2400) | 72.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 21.00 (48) | 18.00 (48) | **-3.0** | [-3.0, -3.0] |
| draft/weapon_share | 20.0 (2400) | 20.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (692) | 100.0 (681) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 97.5 (692) | 97.8 (681) | +0.3 | [-1.0, +1.5] |
| gate/Surge/portal_with_eligible_weapon | 61.1 (1132) | 59.3 (1148) | -1.8 | [-4.7, +0.9] |
| gate/portal_per_legal_turn | 99.9 (1133) | 99.7 (1151) | -0.2 | [-0.4, +0.0] |
| ikz/held_at_end_of_own_turn | 25.4 (1463) | 27.1 (1448) | +1.7 | [-0.0, +3.5] |
| ikz/held_then_spent_in_opp_turn | 39.5 (372) | 42.0 (393) | +2.5 | [-1.3, +6.4] |
| lightning/weapon_attach_per_legal_turn | 29.0 (1264) | 26.8 (1290) | **-2.2** | [-4.0, -0.4] |
| lightning/weapon_attach_to_entity_share | 0.5 (647) | 0.2 (629) | -0.3 | [-0.8, +0.0] |
| outcome/own_turns | 6.10 (240) | 6.03 (240) | -0.1 | [-0.2, +0.0] |
| outcome/win | 55.8 (240) | 55.8 (240) | +0.0 | [-5.4, +5.4] |
| response/any_nonblock_response_when_legal | 75.8 (207) | 81.2 (224) | +5.4 | [-0.9, +12.2] |
| response/defender_declared_when_legal | 100.0 (16) | 100.0 (15) | +0.0 | [+0.0, +0.0] |
| response/spell_played_when_legal | 81.7 (164) | 86.3 (168) | +4.6 | [-0.8, +11.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.5 (198) | 99.5 (193) | -0.0 | [-1.5, +1.5] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.9 (198) | 91.2 (193) | +2.3 | [-1.1, +5.9] |
| strategy/attacks_by_equipped_attacker | 30.4 (2905) | 30.1 (2836) | -0.3 | [-1.5, +0.9] |
| strategy/face_target_share_when_both_legal | 66.7 (1726) | 64.2 (1627) | -2.5 | [-5.2, +0.2] |
| strategy/favorable_trade_taken_per_available_turn | 44.5 (398) | 50.3 (366) | **+5.8** | [+0.3, +11.3] |
| strategy/spell_cast_per_legal_main_turn | 44.8 (29) | 36.7 (30) | -8.2 | [-24.7, +12.6] |

### u9305 vs u8223 — argmax — fixed|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 36.7 (30) | 31.2 (32) | -5.4 | [-22.8, +12.4] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (222) | 0.0 (219) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (638) | 100.0 (645) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 97.6 (638) | 98.0 (645) | +0.3 | [-0.8, +1.5] |
| gate/Surge/portal_with_eligible_weapon | 70.0 (911) | 70.1 (920) | +0.1 | [-2.1, +2.1] |
| gate/portal_per_legal_turn | 99.9 (1134) | 99.8 (1141) | -0.1 | [-0.3, +0.0] |
| ikz/held_at_end_of_own_turn | 32.6 (1581) | 32.8 (1560) | +0.2 | [-1.2, +1.5] |
| ikz/held_then_spent_in_opp_turn | 47.8 (515) | 50.9 (511) | **+3.1** | [+0.3, +6.1] |
| lightning/weapon_attach_per_legal_turn | 30.2 (1409) | 28.6 (1403) | **-1.7** | [-3.2, -0.1] |
| lightning/weapon_attach_to_entity_share | 0.4 (768) | 0.1 (746) | -0.3 | [-0.7, +0.0] |
| outcome/own_turns | 6.18 (256) | 6.09 (256) | **-0.1** | [-0.2, -0.0] |
| outcome/win | 46.5 (256) | 47.7 (256) | +1.2 | [-3.5, +5.9] |
| outcome/win_vs_EARTH | 43.8 (64) | 46.9 (64) | +3.1 | [-6.7, +12.5] |
| outcome/win_vs_FIRE | 25.0 (64) | 28.1 (64) | +3.1 | [-8.1, +15.2] |
| outcome/win_vs_LIGHTNING | 68.8 (64) | 67.2 (64) | -1.6 | [-9.8, +6.9] |
| outcome/win_vs_WATER | 48.4 (64) | 48.4 (64) | +0.0 | [-7.8, +7.7] |
| response/any_nonblock_response_when_legal | 82.8 (319) | 81.2 (351) | -1.6 | [-6.8, +3.7] |
| response/defender_declared_when_legal | 50.0 (62) | 62.7 (51) | **+12.7** | [+2.4, +25.1] |
| response/spell_played_when_legal | 86.7 (263) | 88.8 (276) | +2.1 | [-2.4, +6.7] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.4 (172) | 100.0 (175) | +0.6 | [+0.0, +1.9] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 89.0 (172) | 91.4 (175) | +2.5 | [-1.0, +6.1] |
| strategy/attacks_by_equipped_attacker | 31.4 (2870) | 31.3 (2805) | -0.0 | [-0.9, +0.9] |
| strategy/face_target_share_when_both_legal | 70.2 (1700) | 69.1 (1611) | -1.1 | [-3.1, +1.1] |
| strategy/favorable_trade_taken_per_available_turn | 38.8 (353) | 47.3 (315) | **+8.5** | [+4.4, +12.8] |
| strategy/spell_cast_per_legal_main_turn | 44.8 (29) | 36.7 (30) | -8.2 | [-24.5, +12.6] |

### u9305 vs u8223 — argmax — fixed|Stormchain/Raizan

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 0.0 (15) | 5.9 (17) | +5.9 | [+0.0, +20.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (222) | 0.0 (219) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (222) | 100.0 (219) | +0.0 | [+0.0, +0.0] |
| ikz/held_at_end_of_own_turn | 43.0 (433) | 41.4 (430) | -1.6 | [-3.7, +0.4] |
| ikz/held_then_spent_in_opp_turn | 53.2 (186) | 57.3 (178) | **+4.1** | [+0.2, +8.4] |
| lightning/weapon_attach_per_legal_turn | 32.3 (378) | 28.9 (374) | **-3.4** | [-7.2, -0.3] |
| lightning/weapon_attach_to_entity_share | 0.0 (202) | 0.0 (194) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.77 (64) | 6.72 (64) | -0.0 | [-0.2, +0.0] |
| outcome/win | 21.9 (64) | 21.9 (64) | +0.0 | [-7.8, +6.2] |
| response/any_nonblock_response_when_legal | 89.2 (120) | 80.9 (136) | -8.3 | [-17.3, +2.4] |
| response/defender_declared_when_legal | 32.6 (46) | 47.2 (36) | **+14.6** | [+5.4, +29.1] |
| response/spell_played_when_legal | 94.9 (99) | 92.6 (108) | -2.4 | [-10.1, +2.9] |
| strategy/attacks_by_equipped_attacker | 20.6 (579) | 18.4 (548) | **-2.1** | [-4.8, -0.1] |
| strategy/face_target_share_when_both_legal | 77.5 (360) | 76.3 (337) | -1.2 | [-3.1, +0.6] |
| strategy/favorable_trade_taken_per_available_turn | 33.3 (81) | 40.3 (77) | +6.9 | [+0.0, +15.7] |

### u9305 vs u8223 — argmax — fixed|Surge/Raizan

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 73.3 (15) | 60.0 (15) | -13.3 | [-35.7, +12.6] |
| gate/Surge/eligible_portal_then_equip | 100.0 (638) | 100.0 (645) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 97.6 (638) | 98.0 (645) | +0.3 | [-0.7, +1.5] |
| gate/Surge/portal_with_eligible_weapon | 70.0 (911) | 70.1 (920) | +0.1 | [-1.9, +2.0] |
| gate/portal_per_legal_turn | 99.9 (912) | 99.8 (922) | -0.1 | [-0.3, +0.0] |
| ikz/held_at_end_of_own_turn | 28.7 (1148) | 29.5 (1130) | +0.8 | [-0.9, +2.5] |
| ikz/held_then_spent_in_opp_turn | 44.7 (329) | 47.4 (333) | +2.8 | [-1.2, +6.6] |
| lightning/weapon_attach_per_legal_turn | 29.5 (1031) | 28.5 (1029) | -1.0 | [-2.6, +0.6] |
| lightning/weapon_attach_to_entity_share | 0.5 (566) | 0.2 (552) | -0.3 | [-0.9, +0.0] |
| outcome/own_turns | 5.98 (192) | 5.89 (192) | **-0.1** | [-0.2, -0.0] |
| outcome/win | 54.7 (192) | 56.2 (192) | +1.6 | [-4.2, +7.3] |
| response/any_nonblock_response_when_legal | 78.9 (199) | 81.4 (215) | +2.5 | [-3.0, +8.3] |
| response/defender_declared_when_legal | 100.0 (16) | 100.0 (15) | +0.0 | [+0.0, +0.0] |
| response/spell_played_when_legal | 81.7 (164) | 86.3 (168) | +4.6 | [-0.8, +11.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 99.4 (172) | 100.0 (175) | +0.6 | [+0.0, +1.8] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 89.0 (172) | 91.4 (175) | +2.5 | [-0.9, +6.1] |
| strategy/attacks_by_equipped_attacker | 34.1 (2291) | 34.5 (2257) | +0.4 | [-0.6, +1.4] |
| strategy/face_target_share_when_both_legal | 68.3 (1340) | 67.3 (1274) | -1.0 | [-3.6, +1.6] |
| strategy/favorable_trade_taken_per_available_turn | 40.4 (272) | 49.6 (238) | **+9.1** | [+4.1, +14.2] |
| strategy/spell_cast_per_legal_main_turn | 44.8 (29) | 36.7 (30) | -8.2 | [-24.4, +12.2] |

### u9305 vs u8223 — argmax — free_draft|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.9 (192) | 18.3 (192) | **-0.6** | [-0.9, -0.4] |
| draft/mean_cost | 2.89 (9600) | 3.02 (9600) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 65.5 (9600) | 72.0 (9600) | **+6.5** | [+6.3, +6.7] |
| draft/spell_share | 0.0 (9600) | 0.0 (9600) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.25 (192) | 17.50 (192) | **-2.8** | [-2.8, -2.7] |
| draft/weapon_share | 20.0 (9600) | 20.0 (9600) | +0.0 | [+0.0, +0.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (188) | 0.0 (231) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (81) | 100.0 (55) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.3 (81) | 92.7 (55) | -3.6 | [-13.3, +4.5] |
| gate/Surge/portal_with_eligible_weapon | 18.5 (439) | 12.2 (449) | -6.2 | [-13.0, +0.8] |
| gate/portal_per_legal_turn | 99.5 (630) | 99.9 (681) | +0.3 | [-0.3, +1.0] |
| ikz/held_at_end_of_own_turn | 18.2 (1267) | 21.5 (1286) | **+3.3** | [+0.4, +6.3] |
| ikz/held_then_spent_in_opp_turn | 1.7 (230) | 3.3 (276) | +1.5 | [-1.2, +4.2] |
| lightning/weapon_attach_per_legal_turn | 20.9 (976) | 21.9 (960) | +1.0 | [-2.2, +4.3] |
| lightning/weapon_attach_to_entity_share | 0.0 (267) | 0.0 (275) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.60 (192) | 6.70 (192) | +0.1 | [-0.1, +0.2] |
| outcome/win | 58.3 (192) | 57.8 (192) | -0.5 | [-7.3, +6.2] |
| outcome/win_vs_EARTH | 70.8 (48) | 72.9 (48) | +2.1 | [-15.0, +18.2] |
| outcome/win_vs_FIRE | 35.4 (48) | 33.3 (48) | -2.1 | [-15.6, +11.8] |
| outcome/win_vs_LIGHTNING | 56.2 (48) | 58.3 (48) | +2.1 | [-10.5, +14.3] |
| outcome/win_vs_WATER | 70.8 (48) | 66.7 (48) | -4.2 | [-14.0, +5.4] |
| response/any_nonblock_response_when_legal | 22.2 (18) | 42.9 (21) | +20.6 | [-15.8, +65.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 95.3 (43) | 93.1 (29) | -2.2 | [-14.2, +8.2] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.4 (43) | 86.2 (29) | -2.2 | [-16.2, +9.2] |
| strategy/attacks_by_equipped_attacker | 10.9 (2296) | 10.2 (2260) | -0.7 | [-2.5, +1.2] |
| strategy/face_target_share_when_both_legal | 60.9 (1504) | 54.5 (1425) | **-6.4** | [-9.6, -3.1] |
| strategy/favorable_trade_taken_per_available_turn | 44.9 (521) | 50.8 (533) | **+5.9** | [+0.6, +11.2] |

### u9305 vs u8223 — argmax — free_draft|Stormchain/Piko

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.6 (48) | 19.6 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 2.90 (2400) | 3.04 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 64.0 (2400) | 72.0 (2400) | **+8.0** | [+8.0, +8.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 17.00 (48) | **-3.0** | [-3.0, -3.0] |
| draft/weapon_share | 20.0 (2400) | 20.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (90) | 0.0 (123) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 98.9 (91) | 100.0 (123) | +1.1 | [+0.0, +3.4] |
| ikz/held_at_end_of_own_turn | 22.8 (320) | 25.9 (332) | +3.1 | [-2.6, +8.9] |
| ikz/held_then_spent_in_opp_turn | 2.7 (73) | 0.0 (86) | -2.7 | [-6.8, +0.0] |
| lightning/weapon_attach_per_legal_turn | 18.0 (250) | 21.8 (239) | +3.8 | [-0.8, +8.4] |
| lightning/weapon_attach_to_entity_share | 0.0 (60) | 0.0 (66) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.67 (48) | 6.92 (48) | **+0.2** | [+0.0, +0.5] |
| outcome/win | 66.7 (48) | 60.4 (48) | -6.2 | [-18.8, +6.2] |
| strategy/attacks_by_equipped_attacker | 7.1 (560) | 7.7 (581) | +0.6 | [-1.5, +2.9] |
| strategy/face_target_share_when_both_legal | 64.5 (383) | 57.0 (374) | **-7.5** | [-14.4, -1.1] |
| strategy/favorable_trade_taken_per_available_turn | 43.0 (142) | 50.3 (153) | +7.4 | [-2.2, +17.6] |

### u9305 vs u8223 — argmax — free_draft|Stormchain/Raizan

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.3 (48) | 19.6 (48) | **+1.3** | [+1.3, +1.3] |
| draft/mean_cost | 2.92 (2400) | 3.00 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 66.0 (2400) | 72.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 18.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 20.0 (2400) | 20.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Stormchain/portal_with_eligible_weapon | 0.0 (98) | 0.0 (108) | +0.0 | [+0.0, +0.0] |
| gate/portal_per_legal_turn | 100.0 (98) | 100.0 (108) | +0.0 | [+0.0, +0.0] |
| ikz/held_at_end_of_own_turn | 19.8 (318) | 21.1 (322) | +1.3 | [-4.6, +7.9] |
| ikz/held_then_spent_in_opp_turn | 3.2 (63) | 1.5 (68) | -1.7 | [-8.0, +3.3] |
| lightning/weapon_attach_per_legal_turn | 18.8 (245) | 22.7 (242) | +4.0 | [-3.3, +11.1] |
| lightning/weapon_attach_to_entity_share | 0.0 (57) | 0.0 (72) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.62 (48) | 6.71 (48) | +0.1 | [-0.2, +0.3] |
| outcome/win | 56.2 (48) | 62.5 (48) | +6.2 | [-6.2, +18.8] |
| response/any_nonblock_response_when_legal | 50.0 (4) | 12.5 (8) | -37.5 | [-100.0, +57.1] |
| strategy/attacks_by_equipped_attacker | 6.9 (567) | 8.8 (557) | +1.9 | [-1.3, +5.4] |
| strategy/face_target_share_when_both_legal | 56.9 (355) | 55.1 (356) | -1.8 | [-9.2, +5.1] |
| strategy/favorable_trade_taken_per_available_turn | 46.2 (130) | 49.3 (134) | +3.1 | [-6.5, +12.9] |

### u9305 vs u8223 — argmax — free_draft|Surge/Piko

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 18.3 (48) | 17.0 (48) | **-1.3** | [-1.3, -1.3] |
| draft/mean_cost | 2.92 (2400) | 3.04 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 66.0 (2400) | 72.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 20.00 (48) | 17.00 (48) | **-3.0** | [-3.0, -3.0] |
| draft/weapon_share | 20.0 (2400) | 20.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (27) | 100.0 (19) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.3 (27) | 89.5 (19) | -6.8 | [-27.8, +7.5] |
| gate/Surge/portal_with_eligible_weapon | 12.4 (218) | 8.6 (221) | -3.8 | [-12.1, +5.5] |
| gate/portal_per_legal_turn | 99.1 (220) | 100.0 (221) | +0.9 | [+0.0, +2.2] |
| ikz/held_at_end_of_own_turn | 16.2 (314) | 19.7 (314) | +3.5 | [-2.3, +9.4] |
| ikz/held_then_spent_in_opp_turn | 0.0 (51) | 1.6 (62) | +1.6 | [+0.0, +5.6] |
| lightning/weapon_attach_per_legal_turn | 20.2 (248) | 22.9 (218) | +2.8 | [-3.4, +9.0] |
| lightning/weapon_attach_to_entity_share | 0.0 (69) | 0.0 (60) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.54 (48) | 6.54 (48) | +0.0 | [-0.3, +0.3] |
| outcome/win | 50.0 (48) | 54.2 (48) | +4.2 | [-8.3, +16.7] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 88.2 (17) | 90.9 (11) | +2.7 | [-23.3, +23.5] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.2 (17) | 81.8 (11) | -6.4 | [-30.8, +11.1] |
| strategy/attacks_by_equipped_attacker | 12.4 (555) | 11.0 (543) | -1.4 | [-6.1, +3.7] |
| strategy/face_target_share_when_both_legal | 60.8 (380) | 52.3 (342) | **-8.5** | [-13.8, -3.3] |
| strategy/favorable_trade_taken_per_available_turn | 37.4 (123) | 52.5 (118) | **+15.1** | [+5.6, +26.8] |

### u9305 vs u8223 — argmax — free_draft|Surge/Raizan

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 19.6 (48) | 17.0 (48) | **-2.5** | [-2.5, -2.5] |
| draft/mean_cost | 2.82 (2400) | 3.00 (2400) | **+0.2** | [+0.2, +0.2] |
| draft/normal_share | 66.0 (2400) | 72.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 21.00 (48) | 18.00 (48) | **-3.0** | [-3.0, -3.0] |
| draft/weapon_share | 20.0 (2400) | 20.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Surge/eligible_portal_then_equip | 100.0 (54) | 100.0 (36) | +0.0 | [+0.0, +0.0] |
| gate/Surge/equip_then_destination_attacks | 96.3 (54) | 94.4 (36) | -1.9 | [-13.5, +6.8] |
| gate/Surge/portal_with_eligible_weapon | 24.4 (221) | 15.8 (228) | -8.6 | [-19.0, +1.8] |
| gate/portal_per_legal_turn | 100.0 (221) | 99.6 (229) | -0.4 | [-1.3, +0.0] |
| ikz/held_at_end_of_own_turn | 13.7 (315) | 18.9 (318) | +5.2 | [-0.6, +11.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (43) | 11.7 (60) | **+11.7** | [+5.5, +17.6] |
| lightning/weapon_attach_per_legal_turn | 27.0 (233) | 20.3 (261) | **-6.7** | [-12.6, -0.5] |
| lightning/weapon_attach_to_entity_share | 0.0 (81) | 0.0 (77) | +0.0 | [+0.0, +0.0] |
| outcome/own_turns | 6.56 (48) | 6.62 (48) | +0.1 | [-0.3, +0.4] |
| outcome/win | 60.4 (48) | 54.2 (48) | -6.2 | [-20.8, +8.3] |
| response/any_nonblock_response_when_legal | 0.0 (8) | 77.8 (9) | **+77.8** | [+44.4, +100.0] |
| sequence/lightning.surge_weapon_recovery_attack/completed_per_eligible_game | 100.0 (26) | 94.4 (18) | -5.6 | [-16.7, +0.0] |
| sequence/lightning.surge_weapon_recovery_attack/converted_per_eligible_game | 88.5 (26) | 88.9 (18) | +0.4 | [-15.6, +14.4] |
| strategy/attacks_by_equipped_attacker | 16.6 (614) | 13.1 (579) | -3.5 | [-7.6, +0.6] |
| strategy/face_target_share_when_both_legal | 61.1 (386) | 53.3 (353) | **-7.9** | [-15.5, -0.9] |
| strategy/favorable_trade_taken_per_available_turn | 53.2 (126) | 51.6 (128) | -1.6 | [-14.1, +11.0] |

## WATER
### u9305 vs u8223 — argmax — all|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 14.2 (106) | 5.6 (107) | **-8.5** | [-14.8, -2.3] |
| draft/max_wjaccard_to_curated | 15.2 (96) | 14.0 (96) | **-1.2** | [-1.2, -1.2] |
| draft/mean_cost | 2.31 (4800) | 2.33 (4800) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 83.0 (4800) | 86.0 (4800) | **+3.0** | [+2.7, +3.3] |
| draft/spell_share | 0.0 (4800) | 0.0 (4800) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 23.50 (96) | 22.00 (96) | **-1.5** | [-1.6, -1.4] |
| draft/weapon_share | 7.0 (4800) | 7.0 (4800) | +0.0 | [+0.0, +0.0] |
| gate/Hydromancy/portal_readied_ikz | 90.7 (1144) | 92.2 (1187) | +1.4 | [-0.1, +3.1] |
| gate/Hydromancy/readied_then_spent_same_turn | 78.4 (1038) | 76.7 (1094) | -1.7 | [-4.3, +0.8] |
| gate/portal_per_legal_turn | 97.3 (1176) | 97.5 (1218) | +0.2 | [-0.9, +1.2] |
| ikz/held_at_end_of_own_turn | 42.0 (1582) | 45.2 (1598) | **+3.1** | [+0.9, +5.4] |
| ikz/held_then_spent_in_opp_turn | 57.7 (665) | 54.8 (722) | -2.9 | [-6.3, +0.5] |
| leader/Shao/target_is_current_attacker | 45.9 (377) | 46.5 (385) | +0.6 | [-5.0, +6.3] |
| leader/use_per_legal_turn | 87.1 (434) | 85.7 (454) | -1.4 | [-4.3, +1.4] |
| outcome/own_turns | 7.06 (224) | 7.13 (224) | +0.1 | [-0.1, +0.2] |
| outcome/win | 57.1 (224) | 57.1 (224) | +0.0 | [-5.8, +5.8] |
| outcome/win_vs_EARTH | 60.7 (56) | 58.9 (56) | -1.8 | [-11.8, +10.0] |
| outcome/win_vs_FIRE | 41.1 (56) | 37.5 (56) | -3.6 | [-12.5, +5.0] |
| outcome/win_vs_LIGHTNING | 69.6 (56) | 71.4 (56) | +1.8 | [-10.7, +15.4] |
| outcome/win_vs_WATER | 57.1 (56) | 60.7 (56) | +3.6 | [-10.0, +17.2] |
| response/any_nonblock_response_when_legal | 67.3 (615) | 66.1 (660) | -1.3 | [-5.4, +2.8] |
| response/defender_declared_when_legal | 28.0 (378) | 26.3 (407) | -1.8 | [-5.9, +2.2] |
| response/spell_played_when_legal | 19.3 (367) | 19.8 (373) | +0.5 | [-2.9, +3.9] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 70.8 (161) | 66.7 (165) | -4.1 | [-12.6, +4.4] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 70.8 (161) | 66.7 (165) | -4.1 | [-12.6, +4.4] |
| strategy/attacks_by_equipped_attacker | 3.7 (3456) | 3.7 (3441) | -0.0 | [-0.7, +0.6] |
| strategy/face_target_share_when_both_legal | 59.5 (2053) | 57.9 (2042) | -1.5 | [-4.5, +1.5] |
| strategy/favorable_trade_taken_per_available_turn | 55.5 (584) | 57.0 (591) | +1.5 | [-2.9, +5.9] |
| strategy/spell_cast_per_legal_main_turn | 44.2 (477) | 43.1 (480) | -1.1 | [-5.2, +2.7] |
| water/bounce_play_per_legal_turn | 70.6 (722) | 70.3 (743) | -0.4 | [-3.4, +2.6] |
| water/bounce_play_removes_opp_entity | 47.9 (541) | 48.8 (551) | +0.9 | [-2.4, +4.3] |

### u9305 vs u8223 — argmax — all|Hydromancy/Benzai

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 13.4 (48) | 12.2 (48) | **-1.2** | [-1.2, -1.2] |
| draft/mean_cost | 2.38 (2400) | 2.38 (2400) | +0.0 | [+0.0, +0.0] |
| draft/normal_share | 88.0 (2400) | 90.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 22.00 (48) | 21.00 (48) | **-1.0** | [-1.0, -1.0] |
| draft/weapon_share | 8.0 (2400) | 8.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Hydromancy/portal_readied_ikz | 90.0 (261) | 95.5 (267) | **+5.5** | [+2.0, +8.8] |
| gate/Hydromancy/readied_then_spent_same_turn | 83.4 (235) | 82.0 (255) | -1.4 | [-6.9, +3.7] |
| gate/portal_per_legal_turn | 96.3 (271) | 98.5 (271) | **+2.2** | [+0.1, +4.2] |
| ikz/held_at_end_of_own_turn | 38.1 (312) | 41.2 (320) | +3.1 | [-3.2, +9.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (119) | 0.0 (132) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 7.1 (14) | 23.5 (17) | +16.4 | [-0.4, +38.9] |
| outcome/own_turns | 6.50 (48) | 6.67 (48) | +0.2 | [-0.1, +0.4] |
| outcome/win | 70.8 (48) | 68.8 (48) | -2.1 | [-16.7, +14.6] |
| strategy/attacks_by_equipped_attacker | 3.4 (794) | 3.0 (770) | -0.4 | [-1.8, +0.9] |
| strategy/face_target_share_when_both_legal | 59.5 (491) | 59.2 (473) | -0.3 | [-6.6, +7.0] |
| strategy/favorable_trade_taken_per_available_turn | 53.8 (132) | 52.7 (131) | -1.1 | [-12.3, +10.6] |
| water/bounce_play_per_legal_turn | 52.0 (50) | 55.6 (54) | +3.6 | [-14.0, +20.3] |
| water/bounce_play_removes_opp_entity | 38.5 (26) | 53.3 (30) | +14.9 | [-7.1, +41.2] |

### u9305 vs u8223 — argmax — all|Hydromancy/Shao

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 14.2 (106) | 5.6 (107) | **-8.5** | [-14.9, -2.4] |
| draft/max_wjaccard_to_curated | 17.0 (48) | 15.8 (48) | **-1.2** | [-1.2, -1.2] |
| draft/mean_cost | 2.24 (2400) | 2.28 (2400) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 78.0 (2400) | 82.0 (2400) | **+4.0** | [+4.0, +4.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 25.00 (48) | 23.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 6.0 (2400) | 6.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Hydromancy/portal_readied_ikz | 90.9 (883) | 91.2 (920) | +0.3 | [-1.4, +2.0] |
| gate/Hydromancy/readied_then_spent_same_turn | 77.0 (803) | 75.1 (839) | -1.9 | [-4.8, +1.0] |
| gate/portal_per_legal_turn | 97.6 (905) | 97.1 (947) | -0.4 | [-1.7, +0.9] |
| ikz/held_at_end_of_own_turn | 43.0 (1270) | 46.2 (1278) | **+3.2** | [+0.9, +5.5] |
| ikz/held_then_spent_in_opp_turn | 70.3 (546) | 67.1 (590) | -3.2 | [-6.9, +0.2] |
| leader/Shao/target_is_current_attacker | 45.9 (377) | 46.5 (385) | +0.6 | [-4.9, +6.1] |
| leader/use_per_legal_turn | 89.8 (420) | 88.1 (437) | -1.7 | [-4.2, +1.0] |
| outcome/own_turns | 7.22 (176) | 7.26 (176) | +0.0 | [-0.1, +0.2] |
| outcome/win | 53.4 (176) | 54.0 (176) | +0.6 | [-5.1, +6.2] |
| response/any_nonblock_response_when_legal | 67.3 (615) | 66.1 (660) | -1.3 | [-5.4, +2.7] |
| response/defender_declared_when_legal | 28.0 (378) | 26.3 (407) | -1.8 | [-5.9, +2.3] |
| response/spell_played_when_legal | 19.3 (367) | 19.8 (373) | +0.5 | [-2.8, +3.9] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 70.8 (161) | 66.7 (165) | -4.1 | [-12.6, +4.2] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 70.8 (161) | 66.7 (165) | -4.1 | [-12.6, +4.2] |
| strategy/attacks_by_equipped_attacker | 3.8 (2662) | 3.9 (2671) | +0.1 | [-0.5, +0.8] |
| strategy/face_target_share_when_both_legal | 59.5 (1562) | 57.6 (1569) | -1.9 | [-5.3, +1.2] |
| strategy/favorable_trade_taken_per_available_turn | 56.0 (452) | 58.3 (460) | +2.3 | [-2.4, +6.8] |
| strategy/spell_cast_per_legal_main_turn | 44.2 (477) | 43.1 (480) | -1.1 | [-5.1, +2.8] |
| water/bounce_play_per_legal_turn | 72.0 (672) | 71.4 (689) | -0.6 | [-3.4, +2.2] |
| water/bounce_play_removes_opp_entity | 48.3 (515) | 48.6 (521) | +0.2 | [-3.1, +3.5] |

### u9305 vs u8223 — argmax — fixed|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 14.2 (106) | 5.6 (107) | **-8.5** | [-14.7, -2.3] |
| gate/Hydromancy/portal_readied_ikz | 90.0 (630) | 89.5 (656) | -0.5 | [-2.4, +1.3] |
| gate/Hydromancy/readied_then_spent_same_turn | 73.2 (567) | 73.8 (587) | +0.6 | [-2.4, +3.5] |
| gate/portal_per_legal_turn | 97.1 (649) | 96.6 (679) | -0.5 | [-2.1, +1.2] |
| ikz/held_at_end_of_own_turn | 45.2 (954) | 48.1 (964) | **+3.0** | [+0.6, +5.5] |
| ikz/held_then_spent_in_opp_turn | 71.5 (431) | 68.3 (464) | -3.1 | [-7.0, +0.6] |
| leader/Shao/target_is_current_attacker | 46.7 (300) | 47.1 (306) | +0.4 | [-5.6, +6.4] |
| leader/use_per_legal_turn | 88.0 (341) | 86.0 (356) | -2.0 | [-5.1, +0.8] |
| outcome/own_turns | 7.45 (128) | 7.53 (128) | +0.1 | [-0.1, +0.3] |
| outcome/win | 43.8 (128) | 43.0 (128) | -0.8 | [-7.8, +5.5] |
| outcome/win_vs_EARTH | 50.0 (32) | 40.6 (32) | -9.4 | [-19.4, +0.0] |
| outcome/win_vs_FIRE | 21.9 (32) | 21.9 (32) | +0.0 | [+0.0, +0.0] |
| outcome/win_vs_LIGHTNING | 56.2 (32) | 62.5 (32) | +6.2 | [-10.7, +25.0] |
| outcome/win_vs_WATER | 46.9 (32) | 46.9 (32) | +0.0 | [-18.8, +16.7] |
| response/any_nonblock_response_when_legal | 63.1 (534) | 61.7 (579) | -1.5 | [-5.7, +2.8] |
| response/defender_declared_when_legal | 28.0 (378) | 26.3 (407) | -1.8 | [-5.9, +2.3] |
| response/spell_played_when_legal | 19.3 (367) | 19.8 (373) | +0.5 | [-2.9, +3.9] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 72.1 (122) | 68.5 (124) | -3.6 | [-12.2, +5.3] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 72.1 (122) | 68.5 (124) | -3.6 | [-12.2, +5.3] |
| strategy/attacks_by_equipped_attacker | 4.1 (1869) | 4.1 (1869) | -0.1 | [-0.8, +0.7] |
| strategy/face_target_share_when_both_legal | 56.4 (1109) | 55.5 (1131) | -0.9 | [-4.8, +2.6] |
| strategy/favorable_trade_taken_per_available_turn | 59.2 (338) | 61.4 (339) | +2.2 | [-3.0, +7.5] |
| strategy/spell_cast_per_legal_main_turn | 44.2 (477) | 43.1 (480) | -1.1 | [-5.1, +2.9] |
| water/bounce_play_per_legal_turn | 72.9 (627) | 72.3 (635) | -0.6 | [-2.9, +1.8] |
| water/bounce_play_removes_opp_entity | 48.2 (488) | 49.1 (487) | +0.9 | [-2.3, +4.2] |

### u9305 vs u8223 — argmax — fixed|Hydromancy/Shao

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 14.2 (106) | 5.6 (107) | **-8.5** | [-14.7, -2.3] |
| gate/Hydromancy/portal_readied_ikz | 90.0 (630) | 89.5 (656) | -0.5 | [-2.4, +1.3] |
| gate/Hydromancy/readied_then_spent_same_turn | 73.2 (567) | 73.8 (587) | +0.6 | [-2.4, +3.5] |
| gate/portal_per_legal_turn | 97.1 (649) | 96.6 (679) | -0.5 | [-2.1, +1.2] |
| ikz/held_at_end_of_own_turn | 45.2 (954) | 48.1 (964) | **+3.0** | [+0.6, +5.5] |
| ikz/held_then_spent_in_opp_turn | 71.5 (431) | 68.3 (464) | -3.1 | [-7.0, +0.6] |
| leader/Shao/target_is_current_attacker | 46.7 (300) | 47.1 (306) | +0.4 | [-5.6, +6.4] |
| leader/use_per_legal_turn | 88.0 (341) | 86.0 (356) | -2.0 | [-5.1, +0.8] |
| outcome/own_turns | 7.45 (128) | 7.53 (128) | +0.1 | [-0.1, +0.3] |
| outcome/win | 43.8 (128) | 43.0 (128) | -0.8 | [-7.8, +5.5] |
| response/any_nonblock_response_when_legal | 63.1 (534) | 61.7 (579) | -1.5 | [-5.7, +2.8] |
| response/defender_declared_when_legal | 28.0 (378) | 26.3 (407) | -1.8 | [-5.9, +2.3] |
| response/spell_played_when_legal | 19.3 (367) | 19.8 (373) | +0.5 | [-2.9, +3.9] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 72.1 (122) | 68.5 (124) | -3.6 | [-12.2, +5.3] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 72.1 (122) | 68.5 (124) | -3.6 | [-12.2, +5.3] |
| strategy/attacks_by_equipped_attacker | 4.1 (1869) | 4.1 (1869) | -0.1 | [-0.8, +0.7] |
| strategy/face_target_share_when_both_legal | 56.4 (1109) | 55.5 (1131) | -0.9 | [-4.8, +2.6] |
| strategy/favorable_trade_taken_per_available_turn | 59.2 (338) | 61.4 (339) | +2.2 | [-3.0, +7.5] |
| strategy/spell_cast_per_legal_main_turn | 44.2 (477) | 43.1 (480) | -1.1 | [-5.1, +2.9] |
| water/bounce_play_per_legal_turn | 72.9 (627) | 72.3 (635) | -0.6 | [-2.9, +1.8] |
| water/bounce_play_removes_opp_entity | 48.2 (488) | 49.1 (487) | +0.9 | [-2.3, +4.2] |

### u9305 vs u8223 — argmax — free_draft|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 15.2 (96) | 14.0 (96) | **-1.2** | [-1.2, -1.2] |
| draft/mean_cost | 2.31 (4800) | 2.33 (4800) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 83.0 (4800) | 86.0 (4800) | **+3.0** | [+2.7, +3.3] |
| draft/spell_share | 0.0 (4800) | 0.0 (4800) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 23.50 (96) | 22.00 (96) | **-1.5** | [-1.6, -1.4] |
| draft/weapon_share | 7.0 (4800) | 7.0 (4800) | +0.0 | [+0.0, +0.0] |
| gate/Hydromancy/portal_readied_ikz | 91.6 (514) | 95.5 (531) | **+3.8** | [+1.1, +6.3] |
| gate/Hydromancy/readied_then_spent_same_turn | 84.7 (471) | 80.1 (507) | **-4.6** | [-8.9, -0.6] |
| gate/portal_per_legal_turn | 97.5 (527) | 98.5 (539) | +1.0 | [-0.5, +2.5] |
| ikz/held_at_end_of_own_turn | 37.3 (628) | 40.7 (634) | +3.4 | [-0.8, +7.6] |
| ikz/held_then_spent_in_opp_turn | 32.5 (234) | 30.6 (258) | -1.9 | [-8.8, +5.5] |
| leader/Shao/target_is_current_attacker | 42.9 (77) | 44.3 (79) | +1.4 | [-12.1, +13.6] |
| leader/use_per_legal_turn | 83.9 (93) | 84.7 (98) | +0.8 | [-6.6, +8.8] |
| outcome/own_turns | 6.54 (96) | 6.60 (96) | +0.1 | [-0.2, +0.3] |
| outcome/win | 75.0 (96) | 76.0 (96) | +1.0 | [-8.3, +11.5] |
| outcome/win_vs_EARTH | 75.0 (24) | 83.3 (24) | +8.3 | [-10.0, +31.2] |
| outcome/win_vs_FIRE | 66.7 (24) | 58.3 (24) | -8.3 | [-28.6, +13.6] |
| outcome/win_vs_LIGHTNING | 87.5 (24) | 83.3 (24) | -4.2 | [-22.7, +16.7] |
| outcome/win_vs_WATER | 70.8 (24) | 79.2 (24) | +8.3 | [-12.5, +28.1] |
| response/any_nonblock_response_when_legal | 95.1 (81) | 97.5 (81) | +2.5 | [-3.0, +9.6] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 66.7 (39) | 61.0 (41) | -5.7 | [-28.4, +16.2] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 66.7 (39) | 61.0 (41) | -5.7 | [-28.4, +16.2] |
| strategy/attacks_by_equipped_attacker | 3.2 (1587) | 3.2 (1572) | +0.0 | [-1.0, +1.1] |
| strategy/face_target_share_when_both_legal | 63.0 (944) | 60.9 (911) | -2.1 | [-6.9, +2.5] |
| strategy/favorable_trade_taken_per_available_turn | 50.4 (246) | 51.2 (252) | +0.8 | [-6.8, +7.9] |
| water/bounce_play_per_legal_turn | 55.8 (95) | 58.3 (108) | +2.5 | [-10.8, +15.8] |
| water/bounce_play_removes_opp_entity | 45.3 (53) | 46.9 (64) | +1.6 | [-13.7, +19.8] |

### u9305 vs u8223 — argmax — free_draft|Hydromancy/Benzai

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 13.4 (48) | 12.2 (48) | **-1.2** | [-1.2, -1.2] |
| draft/mean_cost | 2.38 (2400) | 2.38 (2400) | +0.0 | [+0.0, +0.0] |
| draft/normal_share | 88.0 (2400) | 90.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 22.00 (48) | 21.00 (48) | **-1.0** | [-1.0, -1.0] |
| draft/weapon_share | 8.0 (2400) | 8.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Hydromancy/portal_readied_ikz | 90.0 (261) | 95.5 (267) | **+5.5** | [+2.0, +8.8] |
| gate/Hydromancy/readied_then_spent_same_turn | 83.4 (235) | 82.0 (255) | -1.4 | [-6.9, +3.7] |
| gate/portal_per_legal_turn | 96.3 (271) | 98.5 (271) | **+2.2** | [+0.1, +4.2] |
| ikz/held_at_end_of_own_turn | 38.1 (312) | 41.2 (320) | +3.1 | [-3.2, +9.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (119) | 0.0 (132) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 7.1 (14) | 23.5 (17) | +16.4 | [-0.4, +38.9] |
| outcome/own_turns | 6.50 (48) | 6.67 (48) | +0.2 | [-0.1, +0.4] |
| outcome/win | 70.8 (48) | 68.8 (48) | -2.1 | [-16.7, +14.6] |
| strategy/attacks_by_equipped_attacker | 3.4 (794) | 3.0 (770) | -0.4 | [-1.8, +0.9] |
| strategy/face_target_share_when_both_legal | 59.5 (491) | 59.2 (473) | -0.3 | [-6.6, +7.0] |
| strategy/favorable_trade_taken_per_available_turn | 53.8 (132) | 52.7 (131) | -1.1 | [-12.3, +10.6] |
| water/bounce_play_per_legal_turn | 52.0 (50) | 55.6 (54) | +3.6 | [-14.0, +20.3] |
| water/bounce_play_removes_opp_entity | 38.5 (26) | 53.3 (30) | +14.9 | [-7.1, +41.2] |

### u9305 vs u8223 — argmax — free_draft|Hydromancy/Shao

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 17.0 (48) | 15.8 (48) | **-1.2** | [-1.2, -1.2] |
| draft/mean_cost | 2.24 (2400) | 2.28 (2400) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 78.0 (2400) | 82.0 (2400) | **+4.0** | [+4.0, +4.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 25.00 (48) | 23.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 6.0 (2400) | 6.0 (2400) | +0.0 | [+0.0, +0.0] |
| gate/Hydromancy/portal_readied_ikz | 93.3 (253) | 95.5 (264) | +2.2 | [-1.9, +6.0] |
| gate/Hydromancy/readied_then_spent_same_turn | 86.0 (236) | 78.2 (252) | **-7.8** | [-14.3, -1.6] |
| gate/portal_per_legal_turn | 98.8 (256) | 98.5 (268) | -0.3 | [-2.2, +1.6] |
| ikz/held_at_end_of_own_turn | 36.4 (316) | 40.1 (314) | +3.7 | [-1.9, +9.4] |
| ikz/held_then_spent_in_opp_turn | 66.1 (115) | 62.7 (126) | -3.4 | [-11.8, +6.6] |
| leader/Shao/target_is_current_attacker | 42.9 (77) | 44.3 (79) | +1.4 | [-12.6, +13.4] |
| leader/use_per_legal_turn | 97.5 (79) | 97.5 (81) | +0.1 | [-3.5, +3.9] |
| outcome/own_turns | 6.58 (48) | 6.54 (48) | -0.0 | [-0.4, +0.2] |
| outcome/win | 79.2 (48) | 83.3 (48) | +4.2 | [-8.3, +16.7] |
| response/any_nonblock_response_when_legal | 95.1 (81) | 97.5 (81) | +2.5 | [-2.8, +9.2] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 66.7 (39) | 61.0 (41) | -5.7 | [-28.7, +15.4] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 66.7 (39) | 61.0 (41) | -5.7 | [-28.7, +15.4] |
| strategy/attacks_by_equipped_attacker | 2.9 (793) | 3.4 (802) | +0.5 | [-1.0, +2.1] |
| strategy/face_target_share_when_both_legal | 66.9 (453) | 62.8 (438) | -4.1 | [-11.1, +2.2] |
| strategy/favorable_trade_taken_per_available_turn | 46.5 (114) | 49.6 (121) | +3.1 | [-6.8, +11.9] |
| water/bounce_play_per_legal_turn | 60.0 (45) | 61.1 (54) | +1.1 | [-21.4, +23.6] |
| water/bounce_play_removes_opp_entity | 51.9 (27) | 41.2 (34) | -10.7 | [-30.5, +11.7] |

### water vs u8223 — argmax — all|ALL

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 14.2 (106) | 8.7 (115) | -5.5 | [-13.6, +3.4] |
| draft/max_wjaccard_to_curated | 15.2 (96) | 13.4 (96) | **-1.8** | [-2.3, -1.3] |
| draft/mean_cost | 2.31 (4800) | 2.06 (4800) | **-0.2** | [-0.3, -0.2] |
| draft/normal_share | 83.0 (4800) | 82.0 (4800) | -1.0 | [-2.4, +0.4] |
| draft/spell_share | 0.0 (4800) | 0.0 (4800) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 23.50 (96) | 18.00 (96) | **-5.5** | [-5.9, -5.1] |
| draft/weapon_share | 7.0 (4800) | 12.0 (4800) | **+5.0** | [+4.2, +5.9] |
| gate/Hydromancy/portal_readied_ikz | 90.7 (1144) | 92.5 (1153) | +1.7 | [-0.1, +3.6] |
| gate/Hydromancy/readied_then_spent_same_turn | 78.4 (1038) | 77.0 (1066) | -1.4 | [-4.3, +1.6] |
| gate/portal_per_legal_turn | 97.3 (1176) | 97.5 (1182) | +0.3 | [-1.1, +1.7] |
| ikz/held_at_end_of_own_turn | 42.0 (1582) | 44.3 (1605) | +2.3 | [-0.6, +5.0] |
| ikz/held_then_spent_in_opp_turn | 57.7 (665) | 58.6 (711) | +0.9 | [-3.3, +4.9] |
| leader/Shao/target_is_current_attacker | 45.9 (377) | 51.7 (400) | +5.9 | [-0.9, +12.2] |
| leader/use_per_legal_turn | 87.1 (434) | 87.0 (463) | -0.1 | [-2.9, +2.9] |
| outcome/own_turns | 7.06 (224) | 7.17 (224) | +0.1 | [-0.1, +0.3] |
| outcome/win | 57.1 (224) | 63.8 (224) | **+6.7** | [+0.4, +13.4] |
| outcome/win_vs_EARTH | 60.7 (56) | 66.1 (56) | +5.4 | [-10.9, +21.4] |
| outcome/win_vs_FIRE | 41.1 (56) | 48.2 (56) | +7.1 | [-2.1, +17.2] |
| outcome/win_vs_LIGHTNING | 69.6 (56) | 69.6 (56) | +0.0 | [-11.5, +12.0] |
| outcome/win_vs_WATER | 57.1 (56) | 71.4 (56) | **+14.3** | [+3.0, +26.7] |
| response/any_nonblock_response_when_legal | 67.3 (615) | 73.5 (618) | **+6.1** | [+0.5, +12.2] |
| response/defender_declared_when_legal | 28.0 (378) | 31.3 (367) | +3.3 | [-1.5, +8.1] |
| response/spell_played_when_legal | 19.3 (367) | 22.1 (344) | +2.7 | [-2.3, +8.4] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 70.8 (161) | 75.8 (161) | +5.0 | [-3.5, +13.5] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 70.8 (161) | 75.8 (161) | +5.0 | [-3.5, +13.5] |
| strategy/attacks_by_equipped_attacker | 3.7 (3456) | 3.9 (3526) | +0.2 | [-0.5, +0.8] |
| strategy/face_target_share_when_both_legal | 59.5 (2053) | 61.1 (2058) | +1.7 | [-2.2, +5.4] |
| strategy/favorable_trade_taken_per_available_turn | 55.5 (584) | 57.0 (584) | +1.5 | [-3.1, +6.7] |
| strategy/spell_cast_per_legal_main_turn | 44.2 (477) | 46.8 (485) | +2.6 | [-2.2, +7.0] |
| water/bounce_play_per_legal_turn | 70.6 (722) | 65.3 (862) | **-5.3** | [-9.3, -1.2] |
| water/bounce_play_removes_opp_entity | 47.9 (541) | 45.5 (593) | -2.3 | [-6.5, +1.7] |

### water vs u8223 — argmax — all|Hydromancy/Benzai

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 13.4 (48) | 13.4 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 2.38 (2400) | 2.20 (2400) | **-0.2** | [-0.2, -0.2] |
| draft/normal_share | 88.0 (2400) | 82.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 22.00 (48) | 18.00 (48) | **-4.0** | [-4.0, -4.0] |
| draft/weapon_share | 8.0 (2400) | 16.0 (2400) | **+8.0** | [+8.0, +8.0] |
| gate/Hydromancy/portal_readied_ikz | 90.0 (261) | 95.2 (249) | **+5.1** | [+0.5, +9.8] |
| gate/Hydromancy/readied_then_spent_same_turn | 83.4 (235) | 83.1 (237) | -0.3 | [-7.5, +6.2] |
| gate/portal_per_legal_turn | 96.3 (271) | 98.8 (252) | **+2.5** | [+0.4, +4.8] |
| ikz/held_at_end_of_own_turn | 38.1 (312) | 37.6 (322) | -0.6 | [-8.4, +6.7] |
| ikz/held_then_spent_in_opp_turn | 0.0 (119) | 0.0 (121) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 7.1 (14) | 18.8 (16) | +11.6 | [-10.9, +33.3] |
| outcome/own_turns | 6.50 (48) | 6.71 (48) | +0.2 | [-0.1, +0.6] |
| outcome/win | 70.8 (48) | 81.2 (48) | +10.4 | [-4.2, +27.1] |
| strategy/attacks_by_equipped_attacker | 3.4 (794) | 4.6 (791) | +1.2 | [-0.4, +2.7] |
| strategy/face_target_share_when_both_legal | 59.5 (491) | 64.0 (519) | +4.5 | [-4.3, +13.8] |
| strategy/favorable_trade_taken_per_available_turn | 53.8 (132) | 54.8 (126) | +1.0 | [-9.2, +11.5] |
| water/bounce_play_per_legal_turn | 52.0 (50) | 47.0 (115) | -5.0 | [-22.3, +12.3] |
| water/bounce_play_removes_opp_entity | 38.5 (26) | 35.1 (57) | -3.4 | [-23.3, +20.1] |

### water vs u8223 — argmax — all|Hydromancy/Shao

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 14.2 (106) | 8.7 (115) | -5.5 | [-14.0, +3.6] |
| draft/max_wjaccard_to_curated | 17.0 (48) | 13.4 (48) | **-3.6** | [-3.6, -3.6] |
| draft/mean_cost | 2.24 (2400) | 1.92 (2400) | **-0.3** | [-0.3, -0.3] |
| draft/normal_share | 78.0 (2400) | 82.0 (2400) | **+4.0** | [+4.0, +4.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 25.00 (48) | 18.00 (48) | **-7.0** | [-7.0, -7.0] |
| draft/weapon_share | 6.0 (2400) | 8.0 (2400) | **+2.0** | [+2.0, +2.0] |
| gate/Hydromancy/portal_readied_ikz | 90.9 (883) | 91.7 (904) | +0.8 | [-1.2, +2.7] |
| gate/Hydromancy/readied_then_spent_same_turn | 77.0 (803) | 75.3 (829) | -1.7 | [-4.9, +1.6] |
| gate/portal_per_legal_turn | 97.6 (905) | 97.2 (930) | -0.4 | [-2.0, +1.3] |
| ikz/held_at_end_of_own_turn | 43.0 (1270) | 46.0 (1283) | **+3.0** | [+0.0, +5.9] |
| ikz/held_then_spent_in_opp_turn | 70.3 (546) | 70.7 (590) | +0.3 | [-3.3, +4.0] |
| leader/Shao/target_is_current_attacker | 45.9 (377) | 51.7 (400) | +5.9 | [-1.1, +12.3] |
| leader/use_per_legal_turn | 89.8 (420) | 89.5 (447) | -0.3 | [-2.9, +2.4] |
| outcome/own_turns | 7.22 (176) | 7.29 (176) | +0.1 | [-0.1, +0.3] |
| outcome/win | 53.4 (176) | 59.1 (176) | +5.7 | [-0.6, +12.5] |
| response/any_nonblock_response_when_legal | 67.3 (615) | 73.5 (618) | **+6.1** | [+0.6, +11.8] |
| response/defender_declared_when_legal | 28.0 (378) | 31.3 (367) | +3.3 | [-1.7, +8.4] |
| response/spell_played_when_legal | 19.3 (367) | 22.1 (344) | +2.7 | [-2.3, +8.1] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 70.8 (161) | 75.8 (161) | +5.0 | [-3.5, +13.5] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 70.8 (161) | 75.8 (161) | +5.0 | [-3.5, +13.5] |
| strategy/attacks_by_equipped_attacker | 3.8 (2662) | 3.7 (2735) | -0.1 | [-0.8, +0.6] |
| strategy/face_target_share_when_both_legal | 59.5 (1562) | 60.2 (1539) | +0.7 | [-3.2, +4.6] |
| strategy/favorable_trade_taken_per_available_turn | 56.0 (452) | 57.6 (458) | +1.7 | [-3.7, +7.4] |
| strategy/spell_cast_per_legal_main_turn | 44.2 (477) | 46.8 (485) | +2.6 | [-1.9, +7.0] |
| water/bounce_play_per_legal_turn | 72.0 (672) | 68.1 (747) | **-3.9** | [-7.8, -0.2] |
| water/bounce_play_removes_opp_entity | 48.3 (515) | 46.6 (536) | -1.7 | [-6.0, +2.4] |

### water vs u8223 — argmax — fixed|ALL

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 14.2 (106) | 8.7 (115) | -5.5 | [-13.8, +3.6] |
| gate/Hydromancy/portal_readied_ikz | 90.0 (630) | 90.7 (655) | +0.7 | [-1.7, +3.0] |
| gate/Hydromancy/readied_then_spent_same_turn | 73.2 (567) | 72.6 (594) | -0.6 | [-4.0, +2.8] |
| gate/portal_per_legal_turn | 97.1 (649) | 96.9 (676) | -0.2 | [-2.2, +2.0] |
| ikz/held_at_end_of_own_turn | 45.2 (954) | 48.1 (983) | +2.9 | [-0.5, +6.4] |
| ikz/held_then_spent_in_opp_turn | 71.5 (431) | 73.4 (473) | +1.9 | [-2.0, +5.6] |
| leader/Shao/target_is_current_attacker | 46.7 (300) | 53.2 (329) | +6.5 | [-1.5, +14.0] |
| leader/use_per_legal_turn | 88.0 (341) | 87.7 (375) | -0.2 | [-3.2, +2.8] |
| outcome/own_turns | 7.45 (128) | 7.68 (128) | +0.2 | [-0.0, +0.5] |
| outcome/win | 43.8 (128) | 50.0 (128) | +6.2 | [-1.6, +14.1] |
| outcome/win_vs_EARTH | 50.0 (32) | 50.0 (32) | +0.0 | [-20.0, +20.0] |
| outcome/win_vs_FIRE | 21.9 (32) | 28.1 (32) | +6.2 | [-5.9, +19.2] |
| outcome/win_vs_LIGHTNING | 56.2 (32) | 59.4 (32) | +3.1 | [-11.5, +17.9] |
| outcome/win_vs_WATER | 46.9 (32) | 62.5 (32) | **+15.6** | [+3.1, +31.2] |
| response/any_nonblock_response_when_legal | 63.1 (534) | 70.1 (546) | **+7.0** | [+1.1, +13.5] |
| response/defender_declared_when_legal | 28.0 (378) | 31.3 (367) | +3.3 | [-1.5, +8.4] |
| response/spell_played_when_legal | 19.3 (367) | 22.1 (344) | +2.7 | [-2.2, +8.3] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 72.1 (122) | 80.5 (123) | +8.4 | [-0.6, +18.1] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 72.1 (122) | 80.5 (123) | +8.4 | [-0.6, +18.1] |
| strategy/attacks_by_equipped_attacker | 4.1 (1869) | 3.8 (1937) | -0.4 | [-1.2, +0.5] |
| strategy/face_target_share_when_both_legal | 56.4 (1109) | 56.7 (1097) | +0.3 | [-3.9, +4.6] |
| strategy/favorable_trade_taken_per_available_turn | 59.2 (338) | 60.1 (358) | +0.9 | [-4.9, +6.6] |
| strategy/spell_cast_per_legal_main_turn | 44.2 (477) | 46.8 (485) | +2.6 | [-2.1, +7.0] |
| water/bounce_play_per_legal_turn | 72.9 (627) | 72.0 (639) | -0.9 | [-4.4, +2.6] |
| water/bounce_play_removes_opp_entity | 48.2 (488) | 47.9 (486) | -0.2 | [-4.3, +4.0] |

### water vs u8223 — argmax — fixed|Hydromancy/Shao

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 14.2 (106) | 8.7 (115) | -5.5 | [-13.8, +3.6] |
| gate/Hydromancy/portal_readied_ikz | 90.0 (630) | 90.7 (655) | +0.7 | [-1.7, +3.0] |
| gate/Hydromancy/readied_then_spent_same_turn | 73.2 (567) | 72.6 (594) | -0.6 | [-4.0, +2.8] |
| gate/portal_per_legal_turn | 97.1 (649) | 96.9 (676) | -0.2 | [-2.2, +2.0] |
| ikz/held_at_end_of_own_turn | 45.2 (954) | 48.1 (983) | +2.9 | [-0.5, +6.4] |
| ikz/held_then_spent_in_opp_turn | 71.5 (431) | 73.4 (473) | +1.9 | [-2.0, +5.6] |
| leader/Shao/target_is_current_attacker | 46.7 (300) | 53.2 (329) | +6.5 | [-1.5, +14.0] |
| leader/use_per_legal_turn | 88.0 (341) | 87.7 (375) | -0.2 | [-3.2, +2.8] |
| outcome/own_turns | 7.45 (128) | 7.68 (128) | +0.2 | [-0.0, +0.5] |
| outcome/win | 43.8 (128) | 50.0 (128) | +6.2 | [-1.6, +14.1] |
| response/any_nonblock_response_when_legal | 63.1 (534) | 70.1 (546) | **+7.0** | [+1.1, +13.5] |
| response/defender_declared_when_legal | 28.0 (378) | 31.3 (367) | +3.3 | [-1.5, +8.4] |
| response/spell_played_when_legal | 19.3 (367) | 22.1 (344) | +2.7 | [-2.2, +8.3] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 72.1 (122) | 80.5 (123) | +8.4 | [-0.6, +18.1] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 72.1 (122) | 80.5 (123) | +8.4 | [-0.6, +18.1] |
| strategy/attacks_by_equipped_attacker | 4.1 (1869) | 3.8 (1937) | -0.4 | [-1.2, +0.5] |
| strategy/face_target_share_when_both_legal | 56.4 (1109) | 56.7 (1097) | +0.3 | [-3.9, +4.6] |
| strategy/favorable_trade_taken_per_available_turn | 59.2 (338) | 60.1 (358) | +0.9 | [-4.9, +6.6] |
| strategy/spell_cast_per_legal_main_turn | 44.2 (477) | 46.8 (485) | +2.6 | [-2.1, +7.0] |
| water/bounce_play_per_legal_turn | 72.9 (627) | 72.0 (639) | -0.9 | [-4.4, +2.6] |
| water/bounce_play_removes_opp_entity | 48.2 (488) | 47.9 (486) | -0.2 | [-4.3, +4.0] |

### water vs u8223 — argmax — free_draft|ALL

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 15.2 (96) | 13.4 (96) | **-1.8** | [-2.3, -1.3] |
| draft/mean_cost | 2.31 (4800) | 2.06 (4800) | **-0.2** | [-0.3, -0.2] |
| draft/normal_share | 83.0 (4800) | 82.0 (4800) | -1.0 | [-2.5, +0.5] |
| draft/spell_share | 0.0 (4800) | 0.0 (4800) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 23.50 (96) | 18.00 (96) | **-5.5** | [-5.9, -5.1] |
| draft/weapon_share | 7.0 (4800) | 12.0 (4800) | **+5.0** | [+4.1, +5.9] |
| gate/Hydromancy/portal_readied_ikz | 91.6 (514) | 94.8 (498) | **+3.1** | [+0.2, +6.1] |
| gate/Hydromancy/readied_then_spent_same_turn | 84.7 (471) | 82.6 (472) | -2.1 | [-6.9, +3.1] |
| gate/portal_per_legal_turn | 97.5 (527) | 98.4 (506) | +0.9 | [-0.6, +2.5] |
| ikz/held_at_end_of_own_turn | 37.3 (628) | 38.3 (622) | +1.0 | [-4.0, +5.8] |
| ikz/held_then_spent_in_opp_turn | 32.5 (234) | 29.4 (238) | -3.1 | [-11.9, +6.1] |
| leader/Shao/target_is_current_attacker | 42.9 (77) | 45.1 (71) | +2.2 | [-11.0, +14.9] |
| leader/use_per_legal_turn | 83.9 (93) | 84.1 (88) | +0.2 | [-7.8, +9.2] |
| outcome/own_turns | 6.54 (96) | 6.48 (96) | -0.1 | [-0.3, +0.2] |
| outcome/win | 75.0 (96) | 82.3 (96) | +7.3 | [-3.1, +17.7] |
| outcome/win_vs_EARTH | 75.0 (24) | 87.5 (24) | +12.5 | [-12.5, +42.3] |
| outcome/win_vs_FIRE | 66.7 (24) | 75.0 (24) | +8.3 | [-7.7, +25.0] |
| outcome/win_vs_LIGHTNING | 87.5 (24) | 83.3 (24) | -4.2 | [-22.7, +15.4] |
| outcome/win_vs_WATER | 70.8 (24) | 83.3 (24) | +12.5 | [-8.3, +35.0] |
| response/any_nonblock_response_when_legal | 95.1 (81) | 98.6 (72) | +3.5 | [-0.5, +9.9] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 66.7 (39) | 60.5 (38) | -6.1 | [-24.9, +12.8] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 66.7 (39) | 60.5 (38) | -6.1 | [-24.9, +12.8] |
| strategy/attacks_by_equipped_attacker | 3.2 (1587) | 4.0 (1589) | +0.9 | [-0.1, +1.9] |
| strategy/face_target_share_when_both_legal | 63.0 (944) | 66.2 (961) | +3.2 | [-3.2, +9.5] |
| strategy/favorable_trade_taken_per_available_turn | 50.4 (246) | 52.2 (226) | +1.8 | [-6.8, +10.6] |
| water/bounce_play_per_legal_turn | 55.8 (95) | 46.2 (223) | -9.6 | [-22.6, +3.2] |
| water/bounce_play_removes_opp_entity | 45.3 (53) | 34.6 (107) | -10.7 | [-23.4, +3.3] |

### water vs u8223 — argmax — free_draft|Hydromancy/Benzai

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 13.4 (48) | 13.4 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 2.38 (2400) | 2.20 (2400) | **-0.2** | [-0.2, -0.2] |
| draft/normal_share | 88.0 (2400) | 82.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 22.00 (48) | 18.00 (48) | **-4.0** | [-4.0, -4.0] |
| draft/weapon_share | 8.0 (2400) | 16.0 (2400) | **+8.0** | [+8.0, +8.0] |
| gate/Hydromancy/portal_readied_ikz | 90.0 (261) | 95.2 (249) | **+5.1** | [+0.5, +9.8] |
| gate/Hydromancy/readied_then_spent_same_turn | 83.4 (235) | 83.1 (237) | -0.3 | [-7.5, +6.2] |
| gate/portal_per_legal_turn | 96.3 (271) | 98.8 (252) | **+2.5** | [+0.4, +4.8] |
| ikz/held_at_end_of_own_turn | 38.1 (312) | 37.6 (322) | -0.6 | [-8.4, +6.7] |
| ikz/held_then_spent_in_opp_turn | 0.0 (119) | 0.0 (121) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 7.1 (14) | 18.8 (16) | +11.6 | [-10.9, +33.3] |
| outcome/own_turns | 6.50 (48) | 6.71 (48) | +0.2 | [-0.1, +0.6] |
| outcome/win | 70.8 (48) | 81.2 (48) | +10.4 | [-4.2, +27.1] |
| strategy/attacks_by_equipped_attacker | 3.4 (794) | 4.6 (791) | +1.2 | [-0.4, +2.7] |
| strategy/face_target_share_when_both_legal | 59.5 (491) | 64.0 (519) | +4.5 | [-4.3, +13.8] |
| strategy/favorable_trade_taken_per_available_turn | 53.8 (132) | 54.8 (126) | +1.0 | [-9.2, +11.5] |
| water/bounce_play_per_legal_turn | 52.0 (50) | 47.0 (115) | -5.0 | [-22.3, +12.3] |
| water/bounce_play_removes_opp_entity | 38.5 (26) | 35.1 (57) | -3.4 | [-23.3, +20.1] |

### water vs u8223 — argmax — free_draft|Hydromancy/Shao

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| draft/max_wjaccard_to_curated | 17.0 (48) | 13.4 (48) | **-3.6** | [-3.6, -3.6] |
| draft/mean_cost | 2.24 (2400) | 1.92 (2400) | **-0.3** | [-0.3, -0.3] |
| draft/normal_share | 78.0 (2400) | 82.0 (2400) | **+4.0** | [+4.0, +4.0] |
| draft/spell_share | 0.0 (2400) | 0.0 (2400) | +0.0 | [+0.0, +0.0] |
| draft/unique_cards | 25.00 (48) | 18.00 (48) | **-7.0** | [-7.0, -7.0] |
| draft/weapon_share | 6.0 (2400) | 8.0 (2400) | **+2.0** | [+2.0, +2.0] |
| gate/Hydromancy/portal_readied_ikz | 93.3 (253) | 94.4 (249) | +1.1 | [-2.7, +4.7] |
| gate/Hydromancy/readied_then_spent_same_turn | 86.0 (236) | 82.1 (235) | -3.9 | [-11.0, +2.9] |
| gate/portal_per_legal_turn | 98.8 (256) | 98.0 (254) | -0.8 | [-2.6, +1.1] |
| ikz/held_at_end_of_own_turn | 36.4 (316) | 39.0 (300) | +2.6 | [-3.1, +8.9] |
| ikz/held_then_spent_in_opp_turn | 66.1 (115) | 59.8 (117) | -6.3 | [-16.7, +5.6] |
| leader/Shao/target_is_current_attacker | 42.9 (77) | 45.1 (71) | +2.2 | [-11.0, +15.2] |
| leader/use_per_legal_turn | 97.5 (79) | 98.6 (72) | +1.1 | [-0.5, +4.5] |
| outcome/own_turns | 6.58 (48) | 6.25 (48) | -0.3 | [-0.7, +0.0] |
| outcome/win | 79.2 (48) | 83.3 (48) | +4.2 | [-8.3, +16.7] |
| response/any_nonblock_response_when_legal | 95.1 (81) | 98.6 (72) | +3.5 | [-0.3, +9.7] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 66.7 (39) | 60.5 (38) | -6.1 | [-24.1, +12.9] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 66.7 (39) | 60.5 (38) | -6.1 | [-24.1, +12.9] |
| strategy/attacks_by_equipped_attacker | 2.9 (793) | 3.5 (798) | +0.6 | [-0.7, +1.9] |
| strategy/face_target_share_when_both_legal | 66.9 (453) | 68.8 (442) | +1.9 | [-7.4, +11.1] |
| strategy/favorable_trade_taken_per_available_turn | 46.5 (114) | 49.0 (100) | +2.5 | [-12.0, +17.5] |
| water/bounce_play_per_legal_turn | 60.0 (45) | 45.4 (108) | -14.6 | [-33.8, +3.9] |
| water/bounce_play_removes_opp_entity | 51.9 (27) | 34.0 (50) | **-17.9** | [-34.7, -1.3] |

### water vs u8223 — sample — all|ALL

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 10.3 (117) | 3.6 (112) | -6.7 | [-14.2, +0.5] |
| draft/max_wjaccard_to_curated | 13.3 (96) | 13.2 (96) | -0.1 | [-0.5, +0.2] |
| draft/mean_cost | 2.58 (4800) | 2.61 (4800) | **+0.0** | [+0.0, +0.1] |
| draft/normal_share | 75.8 (4800) | 77.8 (4800) | **+2.0** | [+1.4, +2.7] |
| draft/spell_share | 7.6 (4800) | 6.2 (4800) | **-1.4** | [-1.8, -1.0] |
| draft/unique_cards | 33.64 (96) | 32.11 (96) | **-1.5** | [-2.0, -1.0] |
| draft/weapon_share | 8.8 (4800) | 10.1 (4800) | **+1.3** | [+0.7, +1.9] |
| gate/Hydromancy/portal_readied_ikz | 90.8 (1295) | 92.2 (1187) | +1.4 | [-0.4, +3.3] |
| gate/Hydromancy/readied_then_spent_same_turn | 66.8 (1176) | 70.1 (1095) | +3.3 | [-3.3, +11.0] |
| gate/portal_per_legal_turn | 98.2 (1319) | 98.3 (1207) | +0.2 | [-0.8, +1.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 32.0 (50) | 41.9 (31) | +9.9 | [-10.3, +33.7] |
| ikz/held_at_end_of_own_turn | 49.7 (1790) | 50.3 (1648) | +0.6 | [-3.9, +5.2] |
| ikz/held_then_spent_in_opp_turn | 48.5 (889) | 55.9 (829) | +7.4 | [-0.9, +15.3] |
| leader/Benzai/use_then_card_played_same_turn | 13.7 (51) | 30.0 (10) | +16.3 | [-32.9, +42.5] |
| leader/Shao/target_is_current_attacker | 50.0 (418) | 53.3 (439) | +3.3 | [-3.0, +9.7] |
| leader/use_per_legal_turn | 89.3 (525) | 87.2 (515) | -2.1 | [-5.7, +1.5] |
| outcome/own_turns | 7.99 (224) | 7.36 (224) | **-0.6** | [-1.4, -0.1] |
| outcome/win | 46.0 (224) | 55.8 (224) | **+9.8** | [+3.1, +16.1] |
| outcome/win_vs_EARTH | 50.0 (56) | 57.1 (56) | +7.1 | [-9.1, +24.0] |
| outcome/win_vs_FIRE | 19.6 (56) | 28.6 (56) | **+8.9** | [+1.6, +18.7] |
| outcome/win_vs_LIGHTNING | 58.9 (56) | 69.6 (56) | +10.7 | [+0.0, +21.2] |
| outcome/win_vs_WATER | 55.4 (56) | 67.9 (56) | +12.5 | [+0.0, +26.8] |
| response/any_nonblock_response_when_legal | 69.0 (677) | 73.0 (686) | +4.1 | [-3.3, +11.3] |
| response/defender_declared_when_legal | 29.6 (395) | 28.5 (393) | -1.1 | [-5.4, +3.1] |
| response/spell_played_when_legal | 21.5 (382) | 24.1 (394) | +2.6 | [-3.1, +8.6] |
| sequence/water.healing_flutter_timing/completed_per_eligible_game | 78.9 (19) | 92.3 (13) | +13.4 | [-13.3, +38.5] |
| sequence/water.healing_flutter_timing/converted_per_eligible_game | 78.9 (19) | 92.3 (13) | +13.4 | [-13.3, +38.5] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 79.2 (168) | 79.3 (169) | +0.1 | [-8.9, +8.3] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 79.2 (168) | 79.3 (169) | +0.1 | [-8.9, +8.3] |
| strategy/attacks_by_equipped_attacker | 4.2 (3272) | 4.2 (3445) | +0.1 | [-0.7, +0.9] |
| strategy/face_target_share_when_both_legal | 53.3 (1970) | 56.6 (2056) | +3.3 | [-0.3, +6.9] |
| strategy/favorable_trade_taken_per_available_turn | 60.4 (611) | 63.2 (620) | +2.8 | [-1.9, +7.5] |
| strategy/spell_cast_per_legal_main_turn | 36.8 (774) | 38.3 (687) | +1.5 | [-2.9, +5.8] |
| water/bounce_play_per_legal_turn | 71.3 (954) | 68.8 (791) | -2.5 | [-8.1, +3.3] |
| water/bounce_play_removes_opp_entity | 60.2 (709) | 51.6 (566) | **-8.6** | [-16.9, -0.1] |

### water vs u8223 — sample — all|Hydromancy/Benzai

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 6.2 (16) | 0.0 (7) | -6.2 | [-21.4, +0.0] |
| draft/max_wjaccard_to_curated | 13.4 (48) | 13.3 (48) | -0.1 | [-0.7, +0.4] |
| draft/mean_cost | 2.60 (2400) | 2.66 (2400) | **+0.1** | [+0.0, +0.1] |
| draft/normal_share | 76.3 (2400) | 78.2 (2400) | **+1.8** | [+1.0, +2.7] |
| draft/spell_share | 7.7 (2400) | 6.2 (2400) | **-1.4** | [-1.9, -1.0] |
| draft/unique_cards | 33.27 (48) | 31.79 (48) | **-1.5** | [-2.1, -0.8] |
| draft/weapon_share | 9.2 (2400) | 10.9 (2400) | **+1.8** | [+0.8, +2.7] |
| gate/Hydromancy/portal_readied_ikz | 92.8 (291) | 95.5 (247) | +2.8 | [-0.8, +7.1] |
| gate/Hydromancy/readied_then_spent_same_turn | 77.8 (270) | 74.2 (236) | -3.6 | [-10.6, +3.7] |
| gate/portal_per_legal_turn | 99.3 (293) | 100.0 (247) | +0.7 | [+0.0, +1.8] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.0 (24) | 33.3 (21) | +8.3 | [-25.8, +41.7] |
| ikz/held_at_end_of_own_turn | 46.9 (371) | 47.5 (335) | +0.6 | [-9.1, +11.0] |
| ikz/held_then_spent_in_opp_turn | 1.7 (174) | 3.8 (159) | +2.0 | [-1.1, +5.2] |
| leader/Benzai/use_then_card_played_same_turn | 13.7 (51) | 30.0 (10) | +16.3 | [-28.6, +42.0] |
| leader/use_per_legal_turn | 86.4 (59) | 38.5 (26) | **-48.0** | [-65.9, -11.0] |
| outcome/own_turns | 7.73 (48) | 6.98 (48) | -0.8 | [-2.4, +0.2] |
| outcome/win | 41.7 (48) | 52.1 (48) | +10.4 | [-4.2, +27.1] |
| response/any_nonblock_response_when_legal | 36.0 (25) | 100.0 (6) | +64.0 | [+0.0, +86.7] |
| response/defender_declared_when_legal | 31.4 (51) | 20.6 (34) | -10.8 | [-23.9, +1.4] |
| response/spell_played_when_legal | 15.8 (19) | 100.0 (6) | +84.2 | [+0.0, +100.0] |
| sequence/water.healing_flutter_timing/completed_per_eligible_game | 75.0 (8) | 100.0 (6) | +25.0 | [+0.0, +60.0] |
| sequence/water.healing_flutter_timing/converted_per_eligible_game | 75.0 (8) | 100.0 (6) | +25.0 | [+0.0, +60.0] |
| strategy/attacks_by_equipped_attacker | 4.1 (701) | 4.1 (763) | -0.1 | [-2.1, +1.9] |
| strategy/face_target_share_when_both_legal | 48.9 (454) | 54.6 (500) | +5.7 | [-2.3, +13.5] |
| strategy/favorable_trade_taken_per_available_turn | 56.8 (148) | 64.9 (151) | +8.1 | [-2.2, +17.9] |
| strategy/spell_cast_per_legal_main_turn | 20.1 (139) | 19.0 (121) | -1.1 | [-10.5, +7.1] |
| water/bounce_play_per_legal_turn | 60.2 (113) | 46.5 (71) | -13.7 | [-34.0, +17.3] |
| water/bounce_play_removes_opp_entity | 73.9 (69) | 52.9 (34) | -21.0 | [-44.9, +24.3] |

### water vs u8223 — sample — all|Hydromancy/Shao

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 10.9 (101) | 3.8 (105) | -7.1 | [-15.4, +0.8] |
| draft/max_wjaccard_to_curated | 13.2 (48) | 13.0 (48) | -0.2 | [-0.7, +0.4] |
| draft/mean_cost | 2.55 (2400) | 2.56 (2400) | +0.0 | [-0.0, +0.0] |
| draft/normal_share | 75.3 (2400) | 77.5 (2400) | **+2.2** | [+1.2, +3.2] |
| draft/spell_share | 7.6 (2400) | 6.2 (2400) | **-1.4** | [-2.0, -0.7] |
| draft/unique_cards | 34.00 (48) | 32.44 (48) | **-1.6** | [-2.3, -0.8] |
| draft/weapon_share | 8.5 (2400) | 9.3 (2400) | **+0.8** | [+0.2, +1.5] |
| gate/Hydromancy/portal_readied_ikz | 90.2 (1004) | 91.4 (940) | +1.1 | [-0.9, +3.3] |
| gate/Hydromancy/readied_then_spent_same_turn | 63.6 (906) | 69.0 (859) | +5.5 | [-2.7, +14.5] |
| gate/portal_per_legal_turn | 97.9 (1026) | 97.9 (960) | +0.1 | [-1.2, +1.3] |
| heal/heal_spell_per_legal_turn_below_max_hp | 38.5 (26) | 60.0 (10) | +21.5 | [-6.9, +45.8] |
| ikz/held_at_end_of_own_turn | 50.4 (1419) | 51.0 (1313) | +0.6 | [-4.4, +5.5] |
| ikz/held_then_spent_in_opp_turn | 59.9 (715) | 68.2 (670) | +8.3 | [-2.1, +19.1] |
| leader/Shao/target_is_current_attacker | 50.0 (418) | 53.3 (439) | +3.3 | [-3.3, +9.7] |
| leader/use_per_legal_turn | 89.7 (466) | 89.8 (489) | +0.1 | [-2.9, +2.9] |
| outcome/own_turns | 8.06 (176) | 7.46 (176) | -0.6 | [-1.5, +0.0] |
| outcome/win | 47.2 (176) | 56.8 (176) | **+9.7** | [+2.8, +17.0] |
| response/any_nonblock_response_when_legal | 70.2 (652) | 72.8 (680) | +2.5 | [-4.9, +9.8] |
| response/defender_declared_when_legal | 29.4 (344) | 29.2 (359) | -0.1 | [-4.4, +4.6] |
| response/spell_played_when_legal | 21.8 (363) | 22.9 (388) | +1.2 | [-4.5, +6.9] |
| sequence/water.healing_flutter_timing/completed_per_eligible_game | 81.8 (11) | 85.7 (7) | +3.9 | [-40.9, +40.0] |
| sequence/water.healing_flutter_timing/converted_per_eligible_game | 81.8 (11) | 85.7 (7) | +3.9 | [-40.9, +40.0] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 79.2 (168) | 79.3 (169) | +0.1 | [-8.3, +8.5] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 79.2 (168) | 79.3 (169) | +0.1 | [-8.3, +8.5] |
| strategy/attacks_by_equipped_attacker | 4.2 (2571) | 4.3 (2682) | +0.1 | [-0.8, +0.9] |
| strategy/face_target_share_when_both_legal | 54.6 (1516) | 57.2 (1556) | +2.6 | [-1.9, +6.8] |
| strategy/favorable_trade_taken_per_available_turn | 61.6 (463) | 62.7 (469) | +1.1 | [-4.4, +6.5] |
| strategy/spell_cast_per_legal_main_turn | 40.5 (635) | 42.4 (566) | +1.9 | [-2.6, +6.4] |
| water/bounce_play_per_legal_turn | 72.8 (841) | 71.0 (720) | -1.8 | [-7.4, +3.8] |
| water/bounce_play_removes_opp_entity | 58.8 (640) | 51.5 (532) | -7.2 | [-15.8, +0.7] |

### water vs u8223 — sample — fixed|ALL

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 9.7 (93) | 3.9 (103) | -5.8 | [-14.0, +1.7] |
| gate/Hydromancy/portal_readied_ikz | 89.8 (763) | 90.4 (686) | +0.6 | [-1.8, +3.3] |
| gate/Hydromancy/readied_then_spent_same_turn | 62.3 (685) | 68.2 (620) | +5.9 | [-4.5, +16.7] |
| gate/portal_per_legal_turn | 98.1 (778) | 97.6 (703) | -0.5 | [-2.0, +0.9] |
| ikz/held_at_end_of_own_turn | 51.1 (1082) | 51.2 (992) | +0.1 | [-5.9, +5.9] |
| ikz/held_then_spent_in_opp_turn | 55.2 (553) | 67.7 (508) | **+12.6** | [+0.4, +24.4] |
| leader/Shao/target_is_current_attacker | 51.5 (295) | 55.2 (326) | +3.7 | [-4.2, +11.0] |
| leader/use_per_legal_turn | 87.5 (337) | 88.1 (370) | +0.6 | [-3.0, +4.1] |
| outcome/own_turns | 8.45 (128) | 7.75 (128) | -0.7 | [-1.9, +0.1] |
| outcome/win | 45.3 (128) | 53.9 (128) | **+8.6** | [+1.6, +15.6] |
| outcome/win_vs_EARTH | 50.0 (32) | 56.2 (32) | +6.2 | [-8.3, +21.9] |
| outcome/win_vs_FIRE | 21.9 (32) | 25.0 (32) | +3.1 | [+0.0, +10.0] |
| outcome/win_vs_LIGHTNING | 56.2 (32) | 68.8 (32) | +12.5 | [+0.0, +26.9] |
| outcome/win_vs_WATER | 53.1 (32) | 65.6 (32) | +12.5 | [-5.6, +33.3] |
| response/any_nonblock_response_when_legal | 65.6 (509) | 70.4 (540) | +4.8 | [-2.9, +12.4] |
| response/defender_declared_when_legal | 28.7 (324) | 29.4 (350) | +0.7 | [-3.8, +5.4] |
| response/spell_played_when_legal | 22.0 (345) | 24.1 (369) | +2.1 | [-3.5, +7.9] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 75.6 (123) | 79.7 (123) | +4.1 | [-5.7, +13.1] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 75.6 (123) | 79.7 (123) | +4.1 | [-5.7, +13.1] |
| strategy/attacks_by_equipped_attacker | 4.0 (1858) | 4.1 (1933) | +0.1 | [-0.8, +0.8] |
| strategy/face_target_share_when_both_legal | 53.0 (1078) | 57.0 (1108) | +4.1 | [-0.8, +8.9] |
| strategy/favorable_trade_taken_per_available_turn | 63.5 (340) | 62.4 (335) | -1.1 | [-7.4, +4.7] |
| strategy/spell_cast_per_legal_main_turn | 43.6 (500) | 44.7 (494) | +1.1 | [-3.6, +5.9] |
| water/bounce_play_per_legal_turn | 75.6 (757) | 72.3 (657) | -3.3 | [-8.9, +2.0] |
| water/bounce_play_removes_opp_entity | 59.4 (599) | 52.5 (495) | -6.9 | [-15.4, +1.2] |

### water vs u8223 — sample — fixed|Hydromancy/Shao

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 9.7 (93) | 3.9 (103) | -5.8 | [-14.0, +1.7] |
| gate/Hydromancy/portal_readied_ikz | 89.8 (763) | 90.4 (686) | +0.6 | [-1.8, +3.3] |
| gate/Hydromancy/readied_then_spent_same_turn | 62.3 (685) | 68.2 (620) | +5.9 | [-4.5, +16.7] |
| gate/portal_per_legal_turn | 98.1 (778) | 97.6 (703) | -0.5 | [-2.0, +0.9] |
| ikz/held_at_end_of_own_turn | 51.1 (1082) | 51.2 (992) | +0.1 | [-5.9, +5.9] |
| ikz/held_then_spent_in_opp_turn | 55.2 (553) | 67.7 (508) | **+12.6** | [+0.4, +24.4] |
| leader/Shao/target_is_current_attacker | 51.5 (295) | 55.2 (326) | +3.7 | [-4.2, +11.0] |
| leader/use_per_legal_turn | 87.5 (337) | 88.1 (370) | +0.6 | [-3.0, +4.1] |
| outcome/own_turns | 8.45 (128) | 7.75 (128) | -0.7 | [-1.9, +0.1] |
| outcome/win | 45.3 (128) | 53.9 (128) | **+8.6** | [+1.6, +15.6] |
| response/any_nonblock_response_when_legal | 65.6 (509) | 70.4 (540) | +4.8 | [-2.9, +12.4] |
| response/defender_declared_when_legal | 28.7 (324) | 29.4 (350) | +0.7 | [-3.8, +5.4] |
| response/spell_played_when_legal | 22.0 (345) | 24.1 (369) | +2.1 | [-3.5, +7.9] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 75.6 (123) | 79.7 (123) | +4.1 | [-5.7, +13.1] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 75.6 (123) | 79.7 (123) | +4.1 | [-5.7, +13.1] |
| strategy/attacks_by_equipped_attacker | 4.0 (1858) | 4.1 (1933) | +0.1 | [-0.8, +0.8] |
| strategy/face_target_share_when_both_legal | 53.0 (1078) | 57.0 (1108) | +4.1 | [-0.8, +8.9] |
| strategy/favorable_trade_taken_per_available_turn | 63.5 (340) | 62.4 (335) | -1.1 | [-7.4, +4.7] |
| strategy/spell_cast_per_legal_main_turn | 43.6 (500) | 44.7 (494) | +1.1 | [-3.6, +5.9] |
| water/bounce_play_per_legal_turn | 75.6 (757) | 72.3 (657) | -3.3 | [-8.9, +2.0] |
| water/bounce_play_removes_opp_entity | 59.4 (599) | 52.5 (495) | -6.9 | [-15.4, +1.2] |

### water vs u8223 — sample — free_draft|ALL

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 12.5 (24) | 0.0 (9) | -12.5 | [-31.2, +0.0] |
| draft/max_wjaccard_to_curated | 13.3 (96) | 13.2 (96) | -0.1 | [-0.5, +0.2] |
| draft/mean_cost | 2.58 (4800) | 2.61 (4800) | **+0.0** | [+0.0, +0.1] |
| draft/normal_share | 75.8 (4800) | 77.8 (4800) | **+2.0** | [+1.4, +2.7] |
| draft/spell_share | 7.6 (4800) | 6.2 (4800) | **-1.4** | [-1.8, -1.0] |
| draft/unique_cards | 33.64 (96) | 32.11 (96) | **-1.5** | [-2.0, -1.1] |
| draft/weapon_share | 8.8 (4800) | 10.1 (4800) | **+1.3** | [+0.7, +1.9] |
| gate/Hydromancy/portal_readied_ikz | 92.3 (532) | 94.8 (501) | +2.5 | [-0.4, +5.6] |
| gate/Hydromancy/readied_then_spent_same_turn | 73.1 (491) | 72.6 (475) | -0.5 | [-6.0, +5.2] |
| gate/portal_per_legal_turn | 98.3 (541) | 99.4 (504) | +1.1 | [-0.0, +2.3] |
| heal/heal_spell_per_legal_turn_below_max_hp | 32.0 (50) | 41.9 (31) | +9.9 | [-9.4, +35.4] |
| ikz/held_at_end_of_own_turn | 47.5 (708) | 48.9 (656) | +1.5 | [-4.6, +8.0] |
| ikz/held_then_spent_in_opp_turn | 37.5 (336) | 37.1 (321) | -0.4 | [-10.9, +8.7] |
| leader/Benzai/use_then_card_played_same_turn | 13.7 (51) | 30.0 (10) | +16.3 | [-30.6, +41.9] |
| leader/Shao/target_is_current_attacker | 46.3 (123) | 47.8 (113) | +1.4 | [-10.9, +12.8] |
| leader/use_per_legal_turn | 92.6 (188) | 84.8 (145) | **-7.7** | [-16.5, -0.3] |
| outcome/own_turns | 7.38 (96) | 6.83 (96) | -0.5 | [-1.4, +0.0] |
| outcome/win | 46.9 (96) | 58.3 (96) | **+11.5** | [+1.0, +22.9] |
| outcome/win_vs_EARTH | 50.0 (24) | 58.3 (24) | +8.3 | [-22.7, +43.8] |
| outcome/win_vs_FIRE | 16.7 (24) | 33.3 (24) | +16.7 | [+0.0, +36.4] |
| outcome/win_vs_LIGHTNING | 62.5 (24) | 70.8 (24) | +8.3 | [-8.3, +25.0] |
| outcome/win_vs_WATER | 58.3 (24) | 70.8 (24) | +12.5 | [-7.1, +35.7] |
| response/any_nonblock_response_when_legal | 79.2 (168) | 82.9 (146) | +3.7 | [-16.7, +23.4] |
| response/defender_declared_when_legal | 33.8 (71) | 20.9 (43) | **-12.9** | [-23.0, -2.7] |
| response/spell_played_when_legal | 16.2 (37) | 24.0 (25) | +7.8 | [-16.8, +87.0] |
| sequence/water.healing_flutter_timing/completed_per_eligible_game | 78.9 (19) | 92.3 (13) | +13.4 | [-11.9, +40.0] |
| sequence/water.healing_flutter_timing/converted_per_eligible_game | 78.9 (19) | 92.3 (13) | +13.4 | [-11.9, +40.0] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 88.9 (45) | 78.3 (46) | -10.6 | [-26.2, +5.6] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 88.9 (45) | 78.3 (46) | -10.6 | [-26.2, +5.6] |
| strategy/attacks_by_equipped_attacker | 4.3 (1414) | 4.4 (1512) | +0.1 | [-1.3, +1.4] |
| strategy/face_target_share_when_both_legal | 53.7 (892) | 56.0 (948) | +2.3 | [-3.6, +8.3] |
| strategy/favorable_trade_taken_per_available_turn | 56.5 (271) | 64.2 (285) | **+7.8** | [+0.4, +15.3] |
| strategy/spell_cast_per_legal_main_turn | 24.5 (274) | 21.8 (193) | -2.7 | [-9.9, +4.7] |
| water/bounce_play_per_legal_turn | 54.8 (197) | 51.5 (134) | -3.3 | [-19.6, +15.0] |
| water/bounce_play_removes_opp_entity | 64.5 (110) | 45.1 (71) | -19.5 | [-40.3, +10.3] |

### water vs u8223 — sample — free_draft|Hydromancy/Benzai

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 6.2 (16) | 0.0 (7) | -6.2 | [-21.4, +0.0] |
| draft/max_wjaccard_to_curated | 13.4 (48) | 13.3 (48) | -0.1 | [-0.7, +0.4] |
| draft/mean_cost | 2.60 (2400) | 2.66 (2400) | **+0.1** | [+0.0, +0.1] |
| draft/normal_share | 76.3 (2400) | 78.2 (2400) | **+1.8** | [+1.0, +2.7] |
| draft/spell_share | 7.7 (2400) | 6.2 (2400) | **-1.4** | [-1.9, -1.0] |
| draft/unique_cards | 33.27 (48) | 31.79 (48) | **-1.5** | [-2.1, -0.8] |
| draft/weapon_share | 9.2 (2400) | 10.9 (2400) | **+1.8** | [+0.8, +2.7] |
| gate/Hydromancy/portal_readied_ikz | 92.8 (291) | 95.5 (247) | +2.8 | [-0.8, +7.1] |
| gate/Hydromancy/readied_then_spent_same_turn | 77.8 (270) | 74.2 (236) | -3.6 | [-10.6, +3.7] |
| gate/portal_per_legal_turn | 99.3 (293) | 100.0 (247) | +0.7 | [+0.0, +1.8] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.0 (24) | 33.3 (21) | +8.3 | [-25.8, +41.7] |
| ikz/held_at_end_of_own_turn | 46.9 (371) | 47.5 (335) | +0.6 | [-9.1, +11.0] |
| ikz/held_then_spent_in_opp_turn | 1.7 (174) | 3.8 (159) | +2.0 | [-1.1, +5.2] |
| leader/Benzai/use_then_card_played_same_turn | 13.7 (51) | 30.0 (10) | +16.3 | [-28.6, +42.0] |
| leader/use_per_legal_turn | 86.4 (59) | 38.5 (26) | **-48.0** | [-65.9, -11.0] |
| outcome/own_turns | 7.73 (48) | 6.98 (48) | -0.8 | [-2.4, +0.2] |
| outcome/win | 41.7 (48) | 52.1 (48) | +10.4 | [-4.2, +27.1] |
| response/any_nonblock_response_when_legal | 36.0 (25) | 100.0 (6) | +64.0 | [+0.0, +86.7] |
| response/defender_declared_when_legal | 31.4 (51) | 20.6 (34) | -10.8 | [-23.9, +1.4] |
| response/spell_played_when_legal | 15.8 (19) | 100.0 (6) | +84.2 | [+0.0, +100.0] |
| sequence/water.healing_flutter_timing/completed_per_eligible_game | 75.0 (8) | 100.0 (6) | +25.0 | [+0.0, +60.0] |
| sequence/water.healing_flutter_timing/converted_per_eligible_game | 75.0 (8) | 100.0 (6) | +25.0 | [+0.0, +60.0] |
| strategy/attacks_by_equipped_attacker | 4.1 (701) | 4.1 (763) | -0.1 | [-2.1, +1.9] |
| strategy/face_target_share_when_both_legal | 48.9 (454) | 54.6 (500) | +5.7 | [-2.3, +13.5] |
| strategy/favorable_trade_taken_per_available_turn | 56.8 (148) | 64.9 (151) | +8.1 | [-2.2, +17.9] |
| strategy/spell_cast_per_legal_main_turn | 20.1 (139) | 19.0 (121) | -1.1 | [-10.5, +7.1] |
| water/bounce_play_per_legal_turn | 60.2 (113) | 46.5 (71) | -13.7 | [-34.0, +17.3] |
| water/bounce_play_removes_opp_entity | 73.9 (69) | 52.9 (34) | -21.0 | [-44.9, +24.3] |

### water vs u8223 — sample — free_draft|Hydromancy/Shao

| metric | u8223 | water | Δ | 95% CI |
|---|---|---|---|---|
| block/favorable_survive_or_kill | 25.0 (8) | 0.0 (2) | -25.0 | [-75.0, +0.0] |
| draft/max_wjaccard_to_curated | 13.2 (48) | 13.0 (48) | -0.2 | [-0.7, +0.4] |
| draft/mean_cost | 2.55 (2400) | 2.56 (2400) | +0.0 | [-0.0, +0.0] |
| draft/normal_share | 75.3 (2400) | 77.5 (2400) | **+2.2** | [+1.2, +3.2] |
| draft/spell_share | 7.6 (2400) | 6.2 (2400) | **-1.4** | [-2.0, -0.7] |
| draft/unique_cards | 34.00 (48) | 32.44 (48) | **-1.6** | [-2.3, -0.8] |
| draft/weapon_share | 8.5 (2400) | 9.3 (2400) | **+0.8** | [+0.2, +1.5] |
| gate/Hydromancy/portal_readied_ikz | 91.7 (241) | 94.1 (254) | +2.4 | [-1.8, +6.7] |
| gate/Hydromancy/readied_then_spent_same_turn | 67.4 (221) | 71.1 (239) | +3.7 | [-2.8, +10.4] |
| gate/portal_per_legal_turn | 97.2 (248) | 98.8 (257) | +1.7 | [-0.4, +3.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 38.5 (26) | 60.0 (10) | +21.5 | [-6.5, +44.4] |
| ikz/held_at_end_of_own_turn | 48.1 (337) | 50.5 (321) | +2.4 | [-5.1, +9.9] |
| ikz/held_then_spent_in_opp_turn | 75.9 (162) | 69.8 (162) | -6.2 | [-16.1, +3.3] |
| leader/Shao/target_is_current_attacker | 46.3 (123) | 47.8 (113) | +1.4 | [-10.4, +12.6] |
| leader/use_per_legal_turn | 95.3 (129) | 95.0 (119) | -0.4 | [-5.6, +4.9] |
| outcome/own_turns | 7.02 (48) | 6.69 (48) | -0.3 | [-0.8, +0.1] |
| outcome/win | 52.1 (48) | 64.6 (48) | +12.5 | [-4.2, +31.2] |
| response/any_nonblock_response_when_legal | 86.7 (143) | 82.1 (140) | -4.6 | [-23.7, +13.1] |
| response/defender_declared_when_legal | 40.0 (20) | 22.2 (9) | -17.8 | [-42.1, +13.0] |
| response/spell_played_when_legal | 16.7 (18) | 0.0 (19) | -16.7 | [-42.9, +0.0] |
| sequence/water.healing_flutter_timing/completed_per_eligible_game | 81.8 (11) | 85.7 (7) | +3.9 | [-40.9, +40.0] |
| sequence/water.healing_flutter_timing/converted_per_eligible_game | 81.8 (11) | 85.7 (7) | +3.9 | [-40.9, +40.0] |
| sequence/water.shao_attacker_target/completed_per_eligible_game | 88.9 (45) | 78.3 (46) | -10.6 | [-27.7, +4.3] |
| sequence/water.shao_attacker_target/converted_per_eligible_game | 88.9 (45) | 78.3 (46) | -10.6 | [-27.7, +4.3] |
| strategy/attacks_by_equipped_attacker | 4.5 (713) | 4.7 (749) | +0.2 | [-1.8, +2.2] |
| strategy/face_target_share_when_both_legal | 58.7 (438) | 57.6 (448) | -1.1 | [-10.3, +8.0] |
| strategy/favorable_trade_taken_per_available_turn | 56.1 (123) | 63.4 (134) | +7.3 | [-3.2, +18.4] |
| strategy/spell_cast_per_legal_main_turn | 28.9 (135) | 26.4 (72) | -2.5 | [-15.5, +12.5] |
| water/bounce_play_per_legal_turn | 47.6 (84) | 57.1 (63) | +9.5 | [-3.0, +21.6] |
| water/bounce_play_removes_opp_entity | 48.8 (41) | 37.8 (37) | -10.9 | [-28.6, +11.7] |

## EARTH
### earth_c100 vs u8223 — argmax — all|ALL

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 29.5 (543) | 36.3 (797) | **+6.8** | [+1.9, +11.6] |
| block/blocker_survives | 41.6 (543) | 33.2 (797) | **-8.4** | [-13.8, -3.5] |
| block/favorable_survive_or_kill | 50.8 (543) | 50.6 (797) | -0.3 | [-5.3, +4.6] |
| draft/max_wjaccard_to_curated | 25.4 (96) | 15.8 (96) | **-9.6** | [-10.8, -8.2] |
| draft/mean_cost | 3.36 (4800) | 3.39 (4800) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 58.0 (4800) | 63.0 (4800) | **+5.0** | [+4.7, +5.3] |
| draft/spell_share | 15.0 (4800) | 2.0 (4800) | **-13.0** | [-13.3, -12.7] |
| draft/unique_cards | 17.00 (96) | 18.00 (96) | **+1.0** | [+1.0, +1.0] |
| draft/weapon_share | 10.0 (4800) | 0.0 (4800) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 86.9 (168) | 83.7 (123) | -3.2 | [-10.0, +3.7] |
| earth/quicksand_cast_per_legal_turn | 47.5 (354) | 37.4 (329) | **-10.1** | [-17.1, -3.1] |
| gate/Stonehaven/grant_then_block | 40.7 (1009) | 52.8 (1318) | **+12.1** | [+8.4, +15.8] |
| gate/Stonehaven/portal_then_grant | 75.2 (1342) | 83.6 (1577) | **+8.4** | [+5.5, +11.2] |
| gate/portal_per_legal_turn | 99.6 (1348) | 97.3 (1621) | **-2.3** | [-3.1, -1.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 28.9 (757) | 26.7 (673) | -2.2 | [-5.4, +1.1] |
| ikz/held_at_end_of_own_turn | 33.1 (2302) | 33.4 (2425) | +0.3 | [-1.8, +2.4] |
| ikz/held_then_spent_in_opp_turn | 25.3 (763) | 26.9 (810) | +1.6 | [-1.2, +4.3] |
| leader/Bobu/earth_loss_before_next_turn | 57.1 (28) | 100.0 (1) | **+42.9** | [+25.0, +61.9] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 33.3 (3) | +18.5 | [-20.4, +86.2] |
| leader/Goro/target_then_attacks_same_turn | 13.8 (65) | 14.3 (56) | +0.4 | [-12.5, +14.8] |
| leader/Goro/use_with_target | 100.0 (65) | 100.0 (56) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.4 (2193) | 2.5 (2345) | **-2.9** | [-4.1, -1.6] |
| outcome/own_turns | 7.99 (288) | 8.42 (288) | **+0.4** | [+0.2, +0.6] |
| outcome/win | 54.9 (288) | 66.0 (288) | **+11.1** | [+5.2, +16.7] |
| outcome/win_vs_EARTH | 52.8 (72) | 73.6 (72) | **+20.8** | [+7.3, +35.2] |
| outcome/win_vs_FIRE | 40.3 (72) | 50.0 (72) | +9.7 | [-1.4, +21.1] |
| outcome/win_vs_LIGHTNING | 61.1 (72) | 68.1 (72) | +6.9 | [-6.5, +20.0] |
| outcome/win_vs_WATER | 65.3 (72) | 72.2 (72) | +6.9 | [-3.8, +17.2] |
| response/any_nonblock_response_when_legal | 61.2 (454) | 62.5 (648) | +1.3 | [-6.6, +8.4] |
| response/defender_declared_when_legal | 55.2 (982) | 46.1 (1726) | **-9.1** | [-13.7, -4.0] |
| response/spell_played_when_legal | 70.5 (281) | 57.4 (390) | **-13.0** | [-22.5, -3.3] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.0 (176) | 0.6 (176) | **-7.4** | [-11.5, -3.7] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.7 (176) | 0.6 (176) | **-5.1** | [-8.9, -1.6] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.7 (264) | 78.9 (279) | **+12.2** | [+5.3, +18.7] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 41.7 (264) | 48.7 (279) | +7.1 | [-1.1, +15.2] |
| strategy/attacks_by_equipped_attacker | 3.8 (2829) | 1.3 (3041) | **-2.5** | [-3.4, -1.7] |
| strategy/face_target_share_when_both_legal | 50.8 (1532) | 40.6 (1597) | **-10.3** | [-15.3, -5.3] |
| strategy/favorable_trade_taken_per_available_turn | 55.1 (670) | 65.9 (754) | **+10.8** | [+5.4, +16.0] |
| strategy/spell_cast_per_legal_main_turn | 28.3 (1453) | 24.6 (1263) | **-3.7** | [-6.0, -1.3] |

### earth_c100 vs u8223 — argmax — all|Stonehaven/Bobu

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 23.5 (357) | 29.7 (548) | **+6.2** | [+0.4, +12.0] |
| block/blocker_survives | 37.3 (357) | 28.6 (548) | **-8.6** | [-15.6, -2.2] |
| block/favorable_survive_or_kill | 45.7 (357) | 45.3 (548) | -0.4 | [-6.1, +5.0] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 13.4 (48) | **-4.9** | [-4.9, -4.9] |
| draft/mean_cost | 3.36 (2400) | 3.42 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (2400) | 62.0 (2400) | **+4.0** | [+4.0, +4.0] |
| draft/spell_share | 16.0 (2400) | 2.0 (2400) | **-14.0** | [-14.0, -14.0] |
| draft/unique_cards | 17.00 (48) | 18.00 (48) | **+1.0** | [+1.0, +1.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 89.2 (102) | 82.4 (91) | -6.8 | [-14.9, +2.2] |
| earth/quicksand_cast_per_legal_turn | 47.0 (217) | 39.7 (229) | -7.3 | [-15.6, +1.2] |
| gate/Stonehaven/grant_then_block | 38.7 (646) | 52.9 (845) | **+14.2** | [+9.5, +18.7] |
| gate/Stonehaven/portal_then_grant | 73.0 (885) | 81.2 (1040) | **+8.3** | [+4.3, +12.3] |
| gate/portal_per_legal_turn | 99.3 (891) | 97.1 (1071) | **-2.2** | [-3.4, -1.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 29.6 (494) | 26.7 (490) | -2.8 | [-6.7, +1.2] |
| ikz/held_at_end_of_own_turn | 36.5 (1470) | 37.3 (1557) | +0.7 | [-2.0, +3.3] |
| ikz/held_then_spent_in_opp_turn | 27.6 (537) | 28.8 (580) | +1.2 | [-2.3, +4.9] |
| leader/Bobu/earth_loss_before_next_turn | 57.1 (28) | 100.0 (1) | **+42.9** | [+25.0, +62.1] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 33.3 (3) | +18.5 | [-20.8, +87.0] |
| leader/use_per_legal_turn | 3.7 (1470) | 0.2 (1557) | **-3.5** | [-4.7, -2.4] |
| outcome/own_turns | 8.35 (176) | 8.85 (176) | **+0.5** | [+0.3, +0.7] |
| outcome/win | 56.8 (176) | 67.0 (176) | **+10.2** | [+2.8, +17.0] |
| response/any_nonblock_response_when_legal | 58.5 (342) | 60.8 (475) | +2.4 | [-7.0, +10.7] |
| response/defender_declared_when_legal | 50.4 (707) | 45.0 (1215) | -5.3 | [-10.9, +0.1] |
| response/spell_played_when_legal | 73.9 (203) | 57.9 (299) | **-16.0** | [-26.9, -4.7] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.0 (176) | 0.6 (176) | **-7.4** | [-11.9, -3.4] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.7 (176) | 0.6 (176) | **-5.1** | [-9.7, -1.1] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 64.5 (166) | 78.5 (172) | **+14.0** | [+6.1, +22.3] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 38.6 (166) | 48.8 (172) | **+10.3** | [+1.1, +19.7] |
| strategy/attacks_by_equipped_attacker | 2.4 (1659) | 0.0 (1712) | **-2.4** | [-3.3, -1.5] |
| strategy/face_target_share_when_both_legal | 44.0 (877) | 36.3 (871) | **-7.7** | [-14.2, -0.7] |
| strategy/favorable_trade_taken_per_available_turn | 59.9 (377) | 72.0 (425) | **+12.1** | [+4.0, +19.1] |
| strategy/spell_cast_per_legal_main_turn | 27.3 (974) | 25.0 (911) | -2.3 | [-4.8, +0.3] |

### earth_c100 vs u8223 — argmax — all|Stonehaven/Goro

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 40.9 (186) | 50.6 (249) | **+9.7** | [+1.3, +18.5] |
| block/blocker_survives | 50.0 (186) | 43.4 (249) | -6.6 | [-16.4, +3.2] |
| block/favorable_survive_or_kill | 60.8 (186) | 62.2 (249) | +1.5 | [-7.9, +10.6] |
| draft/max_wjaccard_to_curated | 32.5 (48) | 18.3 (48) | **-14.3** | [-14.3, -14.3] |
| draft/mean_cost | 3.36 (2400) | 3.36 (2400) | +0.0 | [+0.0, +0.0] |
| draft/normal_share | 58.0 (2400) | 64.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 14.0 (2400) | 2.0 (2400) | **-12.0** | [-12.0, -12.0] |
| draft/unique_cards | 17.00 (48) | 18.00 (48) | **+1.0** | [+1.0, +1.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 83.3 (66) | 87.5 (32) | +4.2 | [-5.7, +12.7] |
| earth/quicksand_cast_per_legal_turn | 48.2 (137) | 32.0 (100) | **-16.2** | [-29.0, -4.1] |
| gate/Stonehaven/grant_then_block | 44.4 (363) | 52.6 (473) | **+8.3** | [+2.3, +14.5] |
| gate/Stonehaven/portal_then_grant | 79.4 (457) | 88.1 (537) | **+8.7** | [+4.1, +13.2] |
| gate/portal_per_legal_turn | 100.0 (457) | 97.6 (550) | **-2.4** | [-3.7, -1.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.8 (263) | 26.8 (183) | -1.0 | [-6.8, +5.4] |
| ikz/held_at_end_of_own_turn | 27.2 (832) | 26.5 (868) | -0.7 | [-4.2, +3.0] |
| ikz/held_then_spent_in_opp_turn | 19.9 (226) | 22.2 (230) | +2.3 | [-1.8, +6.2] |
| leader/Goro/target_then_attacks_same_turn | 13.8 (65) | 14.3 (56) | +0.4 | [-11.8, +14.2] |
| leader/Goro/use_with_target | 100.0 (65) | 100.0 (56) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.0 (723) | 7.1 (788) | -1.9 | [-4.8, +1.1] |
| outcome/own_turns | 7.43 (112) | 7.75 (112) | **+0.3** | [+0.0, +0.6] |
| outcome/win | 51.8 (112) | 64.3 (112) | **+12.5** | [+1.8, +23.2] |
| response/any_nonblock_response_when_legal | 69.6 (112) | 67.1 (173) | -2.6 | [-16.8, +13.4] |
| response/defender_declared_when_legal | 67.6 (275) | 48.7 (511) | **-18.9** | [-26.9, -10.0] |
| response/spell_played_when_legal | 61.5 (78) | 56.0 (91) | -5.5 | [-23.8, +13.4] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 70.4 (98) | 79.4 (107) | +9.0 | [-2.5, +20.2] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 46.9 (98) | 48.6 (107) | +1.7 | [-11.4, +15.6] |
| strategy/attacks_by_equipped_attacker | 5.9 (1170) | 2.9 (1329) | **-3.0** | [-4.6, -1.2] |
| strategy/face_target_share_when_both_legal | 60.0 (655) | 45.7 (726) | **-14.3** | [-21.9, -6.9] |
| strategy/favorable_trade_taken_per_available_turn | 48.8 (293) | 58.1 (329) | **+9.2** | [+1.1, +17.0] |
| strategy/spell_cast_per_legal_main_turn | 30.3 (479) | 23.6 (352) | **-6.7** | [-11.9, -1.7] |

### earth_c100 vs u8223 — argmax — fixed|ALL

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 25.9 (344) | 31.2 (555) | +5.3 | [-0.7, +11.3] |
| block/blocker_survives | 40.1 (344) | 33.5 (555) | **-6.6** | [-12.8, -0.3] |
| block/favorable_survive_or_kill | 49.1 (344) | 48.1 (555) | -1.0 | [-6.6, +4.6] |
| earth/quicksand_cast_multi_removal | 85.8 (113) | 85.2 (108) | -0.7 | [-8.2, +7.5] |
| earth/quicksand_cast_per_legal_turn | 52.8 (214) | 39.1 (276) | **-13.7** | [-22.3, -5.6] |
| gate/Stonehaven/grant_then_block | 40.6 (645) | 55.2 (822) | **+14.6** | [+10.3, +19.1] |
| gate/Stonehaven/portal_then_grant | 69.4 (929) | 81.3 (1011) | **+11.9** | [+8.7, +15.3] |
| gate/portal_per_legal_turn | 99.5 (934) | 97.8 (1034) | **-1.7** | [-2.8, -0.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 32.2 (544) | 26.7 (673) | **-5.4** | [-8.9, -2.1] |
| ikz/held_at_end_of_own_turn | 36.1 (1557) | 40.2 (1650) | **+4.1** | [+1.8, +6.5] |
| ikz/held_then_spent_in_opp_turn | 34.3 (562) | 32.9 (663) | -1.5 | [-4.8, +2.2] |
| leader/Bobu/earth_loss_before_next_turn | 56.5 (23) | 100.0 (1) | **+43.5** | [+22.7, +63.0] |
| leader/Bobu/use_then_heal_observed | 13.2 (38) | 100.0 (1) | **+86.8** | [+79.1, +96.4] |
| leader/Goro/target_then_attacks_same_turn | 18.9 (37) | 11.5 (52) | -7.4 | [-25.2, +9.7] |
| leader/Goro/use_with_target | 100.0 (37) | 100.0 (52) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.0 (1509) | 3.3 (1602) | **-1.7** | [-3.2, -0.1] |
| outcome/own_turns | 8.11 (192) | 8.59 (192) | **+0.5** | [+0.3, +0.7] |
| outcome/win | 53.6 (192) | 63.0 (192) | **+9.4** | [+3.1, +15.6] |
| outcome/win_vs_EARTH | 54.2 (48) | 66.7 (48) | +12.5 | [+0.0, +25.9] |
| outcome/win_vs_FIRE | 35.4 (48) | 52.1 (48) | **+16.7** | [+5.0, +29.2] |
| outcome/win_vs_LIGHTNING | 62.5 (48) | 64.6 (48) | +2.1 | [-12.5, +15.4] |
| outcome/win_vs_WATER | 62.5 (48) | 68.8 (48) | +6.2 | [-5.9, +18.2] |
| response/any_nonblock_response_when_legal | 61.2 (454) | 62.5 (648) | +1.3 | [-6.7, +8.1] |
| response/defender_declared_when_legal | 47.3 (725) | 43.4 (1276) | -3.9 | [-9.7, +1.5] |
| response/spell_played_when_legal | 70.5 (281) | 57.4 (390) | **-13.0** | [-23.4, -3.3] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.6 (128) | 0.8 (128) | **-7.8** | [-13.2, -3.2] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.5 (128) | 0.8 (128) | -4.7 | [-9.8, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.2 (176) | 71.7 (184) | **+11.5** | [+2.5, +20.2] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 34.7 (176) | 42.9 (184) | +8.3 | [-0.9, +17.3] |
| strategy/attacks_by_equipped_attacker | 2.0 (1710) | 2.4 (1655) | +0.4 | [-0.2, +1.0] |
| strategy/face_target_share_when_both_legal | 49.8 (941) | 48.9 (876) | -1.0 | [-5.6, +4.0] |
| strategy/favorable_trade_taken_per_available_turn | 54.9 (428) | 59.4 (451) | +4.5 | [-2.3, +11.1] |
| strategy/spell_cast_per_legal_main_turn | 27.6 (1113) | 24.5 (1210) | **-3.1** | [-5.6, -0.7] |

### earth_c100 vs u8223 — argmax — fixed|Stonehaven/Bobu

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 18.7 (262) | 25.2 (425) | +6.5 | [-0.4, +12.9] |
| block/blocker_survives | 34.4 (262) | 28.7 (425) | -5.6 | [-13.1, +1.7] |
| block/favorable_survive_or_kill | 43.5 (262) | 42.8 (425) | -0.7 | [-7.2, +5.9] |
| earth/quicksand_cast_multi_removal | 88.6 (79) | 84.5 (84) | -4.1 | [-12.8, +5.6] |
| earth/quicksand_cast_per_legal_turn | 50.6 (156) | 39.8 (211) | **-10.8** | [-20.5, -1.2] |
| gate/Stonehaven/grant_then_block | 39.0 (462) | 53.9 (601) | **+14.9** | [+9.8, +20.0] |
| gate/Stonehaven/portal_then_grant | 67.7 (682) | 78.9 (762) | **+11.1** | [+7.1, +15.5] |
| gate/portal_per_legal_turn | 99.3 (687) | 97.1 (785) | **-2.2** | [-3.6, -0.9] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.3 (387) | 26.7 (490) | **-6.6** | [-10.6, -2.6] |
| ikz/held_at_end_of_own_turn | 39.1 (1103) | 43.3 (1167) | **+4.2** | [+1.5, +6.7] |
| ikz/held_then_spent_in_opp_turn | 34.3 (431) | 33.1 (505) | -1.3 | [-5.2, +2.9] |
| leader/Bobu/earth_loss_before_next_turn | 56.5 (23) | 100.0 (1) | **+43.5** | [+22.7, +63.6] |
| leader/Bobu/use_then_heal_observed | 13.2 (38) | 100.0 (1) | **+86.8** | [+79.4, +96.0] |
| leader/use_per_legal_turn | 3.4 (1103) | 0.1 (1167) | **-3.4** | [-4.7, -2.1] |
| outcome/own_turns | 8.62 (128) | 9.12 (128) | **+0.5** | [+0.2, +0.8] |
| outcome/win | 55.5 (128) | 65.6 (128) | **+10.2** | [+2.3, +18.0] |
| response/any_nonblock_response_when_legal | 58.5 (342) | 60.8 (475) | +2.4 | [-6.7, +10.4] |
| response/defender_declared_when_legal | 44.3 (589) | 42.5 (997) | -1.8 | [-7.4, +4.2] |
| response/spell_played_when_legal | 73.9 (203) | 57.9 (299) | **-16.0** | [-26.9, -5.0] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.6 (128) | 0.8 (128) | **-7.8** | [-13.3, -3.1] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.5 (128) | 0.8 (128) | -4.7 | [-10.2, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 57.4 (122) | 72.6 (124) | **+15.2** | [+6.0, +24.7] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 31.1 (122) | 42.7 (124) | **+11.6** | [+1.1, +22.4] |
| strategy/attacks_by_equipped_attacker | 0.0 (1081) | 0.0 (1028) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 42.2 (573) | 42.3 (525) | +0.1 | [-6.1, +6.4] |
| strategy/favorable_trade_taken_per_available_turn | 59.4 (261) | 68.7 (281) | **+9.3** | [+1.4, +17.3] |
| strategy/spell_cast_per_legal_main_turn | 27.5 (810) | 24.7 (893) | **-2.8** | [-5.3, -0.3] |

### earth_c100 vs u8223 — argmax — fixed|Stonehaven/Goro

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 48.8 (82) | 50.8 (130) | +2.0 | [-8.5, +13.4] |
| block/blocker_survives | 58.5 (82) | 49.2 (130) | -9.3 | [-21.2, +3.6] |
| block/favorable_survive_or_kill | 67.1 (82) | 65.4 (130) | -1.7 | [-13.4, +10.5] |
| earth/quicksand_cast_multi_removal | 79.4 (34) | 87.5 (24) | +8.1 | [-3.8, +19.3] |
| earth/quicksand_cast_per_legal_turn | 58.6 (58) | 36.9 (65) | **-21.7** | [-41.1, -4.6] |
| gate/Stonehaven/grant_then_block | 44.8 (183) | 58.8 (221) | **+14.0** | [+6.0, +21.6] |
| gate/Stonehaven/portal_then_grant | 74.1 (247) | 88.8 (249) | **+14.7** | [+10.3, +19.5] |
| gate/portal_per_legal_turn | 100.0 (247) | 100.0 (249) | +0.0 | [+0.0, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 29.3 (157) | 26.8 (183) | -2.5 | [-9.0, +3.0] |
| ikz/held_at_end_of_own_turn | 28.9 (454) | 32.7 (483) | +3.9 | [-1.0, +9.1] |
| ikz/held_then_spent_in_opp_turn | 34.4 (131) | 32.3 (158) | -2.1 | [-9.4, +4.9] |
| leader/Goro/target_then_attacks_same_turn | 18.9 (37) | 11.5 (52) | -7.4 | [-24.2, +8.5] |
| leader/Goro/use_with_target | 100.0 (37) | 100.0 (52) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.1 (406) | 12.0 (435) | +2.8 | [-1.0, +6.8] |
| outcome/own_turns | 7.09 (64) | 7.55 (64) | **+0.5** | [+0.2, +0.8] |
| outcome/win | 50.0 (64) | 57.8 (64) | +7.8 | [-3.1, +17.2] |
| response/any_nonblock_response_when_legal | 69.6 (112) | 67.1 (173) | -2.6 | [-16.6, +13.0] |
| response/defender_declared_when_legal | 60.3 (136) | 46.6 (279) | **-13.7** | [-26.1, -1.1] |
| response/spell_played_when_legal | 61.5 (78) | 56.0 (91) | -5.5 | [-23.7, +13.1] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.7 (54) | 70.0 (60) | +3.3 | [-14.8, +21.0] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 42.6 (54) | 43.3 (60) | +0.7 | [-17.8, +17.3] |
| strategy/attacks_by_equipped_attacker | 5.4 (629) | 6.2 (627) | +0.8 | [-0.8, +2.6] |
| strategy/face_target_share_when_both_legal | 61.7 (368) | 58.7 (351) | -3.0 | [-9.2, +3.9] |
| strategy/favorable_trade_taken_per_available_turn | 47.9 (167) | 44.1 (170) | -3.8 | [-15.2, +6.3] |
| strategy/spell_cast_per_legal_main_turn | 27.7 (303) | 23.7 (317) | -4.1 | [-9.6, +1.5] |

### earth_c100 vs u8223 — argmax — free_draft|ALL

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 35.7 (199) | 47.9 (242) | **+12.3** | [+3.6, +21.0] |
| block/blocker_survives | 44.2 (199) | 32.6 (242) | **-11.6** | [-21.4, -2.3] |
| block/favorable_survive_or_kill | 53.8 (199) | 56.2 (242) | +2.4 | [-6.8, +11.6] |
| draft/max_wjaccard_to_curated | 25.4 (96) | 15.8 (96) | **-9.6** | [-10.9, -8.2] |
| draft/mean_cost | 3.36 (4800) | 3.39 (4800) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 58.0 (4800) | 63.0 (4800) | **+5.0** | [+4.7, +5.3] |
| draft/spell_share | 15.0 (4800) | 2.0 (4800) | **-13.0** | [-13.3, -12.7] |
| draft/unique_cards | 17.00 (96) | 18.00 (96) | **+1.0** | [+1.0, +1.0] |
| draft/weapon_share | 10.0 (4800) | 0.0 (4800) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 89.1 (55) | 73.3 (15) | -15.8 | [-39.7, +4.6] |
| earth/quicksand_cast_per_legal_turn | 39.3 (140) | 28.3 (53) | -11.0 | [-24.8, +4.9] |
| gate/Stonehaven/grant_then_block | 40.9 (364) | 48.8 (496) | **+7.9** | [+1.3, +14.2] |
| gate/Stonehaven/portal_then_grant | 88.1 (413) | 87.6 (566) | -0.5 | [-5.4, +4.1] |
| gate/portal_per_legal_turn | 99.8 (414) | 96.4 (587) | **-3.3** | [-4.8, -1.9] |
| heal/heal_spell_per_legal_turn_below_max_hp | 20.7 (213) | – (0) | – | |
| ikz/held_at_end_of_own_turn | 27.0 (745) | 19.0 (775) | **-8.0** | [-10.9, -4.8] |
| ikz/held_then_spent_in_opp_turn | 0.0 (201) | 0.0 (147) | +0.0 | [+0.0, +0.0] |
| leader/Bobu/use_then_heal_observed | 18.8 (16) | 0.0 (2) | -18.8 | [-37.5, +0.0] |
| leader/Goro/target_then_attacks_same_turn | 7.1 (28) | 50.0 (4) | +42.9 | [-6.1, +95.2] |
| leader/Goro/use_with_target | 100.0 (28) | 100.0 (4) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 6.4 (684) | 0.8 (743) | **-5.6** | [-7.8, -3.5] |
| outcome/own_turns | 7.76 (96) | 8.07 (96) | +0.3 | [-0.1, +0.7] |
| outcome/win | 57.3 (96) | 71.9 (96) | **+14.6** | [+2.1, +28.1] |
| outcome/win_vs_EARTH | 50.0 (24) | 87.5 (24) | **+37.5** | [+5.0, +66.7] |
| outcome/win_vs_FIRE | 50.0 (24) | 45.8 (24) | -4.2 | [-23.1, +15.4] |
| outcome/win_vs_LIGHTNING | 58.3 (24) | 75.0 (24) | +16.7 | [-11.1, +45.5] |
| outcome/win_vs_WATER | 70.8 (24) | 79.2 (24) | +8.3 | [-11.5, +30.0] |
| response/defender_declared_when_legal | 77.4 (257) | 53.8 (450) | **-23.7** | [-31.6, -15.5] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-13.9, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-13.9, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 79.5 (88) | 92.6 (95) | **+13.1** | [+3.2, +23.0] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 55.7 (88) | 60.0 (95) | +4.3 | [-10.3, +18.5] |
| strategy/attacks_by_equipped_attacker | 6.6 (1119) | 0.0 (1386) | **-6.6** | [-8.0, -5.3] |
| strategy/face_target_share_when_both_legal | 52.5 (591) | 30.5 (721) | **-21.9** | [-30.3, -12.7] |
| strategy/favorable_trade_taken_per_available_turn | 55.4 (242) | 75.6 (303) | **+20.2** | [+11.8, +28.7] |
| strategy/spell_cast_per_legal_main_turn | 30.6 (340) | 28.3 (53) | -2.3 | [-14.9, +15.7] |

### earth_c100 vs u8223 — argmax — free_draft|Stonehaven/Bobu

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 36.8 (95) | 45.5 (123) | +8.7 | [-3.9, +20.4] |
| block/blocker_survives | 45.3 (95) | 28.5 (123) | **-16.8** | [-30.7, -5.1] |
| block/favorable_survive_or_kill | 51.6 (95) | 53.7 (123) | +2.1 | [-9.7, +13.5] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 13.4 (48) | **-4.9** | [-4.9, -4.9] |
| draft/mean_cost | 3.36 (2400) | 3.42 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (2400) | 62.0 (2400) | **+4.0** | [+4.0, +4.0] |
| draft/spell_share | 16.0 (2400) | 2.0 (2400) | **-14.0** | [-14.0, -14.0] |
| draft/unique_cards | 17.00 (48) | 18.00 (48) | **+1.0** | [+1.0, +1.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 91.3 (23) | 57.1 (7) | -34.2 | [-73.0, +4.8] |
| earth/quicksand_cast_per_legal_turn | 37.7 (61) | 38.9 (18) | +1.2 | [-19.0, +36.9] |
| gate/Stonehaven/grant_then_block | 38.0 (184) | 50.4 (244) | **+12.4** | [+2.8, +21.8] |
| gate/Stonehaven/portal_then_grant | 90.6 (203) | 87.8 (278) | -2.9 | [-10.3, +4.4] |
| gate/portal_per_legal_turn | 99.5 (204) | 97.2 (286) | **-2.3** | [-4.5, -0.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 15.9 (107) | – (0) | – | |
| ikz/held_at_end_of_own_turn | 28.9 (367) | 19.2 (390) | **-9.7** | [-14.4, -4.9] |
| ikz/held_then_spent_in_opp_turn | 0.0 (106) | 0.0 (75) | +0.0 | [+0.0, +0.0] |
| leader/Bobu/use_then_heal_observed | 18.8 (16) | 0.0 (2) | -18.8 | [-36.4, +0.0] |
| leader/use_per_legal_turn | 4.4 (367) | 0.5 (390) | **-3.8** | [-6.3, -1.7] |
| outcome/own_turns | 7.65 (48) | 8.12 (48) | +0.5 | [+0.0, +0.9] |
| outcome/win | 60.4 (48) | 70.8 (48) | +10.4 | [-6.2, +25.0] |
| response/defender_declared_when_legal | 80.5 (118) | 56.4 (218) | **-24.1** | [-35.9, -12.9] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-12.5, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-12.5, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 84.1 (44) | 93.8 (48) | +9.7 | [-4.0, +24.7] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 59.1 (44) | 64.6 (48) | +5.5 | [-12.0, +24.0] |
| strategy/attacks_by_equipped_attacker | 6.7 (578) | 0.0 (684) | **-6.7** | [-8.4, -5.4] |
| strategy/face_target_share_when_both_legal | 47.4 (304) | 27.2 (346) | **-20.2** | [-33.0, -6.9] |
| strategy/favorable_trade_taken_per_available_turn | 61.2 (116) | 78.5 (144) | **+17.3** | [+1.7, +32.0] |
| strategy/spell_cast_per_legal_main_turn | 26.2 (164) | 38.9 (18) | +12.7 | [-4.4, +50.0] |

### earth_c100 vs u8223 — argmax — free_draft|Stonehaven/Goro

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 34.6 (104) | 50.4 (119) | **+15.8** | [+2.7, +27.8] |
| block/blocker_survives | 43.3 (104) | 37.0 (119) | -6.3 | [-20.2, +7.3] |
| block/favorable_survive_or_kill | 55.8 (104) | 58.8 (119) | +3.1 | [-11.6, +17.1] |
| draft/max_wjaccard_to_curated | 32.5 (48) | 18.3 (48) | **-14.3** | [-14.3, -14.3] |
| draft/mean_cost | 3.36 (2400) | 3.36 (2400) | +0.0 | [+0.0, +0.0] |
| draft/normal_share | 58.0 (2400) | 64.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 14.0 (2400) | 2.0 (2400) | **-12.0** | [-12.0, -12.0] |
| draft/unique_cards | 17.00 (48) | 18.00 (48) | **+1.0** | [+1.0, +1.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 87.5 (32) | 87.5 (8) | +0.0 | [-24.1, +17.1] |
| earth/quicksand_cast_per_legal_turn | 40.5 (79) | 22.9 (35) | -17.6 | [-36.5, +2.2] |
| gate/Stonehaven/grant_then_block | 43.9 (180) | 47.2 (252) | +3.3 | [-5.2, +11.7] |
| gate/Stonehaven/portal_then_grant | 85.7 (210) | 87.5 (288) | +1.8 | [-3.9, +8.1] |
| gate/portal_per_legal_turn | 100.0 (210) | 95.7 (301) | **-4.3** | [-6.2, -2.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.5 (106) | – (0) | – | |
| ikz/held_at_end_of_own_turn | 25.1 (378) | 18.7 (385) | **-6.4** | [-10.1, -2.6] |
| ikz/held_then_spent_in_opp_turn | 0.0 (95) | 0.0 (72) | +0.0 | [+0.0, +0.0] |
| leader/Goro/target_then_attacks_same_turn | 7.1 (28) | 50.0 (4) | +42.9 | [-6.2, +96.0] |
| leader/Goro/use_with_target | 100.0 (28) | 100.0 (4) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.8 (317) | 1.1 (353) | **-7.7** | [-11.2, -4.3] |
| outcome/own_turns | 7.88 (48) | 8.02 (48) | +0.1 | [-0.5, +0.7] |
| outcome/win | 54.2 (48) | 72.9 (48) | +18.8 | [+0.0, +39.6] |
| response/defender_declared_when_legal | 74.8 (139) | 51.3 (232) | **-23.5** | [-34.6, -12.8] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 75.0 (44) | 91.5 (47) | **+16.5** | [+1.8, +28.9] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 52.3 (44) | 55.3 (47) | +3.0 | [-20.5, +24.8] |
| strategy/attacks_by_equipped_attacker | 6.5 (541) | 0.0 (702) | **-6.5** | [-9.0, -4.4] |
| strategy/face_target_share_when_both_legal | 57.8 (287) | 33.6 (375) | **-24.2** | [-35.7, -11.9] |
| strategy/favorable_trade_taken_per_available_turn | 50.0 (126) | 73.0 (159) | **+23.0** | [+14.1, +31.1] |
| strategy/spell_cast_per_legal_main_turn | 34.7 (176) | 22.9 (35) | -11.8 | [-27.3, +12.1] |

### earth_c100 vs u8223 — sample — all|ALL

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 31.2 (475) | 30.7 (776) | -0.5 | [-6.6, +5.1] |
| block/blocker_survives | 39.2 (475) | 29.8 (776) | **-9.4** | [-14.4, -4.4] |
| block/favorable_survive_or_kill | 50.7 (475) | 46.0 (776) | -4.7 | [-10.6, +0.8] |
| draft/max_wjaccard_to_curated | 12.2 (96) | 13.0 (96) | +0.9 | [-0.2, +2.0] |
| draft/mean_cost | 2.98 (4800) | 3.26 (4800) | **+0.3** | [+0.2, +0.3] |
| draft/normal_share | 77.0 (4800) | 64.7 (4800) | **-12.3** | [-13.7, -10.9] |
| draft/spell_share | 9.3 (4800) | 8.7 (4800) | -0.5 | [-1.2, +0.2] |
| draft/unique_cards | 33.14 (96) | 34.79 (96) | **+1.7** | [+1.1, +2.2] |
| draft/weapon_share | 10.5 (4800) | 2.8 (4800) | **-7.6** | [-8.3, -7.0] |
| earth/quicksand_cast_multi_removal | 82.8 (134) | 82.7 (133) | -0.1 | [-7.3, +7.4] |
| earth/quicksand_cast_per_legal_turn | 53.6 (250) | 40.4 (329) | **-13.2** | [-21.0, -5.1] |
| gate/Stonehaven/grant_then_block | 38.9 (906) | 53.1 (1210) | **+14.3** | [+10.3, +18.6] |
| gate/Stonehaven/portal_then_grant | 69.9 (1297) | 81.5 (1485) | **+11.6** | [+8.4, +14.9] |
| gate/portal_per_legal_turn | 99.2 (1308) | 96.9 (1531) | **-2.2** | [-3.2, -1.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 31.7 (625) | 27.7 (793) | **-3.9** | [-7.5, -0.4] |
| ikz/held_at_end_of_own_turn | 30.7 (2263) | 35.9 (2423) | **+5.2** | [+3.1, +7.4] |
| ikz/held_then_spent_in_opp_turn | 28.6 (695) | 25.6 (871) | **-3.0** | [-5.8, -0.3] |
| leader/Bobu/earth_loss_before_next_turn | 47.4 (19) | 66.7 (6) | +19.3 | [-21.4, +66.7] |
| leader/Bobu/use_then_heal_observed | 12.2 (41) | 28.6 (7) | +16.4 | [-14.9, +87.5] |
| leader/Goro/target_then_attacks_same_turn | 15.6 (64) | 6.8 (73) | -8.8 | [-20.5, +1.5] |
| leader/Goro/use_with_target | 100.0 (64) | 100.0 (73) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 4.8 (2165) | 3.4 (2337) | **-1.4** | [-2.6, -0.3] |
| outcome/own_turns | 7.86 (288) | 8.41 (288) | **+0.6** | [+0.3, +0.8] |
| outcome/win | 48.6 (288) | 58.7 (288) | **+10.1** | [+4.2, +16.0] |
| outcome/win_vs_EARTH | 48.6 (72) | 61.1 (72) | **+12.5** | [+1.3, +25.5] |
| outcome/win_vs_FIRE | 26.4 (72) | 40.3 (72) | **+13.9** | [+1.5, +26.8] |
| outcome/win_vs_LIGHTNING | 54.2 (72) | 63.9 (72) | +9.7 | [-1.3, +20.3] |
| outcome/win_vs_WATER | 65.3 (72) | 69.4 (72) | +4.2 | [-5.6, +15.0] |
| response/any_nonblock_response_when_legal | 58.6 (503) | 62.1 (663) | +3.5 | [-3.1, +10.1] |
| response/defender_declared_when_legal | 50.6 (937) | 45.1 (1719) | **-5.4** | [-10.3, -0.6] |
| response/spell_played_when_legal | 64.4 (317) | 58.9 (389) | -5.5 | [-14.6, +3.3] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.0 (176) | 2.3 (176) | -1.7 | [-4.9, +1.2] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (176) | 1.1 (176) | -1.1 | [-3.5, +0.7] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 62.8 (253) | 68.1 (276) | +5.3 | [-2.3, +12.8] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.7 (253) | 40.6 (276) | -0.1 | [-7.4, +7.6] |
| strategy/attacks_by_equipped_attacker | 3.5 (2864) | 1.6 (2801) | **-1.9** | [-2.7, -1.1] |
| strategy/face_target_share_when_both_legal | 50.1 (1665) | 39.8 (1593) | **-10.3** | [-15.4, -5.4] |
| strategy/favorable_trade_taken_per_available_turn | 55.0 (696) | 67.2 (723) | **+12.2** | [+7.2, +17.6] |
| strategy/spell_cast_per_legal_main_turn | 26.7 (1338) | 24.3 (1517) | **-2.4** | [-5.0, -0.0] |

### earth_c100 vs u8223 — sample — all|Stonehaven/Bobu

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 26.9 (334) | 26.4 (534) | -0.5 | [-7.5, +6.0] |
| block/blocker_survives | 34.1 (334) | 27.3 (534) | **-6.8** | [-12.3, -1.2] |
| block/favorable_survive_or_kill | 46.4 (334) | 42.7 (534) | -3.7 | [-10.3, +2.5] |
| draft/max_wjaccard_to_curated | 9.0 (48) | 12.7 (48) | **+3.8** | [+2.6, +5.0] |
| draft/mean_cost | 2.99 (2400) | 3.25 (2400) | **+0.3** | [+0.2, +0.3] |
| draft/normal_share | 76.5 (2400) | 64.3 (2400) | **-12.2** | [-14.2, -10.1] |
| draft/spell_share | 8.9 (2400) | 8.3 (2400) | -0.6 | [-1.8, +0.5] |
| draft/unique_cards | 32.90 (48) | 34.65 (48) | **+1.8** | [+1.1, +2.3] |
| draft/weapon_share | 10.5 (2400) | 2.7 (2400) | **-7.8** | [-8.7, -6.9] |
| earth/quicksand_cast_multi_removal | 86.7 (90) | 84.9 (93) | -1.7 | [-9.5, +5.7] |
| earth/quicksand_cast_per_legal_turn | 51.4 (175) | 38.1 (244) | **-13.3** | [-22.1, -4.1] |
| gate/Stonehaven/grant_then_block | 39.4 (587) | 51.3 (807) | **+11.9** | [+6.4, +17.0] |
| gate/Stonehaven/portal_then_grant | 69.1 (850) | 80.2 (1006) | **+11.2** | [+7.0, +15.5] |
| gate/portal_per_legal_turn | 98.8 (860) | 96.3 (1045) | **-2.6** | [-3.9, -1.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.8 (417) | 28.8 (542) | **-5.0** | [-9.4, -0.9] |
| ikz/held_at_end_of_own_turn | 34.4 (1448) | 39.6 (1568) | **+5.2** | [+2.7, +7.8] |
| ikz/held_then_spent_in_opp_turn | 31.5 (498) | 26.7 (621) | **-4.8** | [-7.9, -1.4] |
| leader/Bobu/earth_loss_before_next_turn | 47.4 (19) | 66.7 (6) | +19.3 | [-23.2, +66.7] |
| leader/Bobu/use_then_heal_observed | 12.2 (41) | 28.6 (7) | +16.4 | [-15.4, +87.5] |
| leader/use_per_legal_turn | 2.8 (1448) | 0.4 (1568) | **-2.4** | [-3.4, -1.5] |
| outcome/own_turns | 8.23 (176) | 8.91 (176) | **+0.7** | [+0.4, +1.0] |
| outcome/win | 49.4 (176) | 60.8 (176) | **+11.4** | [+4.5, +19.3] |
| response/any_nonblock_response_when_legal | 56.8 (377) | 59.5 (482) | +2.8 | [-4.6, +10.3] |
| response/defender_declared_when_legal | 47.8 (696) | 43.2 (1236) | -4.6 | [-10.7, +1.1] |
| response/spell_played_when_legal | 68.1 (235) | 59.1 (291) | -9.0 | [-18.7, +1.1] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.0 (176) | 2.3 (176) | -1.7 | [-4.5, +1.1] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (176) | 1.1 (176) | -1.1 | [-3.4, +1.1] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 61.0 (159) | 67.4 (172) | +6.4 | [-2.9, +15.2] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.9 (159) | 37.2 (172) | -3.7 | [-13.7, +6.5] |
| strategy/attacks_by_equipped_attacker | 1.9 (1656) | 0.2 (1631) | **-1.7** | [-2.6, -0.9] |
| strategy/face_target_share_when_both_legal | 46.5 (946) | 34.9 (886) | **-11.6** | [-17.6, -5.6] |
| strategy/favorable_trade_taken_per_available_turn | 56.5 (395) | 72.8 (423) | **+16.4** | [+9.4, +23.5] |
| strategy/spell_cast_per_legal_main_turn | 27.3 (911) | 24.7 (1046) | **-2.7** | [-5.5, -0.0] |

### earth_c100 vs u8223 — sample — all|Stonehaven/Goro

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 41.1 (141) | 40.1 (242) | -1.1 | [-11.6, +9.2] |
| block/blocker_survives | 51.1 (141) | 35.1 (242) | **-15.9** | [-26.0, -5.6] |
| block/favorable_survive_or_kill | 61.0 (141) | 53.3 (242) | -7.7 | [-16.8, +3.0] |
| draft/max_wjaccard_to_curated | 15.4 (48) | 13.3 (48) | **-2.0** | [-2.9, -1.2] |
| draft/mean_cost | 2.98 (2400) | 3.27 (2400) | **+0.3** | [+0.2, +0.3] |
| draft/normal_share | 77.4 (2400) | 65.1 (2400) | **-12.3** | [-14.2, -10.4] |
| draft/spell_share | 9.6 (2400) | 9.1 (2400) | -0.5 | [-1.4, +0.4] |
| draft/unique_cards | 33.38 (48) | 34.94 (48) | **+1.6** | [+0.6, +2.4] |
| draft/weapon_share | 10.4 (2400) | 3.0 (2400) | **-7.5** | [-8.5, -6.5] |
| earth/quicksand_cast_multi_removal | 75.0 (44) | 77.5 (40) | +2.5 | [-14.4, +18.5] |
| earth/quicksand_cast_per_legal_turn | 58.7 (75) | 47.1 (85) | -11.6 | [-26.8, +2.6] |
| gate/Stonehaven/grant_then_block | 37.9 (319) | 56.8 (403) | **+18.9** | [+12.2, +26.1] |
| gate/Stonehaven/portal_then_grant | 71.4 (447) | 84.1 (479) | **+12.8** | [+8.0, +17.3] |
| gate/portal_per_legal_turn | 99.8 (448) | 98.4 (486) | **-1.4** | [-2.9, -0.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.4 (208) | 25.5 (251) | -1.9 | [-8.9, +4.2] |
| ikz/held_at_end_of_own_turn | 24.2 (815) | 29.2 (855) | **+5.1** | [+1.1, +9.2] |
| ikz/held_then_spent_in_opp_turn | 21.3 (197) | 22.8 (250) | +1.5 | [-3.0, +5.8] |
| leader/Goro/target_then_attacks_same_turn | 15.6 (64) | 6.8 (73) | -8.8 | [-20.1, +1.3] |
| leader/Goro/use_with_target | 100.0 (64) | 100.0 (73) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.9 (717) | 9.5 (769) | +0.6 | [-2.0, +3.3] |
| outcome/own_turns | 7.28 (112) | 7.63 (112) | **+0.4** | [+0.0, +0.7] |
| outcome/win | 47.3 (112) | 55.4 (112) | +8.0 | [-0.9, +17.0] |
| response/any_nonblock_response_when_legal | 64.3 (126) | 69.1 (181) | +4.8 | [-12.4, +20.8] |
| response/defender_declared_when_legal | 58.5 (241) | 50.1 (483) | -8.4 | [-17.5, +1.1] |
| response/spell_played_when_legal | 53.7 (82) | 58.2 (98) | +4.5 | [-17.5, +21.3] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.0 (94) | 69.2 (104) | +3.3 | [-10.9, +17.1] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.4 (94) | 46.2 (104) | +5.7 | [-5.3, +17.0] |
| strategy/attacks_by_equipped_attacker | 5.5 (1208) | 3.5 (1170) | **-2.0** | [-3.5, -0.5] |
| strategy/face_target_share_when_both_legal | 54.8 (719) | 46.0 (707) | **-8.8** | [-17.0, -1.5] |
| strategy/favorable_trade_taken_per_available_turn | 53.2 (301) | 59.3 (300) | +6.2 | [-1.6, +13.9] |
| strategy/spell_cast_per_legal_main_turn | 25.3 (427) | 23.4 (471) | -1.9 | [-7.1, +3.5] |

### earth_c100 vs u8223 — sample — fixed|ALL

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 28.4 (349) | 31.0 (551) | +2.7 | [-3.8, +8.6] |
| block/blocker_survives | 40.4 (349) | 31.9 (551) | **-8.5** | [-14.1, -2.8] |
| block/favorable_survive_or_kill | 50.7 (349) | 47.9 (551) | -2.8 | [-8.9, +2.9] |
| earth/quicksand_cast_multi_removal | 85.1 (114) | 84.1 (113) | -1.0 | [-8.2, +6.6] |
| earth/quicksand_cast_per_legal_turn | 53.3 (214) | 40.8 (277) | **-12.5** | [-20.7, -3.3] |
| gate/Stonehaven/grant_then_block | 40.5 (650) | 54.3 (827) | **+13.8** | [+8.7, +19.0] |
| gate/Stonehaven/portal_then_grant | 69.7 (932) | 80.2 (1031) | **+10.5** | [+6.6, +14.7] |
| gate/portal_per_legal_turn | 99.4 (938) | 97.0 (1063) | **-2.4** | [-3.5, -1.3] |
| heal/heal_spell_per_legal_turn_below_max_hp | 31.9 (540) | 28.6 (668) | -3.3 | [-6.9, +0.5] |
| ikz/held_at_end_of_own_turn | 36.3 (1549) | 41.0 (1668) | **+4.7** | [+2.6, +7.0] |
| ikz/held_then_spent_in_opp_turn | 35.1 (562) | 31.1 (684) | **-3.9** | [-7.2, -0.8] |
| leader/Bobu/earth_loss_before_next_turn | 46.2 (13) | 50.0 (4) | +3.8 | [-52.6, +64.3] |
| leader/Bobu/use_then_heal_observed | 12.9 (31) | 0.0 (5) | -12.9 | [-26.3, +0.0] |
| leader/Goro/target_then_attacks_same_turn | 21.1 (38) | 8.8 (57) | -12.3 | [-29.6, +2.1] |
| leader/Goro/use_with_target | 100.0 (38) | 100.0 (57) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 4.6 (1497) | 3.8 (1616) | -0.8 | [-2.1, +0.6] |
| outcome/own_turns | 8.07 (192) | 8.69 (192) | **+0.6** | [+0.3, +0.9] |
| outcome/win | 52.1 (192) | 64.6 (192) | **+12.5** | [+6.2, +19.8] |
| outcome/win_vs_EARTH | 60.4 (48) | 62.5 (48) | +2.1 | [-10.9, +14.6] |
| outcome/win_vs_FIRE | 25.0 (48) | 50.0 (48) | **+25.0** | [+8.3, +40.4] |
| outcome/win_vs_LIGHTNING | 56.2 (48) | 68.8 (48) | +12.5 | [+0.0, +26.9] |
| outcome/win_vs_WATER | 66.7 (48) | 77.1 (48) | +10.4 | [+0.0, +22.0] |
| response/any_nonblock_response_when_legal | 58.3 (499) | 61.2 (634) | +2.9 | [-4.2, +9.4] |
| response/defender_declared_when_legal | 48.9 (711) | 44.8 (1231) | -4.2 | [-10.2, +1.5] |
| response/spell_played_when_legal | 64.1 (315) | 57.9 (378) | -6.2 | [-15.1, +3.2] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 3.9 (128) | 1.6 (128) | -2.3 | [-5.9, +0.8] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (128) | 0.0 (128) | -2.3 | [-5.2, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 63.8 (177) | 70.4 (186) | +6.6 | [-1.4, +15.0] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.1 (177) | 39.2 (186) | -0.9 | [-8.8, +7.5] |
| strategy/attacks_by_equipped_attacker | 2.1 (1672) | 2.2 (1696) | +0.1 | [-0.5, +0.8] |
| strategy/face_target_share_when_both_legal | 50.8 (928) | 44.5 (878) | **-6.2** | [-11.8, -0.2] |
| strategy/favorable_trade_taken_per_available_turn | 52.6 (424) | 63.0 (440) | **+10.4** | [+4.2, +16.7] |
| strategy/spell_cast_per_legal_main_turn | 27.4 (1105) | 25.3 (1230) | -2.1 | [-4.5, +0.2] |

### earth_c100 vs u8223 — sample — fixed|Stonehaven/Bobu

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 21.1 (265) | 26.1 (421) | +5.0 | [-2.1, +11.8] |
| block/blocker_survives | 34.7 (265) | 27.8 (421) | **-6.9** | [-13.5, -0.4] |
| block/favorable_survive_or_kill | 44.2 (265) | 43.2 (421) | -0.9 | [-7.9, +5.9] |
| earth/quicksand_cast_multi_removal | 87.7 (81) | 86.6 (82) | -1.1 | [-8.8, +6.4] |
| earth/quicksand_cast_per_legal_turn | 50.9 (159) | 38.5 (213) | **-12.4** | [-21.5, -2.4] |
| gate/Stonehaven/grant_then_block | 39.2 (457) | 52.0 (614) | **+12.8** | [+6.6, +18.7] |
| gate/Stonehaven/portal_then_grant | 67.9 (673) | 78.5 (782) | **+10.6** | [+5.8, +15.6] |
| gate/portal_per_legal_turn | 99.1 (679) | 96.2 (813) | **-2.9** | [-4.3, -1.5] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.9 (381) | 29.5 (485) | **-4.4** | [-8.7, -0.1] |
| ikz/held_at_end_of_own_turn | 39.2 (1086) | 43.9 (1188) | **+4.7** | [+2.0, +7.4] |
| ikz/held_then_spent_in_opp_turn | 36.4 (426) | 30.8 (522) | **-5.5** | [-9.3, -2.0] |
| leader/Bobu/earth_loss_before_next_turn | 46.2 (13) | 50.0 (4) | +3.8 | [-54.5, +68.4] |
| leader/Bobu/use_then_heal_observed | 12.9 (31) | 0.0 (5) | -12.9 | [-26.3, +0.0] |
| leader/use_per_legal_turn | 2.9 (1086) | 0.4 (1188) | **-2.4** | [-3.5, -1.4] |
| outcome/own_turns | 8.48 (128) | 9.28 (128) | **+0.8** | [+0.4, +1.2] |
| outcome/win | 51.6 (128) | 67.2 (128) | **+15.6** | [+7.0, +24.2] |
| response/any_nonblock_response_when_legal | 56.3 (373) | 58.7 (463) | +2.4 | [-5.0, +9.4] |
| response/defender_declared_when_legal | 46.3 (570) | 42.7 (987) | -3.7 | [-9.7, +2.3] |
| response/spell_played_when_legal | 67.8 (233) | 58.6 (285) | -9.2 | [-18.6, +0.6] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 3.9 (128) | 1.6 (128) | -2.3 | [-5.5, +0.8] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (128) | 0.0 (128) | -2.3 | [-4.7, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.2 (123) | 68.8 (125) | +8.6 | [-1.9, +18.2] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 36.6 (123) | 36.8 (125) | +0.2 | [-11.2, +10.8] |
| strategy/attacks_by_equipped_attacker | 0.0 (1039) | 0.0 (1083) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 44.4 (565) | 34.7 (522) | **-9.8** | [-17.4, -1.5] |
| strategy/favorable_trade_taken_per_available_turn | 56.0 (259) | 73.4 (274) | **+17.4** | [+9.1, +25.7] |
| strategy/spell_cast_per_legal_main_turn | 27.6 (800) | 25.5 (909) | -2.1 | [-5.0, +0.6] |

### earth_c100 vs u8223 — sample — fixed|Stonehaven/Goro

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 51.2 (84) | 46.9 (130) | -4.3 | [-17.9, +9.1] |
| block/blocker_survives | 58.3 (84) | 45.4 (130) | **-12.9** | [-24.6, -2.1] |
| block/favorable_survive_or_kill | 71.4 (84) | 63.1 (130) | -8.4 | [-19.5, +2.5] |
| earth/quicksand_cast_multi_removal | 78.8 (33) | 77.4 (31) | -1.4 | [-19.1, +14.2] |
| earth/quicksand_cast_per_legal_turn | 60.0 (55) | 48.4 (64) | -11.6 | [-28.6, +6.5] |
| gate/Stonehaven/grant_then_block | 43.5 (193) | 61.0 (213) | **+17.5** | [+9.3, +25.6] |
| gate/Stonehaven/portal_then_grant | 74.5 (259) | 85.5 (249) | **+11.0** | [+5.6, +16.5] |
| gate/portal_per_legal_turn | 100.0 (259) | 99.6 (250) | -0.4 | [-1.3, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.0 (159) | 26.2 (183) | -0.8 | [-8.1, +5.7] |
| ikz/held_at_end_of_own_turn | 29.4 (463) | 33.8 (480) | +4.4 | [-0.1, +8.4] |
| ikz/held_then_spent_in_opp_turn | 30.9 (136) | 32.1 (162) | +1.2 | [-4.3, +6.5] |
| leader/Goro/target_then_attacks_same_turn | 21.1 (38) | 8.8 (57) | -12.3 | [-29.0, +2.4] |
| leader/Goro/use_with_target | 100.0 (38) | 100.0 (57) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.2 (411) | 13.3 (428) | **+4.1** | [+0.2, +7.7] |
| outcome/own_turns | 7.23 (64) | 7.50 (64) | +0.3 | [-0.1, +0.7] |
| outcome/win | 53.1 (64) | 59.4 (64) | +6.2 | [-4.7, +17.2] |
| response/any_nonblock_response_when_legal | 64.3 (126) | 67.8 (171) | +3.6 | [-14.0, +19.7] |
| response/defender_declared_when_legal | 59.6 (141) | 53.3 (244) | -6.3 | [-17.5, +5.5] |
| response/spell_played_when_legal | 53.7 (82) | 55.9 (93) | +2.3 | [-20.2, +19.8] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 72.2 (54) | 73.8 (61) | +1.5 | [-11.7, +16.0] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 48.1 (54) | 44.3 (61) | -3.9 | [-13.4, +7.0] |
| strategy/attacks_by_equipped_attacker | 5.5 (633) | 6.2 (613) | +0.7 | [-1.0, +2.3] |
| strategy/face_target_share_when_both_legal | 60.6 (363) | 59.0 (356) | -1.6 | [-9.0, +6.6] |
| strategy/favorable_trade_taken_per_available_turn | 47.3 (165) | 45.8 (166) | -1.5 | [-11.3, +7.2] |
| strategy/spell_cast_per_legal_main_turn | 26.9 (305) | 24.6 (321) | -2.3 | [-7.0, +2.4] |

### earth_c100 vs u8223 — sample — free_draft|ALL

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 38.9 (126) | 29.8 (225) | -9.1 | [-20.6, +1.8] |
| block/blocker_survives | 35.7 (126) | 24.4 (225) | **-11.3** | [-22.4, -0.5] |
| block/favorable_survive_or_kill | 50.8 (126) | 41.3 (225) | -9.5 | [-20.9, +2.7] |
| draft/max_wjaccard_to_curated | 12.2 (96) | 13.0 (96) | +0.9 | [-0.2, +2.0] |
| draft/mean_cost | 2.98 (4800) | 3.26 (4800) | **+0.3** | [+0.2, +0.3] |
| draft/normal_share | 77.0 (4800) | 64.7 (4800) | **-12.3** | [-13.6, -11.0] |
| draft/spell_share | 9.3 (4800) | 8.7 (4800) | -0.5 | [-1.2, +0.1] |
| draft/unique_cards | 33.14 (96) | 34.79 (96) | **+1.7** | [+1.1, +2.2] |
| draft/weapon_share | 10.5 (4800) | 2.8 (4800) | **-7.6** | [-8.3, -7.0] |
| earth/quicksand_cast_multi_removal | 70.0 (20) | 75.0 (20) | +5.0 | [-23.1, +29.8] |
| earth/quicksand_cast_per_legal_turn | 55.6 (36) | 38.5 (52) | **-17.1** | [-39.3, -0.1] |
| gate/Stonehaven/grant_then_block | 34.8 (256) | 50.7 (383) | **+15.9** | [+7.3, +24.3] |
| gate/Stonehaven/portal_then_grant | 70.1 (365) | 84.4 (454) | **+14.2** | [+8.6, +19.9] |
| gate/portal_per_legal_turn | 98.6 (370) | 96.8 (468) | -1.9 | [-3.9, +0.3] |
| heal/heal_spell_per_legal_turn_below_max_hp | 30.6 (85) | 23.2 (125) | -7.4 | [-18.7, +3.7] |
| ikz/held_at_end_of_own_turn | 18.6 (714) | 24.8 (755) | **+6.1** | [+1.7, +10.9] |
| ikz/held_then_spent_in_opp_turn | 1.5 (133) | 5.3 (187) | **+3.8** | [+0.5, +7.4] |
| leader/Bobu/use_then_heal_observed | 10.0 (10) | 100.0 (2) | **+90.0** | [+50.0, +100.0] |
| leader/Goro/target_then_attacks_same_turn | 7.7 (26) | 0.0 (16) | -7.7 | [-21.7, +0.0] |
| leader/Goro/use_with_target | 100.0 (26) | 100.0 (16) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.4 (668) | 2.5 (721) | **-2.9** | [-4.8, -1.0] |
| outcome/own_turns | 7.44 (96) | 7.86 (96) | **+0.4** | [+0.1, +0.8] |
| outcome/win | 41.7 (96) | 46.9 (96) | +5.2 | [-5.2, +15.6] |
| outcome/win_vs_EARTH | 25.0 (24) | 58.3 (24) | **+33.3** | [+11.5, +56.2] |
| outcome/win_vs_FIRE | 29.2 (24) | 20.8 (24) | -8.3 | [-25.0, +7.1] |
| outcome/win_vs_LIGHTNING | 50.0 (24) | 54.2 (24) | +4.2 | [-16.7, +22.7] |
| outcome/win_vs_WATER | 62.5 (24) | 54.2 (24) | -8.3 | [-28.6, +13.6] |
| response/any_nonblock_response_when_legal | 100.0 (4) | 82.8 (29) | **-17.2** | [-29.7, -4.5] |
| response/defender_declared_when_legal | 55.8 (226) | 46.1 (488) | -9.6 | [-20.7, +2.0] |
| response/spell_played_when_legal | 100.0 (2) | 90.9 (11) | -9.1 | [-33.3, +0.0] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.2 (48) | 4.2 (48) | +0.0 | [-6.0, +5.8] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.1 (48) | 4.2 (48) | +2.1 | [+0.0, +6.8] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.5 (76) | 63.3 (90) | +2.8 | [-13.5, +19.0] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 42.1 (76) | 43.3 (90) | +1.2 | [-13.6, +16.8] |
| strategy/attacks_by_equipped_attacker | 5.4 (1192) | 0.5 (1105) | **-4.8** | [-6.0, -3.6] |
| strategy/face_target_share_when_both_legal | 49.3 (737) | 34.0 (715) | **-15.3** | [-23.0, -8.3] |
| strategy/favorable_trade_taken_per_available_turn | 58.8 (272) | 73.9 (283) | **+15.0** | [+5.9, +23.2] |
| strategy/spell_cast_per_legal_main_turn | 23.2 (233) | 19.9 (287) | -3.3 | [-11.4, +4.9] |

### earth_c100 vs u8223 — sample — free_draft|Stonehaven/Bobu

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 49.3 (69) | 27.4 (113) | **-21.8** | [-37.9, -7.6] |
| block/blocker_survives | 31.9 (69) | 25.7 (113) | -6.2 | [-18.8, +5.1] |
| block/favorable_survive_or_kill | 55.1 (69) | 40.7 (113) | -14.4 | [-32.1, +2.2] |
| draft/max_wjaccard_to_curated | 9.0 (48) | 12.7 (48) | **+3.8** | [+2.6, +5.0] |
| draft/mean_cost | 2.99 (2400) | 3.25 (2400) | **+0.3** | [+0.2, +0.3] |
| draft/normal_share | 76.5 (2400) | 64.3 (2400) | **-12.2** | [-14.2, -10.2] |
| draft/spell_share | 8.9 (2400) | 8.3 (2400) | -0.6 | [-1.7, +0.5] |
| draft/unique_cards | 32.90 (48) | 34.65 (48) | **+1.8** | [+1.1, +2.4] |
| draft/weapon_share | 10.5 (2400) | 2.7 (2400) | **-7.8** | [-8.7, -6.9] |
| earth/quicksand_cast_multi_removal | 77.8 (9) | 72.7 (11) | -5.1 | [-42.9, +22.2] |
| earth/quicksand_cast_per_legal_turn | 56.2 (16) | 35.5 (31) | -20.8 | [-63.5, +4.3] |
| gate/Stonehaven/grant_then_block | 40.0 (130) | 49.2 (193) | +9.2 | [-1.6, +19.9] |
| gate/Stonehaven/portal_then_grant | 73.4 (177) | 86.2 (224) | **+12.7** | [+5.0, +21.4] |
| gate/portal_per_legal_turn | 97.8 (181) | 96.6 (232) | -1.2 | [-4.3, +2.5] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.3 (36) | 22.8 (57) | -10.5 | [-28.2, +6.8] |
| ikz/held_at_end_of_own_turn | 19.9 (362) | 26.1 (380) | +6.2 | [-0.3, +12.6] |
| ikz/held_then_spent_in_opp_turn | 2.8 (72) | 5.1 (99) | +2.3 | [-2.2, +7.6] |
| leader/Bobu/use_then_heal_observed | 10.0 (10) | 100.0 (2) | **+90.0** | [+58.3, +100.0] |
| leader/use_per_legal_turn | 2.8 (362) | 0.5 (380) | **-2.2** | [-4.7, -0.3] |
| outcome/own_turns | 7.54 (48) | 7.92 (48) | +0.4 | [-0.1, +0.9] |
| outcome/win | 43.8 (48) | 43.8 (48) | +0.0 | [-12.5, +12.5] |
| response/any_nonblock_response_when_legal | 100.0 (4) | 78.9 (19) | -21.1 | [-42.9, +0.0] |
| response/defender_declared_when_legal | 54.8 (126) | 45.4 (249) | -9.4 | [-27.0, +6.8] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.2 (48) | 4.2 (48) | +0.0 | [-6.2, +6.2] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.1 (48) | 4.2 (48) | +2.1 | [+0.0, +6.2] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 63.9 (36) | 63.8 (47) | -0.1 | [-20.1, +22.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 55.6 (36) | 38.3 (47) | -17.3 | [-38.4, +5.2] |
| strategy/attacks_by_equipped_attacker | 5.2 (617) | 0.5 (548) | **-4.6** | [-6.3, -3.0] |
| strategy/face_target_share_when_both_legal | 49.6 (381) | 35.2 (364) | **-14.4** | [-23.1, -6.2] |
| strategy/favorable_trade_taken_per_available_turn | 57.4 (136) | 71.8 (149) | **+14.5** | [+2.0, +26.8] |
| strategy/spell_cast_per_legal_main_turn | 25.2 (111) | 19.0 (137) | -6.2 | [-17.1, +4.1] |

### earth_c100 vs u8223 — sample — free_draft|Stonehaven/Goro

| metric | u8223 | earth_c100 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 26.3 (57) | 32.1 (112) | +5.8 | [-9.1, +21.3] |
| block/blocker_survives | 40.4 (57) | 23.2 (112) | -17.1 | [-36.0, +1.3] |
| block/favorable_survive_or_kill | 45.6 (57) | 42.0 (112) | -3.6 | [-20.7, +14.2] |
| draft/max_wjaccard_to_curated | 15.4 (48) | 13.3 (48) | **-2.0** | [-2.9, -1.2] |
| draft/mean_cost | 2.98 (2400) | 3.27 (2400) | **+0.3** | [+0.2, +0.3] |
| draft/normal_share | 77.4 (2400) | 65.1 (2400) | **-12.3** | [-14.2, -10.3] |
| draft/spell_share | 9.6 (2400) | 9.1 (2400) | -0.5 | [-1.4, +0.4] |
| draft/unique_cards | 33.38 (48) | 34.94 (48) | **+1.6** | [+0.6, +2.4] |
| draft/weapon_share | 10.4 (2400) | 3.0 (2400) | **-7.5** | [-8.5, -6.5] |
| earth/quicksand_cast_multi_removal | 63.6 (11) | 77.8 (9) | +14.1 | [-33.3, +56.7] |
| earth/quicksand_cast_per_legal_turn | 55.0 (20) | 42.9 (21) | -12.1 | [-43.7, +10.0] |
| gate/Stonehaven/grant_then_block | 29.4 (126) | 52.1 (190) | **+22.7** | [+10.9, +35.0] |
| gate/Stonehaven/portal_then_grant | 67.0 (188) | 82.6 (230) | **+15.6** | [+7.5, +23.0] |
| gate/portal_per_legal_turn | 99.5 (189) | 97.0 (236) | -2.4 | [-5.2, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 28.6 (49) | 23.5 (68) | -5.0 | [-22.9, +10.9] |
| ikz/held_at_end_of_own_turn | 17.3 (352) | 23.5 (375) | +6.1 | [-1.3, +13.5] |
| ikz/held_then_spent_in_opp_turn | 0.0 (61) | 5.7 (88) | **+5.7** | [+1.2, +11.1] |
| leader/Goro/target_then_attacks_same_turn | 7.7 (26) | 0.0 (16) | -7.7 | [-22.2, +0.0] |
| leader/Goro/use_with_target | 100.0 (26) | 100.0 (16) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.5 (306) | 4.7 (341) | **-3.8** | [-7.0, -0.6] |
| outcome/own_turns | 7.33 (48) | 7.81 (48) | +0.5 | [-0.1, +1.1] |
| outcome/win | 39.6 (48) | 50.0 (48) | +10.4 | [-4.2, +25.0] |
| response/any_nonblock_response_when_legal | – (0) | 90.0 (10) | – | |
| response/defender_declared_when_legal | 57.0 (100) | 46.9 (239) | -10.1 | [-26.0, +4.4] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 57.5 (40) | 62.8 (43) | +5.3 | [-21.2, +30.6] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 30.0 (40) | 48.8 (43) | +18.8 | [-3.3, +38.9] |
| strategy/attacks_by_equipped_attacker | 5.6 (575) | 0.5 (557) | **-5.0** | [-6.6, -3.2] |
| strategy/face_target_share_when_both_legal | 48.9 (356) | 32.8 (351) | **-16.1** | [-29.4, -4.2] |
| strategy/favorable_trade_taken_per_available_turn | 60.3 (136) | 76.1 (134) | **+15.8** | [+4.8, +27.7] |
| strategy/spell_cast_per_legal_main_turn | 21.3 (122) | 20.7 (150) | -0.6 | [-13.7, +12.1] |

### earth_c75 vs u8223 — argmax — all|ALL

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 29.5 (543) | 34.2 (704) | +4.8 | [-0.1, +9.5] |
| block/blocker_survives | 41.6 (543) | 34.1 (704) | **-7.5** | [-12.7, -2.5] |
| block/favorable_survive_or_kill | 50.8 (543) | 48.0 (704) | -2.8 | [-7.7, +2.1] |
| draft/max_wjaccard_to_curated | 25.4 (96) | 18.9 (96) | **-6.5** | [-8.3, -4.7] |
| draft/mean_cost | 3.36 (4800) | 3.48 (4800) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (4800) | 60.0 (4800) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 15.0 (4800) | 6.0 (4800) | **-9.0** | [-9.3, -8.7] |
| draft/unique_cards | 17.00 (96) | 19.00 (96) | **+2.0** | [+2.0, +2.0] |
| draft/weapon_share | 10.0 (4800) | 0.0 (4800) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 86.9 (168) | 84.7 (137) | -2.2 | [-9.2, +4.9] |
| earth/quicksand_cast_per_legal_turn | 47.5 (354) | 37.6 (364) | **-9.8** | [-16.9, -2.8] |
| gate/Stonehaven/grant_then_block | 40.7 (1009) | 48.6 (1270) | **+7.8** | [+4.1, +11.9] |
| gate/Stonehaven/portal_then_grant | 75.2 (1342) | 83.0 (1530) | **+7.8** | [+4.8, +10.9] |
| gate/portal_per_legal_turn | 99.6 (1348) | 96.8 (1580) | **-2.7** | [-3.8, -1.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 28.9 (757) | 23.7 (786) | **-5.3** | [-8.9, -1.8] |
| ikz/held_at_end_of_own_turn | 33.1 (2302) | 33.3 (2402) | +0.2 | [-2.0, +2.4] |
| ikz/held_then_spent_in_opp_turn | 25.3 (763) | 27.8 (801) | +2.5 | [-0.3, +5.4] |
| leader/Bobu/earth_loss_before_next_turn | 57.1 (28) | 75.0 (4) | +17.9 | [-52.2, +57.1] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 40.0 (5) | +25.2 | [-18.6, +85.7] |
| leader/Goro/target_then_attacks_same_turn | 13.8 (65) | 16.7 (60) | +2.8 | [-12.4, +14.8] |
| leader/Goro/use_with_target | 100.0 (65) | 100.0 (60) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.4 (2193) | 2.8 (2316) | **-2.6** | [-4.0, -1.2] |
| outcome/own_turns | 7.99 (288) | 8.34 (288) | **+0.3** | [+0.1, +0.5] |
| outcome/win | 54.9 (288) | 63.2 (288) | **+8.3** | [+2.8, +14.2] |
| outcome/win_vs_EARTH | 52.8 (72) | 66.7 (72) | **+13.9** | [+3.3, +25.8] |
| outcome/win_vs_FIRE | 40.3 (72) | 47.2 (72) | +6.9 | [-5.7, +20.7] |
| outcome/win_vs_LIGHTNING | 61.1 (72) | 65.3 (72) | +4.2 | [-7.6, +16.1] |
| outcome/win_vs_WATER | 65.3 (72) | 73.6 (72) | +8.3 | [-2.6, +18.8] |
| response/any_nonblock_response_when_legal | 61.2 (454) | 64.0 (698) | +2.8 | [-5.4, +11.2] |
| response/defender_declared_when_legal | 55.2 (982) | 42.4 (1662) | **-12.8** | [-17.6, -7.8] |
| response/spell_played_when_legal | 70.5 (281) | 62.3 (371) | -8.2 | [-18.3, +2.4] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.0 (176) | 1.7 (176) | **-6.2** | [-10.6, -2.1] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.7 (176) | 1.1 (176) | **-4.5** | [-8.5, -0.6] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.7 (264) | 72.2 (281) | +5.6 | [-1.7, +12.9] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 41.7 (264) | 47.3 (281) | +5.7 | [-1.5, +13.1] |
| strategy/attacks_by_equipped_attacker | 3.8 (2829) | 1.2 (3060) | **-2.6** | [-3.4, -1.8] |
| strategy/face_target_share_when_both_legal | 50.8 (1532) | 40.2 (1605) | **-10.6** | [-14.9, -6.0] |
| strategy/favorable_trade_taken_per_available_turn | 55.1 (670) | 69.4 (756) | **+14.4** | [+9.6, +19.1] |
| strategy/spell_cast_per_legal_main_turn | 28.3 (1453) | 24.4 (1381) | **-3.9** | [-6.5, -1.4] |

### earth_c75 vs u8223 — argmax — all|Stonehaven/Bobu

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 23.5 (357) | 29.7 (481) | **+6.2** | [+0.3, +11.9] |
| block/blocker_survives | 37.3 (357) | 30.6 (481) | **-6.7** | [-12.9, -0.5] |
| block/favorable_survive_or_kill | 45.7 (357) | 43.9 (481) | -1.8 | [-7.1, +3.6] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 18.3 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 3.36 (2400) | 3.48 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (2400) | 60.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 16.0 (2400) | 6.0 (2400) | **-10.0** | [-10.0, -10.0] |
| draft/unique_cards | 17.00 (48) | 19.00 (48) | **+2.0** | [+2.0, +2.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 89.2 (102) | 82.6 (92) | -6.6 | [-15.7, +2.4] |
| earth/quicksand_cast_per_legal_turn | 47.0 (217) | 38.7 (238) | -8.3 | [-16.9, +0.9] |
| gate/Stonehaven/grant_then_block | 38.7 (646) | 47.4 (832) | **+8.7** | [+4.1, +13.0] |
| gate/Stonehaven/portal_then_grant | 73.0 (885) | 81.2 (1025) | **+8.2** | [+4.3, +12.1] |
| gate/portal_per_legal_turn | 99.3 (891) | 96.6 (1061) | **-2.7** | [-4.1, -1.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 29.6 (494) | 23.9 (553) | **-5.7** | [-9.6, -1.8] |
| ikz/held_at_end_of_own_turn | 36.5 (1470) | 36.3 (1541) | -0.3 | [-3.2, +2.7] |
| ikz/held_then_spent_in_opp_turn | 27.6 (537) | 30.1 (559) | +2.5 | [-1.2, +6.1] |
| leader/Bobu/earth_loss_before_next_turn | 57.1 (28) | 75.0 (4) | +17.9 | [-50.0, +57.7] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 40.0 (5) | +25.2 | [-18.9, +82.8] |
| leader/use_per_legal_turn | 3.7 (1470) | 0.3 (1541) | **-3.3** | [-4.5, -2.2] |
| outcome/own_turns | 8.35 (176) | 8.76 (176) | **+0.4** | [+0.1, +0.7] |
| outcome/win | 56.8 (176) | 64.8 (176) | +8.0 | [+0.0, +15.3] |
| response/any_nonblock_response_when_legal | 58.5 (342) | 61.2 (495) | +2.7 | [-7.1, +11.8] |
| response/defender_declared_when_legal | 50.4 (707) | 39.8 (1208) | **-10.5** | [-16.1, -5.1] |
| response/spell_played_when_legal | 73.9 (203) | 62.9 (280) | -11.0 | [-22.6, +0.7] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.0 (176) | 1.7 (176) | **-6.2** | [-10.8, -2.3] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.7 (176) | 1.1 (176) | **-4.5** | [-9.1, -0.6] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 64.5 (166) | 71.8 (174) | +7.4 | [-2.4, +16.8] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 38.6 (166) | 46.6 (174) | +8.0 | [-1.1, +17.2] |
| strategy/attacks_by_equipped_attacker | 2.4 (1659) | 0.0 (1795) | **-2.4** | [-3.3, -1.5] |
| strategy/face_target_share_when_both_legal | 44.0 (877) | 35.2 (925) | **-8.8** | [-14.4, -2.9] |
| strategy/favorable_trade_taken_per_available_turn | 59.9 (377) | 77.6 (433) | **+17.7** | [+11.2, +23.9] |
| strategy/spell_cast_per_legal_main_turn | 27.3 (974) | 24.7 (952) | -2.6 | [-5.7, +0.4] |

### earth_c75 vs u8223 — argmax — all|Stonehaven/Goro

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 40.9 (186) | 43.9 (223) | +3.1 | [-5.4, +11.4] |
| block/blocker_survives | 50.0 (186) | 41.7 (223) | -8.3 | [-18.6, +1.9] |
| block/favorable_survive_or_kill | 60.8 (186) | 57.0 (223) | -3.8 | [-14.2, +6.4] |
| draft/max_wjaccard_to_curated | 32.5 (48) | 19.6 (48) | **-13.0** | [-13.0, -13.0] |
| draft/mean_cost | 3.36 (2400) | 3.48 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (2400) | 60.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 14.0 (2400) | 6.0 (2400) | **-8.0** | [-8.0, -8.0] |
| draft/unique_cards | 17.00 (48) | 19.00 (48) | **+2.0** | [+2.0, +2.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 83.3 (66) | 88.9 (45) | +5.6 | [-4.0, +14.6] |
| earth/quicksand_cast_per_legal_turn | 48.2 (137) | 35.7 (126) | **-12.5** | [-24.1, -0.8] |
| gate/Stonehaven/grant_then_block | 44.4 (363) | 50.9 (438) | +6.6 | [-1.0, +13.8] |
| gate/Stonehaven/portal_then_grant | 79.4 (457) | 86.7 (505) | **+7.3** | [+1.9, +12.3] |
| gate/portal_per_legal_turn | 100.0 (457) | 97.3 (519) | **-2.7** | [-4.4, -1.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.8 (263) | 23.2 (233) | -4.6 | [-12.0, +2.3] |
| ikz/held_at_end_of_own_turn | 27.2 (832) | 28.1 (861) | +0.9 | [-2.8, +4.9] |
| ikz/held_then_spent_in_opp_turn | 19.9 (226) | 22.7 (242) | +2.8 | [-1.5, +7.4] |
| leader/Goro/target_then_attacks_same_turn | 13.8 (65) | 16.7 (60) | +2.8 | [-10.9, +14.4] |
| leader/Goro/use_with_target | 100.0 (65) | 100.0 (60) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.0 (723) | 7.7 (775) | -1.2 | [-4.6, +2.3] |
| outcome/own_turns | 7.43 (112) | 7.69 (112) | +0.3 | [-0.0, +0.6] |
| outcome/win | 51.8 (112) | 60.7 (112) | +8.9 | [+0.0, +17.9] |
| response/any_nonblock_response_when_legal | 69.6 (112) | 70.9 (203) | +1.3 | [-13.7, +18.5] |
| response/defender_declared_when_legal | 67.6 (275) | 49.1 (454) | **-18.5** | [-27.4, -9.7] |
| response/spell_played_when_legal | 61.5 (78) | 60.4 (91) | -1.1 | [-20.4, +19.2] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 70.4 (98) | 72.9 (107) | +2.5 | [-7.9, +13.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 46.9 (98) | 48.6 (107) | +1.7 | [-9.9, +13.4] |
| strategy/attacks_by_equipped_attacker | 5.9 (1170) | 3.0 (1265) | **-2.9** | [-4.6, -1.1] |
| strategy/face_target_share_when_both_legal | 60.0 (655) | 47.1 (680) | **-12.9** | [-19.3, -6.2] |
| strategy/favorable_trade_taken_per_available_turn | 48.8 (293) | 58.5 (323) | **+9.7** | [+1.6, +16.9] |
| strategy/spell_cast_per_legal_main_turn | 30.3 (479) | 23.8 (429) | **-6.5** | [-11.1, -1.8] |

### earth_c75 vs u8223 — argmax — fixed|ALL

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 25.9 (344) | 30.7 (502) | +4.8 | [-1.4, +10.7] |
| block/blocker_survives | 40.1 (344) | 32.5 (502) | **-7.6** | [-14.3, -1.1] |
| block/favorable_survive_or_kill | 49.1 (344) | 46.2 (502) | -2.9 | [-8.6, +3.3] |
| earth/quicksand_cast_multi_removal | 85.8 (113) | 84.8 (105) | -1.1 | [-9.6, +7.4] |
| earth/quicksand_cast_per_legal_turn | 52.8 (214) | 40.5 (259) | **-12.3** | [-21.4, -2.8] |
| gate/Stonehaven/grant_then_block | 40.6 (645) | 51.2 (811) | **+10.6** | [+6.1, +14.8] |
| gate/Stonehaven/portal_then_grant | 69.4 (929) | 80.8 (1004) | **+11.3** | [+7.9, +14.9] |
| gate/portal_per_legal_turn | 99.5 (934) | 97.8 (1027) | **-1.7** | [-2.9, -0.6] |
| heal/heal_spell_per_legal_turn_below_max_hp | 32.2 (544) | 27.2 (655) | **-5.0** | [-8.7, -1.5] |
| ikz/held_at_end_of_own_turn | 36.1 (1557) | 39.3 (1633) | **+3.2** | [+0.8, +5.7] |
| ikz/held_then_spent_in_opp_turn | 34.3 (562) | 34.8 (641) | +0.4 | [-3.0, +4.2] |
| leader/Bobu/earth_loss_before_next_turn | 56.5 (23) | 75.0 (4) | +18.5 | [-52.6, +61.9] |
| leader/Bobu/use_then_heal_observed | 13.2 (38) | 40.0 (5) | +26.8 | [-17.8, +85.7] |
| leader/Goro/target_then_attacks_same_turn | 18.9 (37) | 17.6 (51) | -1.3 | [-20.5, +14.3] |
| leader/Goro/use_with_target | 100.0 (37) | 100.0 (51) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.0 (1509) | 3.5 (1583) | -1.4 | [-3.2, +0.5] |
| outcome/own_turns | 8.11 (192) | 8.51 (192) | **+0.4** | [+0.1, +0.7] |
| outcome/win | 53.6 (192) | 58.9 (192) | +5.2 | [-1.0, +11.5] |
| outcome/win_vs_EARTH | 54.2 (48) | 56.2 (48) | +2.1 | [-8.7, +13.0] |
| outcome/win_vs_FIRE | 35.4 (48) | 45.8 (48) | +10.4 | [-4.5, +25.9] |
| outcome/win_vs_LIGHTNING | 62.5 (48) | 64.6 (48) | +2.1 | [-10.9, +15.0] |
| outcome/win_vs_WATER | 62.5 (48) | 68.8 (48) | +6.2 | [-5.6, +19.2] |
| response/any_nonblock_response_when_legal | 61.2 (454) | 63.0 (675) | +1.7 | [-6.7, +9.2] |
| response/defender_declared_when_legal | 47.3 (725) | 39.0 (1288) | **-8.3** | [-14.0, -3.1] |
| response/spell_played_when_legal | 70.5 (281) | 62.3 (371) | -8.2 | [-18.8, +1.9] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.6 (128) | 2.3 (128) | **-6.2** | [-12.1, -0.8] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.5 (128) | 1.6 (128) | -3.9 | [-9.4, +0.8] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.2 (176) | 68.3 (186) | +8.1 | [-1.0, +17.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 34.7 (176) | 40.3 (186) | +5.7 | [-3.2, +14.2] |
| strategy/attacks_by_equipped_attacker | 2.0 (1710) | 2.2 (1700) | +0.2 | [-0.4, +0.9] |
| strategy/face_target_share_when_both_legal | 49.8 (941) | 46.2 (900) | -3.6 | [-8.7, +1.4] |
| strategy/favorable_trade_taken_per_available_turn | 54.9 (428) | 65.6 (451) | **+10.7** | [+4.2, +17.3] |
| strategy/spell_cast_per_legal_main_turn | 27.6 (1113) | 25.3 (1159) | -2.3 | [-4.9, +0.3] |

### earth_c75 vs u8223 — argmax — fixed|Stonehaven/Bobu

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 18.7 (262) | 25.4 (378) | +6.7 | [-0.2, +13.1] |
| block/blocker_survives | 34.4 (262) | 28.3 (378) | -6.0 | [-13.0, +1.2] |
| block/favorable_survive_or_kill | 43.5 (262) | 41.0 (378) | -2.5 | [-8.8, +3.5] |
| earth/quicksand_cast_multi_removal | 88.6 (79) | 84.8 (79) | -3.8 | [-14.0, +6.1] |
| earth/quicksand_cast_per_legal_turn | 50.6 (156) | 40.1 (197) | **-10.5** | [-21.0, -0.6] |
| gate/Stonehaven/grant_then_block | 39.0 (462) | 49.2 (591) | **+10.3** | [+5.6, +15.1] |
| gate/Stonehaven/portal_then_grant | 67.7 (682) | 78.5 (753) | **+10.7** | [+6.3, +15.3] |
| gate/portal_per_legal_turn | 99.3 (687) | 97.0 (776) | **-2.2** | [-3.7, -0.8] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.3 (387) | 26.9 (483) | **-6.4** | [-10.6, -2.3] |
| ikz/held_at_end_of_own_turn | 39.1 (1103) | 42.2 (1154) | **+3.1** | [+0.4, +5.7] |
| ikz/held_then_spent_in_opp_turn | 34.3 (431) | 34.5 (487) | +0.2 | [-3.9, +4.1] |
| leader/Bobu/earth_loss_before_next_turn | 56.5 (23) | 75.0 (4) | +18.5 | [-50.0, +59.4] |
| leader/Bobu/use_then_heal_observed | 13.2 (38) | 40.0 (5) | +26.8 | [-17.0, +83.3] |
| leader/use_per_legal_turn | 3.4 (1103) | 0.4 (1154) | **-3.0** | [-4.4, -1.8] |
| outcome/own_turns | 8.62 (128) | 9.02 (128) | **+0.4** | [+0.1, +0.7] |
| outcome/win | 55.5 (128) | 61.7 (128) | +6.2 | [-2.3, +14.8] |
| response/any_nonblock_response_when_legal | 58.5 (342) | 60.4 (485) | +1.9 | [-7.9, +10.8] |
| response/defender_declared_when_legal | 44.3 (589) | 37.0 (1021) | **-7.3** | [-13.0, -1.6] |
| response/spell_played_when_legal | 73.9 (203) | 62.9 (280) | -11.0 | [-22.4, +0.3] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.6 (128) | 2.3 (128) | **-6.2** | [-11.7, -0.8] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.5 (128) | 1.6 (128) | -3.9 | [-9.4, +0.8] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 57.4 (122) | 66.7 (126) | +9.3 | [-2.1, +20.1] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 31.1 (122) | 38.9 (126) | +7.7 | [-3.1, +18.4] |
| strategy/attacks_by_equipped_attacker | 0.0 (1081) | 0.0 (1077) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 42.2 (573) | 39.7 (549) | -2.5 | [-9.5, +4.5] |
| strategy/favorable_trade_taken_per_available_turn | 59.4 (261) | 75.2 (278) | **+15.8** | [+8.1, +23.6] |
| strategy/spell_cast_per_legal_main_turn | 27.5 (810) | 25.8 (849) | -1.7 | [-4.6, +1.3] |

### earth_c75 vs u8223 — argmax — fixed|Stonehaven/Goro

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 48.8 (82) | 46.8 (124) | -2.0 | [-12.6, +9.2] |
| block/blocker_survives | 58.5 (82) | 45.2 (124) | -13.4 | [-27.8, +1.0] |
| block/favorable_survive_or_kill | 67.1 (82) | 62.1 (124) | -5.0 | [-18.9, +8.9] |
| earth/quicksand_cast_multi_removal | 79.4 (34) | 84.6 (26) | +5.2 | [-10.1, +19.0] |
| earth/quicksand_cast_per_legal_turn | 58.6 (58) | 41.9 (62) | -16.7 | [-38.0, +4.2] |
| gate/Stonehaven/grant_then_block | 44.8 (183) | 56.4 (220) | **+11.6** | [+3.0, +19.6] |
| gate/Stonehaven/portal_then_grant | 74.1 (247) | 87.6 (251) | **+13.6** | [+8.3, +19.5] |
| gate/portal_per_legal_turn | 100.0 (247) | 100.0 (251) | +0.0 | [+0.0, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 29.3 (157) | 27.9 (172) | -1.4 | [-8.4, +5.0] |
| ikz/held_at_end_of_own_turn | 28.9 (454) | 32.2 (479) | +3.3 | [-1.8, +9.1] |
| ikz/held_then_spent_in_opp_turn | 34.4 (131) | 35.7 (154) | +1.4 | [-6.7, +9.4] |
| leader/Goro/target_then_attacks_same_turn | 18.9 (37) | 17.6 (51) | -1.3 | [-19.4, +14.4] |
| leader/Goro/use_with_target | 100.0 (37) | 100.0 (51) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.1 (406) | 11.9 (429) | +2.8 | [-1.7, +7.7] |
| outcome/own_turns | 7.09 (64) | 7.48 (64) | +0.4 | [+0.0, +0.9] |
| outcome/win | 50.0 (64) | 53.1 (64) | +3.1 | [-6.2, +12.5] |
| response/any_nonblock_response_when_legal | 69.6 (112) | 69.5 (190) | -0.2 | [-15.4, +16.4] |
| response/defender_declared_when_legal | 60.3 (136) | 46.4 (267) | **-13.9** | [-27.0, -0.9] |
| response/spell_played_when_legal | 61.5 (78) | 60.4 (91) | -1.1 | [-21.1, +18.3] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.7 (54) | 71.7 (60) | +5.0 | [-9.2, +19.5] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 42.6 (54) | 43.3 (60) | +0.7 | [-15.3, +15.8] |
| strategy/attacks_by_equipped_attacker | 5.4 (629) | 6.1 (623) | +0.7 | [-1.0, +2.4] |
| strategy/face_target_share_when_both_legal | 61.7 (368) | 56.4 (351) | -5.3 | [-11.8, +2.0] |
| strategy/favorable_trade_taken_per_available_turn | 47.9 (167) | 50.3 (173) | +2.4 | [-10.2, +13.5] |
| strategy/spell_cast_per_legal_main_turn | 27.7 (303) | 23.9 (310) | -3.9 | [-8.7, +1.4] |

### earth_c75 vs u8223 — argmax — free_draft|ALL

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 35.7 (199) | 43.1 (202) | +7.4 | [-1.0, +15.6] |
| block/blocker_survives | 44.2 (199) | 38.1 (202) | -6.1 | [-14.8, +2.1] |
| block/favorable_survive_or_kill | 53.8 (199) | 52.5 (202) | -1.3 | [-10.6, +7.8] |
| draft/max_wjaccard_to_curated | 25.4 (96) | 18.9 (96) | **-6.5** | [-8.4, -4.6] |
| draft/mean_cost | 3.36 (4800) | 3.48 (4800) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (4800) | 60.0 (4800) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 15.0 (4800) | 6.0 (4800) | **-9.0** | [-9.3, -8.7] |
| draft/unique_cards | 17.00 (96) | 19.00 (96) | **+2.0** | [+2.0, +2.0] |
| draft/weapon_share | 10.0 (4800) | 0.0 (4800) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 89.1 (55) | 84.4 (32) | -4.7 | [-17.3, +8.2] |
| earth/quicksand_cast_per_legal_turn | 39.3 (140) | 30.5 (105) | -8.8 | [-19.9, +3.6] |
| gate/Stonehaven/grant_then_block | 40.9 (364) | 44.0 (459) | +3.1 | [-4.5, +10.4] |
| gate/Stonehaven/portal_then_grant | 88.1 (413) | 87.3 (526) | -0.9 | [-5.9, +4.1] |
| gate/portal_per_legal_turn | 99.8 (414) | 95.1 (553) | **-4.6** | [-6.6, -2.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 20.7 (213) | 6.1 (131) | **-14.6** | [-21.5, -7.2] |
| ikz/held_at_end_of_own_turn | 27.0 (745) | 20.8 (769) | **-6.2** | [-10.5, -2.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (201) | 0.0 (160) | +0.0 | [+0.0, +0.0] |
| leader/Bobu/use_then_heal_observed | 18.8 (16) | – (0) | – | |
| leader/Goro/target_then_attacks_same_turn | 7.1 (28) | 11.1 (9) | +4.0 | [-15.6, +20.8] |
| leader/Goro/use_with_target | 100.0 (28) | 100.0 (9) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 6.4 (684) | 1.2 (733) | **-5.2** | [-7.5, -2.8] |
| outcome/own_turns | 7.76 (96) | 8.01 (96) | +0.2 | [-0.1, +0.6] |
| outcome/win | 57.3 (96) | 71.9 (96) | **+14.6** | [+2.1, +27.1] |
| outcome/win_vs_EARTH | 50.0 (24) | 87.5 (24) | **+37.5** | [+16.7, +59.4] |
| outcome/win_vs_FIRE | 50.0 (24) | 50.0 (24) | +0.0 | [-25.0, +22.7] |
| outcome/win_vs_LIGHTNING | 58.3 (24) | 66.7 (24) | +8.3 | [-18.8, +35.0] |
| outcome/win_vs_WATER | 70.8 (24) | 83.3 (24) | +12.5 | [-10.0, +33.3] |
| response/any_nonblock_response_when_legal | – (0) | 95.7 (23) | – | |
| response/defender_declared_when_legal | 77.4 (257) | 54.0 (374) | **-23.4** | [-30.9, -15.6] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-13.9, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-13.9, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 79.5 (88) | 80.0 (95) | +0.5 | [-11.0, +11.6] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 55.7 (88) | 61.1 (95) | +5.4 | [-8.0, +17.6] |
| strategy/attacks_by_equipped_attacker | 6.6 (1119) | 0.0 (1360) | **-6.6** | [-8.0, -5.3] |
| strategy/face_target_share_when_both_legal | 52.5 (591) | 32.6 (705) | **-19.8** | [-26.2, -12.6] |
| strategy/favorable_trade_taken_per_available_turn | 55.4 (242) | 75.1 (305) | **+19.7** | [+13.3, +26.5] |
| strategy/spell_cast_per_legal_main_turn | 30.6 (340) | 19.8 (222) | **-10.8** | [-18.4, -2.7] |

### earth_c75 vs u8223 — argmax — free_draft|Stonehaven/Bobu

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 36.8 (95) | 45.6 (103) | +8.8 | [-3.5, +20.5] |
| block/blocker_survives | 45.3 (95) | 38.8 (103) | -6.4 | [-18.3, +3.8] |
| block/favorable_survive_or_kill | 51.6 (95) | 54.4 (103) | +2.8 | [-10.1, +14.7] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 18.3 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 3.36 (2400) | 3.48 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (2400) | 60.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 16.0 (2400) | 6.0 (2400) | **-10.0** | [-10.0, -10.0] |
| draft/unique_cards | 17.00 (48) | 19.00 (48) | **+2.0** | [+2.0, +2.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 91.3 (23) | 69.2 (13) | -22.1 | [-42.0, +4.8] |
| earth/quicksand_cast_per_legal_turn | 37.7 (61) | 31.7 (41) | -6.0 | [-24.8, +15.0] |
| gate/Stonehaven/grant_then_block | 38.0 (184) | 42.7 (241) | +4.7 | [-5.0, +15.0] |
| gate/Stonehaven/portal_then_grant | 90.6 (203) | 88.6 (272) | -2.0 | [-8.2, +5.0] |
| gate/portal_per_legal_turn | 99.5 (204) | 95.4 (285) | **-4.1** | [-6.7, -1.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 15.9 (107) | 2.9 (70) | **-13.0** | [-19.8, -6.3] |
| ikz/held_at_end_of_own_turn | 28.9 (367) | 18.6 (387) | **-10.3** | [-16.3, -4.2] |
| ikz/held_then_spent_in_opp_turn | 0.0 (106) | 0.0 (72) | +0.0 | [+0.0, +0.0] |
| leader/Bobu/use_then_heal_observed | 18.8 (16) | – (0) | – | |
| leader/use_per_legal_turn | 4.4 (367) | 0.0 (387) | **-4.4** | [-6.6, -2.2] |
| outcome/own_turns | 7.65 (48) | 8.06 (48) | +0.4 | [-0.1, +0.9] |
| outcome/win | 60.4 (48) | 72.9 (48) | +12.5 | [-6.2, +29.2] |
| response/any_nonblock_response_when_legal | – (0) | 100.0 (10) | – | |
| response/defender_declared_when_legal | 80.5 (118) | 55.1 (187) | **-25.4** | [-35.1, -15.4] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-12.5, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-12.5, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 84.1 (44) | 85.4 (48) | +1.3 | [-16.1, +18.8] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 59.1 (44) | 66.7 (48) | +7.6 | [-10.6, +26.0] |
| strategy/attacks_by_equipped_attacker | 6.7 (578) | 0.0 (718) | **-6.7** | [-8.4, -5.4] |
| strategy/face_target_share_when_both_legal | 47.4 (304) | 28.7 (376) | **-18.6** | [-27.6, -9.9] |
| strategy/favorable_trade_taken_per_available_turn | 61.2 (116) | 81.9 (155) | **+20.7** | [+10.7, +31.9] |
| strategy/spell_cast_per_legal_main_turn | 26.2 (164) | 15.5 (103) | -10.7 | [-20.0, +1.1] |

### earth_c75 vs u8223 — argmax — free_draft|Stonehaven/Goro

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 34.6 (104) | 40.4 (99) | +5.8 | [-5.7, +16.2] |
| block/blocker_survives | 43.3 (104) | 37.4 (99) | -5.9 | [-19.0, +6.8] |
| block/favorable_survive_or_kill | 55.8 (104) | 50.5 (99) | -5.3 | [-18.6, +7.6] |
| draft/max_wjaccard_to_curated | 32.5 (48) | 19.6 (48) | **-13.0** | [-13.0, -13.0] |
| draft/mean_cost | 3.36 (2400) | 3.48 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (2400) | 60.0 (2400) | **+2.0** | [+2.0, +2.0] |
| draft/spell_share | 14.0 (2400) | 6.0 (2400) | **-8.0** | [-8.0, -8.0] |
| draft/unique_cards | 17.00 (48) | 19.00 (48) | **+2.0** | [+2.0, +2.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 87.5 (32) | 94.7 (19) | +7.2 | [-2.5, +19.4] |
| earth/quicksand_cast_per_legal_turn | 40.5 (79) | 29.7 (64) | -10.8 | [-25.8, +3.8] |
| gate/Stonehaven/grant_then_block | 43.9 (180) | 45.4 (218) | +1.5 | [-10.8, +13.2] |
| gate/Stonehaven/portal_then_grant | 85.7 (210) | 85.8 (254) | +0.1 | [-6.7, +8.0] |
| gate/portal_per_legal_turn | 100.0 (210) | 94.8 (268) | **-5.2** | [-8.0, -2.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.5 (106) | 9.8 (61) | -15.6 | [-29.7, +1.8] |
| ikz/held_at_end_of_own_turn | 25.1 (378) | 23.0 (382) | -2.1 | [-7.1, +3.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (95) | 0.0 (88) | +0.0 | [+0.0, +0.0] |
| leader/Goro/target_then_attacks_same_turn | 7.1 (28) | 11.1 (9) | +4.0 | [-16.0, +20.2] |
| leader/Goro/use_with_target | 100.0 (28) | 100.0 (9) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.8 (317) | 2.6 (346) | **-6.2** | [-10.3, -2.3] |
| outcome/own_turns | 7.88 (48) | 7.96 (48) | +0.1 | [-0.4, +0.5] |
| outcome/win | 54.2 (48) | 70.8 (48) | +16.7 | [+0.0, +33.3] |
| response/any_nonblock_response_when_legal | – (0) | 92.3 (13) | – | |
| response/defender_declared_when_legal | 74.8 (139) | 52.9 (187) | **-21.9** | [-34.4, -11.3] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 75.0 (44) | 74.5 (47) | -0.5 | [-16.4, +14.0] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 52.3 (44) | 55.3 (47) | +3.0 | [-16.7, +20.9] |
| strategy/attacks_by_equipped_attacker | 6.5 (541) | 0.0 (642) | **-6.5** | [-9.0, -4.4] |
| strategy/face_target_share_when_both_legal | 57.8 (287) | 37.1 (329) | **-20.8** | [-30.7, -9.1] |
| strategy/favorable_trade_taken_per_available_turn | 50.0 (126) | 68.0 (150) | **+18.0** | [+9.8, +25.1] |
| strategy/spell_cast_per_legal_main_turn | 34.7 (176) | 23.5 (119) | **-11.1** | [-22.5, -0.5] |

### earth_c75 vs u8223 — sample — all|ALL

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 31.2 (475) | 32.6 (688) | +1.4 | [-4.2, +7.1] |
| block/blocker_survives | 39.2 (475) | 30.7 (688) | **-8.5** | [-14.2, -3.1] |
| block/favorable_survive_or_kill | 50.7 (475) | 46.5 (688) | -4.2 | [-9.8, +1.5] |
| draft/max_wjaccard_to_curated | 12.2 (96) | 13.5 (96) | **+1.4** | [+0.3, +2.5] |
| draft/mean_cost | 2.98 (4800) | 3.19 (4800) | **+0.2** | [+0.2, +0.3] |
| draft/normal_share | 77.0 (4800) | 63.7 (4800) | **-13.2** | [-14.8, -11.7] |
| draft/spell_share | 9.3 (4800) | 10.1 (4800) | **+0.8** | [+0.0, +1.6] |
| draft/unique_cards | 33.14 (96) | 35.29 (96) | **+2.2** | [+1.6, +2.7] |
| draft/weapon_share | 10.5 (4800) | 2.8 (4800) | **-7.6** | [-8.3, -6.9] |
| earth/quicksand_cast_multi_removal | 82.8 (134) | 87.6 (129) | +4.8 | [-2.1, +11.1] |
| earth/quicksand_cast_per_legal_turn | 53.6 (250) | 44.5 (290) | **-9.1** | [-16.8, -1.8] |
| gate/Stonehaven/grant_then_block | 38.9 (906) | 48.0 (1186) | **+9.1** | [+5.2, +13.1] |
| gate/Stonehaven/portal_then_grant | 69.9 (1297) | 82.4 (1439) | **+12.6** | [+9.5, +15.8] |
| gate/portal_per_legal_turn | 99.2 (1308) | 97.7 (1473) | **-1.5** | [-2.4, -0.6] |
| heal/heal_spell_per_legal_turn_below_max_hp | 31.7 (625) | 24.9 (838) | **-6.7** | [-10.2, -3.2] |
| ikz/held_at_end_of_own_turn | 30.7 (2263) | 34.7 (2426) | **+4.0** | [+2.0, +6.0] |
| ikz/held_then_spent_in_opp_turn | 28.6 (695) | 27.5 (843) | -1.1 | [-4.1, +1.7] |
| leader/Bobu/earth_loss_before_next_turn | 47.4 (19) | 28.6 (7) | -18.8 | [-60.7, +33.3] |
| leader/Bobu/use_then_heal_observed | 12.2 (41) | 16.7 (12) | +4.5 | [-18.8, +33.7] |
| leader/Goro/target_then_attacks_same_turn | 15.6 (64) | 6.4 (78) | -9.2 | [-21.4, +1.0] |
| leader/Goro/use_with_target | 100.0 (64) | 100.0 (78) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 4.8 (2165) | 3.8 (2341) | -1.0 | [-2.2, +0.2] |
| outcome/own_turns | 7.86 (288) | 8.42 (288) | **+0.6** | [+0.4, +0.8] |
| outcome/win | 48.6 (288) | 60.8 (288) | **+12.2** | [+6.9, +17.4] |
| outcome/win_vs_EARTH | 48.6 (72) | 58.3 (72) | +9.7 | [-1.1, +21.4] |
| outcome/win_vs_FIRE | 26.4 (72) | 44.4 (72) | **+18.1** | [+7.1, +28.8] |
| outcome/win_vs_LIGHTNING | 54.2 (72) | 63.9 (72) | +9.7 | [+0.0, +19.7] |
| outcome/win_vs_WATER | 65.3 (72) | 76.4 (72) | +11.1 | [+0.0, +22.4] |
| response/any_nonblock_response_when_legal | 58.6 (503) | 63.2 (778) | +4.6 | [-2.7, +11.7] |
| response/defender_declared_when_legal | 50.6 (937) | 40.0 (1718) | **-10.5** | [-15.7, -5.7] |
| response/spell_played_when_legal | 64.4 (317) | 61.3 (395) | -3.1 | [-13.0, +6.5] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.0 (176) | 1.1 (176) | -2.8 | [-6.2, +0.5] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (176) | 1.1 (176) | -1.1 | [-4.0, +1.6] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 62.8 (253) | 65.7 (280) | +2.9 | [-5.0, +10.6] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.7 (253) | 43.2 (280) | +2.5 | [-4.5, +9.6] |
| strategy/attacks_by_equipped_attacker | 3.5 (2864) | 1.8 (2880) | **-1.6** | [-2.4, -0.9] |
| strategy/face_target_share_when_both_legal | 50.1 (1665) | 42.6 (1609) | **-7.5** | [-12.2, -2.6] |
| strategy/favorable_trade_taken_per_available_turn | 55.0 (696) | 68.4 (734) | **+13.4** | [+7.9, +18.4] |
| strategy/spell_cast_per_legal_main_turn | 26.7 (1338) | 24.0 (1516) | **-2.7** | [-5.1, -0.3] |

### earth_c75 vs u8223 — sample — all|Stonehaven/Bobu

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 26.9 (334) | 27.0 (455) | +0.1 | [-6.8, +6.6] |
| block/blocker_survives | 34.1 (334) | 26.2 (455) | **-8.0** | [-14.6, -1.6] |
| block/favorable_survive_or_kill | 46.4 (334) | 40.4 (455) | -6.0 | [-12.7, +0.7] |
| draft/max_wjaccard_to_curated | 9.0 (48) | 13.3 (48) | **+4.3** | [+3.2, +5.5] |
| draft/mean_cost | 2.99 (2400) | 3.19 (2400) | **+0.2** | [+0.1, +0.3] |
| draft/normal_share | 76.5 (2400) | 63.6 (2400) | **-12.9** | [-15.2, -10.8] |
| draft/spell_share | 8.9 (2400) | 9.6 (2400) | +0.7 | [-0.5, +1.8] |
| draft/unique_cards | 32.90 (48) | 35.12 (48) | **+2.2** | [+1.6, +2.9] |
| draft/weapon_share | 10.5 (2400) | 2.7 (2400) | **-7.8** | [-8.8, -6.8] |
| earth/quicksand_cast_multi_removal | 86.7 (90) | 89.0 (91) | +2.3 | [-5.6, +10.3] |
| earth/quicksand_cast_per_legal_turn | 51.4 (175) | 43.5 (209) | -7.9 | [-16.5, +0.7] |
| gate/Stonehaven/grant_then_block | 39.4 (587) | 45.5 (776) | **+6.1** | [+1.4, +10.6] |
| gate/Stonehaven/portal_then_grant | 69.1 (850) | 80.9 (959) | **+11.9** | [+7.9, +16.1] |
| gate/portal_per_legal_turn | 98.8 (860) | 97.3 (986) | **-1.6** | [-2.8, -0.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.8 (417) | 25.9 (532) | **-7.9** | [-11.7, -3.9] |
| ikz/held_at_end_of_own_turn | 34.4 (1448) | 38.8 (1552) | **+4.4** | [+2.0, +6.7] |
| ikz/held_then_spent_in_opp_turn | 31.5 (498) | 29.4 (602) | -2.1 | [-5.7, +1.5] |
| leader/Bobu/earth_loss_before_next_turn | 47.4 (19) | 28.6 (7) | -18.8 | [-62.5, +34.0] |
| leader/Bobu/use_then_heal_observed | 12.2 (41) | 16.7 (12) | +4.5 | [-18.9, +33.1] |
| leader/use_per_legal_turn | 2.8 (1448) | 0.8 (1552) | **-2.1** | [-2.9, -1.2] |
| outcome/own_turns | 8.23 (176) | 8.82 (176) | **+0.6** | [+0.3, +0.9] |
| outcome/win | 49.4 (176) | 65.3 (176) | **+15.9** | [+8.5, +22.7] |
| response/any_nonblock_response_when_legal | 56.8 (377) | 60.3 (557) | +3.6 | [-4.6, +11.7] |
| response/defender_declared_when_legal | 47.8 (696) | 37.8 (1205) | **-10.1** | [-16.1, -4.4] |
| response/spell_played_when_legal | 68.1 (235) | 60.7 (308) | -7.4 | [-18.3, +3.5] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.0 (176) | 1.1 (176) | -2.8 | [-6.2, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (176) | 1.1 (176) | -1.1 | [-4.0, +1.7] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 61.0 (159) | 66.5 (173) | +5.5 | [-4.0, +14.1] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.9 (159) | 41.6 (173) | +0.7 | [-9.2, +10.1] |
| strategy/attacks_by_equipped_attacker | 1.9 (1656) | 0.4 (1683) | **-1.5** | [-2.4, -0.7] |
| strategy/face_target_share_when_both_legal | 46.5 (946) | 38.6 (905) | **-7.9** | [-14.4, -1.5] |
| strategy/favorable_trade_taken_per_available_turn | 56.5 (395) | 74.8 (412) | **+18.3** | [+10.7, +25.2] |
| strategy/spell_cast_per_legal_main_turn | 27.3 (911) | 24.5 (1024) | **-2.8** | [-5.5, -0.3] |

### earth_c75 vs u8223 — sample — all|Stonehaven/Goro

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 41.1 (141) | 43.3 (233) | +2.2 | [-7.6, +12.6] |
| block/blocker_survives | 51.1 (141) | 39.5 (233) | **-11.6** | [-21.6, -1.1] |
| block/favorable_survive_or_kill | 61.0 (141) | 58.4 (233) | -2.6 | [-12.3, +8.2] |
| draft/max_wjaccard_to_curated | 15.4 (48) | 13.8 (48) | **-1.6** | [-2.5, -0.8] |
| draft/mean_cost | 2.98 (2400) | 3.19 (2400) | **+0.2** | [+0.2, +0.3] |
| draft/normal_share | 77.4 (2400) | 63.9 (2400) | **-13.5** | [-15.7, -11.5] |
| draft/spell_share | 9.6 (2400) | 10.5 (2400) | +0.9 | [-0.1, +2.0] |
| draft/unique_cards | 33.38 (48) | 35.46 (48) | **+2.1** | [+1.1, +2.9] |
| draft/weapon_share | 10.4 (2400) | 3.0 (2400) | **-7.5** | [-8.5, -6.5] |
| earth/quicksand_cast_multi_removal | 75.0 (44) | 84.2 (38) | +9.2 | [-2.7, +20.7] |
| earth/quicksand_cast_per_legal_turn | 58.7 (75) | 46.9 (81) | -11.8 | [-27.9, +2.6] |
| gate/Stonehaven/grant_then_block | 37.9 (319) | 52.7 (410) | **+14.8** | [+8.1, +21.8] |
| gate/Stonehaven/portal_then_grant | 71.4 (447) | 85.4 (480) | **+14.1** | [+8.7, +19.1] |
| gate/portal_per_legal_turn | 99.8 (448) | 98.6 (487) | -1.2 | [-2.5, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.4 (208) | 23.2 (306) | -4.2 | [-11.5, +2.1] |
| ikz/held_at_end_of_own_turn | 24.2 (815) | 27.6 (874) | +3.4 | [-0.2, +7.3] |
| ikz/held_then_spent_in_opp_turn | 21.3 (197) | 22.8 (241) | +1.5 | [-2.8, +5.9] |
| leader/Goro/target_then_attacks_same_turn | 15.6 (64) | 6.4 (78) | -9.2 | [-21.0, +1.2] |
| leader/Goro/use_with_target | 100.0 (64) | 100.0 (78) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.9 (717) | 9.9 (789) | +1.0 | [-2.0, +3.8] |
| outcome/own_turns | 7.28 (112) | 7.80 (112) | **+0.5** | [+0.2, +0.9] |
| outcome/win | 47.3 (112) | 53.6 (112) | +6.2 | [-1.8, +14.3] |
| response/any_nonblock_response_when_legal | 64.3 (126) | 70.6 (221) | +6.3 | [-10.9, +23.0] |
| response/defender_declared_when_legal | 58.5 (241) | 45.4 (513) | **-13.1** | [-22.0, -4.4] |
| response/spell_played_when_legal | 53.7 (82) | 63.2 (87) | +9.6 | [-14.4, +27.6] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.0 (94) | 64.5 (107) | -1.5 | [-14.4, +11.6] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.4 (94) | 45.8 (107) | +5.4 | [-5.7, +17.0] |
| strategy/attacks_by_equipped_attacker | 5.5 (1208) | 3.8 (1197) | **-1.7** | [-3.2, -0.3] |
| strategy/face_target_share_when_both_legal | 54.8 (719) | 47.7 (704) | **-7.1** | [-14.5, -0.1] |
| strategy/favorable_trade_taken_per_available_turn | 53.2 (301) | 60.2 (322) | +7.1 | [-0.6, +14.4] |
| strategy/spell_cast_per_legal_main_turn | 25.3 (427) | 23.0 (492) | -2.3 | [-7.3, +2.7] |

### earth_c75 vs u8223 — sample — fixed|ALL

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 28.4 (349) | 31.9 (480) | +3.5 | [-3.3, +9.5] |
| block/blocker_survives | 40.4 (349) | 31.7 (480) | **-8.7** | [-14.9, -2.1] |
| block/favorable_survive_or_kill | 50.7 (349) | 46.2 (480) | -4.5 | [-10.8, +2.0] |
| earth/quicksand_cast_multi_removal | 85.1 (114) | 87.8 (115) | +2.7 | [-4.7, +9.7] |
| earth/quicksand_cast_per_legal_turn | 53.3 (214) | 44.6 (258) | -8.7 | [-16.8, +0.3] |
| gate/Stonehaven/grant_then_block | 40.5 (650) | 49.8 (804) | **+9.3** | [+4.8, +13.8] |
| gate/Stonehaven/portal_then_grant | 69.7 (932) | 81.1 (991) | **+11.4** | [+7.6, +15.5] |
| gate/portal_per_legal_turn | 99.4 (938) | 97.7 (1014) | **-1.6** | [-2.8, -0.6] |
| heal/heal_spell_per_legal_turn_below_max_hp | 31.9 (540) | 25.9 (649) | **-6.0** | [-9.7, -2.2] |
| ikz/held_at_end_of_own_turn | 36.3 (1549) | 40.4 (1654) | **+4.1** | [+1.7, +6.7] |
| ikz/held_then_spent_in_opp_turn | 35.1 (562) | 33.2 (668) | -1.8 | [-5.3, +1.6] |
| leader/Bobu/earth_loss_before_next_turn | 46.2 (13) | 40.0 (5) | -6.2 | [-66.7, +58.3] |
| leader/Bobu/use_then_heal_observed | 12.9 (31) | 33.3 (6) | +20.4 | [-21.4, +67.9] |
| leader/Goro/target_then_attacks_same_turn | 21.1 (38) | 5.7 (53) | **-15.4** | [-32.7, -2.1] |
| leader/Goro/use_with_target | 100.0 (38) | 100.0 (53) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 4.6 (1497) | 3.7 (1600) | -0.9 | [-2.4, +0.6] |
| outcome/own_turns | 8.07 (192) | 8.61 (192) | **+0.5** | [+0.3, +0.8] |
| outcome/win | 52.1 (192) | 65.1 (192) | **+13.0** | [+6.8, +19.3] |
| outcome/win_vs_EARTH | 60.4 (48) | 68.8 (48) | +8.3 | [-3.2, +19.4] |
| outcome/win_vs_FIRE | 25.0 (48) | 47.9 (48) | **+22.9** | [+8.3, +37.5] |
| outcome/win_vs_LIGHTNING | 56.2 (48) | 66.7 (48) | +10.4 | [+0.0, +22.5] |
| outcome/win_vs_WATER | 66.7 (48) | 77.1 (48) | +10.4 | [-2.0, +21.7] |
| response/any_nonblock_response_when_legal | 58.3 (499) | 61.7 (687) | +3.4 | [-4.2, +10.8] |
| response/defender_declared_when_legal | 48.9 (711) | 40.0 (1201) | **-9.0** | [-14.8, -3.2] |
| response/spell_played_when_legal | 64.1 (315) | 61.1 (380) | -3.1 | [-13.7, +7.1] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 3.9 (128) | 1.6 (128) | -2.3 | [-6.2, +1.5] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (128) | 1.6 (128) | -0.8 | [-4.2, +2.5] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 63.8 (177) | 68.1 (185) | +4.3 | [-4.0, +13.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.1 (177) | 44.3 (185) | +4.2 | [-3.6, +13.0] |
| strategy/attacks_by_equipped_attacker | 2.1 (1672) | 2.1 (1733) | -0.0 | [-0.7, +0.6] |
| strategy/face_target_share_when_both_legal | 50.8 (928) | 47.0 (873) | -3.8 | [-9.5, +1.9] |
| strategy/favorable_trade_taken_per_available_turn | 52.6 (424) | 66.3 (430) | **+13.7** | [+7.9, +19.7] |
| strategy/spell_cast_per_legal_main_turn | 27.4 (1105) | 25.1 (1200) | -2.3 | [-5.0, +0.2] |

### earth_c75 vs u8223 — sample — fixed|Stonehaven/Bobu

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 21.1 (265) | 25.1 (358) | +4.0 | [-3.2, +11.3] |
| block/blocker_survives | 34.7 (265) | 26.0 (358) | **-8.7** | [-16.2, -1.2] |
| block/favorable_survive_or_kill | 44.2 (265) | 39.4 (358) | -4.8 | [-11.9, +2.6] |
| earth/quicksand_cast_multi_removal | 87.7 (81) | 88.4 (86) | +0.7 | [-7.7, +9.0] |
| earth/quicksand_cast_per_legal_turn | 50.9 (159) | 42.8 (201) | -8.2 | [-16.8, +1.1] |
| gate/Stonehaven/grant_then_block | 39.2 (457) | 46.9 (593) | **+7.7** | [+2.3, +12.8] |
| gate/Stonehaven/portal_then_grant | 67.9 (673) | 79.8 (743) | **+11.9** | [+7.2, +17.0] |
| gate/portal_per_legal_turn | 99.1 (679) | 97.1 (765) | **-2.0** | [-3.4, -0.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.9 (381) | 25.9 (471) | **-8.0** | [-12.0, -4.0] |
| ikz/held_at_end_of_own_turn | 39.2 (1086) | 43.8 (1163) | **+4.5** | [+1.8, +7.4] |
| ikz/held_then_spent_in_opp_turn | 36.4 (426) | 33.4 (509) | -3.0 | [-7.0, +0.9] |
| leader/Bobu/earth_loss_before_next_turn | 46.2 (13) | 40.0 (5) | -6.2 | [-66.7, +55.6] |
| leader/Bobu/use_then_heal_observed | 12.9 (31) | 33.3 (6) | +20.4 | [-20.0, +68.2] |
| leader/use_per_legal_turn | 2.9 (1086) | 0.5 (1163) | **-2.3** | [-3.4, -1.4] |
| outcome/own_turns | 8.48 (128) | 9.09 (128) | **+0.6** | [+0.3, +0.9] |
| outcome/win | 51.6 (128) | 68.0 (128) | **+16.4** | [+8.6, +24.2] |
| response/any_nonblock_response_when_legal | 56.3 (373) | 58.4 (503) | +2.1 | [-6.2, +10.4] |
| response/defender_declared_when_legal | 46.3 (570) | 36.8 (972) | **-9.5** | [-16.0, -3.4] |
| response/spell_played_when_legal | 67.8 (233) | 60.8 (296) | -7.0 | [-18.2, +3.8] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 3.9 (128) | 1.6 (128) | -2.3 | [-6.2, +1.6] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (128) | 1.6 (128) | -0.8 | [-3.9, +2.3] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.2 (123) | 67.5 (126) | +7.3 | [-3.8, +17.9] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 36.6 (123) | 43.7 (126) | +7.1 | [-3.7, +17.4] |
| strategy/attacks_by_equipped_attacker | 0.0 (1039) | 0.0 (1082) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 44.4 (565) | 40.1 (521) | -4.3 | [-12.1, +3.1] |
| strategy/favorable_trade_taken_per_available_turn | 56.0 (259) | 74.7 (265) | **+18.7** | [+10.8, +26.9] |
| strategy/spell_cast_per_legal_main_turn | 27.6 (800) | 25.3 (878) | -2.3 | [-5.0, +0.3] |

### earth_c75 vs u8223 — sample — fixed|Stonehaven/Goro

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 51.2 (84) | 51.6 (122) | +0.4 | [-12.4, +12.4] |
| block/blocker_survives | 58.3 (84) | 48.4 (122) | -10.0 | [-22.9, +2.1] |
| block/favorable_survive_or_kill | 71.4 (84) | 66.4 (122) | -5.0 | [-16.1, +5.2] |
| earth/quicksand_cast_multi_removal | 78.8 (33) | 86.2 (29) | +7.4 | [-2.4, +18.5] |
| earth/quicksand_cast_per_legal_turn | 60.0 (55) | 50.9 (57) | -9.1 | [-26.1, +8.3] |
| gate/Stonehaven/grant_then_block | 43.5 (193) | 57.8 (211) | **+14.3** | [+5.8, +23.2] |
| gate/Stonehaven/portal_then_grant | 74.5 (259) | 85.1 (248) | **+10.6** | [+4.8, +17.0] |
| gate/portal_per_legal_turn | 100.0 (259) | 99.6 (249) | -0.4 | [-1.3, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.0 (159) | 25.8 (178) | -1.2 | [-8.6, +6.7] |
| ikz/held_at_end_of_own_turn | 29.4 (463) | 32.4 (491) | +3.0 | [-2.4, +8.3] |
| ikz/held_then_spent_in_opp_turn | 30.9 (136) | 32.7 (159) | +1.8 | [-4.6, +8.5] |
| leader/Goro/target_then_attacks_same_turn | 21.1 (38) | 5.7 (53) | **-15.4** | [-32.3, -1.5] |
| leader/Goro/use_with_target | 100.0 (38) | 100.0 (53) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.2 (411) | 12.1 (437) | +2.9 | [-1.4, +7.0] |
| outcome/own_turns | 7.23 (64) | 7.67 (64) | +0.4 | [+0.0, +1.0] |
| outcome/win | 53.1 (64) | 59.4 (64) | +6.2 | [-3.1, +15.6] |
| response/any_nonblock_response_when_legal | 64.3 (126) | 70.7 (184) | +6.4 | [-11.3, +23.8] |
| response/defender_declared_when_legal | 59.6 (141) | 53.3 (229) | -6.3 | [-17.3, +3.5] |
| response/spell_played_when_legal | 53.7 (82) | 61.9 (84) | +8.2 | [-16.1, +26.7] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 72.2 (54) | 69.5 (59) | -2.7 | [-17.3, +12.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 48.1 (54) | 45.8 (59) | -2.4 | [-16.1, +11.3] |
| strategy/attacks_by_equipped_attacker | 5.5 (633) | 5.5 (651) | +0.0 | [-1.8, +1.7] |
| strategy/face_target_share_when_both_legal | 60.6 (363) | 57.1 (352) | -3.5 | [-11.8, +5.3] |
| strategy/favorable_trade_taken_per_available_turn | 47.3 (165) | 52.7 (165) | +5.5 | [-3.8, +14.0] |
| strategy/spell_cast_per_legal_main_turn | 26.9 (305) | 24.5 (322) | -2.4 | [-7.4, +3.2] |

### earth_c75 vs u8223 — sample — free_draft|ALL

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 38.9 (126) | 34.1 (208) | -4.8 | [-16.2, +7.8] |
| block/blocker_survives | 35.7 (126) | 28.4 (208) | -7.3 | [-18.3, +3.4] |
| block/favorable_survive_or_kill | 50.8 (126) | 47.1 (208) | -3.7 | [-15.9, +8.9] |
| draft/max_wjaccard_to_curated | 12.2 (96) | 13.5 (96) | **+1.4** | [+0.3, +2.5] |
| draft/mean_cost | 2.98 (4800) | 3.19 (4800) | **+0.2** | [+0.2, +0.3] |
| draft/normal_share | 77.0 (4800) | 63.7 (4800) | **-13.2** | [-14.8, -11.8] |
| draft/spell_share | 9.3 (4800) | 10.1 (4800) | **+0.8** | [+0.0, +1.6] |
| draft/unique_cards | 33.14 (96) | 35.29 (96) | **+2.2** | [+1.6, +2.7] |
| draft/weapon_share | 10.5 (4800) | 2.8 (4800) | **-7.6** | [-8.3, -7.0] |
| earth/quicksand_cast_multi_removal | 70.0 (20) | 85.7 (14) | +15.7 | [-10.0, +34.5] |
| earth/quicksand_cast_per_legal_turn | 55.6 (36) | 43.8 (32) | -11.8 | [-38.3, +9.2] |
| gate/Stonehaven/grant_then_block | 34.8 (256) | 44.2 (382) | **+9.5** | [+1.2, +17.2] |
| gate/Stonehaven/portal_then_grant | 70.1 (365) | 85.3 (448) | **+15.1** | [+9.7, +21.0] |
| gate/portal_per_legal_turn | 98.6 (370) | 97.6 (459) | -1.0 | [-3.0, +0.8] |
| heal/heal_spell_per_legal_turn_below_max_hp | 30.6 (85) | 21.7 (189) | -8.9 | [-20.2, +0.3] |
| ikz/held_at_end_of_own_turn | 18.6 (714) | 22.7 (772) | **+4.0** | [+0.3, +7.8] |
| ikz/held_then_spent_in_opp_turn | 1.5 (133) | 5.7 (175) | **+4.2** | [+0.8, +8.4] |
| leader/Bobu/use_then_heal_observed | 10.0 (10) | 0.0 (6) | -10.0 | [-50.0, +0.0] |
| leader/Goro/target_then_attacks_same_turn | 7.7 (26) | 8.0 (25) | +0.3 | [-17.0, +15.8] |
| leader/Goro/use_with_target | 100.0 (26) | 100.0 (25) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.4 (668) | 4.2 (741) | -1.2 | [-2.9, +0.5] |
| outcome/own_turns | 7.44 (96) | 8.04 (96) | **+0.6** | [+0.3, +0.9] |
| outcome/win | 41.7 (96) | 52.1 (96) | +10.4 | [+0.0, +20.8] |
| outcome/win_vs_EARTH | 25.0 (24) | 37.5 (24) | +12.5 | [-11.1, +38.5] |
| outcome/win_vs_FIRE | 29.2 (24) | 37.5 (24) | +8.3 | [-8.3, +25.0] |
| outcome/win_vs_LIGHTNING | 50.0 (24) | 58.3 (24) | +8.3 | [-12.5, +28.6] |
| outcome/win_vs_WATER | 62.5 (24) | 75.0 (24) | +12.5 | [-13.6, +37.5] |
| response/any_nonblock_response_when_legal | 100.0 (4) | 74.7 (91) | **-25.3** | [-34.4, -15.4] |
| response/defender_declared_when_legal | 55.8 (226) | 40.2 (517) | **-15.5** | [-26.5, -4.9] |
| response/spell_played_when_legal | 100.0 (2) | 66.7 (15) | **-33.3** | [-45.5, -10.0] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.2 (48) | 0.0 (48) | -4.2 | [-10.9, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.1 (48) | 0.0 (48) | -2.1 | [-7.5, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.5 (76) | 61.1 (95) | +0.5 | [-13.7, +15.1] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 42.1 (76) | 41.1 (95) | -1.1 | [-15.2, +13.5] |
| strategy/attacks_by_equipped_attacker | 5.4 (1192) | 1.5 (1147) | **-3.9** | [-5.2, -2.5] |
| strategy/face_target_share_when_both_legal | 49.3 (737) | 37.4 (736) | **-11.9** | [-19.9, -4.4] |
| strategy/favorable_trade_taken_per_available_turn | 58.8 (272) | 71.4 (304) | **+12.6** | [+3.0, +21.9] |
| strategy/spell_cast_per_legal_main_turn | 23.2 (233) | 19.9 (316) | -3.2 | [-10.4, +3.2] |

### earth_c75 vs u8223 — sample — free_draft|Stonehaven/Bobu

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 49.3 (69) | 34.0 (97) | **-15.3** | [-30.2, -1.2] |
| block/blocker_survives | 31.9 (69) | 26.8 (97) | -5.1 | [-16.6, +6.6] |
| block/favorable_survive_or_kill | 55.1 (69) | 44.3 (97) | -10.7 | [-26.2, +3.5] |
| draft/max_wjaccard_to_curated | 9.0 (48) | 13.3 (48) | **+4.3** | [+3.3, +5.5] |
| draft/mean_cost | 2.99 (2400) | 3.19 (2400) | **+0.2** | [+0.1, +0.3] |
| draft/normal_share | 76.5 (2400) | 63.6 (2400) | **-12.9** | [-15.1, -10.8] |
| draft/spell_share | 8.9 (2400) | 9.6 (2400) | +0.7 | [-0.5, +1.8] |
| draft/unique_cards | 32.90 (48) | 35.12 (48) | **+2.2** | [+1.6, +2.9] |
| draft/weapon_share | 10.5 (2400) | 2.7 (2400) | **-7.8** | [-8.8, -6.8] |
| earth/quicksand_cast_multi_removal | 77.8 (9) | 100.0 (5) | +22.2 | [+0.0, +37.5] |
| earth/quicksand_cast_per_legal_turn | 56.2 (16) | 62.5 (8) | +6.2 | [-14.7, +50.0] |
| gate/Stonehaven/grant_then_block | 40.0 (130) | 41.0 (183) | +1.0 | [-8.7, +9.9] |
| gate/Stonehaven/portal_then_grant | 73.4 (177) | 84.7 (216) | **+11.3** | [+5.3, +17.8] |
| gate/portal_per_legal_turn | 97.8 (181) | 97.7 (221) | -0.1 | [-2.7, +2.5] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.3 (36) | 26.2 (61) | -7.1 | [-25.8, +9.4] |
| ikz/held_at_end_of_own_turn | 19.9 (362) | 23.9 (389) | +4.0 | [-1.4, +9.7] |
| ikz/held_then_spent_in_opp_turn | 2.8 (72) | 7.5 (93) | +4.7 | [-1.0, +11.5] |
| leader/Bobu/use_then_heal_observed | 10.0 (10) | 0.0 (6) | -10.0 | [-40.0, +0.0] |
| leader/use_per_legal_turn | 2.8 (362) | 1.5 (389) | -1.2 | [-2.9, +0.3] |
| outcome/own_turns | 7.54 (48) | 8.10 (48) | **+0.6** | [+0.0, +1.1] |
| outcome/win | 43.8 (48) | 58.3 (48) | +14.6 | [-2.1, +31.2] |
| response/any_nonblock_response_when_legal | 100.0 (4) | 77.8 (54) | **-22.2** | [-34.1, -10.8] |
| response/defender_declared_when_legal | 54.8 (126) | 41.6 (233) | -13.1 | [-30.0, +2.2] |
| response/spell_played_when_legal | 100.0 (2) | 58.3 (12) | **-41.7** | [-50.0, -20.0] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.2 (48) | 0.0 (48) | -4.2 | [-10.4, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.1 (48) | 0.0 (48) | -2.1 | [-6.2, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 63.9 (36) | 63.8 (47) | -0.1 | [-19.5, +20.6] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 55.6 (36) | 36.2 (47) | -19.4 | [-39.6, +2.8] |
| strategy/attacks_by_equipped_attacker | 5.2 (617) | 1.2 (601) | **-4.0** | [-5.7, -2.2] |
| strategy/face_target_share_when_both_legal | 49.6 (381) | 36.5 (384) | **-13.1** | [-24.3, -2.6] |
| strategy/favorable_trade_taken_per_available_turn | 57.4 (136) | 74.8 (147) | **+17.5** | [+3.7, +31.3] |
| strategy/spell_cast_per_legal_main_turn | 25.2 (111) | 19.9 (146) | -5.4 | [-15.0, +2.8] |

### earth_c75 vs u8223 — sample — free_draft|Stonehaven/Goro

| metric | u8223 | earth_c75 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 26.3 (57) | 34.2 (111) | +7.9 | [-7.4, +25.7] |
| block/blocker_survives | 40.4 (57) | 29.7 (111) | -10.6 | [-29.3, +7.3] |
| block/favorable_survive_or_kill | 45.6 (57) | 49.5 (111) | +3.9 | [-15.2, +22.4] |
| draft/max_wjaccard_to_curated | 15.4 (48) | 13.8 (48) | **-1.6** | [-2.4, -0.8] |
| draft/mean_cost | 2.98 (2400) | 3.19 (2400) | **+0.2** | [+0.2, +0.3] |
| draft/normal_share | 77.4 (2400) | 63.9 (2400) | **-13.5** | [-15.7, -11.5] |
| draft/spell_share | 9.6 (2400) | 10.5 (2400) | +0.9 | [-0.1, +1.9] |
| draft/unique_cards | 33.38 (48) | 35.46 (48) | **+2.1** | [+1.2, +2.9] |
| draft/weapon_share | 10.4 (2400) | 3.0 (2400) | **-7.5** | [-8.5, -6.5] |
| earth/quicksand_cast_multi_removal | 63.6 (11) | 77.8 (9) | +14.1 | [-22.7, +44.4] |
| earth/quicksand_cast_per_legal_turn | 55.0 (20) | 37.5 (24) | -17.5 | [-58.5, +9.6] |
| gate/Stonehaven/grant_then_block | 29.4 (126) | 47.2 (199) | **+17.9** | [+6.7, +29.2] |
| gate/Stonehaven/portal_then_grant | 67.0 (188) | 85.8 (232) | **+18.8** | [+10.1, +27.1] |
| gate/portal_per_legal_turn | 99.5 (189) | 97.5 (238) | -2.0 | [-4.7, +0.3] |
| heal/heal_spell_per_legal_turn_below_max_hp | 28.6 (49) | 19.5 (128) | -9.0 | [-25.9, +4.2] |
| ikz/held_at_end_of_own_turn | 17.3 (352) | 21.4 (383) | +4.1 | [-1.2, +9.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (61) | 3.7 (82) | +3.7 | [+0.0, +8.8] |
| leader/Goro/target_then_attacks_same_turn | 7.7 (26) | 8.0 (25) | +0.3 | [-17.1, +15.8] |
| leader/Goro/use_with_target | 100.0 (26) | 100.0 (25) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.5 (306) | 7.1 (352) | -1.4 | [-4.7, +1.9] |
| outcome/own_turns | 7.33 (48) | 7.98 (48) | **+0.6** | [+0.2, +1.1] |
| outcome/win | 39.6 (48) | 45.8 (48) | +6.2 | [-8.3, +18.8] |
| response/any_nonblock_response_when_legal | – (0) | 70.3 (37) | – | |
| response/defender_declared_when_legal | 57.0 (100) | 39.1 (284) | **-17.9** | [-34.3, -3.7] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 57.5 (40) | 58.3 (48) | +0.8 | [-20.7, +21.5] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 30.0 (40) | 45.8 (48) | +15.8 | [-1.5, +32.9] |
| strategy/attacks_by_equipped_attacker | 5.6 (575) | 1.8 (546) | **-3.7** | [-5.8, -1.7] |
| strategy/face_target_share_when_both_legal | 48.9 (356) | 38.4 (352) | -10.5 | [-20.9, +0.7] |
| strategy/favorable_trade_taken_per_available_turn | 60.3 (136) | 68.2 (157) | +7.9 | [-3.9, +19.3] |
| strategy/spell_cast_per_legal_main_turn | 21.3 (122) | 20.0 (170) | -1.3 | [-12.1, +8.2] |

### earth vs u8223 — argmax — all|ALL

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 29.5 (543) | 32.2 (583) | +2.8 | [-2.1, +7.7] |
| block/blocker_survives | 41.6 (543) | 36.2 (583) | **-5.4** | [-10.4, -0.8] |
| block/favorable_survive_or_kill | 50.8 (543) | 47.5 (583) | -3.3 | [-8.2, +1.4] |
| draft/max_wjaccard_to_curated | 25.4 (96) | 19.1 (96) | **-6.3** | [-7.0, -5.6] |
| draft/mean_cost | 3.36 (4800) | 3.42 (4800) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (4800) | 55.0 (4800) | **-3.0** | [-3.3, -2.7] |
| draft/spell_share | 15.0 (4800) | 5.0 (4800) | **-10.0** | [-10.0, -10.0] |
| draft/unique_cards | 17.00 (96) | 20.50 (96) | **+3.5** | [+3.4, +3.6] |
| draft/weapon_share | 10.0 (4800) | 0.0 (4800) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 86.9 (168) | 84.7 (124) | -2.2 | [-9.3, +4.7] |
| earth/quicksand_cast_per_legal_turn | 47.5 (354) | 45.9 (270) | -1.5 | [-7.7, +4.8] |
| gate/Stonehaven/grant_then_block | 40.7 (1009) | 43.9 (1083) | +3.1 | [-0.4, +6.8] |
| gate/Stonehaven/portal_then_grant | 75.2 (1342) | 73.8 (1468) | -1.4 | [-4.6, +1.8] |
| gate/portal_per_legal_turn | 99.6 (1348) | 98.2 (1495) | **-1.4** | [-2.2, -0.6] |
| heal/heal_spell_per_legal_turn_below_max_hp | 28.9 (757) | 27.3 (699) | -1.6 | [-4.6, +1.3] |
| ikz/held_at_end_of_own_turn | 33.1 (2302) | 33.1 (2350) | -0.1 | [-1.9, +1.8] |
| ikz/held_then_spent_in_opp_turn | 25.3 (763) | 26.9 (777) | +1.6 | [-1.2, +4.2] |
| leader/Bobu/earth_loss_before_next_turn | 57.1 (28) | 75.0 (16) | +17.9 | [-10.0, +41.0] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 16.0 (25) | +1.2 | [-13.5, +15.5] |
| leader/Goro/target_then_attacks_same_turn | 13.8 (65) | 9.8 (92) | -4.1 | [-15.0, +5.9] |
| leader/Goro/use_with_target | 100.0 (65) | 100.0 (92) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.4 (2193) | 5.2 (2260) | -0.2 | [-1.9, +1.7] |
| outcome/own_turns | 7.99 (288) | 8.16 (288) | +0.2 | [-0.0, +0.4] |
| outcome/win | 54.9 (288) | 59.0 (288) | +4.2 | [-1.4, +10.1] |
| outcome/win_vs_EARTH | 52.8 (72) | 59.7 (72) | +6.9 | [-5.0, +19.4] |
| outcome/win_vs_FIRE | 40.3 (72) | 41.7 (72) | +1.4 | [-10.0, +12.5] |
| outcome/win_vs_LIGHTNING | 61.1 (72) | 65.3 (72) | +4.2 | [-7.7, +17.1] |
| outcome/win_vs_WATER | 65.3 (72) | 69.4 (72) | +4.2 | [-7.8, +16.1] |
| response/any_nonblock_response_when_legal | 61.2 (454) | 56.6 (567) | -4.6 | [-12.2, +3.1] |
| response/defender_declared_when_legal | 55.2 (982) | 49.0 (1189) | **-6.2** | [-10.3, -1.7] |
| response/spell_played_when_legal | 70.5 (281) | 63.4 (339) | -7.0 | [-15.1, +1.1] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.0 (176) | 5.1 (176) | -2.8 | [-6.9, +1.1] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.7 (176) | 2.3 (176) | -3.4 | [-7.4, +0.5] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.7 (264) | 68.3 (265) | +1.6 | [-5.2, +8.7] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 41.7 (264) | 42.6 (265) | +1.0 | [-6.5, +8.7] |
| strategy/attacks_by_equipped_attacker | 3.8 (2829) | 1.4 (2984) | **-2.4** | [-3.2, -1.6] |
| strategy/face_target_share_when_both_legal | 50.8 (1532) | 43.1 (1643) | **-7.8** | [-12.0, -3.1] |
| strategy/favorable_trade_taken_per_available_turn | 55.1 (670) | 66.3 (725) | **+11.3** | [+6.7, +15.5] |
| strategy/spell_cast_per_legal_main_turn | 28.3 (1453) | 25.1 (1299) | **-3.2** | [-5.2, -1.3] |

### earth vs u8223 — argmax — all|Stonehaven/Bobu

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 23.5 (357) | 26.1 (399) | +2.5 | [-3.3, +8.2] |
| block/blocker_survives | 37.3 (357) | 32.6 (399) | -4.7 | [-10.1, +0.8] |
| block/favorable_survive_or_kill | 45.7 (357) | 43.1 (399) | -2.6 | [-8.0, +2.8] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 14.6 (48) | **-3.7** | [-3.7, -3.7] |
| draft/mean_cost | 3.36 (2400) | 3.40 (2400) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 58.0 (2400) | 54.0 (2400) | **-4.0** | [-4.0, -4.0] |
| draft/spell_share | 16.0 (2400) | 6.0 (2400) | **-10.0** | [-10.0, -10.0] |
| draft/unique_cards | 17.00 (48) | 20.00 (48) | **+3.0** | [+3.0, +3.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 89.2 (102) | 90.1 (81) | +0.9 | [-7.0, +9.6] |
| earth/quicksand_cast_per_legal_turn | 47.0 (217) | 44.0 (184) | -3.0 | [-10.2, +4.5] |
| gate/Stonehaven/grant_then_block | 38.7 (646) | 43.3 (688) | **+4.6** | [+0.5, +8.7] |
| gate/Stonehaven/portal_then_grant | 73.0 (885) | 71.6 (961) | -1.4 | [-5.6, +2.5] |
| gate/portal_per_legal_turn | 99.3 (891) | 97.9 (982) | **-1.5** | [-2.7, -0.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 29.6 (494) | 28.9 (491) | -0.6 | [-4.3, +2.8] |
| ikz/held_at_end_of_own_turn | 36.5 (1470) | 36.4 (1502) | -0.1 | [-2.4, +2.0] |
| ikz/held_then_spent_in_opp_turn | 27.6 (537) | 29.8 (547) | +2.2 | [-1.0, +5.8] |
| leader/Bobu/earth_loss_before_next_turn | 57.1 (28) | 75.0 (16) | +17.9 | [-10.8, +40.1] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 16.0 (25) | +1.2 | [-13.8, +16.7] |
| leader/use_per_legal_turn | 3.7 (1470) | 1.7 (1502) | **-2.0** | [-3.3, -0.7] |
| outcome/own_turns | 8.35 (176) | 8.53 (176) | +0.2 | [-0.1, +0.4] |
| outcome/win | 56.8 (176) | 61.9 (176) | +5.1 | [-2.3, +13.1] |
| response/any_nonblock_response_when_legal | 58.5 (342) | 53.3 (430) | -5.2 | [-14.0, +3.3] |
| response/defender_declared_when_legal | 50.4 (707) | 46.8 (853) | -3.6 | [-8.4, +1.4] |
| response/spell_played_when_legal | 73.9 (203) | 64.1 (262) | **-9.8** | [-17.5, -1.2] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.0 (176) | 5.1 (176) | -2.8 | [-7.4, +1.1] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.7 (176) | 2.3 (176) | -3.4 | [-7.4, +0.6] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 64.5 (166) | 67.3 (168) | +2.8 | [-5.8, +11.5] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 38.6 (166) | 39.9 (168) | +1.3 | [-8.2, +11.0] |
| strategy/attacks_by_equipped_attacker | 2.4 (1659) | 0.0 (1766) | **-2.4** | [-3.3, -1.5] |
| strategy/face_target_share_when_both_legal | 44.0 (877) | 39.9 (938) | -4.1 | [-10.2, +2.3] |
| strategy/favorable_trade_taken_per_available_turn | 59.9 (377) | 69.1 (408) | **+9.2** | [+2.3, +15.6] |
| strategy/spell_cast_per_legal_main_turn | 27.3 (974) | 25.3 (925) | -2.0 | [-4.4, +0.4] |

### earth vs u8223 — argmax — all|Stonehaven/Goro

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 40.9 (186) | 45.7 (184) | +4.8 | [-4.3, +14.2] |
| block/blocker_survives | 50.0 (186) | 44.0 (184) | -6.0 | [-16.0, +3.8] |
| block/favorable_survive_or_kill | 60.8 (186) | 57.1 (184) | -3.7 | [-13.4, +6.1] |
| draft/max_wjaccard_to_curated | 32.5 (48) | 23.6 (48) | **-8.9** | [-8.9, -8.9] |
| draft/mean_cost | 3.36 (2400) | 3.44 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (2400) | 56.0 (2400) | **-2.0** | [-2.0, -2.0] |
| draft/spell_share | 14.0 (2400) | 4.0 (2400) | **-10.0** | [-10.0, -10.0] |
| draft/unique_cards | 17.00 (48) | 21.00 (48) | **+4.0** | [+4.0, +4.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 83.3 (66) | 74.4 (43) | -8.9 | [-21.1, +3.9] |
| earth/quicksand_cast_per_legal_turn | 48.2 (137) | 50.0 (86) | +1.8 | [-10.8, +13.9] |
| gate/Stonehaven/grant_then_block | 44.4 (363) | 44.8 (395) | +0.5 | [-6.2, +6.7] |
| gate/Stonehaven/portal_then_grant | 79.4 (457) | 77.9 (507) | -1.5 | [-6.9, +3.4] |
| gate/portal_per_legal_turn | 100.0 (457) | 98.8 (513) | **-1.2** | [-2.2, -0.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.8 (263) | 23.6 (208) | -4.2 | [-9.7, +0.7] |
| ikz/held_at_end_of_own_turn | 27.2 (832) | 27.1 (848) | -0.0 | [-3.5, +3.6] |
| ikz/held_then_spent_in_opp_turn | 19.9 (226) | 20.0 (230) | +0.1 | [-4.1, +4.3] |
| leader/Goro/target_then_attacks_same_turn | 13.8 (65) | 9.8 (92) | -4.1 | [-15.1, +5.2] |
| leader/Goro/use_with_target | 100.0 (65) | 100.0 (92) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.0 (723) | 12.1 (758) | +3.1 | [-1.1, +8.4] |
| outcome/own_turns | 7.43 (112) | 7.57 (112) | +0.1 | [-0.2, +0.5] |
| outcome/win | 51.8 (112) | 54.5 (112) | +2.7 | [-6.2, +11.6] |
| response/any_nonblock_response_when_legal | 69.6 (112) | 67.2 (137) | -2.5 | [-15.9, +10.6] |
| response/defender_declared_when_legal | 67.6 (275) | 54.8 (336) | **-12.9** | [-21.4, -4.8] |
| response/spell_played_when_legal | 61.5 (78) | 61.0 (77) | -0.5 | [-17.3, +14.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 70.4 (98) | 70.1 (97) | -0.3 | [-10.5, +10.2] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 46.9 (98) | 47.4 (97) | +0.5 | [-12.1, +12.9] |
| strategy/attacks_by_equipped_attacker | 5.9 (1170) | 3.4 (1218) | **-2.5** | [-4.2, -1.0] |
| strategy/face_target_share_when_both_legal | 60.0 (655) | 47.4 (705) | **-12.6** | [-18.9, -5.9] |
| strategy/favorable_trade_taken_per_available_turn | 48.8 (293) | 62.8 (317) | **+14.0** | [+8.1, +19.6] |
| strategy/spell_cast_per_legal_main_turn | 30.3 (479) | 24.6 (374) | **-5.7** | [-9.7, -1.9] |

### earth vs u8223 — argmax — fixed|ALL

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 25.9 (344) | 28.9 (426) | +3.0 | [-2.7, +8.7] |
| block/blocker_survives | 40.1 (344) | 35.2 (426) | -4.9 | [-10.8, +0.7] |
| block/favorable_survive_or_kill | 49.1 (344) | 46.0 (426) | -3.1 | [-8.3, +1.9] |
| earth/quicksand_cast_multi_removal | 85.8 (113) | 84.1 (107) | -1.7 | [-8.9, +5.5] |
| earth/quicksand_cast_per_legal_turn | 52.8 (214) | 46.9 (228) | -5.9 | [-13.0, +1.5] |
| gate/Stonehaven/grant_then_block | 40.6 (645) | 45.7 (728) | **+5.1** | [+0.8, +9.5] |
| gate/Stonehaven/portal_then_grant | 69.4 (929) | 72.4 (1006) | +2.9 | [-0.4, +6.1] |
| gate/portal_per_legal_turn | 99.5 (934) | 98.3 (1023) | **-1.1** | [-2.3, -0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 32.2 (544) | 28.0 (625) | **-4.2** | [-7.3, -1.2] |
| ikz/held_at_end_of_own_turn | 36.1 (1557) | 37.5 (1617) | +1.4 | [-0.5, +3.2] |
| ikz/held_then_spent_in_opp_turn | 34.3 (562) | 34.5 (606) | +0.1 | [-3.0, +3.6] |
| leader/Bobu/earth_loss_before_next_turn | 56.5 (23) | 76.9 (13) | +20.4 | [-11.3, +43.3] |
| leader/Bobu/use_then_heal_observed | 13.2 (38) | 21.1 (19) | +7.9 | [-8.3, +23.3] |
| leader/Goro/target_then_attacks_same_turn | 18.9 (37) | 13.8 (58) | -5.1 | [-23.3, +10.1] |
| leader/Goro/use_with_target | 100.0 (37) | 100.0 (58) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.0 (1509) | 4.9 (1564) | -0.0 | [-2.1, +2.6] |
| outcome/own_turns | 8.11 (192) | 8.42 (192) | **+0.3** | [+0.1, +0.6] |
| outcome/win | 53.6 (192) | 62.0 (192) | **+8.3** | [+1.6, +14.6] |
| outcome/win_vs_EARTH | 54.2 (48) | 58.3 (48) | +4.2 | [-9.1, +17.9] |
| outcome/win_vs_FIRE | 35.4 (48) | 45.8 (48) | +10.4 | [+0.0, +21.2] |
| outcome/win_vs_LIGHTNING | 62.5 (48) | 72.9 (48) | +10.4 | [-1.9, +24.0] |
| outcome/win_vs_WATER | 62.5 (48) | 70.8 (48) | +8.3 | [-6.5, +23.8] |
| response/any_nonblock_response_when_legal | 61.2 (454) | 56.6 (567) | -4.6 | [-12.4, +3.4] |
| response/defender_declared_when_legal | 47.3 (725) | 46.5 (917) | -0.9 | [-5.3, +3.9] |
| response/spell_played_when_legal | 70.5 (281) | 63.4 (339) | -7.0 | [-14.6, +1.3] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.6 (128) | 6.2 (128) | -2.3 | [-7.7, +2.6] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.5 (128) | 3.1 (128) | -2.3 | [-7.4, +2.3] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.2 (176) | 63.0 (181) | +2.8 | [-5.1, +10.3] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 34.7 (176) | 37.6 (181) | +2.9 | [-5.4, +11.6] |
| strategy/attacks_by_equipped_attacker | 2.0 (1710) | 2.3 (1809) | +0.3 | [-0.2, +0.8] |
| strategy/face_target_share_when_both_legal | 49.8 (941) | 49.4 (977) | -0.4 | [-5.0, +4.8] |
| strategy/favorable_trade_taken_per_available_turn | 54.9 (428) | 62.0 (445) | **+7.1** | [+1.3, +12.5] |
| strategy/spell_cast_per_legal_main_turn | 27.6 (1113) | 24.5 (1182) | **-3.0** | [-5.2, -1.1] |

### earth vs u8223 — argmax — fixed|Stonehaven/Bobu

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 18.7 (262) | 22.7 (321) | +4.0 | [-2.4, +10.2] |
| block/blocker_survives | 34.4 (262) | 30.5 (321) | -3.8 | [-9.7, +1.9] |
| block/favorable_survive_or_kill | 43.5 (262) | 41.1 (321) | -2.4 | [-8.4, +2.7] |
| earth/quicksand_cast_multi_removal | 88.6 (79) | 90.4 (73) | +1.8 | [-7.7, +11.1] |
| earth/quicksand_cast_per_legal_turn | 50.6 (156) | 43.7 (167) | -6.9 | [-14.9, +1.0] |
| gate/Stonehaven/grant_then_block | 39.0 (462) | 43.7 (522) | +4.7 | [-0.1, +9.5] |
| gate/Stonehaven/portal_then_grant | 67.7 (682) | 69.9 (747) | +2.1 | [-1.7, +6.0] |
| gate/portal_per_legal_turn | 99.3 (687) | 98.0 (762) | -1.2 | [-2.7, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.3 (387) | 29.4 (439) | **-3.9** | [-7.8, -0.2] |
| ikz/held_at_end_of_own_turn | 39.1 (1103) | 40.5 (1137) | +1.5 | [-0.6, +3.6] |
| ikz/held_then_spent_in_opp_turn | 34.3 (431) | 35.4 (461) | +1.0 | [-2.8, +4.9] |
| leader/Bobu/earth_loss_before_next_turn | 56.5 (23) | 76.9 (13) | +20.4 | [-11.5, +44.4] |
| leader/Bobu/use_then_heal_observed | 13.2 (38) | 21.1 (19) | +7.9 | [-7.3, +25.3] |
| leader/use_per_legal_turn | 3.4 (1103) | 1.7 (1137) | **-1.8** | [-3.2, -0.4] |
| outcome/own_turns | 8.62 (128) | 8.88 (128) | +0.3 | [-0.0, +0.6] |
| outcome/win | 55.5 (128) | 62.5 (128) | +7.0 | [-0.8, +15.6] |
| response/any_nonblock_response_when_legal | 58.5 (342) | 53.3 (430) | -5.2 | [-14.4, +3.5] |
| response/defender_declared_when_legal | 44.3 (589) | 43.9 (731) | -0.4 | [-5.5, +4.7] |
| response/spell_played_when_legal | 73.9 (203) | 64.1 (262) | **-9.8** | [-17.8, -1.4] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.6 (128) | 6.2 (128) | -2.3 | [-7.8, +2.3] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.5 (128) | 3.1 (128) | -2.3 | [-7.8, +1.6] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 57.4 (122) | 63.5 (126) | +6.1 | [-3.5, +15.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 31.1 (122) | 34.1 (126) | +3.0 | [-7.0, +12.8] |
| strategy/attacks_by_equipped_attacker | 0.0 (1081) | 0.0 (1149) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 42.2 (573) | 42.5 (588) | +0.3 | [-6.0, +7.0] |
| strategy/favorable_trade_taken_per_available_turn | 59.4 (261) | 67.8 (267) | **+8.4** | [+0.6, +15.7] |
| strategy/spell_cast_per_legal_main_turn | 27.5 (810) | 24.8 (855) | **-2.7** | [-5.1, -0.5] |

### earth vs u8223 — argmax — fixed|Stonehaven/Goro

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 48.8 (82) | 47.6 (105) | -1.2 | [-12.0, +11.0] |
| block/blocker_survives | 58.5 (82) | 49.5 (105) | -9.0 | [-25.3, +6.5] |
| block/favorable_survive_or_kill | 67.1 (82) | 61.0 (105) | -6.1 | [-18.9, +6.4] |
| earth/quicksand_cast_multi_removal | 79.4 (34) | 70.6 (34) | -8.8 | [-19.6, +0.0] |
| earth/quicksand_cast_per_legal_turn | 58.6 (58) | 55.7 (61) | -2.9 | [-23.6, +13.7] |
| gate/Stonehaven/grant_then_block | 44.8 (183) | 51.0 (206) | +6.2 | [-2.6, +13.7] |
| gate/Stonehaven/portal_then_grant | 74.1 (247) | 79.5 (259) | **+5.4** | [+0.2, +10.8] |
| gate/portal_per_legal_turn | 100.0 (247) | 99.2 (261) | -0.8 | [-2.3, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 29.3 (157) | 24.7 (186) | **-4.6** | [-10.1, -0.0] |
| ikz/held_at_end_of_own_turn | 28.9 (454) | 30.2 (480) | +1.4 | [-2.4, +5.8] |
| ikz/held_then_spent_in_opp_turn | 34.4 (131) | 31.7 (145) | -2.6 | [-9.2, +3.6] |
| leader/Goro/target_then_attacks_same_turn | 18.9 (37) | 13.8 (58) | -5.1 | [-21.2, +10.2] |
| leader/Goro/use_with_target | 100.0 (37) | 100.0 (58) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.1 (406) | 13.6 (427) | +4.5 | [-2.3, +11.8] |
| outcome/own_turns | 7.09 (64) | 7.50 (64) | **+0.4** | [+0.0, +0.9] |
| outcome/win | 50.0 (64) | 60.9 (64) | **+10.9** | [+3.1, +18.8] |
| response/any_nonblock_response_when_legal | 69.6 (112) | 67.2 (137) | -2.5 | [-15.9, +10.9] |
| response/defender_declared_when_legal | 60.3 (136) | 56.5 (186) | -3.8 | [-15.0, +6.0] |
| response/spell_played_when_legal | 61.5 (78) | 61.0 (77) | -0.5 | [-17.4, +14.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.7 (54) | 61.8 (55) | -4.8 | [-17.5, +7.5] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 42.6 (54) | 45.5 (55) | +2.9 | [-12.3, +17.1] |
| strategy/attacks_by_equipped_attacker | 5.4 (629) | 6.2 (660) | +0.8 | [-0.5, +2.0] |
| strategy/face_target_share_when_both_legal | 61.7 (368) | 59.9 (389) | -1.8 | [-8.2, +5.7] |
| strategy/favorable_trade_taken_per_available_turn | 47.9 (167) | 53.4 (178) | +5.5 | [-2.7, +12.7] |
| strategy/spell_cast_per_legal_main_turn | 27.7 (303) | 23.9 (327) | -3.9 | [-8.2, +0.1] |

### earth vs u8223 — argmax — free_draft|ALL

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 35.7 (199) | 41.4 (157) | +5.7 | [-4.8, +15.2] |
| block/blocker_survives | 44.2 (199) | 38.9 (157) | -5.4 | [-15.2, +3.9] |
| block/favorable_survive_or_kill | 53.8 (199) | 51.6 (157) | -2.2 | [-12.4, +7.4] |
| draft/max_wjaccard_to_curated | 25.4 (96) | 19.1 (96) | **-6.3** | [-7.1, -5.6] |
| draft/mean_cost | 3.36 (4800) | 3.42 (4800) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (4800) | 55.0 (4800) | **-3.0** | [-3.3, -2.7] |
| draft/spell_share | 15.0 (4800) | 5.0 (4800) | **-10.0** | [-10.0, -10.0] |
| draft/unique_cards | 17.00 (96) | 20.50 (96) | **+3.5** | [+3.4, +3.6] |
| draft/weapon_share | 10.0 (4800) | 0.0 (4800) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 89.1 (55) | 88.2 (17) | -0.9 | [-20.6, +15.0] |
| earth/quicksand_cast_per_legal_turn | 39.3 (140) | 40.5 (42) | +1.2 | [-14.4, +23.7] |
| gate/Stonehaven/grant_then_block | 40.9 (364) | 40.0 (355) | -0.9 | [-7.7, +5.7] |
| gate/Stonehaven/portal_then_grant | 88.1 (413) | 76.8 (462) | **-11.3** | [-17.6, -5.2] |
| gate/portal_per_legal_turn | 99.8 (414) | 97.9 (472) | **-1.9** | [-3.1, -0.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 20.7 (213) | 21.6 (74) | +1.0 | [-8.3, +11.6] |
| ikz/held_at_end_of_own_turn | 27.0 (745) | 23.3 (733) | -3.7 | [-7.7, +0.5] |
| ikz/held_then_spent_in_opp_turn | 0.0 (201) | 0.0 (171) | +0.0 | [+0.0, +0.0] |
| leader/Bobu/use_then_heal_observed | 18.8 (16) | 0.0 (6) | -18.8 | [-37.5, +0.0] |
| leader/Goro/target_then_attacks_same_turn | 7.1 (28) | 2.9 (34) | -4.2 | [-16.0, +6.5] |
| leader/Goro/use_with_target | 100.0 (28) | 100.0 (34) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 6.4 (684) | 5.7 (696) | -0.7 | [-3.9, +2.9] |
| outcome/own_turns | 7.76 (96) | 7.64 (96) | -0.1 | [-0.5, +0.3] |
| outcome/win | 57.3 (96) | 53.1 (96) | -4.2 | [-16.7, +8.3] |
| outcome/win_vs_EARTH | 50.0 (24) | 62.5 (24) | +12.5 | [-15.0, +40.9] |
| outcome/win_vs_FIRE | 50.0 (24) | 33.3 (24) | -16.7 | [-40.9, +10.0] |
| outcome/win_vs_LIGHTNING | 58.3 (24) | 50.0 (24) | -8.3 | [-31.8, +16.7] |
| outcome/win_vs_WATER | 70.8 (24) | 66.7 (24) | -4.2 | [-23.1, +15.6] |
| response/defender_declared_when_legal | 77.4 (257) | 57.7 (272) | **-19.7** | [-28.7, -10.5] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 6.2 (48) | 2.1 (48) | -4.2 | [-10.4, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-13.9, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 79.5 (88) | 79.8 (84) | +0.2 | [-11.6, +12.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 55.7 (88) | 53.6 (84) | -2.1 | [-18.2, +14.0] |
| strategy/attacks_by_equipped_attacker | 6.6 (1119) | 0.0 (1175) | **-6.6** | [-8.0, -5.3] |
| strategy/face_target_share_when_both_legal | 52.5 (591) | 33.8 (666) | **-18.7** | [-25.9, -10.4] |
| strategy/favorable_trade_taken_per_available_turn | 55.4 (242) | 73.2 (280) | **+17.8** | [+10.9, +25.0] |
| strategy/spell_cast_per_legal_main_turn | 30.6 (340) | 30.8 (117) | +0.2 | [-7.5, +10.0] |

### earth vs u8223 — argmax — free_draft|Stonehaven/Bobu

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 36.8 (95) | 39.7 (78) | +2.9 | [-11.5, +15.8] |
| block/blocker_survives | 45.3 (95) | 41.0 (78) | -4.2 | [-18.6, +9.5] |
| block/favorable_survive_or_kill | 51.6 (95) | 51.3 (78) | -0.3 | [-14.5, +12.8] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 14.6 (48) | **-3.7** | [-3.7, -3.7] |
| draft/mean_cost | 3.36 (2400) | 3.40 (2400) | **+0.0** | [+0.0, +0.0] |
| draft/normal_share | 58.0 (2400) | 54.0 (2400) | **-4.0** | [-4.0, -4.0] |
| draft/spell_share | 16.0 (2400) | 6.0 (2400) | **-10.0** | [-10.0, -10.0] |
| draft/unique_cards | 17.00 (48) | 20.00 (48) | **+3.0** | [+3.0, +3.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 91.3 (23) | 87.5 (8) | -3.8 | [-38.0, +17.9] |
| earth/quicksand_cast_per_legal_turn | 37.7 (61) | 47.1 (17) | +9.4 | [-17.9, +57.1] |
| gate/Stonehaven/grant_then_block | 38.0 (184) | 42.2 (166) | +4.1 | [-4.5, +12.8] |
| gate/Stonehaven/portal_then_grant | 90.6 (203) | 77.6 (214) | **-13.1** | [-22.4, -3.6] |
| gate/portal_per_legal_turn | 99.5 (204) | 97.3 (220) | **-2.2** | [-4.5, -0.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 15.9 (107) | 25.0 (52) | +9.1 | [-1.2, +23.7] |
| ikz/held_at_end_of_own_turn | 28.9 (367) | 23.6 (365) | **-5.3** | [-10.6, -0.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (106) | 0.0 (86) | +0.0 | [+0.0, +0.0] |
| leader/Bobu/use_then_heal_observed | 18.8 (16) | 0.0 (6) | -18.8 | [-36.4, +0.0] |
| leader/use_per_legal_turn | 4.4 (367) | 1.6 (365) | -2.7 | [-5.6, +0.3] |
| outcome/own_turns | 7.65 (48) | 7.60 (48) | -0.0 | [-0.6, +0.5] |
| outcome/win | 60.4 (48) | 60.4 (48) | +0.0 | [-16.7, +16.7] |
| response/defender_declared_when_legal | 80.5 (118) | 63.9 (122) | **-16.6** | [-27.4, -6.0] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 6.2 (48) | 2.1 (48) | -4.2 | [-10.4, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 6.2 (48) | 0.0 (48) | -6.2 | [-12.5, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 84.1 (44) | 78.6 (42) | -5.5 | [-24.1, +13.0] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 59.1 (44) | 57.1 (42) | -1.9 | [-24.9, +20.3] |
| strategy/attacks_by_equipped_attacker | 6.7 (578) | 0.0 (617) | **-6.7** | [-8.4, -5.4] |
| strategy/face_target_share_when_both_legal | 47.4 (304) | 35.4 (350) | -11.9 | [-23.8, +1.1] |
| strategy/favorable_trade_taken_per_available_turn | 61.2 (116) | 71.6 (141) | +10.4 | [-1.7, +23.4] |
| strategy/spell_cast_per_legal_main_turn | 26.2 (164) | 31.4 (70) | +5.2 | [-4.6, +19.3] |

### earth vs u8223 — argmax — free_draft|Stonehaven/Goro

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 34.6 (104) | 43.0 (79) | +8.4 | [-6.2, +22.3] |
| block/blocker_survives | 43.3 (104) | 36.7 (79) | -6.6 | [-18.9, +6.0] |
| block/favorable_survive_or_kill | 55.8 (104) | 51.9 (79) | -3.9 | [-17.8, +10.7] |
| draft/max_wjaccard_to_curated | 32.5 (48) | 23.6 (48) | **-8.9** | [-8.9, -8.9] |
| draft/mean_cost | 3.36 (2400) | 3.44 (2400) | **+0.1** | [+0.1, +0.1] |
| draft/normal_share | 58.0 (2400) | 56.0 (2400) | **-2.0** | [-2.0, -2.0] |
| draft/spell_share | 14.0 (2400) | 4.0 (2400) | **-10.0** | [-10.0, -10.0] |
| draft/unique_cards | 17.00 (48) | 21.00 (48) | **+4.0** | [+4.0, +4.0] |
| draft/weapon_share | 10.0 (2400) | 0.0 (2400) | **-10.0** | [-10.0, -10.0] |
| earth/quicksand_cast_multi_removal | 87.5 (32) | 88.9 (9) | +1.4 | [-29.3, +21.9] |
| earth/quicksand_cast_per_legal_turn | 40.5 (79) | 36.0 (25) | -4.5 | [-22.9, +31.2] |
| gate/Stonehaven/grant_then_block | 43.9 (180) | 38.1 (189) | -5.8 | [-15.7, +3.2] |
| gate/Stonehaven/portal_then_grant | 85.7 (210) | 76.2 (248) | **-9.5** | [-17.2, -1.6] |
| gate/portal_per_legal_turn | 100.0 (210) | 98.4 (252) | **-1.6** | [-3.0, -0.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.5 (106) | 13.6 (22) | -11.8 | [-27.9, +1.2] |
| ikz/held_at_end_of_own_turn | 25.1 (378) | 23.1 (368) | -2.0 | [-8.1, +4.5] |
| ikz/held_then_spent_in_opp_turn | 0.0 (95) | 0.0 (85) | +0.0 | [+0.0, +0.0] |
| leader/Goro/target_then_attacks_same_turn | 7.1 (28) | 2.9 (34) | -4.2 | [-16.7, +6.7] |
| leader/Goro/use_with_target | 100.0 (28) | 100.0 (34) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.8 (317) | 10.3 (331) | +1.4 | [-4.1, +7.6] |
| outcome/own_turns | 7.88 (48) | 7.67 (48) | -0.2 | [-0.8, +0.3] |
| outcome/win | 54.2 (48) | 45.8 (48) | -8.3 | [-25.0, +10.4] |
| response/defender_declared_when_legal | 74.8 (139) | 52.7 (150) | **-22.2** | [-37.0, -9.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 75.0 (44) | 81.0 (42) | +6.0 | [-10.4, +21.7] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 52.3 (44) | 50.0 (42) | -2.3 | [-24.9, +20.1] |
| strategy/attacks_by_equipped_attacker | 6.5 (541) | 0.0 (558) | **-6.5** | [-9.0, -4.4] |
| strategy/face_target_share_when_both_legal | 57.8 (287) | 32.0 (316) | **-25.9** | [-34.2, -16.5] |
| strategy/favorable_trade_taken_per_available_turn | 50.0 (126) | 74.8 (139) | **+24.8** | [+18.2, +32.3] |
| strategy/spell_cast_per_legal_main_turn | 34.7 (176) | 29.8 (47) | -4.9 | [-15.9, +11.0] |

### earth vs u8223 — sample — all|ALL

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 31.2 (475) | 30.1 (612) | -1.1 | [-5.5, +3.3] |
| block/blocker_survives | 39.2 (475) | 34.6 (612) | -4.5 | [-9.5, +0.7] |
| block/favorable_survive_or_kill | 50.7 (475) | 44.6 (612) | **-6.1** | [-11.1, -1.3] |
| draft/max_wjaccard_to_curated | 12.2 (96) | 12.7 (96) | +0.5 | [-0.2, +1.3] |
| draft/mean_cost | 2.98 (4800) | 3.29 (4800) | **+0.3** | [+0.3, +0.3] |
| draft/normal_share | 77.0 (4800) | 70.0 (4800) | **-6.9** | [-8.0, -5.9] |
| draft/spell_share | 9.3 (4800) | 8.8 (4800) | -0.5 | [-1.1, +0.0] |
| draft/unique_cards | 33.14 (96) | 34.51 (96) | **+1.4** | [+0.9, +1.8] |
| draft/weapon_share | 10.5 (4800) | 5.4 (4800) | **-5.1** | [-5.7, -4.5] |
| earth/quicksand_cast_multi_removal | 82.8 (134) | 81.5 (130) | -1.3 | [-9.8, +6.4] |
| earth/quicksand_cast_per_legal_turn | 53.6 (250) | 44.7 (291) | **-8.9** | [-15.3, -2.2] |
| gate/Stonehaven/grant_then_block | 38.9 (906) | 45.9 (1058) | **+7.1** | [+3.5, +10.6] |
| gate/Stonehaven/portal_then_grant | 69.9 (1297) | 71.6 (1478) | +1.7 | [-1.2, +4.8] |
| gate/portal_per_legal_turn | 99.2 (1308) | 97.9 (1509) | **-1.2** | [-1.9, -0.5] |
| heal/heal_spell_per_legal_turn_below_max_hp | 31.7 (625) | 25.3 (762) | **-6.4** | [-9.5, -3.1] |
| ikz/held_at_end_of_own_turn | 30.7 (2263) | 33.3 (2364) | **+2.6** | [+0.5, +4.7] |
| ikz/held_then_spent_in_opp_turn | 28.6 (695) | 26.7 (787) | -1.9 | [-4.9, +1.1] |
| leader/Bobu/earth_loss_before_next_turn | 47.4 (19) | 77.8 (18) | **+30.4** | [+1.6, +58.3] |
| leader/Bobu/use_then_heal_observed | 12.2 (41) | 26.9 (26) | +14.7 | [-6.1, +36.8] |
| leader/Goro/target_then_attacks_same_turn | 15.6 (64) | 15.4 (117) | -0.2 | [-12.8, +10.5] |
| leader/Goro/use_with_target | 100.0 (64) | 100.0 (117) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 4.8 (2165) | 6.3 (2263) | +1.5 | [-0.7, +4.1] |
| outcome/own_turns | 7.86 (288) | 8.21 (288) | **+0.4** | [+0.1, +0.6] |
| outcome/win | 48.6 (288) | 52.1 (288) | +3.5 | [-2.1, +9.0] |
| outcome/win_vs_EARTH | 48.6 (72) | 54.2 (72) | +5.6 | [-6.4, +18.6] |
| outcome/win_vs_FIRE | 26.4 (72) | 31.9 (72) | +5.6 | [-5.6, +16.7] |
| outcome/win_vs_LIGHTNING | 54.2 (72) | 61.1 (72) | +6.9 | [-2.8, +18.1] |
| outcome/win_vs_WATER | 65.3 (72) | 61.1 (72) | -4.2 | [-13.9, +5.6] |
| response/any_nonblock_response_when_legal | 58.6 (503) | 53.4 (633) | -5.3 | [-12.7, +2.6] |
| response/defender_declared_when_legal | 50.6 (937) | 45.0 (1357) | **-5.6** | [-9.8, -1.5] |
| response/spell_played_when_legal | 64.4 (317) | 63.1 (344) | -1.3 | [-10.1, +6.8] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.0 (176) | 5.1 (176) | +1.1 | [-3.1, +5.3] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (176) | 3.4 (176) | +1.1 | [-1.9, +4.2] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 62.8 (253) | 70.1 (268) | +7.3 | [-0.2, +14.5] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.7 (253) | 42.2 (268) | +1.5 | [-5.4, +8.8] |
| strategy/attacks_by_equipped_attacker | 3.5 (2864) | 2.2 (2889) | **-1.2** | [-1.9, -0.5] |
| strategy/face_target_share_when_both_legal | 50.1 (1665) | 41.7 (1636) | **-8.4** | [-12.6, -3.9] |
| strategy/favorable_trade_taken_per_available_turn | 55.0 (696) | 66.1 (710) | **+11.0** | [+5.9, +16.1] |
| strategy/spell_cast_per_legal_main_turn | 26.7 (1338) | 23.8 (1426) | **-2.9** | [-5.0, -0.8] |

### earth vs u8223 — sample — all|Stonehaven/Bobu

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 26.9 (334) | 24.7 (409) | -2.3 | [-7.4, +3.1] |
| block/blocker_survives | 34.1 (334) | 33.0 (409) | -1.1 | [-6.9, +4.4] |
| block/favorable_survive_or_kill | 46.4 (334) | 41.8 (409) | -4.6 | [-10.3, +0.7] |
| draft/max_wjaccard_to_curated | 9.0 (48) | 11.3 (48) | **+2.3** | [+1.5, +3.2] |
| draft/mean_cost | 2.99 (2400) | 3.29 (2400) | **+0.3** | [+0.2, +0.4] |
| draft/normal_share | 76.5 (2400) | 69.4 (2400) | **-7.1** | [-9.0, -5.3] |
| draft/spell_share | 8.9 (2400) | 8.6 (2400) | -0.3 | [-1.0, +0.4] |
| draft/unique_cards | 32.90 (48) | 34.54 (48) | **+1.6** | [+1.1, +2.2] |
| draft/weapon_share | 10.5 (2400) | 4.9 (2400) | **-5.6** | [-6.5, -4.8] |
| earth/quicksand_cast_multi_removal | 86.7 (90) | 88.0 (83) | +1.3 | [-6.3, +8.8] |
| earth/quicksand_cast_per_legal_turn | 51.4 (175) | 39.2 (212) | **-12.3** | [-19.7, -5.1] |
| gate/Stonehaven/grant_then_block | 39.4 (587) | 44.1 (678) | **+4.7** | [+0.0, +9.2] |
| gate/Stonehaven/portal_then_grant | 69.1 (850) | 70.8 (958) | +1.7 | [-2.2, +5.9] |
| gate/portal_per_legal_turn | 98.8 (860) | 97.6 (982) | **-1.3** | [-2.4, -0.3] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.8 (417) | 26.4 (485) | **-7.4** | [-11.3, -3.8] |
| ikz/held_at_end_of_own_turn | 34.4 (1448) | 37.4 (1483) | **+3.0** | [+0.6, +5.6] |
| ikz/held_then_spent_in_opp_turn | 31.5 (498) | 28.1 (555) | -3.4 | [-7.2, +0.7] |
| leader/Bobu/earth_loss_before_next_turn | 47.4 (19) | 77.8 (18) | **+30.4** | [+0.6, +57.1] |
| leader/Bobu/use_then_heal_observed | 12.2 (41) | 26.9 (26) | +14.7 | [-5.6, +37.0] |
| leader/use_per_legal_turn | 2.8 (1448) | 1.8 (1483) | **-1.1** | [-2.2, -0.0] |
| outcome/own_turns | 8.23 (176) | 8.43 (176) | +0.2 | [-0.1, +0.5] |
| outcome/win | 49.4 (176) | 51.7 (176) | +2.3 | [-5.1, +9.7] |
| response/any_nonblock_response_when_legal | 56.8 (377) | 50.3 (465) | -6.4 | [-14.9, +2.3] |
| response/defender_declared_when_legal | 47.8 (696) | 42.0 (972) | **-5.9** | [-10.9, -1.1] |
| response/spell_played_when_legal | 68.1 (235) | 64.7 (249) | -3.4 | [-12.9, +6.7] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.0 (176) | 5.1 (176) | +1.1 | [-2.8, +5.1] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (176) | 3.4 (176) | +1.1 | [-1.7, +4.5] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 61.0 (159) | 68.2 (170) | +7.2 | [-2.1, +16.5] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.9 (159) | 38.8 (170) | -2.1 | [-11.1, +7.3] |
| strategy/attacks_by_equipped_attacker | 1.9 (1656) | 0.4 (1649) | **-1.5** | [-2.4, -0.7] |
| strategy/face_target_share_when_both_legal | 46.5 (946) | 38.6 (914) | **-7.9** | [-13.7, -1.9] |
| strategy/favorable_trade_taken_per_available_turn | 56.5 (395) | 68.9 (405) | **+12.4** | [+5.2, +19.7] |
| strategy/spell_cast_per_legal_main_turn | 27.3 (911) | 23.3 (949) | **-4.0** | [-6.4, -1.7] |

### earth vs u8223 — sample — all|Stonehaven/Goro

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 41.1 (141) | 40.9 (203) | -0.2 | [-8.5, +7.8] |
| block/blocker_survives | 51.1 (141) | 37.9 (203) | **-13.1** | [-23.0, -2.9] |
| block/favorable_survive_or_kill | 61.0 (141) | 50.2 (203) | -10.7 | [-20.6, +0.1] |
| draft/max_wjaccard_to_curated | 15.4 (48) | 14.1 (48) | **-1.2** | [-1.8, -0.7] |
| draft/mean_cost | 2.98 (2400) | 3.29 (2400) | **+0.3** | [+0.3, +0.4] |
| draft/normal_share | 77.4 (2400) | 70.7 (2400) | **-6.8** | [-7.8, -5.7] |
| draft/spell_share | 9.6 (2400) | 8.9 (2400) | -0.8 | [-1.6, +0.0] |
| draft/unique_cards | 33.38 (48) | 34.48 (48) | **+1.1** | [+0.4, +1.7] |
| draft/weapon_share | 10.4 (2400) | 5.9 (2400) | **-4.5** | [-5.4, -3.7] |
| earth/quicksand_cast_multi_removal | 75.0 (44) | 70.2 (47) | -4.8 | [-22.5, +12.0] |
| earth/quicksand_cast_per_legal_turn | 58.7 (75) | 59.5 (79) | +0.8 | [-13.1, +12.9] |
| gate/Stonehaven/grant_then_block | 37.9 (319) | 49.2 (380) | **+11.3** | [+6.0, +16.7] |
| gate/Stonehaven/portal_then_grant | 71.4 (447) | 73.1 (520) | +1.7 | [-3.3, +6.5] |
| gate/portal_per_legal_turn | 99.8 (448) | 98.7 (527) | **-1.1** | [-2.1, -0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.4 (208) | 23.5 (277) | -3.9 | [-10.1, +1.3] |
| ikz/held_at_end_of_own_turn | 24.2 (815) | 26.3 (881) | +2.2 | [-1.4, +5.9] |
| ikz/held_then_spent_in_opp_turn | 21.3 (197) | 23.3 (232) | +2.0 | [-2.1, +6.3] |
| leader/Goro/target_then_attacks_same_turn | 15.6 (64) | 15.4 (117) | -0.2 | [-12.9, +10.4] |
| leader/Goro/use_with_target | 100.0 (64) | 100.0 (117) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.9 (717) | 15.0 (780) | **+6.1** | [+0.7, +12.6] |
| outcome/own_turns | 7.28 (112) | 7.87 (112) | **+0.6** | [+0.2, +1.0] |
| outcome/win | 47.3 (112) | 52.7 (112) | +5.4 | [-2.7, +13.4] |
| response/any_nonblock_response_when_legal | 64.3 (126) | 61.9 (168) | -2.4 | [-19.9, +13.0] |
| response/defender_declared_when_legal | 58.5 (241) | 52.7 (385) | -5.8 | [-13.7, +2.0] |
| response/spell_played_when_legal | 53.7 (82) | 58.9 (95) | +5.3 | [-14.3, +19.4] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.0 (94) | 73.5 (98) | +7.5 | [-5.0, +20.3] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.4 (94) | 48.0 (98) | +7.5 | [-3.5, +19.6] |
| strategy/attacks_by_equipped_attacker | 5.5 (1208) | 4.7 (1240) | -0.9 | [-2.1, +0.4] |
| strategy/face_target_share_when_both_legal | 54.8 (719) | 45.6 (722) | **-9.2** | [-15.8, -2.4] |
| strategy/favorable_trade_taken_per_available_turn | 53.2 (301) | 62.3 (305) | **+9.1** | [+2.4, +15.9] |
| strategy/spell_cast_per_legal_main_turn | 25.3 (427) | 24.7 (477) | -0.6 | [-5.1, +3.9] |

### earth vs u8223 — sample — fixed|ALL

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 28.4 (349) | 29.7 (424) | +1.4 | [-4.0, +6.5] |
| block/blocker_survives | 40.4 (349) | 36.6 (424) | -3.8 | [-9.4, +1.4] |
| block/favorable_survive_or_kill | 50.7 (349) | 46.0 (424) | -4.7 | [-10.2, +0.3] |
| earth/quicksand_cast_multi_removal | 85.1 (114) | 81.2 (112) | -3.8 | [-11.8, +4.0] |
| earth/quicksand_cast_per_legal_turn | 53.3 (214) | 43.8 (256) | **-9.5** | [-16.4, -2.4] |
| gate/Stonehaven/grant_then_block | 40.5 (650) | 45.8 (725) | **+5.3** | [+0.7, +9.8] |
| gate/Stonehaven/portal_then_grant | 69.7 (932) | 71.9 (1009) | +2.1 | [-1.4, +5.4] |
| gate/portal_per_legal_turn | 99.4 (938) | 98.2 (1027) | **-1.1** | [-1.9, -0.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 31.9 (540) | 26.8 (623) | **-5.0** | [-8.5, -1.5] |
| ikz/held_at_end_of_own_turn | 36.3 (1549) | 38.1 (1620) | +1.8 | [-0.6, +4.2] |
| ikz/held_then_spent_in_opp_turn | 35.1 (562) | 33.4 (617) | -1.7 | [-5.2, +1.9] |
| leader/Bobu/earth_loss_before_next_turn | 46.2 (13) | 76.5 (17) | +30.3 | [-6.0, +63.2] |
| leader/Bobu/use_then_heal_observed | 12.9 (31) | 28.6 (21) | +15.7 | [-6.5, +38.9] |
| leader/Goro/target_then_attacks_same_turn | 21.1 (38) | 13.9 (79) | -7.1 | [-25.8, +7.1] |
| leader/Goro/use_with_target | 100.0 (38) | 100.0 (79) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 4.6 (1497) | 6.4 (1564) | +1.8 | [-0.7, +5.2] |
| outcome/own_turns | 8.07 (192) | 8.44 (192) | **+0.4** | [+0.1, +0.7] |
| outcome/win | 52.1 (192) | 55.7 (192) | +3.6 | [-2.1, +9.9] |
| outcome/win_vs_EARTH | 60.4 (48) | 54.2 (48) | -6.2 | [-18.8, +6.7] |
| outcome/win_vs_FIRE | 25.0 (48) | 41.7 (48) | **+16.7** | [+5.4, +28.0] |
| outcome/win_vs_LIGHTNING | 56.2 (48) | 64.6 (48) | +8.3 | [-4.0, +22.5] |
| outcome/win_vs_WATER | 66.7 (48) | 62.5 (48) | -4.2 | [-16.0, +7.7] |
| response/any_nonblock_response_when_legal | 58.3 (499) | 52.5 (611) | -5.8 | [-13.5, +2.5] |
| response/defender_declared_when_legal | 48.9 (711) | 43.9 (964) | **-5.1** | [-9.8, -0.6] |
| response/spell_played_when_legal | 64.1 (315) | 63.0 (338) | -1.1 | [-10.2, +7.5] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 3.9 (128) | 6.2 (128) | +2.3 | [-2.6, +7.6] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (128) | 3.9 (128) | +1.6 | [-2.2, +5.6] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 63.8 (177) | 66.9 (181) | +3.0 | [-5.4, +11.8] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 40.1 (177) | 39.2 (181) | -0.9 | [-8.4, +7.4] |
| strategy/attacks_by_equipped_attacker | 2.1 (1672) | 2.4 (1740) | +0.3 | [-0.1, +0.8] |
| strategy/face_target_share_when_both_legal | 50.8 (928) | 43.5 (915) | **-7.3** | [-13.1, -1.3] |
| strategy/favorable_trade_taken_per_available_turn | 52.6 (424) | 63.7 (430) | **+11.1** | [+5.1, +17.4] |
| strategy/spell_cast_per_legal_main_turn | 27.4 (1105) | 24.4 (1180) | **-3.0** | [-5.4, -0.6] |

### earth vs u8223 — sample — fixed|Stonehaven/Bobu

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 21.1 (265) | 22.7 (317) | +1.6 | [-4.5, +7.9] |
| block/blocker_survives | 34.7 (265) | 32.8 (317) | -1.9 | [-8.2, +3.8] |
| block/favorable_survive_or_kill | 44.2 (265) | 41.3 (317) | -2.8 | [-9.6, +3.2] |
| earth/quicksand_cast_multi_removal | 87.7 (81) | 89.2 (74) | +1.5 | [-6.1, +8.7] |
| earth/quicksand_cast_per_legal_turn | 50.9 (159) | 38.3 (193) | **-12.6** | [-20.2, -4.6] |
| gate/Stonehaven/grant_then_block | 39.2 (457) | 43.5 (517) | +4.4 | [-1.4, +9.7] |
| gate/Stonehaven/portal_then_grant | 67.9 (673) | 70.1 (738) | +2.1 | [-2.0, +6.4] |
| gate/portal_per_legal_turn | 99.1 (679) | 97.7 (755) | **-1.4** | [-2.4, -0.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.9 (381) | 27.4 (430) | **-6.4** | [-10.2, -2.6] |
| ikz/held_at_end_of_own_turn | 39.2 (1086) | 41.1 (1115) | +1.8 | [-0.7, +4.4] |
| ikz/held_then_spent_in_opp_turn | 36.4 (426) | 33.4 (458) | -3.0 | [-7.2, +1.2] |
| leader/Bobu/earth_loss_before_next_turn | 46.2 (13) | 76.5 (17) | +30.3 | [-7.1, +64.6] |
| leader/Bobu/use_then_heal_observed | 12.9 (31) | 28.6 (21) | +15.7 | [-6.7, +40.5] |
| leader/use_per_legal_turn | 2.9 (1086) | 1.9 (1115) | -1.0 | [-2.2, +0.2] |
| outcome/own_turns | 8.48 (128) | 8.71 (128) | +0.2 | [-0.1, +0.5] |
| outcome/win | 51.6 (128) | 56.2 (128) | +4.7 | [-3.1, +12.5] |
| response/any_nonblock_response_when_legal | 56.3 (373) | 49.6 (452) | -6.7 | [-15.0, +2.0] |
| response/defender_declared_when_legal | 46.3 (570) | 41.4 (764) | -5.0 | [-10.2, +0.1] |
| response/spell_played_when_legal | 67.8 (233) | 64.2 (246) | -3.6 | [-13.4, +6.4] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 3.9 (128) | 6.2 (128) | +2.3 | [-2.3, +7.8] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.3 (128) | 3.9 (128) | +1.6 | [-2.3, +5.5] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.2 (123) | 67.2 (125) | +7.0 | [-3.8, +17.3] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 36.6 (123) | 36.8 (125) | +0.2 | [-9.5, +9.8] |
| strategy/attacks_by_equipped_attacker | 0.0 (1039) | 0.0 (1068) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 44.4 (565) | 39.0 (559) | -5.4 | [-12.8, +2.2] |
| strategy/favorable_trade_taken_per_available_turn | 56.0 (259) | 66.7 (270) | **+10.7** | [+2.2, +19.0] |
| strategy/spell_cast_per_legal_main_turn | 27.6 (800) | 23.4 (847) | **-4.2** | [-6.8, -1.9] |

### earth vs u8223 — sample — fixed|Stonehaven/Goro

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 51.2 (84) | 50.5 (107) | -0.7 | [-11.6, +8.8] |
| block/blocker_survives | 58.3 (84) | 47.7 (107) | -10.7 | [-20.6, +1.7] |
| block/favorable_survive_or_kill | 71.4 (84) | 59.8 (107) | **-11.6** | [-21.2, -2.1] |
| earth/quicksand_cast_multi_removal | 78.8 (33) | 65.8 (38) | -13.0 | [-31.8, +4.4] |
| earth/quicksand_cast_per_legal_turn | 60.0 (55) | 60.3 (63) | +0.3 | [-13.1, +12.0] |
| gate/Stonehaven/grant_then_block | 43.5 (193) | 51.4 (208) | +7.9 | [-0.2, +15.8] |
| gate/Stonehaven/portal_then_grant | 74.5 (259) | 76.8 (271) | +2.2 | [-3.5, +7.7] |
| gate/portal_per_legal_turn | 100.0 (259) | 99.6 (272) | -0.4 | [-1.1, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.0 (159) | 25.4 (193) | -1.7 | [-8.5, +4.3] |
| ikz/held_at_end_of_own_turn | 29.4 (463) | 31.5 (505) | +2.1 | [-2.9, +7.4] |
| ikz/held_then_spent_in_opp_turn | 30.9 (136) | 33.3 (159) | +2.5 | [-3.2, +8.8] |
| leader/Goro/target_then_attacks_same_turn | 21.1 (38) | 13.9 (79) | -7.1 | [-25.3, +7.8] |
| leader/Goro/use_with_target | 100.0 (38) | 100.0 (79) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.2 (411) | 17.6 (449) | **+8.3** | [+0.3, +17.7] |
| outcome/own_turns | 7.23 (64) | 7.89 (64) | **+0.7** | [+0.1, +1.4] |
| outcome/win | 53.1 (64) | 54.7 (64) | +1.6 | [-6.2, +9.4] |
| response/any_nonblock_response_when_legal | 64.3 (126) | 61.0 (159) | -3.3 | [-21.3, +12.0] |
| response/defender_declared_when_legal | 59.6 (141) | 53.5 (200) | -6.1 | [-15.9, +2.6] |
| response/spell_played_when_legal | 53.7 (82) | 59.8 (92) | +6.1 | [-13.8, +19.8] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 72.2 (54) | 66.1 (56) | -6.2 | [-20.4, +7.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 48.1 (54) | 44.6 (56) | -3.5 | [-15.8, +9.1] |
| strategy/attacks_by_equipped_attacker | 5.5 (633) | 6.2 (672) | +0.7 | [-0.4, +1.9] |
| strategy/face_target_share_when_both_legal | 60.6 (363) | 50.6 (356) | **-10.0** | [-19.4, -0.5] |
| strategy/favorable_trade_taken_per_available_turn | 47.3 (165) | 58.8 (160) | **+11.5** | [+3.9, +18.9] |
| strategy/spell_cast_per_legal_main_turn | 26.9 (305) | 27.0 (333) | +0.1 | [-5.2, +5.2] |

### earth vs u8223 — sample — free_draft|ALL

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 38.9 (126) | 30.9 (188) | -8.0 | [-15.9, +0.6] |
| block/blocker_survives | 35.7 (126) | 30.3 (188) | -5.4 | [-18.6, +7.3] |
| block/favorable_survive_or_kill | 50.8 (126) | 41.5 (188) | -9.3 | [-20.9, +2.8] |
| draft/max_wjaccard_to_curated | 12.2 (96) | 12.7 (96) | +0.5 | [-0.2, +1.3] |
| draft/mean_cost | 2.98 (4800) | 3.29 (4800) | **+0.3** | [+0.3, +0.3] |
| draft/normal_share | 77.0 (4800) | 70.0 (4800) | **-6.9** | [-8.0, -5.9] |
| draft/spell_share | 9.3 (4800) | 8.8 (4800) | -0.5 | [-1.0, +0.0] |
| draft/unique_cards | 33.14 (96) | 34.51 (96) | **+1.4** | [+0.9, +1.8] |
| draft/weapon_share | 10.5 (4800) | 5.4 (4800) | **-5.1** | [-5.7, -4.5] |
| earth/quicksand_cast_multi_removal | 70.0 (20) | 83.3 (18) | +13.3 | [-15.8, +40.0] |
| earth/quicksand_cast_per_legal_turn | 55.6 (36) | 51.4 (35) | -4.1 | [-31.1, +23.0] |
| gate/Stonehaven/grant_then_block | 34.8 (256) | 46.2 (333) | **+11.5** | [+5.8, +17.2] |
| gate/Stonehaven/portal_then_grant | 70.1 (365) | 71.0 (469) | +0.9 | [-5.5, +7.5] |
| gate/portal_per_legal_turn | 98.6 (370) | 97.3 (482) | -1.3 | [-3.3, +0.6] |
| heal/heal_spell_per_legal_turn_below_max_hp | 30.6 (85) | 18.7 (139) | **-11.9** | [-22.2, -3.1] |
| ikz/held_at_end_of_own_turn | 18.6 (714) | 22.8 (744) | **+4.2** | [+0.1, +8.2] |
| ikz/held_then_spent_in_opp_turn | 1.5 (133) | 2.4 (170) | +0.8 | [-2.6, +4.9] |
| leader/Bobu/use_then_heal_observed | 10.0 (10) | 20.0 (5) | +10.0 | [-40.0, +66.7] |
| leader/Goro/target_then_attacks_same_turn | 7.7 (26) | 18.4 (38) | +10.7 | [-7.7, +23.3] |
| leader/Goro/use_with_target | 100.0 (26) | 100.0 (38) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.4 (668) | 6.2 (699) | +0.8 | [-2.7, +4.9] |
| outcome/own_turns | 7.44 (96) | 7.75 (96) | +0.3 | [-0.0, +0.6] |
| outcome/win | 41.7 (96) | 44.8 (96) | +3.1 | [-8.3, +13.5] |
| outcome/win_vs_EARTH | 25.0 (24) | 54.2 (24) | **+29.2** | [+4.2, +54.5] |
| outcome/win_vs_FIRE | 29.2 (24) | 12.5 (24) | -16.7 | [-35.0, +0.0] |
| outcome/win_vs_LIGHTNING | 50.0 (24) | 54.2 (24) | +4.2 | [-15.4, +23.5] |
| outcome/win_vs_WATER | 62.5 (24) | 58.3 (24) | -4.2 | [-22.7, +15.0] |
| response/any_nonblock_response_when_legal | 100.0 (4) | 77.3 (22) | -22.7 | [-47.6, +0.0] |
| response/defender_declared_when_legal | 55.8 (226) | 47.8 (393) | -7.9 | [-18.2, +1.5] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.2 (48) | 2.1 (48) | -2.1 | [-10.0, +5.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.1 (48) | 2.1 (48) | +0.0 | [-6.5, +6.2] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.5 (76) | 77.0 (87) | **+16.5** | [+2.1, +29.8] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 42.1 (76) | 48.3 (87) | +6.2 | [-8.5, +20.9] |
| strategy/attacks_by_equipped_attacker | 5.4 (1192) | 2.0 (1149) | **-3.4** | [-4.7, -2.1] |
| strategy/face_target_share_when_both_legal | 49.3 (737) | 39.4 (721) | **-9.9** | [-16.2, -3.4] |
| strategy/favorable_trade_taken_per_available_turn | 58.8 (272) | 69.6 (280) | **+10.8** | [+2.0, +19.5] |
| strategy/spell_cast_per_legal_main_turn | 23.2 (233) | 20.7 (246) | -2.4 | [-8.9, +4.0] |

### earth vs u8223 — sample — free_draft|Stonehaven/Bobu

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 49.3 (69) | 31.5 (92) | **-17.8** | [-27.4, -9.5] |
| block/blocker_survives | 31.9 (69) | 33.7 (92) | +1.8 | [-12.7, +16.2] |
| block/favorable_survive_or_kill | 55.1 (69) | 43.5 (92) | **-11.6** | [-24.9, -2.2] |
| draft/max_wjaccard_to_curated | 9.0 (48) | 11.3 (48) | **+2.3** | [+1.5, +3.2] |
| draft/mean_cost | 2.99 (2400) | 3.29 (2400) | **+0.3** | [+0.2, +0.4] |
| draft/normal_share | 76.5 (2400) | 69.4 (2400) | **-7.1** | [-8.9, -5.3] |
| draft/spell_share | 8.9 (2400) | 8.6 (2400) | -0.3 | [-1.0, +0.3] |
| draft/unique_cards | 32.90 (48) | 34.54 (48) | **+1.6** | [+1.1, +2.2] |
| draft/weapon_share | 10.5 (2400) | 4.9 (2400) | **-5.6** | [-6.4, -4.8] |
| earth/quicksand_cast_multi_removal | 77.8 (9) | 77.8 (9) | +0.0 | [-50.0, +33.3] |
| earth/quicksand_cast_per_legal_turn | 56.2 (16) | 47.4 (19) | -8.9 | [-49.4, +40.8] |
| gate/Stonehaven/grant_then_block | 40.0 (130) | 46.0 (161) | +6.0 | [-2.6, +15.3] |
| gate/Stonehaven/portal_then_grant | 73.4 (177) | 73.2 (220) | -0.3 | [-9.9, +9.6] |
| gate/portal_per_legal_turn | 97.8 (181) | 96.9 (227) | -0.9 | [-4.3, +2.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.3 (36) | 18.2 (55) | **-15.2** | [-36.1, -2.0] |
| ikz/held_at_end_of_own_turn | 19.9 (362) | 26.4 (368) | +6.5 | [-0.2, +12.4] |
| ikz/held_then_spent_in_opp_turn | 2.8 (72) | 3.1 (97) | +0.3 | [-5.1, +7.0] |
| leader/Bobu/use_then_heal_observed | 10.0 (10) | 20.0 (5) | +10.0 | [-37.5, +66.7] |
| leader/use_per_legal_turn | 2.8 (362) | 1.4 (368) | -1.4 | [-4.1, +0.5] |
| outcome/own_turns | 7.54 (48) | 7.67 (48) | +0.1 | [-0.4, +0.6] |
| outcome/win | 43.8 (48) | 39.6 (48) | -4.2 | [-20.8, +10.4] |
| response/any_nonblock_response_when_legal | 100.0 (4) | 76.9 (13) | -23.1 | [-55.6, +0.0] |
| response/defender_declared_when_legal | 54.8 (126) | 44.2 (208) | -10.5 | [-25.2, +2.9] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 4.2 (48) | 2.1 (48) | -2.1 | [-10.4, +4.2] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 2.1 (48) | 2.1 (48) | +0.0 | [-6.2, +6.2] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 63.9 (36) | 71.1 (45) | +7.2 | [-12.0, +27.1] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 55.6 (36) | 44.4 (45) | -11.1 | [-32.4, +10.6] |
| strategy/attacks_by_equipped_attacker | 5.2 (617) | 1.2 (581) | **-4.0** | [-5.7, -2.3] |
| strategy/face_target_share_when_both_legal | 49.6 (381) | 38.0 (355) | **-11.6** | [-21.1, -2.2] |
| strategy/favorable_trade_taken_per_available_turn | 57.4 (136) | 73.3 (135) | **+16.0** | [+3.2, +29.9] |
| strategy/spell_cast_per_legal_main_turn | 25.2 (111) | 22.5 (102) | -2.7 | [-11.7, +6.9] |

### earth vs u8223 — sample — free_draft|Stonehaven/Goro

| metric | u8223 | earth | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 26.3 (57) | 30.2 (96) | +3.9 | [-7.6, +15.2] |
| block/blocker_survives | 40.4 (57) | 27.1 (96) | -13.3 | [-34.8, +6.8] |
| block/favorable_survive_or_kill | 45.6 (57) | 39.6 (96) | -6.0 | [-27.7, +15.1] |
| draft/max_wjaccard_to_curated | 15.4 (48) | 14.1 (48) | **-1.2** | [-1.8, -0.7] |
| draft/mean_cost | 2.98 (2400) | 3.29 (2400) | **+0.3** | [+0.3, +0.3] |
| draft/normal_share | 77.4 (2400) | 70.7 (2400) | **-6.8** | [-7.8, -5.8] |
| draft/spell_share | 9.6 (2400) | 8.9 (2400) | -0.8 | [-1.5, +0.0] |
| draft/unique_cards | 33.38 (48) | 34.48 (48) | **+1.1** | [+0.5, +1.7] |
| draft/weapon_share | 10.4 (2400) | 5.9 (2400) | **-4.5** | [-5.4, -3.8] |
| earth/quicksand_cast_multi_removal | 63.6 (11) | 88.9 (9) | +25.3 | [-18.2, +59.2] |
| earth/quicksand_cast_per_legal_turn | 55.0 (20) | 56.2 (16) | +1.2 | [-43.8, +38.6] |
| gate/Stonehaven/grant_then_block | 29.4 (126) | 46.5 (172) | **+17.1** | [+10.5, +24.2] |
| gate/Stonehaven/portal_then_grant | 67.0 (188) | 69.1 (249) | +2.1 | [-6.2, +10.9] |
| gate/portal_per_legal_turn | 99.5 (189) | 97.6 (255) | -1.8 | [-3.5, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 28.6 (49) | 19.0 (84) | -9.5 | [-25.5, +2.6] |
| ikz/held_at_end_of_own_turn | 17.3 (352) | 19.4 (376) | +2.1 | [-2.8, +7.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (61) | 1.4 (73) | +1.4 | [+0.0, +4.5] |
| leader/Goro/target_then_attacks_same_turn | 7.7 (26) | 18.4 (38) | +10.7 | [-8.4, +23.3] |
| leader/Goro/use_with_target | 100.0 (26) | 100.0 (38) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.5 (306) | 11.5 (331) | +3.0 | [-3.4, +11.4] |
| outcome/own_turns | 7.33 (48) | 7.83 (48) | **+0.5** | [+0.1, +1.0] |
| outcome/win | 39.6 (48) | 50.0 (48) | +10.4 | [-4.2, +25.0] |
| response/any_nonblock_response_when_legal | – (0) | 77.8 (9) | – | |
| response/defender_declared_when_legal | 57.0 (100) | 51.9 (185) | -5.1 | [-18.8, +7.4] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 57.5 (40) | 83.3 (42) | **+25.8** | [+6.6, +44.3] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 30.0 (40) | 52.4 (42) | **+22.4** | [+2.7, +41.6] |
| strategy/attacks_by_equipped_attacker | 5.6 (575) | 2.8 (568) | **-2.7** | [-4.6, -0.9] |
| strategy/face_target_share_when_both_legal | 48.9 (356) | 40.7 (366) | -8.2 | [-17.4, +2.5] |
| strategy/favorable_trade_taken_per_available_turn | 60.3 (136) | 66.2 (145) | +5.9 | [-5.0, +16.2] |
| strategy/spell_cast_per_legal_main_turn | 21.3 (122) | 19.4 (144) | -1.9 | [-12.1, +7.1] |

### u9305 vs u8223 — argmax — all|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 29.5 (543) | 34.6 (523) | **+5.1** | [+0.7, +9.6] |
| block/blocker_survives | 41.6 (543) | 40.5 (523) | -1.1 | [-5.2, +2.7] |
| block/favorable_survive_or_kill | 50.8 (543) | 52.0 (523) | +1.2 | [-3.3, +5.6] |
| draft/max_wjaccard_to_curated | 25.4 (96) | 24.6 (96) | **-0.8** | [-1.0, -0.6] |
| draft/mean_cost | 3.36 (4800) | 3.72 (4800) | **+0.4** | [+0.4, +0.4] |
| draft/normal_share | 58.0 (4800) | 65.0 (4800) | **+7.0** | [+6.7, +7.3] |
| draft/spell_share | 15.0 (4800) | 10.0 (4800) | **-5.0** | [-5.3, -4.7] |
| draft/unique_cards | 17.00 (96) | 15.50 (96) | **-1.5** | [-1.6, -1.4] |
| draft/weapon_share | 10.0 (4800) | 8.0 (4800) | **-2.0** | [-2.0, -2.0] |
| earth/quicksand_cast_multi_removal | 86.9 (168) | 85.6 (160) | -1.3 | [-7.2, +4.8] |
| earth/quicksand_cast_per_legal_turn | 47.5 (354) | 40.3 (397) | **-7.2** | [-12.9, -1.8] |
| gate/Stonehaven/grant_then_block | 40.7 (1009) | 39.7 (1018) | -1.0 | [-4.1, +2.0] |
| gate/Stonehaven/portal_then_grant | 75.2 (1342) | 70.4 (1446) | **-4.8** | [-7.8, -2.0] |
| gate/portal_per_legal_turn | 99.6 (1348) | 99.2 (1457) | -0.3 | [-0.8, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 28.9 (757) | 26.7 (660) | -2.3 | [-5.2, +0.6] |
| ikz/held_at_end_of_own_turn | 33.1 (2302) | 34.1 (2295) | +1.0 | [-0.7, +2.8] |
| ikz/held_then_spent_in_opp_turn | 25.3 (763) | 26.6 (783) | +1.3 | [-1.0, +3.6] |
| leader/Bobu/earth_loss_before_next_turn | 57.1 (28) | 53.3 (15) | -3.8 | [-33.3, +25.0] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 17.9 (28) | +3.0 | [-11.7, +20.5] |
| leader/Goro/target_then_attacks_same_turn | 13.8 (65) | 19.6 (56) | +5.8 | [-2.7, +15.1] |
| leader/Goro/use_with_target | 100.0 (65) | 100.0 (56) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.4 (2193) | 3.8 (2185) | **-1.6** | [-2.6, -0.4] |
| outcome/own_turns | 7.99 (288) | 7.97 (288) | -0.0 | [-0.2, +0.2] |
| outcome/win | 54.9 (288) | 53.1 (288) | -1.7 | [-6.6, +2.8] |
| outcome/win_vs_EARTH | 52.8 (72) | 50.0 (72) | -2.8 | [-11.8, +6.9] |
| outcome/win_vs_FIRE | 40.3 (72) | 37.5 (72) | -2.8 | [-11.8, +6.1] |
| outcome/win_vs_LIGHTNING | 61.1 (72) | 63.9 (72) | +2.8 | [-5.0, +10.3] |
| outcome/win_vs_WATER | 65.3 (72) | 61.1 (72) | -4.2 | [-15.8, +7.5] |
| response/any_nonblock_response_when_legal | 61.2 (454) | 61.8 (505) | +0.5 | [-5.9, +6.4] |
| response/defender_declared_when_legal | 55.2 (982) | 55.5 (941) | +0.3 | [-3.2, +3.8] |
| response/spell_played_when_legal | 70.5 (281) | 64.0 (328) | **-6.4** | [-12.2, -0.9] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.0 (176) | 4.5 (176) | -3.4 | [-8.0, +0.7] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.7 (176) | 2.8 (176) | -2.8 | [-6.9, +1.1] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.7 (264) | 66.9 (257) | +0.3 | [-4.8, +6.2] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 41.7 (264) | 39.3 (257) | -2.4 | [-9.0, +4.7] |
| strategy/attacks_by_equipped_attacker | 3.8 (2829) | 3.5 (2741) | -0.4 | [-1.0, +0.2] |
| strategy/face_target_share_when_both_legal | 50.8 (1532) | 50.3 (1503) | -0.5 | [-4.3, +3.2] |
| strategy/favorable_trade_taken_per_available_turn | 55.1 (670) | 57.2 (696) | +2.1 | [-1.8, +6.1] |
| strategy/spell_cast_per_legal_main_turn | 28.3 (1453) | 25.9 (1370) | **-2.4** | [-4.3, -0.5] |

### u9305 vs u8223 — argmax — all|Stonehaven/Bobu

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 23.5 (357) | 27.4 (354) | +3.9 | [-1.4, +9.2] |
| block/blocker_survives | 37.3 (357) | 34.7 (354) | -2.5 | [-7.6, +2.4] |
| block/favorable_survive_or_kill | 45.7 (357) | 44.6 (354) | -1.0 | [-6.4, +4.6] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 18.3 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 3.36 (2400) | 3.72 (2400) | **+0.4** | [+0.4, +0.4] |
| draft/normal_share | 58.0 (2400) | 66.0 (2400) | **+8.0** | [+8.0, +8.0] |
| draft/spell_share | 16.0 (2400) | 10.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/unique_cards | 17.00 (48) | 15.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 10.0 (2400) | 8.0 (2400) | **-2.0** | [-2.0, -2.0] |
| earth/quicksand_cast_multi_removal | 89.2 (102) | 89.9 (99) | +0.7 | [-6.6, +8.0] |
| earth/quicksand_cast_per_legal_turn | 47.0 (217) | 39.0 (254) | **-8.0** | [-14.6, -1.2] |
| gate/Stonehaven/grant_then_block | 38.7 (646) | 38.9 (651) | +0.2 | [-3.2, +3.8] |
| gate/Stonehaven/portal_then_grant | 73.0 (885) | 68.7 (948) | **-4.3** | [-8.3, -0.8] |
| gate/portal_per_legal_turn | 99.3 (891) | 98.9 (959) | -0.5 | [-1.2, +0.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 29.6 (494) | 28.1 (434) | -1.4 | [-4.8, +2.2] |
| ikz/held_at_end_of_own_turn | 36.5 (1470) | 36.6 (1449) | +0.1 | [-1.7, +2.0] |
| ikz/held_then_spent_in_opp_turn | 27.6 (537) | 29.6 (531) | +2.0 | [-0.8, +5.0] |
| leader/Bobu/earth_loss_before_next_turn | 57.1 (28) | 53.3 (15) | -3.8 | [-33.3, +27.4] |
| leader/Bobu/use_then_heal_observed | 14.8 (54) | 17.9 (28) | +3.0 | [-12.0, +22.0] |
| leader/use_per_legal_turn | 3.7 (1470) | 1.9 (1449) | **-1.7** | [-2.6, -0.9] |
| outcome/own_turns | 8.35 (176) | 8.23 (176) | -0.1 | [-0.3, +0.1] |
| outcome/win | 56.8 (176) | 54.0 (176) | -2.8 | [-8.5, +3.4] |
| response/any_nonblock_response_when_legal | 58.5 (342) | 58.7 (373) | +0.2 | [-7.3, +7.1] |
| response/defender_declared_when_legal | 50.4 (707) | 51.2 (689) | +0.9 | [-2.8, +4.5] |
| response/spell_played_when_legal | 73.9 (203) | 64.1 (245) | **-9.8** | [-16.0, -3.6] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.0 (176) | 4.5 (176) | -3.4 | [-8.0, +0.6] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.7 (176) | 2.8 (176) | -2.8 | [-7.4, +1.1] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 64.5 (166) | 63.4 (161) | -1.1 | [-7.8, +6.2] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 38.6 (166) | 36.6 (161) | -1.9 | [-9.8, +6.0] |
| strategy/attacks_by_equipped_attacker | 2.4 (1659) | 1.8 (1594) | -0.6 | [-1.3, +0.1] |
| strategy/face_target_share_when_both_legal | 44.0 (877) | 44.5 (850) | +0.5 | [-4.8, +5.5] |
| strategy/favorable_trade_taken_per_available_turn | 59.9 (377) | 62.3 (393) | +2.4 | [-3.5, +8.2] |
| strategy/spell_cast_per_legal_main_turn | 27.3 (974) | 25.7 (920) | -1.7 | [-3.8, +0.4] |

### u9305 vs u8223 — argmax — all|Stonehaven/Goro

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 40.9 (186) | 49.7 (169) | **+8.8** | [+1.2, +16.4] |
| block/blocker_survives | 50.0 (186) | 52.7 (169) | +2.7 | [-4.9, +9.5] |
| block/favorable_survive_or_kill | 60.8 (186) | 67.5 (169) | +6.7 | [-1.2, +14.4] |
| draft/max_wjaccard_to_curated | 32.5 (48) | 31.0 (48) | **-1.6** | [-1.6, -1.6] |
| draft/mean_cost | 3.36 (2400) | 3.72 (2400) | **+0.4** | [+0.4, +0.4] |
| draft/normal_share | 58.0 (2400) | 64.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 14.0 (2400) | 10.0 (2400) | **-4.0** | [-4.0, -4.0] |
| draft/unique_cards | 17.00 (48) | 16.00 (48) | **-1.0** | [-1.0, -1.0] |
| draft/weapon_share | 10.0 (2400) | 8.0 (2400) | **-2.0** | [-2.0, -2.0] |
| earth/quicksand_cast_multi_removal | 83.3 (66) | 78.7 (61) | -4.6 | [-15.3, +5.4] |
| earth/quicksand_cast_per_legal_turn | 48.2 (137) | 42.7 (143) | -5.5 | [-15.9, +3.5] |
| gate/Stonehaven/grant_then_block | 44.4 (363) | 41.1 (367) | -3.2 | [-9.3, +2.6] |
| gate/Stonehaven/portal_then_grant | 79.4 (457) | 73.7 (498) | **-5.7** | [-10.4, -1.6] |
| gate/portal_per_legal_turn | 100.0 (457) | 100.0 (498) | +0.0 | [+0.0, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 27.8 (263) | 23.9 (226) | -3.9 | [-9.6, +1.5] |
| ikz/held_at_end_of_own_turn | 27.2 (832) | 29.8 (846) | +2.6 | [-1.1, +6.2] |
| ikz/held_then_spent_in_opp_turn | 19.9 (226) | 20.2 (252) | +0.3 | [-3.1, +3.5] |
| leader/Goro/target_then_attacks_same_turn | 13.8 (65) | 19.6 (56) | +5.8 | [-2.7, +15.2] |
| leader/Goro/use_with_target | 100.0 (65) | 100.0 (56) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.0 (723) | 7.6 (736) | -1.4 | [-3.9, +1.4] |
| outcome/own_turns | 7.43 (112) | 7.55 (112) | +0.1 | [-0.2, +0.5] |
| outcome/win | 51.8 (112) | 51.8 (112) | +0.0 | [-8.0, +8.0] |
| response/any_nonblock_response_when_legal | 69.6 (112) | 70.5 (132) | +0.8 | [-10.9, +11.3] |
| response/defender_declared_when_legal | 67.6 (275) | 67.1 (252) | -0.6 | [-8.2, +6.8] |
| response/spell_played_when_legal | 61.5 (78) | 63.9 (83) | +2.3 | [-10.5, +11.8] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 70.4 (98) | 72.9 (96) | +2.5 | [-6.6, +11.7] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 46.9 (98) | 43.8 (96) | -3.2 | [-15.0, +8.8] |
| strategy/attacks_by_equipped_attacker | 5.9 (1170) | 5.8 (1147) | -0.1 | [-1.3, +1.1] |
| strategy/face_target_share_when_both_legal | 60.0 (655) | 57.9 (653) | -2.1 | [-6.9, +2.4] |
| strategy/favorable_trade_taken_per_available_turn | 48.8 (293) | 50.5 (303) | +1.7 | [-3.5, +7.2] |
| strategy/spell_cast_per_legal_main_turn | 30.3 (479) | 26.4 (450) | **-3.8** | [-7.7, -0.1] |

### u9305 vs u8223 — argmax — fixed|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 25.9 (344) | 30.1 (376) | +4.2 | [-0.6, +9.2] |
| block/blocker_survives | 40.1 (344) | 35.6 (376) | **-4.5** | [-9.1, -0.1] |
| block/favorable_survive_or_kill | 49.1 (344) | 48.4 (376) | -0.7 | [-5.9, +4.4] |
| earth/quicksand_cast_multi_removal | 85.8 (113) | 88.3 (111) | +2.4 | [-1.9, +7.1] |
| earth/quicksand_cast_per_legal_turn | 52.8 (214) | 45.1 (246) | **-7.7** | [-14.3, -1.4] |
| gate/Stonehaven/grant_then_block | 40.6 (645) | 41.2 (699) | +0.6 | [-2.6, +4.1] |
| gate/Stonehaven/portal_then_grant | 69.4 (929) | 69.1 (1012) | -0.4 | [-2.9, +2.2] |
| gate/portal_per_legal_turn | 99.5 (934) | 99.0 (1022) | -0.4 | [-1.0, +0.1] |
| heal/heal_spell_per_legal_turn_below_max_hp | 32.2 (544) | 30.3 (558) | -1.9 | [-4.6, +0.7] |
| ikz/held_at_end_of_own_turn | 36.1 (1557) | 37.8 (1579) | +1.7 | [-0.0, +3.6] |
| ikz/held_then_spent_in_opp_turn | 34.3 (562) | 34.8 (597) | +0.5 | [-2.3, +3.6] |
| leader/Bobu/earth_loss_before_next_turn | 56.5 (23) | 50.0 (14) | -6.5 | [-36.6, +27.6] |
| leader/Bobu/use_then_heal_observed | 13.2 (38) | 16.0 (25) | +2.8 | [-13.0, +25.0] |
| leader/Goro/target_then_attacks_same_turn | 18.9 (37) | 20.5 (44) | +1.5 | [-8.1, +10.2] |
| leader/Goro/use_with_target | 100.0 (37) | 100.0 (44) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 5.0 (1509) | 4.5 (1532) | -0.5 | [-1.5, +0.8] |
| outcome/own_turns | 8.11 (192) | 8.22 (192) | +0.1 | [-0.1, +0.4] |
| outcome/win | 53.6 (192) | 51.0 (192) | -2.6 | [-7.8, +2.6] |
| outcome/win_vs_EARTH | 54.2 (48) | 54.2 (48) | +0.0 | [-10.0, +10.3] |
| outcome/win_vs_FIRE | 35.4 (48) | 35.4 (48) | +0.0 | [-9.6, +9.6] |
| outcome/win_vs_LIGHTNING | 62.5 (48) | 60.4 (48) | -2.1 | [-8.9, +5.0] |
| outcome/win_vs_WATER | 62.5 (48) | 54.2 (48) | -8.3 | [-22.2, +5.6] |
| response/any_nonblock_response_when_legal | 61.2 (454) | 61.8 (505) | +0.5 | [-6.0, +6.6] |
| response/defender_declared_when_legal | 47.3 (725) | 51.1 (734) | **+3.8** | [+0.5, +7.5] |
| response/spell_played_when_legal | 70.5 (281) | 64.0 (328) | **-6.4** | [-12.6, -0.6] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.6 (128) | 5.5 (128) | -3.1 | [-8.8, +2.3] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.5 (128) | 3.1 (128) | -2.3 | [-7.7, +2.5] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 60.2 (176) | 61.7 (175) | +1.5 | [-5.3, +7.8] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 34.7 (176) | 35.4 (175) | +0.8 | [-5.4, +7.1] |
| strategy/attacks_by_equipped_attacker | 2.0 (1710) | 2.4 (1715) | **+0.4** | [+0.1, +0.7] |
| strategy/face_target_share_when_both_legal | 49.8 (941) | 53.1 (956) | +3.3 | [-0.5, +7.2] |
| strategy/favorable_trade_taken_per_available_turn | 54.9 (428) | 57.7 (426) | +2.8 | [-1.9, +7.5] |
| strategy/spell_cast_per_legal_main_turn | 27.6 (1113) | 26.4 (1125) | -1.2 | [-3.0, +0.6] |

### u9305 vs u8223 — argmax — fixed|Stonehaven/Bobu

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 18.7 (262) | 20.9 (277) | +2.2 | [-3.3, +7.8] |
| block/blocker_survives | 34.4 (262) | 29.2 (277) | **-5.1** | [-10.7, -0.2] |
| block/favorable_survive_or_kill | 43.5 (262) | 40.1 (277) | -3.4 | [-9.4, +1.7] |
| earth/quicksand_cast_multi_removal | 88.6 (79) | 93.5 (77) | +4.9 | [-0.8, +10.9] |
| earth/quicksand_cast_per_legal_turn | 50.6 (156) | 42.8 (180) | **-7.9** | [-16.3, -0.4] |
| gate/Stonehaven/grant_then_block | 39.0 (462) | 38.6 (490) | -0.4 | [-4.3, +3.5] |
| gate/Stonehaven/portal_then_grant | 67.7 (682) | 66.9 (732) | -0.8 | [-4.5, +2.3] |
| gate/portal_per_legal_turn | 99.3 (687) | 98.7 (742) | -0.6 | [-1.4, +0.2] |
| heal/heal_spell_per_legal_turn_below_max_hp | 33.3 (387) | 30.2 (394) | -3.1 | [-6.7, +0.4] |
| ikz/held_at_end_of_own_turn | 39.1 (1103) | 39.7 (1098) | +0.6 | [-1.0, +2.4] |
| ikz/held_then_spent_in_opp_turn | 34.3 (431) | 36.0 (436) | +1.7 | [-1.8, +5.3] |
| leader/Bobu/earth_loss_before_next_turn | 56.5 (23) | 50.0 (14) | -6.5 | [-38.3, +26.5] |
| leader/Bobu/use_then_heal_observed | 13.2 (38) | 16.0 (25) | +2.8 | [-13.8, +26.0] |
| leader/use_per_legal_turn | 3.4 (1103) | 2.3 (1098) | **-1.2** | [-2.1, -0.3] |
| outcome/own_turns | 8.62 (128) | 8.58 (128) | -0.0 | [-0.3, +0.2] |
| outcome/win | 55.5 (128) | 53.1 (128) | -2.3 | [-9.4, +4.7] |
| response/any_nonblock_response_when_legal | 58.5 (342) | 58.7 (373) | +0.2 | [-7.1, +7.1] |
| response/defender_declared_when_legal | 44.3 (589) | 46.9 (588) | +2.6 | [-1.4, +6.5] |
| response/spell_played_when_legal | 73.9 (203) | 64.1 (245) | **-9.8** | [-16.1, -3.8] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 8.6 (128) | 5.5 (128) | -3.1 | [-9.4, +2.3] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 5.5 (128) | 3.1 (128) | -2.3 | [-7.8, +2.3] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 57.4 (122) | 58.0 (119) | +0.6 | [-7.7, +9.1] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 31.1 (122) | 31.9 (119) | +0.8 | [-7.7, +9.1] |
| strategy/attacks_by_equipped_attacker | 0.0 (1081) | 0.0 (1056) | +0.0 | [+0.0, +0.0] |
| strategy/face_target_share_when_both_legal | 42.2 (573) | 46.2 (558) | +4.0 | [-1.3, +9.2] |
| strategy/favorable_trade_taken_per_available_turn | 59.4 (261) | 63.0 (257) | +3.6 | [-3.1, +9.8] |
| strategy/spell_cast_per_legal_main_turn | 27.5 (810) | 25.9 (811) | -1.6 | [-3.8, +0.4] |

### u9305 vs u8223 — argmax — fixed|Stonehaven/Goro

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 48.8 (82) | 55.6 (99) | +6.8 | [-1.8, +15.0] |
| block/blocker_survives | 58.5 (82) | 53.5 (99) | -5.0 | [-14.3, +4.3] |
| block/favorable_survive_or_kill | 67.1 (82) | 71.7 (99) | +4.6 | [-5.3, +14.2] |
| earth/quicksand_cast_multi_removal | 79.4 (34) | 76.5 (34) | -2.9 | [-11.8, +2.9] |
| earth/quicksand_cast_per_legal_turn | 58.6 (58) | 51.5 (66) | -7.1 | [-22.4, +4.7] |
| gate/Stonehaven/grant_then_block | 44.8 (183) | 47.4 (209) | +2.6 | [-4.0, +8.7] |
| gate/Stonehaven/portal_then_grant | 74.1 (247) | 74.6 (280) | +0.6 | [-2.3, +3.6] |
| gate/portal_per_legal_turn | 100.0 (247) | 100.0 (280) | +0.0 | [+0.0, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 29.3 (157) | 30.5 (164) | +1.2 | [-2.4, +4.0] |
| ikz/held_at_end_of_own_turn | 28.9 (454) | 33.5 (481) | **+4.6** | [+0.6, +9.0] |
| ikz/held_then_spent_in_opp_turn | 34.4 (131) | 31.7 (161) | -2.7 | [-8.9, +2.9] |
| leader/Goro/target_then_attacks_same_turn | 18.9 (37) | 20.5 (44) | +1.5 | [-8.8, +10.2] |
| leader/Goro/use_with_target | 100.0 (37) | 100.0 (44) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 9.1 (406) | 10.1 (434) | +1.0 | [-1.7, +4.2] |
| outcome/own_turns | 7.09 (64) | 7.52 (64) | **+0.4** | [+0.1, +0.9] |
| outcome/win | 50.0 (64) | 46.9 (64) | -3.1 | [-10.9, +4.7] |
| response/any_nonblock_response_when_legal | 69.6 (112) | 70.5 (132) | +0.8 | [-10.8, +11.6] |
| response/defender_declared_when_legal | 60.3 (136) | 67.8 (146) | **+7.5** | [+1.0, +14.1] |
| response/spell_played_when_legal | 61.5 (78) | 63.9 (83) | +2.3 | [-10.0, +11.8] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 66.7 (54) | 69.6 (56) | +3.0 | [-8.1, +15.4] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 42.6 (54) | 42.9 (56) | +0.3 | [-11.9, +11.0] |
| strategy/attacks_by_equipped_attacker | 5.4 (629) | 6.2 (659) | +0.8 | [-0.1, +1.9] |
| strategy/face_target_share_when_both_legal | 61.7 (368) | 62.8 (398) | +1.1 | [-3.2, +5.8] |
| strategy/favorable_trade_taken_per_available_turn | 47.9 (167) | 49.7 (169) | +1.8 | [-4.0, +7.0] |
| strategy/spell_cast_per_legal_main_turn | 27.7 (303) | 27.7 (314) | -0.0 | [-3.7, +3.5] |

### u9305 vs u8223 — argmax — free_draft|ALL

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 35.7 (199) | 46.3 (147) | **+10.6** | [+2.0, +19.3] |
| block/blocker_survives | 44.2 (199) | 53.1 (147) | **+8.8** | [+1.1, +17.4] |
| block/favorable_survive_or_kill | 53.8 (199) | 61.2 (147) | +7.5 | [-0.8, +16.7] |
| draft/max_wjaccard_to_curated | 25.4 (96) | 24.6 (96) | **-0.8** | [-1.0, -0.6] |
| draft/mean_cost | 3.36 (4800) | 3.72 (4800) | **+0.4** | [+0.4, +0.4] |
| draft/normal_share | 58.0 (4800) | 65.0 (4800) | **+7.0** | [+6.7, +7.3] |
| draft/spell_share | 15.0 (4800) | 10.0 (4800) | **-5.0** | [-5.3, -4.7] |
| draft/unique_cards | 17.00 (96) | 15.50 (96) | **-1.5** | [-1.6, -1.4] |
| draft/weapon_share | 10.0 (4800) | 8.0 (4800) | **-2.0** | [-2.0, -2.0] |
| earth/quicksand_cast_multi_removal | 89.1 (55) | 79.6 (49) | -9.5 | [-26.7, +6.0] |
| earth/quicksand_cast_per_legal_turn | 39.3 (140) | 32.5 (151) | -6.8 | [-16.3, +2.8] |
| gate/Stonehaven/grant_then_block | 40.9 (364) | 36.4 (319) | -4.6 | [-10.9, +1.4] |
| gate/Stonehaven/portal_then_grant | 88.1 (413) | 73.5 (434) | **-14.6** | [-20.6, -8.8] |
| gate/portal_per_legal_turn | 99.8 (414) | 99.8 (435) | +0.0 | [-0.7, +0.7] |
| heal/heal_spell_per_legal_turn_below_max_hp | 20.7 (213) | 6.9 (102) | **-13.8** | [-20.7, -6.5] |
| ikz/held_at_end_of_own_turn | 27.0 (745) | 26.0 (716) | -1.0 | [-4.8, +3.0] |
| ikz/held_then_spent_in_opp_turn | 0.0 (201) | 0.0 (186) | +0.0 | [+0.0, +0.0] |
| leader/Bobu/use_then_heal_observed | 18.8 (16) | 33.3 (3) | +14.6 | [-25.0, +38.2] |
| leader/Goro/target_then_attacks_same_turn | 7.1 (28) | 16.7 (12) | +9.5 | [-12.0, +50.0] |
| leader/Goro/use_with_target | 100.0 (28) | 100.0 (12) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 6.4 (684) | 2.3 (653) | **-4.1** | [-6.3, -1.9] |
| outcome/own_turns | 7.76 (96) | 7.46 (96) | -0.3 | [-0.6, +0.0] |
| outcome/win | 57.3 (96) | 57.3 (96) | +0.0 | [-9.4, +10.4] |
| outcome/win_vs_EARTH | 50.0 (24) | 41.7 (24) | -8.3 | [-27.8, +11.5] |
| outcome/win_vs_FIRE | 50.0 (24) | 41.7 (24) | -8.3 | [-25.0, +8.3] |
| outcome/win_vs_LIGHTNING | 58.3 (24) | 70.8 (24) | +12.5 | [-5.0, +30.8] |
| outcome/win_vs_WATER | 70.8 (24) | 75.0 (24) | +4.2 | [-16.7, +30.0] |
| response/defender_declared_when_legal | 77.4 (257) | 71.0 (207) | -6.4 | [-15.0, +2.3] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 6.2 (48) | 2.1 (48) | -4.2 | [-10.7, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 6.2 (48) | 2.1 (48) | -4.2 | [-10.7, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 79.5 (88) | 78.0 (82) | -1.5 | [-11.2, +8.7] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 55.7 (88) | 47.6 (82) | -8.1 | [-23.6, +7.0] |
| strategy/attacks_by_equipped_attacker | 6.6 (1119) | 5.3 (1026) | -1.3 | [-3.0, +0.2] |
| strategy/face_target_share_when_both_legal | 52.5 (591) | 45.3 (547) | -7.1 | [-14.2, +0.3] |
| strategy/favorable_trade_taken_per_available_turn | 55.4 (242) | 56.3 (270) | +0.9 | [-6.3, +8.5] |
| strategy/spell_cast_per_legal_main_turn | 30.6 (340) | 23.7 (245) | **-6.9** | [-13.2, -0.7] |

### u9305 vs u8223 — argmax — free_draft|Stonehaven/Bobu

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 36.8 (95) | 50.6 (77) | **+13.8** | [+0.9, +25.6] |
| block/blocker_survives | 45.3 (95) | 54.5 (77) | +9.3 | [-2.4, +22.1] |
| block/favorable_survive_or_kill | 51.6 (95) | 61.0 (77) | +9.5 | [-2.7, +21.6] |
| draft/max_wjaccard_to_curated | 18.3 (48) | 18.3 (48) | +0.0 | [+0.0, +0.0] |
| draft/mean_cost | 3.36 (2400) | 3.72 (2400) | **+0.4** | [+0.4, +0.4] |
| draft/normal_share | 58.0 (2400) | 66.0 (2400) | **+8.0** | [+8.0, +8.0] |
| draft/spell_share | 16.0 (2400) | 10.0 (2400) | **-6.0** | [-6.0, -6.0] |
| draft/unique_cards | 17.00 (48) | 15.00 (48) | **-2.0** | [-2.0, -2.0] |
| draft/weapon_share | 10.0 (2400) | 8.0 (2400) | **-2.0** | [-2.0, -2.0] |
| earth/quicksand_cast_multi_removal | 91.3 (23) | 77.3 (22) | -14.0 | [-36.7, +6.9] |
| earth/quicksand_cast_per_legal_turn | 37.7 (61) | 29.7 (74) | -8.0 | [-20.7, +5.9] |
| gate/Stonehaven/grant_then_block | 38.0 (184) | 39.8 (161) | +1.7 | [-5.9, +8.9] |
| gate/Stonehaven/portal_then_grant | 90.6 (203) | 74.5 (216) | **-16.1** | [-24.9, -7.8] |
| gate/portal_per_legal_turn | 99.5 (204) | 99.5 (217) | +0.0 | [-1.4, +1.4] |
| heal/heal_spell_per_legal_turn_below_max_hp | 15.9 (107) | 7.5 (40) | -8.4 | [-17.5, +3.1] |
| ikz/held_at_end_of_own_turn | 28.9 (367) | 27.1 (351) | -1.8 | [-6.9, +3.3] |
| ikz/held_then_spent_in_opp_turn | 0.0 (106) | 0.0 (95) | +0.0 | [+0.0, +0.0] |
| leader/Bobu/use_then_heal_observed | 18.8 (16) | 33.3 (3) | +14.6 | [-25.0, +38.9] |
| leader/use_per_legal_turn | 4.4 (367) | 0.9 (351) | **-3.5** | [-5.8, -1.5] |
| outcome/own_turns | 7.65 (48) | 7.31 (48) | -0.3 | [-0.8, +0.1] |
| outcome/win | 60.4 (48) | 56.2 (48) | -4.2 | [-14.6, +6.2] |
| response/defender_declared_when_legal | 80.5 (118) | 76.2 (101) | -4.3 | [-14.3, +8.2] |
| sequence/earth.bobu_before_destruction/completed_per_eligible_game | 6.2 (48) | 2.1 (48) | -4.2 | [-10.4, +0.0] |
| sequence/earth.bobu_before_destruction/converted_per_eligible_game | 6.2 (48) | 2.1 (48) | -4.2 | [-10.4, +0.0] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 84.1 (44) | 78.6 (42) | -5.5 | [-18.9, +8.5] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 59.1 (44) | 50.0 (42) | -9.1 | [-28.6, +10.1] |
| strategy/attacks_by_equipped_attacker | 6.7 (578) | 5.2 (538) | -1.5 | [-3.6, +0.6] |
| strategy/face_target_share_when_both_legal | 47.4 (304) | 41.1 (292) | -6.3 | [-17.5, +5.3] |
| strategy/favorable_trade_taken_per_available_turn | 61.2 (116) | 61.0 (136) | -0.2 | [-12.0, +12.1] |
| strategy/spell_cast_per_legal_main_turn | 26.2 (164) | 23.9 (109) | -2.4 | [-10.2, +7.8] |

### u9305 vs u8223 — argmax — free_draft|Stonehaven/Goro

| metric | u8223 | u9305 | Δ | 95% CI |
|---|---|---|---|---|
| block/attacker_destroyed | 34.6 (104) | 41.4 (70) | +6.8 | [-5.3, +18.8] |
| block/blocker_survives | 43.3 (104) | 51.4 (70) | +8.2 | [-4.3, +18.8] |
| block/favorable_survive_or_kill | 55.8 (104) | 61.4 (70) | +5.7 | [-5.9, +16.9] |
| draft/max_wjaccard_to_curated | 32.5 (48) | 31.0 (48) | **-1.6** | [-1.6, -1.6] |
| draft/mean_cost | 3.36 (2400) | 3.72 (2400) | **+0.4** | [+0.4, +0.4] |
| draft/normal_share | 58.0 (2400) | 64.0 (2400) | **+6.0** | [+6.0, +6.0] |
| draft/spell_share | 14.0 (2400) | 10.0 (2400) | **-4.0** | [-4.0, -4.0] |
| draft/unique_cards | 17.00 (48) | 16.00 (48) | **-1.0** | [-1.0, -1.0] |
| draft/weapon_share | 10.0 (2400) | 8.0 (2400) | **-2.0** | [-2.0, -2.0] |
| earth/quicksand_cast_multi_removal | 87.5 (32) | 81.5 (27) | -6.0 | [-29.1, +14.3] |
| earth/quicksand_cast_per_legal_turn | 40.5 (79) | 35.1 (77) | -5.4 | [-20.2, +8.0] |
| gate/Stonehaven/grant_then_block | 43.9 (180) | 32.9 (158) | **-11.0** | [-20.5, -1.2] |
| gate/Stonehaven/portal_then_grant | 85.7 (210) | 72.5 (218) | **-13.2** | [-20.7, -5.5] |
| gate/portal_per_legal_turn | 100.0 (210) | 100.0 (218) | +0.0 | [+0.0, +0.0] |
| heal/heal_spell_per_legal_turn_below_max_hp | 25.5 (106) | 6.5 (62) | **-19.0** | [-30.1, -8.2] |
| ikz/held_at_end_of_own_turn | 25.1 (378) | 24.9 (365) | -0.2 | [-5.8, +5.4] |
| ikz/held_then_spent_in_opp_turn | 0.0 (95) | 0.0 (91) | +0.0 | [+0.0, +0.0] |
| leader/Goro/target_then_attacks_same_turn | 7.1 (28) | 16.7 (12) | +9.5 | [-10.8, +50.0] |
| leader/Goro/use_with_target | 100.0 (28) | 100.0 (12) | +0.0 | [+0.0, +0.0] |
| leader/use_per_legal_turn | 8.8 (317) | 4.0 (302) | **-4.9** | [-8.7, -1.0] |
| outcome/own_turns | 7.88 (48) | 7.60 (48) | -0.3 | [-0.8, +0.2] |
| outcome/win | 54.2 (48) | 58.3 (48) | +4.2 | [-10.4, +20.8] |
| response/defender_declared_when_legal | 74.8 (139) | 66.0 (106) | -8.8 | [-23.2, +4.4] |
| sequence/earth.stone_defender_portal_attack/completed_per_eligible_game | 75.0 (44) | 77.5 (40) | +2.5 | [-11.1, +16.9] |
| sequence/earth.stone_defender_portal_attack/converted_per_eligible_game | 52.3 (44) | 45.0 (40) | -7.3 | [-30.6, +15.9] |
| strategy/attacks_by_equipped_attacker | 6.5 (541) | 5.3 (488) | -1.1 | [-3.8, +1.0] |
| strategy/face_target_share_when_both_legal | 57.8 (287) | 50.2 (255) | -7.6 | [-16.1, +0.7] |
| strategy/favorable_trade_taken_per_available_turn | 50.0 (126) | 51.5 (134) | +1.5 | [-8.1, +11.7] |
| strategy/spell_cast_per_legal_main_turn | 34.7 (176) | 23.5 (136) | **-11.1** | [-20.3, -2.0] |

