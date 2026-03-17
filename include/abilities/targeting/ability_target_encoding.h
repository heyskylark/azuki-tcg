#ifndef AZUKI_ABILITY_TARGET_ENCODING_H
#define AZUKI_ABILITY_TARGET_ENCODING_H

#include <stdbool.h>

int azk_encode_enemy_leader_or_garden_target_index(bool is_leader,
                                                   int zone_index);

bool azk_decode_enemy_leader_or_garden_target_index(int action_index,
                                                    bool *out_is_leader,
                                                    int *out_zone_index);

int azk_encode_any_garden_target_index(bool is_enemy, int zone_index);

bool azk_decode_any_garden_target_index(int action_index, bool *out_is_enemy,
                                        int *out_zone_index);

int azk_encode_any_leader_target_index(bool is_enemy);

bool azk_decode_any_leader_target_index(int action_index, bool *out_is_enemy);

#endif // AZUKI_ABILITY_TARGET_ENCODING_H
