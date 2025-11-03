#ifndef AZUKI_UTILS_ACTIONS_UTIL_H
#define AZUKI_UTILS_ACTIONS_UTIL_H

#include <stdbool.h>

#include "components.h"

#define AZK_USER_ACTION_VALUE_COUNT 4

bool azk_parse_user_action_values(ecs_world_t *world, const int values[AZK_USER_ACTION_VALUE_COUNT], UserAction *out_action);
bool azk_parse_user_action_string(ecs_world_t *world, const char *input, UserAction *out_action);
void azk_store_user_action(ecs_world_t *world, const UserAction *action);
void azk_block_for_user_action(ecs_world_t *world);

#endif
