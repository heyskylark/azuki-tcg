#include "utils/actions_util.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils/cli_rendering_util.h"

#define ACTION_VALUE_COUNT AZK_USER_ACTION_VALUE_COUNT

bool verify_user_action_player(const GameState *gs, const UserAction *action) {
  if (!gs || !action) {
    return false;
  }

  return gs->players[gs->active_player_index] == action->player;
}

static bool parse_action_values(const char *input, int values[ACTION_VALUE_COUNT]) {
  if (!input || !values) {
    return false;
  }

  int count = 0;
  const char *cursor = input;

  while (*cursor != '\0') {
    while (isspace((unsigned char)*cursor)) {
      cursor++;
    }

    if (*cursor == '\0' || *cursor == '\n') {
      break;
    }

    char *endptr = NULL;
    long parsed = strtol(cursor, &endptr, 10);
    if (cursor == endptr) {
      return false;
    }

    if (parsed < INT_MIN || parsed > INT_MAX) {
      return false;
    }

    if (count >= ACTION_VALUE_COUNT) {
      return false;
    }

    values[count++] = (int)parsed;
    cursor = endptr;

    while (isspace((unsigned char)*cursor)) {
      cursor++;
    }

    if (*cursor == ',') {
      cursor++;
      continue;
    }

    if (*cursor == '\0' || *cursor == '\n') {
      break;
    }

    return false;
  }

  return count == ACTION_VALUE_COUNT;
}

bool azk_parse_user_action_values(ecs_world_t *world, const int values[ACTION_VALUE_COUNT], UserAction *out_action) {
  if (!world || !values) {
    return false;
  }

  const GameState *state = ecs_singleton_get(world, GameState);
  ecs_assert(state != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  int active_index = state->active_player_index;
  ecs_assert(active_index >= 0 && active_index < MAX_PLAYERS_PER_MATCH, ECS_INVALID_PARAMETER, "Active player index out of range");

  ActionType type = (ActionType)values[0];
  if (type < ACT_NOOP || type > ACT_MULLIGAN_SHUFFLE) {
    return false;
  }

  ecs_entity_t active_player = state->players[active_index];
  ecs_assert(active_player != 0, ECS_INVALID_PARAMETER, "Active player entity is invalid");

  UserAction action = {
    .player = active_player,
    .type = type,
    .subaction_1 = values[1],
    .subaction_2 = values[2],
    .subaction_3 = values[3]
  };

  if (out_action) {
    *out_action = action;
  }

  return true;
}

bool azk_parse_user_action_string(ecs_world_t *world, const char *input, UserAction *out_action) {
  if (!world || !input) {
    return false;
  }

  int values[ACTION_VALUE_COUNT];
  if (!parse_action_values(input, values)) {
    return false;
  }

  return azk_parse_user_action_values(world, values, out_action);
}

void azk_store_user_action(ecs_world_t *world, const UserAction *action) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World pointer is null");
  ecs_assert(action != NULL, ECS_INVALID_PARAMETER, "UserAction pointer is null");

  ActionContext *ctx = ecs_singleton_get_mut(world, ActionContext);
  ecs_assert(ctx != NULL, ECS_INVALID_PARAMETER, "ActionContext singleton missing");

  // New action submission starts a fresh validity window.
  ctx->invalid_action = false;
  ctx->user_action = *action;

  ctx->user_action_history[ctx->history_head] = *action;
  if (ctx->history_size < MAX_USER_ACTION_HISTORY_SIZE) {
    ctx->history_size++;
  }
  ctx->history_head = (ctx->history_head + 1) % MAX_USER_ACTION_HISTORY_SIZE;

  ecs_singleton_modified(world, ActionContext);
}

void azk_block_for_user_action(ecs_world_t *world) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World pointer is null");

  char buffer[128];
  char message[128] = {0};
  while (true) {
    const GameState *state = ecs_singleton_get(world, GameState);
    ecs_assert(state != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

    const char *message_ptr = message[0] != '\0' ? message : NULL;
    if (!cli_render_prompt_user_action(state->active_player_index, message_ptr, buffer, sizeof buffer)) {
      snprintf(message, sizeof message, "Input error. Please try again.");
      continue;
    }
    message[0] = '\0';

    UserAction action;
    if (!azk_parse_user_action_string(world, buffer, &action)) {
      snprintf(message, sizeof message, "Invalid input. Enter four comma-separated integers (type,p0,p1,p2).");
      continue;
    }

    azk_store_user_action(world, &action);
    break;
  }
}
