#include "validation/action_enumerator.h"

#include <string.h>

#include "validation/action_schema.h"
#include "validation/action_validation.h"
#include "utils/phase_utils.h"

static bool validate_action_for_mask(
  ecs_world_t *world,
  const GameState *gs,
  const AbilityContext *actx,
  ecs_entity_t player,
  const UserAction *action
) {
  switch (action->type) {
    case ACT_PLAY_ENTITY_TO_GARDEN:
      return azk_validate_play_entity_action(world, gs, player, ZONE_GARDEN, action, false, NULL);
    case ACT_PLAY_ENTITY_TO_ALLEY:
      return azk_validate_play_entity_action(world, gs, player, ZONE_ALLEY, action, false, NULL);
    case ACT_GATE_PORTAL:
      return azk_validate_gate_portal_action(world, gs, player, action, false, NULL);
    case ACT_ATTACK:
      return azk_validate_attack_action(world, gs, player, action, false, NULL);
    case ACT_ATTACH_WEAPON_FROM_HAND:
      return azk_validate_attach_weapon_action(world, gs, player, action, false, NULL);
    case ACT_ACTIVATE_ABILITY:
      return azk_validate_activate_ability_action(world, gs, player, action, false, NULL);
    case ACT_SELECT_COST_TARGET:
    case ACT_SELECT_EFFECT_TARGET:
      return azk_validate_select_ability_target_action(world, gs, actx, player, action, false, NULL);
    case ACT_ACCEPT_TRIGGERED_ABILITY:
      return azk_validate_accept_triggered_ability_action(world, gs, actx, player, action, false);
    case ACT_NOOP:
    case ACT_MULLIGAN_SHUFFLE:
      return azk_validate_simple_action(world, gs, player, action->type, false);
    default:
      return false;
  }
}

static void enumerate_parameters(
  ecs_world_t *world,
  const GameState *gs,
  const AbilityContext *actx,
  ecs_entity_t player,
  const AzkActionSpec *spec,
  int param_index,
  UserAction *action,
  AzkActionMaskSet *out_mask
) {
  if (param_index >= 3) {
    if (validate_action_for_mask(world, gs, actx, player, action)) {
      if (out_mask->legal_action_count < AZK_MAX_LEGAL_ACTIONS) {
        out_mask->legal_actions[out_mask->legal_action_count++] = *action;
      }
      out_mask->head0_mask[action->type] = 1;
    }
    return;
  }

  const AzkActionParamSpec *param = &spec->params[param_index];
  int *target_subaction = NULL;
  switch (param_index) {
    case 0: target_subaction = &action->subaction_1; break;
    case 1: target_subaction = &action->subaction_2; break;
    case 2: target_subaction = &action->subaction_3; break;
    default: return;
  }

  if (param->kind == AZK_ACTION_PARAM_UNUSED) {
    *target_subaction = 0;
    enumerate_parameters(world, gs, actx, player, spec, param_index + 1, action, out_mask);
    return;
  }

  for (int value = param->min_value; value <= param->max_value; ++value) {
    *target_subaction = value;
    enumerate_parameters(world, gs, actx, player, spec, param_index + 1, action, out_mask);
  }
}

bool azk_build_action_mask_for_player(
  ecs_world_t *world,
  const GameState *gs,
  int8_t player_index,
  AzkActionMaskSet *out_mask
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(out_mask != NULL, ECS_INVALID_PARAMETER, "Output mask pointer is null");

  memset(out_mask, 0, sizeof(*out_mask));

  if (player_index < 0 || player_index >= MAX_PLAYERS_PER_MATCH) {
    return false;
  }

  const AbilityContext *actx = ecs_singleton_get(world, AbilityContext);
  ecs_entity_t player = gs->players[player_index];
  bool selection_pending = actx != NULL && actx->has_pending && actx->player == player;

  if (gs->winner != -1) {
    return true;
  }

  if (!selection_pending && !phase_requires_user_action(gs->phase)) {
    return true;
  }

  if (player_index != gs->active_player_index) {
    return true;
  }

  if (selection_pending) {
    ActionType selection_type = ACT_NOOP;
    if (actx->phase == ABILITY_SELECTION_COST) {
      selection_type = ACT_SELECT_COST_TARGET;
    } else if (actx->phase == ABILITY_SELECTION_EFFECT) {
      selection_type = ACT_SELECT_EFFECT_TARGET;
    }
    size_t spec_count = 0;
    const AzkActionSpec *specs = azk_get_action_specs(&spec_count);
    for (size_t i = 0; i < spec_count; ++i) {
      const AzkActionSpec *spec = &specs[i];
      if (actx->phase == ABILITY_SELECTION_PROMPT) {
        if (spec->type != ACT_ACCEPT_TRIGGERED_ABILITY) {
          continue;
        }
      } else if (spec->type != selection_type) {
        continue;
      }
      if ((spec->phase_mask & AZK_PHASE_MASK(gs->phase)) == 0) {
        continue;
      }

      UserAction action = {
        .player = player,
        .type = spec->type,
        .subaction_1 = 0,
        .subaction_2 = 0,
        .subaction_3 = 0
      };
      enumerate_parameters(world, gs, actx, player, spec, 0, &action, out_mask);
    }

    // Allow NOOP if it validates (e.g., to end optional selections)
    UserAction noop = { .player = player, .type = ACT_NOOP, .subaction_1 = 0, .subaction_2 = 0, .subaction_3 = 0 };
    if (validate_action_for_mask(world, gs, actx, player, &noop)) {
      out_mask->legal_actions[out_mask->legal_action_count++] = noop;
      out_mask->head0_mask[noop.type] = 1;
    }

    return true;
  }

  size_t spec_count = 0;
  const AzkActionSpec *specs = azk_get_action_specs(&spec_count);

  for (size_t i = 0; i < spec_count; ++i) {
    const AzkActionSpec *spec = &specs[i];
    if ((spec->phase_mask & AZK_PHASE_MASK(gs->phase)) == 0) {
      continue;
    }

    UserAction action = {
      .player = player,
      .type = spec->type,
      .subaction_1 = 0,
      .subaction_2 = 0,
      .subaction_3 = 0
    };

    enumerate_parameters(world, gs, actx, player, spec, 0, &action, out_mask);
  }

  return true;
}
