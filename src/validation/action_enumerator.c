#include "validation/action_enumerator.h"

#include <string.h>

#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "generated/card_defs.h"
#include "validation/action_schema.h"
#include "validation/action_validation.h"
#include "utils/phase_utils.h"
#include "utils/player_util.h"

// Helper to add a valid action to the mask set
static void add_valid_action(AzkActionMaskSet *out_mask, const UserAction *action) {
  if (out_mask->legal_action_count < AZK_MAX_LEGAL_ACTIONS) {
    out_mask->legal_actions[out_mask->legal_action_count++] = *action;
  }
  out_mask->head0_mask[action->type] = 1;
}

// Enumerate ability actions based on current ability phase
static void enumerate_ability_actions(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  AzkActionMaskSet *out_mask
) {
  const AbilityContext* ctx = ecs_singleton_get(world, AbilityContext);
  if (!ctx || ctx->phase == ABILITY_PHASE_NONE) {
    return;
  }

  UserAction action = {
    .player = player,
    .type = ACT_NOOP,
    .subaction_1 = 0,
    .subaction_2 = 0,
    .subaction_3 = 0
  };

  switch (ctx->phase) {
    case ABILITY_PHASE_CONFIRMATION:
      // Can confirm or decline (NOOP)
      action.type = ACT_CONFIRM_ABILITY;
      add_valid_action(out_mask, &action);

      action.type = ACT_NOOP;
      add_valid_action(out_mask, &action);
      break;

    case ABILITY_PHASE_COST_SELECTION: {
      // Get ability def to know target type
      const CardId* card_id = ecs_get(world, ctx->source_card, CardId);
      if (!card_id) break;

      const AbilityDef* def = azk_get_ability_def(card_id->id);
      if (!def) break;

      action.type = ACT_SELECT_COST_TARGET;

      // Enumerate valid cost targets based on type
      uint8_t player_num = get_player_number(world, ctx->owner);
      switch (def->cost_req.type) {
        case ABILITY_TARGET_FRIENDLY_HAND: {
          ecs_entity_t hand = gs->zones[player_num].hand;
          ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
          for (int i = 0; i < hand_cards.count; i++) {
            ecs_entity_t target = hand_cards.ids[i];
            if (def->validate_cost_target &&
                !def->validate_cost_target(world, ctx->source_card, ctx->owner, target)) {
              continue;
            }
            action.subaction_1 = i;
            add_valid_action(out_mask, &action);
          }
          break;
        }
        // Add other target types as needed
        default:
          break;
      }
      break;
    }

    case ABILITY_PHASE_EFFECT_SELECTION: {
      // Get ability def to know target type
      const CardId* card_id = ecs_get(world, ctx->source_card, CardId);
      if (!card_id) break;

      const AbilityDef* def = azk_get_ability_def(card_id->id);
      if (!def) break;

      action.type = ACT_SELECT_EFFECT_TARGET;

      // Enumerate valid effect targets based on type
      uint8_t player_num = get_player_number(world, ctx->owner);
      switch (def->effect_req.type) {
        case ABILITY_TARGET_FRIENDLY_HAND: {
          ecs_entity_t hand = gs->zones[player_num].hand;
          ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
          for (int i = 0; i < hand_cards.count; i++) {
            ecs_entity_t target = hand_cards.ids[i];
            if (def->validate_effect_target &&
                !def->validate_effect_target(world, ctx->source_card, ctx->owner, target)) {
              continue;
            }
            action.subaction_1 = i;
            add_valid_action(out_mask, &action);
          }
          break;
        }
        case ABILITY_TARGET_ENEMY_GARDEN_ENTITY: {
          uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
          ecs_entity_t garden = gs->zones[enemy_num].garden;
          ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
          for (int i = 0; i < garden_cards.count; i++) {
            ecs_entity_t target = garden_cards.ids[i];
            if (def->validate_effect_target &&
                !def->validate_effect_target(world, ctx->source_card, ctx->owner, target)) {
              continue;
            }
            action.subaction_1 = i;
            add_valid_action(out_mask, &action);
          }
          break;
        }
        case ABILITY_TARGET_ANY_GARDEN_ENTITY: {
          // Index encoding: 0-4 = self garden, 5-9 = opponent garden
          // Check self garden
          ecs_entity_t self_garden = gs->zones[player_num].garden;
          ecs_entities_t self_cards = ecs_get_ordered_children(world, self_garden);
          for (int i = 0; i < self_cards.count; i++) {
            ecs_entity_t target = self_cards.ids[i];
            const ZoneIndex* zi = ecs_get(world, target, ZoneIndex);
            if (!zi) continue;
            if (def->validate_effect_target &&
                !def->validate_effect_target(world, ctx->source_card, ctx->owner, target)) {
              continue;
            }
            action.subaction_1 = zi->index;  // 0-4 for self garden
            add_valid_action(out_mask, &action);
          }
          // Check opponent garden
          uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
          ecs_entity_t enemy_garden = gs->zones[enemy_num].garden;
          ecs_entities_t enemy_cards = ecs_get_ordered_children(world, enemy_garden);
          for (int i = 0; i < enemy_cards.count; i++) {
            ecs_entity_t target = enemy_cards.ids[i];
            const ZoneIndex* zi = ecs_get(world, target, ZoneIndex);
            if (!zi) continue;
            if (def->validate_effect_target &&
                !def->validate_effect_target(world, ctx->source_card, ctx->owner, target)) {
              continue;
            }
            action.subaction_1 = zi->index + GARDEN_SIZE;  // 5-9 for opponent garden
            add_valid_action(out_mask, &action);
          }
          break;
        }
        default:
          break;
      }
      break;
    }

    default:
      break;
  }
}

static bool validate_action_for_mask(
  ecs_world_t *world,
  const GameState *gs,
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
    case ACT_PLAY_SPELL_FROM_HAND:
      return azk_validate_play_spell_action(world, gs, player, action, false, NULL);
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
  ecs_entity_t player,
  const AzkActionSpec *spec,
  int param_index,
  UserAction *action,
  AzkActionMaskSet *out_mask
) {
  if (param_index >= 3) {
    if (validate_action_for_mask(world, gs, player, action)) {
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
    enumerate_parameters(world, gs, player, spec, param_index + 1, action, out_mask);
    return;
  }

  for (int value = param->min_value; value <= param->max_value; ++value) {
    *target_subaction = value;
    enumerate_parameters(world, gs, player, spec, param_index + 1, action, out_mask);
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

  if (gs->winner != -1 || !phase_requires_user_action(gs->phase)) {
    return true;
  }

  if (player_index != gs->active_player_index) {
    return true;
  }

  ecs_entity_t player = gs->players[player_index];

  // Check if we're in an ability sub-phase
  if (azk_is_in_ability_phase(world)) {
    enumerate_ability_actions(world, gs, player, out_mask);
    return true;
  }

  // Normal phase enumeration
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

    enumerate_parameters(world, gs, player, spec, 0, &action, out_mask);
  }

  return true;
}
