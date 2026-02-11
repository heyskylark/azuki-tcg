#include "validation/action_enumerator.h"

#include <stdio.h>
#include <string.h>

#include "abilities/ability_registry.h"
#include "utils/debug_log.h"
#include "abilities/ability_system.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/phase_utils.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"
#include "validation/action_schema.h"
#include "validation/action_validation.h"

// Helper to add a valid action to the mask set
static void add_valid_action(AzkActionMaskSet *out_mask,
                             const UserAction *action) {
  if (out_mask->legal_action_count < AZK_MAX_LEGAL_ACTIONS) {
    out_mask->legal_actions[out_mask->legal_action_count++] = *action;
  }
  out_mask->head0_mask[action->type] = 1;
}

// Enumerate ability actions based on current ability phase
static void enumerate_ability_actions(ecs_world_t *world, const GameState *gs,
                                      ecs_entity_t player,
                                      AzkActionMaskSet *out_mask) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (!ctx || ctx->phase == ABILITY_PHASE_NONE) {
    return;
  }

  UserAction action = {.player = player,
                       .type = ACT_NOOP,
                       .subaction_1 = 0,
                       .subaction_2 = 0,
                       .subaction_3 = 0};

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
    const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
    if (!card_id)
      break;

    const AbilityDef *def = azk_get_ability_def(card_id->id);
    if (!def)
      break;

    action.type = ACT_SELECT_COST_TARGET;

    // Enumerate valid cost targets based on type
    uint8_t player_num = get_player_number(world, ctx->owner);
    switch (def->cost_req.type) {
    case ABILITY_TARGET_FRIENDLY_HAND:
    case ABILITY_TARGET_FRIENDLY_HAND_WEAPON: {
      ecs_entity_t hand = gs->zones[player_num].hand;
      ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
      for (int i = 0; i < hand_cards.count; i++) {
        ecs_entity_t target = hand_cards.ids[i];
        if (def->validate_cost_target &&
            !def->validate_cost_target(world, ctx->source_card, ctx->owner,
                                       target)) {
          continue;
        }
        action.subaction_1 = i;
        add_valid_action(out_mask, &action);
      }
      break;
    }
    case ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY: {
      ecs_entity_t garden = gs->zones[player_num].garden;
      ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
      for (int i = 0; i < garden_cards.count; i++) {
        ecs_entity_t target = garden_cards.ids[i];
        const ZoneIndex *zi = ecs_get(world, target, ZoneIndex);
        if (!zi)
          continue;
        if (def->validate_cost_target &&
            !def->validate_cost_target(world, ctx->source_card, ctx->owner,
                                       target)) {
          continue;
        }
        action.subaction_1 = zi->index;
        add_valid_action(out_mask, &action);
      }
      break;
    }
    default:
      break;
    }
    break;
  }

  case ABILITY_PHASE_EFFECT_SELECTION: {
    // Get ability def to know target type
    const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
    if (!card_id)
      break;

    const AbilityDef *def = azk_get_ability_def(card_id->id);
    if (!def)
      break;

    // Allow skipping effect selection if min is 0 ("up to" effects)
    if (ctx->effect_min == 0 && ctx->effect_filled == 0) {
      action.type = ACT_NOOP;
      add_valid_action(out_mask, &action);
    }

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
            !def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                         target)) {
          continue;
        }
        action.subaction_1 = i;
        add_valid_action(out_mask, &action);
      }
      break;
    }
    case ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY: {
      ecs_entity_t garden = gs->zones[player_num].garden;
      ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
      for (int i = 0; i < garden_cards.count; i++) {
        ecs_entity_t target = garden_cards.ids[i];
        const ZoneIndex *zi = ecs_get(world, target, ZoneIndex);
        if (!zi)
          continue;
        if (def->validate_effect_target &&
            !def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                         target)) {
          continue;
        }
        action.subaction_1 = zi->index;
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
        const ZoneIndex *zi = ecs_get(world, target, ZoneIndex);
        if (!zi)
          continue;
        if (def->validate_effect_target &&
            !def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                         target)) {
          continue;
        }
        action.subaction_1 = zi->index;
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
        const ZoneIndex *zi = ecs_get(world, target, ZoneIndex);
        if (!zi)
          continue;
        if (def->validate_effect_target &&
            !def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                         target)) {
          continue;
        }
        action.subaction_1 = zi->index; // 0-4 for self garden
        add_valid_action(out_mask, &action);
      }
      // Check opponent garden
      uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
      ecs_entity_t enemy_garden = gs->zones[enemy_num].garden;
      ecs_entities_t enemy_cards =
          ecs_get_ordered_children(world, enemy_garden);
      for (int i = 0; i < enemy_cards.count; i++) {
        ecs_entity_t target = enemy_cards.ids[i];
        const ZoneIndex *zi = ecs_get(world, target, ZoneIndex);
        if (!zi)
          continue;
        if (def->validate_effect_target &&
            !def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                         target)) {
          continue;
        }
        action.subaction_1 = zi->index + GARDEN_SIZE; // 5-9 for opponent garden
        add_valid_action(out_mask, &action);
      }
      break;
    }
    case ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY: {
      // Index encoding: 0-4 = opponent garden slots (by ZoneIndex), 5 = opponent leader
      uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;

      // Check opponent garden entities
      ecs_entity_t enemy_garden = gs->zones[enemy_num].garden;
      ecs_entities_t enemy_cards =
          ecs_get_ordered_children(world, enemy_garden);
      for (int i = 0; i < enemy_cards.count; i++) {
        ecs_entity_t target = enemy_cards.ids[i];
        const ZoneIndex *zi = ecs_get(world, target, ZoneIndex);
        if (!zi)
          continue;
        if (def->validate_effect_target &&
            !def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                         target)) {
          continue;
        }
        action.subaction_1 = zi->index; // 0-4 for opponent garden
        add_valid_action(out_mask, &action);
      }

      // Check opponent leader (index 5)
      ecs_entity_t leader =
          find_leader_card_in_zone(world, gs->zones[enemy_num].leader);
      if (leader != 0) {
        if (!def->validate_effect_target ||
            def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                        leader)) {
          action.subaction_1 = GARDEN_SIZE; // 5 for leader
          add_valid_action(out_mask, &action);
        }
      }
      break;
    }
    case ABILITY_TARGET_ANY_LEADER: {
      // Index encoding: 0 = friendly leader, 1 = enemy leader
      // Check friendly leader (index 0)
      ecs_entity_t friendly_leader =
          find_leader_card_in_zone(world, gs->zones[player_num].leader);
      if (friendly_leader != 0) {
        if (!def->validate_effect_target ||
            def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                        friendly_leader)) {
          action.subaction_1 = 0; // 0 for friendly leader
          add_valid_action(out_mask, &action);
        }
      }

      // Check enemy leader (index 1)
      uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
      ecs_entity_t enemy_leader =
          find_leader_card_in_zone(world, gs->zones[enemy_num].leader);
      if (enemy_leader != 0) {
        if (!def->validate_effect_target ||
            def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                        enemy_leader)) {
          action.subaction_1 = 1; // 1 for enemy leader
          add_valid_action(out_mask, &action);
        }
      }
      break;
    }
    default:
      break;
    }
    break;
  }

  case ABILITY_PHASE_SELECTION_PICK: {
    // Get ability def for validation
    const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
    if (!card_id)
      break;

    const AbilityDef *def = azk_get_ability_def(card_id->id);
    if (!def)
      break;

    uint8_t owner_player_num = get_player_number(world, ctx->owner);
    bool alley_slot_occupied[ALLEY_SIZE] = {false};
    bool alley_full = false;
    if (def->can_select_to_alley) {
      ecs_entity_t alley_zone = gs->zones[owner_player_num].alley;
      ecs_entities_t alley_cards = ecs_get_ordered_children(world, alley_zone);
      alley_full = alley_cards.count >= ALLEY_SIZE;
      for (int i = 0; i < alley_cards.count; i++) {
        const ZoneIndex *zi = ecs_get(world, alley_cards.ids[i], ZoneIndex);
        if (zi && zi->index < ALLEY_SIZE) {
          alley_slot_occupied[zi->index] = true;
        }
      }
    }

    // Always allow skipping selection pick for "up to" effects
    action.type = ACT_NOOP;
    add_valid_action(out_mask, &action);

    // Enumerate valid selection targets
    for (int i = 0; i < ctx->selection_count; i++) {
      ecs_entity_t target = ctx->selection_cards[i];
      if (target == 0)
        continue; // Already picked or empty

      // Validate against selection target validator if defined
      if (def->validate_selection_target &&
          !def->validate_selection_target(world, ctx->source_card, ctx->owner,
                                          target)) {
        continue;
      }

      // Add ACT_SELECT_FROM_SELECTION (add to hand) if allowed
      // If ability has special selection modes (alley/equip), only add if
      // can_select_to_hand is explicitly true. Otherwise, use default behavior.
      bool has_special_selection = def->can_select_to_alley || def->can_select_to_equip;
      if (!has_special_selection || def->can_select_to_hand) {
        action.type = ACT_SELECT_FROM_SELECTION;
        action.subaction_1 = i;
        action.subaction_2 = 0;
        add_valid_action(out_mask, &action);
      }

      // If can_select_to_alley and target is an entity, enumerate alley slots
      if (def->can_select_to_alley && is_card_type(world, target, CARD_TYPE_ENTITY)) {
        action.type = ACT_SELECT_TO_ALLEY;
        action.subaction_1 = i; // selection index
        for (int slot = 0; slot < ALLEY_SIZE; slot++) {
          // Mirror ability_system.c: occupied slots are only legal when alley
          // is full (forced replacement).
          if (alley_slot_occupied[slot] && !alley_full) {
            continue;
          }
          action.subaction_2 = slot; // alley slot index
          add_valid_action(out_mask, &action);
        }
      }

      // If can_select_to_equip and target is a weapon, enumerate garden/leader
      if (def->can_select_to_equip &&
          is_card_type(world, target, CARD_TYPE_WEAPON)) {
        uint8_t pnum = get_player_number(world, ctx->owner);
        action.type = ACT_SELECT_TO_EQUIP;
        action.subaction_1 = i; // selection index
        // Enumerate garden entities (slots 0-4)
        for (int slot = 0; slot < GARDEN_SIZE; slot++) {
          ecs_entity_t entity =
              find_card_in_zone_index(world, gs->zones[pnum].garden, slot);
          if (entity != 0) {
            action.subaction_2 = slot;
            add_valid_action(out_mask, &action);
          }
        }
        // Enumerate leader (slot 5)
        ecs_entity_t leader =
            find_leader_card_in_zone(world, gs->zones[pnum].leader);
        if (leader != 0) {
          action.subaction_2 = GARDEN_SIZE; // 5 for leader
          add_valid_action(out_mask, &action);
        }
      }
    }
    break;
  }

  case ABILITY_PHASE_BOTTOM_DECK: {
    // Enumerate remaining selection cards for bottom decking
    action.type = ACT_BOTTOM_DECK_CARD;
    for (int i = 0; i < ctx->selection_count; i++) {
      ecs_entity_t card = ctx->selection_cards[i];
      if (card == 0)
        continue; // Already bottom decked

      action.subaction_1 = i;
      add_valid_action(out_mask, &action);
    }

    // Also allow ACT_BOTTOM_DECK_ALL to finish quickly
    action.type = ACT_BOTTOM_DECK_ALL;
    action.subaction_1 = 0;
    add_valid_action(out_mask, &action);
    break;
  }

  default:
    break;
  }
}

static bool validate_action_for_mask(ecs_world_t *world, const GameState *gs,
                                     ecs_entity_t player,
                                     const UserAction *action) {
  bool result = false;
  switch (action->type) {
  case ACT_PLAY_ENTITY_TO_GARDEN:
    result = azk_validate_play_entity_action(world, gs, player, ZONE_GARDEN,
                                           action, false, NULL);
    break;
  case ACT_PLAY_ENTITY_TO_ALLEY:
    result = azk_validate_play_entity_action(world, gs, player, ZONE_ALLEY,
                                           action, false, NULL);
    break;
  case ACT_GATE_PORTAL:
    result = azk_validate_gate_portal_action(world, gs, player, action, false,
                                           NULL);
    break;
  case ACT_ATTACK:
    result = azk_validate_attack_action(world, gs, player, action, false, NULL);
    break;
  case ACT_ATTACH_WEAPON_FROM_HAND:
    result = azk_validate_attach_weapon_action(world, gs, player, action, false,
                                             NULL);
    break;
  case ACT_PLAY_SPELL_FROM_HAND:
    result = azk_validate_play_spell_action(world, gs, player, action, false,
                                          NULL);
    break;
  case ACT_ACTIVATE_ALLEY_ABILITY:
    result = azk_validate_activate_alley_ability_action(world, gs, player, action,
                                                      false, NULL);
    break;
  case ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY:
    result = azk_validate_activate_garden_or_leader_ability_action(
        world, gs, player, action, false, NULL);
    break;
  case ACT_DECLARE_DEFENDER:
    result = azk_validate_declare_defender_action(world, gs, player, action, false,
                                                NULL);
    break;
  case ACT_NOOP:
  case ACT_MULLIGAN_SHUFFLE:
    result = azk_validate_simple_action(world, gs, player, action->type, false);
    AZK_DEBUG_INFO("[ActionMask] validate_action_for_mask: type=%d, result=%d", action->type, result);
    break;
  default:
    AZK_DEBUG_WARN("[ActionMask] validate_action_for_mask: unknown type=%d", action->type);
    result = false;
  }
  return result;
}

static void enumerate_parameters(ecs_world_t *world, const GameState *gs,
                                 ecs_entity_t player, const AzkActionSpec *spec,
                                 int param_index, UserAction *action,
                                 AzkActionMaskSet *out_mask) {
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
  case 0:
    target_subaction = &action->subaction_1;
    break;
  case 1:
    target_subaction = &action->subaction_2;
    break;
  case 2:
    target_subaction = &action->subaction_3;
    break;
  default:
    return;
  }

  if (param->kind == AZK_ACTION_PARAM_UNUSED) {
    *target_subaction = 0;
    enumerate_parameters(world, gs, player, spec, param_index + 1, action,
                         out_mask);
    return;
  }

  for (int value = param->min_value; value <= param->max_value; ++value) {
    *target_subaction = value;
    enumerate_parameters(world, gs, player, spec, param_index + 1, action,
                         out_mask);
  }
}

bool azk_build_action_mask_for_player(ecs_world_t *world, const GameState *gs,
                                      int8_t player_index,
                                      AzkActionMaskSet *out_mask) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(out_mask != NULL, ECS_INVALID_PARAMETER,
             "Output mask pointer is null");

  memset(out_mask, 0, sizeof(*out_mask));

  if (player_index < 0 || player_index >= MAX_PLAYERS_PER_MATCH) {
    AZK_DEBUG_WARN("[ActionMask] player_index %d out of bounds", player_index);
    return false;
  }

  bool phase_requires_action = phase_requires_user_action(world, gs->phase);
  AZK_DEBUG_INFO("[ActionMask] phase=%d, winner=%d, phase_requires_action=%d, active_player=%d, player_index=%d",
                  gs->phase, gs->winner, phase_requires_action, gs->active_player_index, player_index);

  if (gs->winner != -1 || !phase_requires_action) {
    AZK_DEBUG_INFO("[ActionMask] Early return: game over or phase doesn't require action");
    return true;
  }

  if (player_index != gs->active_player_index) {
    AZK_DEBUG_INFO("[ActionMask] Early return: player %d is not active player %d", player_index, gs->active_player_index);
    return true;
  }

  ecs_entity_t player = gs->players[player_index];

  // Check if we're in an ability sub-phase
  if (azk_is_in_ability_phase(world)) {
    AZK_DEBUG_INFO("[ActionMask] In ability phase, enumerating ability actions");
    enumerate_ability_actions(world, gs, player, out_mask);
    return true;
  }

  // Normal phase enumeration
  size_t spec_count = 0;
  const AzkActionSpec *specs = azk_get_action_specs(&spec_count);

  AZK_DEBUG_INFO("[ActionMask] Enumerating %zu action specs for phase %d", spec_count, gs->phase);

  for (size_t i = 0; i < spec_count; ++i) {
    const AzkActionSpec *spec = &specs[i];
    if ((spec->phase_mask & AZK_PHASE_MASK(gs->phase)) == 0) {
      continue;
    }

    AZK_DEBUG_INFO("[ActionMask] Spec %zu (type=%d) matches phase mask", i, spec->type);

    UserAction action = {.player = player,
                         .type = spec->type,
                         .subaction_1 = 0,
                         .subaction_2 = 0,
                         .subaction_3 = 0};

    enumerate_parameters(world, gs, player, spec, 0, &action, out_mask);
  }

  AZK_DEBUG_INFO("[ActionMask] Final mask: legal_action_count=%d, head0_mask[0]=%d, head0_mask[23]=%d",
                  out_mask->legal_action_count, out_mask->head0_mask[0], out_mask->head0_mask[23]);

  return true;
}
