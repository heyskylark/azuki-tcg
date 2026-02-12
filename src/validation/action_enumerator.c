#include "validation/action_enumerator.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

typedef struct {
  bool initialized;
  bool enabled;
  bool verbose;
  uint64_t report_every;
  uint64_t mask_calls;
  uint64_t mask_total_ns;
  uint64_t early_returns;
  uint64_t ability_calls;
  uint64_t ability_total_ns;
  uint64_t normal_calls;
  uint64_t normal_total_ns;
  uint64_t validate_calls;
  uint64_t validate_success;
  uint64_t validate_calls_by_type[AZK_ACTION_TYPE_COUNT];
  uint64_t validate_success_by_type[AZK_ACTION_TYPE_COUNT];
} ActionMaskProfileState;

typedef struct {
  int hand_card_count;
  bool has_ikz_token;
} ActionEnumerationContext;

#define AZK_MAX_PLACEMENT_SLOTS \
  ((GARDEN_SIZE > ALLEY_SIZE) ? GARDEN_SIZE : ALLEY_SIZE)

static ActionMaskProfileState k_mask_profile = {0};

static uint64_t monotonic_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static bool env_enabled(const char *name) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return false;
  }
  if (value[0] == '0' && value[1] == '\0') {
    return false;
  }
  return true;
}

static void init_mask_profile_if_needed(void) {
  if (k_mask_profile.initialized) {
    return;
  }
  k_mask_profile.initialized = true;
  k_mask_profile.enabled = env_enabled("AZK_MASK_PROFILE");
  k_mask_profile.verbose = env_enabled("AZK_MASK_PROFILE_VERBOSE");

  const char *report_every = getenv("AZK_MASK_PROFILE_EVERY");
  if (report_every != NULL && report_every[0] != '\0') {
    char *end_ptr = NULL;
    unsigned long long parsed = strtoull(report_every, &end_ptr, 10);
    if (end_ptr != report_every && parsed > 0ull) {
      k_mask_profile.report_every = (uint64_t)parsed;
    }
  }
  if (k_mask_profile.report_every == 0) {
    k_mask_profile.report_every = 50000;
  }
}

static void maybe_report_mask_profile(void) {
  if (!k_mask_profile.enabled) {
    return;
  }
  if (k_mask_profile.mask_calls == 0 ||
      (k_mask_profile.mask_calls % k_mask_profile.report_every) != 0) {
    return;
  }

  const double avg_us =
      k_mask_profile.mask_total_ns / (double)k_mask_profile.mask_calls / 1000.0;
  const double ability_avg_us =
      k_mask_profile.ability_calls == 0
          ? 0.0
          : k_mask_profile.ability_total_ns /
                (double)k_mask_profile.ability_calls / 1000.0;
  const double normal_avg_us =
      k_mask_profile.normal_calls == 0
          ? 0.0
          : k_mask_profile.normal_total_ns /
                (double)k_mask_profile.normal_calls / 1000.0;
  const double validate_hit_rate =
      k_mask_profile.validate_calls == 0
          ? 0.0
          : (double)k_mask_profile.validate_success /
                (double)k_mask_profile.validate_calls;

  fprintf(stderr,
          "[MaskProfile] calls=%" PRIu64
          " avg_us=%.2f early=%" PRIu64 " ability=%" PRIu64
          " ability_avg_us=%.2f normal=%" PRIu64
          " normal_avg_us=%.2f validate=%" PRIu64
          " validate_ok=%" PRIu64 " validate_hit=%.3f\n",
          k_mask_profile.mask_calls, avg_us, k_mask_profile.early_returns,
          k_mask_profile.ability_calls, ability_avg_us,
          k_mask_profile.normal_calls, normal_avg_us,
          k_mask_profile.validate_calls, k_mask_profile.validate_success,
          validate_hit_rate);

  if (!k_mask_profile.verbose) {
    return;
  }

  for (int type = 0; type < AZK_ACTION_TYPE_COUNT; ++type) {
    uint64_t calls = k_mask_profile.validate_calls_by_type[type];
    if (calls == 0) {
      continue;
    }
    uint64_t successes = k_mask_profile.validate_success_by_type[type];
    const double hit_rate = (double)successes / (double)calls;
    fprintf(stderr,
            "  [MaskProfile] type=%d validate=%" PRIu64
            " validate_ok=%" PRIu64 " hit=%.3f\n",
            type, calls, successes, hit_rate);
  }
}

static bool action_supports_ikz_toggle(ActionType type, int param_index) {
  if (param_index != 2) {
    return false;
  }
  switch (type) {
  case ACT_PLAY_ENTITY_TO_GARDEN:
  case ACT_PLAY_ENTITY_TO_ALLEY:
  case ACT_ATTACH_WEAPON_FROM_HAND:
  case ACT_PLAY_SPELL_FROM_HAND:
    return true;
  default:
    return false;
  }
}

static bool player_has_ready_ikz_token(ecs_world_t *world,
                                       ecs_entity_t player) {
  const IKZToken *ikz_token = ecs_get(world, player, IKZToken);
  if (ikz_token == NULL || ikz_token->ikz_token == 0) {
    return false;
  }

  const TapState *tap_state = ecs_get(world, ikz_token->ikz_token, TapState);
  if (tap_state == NULL) {
    return false;
  }

  return tap_state->tapped == 0;
}

static bool resolve_param_bounds(const AzkActionSpec *spec, int param_index,
                                 const AzkActionParamSpec *param,
                                 const ActionEnumerationContext *ctx,
                                 int *out_min, int *out_max) {
  int min_value = param->min_value;
  int max_value = param->max_value;

  switch (param->kind) {
  case AZK_ACTION_PARAM_HAND_INDEX:
    if (ctx->hand_card_count <= 0) {
      return false;
    }
    if (max_value >= ctx->hand_card_count) {
      max_value = ctx->hand_card_count - 1;
    }
    break;
  case AZK_ACTION_PARAM_BOOL:
    if (!ctx->has_ikz_token &&
        action_supports_ikz_toggle(spec->type, param_index)) {
      min_value = 0;
      max_value = 0;
    }
    break;
  default:
    break;
  }

  if (max_value < min_value) {
    return false;
  }

  *out_min = min_value;
  *out_max = max_value;
  return true;
}

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
  init_mask_profile_if_needed();
  if (k_mask_profile.enabled) {
    k_mask_profile.validate_calls++;
    if (action->type >= 0 && action->type < AZK_ACTION_TYPE_COUNT) {
      k_mask_profile.validate_calls_by_type[action->type]++;
    }
  }

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

  if (k_mask_profile.enabled && result) {
    k_mask_profile.validate_success++;
    if (action->type >= 0 && action->type < AZK_ACTION_TYPE_COUNT) {
      k_mask_profile.validate_success_by_type[action->type]++;
    }
  }

  return result;
}

static void try_add_action_if_valid(ecs_world_t *world, const GameState *gs,
                                    ecs_entity_t player, UserAction *action,
                                    AzkActionMaskSet *out_mask) {
  if (!validate_action_for_mask(world, gs, player, action)) {
    return;
  }
  if (out_mask->legal_action_count < AZK_MAX_LEGAL_ACTIONS) {
    out_mask->legal_actions[out_mask->legal_action_count++] = *action;
  }
  out_mask->head0_mask[action->type] = 1;
}

static int collect_placement_slots(ecs_world_t *world, ecs_entity_t zone,
                                   int max_slots,
                                   int out_slots[AZK_MAX_PLACEMENT_SLOTS]) {
  bool occupied[AZK_MAX_PLACEMENT_SLOTS] = {false};
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  const bool zone_full = cards.count >= max_slots;

  for (int i = 0; i < cards.count; ++i) {
    const ZoneIndex *zi = ecs_get(world, cards.ids[i], ZoneIndex);
    if (zi == NULL || zi->index < 0 || zi->index >= max_slots) {
      continue;
    }
    occupied[zi->index] = true;
  }

  int slot_count = 0;
  for (int slot = 0; slot < max_slots; ++slot) {
    if (zone_full || !occupied[slot]) {
      out_slots[slot_count++] = slot;
    }
  }
  return slot_count;
}

static void enumerate_play_entity_actions(ecs_world_t *world,
                                          const GameState *gs,
                                          ecs_entity_t player,
                                          const ActionEnumerationContext *ctx,
                                          ActionType type,
                                          ZonePlacementType placement_type,
                                          AzkActionMaskSet *out_mask) {
  if (ctx->hand_card_count <= 0) {
    return;
  }

  const uint8_t player_num = get_player_number(world, player);
  const ecs_entity_t zone =
      placement_type == ZONE_GARDEN ? gs->zones[player_num].garden
                                    : gs->zones[player_num].alley;
  const int slot_limit = placement_type == ZONE_GARDEN ? GARDEN_SIZE : ALLEY_SIZE;

  int slot_candidates[AZK_MAX_PLACEMENT_SLOTS] = {0};
  const int slot_count =
      collect_placement_slots(world, zone, slot_limit, slot_candidates);
  if (slot_count == 0) {
    return;
  }

  int bool_values[2] = {0, 1};
  int bool_count = ctx->has_ikz_token ? 2 : 1;

  UserAction action = {.player = player,
                       .type = type,
                       .subaction_1 = 0,
                       .subaction_2 = 0,
                       .subaction_3 = 0};
  for (int hand_index = 0; hand_index < ctx->hand_card_count; ++hand_index) {
    action.subaction_1 = hand_index;
    for (int i = 0; i < slot_count; ++i) {
      action.subaction_2 = slot_candidates[i];
      for (int b = 0; b < bool_count; ++b) {
        action.subaction_3 = bool_values[b];
        try_add_action_if_valid(world, gs, player, &action, out_mask);
      }
    }
  }
}

static void enumerate_play_spell_actions(ecs_world_t *world, const GameState *gs,
                                         ecs_entity_t player,
                                         const ActionEnumerationContext *ctx,
                                         AzkActionMaskSet *out_mask) {
  if (ctx->hand_card_count <= 0) {
    return;
  }

  int bool_values[2] = {0, 1};
  int bool_count = ctx->has_ikz_token ? 2 : 1;

  UserAction action = {.player = player,
                       .type = ACT_PLAY_SPELL_FROM_HAND,
                       .subaction_1 = 0,
                       .subaction_2 = 0,
                       .subaction_3 = 0};
  for (int hand_index = 0; hand_index < ctx->hand_card_count; ++hand_index) {
    action.subaction_1 = hand_index;
    for (int b = 0; b < bool_count; ++b) {
      action.subaction_3 = bool_values[b];
      try_add_action_if_valid(world, gs, player, &action, out_mask);
    }
  }
}

static void enumerate_attach_weapon_actions(
    ecs_world_t *world, const GameState *gs, ecs_entity_t player,
    const ActionEnumerationContext *ctx, AzkActionMaskSet *out_mask) {
  if (ctx->hand_card_count <= 0) {
    return;
  }

  const uint8_t player_num = get_player_number(world, player);
  int target_indices[GARDEN_SIZE + 1] = {0};
  int target_count = 0;
  for (int slot = 0; slot < GARDEN_SIZE; ++slot) {
    if (find_card_in_zone_index(world, gs->zones[player_num].garden, slot) != 0) {
      target_indices[target_count++] = slot;
    }
  }
  if (find_leader_card_in_zone(world, gs->zones[player_num].leader) != 0) {
    target_indices[target_count++] = GARDEN_SIZE;
  }
  if (target_count == 0) {
    return;
  }

  int bool_values[2] = {0, 1};
  int bool_count = ctx->has_ikz_token ? 2 : 1;
  UserAction action = {.player = player,
                       .type = ACT_ATTACH_WEAPON_FROM_HAND,
                       .subaction_1 = 0,
                       .subaction_2 = 0,
                       .subaction_3 = 0};
  for (int hand_index = 0; hand_index < ctx->hand_card_count; ++hand_index) {
    action.subaction_1 = hand_index;
    for (int i = 0; i < target_count; ++i) {
      action.subaction_2 = target_indices[i];
      for (int b = 0; b < bool_count; ++b) {
        action.subaction_3 = bool_values[b];
        try_add_action_if_valid(world, gs, player, &action, out_mask);
      }
    }
  }
}

static void enumerate_gate_portal_actions(ecs_world_t *world, const GameState *gs,
                                          ecs_entity_t player,
                                          AzkActionMaskSet *out_mask) {
  const uint8_t player_num = get_player_number(world, player);

  int alley_indices[ALLEY_SIZE] = {0};
  int alley_count = 0;
  for (int slot = 0; slot < ALLEY_SIZE; ++slot) {
    if (find_card_in_zone_index(world, gs->zones[player_num].alley, slot) != 0) {
      alley_indices[alley_count++] = slot;
    }
  }
  if (alley_count == 0) {
    return;
  }

  int garden_slots[AZK_MAX_PLACEMENT_SLOTS] = {0};
  int garden_slot_count = collect_placement_slots(world, gs->zones[player_num].garden,
                                                  GARDEN_SIZE, garden_slots);
  if (garden_slot_count == 0) {
    return;
  }

  UserAction action = {.player = player,
                       .type = ACT_GATE_PORTAL,
                       .subaction_1 = 0,
                       .subaction_2 = 0,
                       .subaction_3 = 0};
  for (int a = 0; a < alley_count; ++a) {
    action.subaction_1 = alley_indices[a];
    for (int g = 0; g < garden_slot_count; ++g) {
      action.subaction_2 = garden_slots[g];
      try_add_action_if_valid(world, gs, player, &action, out_mask);
    }
  }
}

static void enumerate_activate_garden_or_leader_ability_actions(
    ecs_world_t *world, const GameState *gs, ecs_entity_t player,
    AzkActionMaskSet *out_mask) {
  const uint8_t player_num = get_player_number(world, player);
  UserAction action = {.player = player,
                       .type = ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
                       .subaction_1 = 0,
                       .subaction_2 = 0,
                       .subaction_3 = 0};

  for (int slot = 0; slot < GARDEN_SIZE; ++slot) {
    if (find_card_in_zone_index(world, gs->zones[player_num].garden, slot) == 0) {
      continue;
    }
    action.subaction_1 = slot;
    try_add_action_if_valid(world, gs, player, &action, out_mask);
  }

  if (find_leader_card_in_zone(world, gs->zones[player_num].leader) != 0) {
    action.subaction_1 = GARDEN_SIZE;
    try_add_action_if_valid(world, gs, player, &action, out_mask);
  }
}

static void enumerate_activate_alley_ability_actions(ecs_world_t *world,
                                                     const GameState *gs,
                                                     ecs_entity_t player,
                                                     AzkActionMaskSet *out_mask) {
  const uint8_t player_num = get_player_number(world, player);
  UserAction action = {.player = player,
                       .type = ACT_ACTIVATE_ALLEY_ABILITY,
                       .subaction_1 = 0,
                       .subaction_2 = 0,
                       .subaction_3 = 0};

  for (int slot = 0; slot < ALLEY_SIZE; ++slot) {
    if (find_card_in_zone_index(world, gs->zones[player_num].alley, slot) == 0) {
      continue;
    }
    action.subaction_2 = slot;
    try_add_action_if_valid(world, gs, player, &action, out_mask);
  }
}

static bool enumerate_spec_fast_path(ecs_world_t *world, const GameState *gs,
                                     ecs_entity_t player,
                                     const ActionEnumerationContext *ctx,
                                     const AzkActionSpec *spec,
                                     AzkActionMaskSet *out_mask) {
  switch (spec->type) {
  case ACT_PLAY_ENTITY_TO_GARDEN:
    enumerate_play_entity_actions(world, gs, player, ctx,
                                  ACT_PLAY_ENTITY_TO_GARDEN, ZONE_GARDEN,
                                  out_mask);
    return true;
  case ACT_PLAY_ENTITY_TO_ALLEY:
    enumerate_play_entity_actions(world, gs, player, ctx,
                                  ACT_PLAY_ENTITY_TO_ALLEY, ZONE_ALLEY,
                                  out_mask);
    return true;
  case ACT_ATTACH_WEAPON_FROM_HAND:
    enumerate_attach_weapon_actions(world, gs, player, ctx, out_mask);
    return true;
  case ACT_PLAY_SPELL_FROM_HAND:
    enumerate_play_spell_actions(world, gs, player, ctx, out_mask);
    return true;
  case ACT_GATE_PORTAL:
    enumerate_gate_portal_actions(world, gs, player, out_mask);
    return true;
  case ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY:
    enumerate_activate_garden_or_leader_ability_actions(world, gs, player,
                                                        out_mask);
    return true;
  case ACT_ACTIVATE_ALLEY_ABILITY:
    enumerate_activate_alley_ability_actions(world, gs, player, out_mask);
    return true;
  default:
    return false;
  }
}

static void enumerate_parameters(ecs_world_t *world, const GameState *gs,
                                 ecs_entity_t player, const AzkActionSpec *spec,
                                 const ActionEnumerationContext *ctx,
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
    enumerate_parameters(world, gs, player, spec, ctx, param_index + 1, action,
                         out_mask);
    return;
  }

  int min_value = param->min_value;
  int max_value = param->max_value;
  if (!resolve_param_bounds(spec, param_index, param, ctx, &min_value,
                            &max_value)) {
    return;
  }

  for (int value = min_value; value <= max_value; ++value) {
    *target_subaction = value;
    enumerate_parameters(world, gs, player, spec, ctx, param_index + 1, action,
                         out_mask);
  }
}

bool azk_build_action_mask_for_player(ecs_world_t *world, const GameState *gs,
                                      int8_t player_index,
                                      AzkActionMaskSet *out_mask) {
  init_mask_profile_if_needed();
  const uint64_t start_ns =
      k_mask_profile.enabled ? monotonic_now_ns() : 0;
  bool was_early_return = false;
  bool was_ability_phase = false;

  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(out_mask != NULL, ECS_INVALID_PARAMETER,
             "Output mask pointer is null");

  // Only fields read by downstream consumers must be reset here.
  // `legal_actions` entries above `legal_action_count` are intentionally left
  // untouched to avoid a large per-call memset cost.
  memset(out_mask->head0_mask, 0, sizeof(out_mask->head0_mask));
  out_mask->legal_action_count = 0;

  if (player_index < 0 || player_index >= MAX_PLAYERS_PER_MATCH) {
    AZK_DEBUG_WARN("[ActionMask] player_index %d out of bounds", player_index);
    was_early_return = true;
    if (k_mask_profile.enabled) {
      const uint64_t elapsed_ns = monotonic_now_ns() - start_ns;
      k_mask_profile.mask_calls++;
      k_mask_profile.mask_total_ns += elapsed_ns;
      k_mask_profile.early_returns++;
      maybe_report_mask_profile();
    }
    return false;
  }

  bool phase_requires_action = phase_requires_user_action(world, gs->phase);
  AZK_DEBUG_INFO("[ActionMask] phase=%d, winner=%d, phase_requires_action=%d, active_player=%d, player_index=%d",
                  gs->phase, gs->winner, phase_requires_action, gs->active_player_index, player_index);

  if (gs->winner != -1 || !phase_requires_action) {
    AZK_DEBUG_INFO("[ActionMask] Early return: game over or phase doesn't require action");
    was_early_return = true;
    if (k_mask_profile.enabled) {
      const uint64_t elapsed_ns = monotonic_now_ns() - start_ns;
      k_mask_profile.mask_calls++;
      k_mask_profile.mask_total_ns += elapsed_ns;
      k_mask_profile.early_returns++;
      maybe_report_mask_profile();
    }
    return true;
  }

  if (player_index != gs->active_player_index) {
    AZK_DEBUG_INFO("[ActionMask] Early return: player %d is not active player %d", player_index, gs->active_player_index);
    was_early_return = true;
    if (k_mask_profile.enabled) {
      const uint64_t elapsed_ns = monotonic_now_ns() - start_ns;
      k_mask_profile.mask_calls++;
      k_mask_profile.mask_total_ns += elapsed_ns;
      k_mask_profile.early_returns++;
      maybe_report_mask_profile();
    }
    return true;
  }

  ecs_entity_t player = gs->players[player_index];
  // Check if we're in an ability sub-phase
  if (azk_is_in_ability_phase(world)) {
    AZK_DEBUG_INFO("[ActionMask] In ability phase, enumerating ability actions");
    was_ability_phase = true;
    enumerate_ability_actions(world, gs, player, out_mask);
    if (k_mask_profile.enabled) {
      const uint64_t elapsed_ns = monotonic_now_ns() - start_ns;
      k_mask_profile.mask_calls++;
      k_mask_profile.mask_total_ns += elapsed_ns;
      k_mask_profile.ability_calls++;
      k_mask_profile.ability_total_ns += elapsed_ns;
      maybe_report_mask_profile();
    }
    return true;
  }

  const uint8_t player_number = get_player_number(world, player);
  ActionEnumerationContext enumeration_ctx = {
      .hand_card_count = 0,
      .has_ikz_token = player_has_ready_ikz_token(world, player)};
  {
    ecs_entity_t hand_zone = gs->zones[player_number].hand;
    ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand_zone);
    enumeration_ctx.hand_card_count = hand_cards.count;
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

    if (enumerate_spec_fast_path(world, gs, player, &enumeration_ctx, spec,
                                 out_mask)) {
      continue;
    }

    enumerate_parameters(world, gs, player, spec, &enumeration_ctx, 0, &action,
                         out_mask);
  }

  AZK_DEBUG_INFO("[ActionMask] Final mask: legal_action_count=%d, head0_mask[0]=%d, head0_mask[23]=%d",
                  out_mask->legal_action_count, out_mask->head0_mask[0], out_mask->head0_mask[23]);

  if (k_mask_profile.enabled) {
    const uint64_t elapsed_ns = monotonic_now_ns() - start_ns;
    k_mask_profile.mask_calls++;
    k_mask_profile.mask_total_ns += elapsed_ns;
    if (was_early_return) {
      k_mask_profile.early_returns++;
    } else if (was_ability_phase) {
      k_mask_profile.ability_calls++;
      k_mask_profile.ability_total_ns += elapsed_ns;
    } else {
      k_mask_profile.normal_calls++;
      k_mask_profile.normal_total_ns += elapsed_ns;
    }
    maybe_report_mask_profile();
  }

  return true;
}
