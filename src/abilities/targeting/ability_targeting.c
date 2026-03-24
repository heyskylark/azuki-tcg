#include "abilities/targeting/ability_targeting.h"

#include <stdbool.h>

#include "abilities/targeting/ability_target_encoding.h"
#include "components/abilities.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

typedef bool (*AbilityTargetValidatorFn)(ecs_world_t *, ecs_entity_t,
                                         ecs_entity_t, ecs_entity_t);

static AbilityTargetType get_target_type(const AbilityDef *def,
                                         AbilityTargetScope scope) {
  if (!def) {
    return ABILITY_TARGET_NONE;
  }

  return scope == ABILITY_TARGET_SCOPE_COST ? def->cost_req.type
                                            : def->effect_req.type;
}

static AbilityTargetValidatorFn get_target_validator(const AbilityDef *def,
                                                     AbilityTargetScope scope) {
  if (!def) {
    return NULL;
  }

  return scope == ABILITY_TARGET_SCOPE_COST ? def->validate_cost_target
                                            : def->validate_effect_target;
}

static int append_choice(AbilityTargetChoice *out, int out_cap, int count,
                         int action_index, ecs_entity_t entity) {
  if (out && count < out_cap) {
    out[count] = (AbilityTargetChoice){
        .action_index = action_index,
        .entity = entity,
    };
  }
  return count + 1;
}

static bool is_target_valid(ecs_world_t *world, ecs_entity_t source_card,
                            ecs_entity_t owner, ecs_entity_t target,
                            AbilityTargetValidatorFn validator) {
  if (target == 0) {
    return false;
  }

  if (!validator) {
    return true;
  }

  return validator(world, source_card, owner, target);
}

static ecs_entity_t find_leader_card_if_present(ecs_world_t *world,
                                                ecs_entity_t zone) {
  if (zone == 0) {
    return 0;
  }

  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  if (cards.count != 1) {
    return 0;
  }

  ecs_entity_t leader_card = cards.ids[0];
  if (!is_card_type(world, leader_card, CARD_TYPE_LEADER)) {
    return 0;
  }

  return leader_card;
}

static int collect_hand_targets(ecs_world_t *world, ecs_entity_t zone,
                                ecs_entity_t source_card, ecs_entity_t owner,
                                AbilityTargetValidatorFn validator,
                                AbilityTargetChoice *out, int out_cap,
                                int count) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t target = cards.ids[i];
    if (!is_target_valid(world, source_card, owner, target, validator)) {
      continue;
    }
    count = append_choice(out, out_cap, count, i, target);
  }
  return count;
}

static int collect_zone_index_targets(ecs_world_t *world, ecs_entity_t zone,
                                      ecs_entity_t source_card,
                                      ecs_entity_t owner,
                                      AbilityTargetValidatorFn validator,
                                      AbilityTargetChoice *out, int out_cap,
                                      int count, int index_offset) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t target = cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, target, ZoneIndex);
    if (!zone_index) {
      continue;
    }
    if (!is_target_valid(world, source_card, owner, target, validator)) {
      continue;
    }
    count = append_choice(out, out_cap, count, zone_index->index + index_offset,
                          target);
  }
  return count;
}

static int collect_pending_gate_portal_target(
    ecs_world_t *world, const GameState *gs, uint8_t player_num,
    ecs_entity_t source_card, ecs_entity_t owner,
    AbilityTargetValidatorFn validator, AbilityTargetChoice *out, int out_cap,
    int count) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  if (ctx == NULL || ctx->scratch.kind != ABILITY_SCRATCH_GATE_PORTAL) {
    return count;
  }

  ecs_entity_t portaled_card = ctx->scratch.data.gate_portal.portaled_card;
  if (portaled_card == 0) {
    return count;
  }

  if (ecs_get_target(world, portaled_card, EcsChildOf, 0) ==
      gs->zones[player_num].garden) {
    return count;
  }

  if (!is_target_valid(world, source_card, owner, portaled_card, validator)) {
    return count;
  }

  return append_choice(out, out_cap, count,
                       ctx->scratch.data.gate_portal.garden_index,
                       portaled_card);
}

static int collect_enemy_leader_or_garden_targets(
    ecs_world_t *world, const GameState *gs, uint8_t player_num,
    ecs_entity_t source_card, ecs_entity_t owner,
    AbilityTargetValidatorFn validator, AbilityTargetChoice *out, int out_cap,
    int count) {
  const uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
  count = collect_zone_index_targets(world, gs->zones[enemy_num].garden,
                                     source_card, owner, validator, out,
                                     out_cap, count, 0);

  ecs_entity_t leader =
      find_leader_card_if_present(world, gs->zones[enemy_num].leader);
  if (is_target_valid(world, source_card, owner, leader, validator)) {
    count = append_choice(
        out, out_cap, count,
        azk_encode_enemy_leader_or_garden_target_index(true, -1), leader);
  }

  return count;
}

static int collect_any_garden_targets(ecs_world_t *world, const GameState *gs,
                                      uint8_t player_num,
                                      ecs_entity_t source_card,
                                      ecs_entity_t owner,
                                      AbilityTargetValidatorFn validator,
                                      AbilityTargetChoice *out, int out_cap,
                                      int count) {
  const uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t self_cards =
      ecs_get_ordered_children(world, gs->zones[player_num].garden);
  for (int32_t i = 0; i < self_cards.count; i++) {
    ecs_entity_t target = self_cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, target, ZoneIndex);
    if (!zone_index) {
      continue;
    }
    if (!is_target_valid(world, source_card, owner, target, validator)) {
      continue;
    }
    count = append_choice(
        out, out_cap, count,
        azk_encode_any_garden_target_index(false, zone_index->index), target);
  }

  ecs_entities_t enemy_cards =
      ecs_get_ordered_children(world, gs->zones[enemy_num].garden);
  for (int32_t i = 0; i < enemy_cards.count; i++) {
    ecs_entity_t target = enemy_cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, target, ZoneIndex);
    if (!zone_index) {
      continue;
    }
    if (!is_target_valid(world, source_card, owner, target, validator)) {
      continue;
    }
    count = append_choice(
        out, out_cap, count,
        azk_encode_any_garden_target_index(true, zone_index->index), target);
  }

  return count;
}

static int collect_friendly_garden_or_alley_targets(
    ecs_world_t *world, const GameState *gs, uint8_t player_num,
    ecs_entity_t source_card, ecs_entity_t owner,
    AbilityTargetValidatorFn validator, AbilityTargetChoice *out, int out_cap,
    int count) {
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[player_num].garden);
  for (int32_t i = 0; i < garden_cards.count; i++) {
    ecs_entity_t target = garden_cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, target, ZoneIndex);
    if (!zone_index) {
      continue;
    }
    if (!is_target_valid(world, source_card, owner, target, validator)) {
      continue;
    }
    count = append_choice(
        out, out_cap, count,
        azk_encode_friendly_garden_or_alley_target_index(false,
                                                         zone_index->index),
        target);
  }

  ecs_entities_t alley_cards =
      ecs_get_ordered_children(world, gs->zones[player_num].alley);
  for (int32_t i = 0; i < alley_cards.count; i++) {
    ecs_entity_t target = alley_cards.ids[i];
    const ZoneIndex *zone_index = ecs_get(world, target, ZoneIndex);
    if (!zone_index) {
      continue;
    }
    if (!is_target_valid(world, source_card, owner, target, validator)) {
      continue;
    }
    count = append_choice(
        out, out_cap, count,
        azk_encode_friendly_garden_or_alley_target_index(true,
                                                         zone_index->index),
        target);
  }

  return count;
}

static int collect_any_leader_targets(ecs_world_t *world, const GameState *gs,
                                      uint8_t player_num,
                                      ecs_entity_t source_card,
                                      ecs_entity_t owner,
                                      AbilityTargetValidatorFn validator,
                                      AbilityTargetChoice *out, int out_cap,
                                      int count) {
  ecs_entity_t friendly_leader =
      find_leader_card_if_present(world, gs->zones[player_num].leader);
  if (is_target_valid(world, source_card, owner, friendly_leader, validator)) {
    count = append_choice(out, out_cap, count,
                          azk_encode_any_leader_target_index(false),
                          friendly_leader);
  }

  const uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entity_t enemy_leader =
      find_leader_card_if_present(world, gs->zones[enemy_num].leader);
  if (is_target_valid(world, source_card, owner, enemy_leader, validator)) {
    count = append_choice(out, out_cap, count,
                          azk_encode_any_leader_target_index(true),
                          enemy_leader);
  }

  return count;
}

static int collect_any_leader_or_garden_targets(
    ecs_world_t *world, const GameState *gs, uint8_t player_num,
    ecs_entity_t source_card, ecs_entity_t owner,
    AbilityTargetValidatorFn validator, AbilityTargetChoice *out, int out_cap,
    int count) {
  count = collect_any_garden_targets(world, gs, player_num, source_card, owner,
                                     validator, out, out_cap, count);

  ecs_entity_t friendly_leader =
      find_leader_card_if_present(world, gs->zones[player_num].leader);
  if (is_target_valid(world, source_card, owner, friendly_leader, validator)) {
    count = append_choice(
        out, out_cap, count,
        azk_encode_any_leader_or_garden_target_index(true, false, -1),
        friendly_leader);
  }

  const uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entity_t enemy_leader =
      find_leader_card_if_present(world, gs->zones[enemy_num].leader);
  if (is_target_valid(world, source_card, owner, enemy_leader, validator)) {
    count = append_choice(
        out, out_cap, count,
        azk_encode_any_leader_or_garden_target_index(true, true, -1),
        enemy_leader);
  }

  return count;
}

static int collect_target_choices_internal(
    ecs_world_t *world, const AbilityDef *def, AbilityTargetScope scope,
    ecs_entity_t source_card, ecs_entity_t owner,
    AbilityTargetValidatorFn validator, AbilityTargetChoice *out, int out_cap) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!world || !def || !gs || owner == 0) {
    return 0;
  }

  const uint8_t player_num = get_player_number(world, owner);
  const AbilityTargetType type = get_target_type(def, scope);
  int count = 0;

  switch (type) {
  case ABILITY_TARGET_NONE:
    return 0;
  case ABILITY_TARGET_FRIENDLY_HAND:
  case ABILITY_TARGET_FRIENDLY_HAND_WEAPON:
    return collect_hand_targets(world, gs->zones[player_num].hand, source_card,
                                owner, validator, out, out_cap, count);
  case ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY:
    count = collect_zone_index_targets(world, gs->zones[player_num].garden,
                                       source_card, owner, validator, out,
                                       out_cap, count, 0);
    return collect_pending_gate_portal_target(world, gs, player_num,
                                              source_card, owner, validator,
                                              out, out_cap, count);
  case ABILITY_TARGET_ENEMY_GARDEN_ENTITY: {
    const uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
    return collect_zone_index_targets(world, gs->zones[enemy_num].garden,
                                      source_card, owner, validator, out,
                                      out_cap, count, 0);
  }
  case ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY:
    return collect_enemy_leader_or_garden_targets(
        world, gs, player_num, source_card, owner, validator, out, out_cap,
        count);
  case ABILITY_TARGET_FRIENDLY_GARDEN_OR_ALLEY_ENTITY:
    return collect_friendly_garden_or_alley_targets(
        world, gs, player_num, source_card, owner, validator, out, out_cap,
        count);
  case ABILITY_TARGET_ANY_GARDEN_ENTITY:
    return collect_any_garden_targets(world, gs, player_num, source_card,
                                      owner, validator, out, out_cap, count);
  case ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY:
    return collect_any_leader_or_garden_targets(
        world, gs, player_num, source_card, owner, validator, out, out_cap,
        count);
  case ABILITY_TARGET_ANY_LEADER:
    return collect_any_leader_targets(world, gs, player_num, source_card,
                                      owner, validator, out, out_cap, count);
  default:
    return 0;
  }
}

int azk_collect_ability_target_choices(ecs_world_t *world,
                                       const AbilityDef *def,
                                       AbilityTargetScope scope,
                                       ecs_entity_t source_card,
                                       ecs_entity_t owner,
                                       AbilityTargetChoice *out, int out_cap) {
  return collect_target_choices_internal(world, def, scope, source_card, owner,
                                         get_target_validator(def, scope), out,
                                         out_cap);
}

uint8_t azk_count_ability_target_choices(ecs_world_t *world,
                                         const AbilityDef *def,
                                         AbilityTargetScope scope,
                                         ecs_entity_t source_card,
                                         ecs_entity_t owner) {
  const int count = azk_collect_ability_target_choices(
      world, def, scope, source_card, owner, NULL, 0);
  return count > UINT8_MAX ? UINT8_MAX : (uint8_t)count;
}

ecs_entity_t azk_resolve_ability_target_choice_entity(ecs_world_t *world,
                                                      const AbilityDef *def,
                                                      AbilityTargetScope scope,
                                                      ecs_entity_t owner,
                                                      int action_index) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!world || !def || !gs || owner == 0) {
    return 0;
  }

  const uint8_t player_num = get_player_number(world, owner);
  const AbilityTargetType type = get_target_type(def, scope);

  switch (type) {
  case ABILITY_TARGET_FRIENDLY_HAND:
  case ABILITY_TARGET_FRIENDLY_HAND_WEAPON: {
    ecs_entities_t hand_cards =
        ecs_get_ordered_children(world, gs->zones[player_num].hand);
    if (action_index < 0 || action_index >= hand_cards.count) {
      return 0;
    }
    return hand_cards.ids[action_index];
  }
  case ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY:
    return find_card_in_zone_index(world, gs->zones[player_num].garden,
                                   action_index);
  case ABILITY_TARGET_ENEMY_GARDEN_ENTITY: {
    const uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
    return find_card_in_zone_index(world, gs->zones[enemy_num].garden,
                                   action_index);
  }
  case ABILITY_TARGET_FRIENDLY_GARDEN_OR_ALLEY_ENTITY: {
    bool is_alley = false;
    int zone_index = -1;
    if (!azk_decode_friendly_garden_or_alley_target_index(action_index,
                                                          &is_alley,
                                                          &zone_index)) {
      return 0;
    }

    return find_card_in_zone_index(
        world, is_alley ? gs->zones[player_num].alley
                        : gs->zones[player_num].garden,
        zone_index);
  }
  case ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY: {
    bool is_leader = false;
    int zone_index = -1;
    if (!azk_decode_enemy_leader_or_garden_target_index(action_index,
                                                        &is_leader,
                                                        &zone_index)) {
      return 0;
    }
    const uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
    if (is_leader) {
      return find_leader_card_if_present(world, gs->zones[enemy_num].leader);
    }
    return find_card_in_zone_index(world, gs->zones[enemy_num].garden,
                                   zone_index);
  }
  case ABILITY_TARGET_ANY_GARDEN_ENTITY: {
    bool is_enemy = false;
    int zone_index = -1;
    if (!azk_decode_any_garden_target_index(action_index, &is_enemy,
                                            &zone_index)) {
      return 0;
    }
    const uint8_t target_player_num =
        is_enemy ? (player_num + 1) % MAX_PLAYERS_PER_MATCH : player_num;
    return find_card_in_zone_index(world, gs->zones[target_player_num].garden,
                                   zone_index);
  }
  case ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY: {
    bool is_leader = false;
    bool is_enemy = false;
    int zone_index = -1;
    if (!azk_decode_any_leader_or_garden_target_index(action_index,
                                                      &is_leader, &is_enemy,
                                                      &zone_index)) {
      return 0;
    }

    const uint8_t target_player_num =
        is_enemy ? (player_num + 1) % MAX_PLAYERS_PER_MATCH : player_num;
    if (is_leader) {
      return find_leader_card_if_present(world,
                                         gs->zones[target_player_num].leader);
    }

    return find_card_in_zone_index(world, gs->zones[target_player_num].garden,
                                   zone_index);
  }
  case ABILITY_TARGET_ANY_LEADER: {
    bool is_enemy = false;
    if (!azk_decode_any_leader_target_index(action_index, &is_enemy)) {
      return 0;
    }
    const uint8_t target_player_num =
        is_enemy ? (player_num + 1) % MAX_PLAYERS_PER_MATCH : player_num;
    return find_leader_card_if_present(world, gs->zones[target_player_num].leader);
  }
  default:
    return 0;
  }
}
