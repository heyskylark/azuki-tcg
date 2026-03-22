#include "abilities/cards/azk01_120.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/weapon_util.h"
#include "utils/zone_util.h"

static uint8_t gate_power_from_ctx(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx == NULL || ctx->scratch.kind != ABILITY_SCRATCH_GATE_PORTAL) {
    return 0;
  }

  const GatePoints *gp =
      ecs_get(world, ctx->scratch.data.gate_portal.portaled_card, GatePoints);
  return gp != NULL ? gp->gate_points : 0;
}

static bool host_is_valid_reequip_target(ecs_world_t *world, ecs_entity_t owner,
                                         ecs_entity_t host,
                                         ecs_entity_t current_host) {
  if (host == 0 || host == current_host) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  const ecs_entity_t parent = ecs_get_target(world, host, EcsChildOf, 0);
  return parent == gs->zones[owner_num].garden ||
         parent == gs->zones[owner_num].leader;
}

static bool weapon_has_other_host_option(ecs_world_t *world, ecs_entity_t owner,
                                         ecs_entity_t current_host) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);

  ecs_entity_t leader = find_leader_card_in_zone(world, gs->zones[owner_num].leader);
  if (host_is_valid_reequip_target(world, owner, leader, current_host)) {
    return true;
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    if (host_is_valid_reequip_target(world, owner, garden_cards.ids[i],
                                     current_host)) {
      return true;
    }
  }

  return false;
}

static bool is_valid_reequip_weapon(ecs_world_t *world, ecs_entity_t owner,
                                    ecs_entity_t weapon, uint8_t max_cost) {
  if (!is_weapon_card(world, weapon)) {
    return false;
  }

  const IKZCost *cost = ecs_get(world, weapon, IKZCost);
  if (cost == NULL || cost->ikz_cost > max_cost) {
    return false;
  }

  const ecs_entity_t current_host = ecs_get_target(world, weapon, EcsChildOf, 0);
  return weapon_has_other_host_option(world, owner, current_host);
}

static void detach_weapon_to_selection(ecs_world_t *world, ecs_entity_t weapon,
                                       ecs_entity_t selection_zone,
                                       int8_t selection_index) {
  const ecs_entity_t host = ecs_get_target(world, weapon, EcsChildOf, 0);
  const CurStats *weapon_stats = ecs_get(world, weapon, CurStats);
  const CurStats *host_stats = host != 0 ? ecs_get(world, host, CurStats) : NULL;

  if (host != 0) {
    remove_weapon_combat_modifier_if_any(world, weapon, host);
  }

  if (host_stats != NULL && weapon_stats != NULL) {
    int16_t new_atk = host_stats->cur_atk - weapon_stats->cur_atk;
    if (new_atk < 0) {
      new_atk = 0;
    }

    ecs_set(world, host, CurStats,
            {.cur_atk = (int8_t)new_atk, .cur_hp = host_stats->cur_hp});
    azk_log_card_stat_change(world, host, (int8_t)(-weapon_stats->cur_atk), 0,
                             (int8_t)new_atk, host_stats->cur_hp);
  }

  azk_log_card_zone_moved(world, weapon, GLOG_ZONE_EQUIPPED, -1,
                          GLOG_ZONE_SELECTION, selection_index);
  ecs_add_pair(world, weapon, EcsChildOf, selection_zone);
  ecs_set(world, weapon, ReequipOrigin, {.previous_host = host});
}

static void return_selection_weapon_to_host(ecs_world_t *world,
                                            ecs_entity_t weapon) {
  const ReequipOrigin *origin = ecs_get(world, weapon, ReequipOrigin);
  if (origin == NULL) {
    return;
  }

  if (!ecs_is_valid(world, origin->previous_host)) {
    discard_card(world, weapon);
    ecs_remove(world, weapon, ReequipOrigin);
    return;
  }

  const ecs_entity_t selection_zone = ecs_get_target(world, weapon, EcsChildOf, 0);
  const int8_t from_index =
      azk_get_card_index_in_zone(world, weapon, selection_zone);
  const CurStats *weapon_stats = ecs_get(world, weapon, CurStats);
  if (weapon_stats == NULL) {
    ecs_remove(world, weapon, ReequipOrigin);
    return;
  }

  azk_log_card_zone_moved(world, weapon, GLOG_ZONE_SELECTION, from_index,
                          GLOG_ZONE_EQUIPPED, -1);
  ecs_add_pair(world, weapon, EcsChildOf, origin->previous_host);
  apply_weapon_attack_bonus(world, origin->previous_host, weapon_stats->cur_atk);
  apply_weapon_combat_modifier_if_any(world, weapon, origin->previous_host);
  ecs_remove(world, weapon, ReequipOrigin);
}

bool azk01_120_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  const uint8_t max_cost = gate_power_from_ctx(world, ctx);
  if (max_cost == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t leader = find_leader_card_in_zone(world, gs->zones[owner_num].leader);
  ecs_iter_t leader_children = ecs_children(world, leader);
  while (ecs_children_next(&leader_children)) {
    for (int i = 0; i < leader_children.count; ++i) {
      if (is_valid_reequip_weapon(world, owner, leader_children.entities[i],
                                  max_cost)) {
        return true;
      }
    }
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    ecs_iter_t weapon_it = ecs_children(world, garden_cards.ids[i]);
    while (ecs_children_next(&weapon_it)) {
      for (int j = 0; j < weapon_it.count; ++j) {
        if (is_valid_reequip_weapon(world, owner, weapon_it.entities[j],
                                    max_cost)) {
          return true;
        }
      }
    }
  }

  return false;
}

void azk01_120_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const uint8_t max_cost = gate_power_from_ctx(world, ctx);
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t selection_zone = gs->zones[owner_num].selection;
  ecs_entity_t selection_cards[MAX_SELECTION_ZONE_SIZE] = {0};
  uint8_t selection_count = 0;

  ecs_entity_t leader = find_leader_card_in_zone(world, gs->zones[owner_num].leader);
  ecs_iter_t leader_children = ecs_children(world, leader);
  while (ecs_children_next(&leader_children) &&
         selection_count < MAX_SELECTION_ZONE_SIZE) {
    for (int i = 0; i < leader_children.count &&
                    selection_count < MAX_SELECTION_ZONE_SIZE;
         ++i) {
      ecs_entity_t weapon = leader_children.entities[i];
      if (!is_valid_reequip_weapon(world, ctx->runtime.owner, weapon, max_cost)) {
        continue;
      }

      detach_weapon_to_selection(world, weapon, selection_zone, selection_count);
      selection_cards[selection_count++] = weapon;
    }
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0;
       i < garden_cards.count && selection_count < MAX_SELECTION_ZONE_SIZE; ++i) {
    ecs_iter_t weapon_it = ecs_children(world, garden_cards.ids[i]);
    while (ecs_children_next(&weapon_it) &&
           selection_count < MAX_SELECTION_ZONE_SIZE) {
      for (int j = 0; j < weapon_it.count &&
                      selection_count < MAX_SELECTION_ZONE_SIZE;
           ++j) {
        ecs_entity_t weapon = weapon_it.entities[j];
        if (!is_valid_reequip_weapon(world, ctx->runtime.owner, weapon,
                                     max_cost)) {
          continue;
        }

        detach_weapon_to_selection(world, weapon, selection_zone, selection_count);
        selection_cards[selection_count++] = weapon;
      }
    }
  }

  azk_init_selection_state(ctx, selection_cards, selection_count, 1);
  ctx->runtime.phase =
      selection_count > 0 ? ABILITY_PHASE_SELECTION_PICK : ABILITY_PHASE_NONE;
}

bool azk01_120_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;

  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return target != 0 &&
         is_valid_reequip_weapon(world, owner, target, gate_power_from_ctx(world, ctx));
}

void azk01_120_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  for (uint8_t i = 0;
       i < ctx->selection.count && i < MAX_SELECTION_ZONE_SIZE; ++i) {
    ecs_entity_t weapon = ctx->selection.cards[i];
    if (weapon == 0) {
      continue;
    }

    const GameState *gs = ecs_singleton_get(world, GameState);
    const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
    if (ecs_get_target(world, weapon, EcsChildOf, 0) !=
        gs->zones[owner_num].selection) {
      continue;
    }

    return_selection_weapon_to_host(world, weapon);
  }
}
