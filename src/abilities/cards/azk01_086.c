#include "abilities/cards/azk01_086.h"

#include "abilities/selection/ability_selection_helpers.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"
#include "utils/status_util.h"

static bool is_valid_forging_tricks_target(ecs_world_t *world,
                                           ecs_entity_t card) {
  return is_weapon_card(world, card);
}

static uint8_t move_matching_discard_weapons_to_selection(ecs_world_t *world,
                                                          AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t discard_zone = gs->zones[player_num].discard;
  const ecs_entity_t selection_zone = gs->zones[player_num].selection;
  ecs_entities_t discard_cards = ecs_get_ordered_children(world, discard_zone);

  ecs_entity_t selection_cards[MAX_SELECTION_ZONE_SIZE] = {0};
  uint8_t selection_count = 0;
  for (int32_t i = 0;
       i < discard_cards.count && selection_count < MAX_SELECTION_ZONE_SIZE;
       ++i) {
    ecs_entity_t target = discard_cards.ids[i];
    if (!is_valid_forging_tricks_target(world, target)) {
      continue;
    }

    int8_t from_index = azk_get_card_index_in_zone(world, target, discard_zone);
    ecs_add_pair(world, target, EcsChildOf, selection_zone);
    azk_log_card_zone_moved(world, target, GLOG_ZONE_DISCARD, from_index,
                            GLOG_ZONE_SELECTION, (int8_t)selection_count);
    selection_cards[selection_count++] = target;
  }

  uint8_t pick_max = selection_count;
  if (pick_max > 5) {
    pick_max = 5;
  }
  azk_init_selection_state(ctx, selection_cards, selection_count, pick_max);
  ctx->runtime.phase =
      selection_count > 0 ? ABILITY_PHASE_SELECTION_PICK : ABILITY_PHASE_NONE;
  return selection_count;
}

static void return_unpicked_selection_cards_to_discard(ecs_world_t *world,
                                                       AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t discard_zone = gs->zones[player_num].discard;
  const ecs_entity_t selection_zone = gs->zones[player_num].selection;

  for (uint8_t i = 0;
       i < ctx->selection.count && i < MAX_SELECTION_ZONE_SIZE; ++i) {
    ecs_entity_t card = ctx->selection.cards[i];
    if (card == 0 ||
        ecs_get_target(world, card, EcsChildOf, 0) != selection_zone) {
      continue;
    }

    int8_t from_index = azk_get_card_index_in_zone(world, card, selection_zone);
    ecs_add_pair(world, card, EcsChildOf, discard_zone);
    azk_log_card_zone_moved(world, card, GLOG_ZONE_SELECTION, from_index,
                            GLOG_ZONE_DISCARD, -1);
  }
}

bool azk01_086_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  ecs_entities_t discard_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].discard);
  for (int32_t i = 0; i < discard_cards.count; ++i) {
    if (is_valid_forging_tricks_target(world, discard_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

void azk01_086_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  const uint8_t selection_count =
      move_matching_discard_weapons_to_selection(world, ctx);
  if (selection_count == 0) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
    cli_render_logf("[AZK01-086] No Weapon cards in discard");
    return;
  }

  cli_render_logf("[AZK01-086] Found %u Weapon card(s) in discard",
                  (unsigned)selection_count);
}

bool azk01_086_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;
  return target != 0 && is_valid_forging_tricks_target(world, target);
}

void azk01_086_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t picked_cards[MAX_SELECTION_ZONE_SIZE] = {0};
  const uint8_t picked_count =
      ctx->selection.picked_count > 5 ? 5 : ctx->selection.picked_count;

  for (uint8_t i = 0; i < picked_count && i < MAX_ABILITY_SELECTION; ++i) {
    picked_cards[i] = ctx->selection.picked_cards[i];
  }

  return_unpicked_selection_cards_to_discard(world, ctx);
  azk_init_selection_state(ctx, picked_cards, picked_count, 0);

  if (picked_count == 0) {
    ctx->runtime.phase = ABILITY_PHASE_NONE;
    cli_render_logf("[AZK01-086] Chose not to bottom deck any Weapon cards");
    return;
  }

  ecs_entity_t leader = find_leader_card_in_zone(world, gs->zones[player_num].leader);
  if (leader != 0) {
    apply_attack_modifier(world, leader, ctx->runtime.source_card,
                          (int8_t)picked_count, true);
  }

  ctx->runtime.phase = ABILITY_PHASE_BOTTOM_DECK;
  cli_render_logf("[AZK01-086] Selected %u Weapon card(s) to bottom deck",
                  (unsigned)picked_count);
}
