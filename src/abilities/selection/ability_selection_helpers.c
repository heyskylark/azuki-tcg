#include "abilities/selection/ability_selection_helpers.h"

#include "abilities/core/ability_context.h"
#include "utils/deck_utils.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

static uint8_t clamp_selection_count(uint8_t count) {
  return count > MAX_SELECTION_ZONE_SIZE ? MAX_SELECTION_ZONE_SIZE : count;
}

static uint8_t clamp_pick_max(uint8_t pick_max) {
  const bool within_capacity = pick_max <= MAX_ABILITY_SELECTION;
  ecs_assert(within_capacity, ECS_INVALID_PARAMETER,
             "Selection pick_max %u exceeds MAX_ABILITY_SELECTION %u",
             (unsigned)pick_max, (unsigned)MAX_ABILITY_SELECTION);
  return within_capacity ? pick_max : MAX_ABILITY_SELECTION;
}

void azk_init_selection_state(AbilityContext *ctx, const ecs_entity_t *cards,
                              uint8_t count, uint8_t pick_max) {
  if (!ctx) {
    return;
  }

  const uint8_t actual_count = cards ? clamp_selection_count(count) : 0;
  ctx->selection = (AbilitySelectionState){
      .count = actual_count,
      .pick_max = clamp_pick_max(pick_max),
  };

  for (uint8_t i = 0; i < actual_count; ++i) {
    ctx->selection.cards[i] = cards[i];
  }

  for (uint8_t i = actual_count; i < MAX_SELECTION_ZONE_SIZE; ++i) {
    ctx->selection.cards[i] = 0;
  }

  ctx->selection.picked_count = 0;
  for (uint8_t i = 0; i < MAX_ABILITY_SELECTION; ++i) {
    ctx->selection.picked_cards[i] = 0;
  }
}

uint8_t azk_move_matching_hand_cards_to_selection(
    ecs_world_t *world, AbilityContext *ctx, uint8_t pick_max,
    AbilitySelectionCardPredicate predicate, const void *user_ctx) {
  if (!world || !ctx) {
    return 0;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!gs) {
    return 0;
  }

  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t hand_zone = gs->zones[player_num].hand;
  const ecs_entity_t selection_zone = gs->zones[player_num].selection;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand_zone);

  ecs_entity_t selection_cards[MAX_SELECTION_ZONE_SIZE] = {0};
  int32_t selection_index =
      ecs_get_ordered_children(world, selection_zone).count;
  uint8_t selection_count = 0;

  for (int32_t i = 0;
       i < hand_cards.count && selection_count < MAX_SELECTION_ZONE_SIZE; ++i) {
    ecs_entity_t card = hand_cards.ids[i];
    if (card == 0) {
      continue;
    }

    if (predicate && !predicate(world, card, user_ctx)) {
      continue;
    }

    ecs_add_pair(world, card, EcsChildOf, selection_zone);
    azk_log_card_zone_moved(world, card, GLOG_ZONE_HAND, (int8_t)i,
                            GLOG_ZONE_SELECTION,
                            (int8_t)(selection_index + selection_count));
    selection_cards[selection_count++] = card;
  }

  azk_init_selection_state(ctx, selection_cards, selection_count, pick_max);
  ctx->runtime.phase =
      selection_count > 0 ? ABILITY_PHASE_SELECTION_PICK : ABILITY_PHASE_NONE;
  return selection_count;
}

uint8_t azk_count_selection_cards_matching(
    ecs_world_t *world, const AbilityContext *ctx,
    AbilitySelectionCardPredicate predicate, const void *user_ctx) {
  if (!ctx) {
    return 0;
  }

  uint8_t count = 0;
  for (uint8_t i = 0;
       i < ctx->selection.count && i < MAX_SELECTION_ZONE_SIZE; ++i) {
    const ecs_entity_t card = ctx->selection.cards[i];
    if (card == 0) {
      continue;
    }

    if (!predicate || predicate(world, card, user_ctx)) {
      ++count;
    }
  }

  return count;
}

uint8_t azk_begin_bottom_deck_for_remaining_selection(AbilityContext *ctx) {
  if (!ctx) {
    return 0;
  }

  const uint8_t remaining = azk_count_remaining_selection_cards(ctx);
  if (remaining > 0) {
    ctx->runtime.phase = ABILITY_PHASE_BOTTOM_DECK;
  }

  return remaining;
}

void azk_record_selection_pick(AbilityContext *ctx, int selection_index,
                               ecs_entity_t picked_card) {
  if (!ctx) {
    return;
  }

  if (ctx->selection.picked_count < MAX_ABILITY_SELECTION) {
    ctx->selection.picked_cards[ctx->selection.picked_count] = picked_card;
  }
  ctx->selection.picked_count++;

  if (selection_index >= 0 && selection_index < ctx->selection.count &&
      selection_index < MAX_SELECTION_ZONE_SIZE) {
    ctx->selection.cards[selection_index] = 0;
  }
}

uint8_t azk_move_picked_selection_cards_to_hand(ecs_world_t *world,
                                                const AbilityContext *ctx) {
  if (!ctx) {
    return 0;
  }

  uint8_t moved = 0;
  for (uint8_t i = 0;
       i < ctx->selection.picked_count && i < MAX_ABILITY_SELECTION; ++i) {
    ecs_entity_t picked = ctx->selection.picked_cards[i];
    if (picked == 0) {
      continue;
    }

    move_selection_to_hand(world, picked);
    ++moved;
  }

  return moved;
}

uint8_t azk_move_picked_selection_cards_to_hand_if_still_in_selection(
    ecs_world_t *world, const AbilityContext *ctx) {
  if (!ctx) {
    return 0;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!gs) {
    return 0;
  }

  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t selection_zone = gs->zones[player_num].selection;

  uint8_t moved = 0;
  for (uint8_t i = 0;
       i < ctx->selection.picked_count && i < MAX_ABILITY_SELECTION; ++i) {
    ecs_entity_t picked = ctx->selection.picked_cards[i];
    if (picked == 0) {
      continue;
    }

    if (ecs_get_target(world, picked, EcsChildOf, 0) != selection_zone) {
      continue;
    }

    move_selection_to_hand(world, picked);
    ++moved;
  }

  return moved;
}

uint8_t azk_return_remaining_selection_cards_to_hand(ecs_world_t *world,
                                                     AbilityContext *ctx) {
  if (!ctx) {
    return 0;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!gs) {
    return 0;
  }

  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t selection_zone = gs->zones[player_num].selection;

  uint8_t moved = 0;
  for (uint8_t i = 0;
       i < ctx->selection.count && i < MAX_SELECTION_ZONE_SIZE; ++i) {
    ecs_entity_t card = ctx->selection.cards[i];
    if (card == 0) {
      continue;
    }

    if (ecs_get_target(world, card, EcsChildOf, 0) != selection_zone) {
      ctx->selection.cards[i] = 0;
      continue;
    }

    move_selection_to_hand(world, card);
    ctx->selection.cards[i] = 0;
    ++moved;
  }

  return moved;
}

uint8_t azk_return_remaining_selection_cards_to_discard(ecs_world_t *world,
                                                        AbilityContext *ctx) {
  if (!ctx) {
    return 0;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!gs) {
    return 0;
  }

  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t discard_zone = gs->zones[player_num].discard;
  const ecs_entity_t selection_zone = gs->zones[player_num].selection;

  uint8_t moved = 0;
  for (uint8_t i = 0;
       i < ctx->selection.count && i < MAX_SELECTION_ZONE_SIZE; ++i) {
    ecs_entity_t card = ctx->selection.cards[i];
    if (card == 0) {
      continue;
    }

    const int8_t from_index =
        azk_get_card_index_in_zone(world, card, selection_zone);
    ecs_add_pair(world, card, EcsChildOf, discard_zone);
    azk_log_card_zone_moved(world, card, GLOG_ZONE_SELECTION, from_index,
                            GLOG_ZONE_DISCARD, -1);
    ctx->selection.cards[i] = 0;
    ++moved;
  }

  return moved;
}
