#include "abilities/cards/stt01_002.h"

#include "components/components.h"
#include "constants/game.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

// STT01-002 "Surge": On Gate Portal; you may play from your discard pile
// a weapon card with cost <= gate points of the portaled entity

bool stt01_002_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;

  // Always valid - validation happens during selection
  return true;
}

void stt01_002_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  // Get gate points from the portaled card (stored in ctx->effect_targets[0])
  ecs_entity_t portaled_card = ctx->effect_targets[0];
  const GatePoints *gp = ecs_get(world, portaled_card, GatePoints);

  if (!gp || gp->gate_points == 0) {
    cli_render_logf("[STT01-002] Portaled card has no gate points");
    ctx->phase = ABILITY_PHASE_NONE;
    return;
  }

  uint8_t max_cost = gp->gate_points;
  cli_render_logf("[STT01-002] Searching discard for weapons with cost <= %d",
                  max_cost);

  // Get discard pile
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t discard = gs->zones[player_num].discard;
  ecs_entity_t selection = gs->zones[player_num].selection;

  // Find valid weapons and move to selection zone
  ecs_entities_t discard_cards = ecs_get_ordered_children(world, discard);

  ctx->selection_count = 0;
  for (int32_t i = 0;
       i < discard_cards.count && ctx->selection_count < MAX_SELECTION_ZONE_SIZE;
       i++) {
    ecs_entity_t card = discard_cards.ids[i];

    // Check if weapon
    if (!is_weapon_card(world, card)) {
      continue;
    }

    // Check cost <= gate points
    const IKZCost *cost = ecs_get(world, card, IKZCost);
    if (!cost || cost->ikz_cost > max_cost) {
      continue;
    }

    // Move to selection zone
    int8_t from_index = azk_get_card_index_in_zone(world, card, discard);
    ecs_add_pair(world, card, EcsChildOf, selection);
    azk_log_card_zone_moved(world, card, GLOG_ZONE_DISCARD, from_index,
                            GLOG_ZONE_SELECTION,
                            (int8_t)ctx->selection_count);
    ctx->selection_cards[ctx->selection_count] = card;
    ctx->selection_count++;
  }

  if (ctx->selection_count == 0) {
    cli_render_logf("[STT01-002] No valid weapons in discard pile");
    ctx->phase = ABILITY_PHASE_NONE;
    return;
  }

  // Set up selection pick phase - "up to 1" weapon
  ctx->selection_pick_max = 1;
  ctx->selection_picked = 0;
  ctx->phase = ABILITY_PHASE_SELECTION_PICK;

  // Store max_cost in effect_min for validation during selection
  ctx->effect_min = max_cost;

  cli_render_logf("[STT01-002] Found %d valid weapon(s) to equip",
                  ctx->selection_count);
}

bool stt01_002_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target) {
  (void)card;
  (void)owner;

  // Must be a weapon
  if (!is_weapon_card(world, target)) {
    return false;
  }

  // Check cost against stored max
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  const IKZCost *cost = ecs_get(world, target, IKZCost);
  if (!cost || cost->ikz_cost > ctx->effect_min) {
    return false;
  }

  return true;
}

void stt01_002_on_selection_complete(ecs_world_t *world, AbilityContext *ctx) {
  // Move any unpicked cards back to discard
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t discard = gs->zones[player_num].discard;
  ecs_entity_t selection = gs->zones[player_num].selection;

  for (int i = 0; i < ctx->selection_count; i++) {
    ecs_entity_t card = ctx->selection_cards[i];
    if (card != 0) {
      int8_t from_index = azk_get_card_index_in_zone(world, card, selection);
      ecs_add_pair(world, card, EcsChildOf, discard);
      azk_log_card_zone_moved(world, card, GLOG_ZONE_SELECTION, from_index,
                              GLOG_ZONE_DISCARD, -1);
      ctx->selection_cards[i] = 0;
    }
  }

  cli_render_logf("[STT01-002] Ability complete");
}
