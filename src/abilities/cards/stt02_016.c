#include "abilities/cards/stt02_016.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

// STT02-016: [Response] Discard 1: Reduce a leader's or entity's attack by 2 until the end of the turn.

bool stt02_016_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  // Check player has at least 1 card in hand to discard
  ecs_entity_t hand = gs->zones[owner_num].hand;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
  if (hand_cards.count < 1) {
    return false;
  }

  // Check opponent has at least one valid target (leader or garden entity)
  // Garden entities
  ecs_entity_t garden = gs->zones[opponent_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
  if (garden_cards.count > 0) {
    return true; // At least one garden entity exists
  }

  // Leader always exists
  ecs_entity_t leader = find_leader_card_in_zone(world, gs->zones[opponent_num].leader);
  if (leader != 0) {
    return true;
  }

  return false;
}

bool stt02_016_validate_cost_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  // Target must be in the owner's hand
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t hand = gs->zones[owner_num].hand;

  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);
  return parent == hand;
}

bool stt02_016_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  ecs_entity_t parent = ecs_get_target(world, target, EcsChildOf, 0);

  // Check if target is in opponent's garden
  if (parent == gs->zones[opponent_num].garden) {
    return true;
  }

  // Check if target is opponent's leader
  if (parent == gs->zones[opponent_num].leader) {
    return true;
  }

  return false;
}

void stt02_016_apply_costs(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t to_discard = ctx->cost_targets[0];
  discard_card(world, to_discard);
  cli_render_logf("[STT02-016] Discarded a card as cost");
}

void stt02_016_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect_targets[0];

  if (target == 0) {
    cli_render_logf("[STT02-016] No target for attack debuff");
    return;
  }

  // Apply -2 attack modifier that expires at end of turn
  // Source is the spell card applying the debuff
  apply_attack_modifier(world, target, ctx->source_card, -2, true);
  cli_render_logf("[STT02-016] Reduced target's attack by 2 until end of turn");
}
