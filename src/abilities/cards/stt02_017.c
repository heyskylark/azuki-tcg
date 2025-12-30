#include "abilities/cards/stt02_017.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

// STT02-017 "Shao's Perseverance": [Main] If your leader's Shao, return all
// entities with cost <= 4 in opponent's garden to their owner's hand

bool stt02_017_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);

  // Get player's leader card
  ecs_entity_t leader_zone = gs->zones[player_num].leader;
  ecs_entity_t leader = find_leader_card_in_zone(world, leader_zone);

  // Check if leader has Shao subtype
  return has_subtype(world, leader, ecs_id(TSubtype_Shao));
}

void stt02_017_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, ctx->owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  // Get opponent's garden zone
  ecs_entity_t garden = gs->zones[opponent_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

  // Collect entities to bounce first to avoid iterator invalidation
  ecs_entity_t to_bounce[5]; // Max garden size is 5
  int bounce_count = 0;

  for (int i = 0; i < garden_cards.count; i++) {
    ecs_entity_t entity = garden_cards.ids[i];

    // Must be an entity card
    if (!is_card_type(world, entity, CARD_TYPE_ENTITY)) {
      continue;
    }

    // Must have IKZ cost <= 4
    const IKZCost *cost = ecs_get(world, entity, IKZCost);
    if (!cost || cost->ikz_cost > 4) {
      continue;
    }

    to_bounce[bounce_count++] = entity;
  }

  // Return all collected entities to their owner's hand
  for (int i = 0; i < bounce_count; i++) {
    return_card_to_hand(world, to_bounce[i]);
  }

  cli_render_logf("[STT02-017] Returned %d entities to opponent's hand",
                  bounce_count);
}
