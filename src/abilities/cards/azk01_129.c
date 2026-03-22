#include "abilities/cards/azk01_129.h"

#include "components/abilities.h"
#include "components/components.h"
#include "utils/damage_util.h"
#include "utils/zone_util.h"

static ecs_entity_t leader_at_index(ecs_world_t *world, uint8_t player_num) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  return find_leader_card_in_zone(world, gs->zones[player_num].leader);
}

bool azk01_129_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)owner;

  const DamageTracker *tracker = ecs_get(world, card, DamageTracker);
  return tracker != NULL &&
         (tracker->took_damage_this_turn || tracker->dealt_damage_this_turn);
}

void azk01_129_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  for (uint8_t i = 0; i < MAX_PLAYERS_PER_MATCH; ++i) {
    ecs_entity_t leader = leader_at_index(world, i);
    if (leader != 0) {
      deal_effect_damage(world, leader, 1);
    }
  }
}
