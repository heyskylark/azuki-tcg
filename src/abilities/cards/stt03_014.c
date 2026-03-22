#include "abilities/cards/stt03_014.h"

#include "components/abilities.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool stt03_014_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  return true;
}

void stt03_014_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t enemy_num =
      (get_player_number(world, ctx->runtime.owner) + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t cards = ecs_get_ordered_children(world, gs->zones[enemy_num].garden);
  for (int32_t i = 0; i < cards.count; ++i) {
    const BaseStats *base = ecs_get(world, cards.ids[i], BaseStats);
    if (base != NULL && base->attack <= 3) {
      apply_timed_tag_grant(world, cards.ids[i], ecs_id(Rooted),
                            TAG_GRANT_TICK_START_OF_TURN, 2);
    }
  }
}
