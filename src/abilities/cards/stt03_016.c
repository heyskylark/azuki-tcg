#include "abilities/cards/stt03_016.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

bool stt03_016_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t enemy_num = (get_player_number(world, owner) + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t cards = ecs_get_ordered_children(world, gs->zones[enemy_num].garden);
  for (int32_t i = 0; i < cards.count; ++i) {
    const CurStats *cur = ecs_get(world, cards.ids[i], CurStats);
    if (cur != NULL && cur->cur_hp <= 2) {
      return true;
    }
  }

  return false;
}

void stt03_016_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t enemy_num =
      (get_player_number(world, ctx->runtime.owner) + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t cards = ecs_get_ordered_children(world, gs->zones[enemy_num].garden);
  ecs_entity_t to_destroy[GARDEN_SIZE] = {0};
  uint8_t destroy_count = 0;

  for (int32_t i = 0; i < cards.count && destroy_count < GARDEN_SIZE; ++i) {
    const CurStats *cur = ecs_get(world, cards.ids[i], CurStats);
    if (cur != NULL && cur->cur_hp <= 2) {
      to_destroy[destroy_count++] = cards.ids[i];
    }
  }

  for (uint8_t i = 0; i < destroy_count; ++i) {
    discard_card(world, to_destroy[i]);
  }
}
