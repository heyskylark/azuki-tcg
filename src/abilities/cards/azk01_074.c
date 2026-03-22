#include "abilities/cards/azk01_074.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

bool azk01_074_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[opponent_num].garden);
  return garden_cards.count > 0;
}

bool azk01_074_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || !is_card_type(world, target, CARD_TYPE_ENTITY)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[opponent_num].garden;
}

void azk01_074_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t source = ctx->runtime.source_card;
  ecs_entity_t target = ctx->effect.entities[0];
  const CurStats *source_stats = ecs_get(world, source, CurStats);
  const CurStats *target_stats = ecs_get(world, target, CurStats);
  if (source_stats == NULL || target_stats == NULL) {
    return;
  }

  int16_t modifier =
      (int16_t)target_stats->cur_atk - (int16_t)source_stats->cur_atk;
  if (modifier < INT8_MIN) {
    modifier = INT8_MIN;
  } else if (modifier > INT8_MAX) {
    modifier = INT8_MAX;
  }

  apply_attack_modifier(world, source, source, (int8_t)modifier, true);
}
