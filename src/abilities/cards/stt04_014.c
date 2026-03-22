#include "abilities/cards/stt04_014.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

bool stt04_014_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  const ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[owner_num].leader);
  return leader != 0 && has_subtype(world, leader, ecs_id(TSubtype_Scorchweaver));
}

void stt04_014_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t targets[MAX_PLAYERS_PER_MATCH * GARDEN_SIZE] = {0};
  uint8_t target_count = 0;

  for (uint8_t p = 0; p < MAX_PLAYERS_PER_MATCH; ++p) {
    ecs_entities_t cards = ecs_get_ordered_children(world, gs->zones[p].garden);
    for (int32_t i = 0; i < cards.count &&
                        target_count < MAX_PLAYERS_PER_MATCH * GARDEN_SIZE;
         ++i) {
      targets[target_count++] = cards.ids[i];
    }
  }

  for (uint8_t i = 0; i < target_count; ++i) {
    if (targets[i] != 0) {
      deal_effect_damage(world, targets[i], 1);
    }
  }

  ecs_entities_t garden_cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    ecs_entity_t target = garden_cards.ids[i];
    if (target != ctx->runtime.source_card && is_card_type(world, target, CARD_TYPE_ENTITY)) {
      apply_attack_modifier(world, target, ctx->runtime.source_card, 1, true);
    }
  }
}
