#include "abilities/cards/azk01_083.h"

#include "components/abilities.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/player_util.h"

static bool is_copy_source(ecs_world_t *world, ecs_entity_t card,
                           ecs_entity_t owner, ecs_entity_t target) {
  if (target == 0 || target == card || !is_normal_element_card(world, target)) {
    return false;
  }

  const CardId *card_id = ecs_get(world, target, CardId);
  if (card_id == NULL || card_id->id == CARD_DEF_AZK01_083) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_target(world, target, EcsChildOf, 0) ==
         gs->zones[owner_num].garden;
}

bool azk01_083_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  // [Main] ability: only usable while the Imitator itself is in the Garden.
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].garden) {
    return false;
  }
  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[owner_num].garden);
  for (int32_t i = 0; i < cards.count; ++i) {
    if (is_copy_source(world, card, owner, cards.ids[i])) {
      return true;
    }
  }
  return false;
}

bool azk01_083_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  return is_copy_source(world, card, owner, target);
}

void azk01_083_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0 || ctx->effect.entities[0] == 0) {
    return;
  }

  const CardId *source_id = ecs_get(world, ctx->effect.entities[0], CardId);
  if (source_id == NULL) {
    return;
  }

  (void)azk_grant_copied_card_text(world, ctx->runtime.source_card,
                                   source_id->id);
}
