#include "abilities/cards/stt01_016.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/damage_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

// STT01-016 "Raizan's Zanbato": [When Attacking] If equipped to a (Raizan)
// card, deal 1 damage to all entities in your opponent's garden.

bool stt01_016_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  // Get the entity this weapon is equipped to (parent via EcsChildOf)
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  if (parent == 0) {
    return false;
  }

  // Check if parent has Raizan subtype
  if (!has_subtype(world, parent, ecs_id(TSubtype_Raizan))) {
    return false;
  }

  // Check opponent has at least one non-EffectImmune entity in garden
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  ecs_entity_t garden = gs->zones[opponent_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

  for (int32_t i = 0; i < garden_cards.count; i++) {
    if (!is_effect_immune(world, garden_cards.ids[i])) {
      return true;
    }
  }

  return false;
}

void stt01_016_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, ctx->owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

  ecs_entity_t garden = gs->zones[opponent_num].garden;
  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

  int damage_count = 0;

  for (int32_t i = 0; i < garden_cards.count; i++) {
    ecs_entity_t target = garden_cards.ids[i];

    // deal_effect_damage handles EffectImmune check and death/discard
    if (deal_effect_damage(world, target, 1)) {
      damage_count++;
    }
  }

  cli_render_logf("[STT01-016] Dealt 1 damage to %d entities in opponent's "
                  "garden",
                  damage_count);
}
