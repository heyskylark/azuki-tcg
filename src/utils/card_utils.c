#include "utils/card_utils.h"
#include "generated/card_defs.h"
#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include <stdio.h>

bool is_card_type(ecs_world_t *world, ecs_entity_t card, CardType type) {
  const Type *card_type = ecs_get(world, card, Type);
  ecs_assert(card_type != NULL, ECS_INVALID_PARAMETER, "Type component not found for card %d", card);
  return card_type->value == type;
}

void discard_card(ecs_world_t *world, ecs_entity_t card) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);

  const PlayerNumber *player_number = ecs_get(world, owner, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER, "PlayerNumber component not found for player %d", owner);

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t discard_zone = gs->zones[player_number->player_number].discard;

  ecs_remove_id(world, card, ecs_id(ZoneIndex));
  ecs_set(world, card, TapState, { .tapped = false, .cooldown = false });
  ecs_add_pair(world, card, EcsChildOf, discard_zone);
}

void return_card_to_hand(ecs_world_t *world, ecs_entity_t card) {
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", card);

  const PlayerNumber *player_number = ecs_get(world, owner, PlayerNumber);
  ecs_assert(player_number != NULL, ECS_INVALID_PARAMETER, "PlayerNumber component not found for player %d", owner);

  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t hand_zone = gs->zones[player_number->player_number].hand;

  // Remove zone index if present (entity was in garden/alley)
  ecs_remove_id(world, card, ecs_id(ZoneIndex));
  // Reset tap state
  ecs_set(world, card, TapState, { .tapped = false, .cooldown = false });
  // Reset current stats to base stats
  const BaseStats *base = ecs_get(world, card, BaseStats);
  if (base) {
    ecs_set(world, card, CurStats, { .cur_atk = base->attack, .cur_hp = base->health });
  }
  // Move to hand
  ecs_add_pair(world, card, EcsChildOf, hand_zone);

  cli_render_logf("[CardUtils] Returned card to hand");
}

void set_card_to_tapped(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER, "TapState component not found for card %d", card);
  ecs_set(world, card, TapState, { .tapped = true, .cooldown = tap_state->cooldown });
}

void set_card_to_cooldown(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER, "TapState component not found for card %d", card);
  ecs_set(world, card, TapState, { .tapped = tap_state->tapped, .cooldown = true });
}

bool is_card_tapped(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER, "TapState component not found for card %d", card);
  return tap_state->tapped;
}

bool is_card_cooldown(ecs_world_t *world, ecs_entity_t card) {
  const TapState *tap_state = ecs_get(world, card, TapState);
  ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER, "TapState component not found for card %d", card);
  return tap_state->cooldown;
}
