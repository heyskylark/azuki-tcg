#include "abilities/cards/stt02_001.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

// STT02-001 (Shao): [Response] [Once/Turn] Pay 1 IKZ: Reduce a leader's or
// entity's attack by 1 until the end of the turn.

bool stt02_001_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t owner_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_num + 1) % MAX_PLAYERS_PER_MATCH;

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

bool stt02_001_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
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

void stt02_001_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t target = ctx->effect_targets[0];

  if (target == 0) {
    cli_render_logf("[STT02-001] No target for attack debuff");
    return;
  }

  // Apply -1 attack modifier that expires at end of turn
  // Source is the card applying the debuff (Shao)
  apply_attack_modifier(world, target, ctx->source_card, -1, true);
  cli_render_logf("[STT02-001] Reduced target's attack by 1 until end of turn");
}
