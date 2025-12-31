#include "utils/combat_util.h"
#include "components/abilities.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"

int attack(
  ecs_world_t *world,
  const AttackIntent *intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(intent != NULL, ECS_INVALID_PARAMETER, "AttackIntent is null");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  tap_card(world, intent->attacking_card);

  CombatState combat_state = {
    .attacking_card = intent->attacking_card,
    .defender_card = intent->defending_card,
    .defender_intercepted = false,
  };
  gs->combat_state = combat_state;

  return 0;
}

void resolve_combat(ecs_world_t *world) {
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  ecs_assert(gs->combat_state.attacking_card != 0, ECS_INVALID_PARAMETER, "Combat state attacking card not set");
  ecs_assert(gs->combat_state.defender_card != 0, ECS_INVALID_PARAMETER, "Combat state defender card not set");

  CurStats *attacking_card_cur_stats = ecs_get_mut(world, gs->combat_state.attacking_card, CurStats);
  ecs_assert(attacking_card_cur_stats != NULL, ECS_INVALID_PARAMETER, "Attacking card cur stats not found");
  CurStats *defender_card_cur_stats = ecs_get_mut(world, gs->combat_state.defender_card, CurStats);
  ecs_assert(defender_card_cur_stats != NULL, ECS_INVALID_PARAMETER, "Defender card cur stats not found");

  // If defender is frozen, no damage is dealt by either party
  // (Attacker can never be frozen due to attack validation)
  bool defender_frozen = ecs_has(world, gs->combat_state.defender_card, Frozen);
  if (defender_frozen) {
    cli_render_log("[Combat] Defender is frozen - no damage dealt");
  } else {
    attacking_card_cur_stats->cur_hp -= defender_card_cur_stats->cur_atk;
    defender_card_cur_stats->cur_hp -= attacking_card_cur_stats->cur_atk;
  }

  bool attacking_leader_defeated = false;
  bool defender_leader_defeated = false;
  if (attacking_card_cur_stats->cur_hp <= 0) {
    if (ecs_has(world, gs->combat_state.attacking_card, TLeader)) {
      attacking_leader_defeated = true;
    } else {
      discard_card(world, gs->combat_state.attacking_card);
    }
  }
  
  if (defender_card_cur_stats->cur_hp <= 0) {
    if (ecs_has(world, gs->combat_state.defender_card, TLeader)) {
      defender_leader_defeated = true;
    } else {
      discard_card(world, gs->combat_state.defender_card);
    }
  }

  if (attacking_leader_defeated && defender_leader_defeated) {
    gs->winner = 2;
  } else if (attacking_leader_defeated) {
    gs->winner = (gs->active_player_index + 1) % 2;
  } else if (defender_leader_defeated) {
    gs->winner = gs->active_player_index;
  }

  // TODO: Resolve "after attacking" or "when attacked" effects that trigger from the outcome
}
