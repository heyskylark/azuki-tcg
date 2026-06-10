#include "systems/combat_resolve_phase.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "components/components.h"
#include "utils/ability_util.h"
#include "utils/cli_rendering_util.h"
#include "utils/combat_util.h"
#include "utils/observation_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

static bool is_card_still_in_owner_battle_zone(ecs_world_t *world,
                                               ecs_entity_t card,
                                               const GameState *gs,
                                               bool allow_alley) {
  // Card may have been deleted outright (not just moved zones) before combat
  // resolution; touching a dead entity would crash in release builds where
  // flecs validity checks compile out.
  if (card == 0 || !ecs_is_alive(world, card)) {
    return false;
  }

  if (ecs_has(world, card, TLeader)) {
    return true;
  }

  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (owner == 0) {
    return false;
  }

  uint8_t owner_num = get_player_number(world, owner);
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  return parent == gs->zones[owner_num].garden ||
         (allow_alley && parent == gs->zones[owner_num].alley);
}

void HandleCombatResolution(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);

  bool attacker_valid = is_card_still_in_owner_battle_zone(
      world, gs->combat_state.attacking_card, gs, false);
  bool defender_valid = is_card_still_in_owner_battle_zone(
      world, gs->combat_state.defender_card, gs, true);

  if (!attacker_valid || !defender_valid) {
    cli_render_logf("[CombatResolution] Combat fizzled - %s removed",
                    !attacker_valid ? "attacker" : "defender");
    gs->combat_state.attacking_card = 0;
    gs->combat_state.defender_card = 0;
    gs->combat_state.defender_intercepted = false;
    gs->phase = PHASE_MAIN;
    return;
  }

  resolve_combat(world);

  const ecs_entity_t attacker = gs->combat_state.attacking_card;
  ecs_entity_t attacker_owner = 0;
  if (attacker != 0) {
    attacker_owner = ecs_get_target(world, attacker, Rel_OwnedBy, 0);
    const CardId *attacker_id = ecs_get(world, attacker, CardId);
    if (attacker_id != NULL &&
        azk_has_ability_with_timing(attacker_id->id, ecs_id(AAfterAttacking))) {
      ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
      uint8_t ability_count = azk_collect_card_timed_abilities(
          world, attacker, ecs_id(AAfterAttacking), abilities,
          AZK_MAX_CARD_ABILITIES);
      for (uint8_t i = 0; i < ability_count; ++i) {
        azk_queue_triggered_effect(world, abilities[i], attacker_owner,
                                   TIMING_TAG_AFTER_ATTACKING);
      }
    }
  }

  // Reset combat state
  gs->combat_state.attacking_card = 0;
  gs->combat_state.defender_card = 0;
  gs->combat_state.defender_intercepted = false;

  if (is_game_over(world)) {
    gs->phase = PHASE_END_MATCH;
  } else {
    gs->phase = PHASE_MAIN;
  }

  cli_render_logf("[CombatResolution] Combat resolution");
}

void init_combat_resolve_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "CombatResolvePhaseSystem",
      .add = ecs_ids(TCombatResolve)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) }
    },
    .callback = HandleCombatResolution
  });
}
