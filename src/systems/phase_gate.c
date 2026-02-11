#include <stdlib.h>
#include <flecs.h>
#include "systems/phase_gate.h"
#include "abilities/ability_system.h"
#include "constants/game.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

static const char *PIPELINE_MULLIGAN = "MulliganPipeline";
static const char *PIPELINE_START_OF_TURN = "StartOfTurnPipeline";
static const char *PIPELINE_MAIN = "MainPipeline";
static const char *PIPELINE_RESPONSE = "ResponseWindowPipeline";
static const char *PIPELINE_COMBAT = "CombatResolvePipeline";
static const char *PIPELINE_END_TURN = "EndTurnPipeline";
static const char *PIPELINE_END_MATCH = "EndMatchPipeline";
static const char *PIPELINE_ABILITY = "AbilityResolutionPipeline";
static const char *SYSTEM_PHASE_GATE = "PhaseGate";

static ecs_entity_t lookup_named_entity(ecs_world_t *world,
                                        const char *entity_name) {
  ecs_entity_t entity = ecs_lookup(world, entity_name);
  ecs_assert(entity != 0, ECS_INVALID_PARAMETER,
             "Entity %s not found in world", entity_name);
  return entity;
}

static void create_named_pipeline(ecs_world_t *world, const char *pipeline_name,
                                  ecs_entity_t phase_tag) {
  ecs_pipeline(world,
               {.entity = ecs_entity(world, {.name = pipeline_name}),
                .query.terms = {{.id = EcsSystem}, {.id = phase_tag}}});
}

static void set_pipeline_for_phase(ecs_world_t *world, Phase phase) {
  switch (phase) {
  case PHASE_PREGAME_MULLIGAN:
    ecs_set_pipeline(world, lookup_named_entity(world, PIPELINE_MULLIGAN));
    break;
  case PHASE_START_OF_TURN:
    ecs_set_pipeline(world, lookup_named_entity(world, PIPELINE_START_OF_TURN));
    break;
  case PHASE_MAIN:
    ecs_set_pipeline(world, lookup_named_entity(world, PIPELINE_MAIN));
    break;
  case PHASE_RESPONSE_WINDOW:
    ecs_set_pipeline(world, lookup_named_entity(world, PIPELINE_RESPONSE));
    break;
  case PHASE_COMBAT_RESOLVE:
    ecs_set_pipeline(world, lookup_named_entity(world, PIPELINE_COMBAT));
    break;
  case PHASE_END_TURN:
    ecs_set_pipeline(world, lookup_named_entity(world, PIPELINE_END_TURN));
    break;
  case PHASE_END_MATCH:
    ecs_set_pipeline(world, lookup_named_entity(world, PIPELINE_END_MATCH));
    break;
  default:
    cli_render_logf("[PhaseGate] Unknown phase: %d", phase);
    exit(EXIT_FAILURE);
  }
}

void PhaseGate(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  Phase phase = gs->phase;

  // Check ability phase FIRST (except during END_MATCH)
  if (phase != PHASE_END_MATCH) {
    AbilityPhase ability_phase = azk_get_ability_phase(world);
    if (ability_phase != ABILITY_PHASE_NONE) {
      ecs_set_pipeline(world, lookup_named_entity(world, PIPELINE_ABILITY));
      return;
    }

    // Check for pending combat transition after "when attacking" effects
    // resolved If in MAIN phase with pending combat and no queued effects,
    // transition to response window
    if (phase == PHASE_MAIN && gs->combat_state.attacking_card != 0 &&
        !azk_has_queued_triggered_effects(world)) {
      // Active player is the attacker (we stayed in MAIN after queueing effect)
      uint8_t defender_index =
          (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;

      if (defender_can_respond(world, gs, defender_index)) {
        gs->phase = PHASE_RESPONSE_WINDOW;
        gs->active_player_index = defender_index;
        ecs_singleton_modified(world, GameState);
        cli_render_logf("[PhaseGate] When attacking effects resolved - "
                        "defender has response options");
        set_pipeline_for_phase(world, PHASE_RESPONSE_WINDOW);
        return;
      } else {
        gs->phase = PHASE_COMBAT_RESOLVE;
        ecs_singleton_modified(world, GameState);
        cli_render_logf(
            "[PhaseGate] When attacking effects resolved - proceeding to combat");
        set_pipeline_for_phase(world, PHASE_COMBAT_RESOLVE);
        return;
      }
    }
  }

  set_pipeline_for_phase(world, phase);
}

void init_phase_gate_system(ecs_world_t *world) {
  create_named_pipeline(world, PIPELINE_MULLIGAN, TMulligan);
  create_named_pipeline(world, PIPELINE_START_OF_TURN, TStartOfTurn);
  create_named_pipeline(world, PIPELINE_MAIN, TMain);
  create_named_pipeline(world, PIPELINE_RESPONSE, TResponseWindow);
  create_named_pipeline(world, PIPELINE_COMBAT, TCombatResolve);
  create_named_pipeline(world, PIPELINE_END_TURN, TEndTurn);
  create_named_pipeline(world, PIPELINE_END_MATCH, TEndMatch);
  create_named_pipeline(world, PIPELINE_ABILITY, TAbilityResolution);

  const GameState *gs = ecs_singleton_get(world, GameState);
  set_pipeline_for_phase(world, gs->phase);

  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = SYSTEM_PHASE_GATE,
      .add = ecs_ids( ecs_dependson(EcsOnUpdate) )
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) }
    },
    .callback = PhaseGate
  });
}

void run_phase_gate_system(ecs_world_t *world) {
  ecs_entity_t phase_gate_system = lookup_named_entity(world, SYSTEM_PHASE_GATE);
  ecs_run(world, phase_gate_system, 0, NULL);
}
