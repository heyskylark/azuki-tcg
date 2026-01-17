#include <stdlib.h>
#include <flecs.h>
#include "systems/phase_gate.h"
#include "abilities/ability_system.h"
#include "constants/game.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

static ecs_entity_t MulliganPipeline;
static ecs_entity_t StartOfTurnPipeline;
static ecs_entity_t MainPipeline;
static ecs_entity_t ResponseWindowPipeline;
static ecs_entity_t CombatResolvePipeline;
static ecs_entity_t EndTurnPipeline;
static ecs_entity_t EndMatchPipeline;
static ecs_entity_t AbilityResolutionPipeline;
static ecs_entity_t s_phase_gate_system = 0;

static void set_pipeline_for_phase(ecs_world_t *world, Phase phase) {
  switch (phase) {
  case PHASE_PREGAME_MULLIGAN:
    ecs_set_pipeline(world, MulliganPipeline);
    break;
  case PHASE_START_OF_TURN:
    ecs_set_pipeline(world, StartOfTurnPipeline);
    break;
  case PHASE_MAIN:
    ecs_set_pipeline(world, MainPipeline);
    break;
  case PHASE_RESPONSE_WINDOW:
    ecs_set_pipeline(world, ResponseWindowPipeline);
    break;
  case PHASE_COMBAT_RESOLVE:
    ecs_set_pipeline(world, CombatResolvePipeline);
    break;
  case PHASE_END_TURN:
    ecs_set_pipeline(world, EndTurnPipeline);
    break;
  case PHASE_END_MATCH:
    ecs_set_pipeline(world, EndMatchPipeline);
    break;
  default:
    cli_render_logf("[PhaseGate] Unknown phase: %d", phase);
    exit(EXIT_FAILURE);
  }
}

void PhaseGate(ecs_iter_t *it) {
  ecs_world_t *world = (ecs_world_t *)ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  Phase phase = gs->phase;

  // Check ability phase FIRST (except during END_MATCH)
  if (phase != PHASE_END_MATCH) {
    AbilityPhase ability_phase = azk_get_ability_phase(world);
    if (ability_phase != ABILITY_PHASE_NONE) {
      ecs_set_pipeline(world, AbilityResolutionPipeline);
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
  MulliganPipeline = ecs_pipeline_init(world, &(ecs_pipeline_desc_t){
    .query.terms = {
        { .id = EcsSystem }, // mandatory
        { .id = TMulligan }
    }
  });
  StartOfTurnPipeline = ecs_pipeline_init(world, &(ecs_pipeline_desc_t){
    .query.terms = {
        { .id = EcsSystem }, // mandatory
        { .id = TStartOfTurn }
    }
  });
  MainPipeline = ecs_pipeline_init(world, &(ecs_pipeline_desc_t){
    .query.terms = {
        { .id = EcsSystem }, // mandatory
        { .id = TMain }
    }
  });
  ResponseWindowPipeline = ecs_pipeline_init(world, &(ecs_pipeline_desc_t){
    .query.terms = {
        { .id = EcsSystem }, // mandatory
        { .id = TResponseWindow }
    }
  });
  CombatResolvePipeline = ecs_pipeline_init(world, &(ecs_pipeline_desc_t){
    .query.terms = {
        { .id = EcsSystem }, // mandatory
        { .id = TCombatResolve }
    }
  });
  EndTurnPipeline = ecs_pipeline_init(world, &(ecs_pipeline_desc_t){
    .query.terms = {
        { .id = EcsSystem }, // mandatory
        { .id = TEndTurn }
    }
  });
  EndMatchPipeline = ecs_pipeline_init(world, &(ecs_pipeline_desc_t){
    .query.terms = {
        { .id = EcsSystem }, // mandatory
        { .id = TEndMatch }
    }
  });
  AbilityResolutionPipeline = ecs_pipeline_init(world, &(ecs_pipeline_desc_t){
    .query.terms = {
        { .id = EcsSystem }, // mandatory
        { .id = TAbilityResolution }
    }
  });

  const GameState *gs = ecs_singleton_get(world, GameState);
  set_pipeline_for_phase(world, gs->phase);

  s_phase_gate_system = ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "PhaseGate",
      .add = ecs_ids( ecs_dependson(EcsOnUpdate) )
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) }
    },
    .callback = PhaseGate
  });
}

void run_phase_gate_system(ecs_world_t *world) {
  ecs_assert(s_phase_gate_system != 0, ECS_INVALID_PARAMETER, "Phase gate system not initialized");
  ecs_run(world, s_phase_gate_system, 0, NULL);
}
