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

static const PhaseGateCache *phase_gate_cache(ecs_world_t *world) {
  const PhaseGateCache *cache = ecs_singleton_get(world, PhaseGateCache);
  ecs_assert(cache != NULL, ECS_INVALID_PARAMETER,
             "PhaseGateCache singleton missing");
  return cache;
}

static PhaseGateCache *phase_gate_cache_mut(ecs_world_t *world) {
  PhaseGateCache *cache = ecs_singleton_get_mut(world, PhaseGateCache);
  ecs_assert(cache != NULL, ECS_INVALID_PARAMETER,
             "PhaseGateCache singleton missing");
  return cache;
}

static void set_pipeline_for_phase(ecs_world_t *world,
                                   PhaseGateCache *cache,
                                   Phase phase) {
  ecs_entity_t pipeline = 0;
  switch (phase) {
  case PHASE_PREGAME_MULLIGAN:
    pipeline = cache->pipeline_mulligan;
    break;
  case PHASE_START_OF_TURN:
    pipeline = cache->pipeline_start_of_turn;
    break;
  case PHASE_MAIN:
    pipeline = cache->pipeline_main;
    break;
  case PHASE_RESPONSE_WINDOW:
    pipeline = cache->pipeline_response;
    break;
  case PHASE_COMBAT_RESOLVE:
    pipeline = cache->pipeline_combat;
    break;
  case PHASE_END_TURN:
    pipeline = cache->pipeline_end_turn;
    break;
  case PHASE_END_MATCH:
    pipeline = cache->pipeline_end_match;
    break;
  default:
    cli_render_logf("[PhaseGate] Unknown phase: %d", phase);
    exit(EXIT_FAILURE);
  }

  ecs_assert(pipeline != 0, ECS_INVALID_PARAMETER,
             "Pipeline missing for phase %d", phase);

  if (cache->current_pipeline == pipeline) {
    return;
  }

  ecs_set_pipeline(world, pipeline);
  cache->current_pipeline = pipeline;
}

void PhaseGate(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  PhaseGateCache *cache = phase_gate_cache_mut(world);
  GameState *gs = ecs_field(it, GameState, 0);
  Phase phase = gs->phase;

  // Check ability phase FIRST (except during END_MATCH)
  if (phase != PHASE_END_MATCH) {
    AbilityPhase ability_phase = azk_get_ability_phase(world);
    if (ability_phase != ABILITY_PHASE_NONE) {
      ecs_assert(cache->pipeline_ability != 0, ECS_INVALID_PARAMETER,
                 "Ability pipeline missing");
      if (cache->current_pipeline != cache->pipeline_ability) {
        ecs_set_pipeline(world, cache->pipeline_ability);
        cache->current_pipeline = cache->pipeline_ability;
      }
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
        set_pipeline_for_phase(world, cache, PHASE_RESPONSE_WINDOW);
        return;
      } else {
        gs->phase = PHASE_COMBAT_RESOLVE;
        ecs_singleton_modified(world, GameState);
        cli_render_logf(
            "[PhaseGate] When attacking effects resolved - proceeding to combat");
        set_pipeline_for_phase(world, cache, PHASE_COMBAT_RESOLVE);
        return;
      }
    }
  }

  set_pipeline_for_phase(world, cache, phase);
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

  ecs_entity_t phase_gate_system = ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = SYSTEM_PHASE_GATE,
      .add = ecs_ids( ecs_dependson(EcsOnUpdate) )
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) }
    },
    .callback = PhaseGate
  });

  PhaseGateCache cache = {0};
  cache.pipeline_mulligan = lookup_named_entity(world, PIPELINE_MULLIGAN);
  cache.pipeline_start_of_turn =
      lookup_named_entity(world, PIPELINE_START_OF_TURN);
  cache.pipeline_main = lookup_named_entity(world, PIPELINE_MAIN);
  cache.pipeline_response = lookup_named_entity(world, PIPELINE_RESPONSE);
  cache.pipeline_combat = lookup_named_entity(world, PIPELINE_COMBAT);
  cache.pipeline_end_turn = lookup_named_entity(world, PIPELINE_END_TURN);
  cache.pipeline_end_match = lookup_named_entity(world, PIPELINE_END_MATCH);
  cache.pipeline_ability = lookup_named_entity(world, PIPELINE_ABILITY);
  cache.phase_gate_system = phase_gate_system;
  ecs_singleton_set_ptr(world, PhaseGateCache, &cache);

  const GameState *gs = ecs_singleton_get(world, GameState);
  PhaseGateCache *cache_ptr = phase_gate_cache_mut(world);
  set_pipeline_for_phase(world, cache_ptr, gs->phase);
}

void run_phase_gate_system(ecs_world_t *world) {
  const PhaseGateCache *cache = phase_gate_cache(world);
  ecs_assert(cache->phase_gate_system != 0, ECS_INVALID_PARAMETER,
             "Phase gate system missing");
  ecs_run(world, cache->phase_gate_system, 0, NULL);
}
