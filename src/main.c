#include "world.h"
#include "abilities/ability_system.h"
#include "systems/phase_gate.h"
#include "utils/actions_util.h"
#include "utils/cli_rendering_util.h"
#include "utils/observation_util.h"
#include "utils/phase_utils.h"

int main(void) {
  ecs_world_t *world = azk_world_init(42);
  cli_render_init();

  bool game_over = false;
  while (!game_over) {
    const GameState *gs = ecs_singleton_get(world, GameState);
    ObservationData observation_data =
        create_observation_data(world, gs->active_player_index);
    cli_render_draw(world, &observation_data, gs);

    // Check for queued triggered effects to auto-process
    // Only process when no ability is currently active to avoid overwriting
    // AbilityContext mid-ability (e.g., during BOTTOM_DECK phase)
    if (azk_has_queued_triggered_effects(world) &&
        !azk_is_in_ability_phase(world)) {
      // Auto-process the queued effect (no user input needed)
      // This just validates and sets up AbilityContext - no systems run
      azk_process_triggered_effect_queue(world);
      game_over = is_game_over(world);
      continue;
    }

    // Run phase gate BEFORE checking for user action to handle auto-transitions
    // (e.g., pending combat after "when attacking" effects resolve)
    Phase phase_before = gs->phase;
    run_phase_gate_system(world);

    // Re-fetch gs since phase may have changed
    gs = ecs_singleton_get(world, GameState);

    // If phase changed, re-render before asking for user input
    if (gs->phase != phase_before) {
      game_over = is_game_over(world);
      continue;
    }

    // Normal flow: check if user input needed
    bool requires_user_action = phase_requires_user_action(world, gs->phase);
    if (requires_user_action) {
      azk_block_for_user_action(world);
    }

    run_phase_gate_system(world);
    ecs_progress(world, 0);
    game_over = is_game_over(world);
  }

  cli_render_shutdown();
  azk_world_fini(world);

  return 0;
}
