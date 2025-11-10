#include <stdio.h>

#include "azuki/engine.h"

/*
 * This file is a placeholder entry point for RL-specific harnesses.
 * It demonstrates how to drive the Azuki engine without the CLI renderer
 * so Python bindings or C-based trainers can plug their own loops in.
 */
int main(void) {
  AzkEngine *engine = azk_engine_create(42);
  if (!engine) {
    fprintf(stderr, "Failed to initialize Azuki engine\n");
    return 1;
  }

  ObservationData observation = {0};

  while (!azk_engine_is_game_over(engine)) {
    azk_engine_tick(engine);

    if (!azk_engine_requires_action(engine)) {
      continue;
    }

    if (!azk_engine_observe(engine, &observation)) {
      fprintf(stderr, "Failed to capture observation\n");
      break;
    }

    /*
     * TODO: Replace this stub with calls into the RL policy/bindings so an
     *       action can be produced and submitted via azk_engine_submit_action.
     */
    break;
  }

  azk_engine_destroy(engine);
  return 0;
}
