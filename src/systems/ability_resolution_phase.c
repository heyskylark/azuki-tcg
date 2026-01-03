#include "systems/ability_resolution_phase.h"
#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "components/components.h"
#include "constants/game.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

/**
 * Check if ability processing is complete and if we're in response phase,
 * handle auto-transition to combat resolve if defender has no more options.
 */
static void check_post_ability_transition(ecs_world_t *world, GameState *gs) {
  AbilityPhase current_phase = azk_get_ability_phase(world);
  if (current_phase != ABILITY_PHASE_NONE) {
    return; // Still in ability resolution
  }

  // Don't auto-transition if there are queued effects pending
  if (azk_has_queued_triggered_effects(world)) {
    return;
  }

  // Only handle auto-transition for response phase
  if (gs->phase == PHASE_RESPONSE_WINDOW) {
    // Check if defender can still respond
    if (!defender_can_respond(world, gs, gs->active_player_index)) {
      cli_render_log("[AbilityResolution] Ability complete, no more response "
                     "options - proceeding to combat");
      gs->active_player_index =
          (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;
      gs->phase = PHASE_COMBAT_RESOLVE;
    }
  }
}

void HandleAbilityResolution(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  ActionContext *ac = ecs_field(it, ActionContext, 1);

  AbilityPhase ability_phase = azk_get_ability_phase(world);

  // This system should only run when ability_phase != NONE
  // (enforced by phase_gate.c)
  if (ability_phase == ABILITY_PHASE_NONE) {
    cli_render_log(
        "[AbilityResolution] ERROR: System ran with no active ability phase");
    return;
  }

  switch (ac->user_action.type) {
  case ACT_CONFIRM_ABILITY:
    if (ability_phase == ABILITY_PHASE_CONFIRMATION) {
      if (!azk_process_ability_confirmation(world)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf("[AbilityResolution] ACT_CONFIRM_ABILITY not valid in "
                      "ability phase %d",
                      ability_phase);
      ac->invalid_action = true;
    }
    break;

  case ACT_NOOP:
    // NOOP has different meanings in different ability phases
    if (ability_phase == ABILITY_PHASE_CONFIRMATION) {
      if (!azk_process_ability_decline(world)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else if (ability_phase == ABILITY_PHASE_EFFECT_SELECTION) {
      if (!azk_process_effect_skip(world)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else if (ability_phase == ABILITY_PHASE_SELECTION_PICK) {
      if (!azk_process_skip_selection(world)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf(
          "[AbilityResolution] ACT_NOOP not valid in ability phase %d",
          ability_phase);
      ac->invalid_action = true;
    }
    break;

  case ACT_SELECT_COST_TARGET:
    if (ability_phase == ABILITY_PHASE_COST_SELECTION) {
      if (!azk_process_cost_selection(world, ac->user_action.subaction_1)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf("[AbilityResolution] ACT_SELECT_COST_TARGET not valid in "
                      "ability phase %d",
                      ability_phase);
      ac->invalid_action = true;
    }
    break;

  case ACT_SELECT_EFFECT_TARGET:
    if (ability_phase == ABILITY_PHASE_EFFECT_SELECTION) {
      if (!azk_process_effect_selection(world, ac->user_action.subaction_1)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf("[AbilityResolution] ACT_SELECT_EFFECT_TARGET not valid "
                      "in ability phase %d",
                      ability_phase);
      ac->invalid_action = true;
    }
    break;

  case ACT_SELECT_FROM_SELECTION:
    if (ability_phase == ABILITY_PHASE_SELECTION_PICK) {
      if (!azk_process_selection_pick(world, ac->user_action.subaction_1)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf("[AbilityResolution] ACT_SELECT_FROM_SELECTION not valid "
                      "in ability phase %d",
                      ability_phase);
      ac->invalid_action = true;
    }
    break;

  case ACT_SELECT_TO_ALLEY:
    if (ability_phase == ABILITY_PHASE_SELECTION_PICK) {
      if (!azk_process_selection_to_alley(world, ac->user_action.subaction_1,
                                          ac->user_action.subaction_2)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf("[AbilityResolution] ACT_SELECT_TO_ALLEY not valid in "
                      "ability phase %d",
                      ability_phase);
      ac->invalid_action = true;
    }
    break;

  case ACT_SELECT_TO_EQUIP:
    if (ability_phase == ABILITY_PHASE_SELECTION_PICK) {
      if (!azk_process_selection_to_equip(world, ac->user_action.subaction_1,
                                          ac->user_action.subaction_2)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf("[AbilityResolution] ACT_SELECT_TO_EQUIP not valid in "
                      "ability phase %d",
                      ability_phase);
      ac->invalid_action = true;
    }
    break;

  case ACT_BOTTOM_DECK_CARD:
    if (ability_phase == ABILITY_PHASE_BOTTOM_DECK) {
      if (!azk_process_bottom_deck(world, ac->user_action.subaction_1)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf("[AbilityResolution] ACT_BOTTOM_DECK_CARD not valid in "
                      "ability phase %d",
                      ability_phase);
      ac->invalid_action = true;
    }
    break;

  case ACT_BOTTOM_DECK_ALL:
    if (ability_phase == ABILITY_PHASE_BOTTOM_DECK) {
      if (!azk_process_bottom_deck_all(world)) {
        ac->invalid_action = true;
      } else {
        check_post_ability_transition(world, gs);
      }
    } else {
      cli_render_logf("[AbilityResolution] ACT_BOTTOM_DECK_ALL not valid in "
                      "ability phase %d",
                      ability_phase);
      ac->invalid_action = true;
    }
    break;

  default:
    cli_render_logf(
        "[AbilityResolution] Action type %d not valid during ability phase",
        ac->user_action.type);
    ac->invalid_action = true;
    break;
  }
}

void init_ability_resolution_phase_system(ecs_world_t *world) {
  ecs_system(world,
             {.entity = ecs_entity(world, {.name = "AbilityResolutionSystem",
                                           .add = ecs_ids(TAbilityResolution)}),
              .query.terms = {{.id = ecs_id(GameState),
                               .src.id = ecs_id(GameState),
                               .inout = EcsIn},
                              {.id = ecs_id(ActionContext),
                               .src.id = ecs_id(ActionContext)}},
              .callback = HandleAbilityResolution});
}
