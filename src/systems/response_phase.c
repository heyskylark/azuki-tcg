#include "systems/response_phase.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "components/components.h"
#include "constants/game.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"
#include "validation/action_validation.h"

/**
 * Check if ability processing is complete and defender has no more response
 * options. If so, auto-transition to combat resolve phase. Returns true if
 * auto-transition occurred.
 */
static bool check_auto_transition_after_ability(ecs_world_t *world,
                                                GameState *gs) {
  AbilityPhase current_phase = azk_get_ability_phase(world);
  if (current_phase == ABILITY_PHASE_NONE) {
    // Ability processing complete - check if defender can still respond
    if (!defender_can_respond(world, gs, gs->active_player_index)) {
      cli_render_log("[ResponseAction] Ability complete, no more response "
                     "options - proceeding to combat");
      gs->active_player_index =
          (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;
      gs->phase = PHASE_COMBAT_RESOLVE;
      return true;
    }
  }
  return false;
}

static void handle_play_spell_from_hand(ecs_world_t *world, GameState *gs,
                                        ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_SPELL_FROM_HAND) {
    ac->invalid_action = true;
    return;
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  PlaySpellIntent intent = {0};
  if (!azk_validate_play_spell_action(world, gs, player, &ac->user_action, true,
                                      &intent)) {
    ac->invalid_action = true;
    return;
  }

  // Pay IKZ cost (tap IKZ cards)
  for (int i = 0; i < intent.ikz_card_count; i++) {
    set_card_to_tapped(world, intent.ikz_cards[i]);
  }

  // Move spell card to discard
  discard_card(world, intent.spell_card);

  cli_render_logf("[ResponseAction] Played spell from hand");

  // Trigger the spell's ability
  azk_trigger_spell_ability(world, intent.spell_card, player);
}

void HandleResponseAction(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  ActionContext *ac = ecs_field(it, ActionContext, 1);

  // Check if we're in an ability sub-phase
  AbilityPhase ability_phase = azk_get_ability_phase(world);

  if (ability_phase != ABILITY_PHASE_NONE) {
    // In ability phase - handle ability-related actions
    switch (ac->user_action.type) {
    case ACT_CONFIRM_ABILITY:
      if (ability_phase == ABILITY_PHASE_CONFIRMATION) {
        if (!azk_process_ability_confirmation(world)) {
          ac->invalid_action = true;
        } else {
          check_auto_transition_after_ability(world, gs);
        }
      } else {
        cli_render_logf("[ResponseAction] ACT_CONFIRM_ABILITY not valid in "
                        "ability phase %d",
                        ability_phase);
        ac->invalid_action = true;
      }
      break;

    case ACT_NOOP:
      // In confirmation phase, NOOP means decline
      if (ability_phase == ABILITY_PHASE_CONFIRMATION) {
        if (!azk_process_ability_decline(world)) {
          ac->invalid_action = true;
        } else {
          check_auto_transition_after_ability(world, gs);
        }
      } else if (ability_phase == ABILITY_PHASE_EFFECT_SELECTION) {
        // In effect selection phase, NOOP means skip (for "up to" effects with
        // min=0)
        if (!azk_process_effect_skip(world)) {
          ac->invalid_action = true;
        } else {
          check_auto_transition_after_ability(world, gs);
        }
      } else {
        cli_render_logf(
            "[ResponseAction] ACT_NOOP not valid in ability phase %d",
            ability_phase);
        ac->invalid_action = true;
      }
      break;

    case ACT_SELECT_COST_TARGET:
      if (ability_phase == ABILITY_PHASE_COST_SELECTION) {
        if (!azk_process_cost_selection(world, ac->user_action.subaction_1)) {
          ac->invalid_action = true;
        } else {
          check_auto_transition_after_ability(world, gs);
        }
      } else {
        cli_render_logf("[ResponseAction] ACT_SELECT_COST_TARGET not valid in "
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
          check_auto_transition_after_ability(world, gs);
        }
      } else {
        cli_render_logf("[ResponseAction] ACT_SELECT_EFFECT_TARGET not valid "
                        "in ability phase %d",
                        ability_phase);
        ac->invalid_action = true;
      }
      break;

    default:
      cli_render_logf(
          "[ResponseAction] Action type %d not valid during ability phase",
          ac->user_action.type);
      ac->invalid_action = true;
      break;
    }
    return;
  }

  // No ability sub-phase active - check if defender can still respond
  // If not, auto-transition to combat without requiring NOOP input
  if (!defender_can_respond(world, gs, gs->active_player_index)) {
    cli_render_log("[ResponseAction] Defender has no response options - "
                   "proceeding to combat");
    gs->active_player_index =
        (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;
    gs->phase = PHASE_COMBAT_RESOLVE;
    return;
  }

  // Normal response phase handling
  switch (ac->user_action.type) {
  case ACT_PLAY_SPELL_FROM_HAND:
    handle_play_spell_from_hand(world, gs, ac);
    // Check for auto-transition if spell applied immediately without sub-phases
    if (!ac->invalid_action) {
      check_auto_transition_after_ability(world, gs);
    }
    break;

  case ACT_NOOP:
    // Pass on responding - proceed to combat
    if (!azk_validate_simple_action(world, gs,
                                    gs->players[gs->active_player_index],
                                    ac->user_action.type, true)) {
      ac->invalid_action = true;
      break;
    }
    cli_render_log("[ResponseAction] Defender passes - proceeding to combat");
    // Switch back to attacker for combat resolution tracking
    gs->active_player_index =
        (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;
    gs->phase = PHASE_COMBAT_RESOLVE;
    break;

  default:
    cli_render_logf("[ResponseAction] Unknown response action type: %d",
                    ac->user_action.type);
    ac->invalid_action = true;
    break;
  }
}

void init_response_phase_system(ecs_world_t *world) {
  ecs_system(world,
             {.entity = ecs_entity(world, {.name = "ResponsePhaseSystem",
                                           .add = ecs_ids(TResponseWindow)}),
              .query.terms = {{.id = ecs_id(GameState),
                               .src.id = ecs_id(GameState),
                               .inout = EcsIn},
                              {.id = ecs_id(ActionContext),
                               .src.id = ecs_id(ActionContext)}},
              .callback = HandleResponseAction});
}
