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

static void handle_declare_defender(ecs_world_t *world, GameState *gs,
                                    ActionContext *ac) {
  if (ac->user_action.type != ACT_DECLARE_DEFENDER) {
    ac->invalid_action = true;
    return;
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  DeclareDefenderIntent intent = {0};
  if (!azk_validate_declare_defender_action(world, gs, player, &ac->user_action,
                                            true, &intent)) {
    ac->invalid_action = true;
    return;
  }

  // Update combat state with new defender
  gs->combat_state.defender_card = intent.defender_card;
  gs->combat_state.defender_intercepted = true;

  cli_render_logf("[ResponseAction] Declared defender at garden index %d",
                  intent.garden_index);
}

static void handle_activate_leader_response_ability(ecs_world_t *world,
                                                     GameState *gs,
                                                     ActionContext *ac) {
  if (ac->user_action.type != ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY) {
    ac->invalid_action = true;
    return;
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  ActivateAbilityIntent intent = {0};
  if (!azk_validate_activate_garden_or_leader_ability_action(
          world, gs, player, &ac->user_action, true, &intent)) {
    ac->invalid_action = true;
    return;
  }

  // Pay IKZ cost (tap IKZ cards)
  for (int i = 0; i < intent.ikz_card_count; i++) {
    tap_card(world, intent.ikz_cards[i]);
  }

  // Mark once-per-turn as used
  if (ecs_has(world, intent.card, AOnceTurn)) {
    ecs_set(world, intent.card, AbilityRepeatContext, {
      .is_once_per_turn = true,
      .was_applied = true
    });
  }

  cli_render_logf("[ResponseAction] Activated leader response ability");

  // Trigger the leader's response ability
  azk_trigger_leader_response_ability(world, intent.card, player);
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
    tap_card(world, intent.ikz_cards[i]);
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

  // Ability phase actions are now handled by AbilityResolutionPhaseSystem
  // This system should only run when ability_phase == NONE
  // (enforced by phase_gate.c pipeline selection)

  // Check if defender can still respond
  // If not, auto-transition to combat without requiring NOOP input
  // Also check for queued effects - must process those first
  if (!azk_has_queued_triggered_effects(world) &&
      !defender_can_respond(world, gs, gs->active_player_index)) {
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
    // Spell ability triggers via azk_trigger_spell_ability.
    // If ability requires selection, AbilityResolutionPhaseSystem handles it.
    // Auto-transition after ability completion is handled there.
    break;

  case ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY:
    handle_activate_leader_response_ability(world, gs, ac);
    // Leader ability triggers via azk_trigger_leader_response_ability.
    // If ability requires selection, AbilityResolutionPhaseSystem handles it.
    break;

  case ACT_DECLARE_DEFENDER:
    handle_declare_defender(world, gs, ac);
    break;

  case ACT_NOOP:
    // Pass on responding - but first check for queued effects
    if (azk_has_queued_triggered_effects(world)) {
      // Can't pass while queued abilities need resolution
      cli_render_log("[ResponseAction] Cannot pass - queued effects pending");
      ac->invalid_action = true;
      break;
    }
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
