#include "systems/response_phase.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "components/components.h"
#include "constants/game.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/combat_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/weapon_util.h"
#include "utils/zone_util.h"
#include "validation/action_validation.h"

static int play_entity_to_garden_or_alley_response(ecs_world_t *world,
                                                   GameState *gs,
                                                   ActionContext *ac,
                                                   ZonePlacementType placement_type) {
  ecs_entity_t player = gs->players[gs->active_player_index];
  PlayEntityIntent intent = {0};
  if (!azk_validate_play_entity_action(world, gs, player, placement_type,
                                       &ac->user_action, true, &intent)) {
    return -1;
  }

  int result = summon_card_into_zone_index(world, &intent);
  if (result < 0) {
    return result;
  }

  if (placement_type == ZONE_GARDEN) {
    gs->entities_played_garden_this_turn[gs->active_player_index]++;
  } else {
    gs->entities_played_alley_this_turn[gs->active_player_index]++;
  }
  gs->cards_played_this_turn[gs->active_player_index]++;
  gs->next_card_play_cost_reduction[gs->active_player_index] = 0;

  azk_trigger_on_play_ability(world, intent.card, intent.player);
  return 0;
}

static void handle_play_response_entity_to_garden(ecs_world_t *world,
                                                  GameState *gs,
                                                  ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_GARDEN) {
    ac->invalid_action = true;
    return;
  }

  if (play_entity_to_garden_or_alley_response(world, gs, ac, ZONE_GARDEN) < 0) {
    ac->invalid_action = true;
    return;
  }

  cli_render_logf("[ResponseAction] Played response entity to garden");
}

static void handle_play_response_entity_to_alley(ecs_world_t *world,
                                                 GameState *gs,
                                                 ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_ALLEY) {
    ac->invalid_action = true;
    return;
  }

  if (play_entity_to_garden_or_alley_response(world, gs, ac, ZONE_ALLEY) < 0) {
    ac->invalid_action = true;
    return;
  }

  cli_render_logf("[ResponseAction] Played response entity to alley");
}

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

  // Defender redirects are a tap action, but they ignore cooldown.
  tap_card(world, intent.defender_card);

  // Update combat state with new defender
  gs->combat_state.defender_card = intent.defender_card;
  gs->combat_state.defender_intercepted = true;

  // Log defender declared
  azk_log_defender_declared(world, intent.defender_card);

  cli_render_logf("[ResponseAction] Declared defender at garden index %d",
                  intent.garden_index);
}

static void handle_activate_response_ability(ecs_world_t *world, GameState *gs,
                                             ActionContext *ac) {
  if (ac->user_action.type != ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY &&
      ac->user_action.type != ACT_ACTIVATE_ALLEY_ABILITY) {
    ac->invalid_action = true;
    return;
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  ActivateAbilityIntent intent = {0};
  bool valid = false;
  if (ac->user_action.type == ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY) {
    valid = azk_validate_activate_garden_or_leader_ability_action(
        world, gs, player, &ac->user_action, true, &intent);
  } else {
    valid = azk_validate_activate_alley_ability_action(
        world, gs, player, &ac->user_action, true, &intent);
  }

  if (!valid) {
    ac->invalid_action = true;
    return;
  }

  // Pay IKZ cost (tap IKZ cards)
  for (int i = 0; i < intent.ikz_card_count; i++) {
    tap_card(world, intent.ikz_cards[i]);
  }

  // Note: Once-per-turn marking is now handled in azk_clear_ability_context()
  // when the ability actually completes (not when it's triggered)

  cli_render_logf("[ResponseAction] Activated response ability");

  // Trigger the selected card's response ability
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
  gs->cards_played_this_turn[gs->active_player_index]++;
  gs->next_card_play_cost_reduction[gs->active_player_index] = 0;

  cli_render_logf("[ResponseAction] Played spell from hand");

  // Trigger the spell's ability
  azk_trigger_spell_ability(world, intent.spell_card, player);
}

static void handle_attach_weapon_from_hand(ecs_world_t *world, GameState *gs,
                                           ActionContext *ac) {
  if (ac->user_action.type != ACT_ATTACH_WEAPON_FROM_HAND) {
    ac->invalid_action = true;
    return;
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  AttachWeaponIntent intent = {0};
  if (!azk_validate_attach_weapon_action(world, gs, player, &ac->user_action,
                                         true, &intent)) {
    ac->invalid_action = true;
    return;
  }

  int result = attach_weapon_from_hand(world, &intent);
  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  gs->cards_played_this_turn[gs->active_player_index]++;
  gs->next_card_play_cost_reduction[gs->active_player_index] = 0;

  azk_trigger_on_play_ability(world, intent.weapon_card, intent.player);
  azk_trigger_when_equipped_ability(world, intent.weapon_card, intent.player);
  azk_trigger_when_equipped_ability(world, intent.target_card, intent.player);

  cli_render_logf("[ResponseAction] Attached response weapon");
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
    bool queued_when_attacked = azk_transition_to_combat_resolve(world);
    cli_render_log(queued_when_attacked
                       ? "[ResponseAction] Response window closed - processing "
                         "when attacked effects"
                       : "[ResponseAction] Defender has no response options - "
                         "proceeding to combat");
    return;
  }

  // Normal response phase handling
  switch (ac->user_action.type) {
  case ACT_PLAY_ENTITY_TO_GARDEN:
    handle_play_response_entity_to_garden(world, gs, ac);
    break;

  case ACT_PLAY_ENTITY_TO_ALLEY:
    handle_play_response_entity_to_alley(world, gs, ac);
    break;

  case ACT_PLAY_SPELL_FROM_HAND:
    handle_play_spell_from_hand(world, gs, ac);
    // Spell ability triggers via azk_trigger_spell_ability.
    // If ability requires selection, AbilityResolutionPhaseSystem handles it.
    // Auto-transition after ability completion is handled there.
    break;

  case ACT_ATTACH_WEAPON_FROM_HAND:
    handle_attach_weapon_from_hand(world, gs, ac);
    break;

  case ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY:
  case ACT_ACTIVATE_ALLEY_ABILITY:
    handle_activate_response_ability(world, gs, ac);
    // Card response abilities trigger via azk_trigger_leader_response_ability.
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
    bool queued_when_attacked = azk_transition_to_combat_resolve(world);
    cli_render_log(queued_when_attacked
                       ? "[ResponseAction] Defender passes - processing when "
                         "attacked effects"
                       : "[ResponseAction] Defender passes - proceeding to "
                         "combat");
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
