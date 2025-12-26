#include "systems/main_phase.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "components/components.h"
#include "constants/game.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/combat_util.h"
#include "utils/player_util.h"
#include "utils/weapon_util.h"
#include "utils/zone_util.h"
#include "validation/action_validation.h"

static int play_entity_to_garden_or_alley(ecs_world_t *world, GameState *gs,
                                          ActionContext *ac,
                                          ZonePlacementType placement_type,
                                          ecs_entity_t *out_card,
                                          ecs_entity_t *out_player) {
  ecs_entity_t player = gs->players[gs->active_player_index];
  PlayEntityIntent intent = {0};
  if (!azk_validate_play_entity_action(world, gs, player, placement_type,
                                       &ac->user_action, true, &intent)) {
    return -1;
  }

  int result = summon_card_into_zone_index(world, &intent);
  if (result == 0) {
    azk_trigger_on_play_ability(world, intent.card, intent.player);
  }

  return result;
}

/**
 * Expected Action: ACT_PLAY_ENTITY_TO_GARDEN, hand_index, garden_index, use ikz
 * token
 */
static void handle_play_entity_to_garden(ecs_world_t *world, GameState *gs,
                                         ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_GARDEN) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t played_card = 0;
  ecs_entity_t player = 0;
  int result = play_entity_to_garden_or_alley(world, gs, ac, ZONE_GARDEN,
                                              &played_card, &player);
  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  cli_render_logf("[MainAction] Played entity to garden");
}

/**
 * Expected Action: ACT_PLAY_ENTITY_TO_ALLEY, hand_index, alley_index, use ikz
 * token
 */
static void handle_play_entity_to_alley(ecs_world_t *world, GameState *gs,
                                        ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_ALLEY) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t played_card = 0;
  ecs_entity_t player = 0;
  int result = play_entity_to_garden_or_alley(world, gs, ac, ZONE_ALLEY,
                                              &played_card, &player);
  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  cli_render_logf("[MainAction] Played entity to alley");
}

/**
 * Expected Action: ACT_GATE_PORTAL, alley_index, garden_index, 0
 */
static void handle_gate_portal(ecs_world_t *world, GameState *gs,
                               ActionContext *ac) {
  if (ac->user_action.type != ACT_GATE_PORTAL) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  GatePortalIntent intent = {0};
  if (!azk_validate_gate_portal_action(world, gs, player, &ac->user_action,
                                       true, &intent)) {
    ac->invalid_action = true;
    return;
  }

  int result = gate_card_into_garden(world, &intent);

  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  cli_render_logf("[MainAction] Gate portal");
}

/**
 * Check if the defending player has any response spells they can play.
 * Returns true if defender has at least one response spell in hand with enough
 * IKZ to cast.
 */
static bool defender_can_respond(ecs_world_t *world, const GameState *gs,
                                 uint8_t defender_index) {
  ecs_entity_t hand = gs->zones[defender_index].hand;
  ecs_entity_t ikz_area = gs->zones[defender_index].ikz_area;

  // Count available untapped IKZ
  ecs_entities_t ikz_cards = ecs_get_ordered_children(world, ikz_area);
  uint8_t available_ikz = 0;
  for (int i = 0; i < ikz_cards.count; i++) {
    if (!is_card_tapped(world, ikz_cards.ids[i])) {
      available_ikz++;
    }
  }

  // Check IKZ token
  ecs_entity_t defender_player = gs->players[defender_index];
  const IKZToken *ikz_token = ecs_get(world, defender_player, IKZToken);
  if (ikz_token && ikz_token->ikz_token != 0 &&
      !is_card_tapped(world, ikz_token->ikz_token)) {
    available_ikz++;
  }

  // Check if any card in hand is a response spell with affordable cost
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
  for (int i = 0; i < hand_cards.count; i++) {
    ecs_entity_t card = hand_cards.ids[i];

    // Check if it's a spell with AResponse timing
    if (!is_card_type(world, card, CARD_TYPE_SPELL))
      continue;
    if (!ecs_has(world, card, AResponse))
      continue;

    // Check if we have the ability registered
    const CardId *card_id = ecs_get(world, card, CardId);
    if (!card_id || !azk_has_ability(card_id->id))
      continue;

    // Check IKZ cost
    const IKZCost *ikz_cost = ecs_get(world, card, IKZCost);
    if (!ikz_cost)
      continue;

    if (ikz_cost->ikz_cost <= available_ikz) {
      // Found at least one playable response spell
      return true;
    }
  }

  return false;
}

/**
 * Expected Action: ACT_ATTACK, gaden_attacker_index, defender_index (opponent
 * tapped garden entity or leader) attacker_index and defender_index of 5 is the
 * leader
 */
static void handle_attack(ecs_world_t *world, GameState *gs,
                          ActionContext *ac) {
  if (ac->user_action.type != ACT_ATTACK) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  AttackIntent intent = {0};
  if (!azk_validate_attack_action(world, gs, player, &ac->user_action, true,
                                  &intent)) {
    ac->invalid_action = true;
    return;
  }

  int result = attack(world, &intent);

  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  // Check if defender can respond (has response spells and IKZ)
  uint8_t defender_index =
      (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;
  if (defender_can_respond(world, gs, defender_index)) {
    gs->phase = PHASE_RESPONSE_WINDOW;
    gs->active_player_index = defender_index;
    cli_render_logf(
        "[MainAction] Attack declared - defender has response options");
  } else {
    gs->phase = PHASE_COMBAT_RESOLVE;
    cli_render_logf("[MainAction] Attack declared - proceeding to combat");
  }
}

/**
 * Expected Action: ACT_ACTIVATE_ALLEY_ABILITY, ability_index, alley_index,
 * unused ability_index is 0 for now (single ability per card)
 */
static void handle_activate_alley_ability(ecs_world_t *world, GameState *gs,
                                          ActionContext *ac) {
  if (ac->user_action.type != ACT_ACTIVATE_ALLEY_ABILITY) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  ActivateAbilityIntent intent = {0};
  if (!azk_validate_activate_alley_ability_action(
          world, gs, player, &ac->user_action, true, &intent)) {
    ac->invalid_action = true;
    return;
  }

  // Trigger the main phase ability
  azk_trigger_main_ability(world, intent.card, player);

  cli_render_logf("[MainAction] Activated alley ability");
}

/**
 * Expected Action: ACT_ATTACH_WEAPON_FROM_HAND, hand_index, entity_index, use
 * ikz token entity_index of 5 is the leader
 */
static void handle_attach_weapon_from_hand(ecs_world_t *world, GameState *gs,
                                           ActionContext *ac) {
  if (ac->user_action.type != ACT_ATTACH_WEAPON_FROM_HAND) {
    exit(EXIT_FAILURE);
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

  cli_render_logf("[MainAction] Attach weapon");
}

void HandleMainAction(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  ActionContext *ac = ecs_field(it, ActionContext, 1);

  // Check if we're in an ability sub-phase
  AbilityPhase ability_phase = azk_get_ability_phase(world);

  if (ability_phase != ABILITY_PHASE_NONE) {
    // In ability phase - only handle ability-related actions
    switch (ac->user_action.type) {
    case ACT_CONFIRM_ABILITY:
      if (ability_phase == ABILITY_PHASE_CONFIRMATION) {
        if (!azk_process_ability_confirmation(world)) {
          ac->invalid_action = true;
        }
      } else {
        cli_render_logf(
            "[MainAction] ACT_CONFIRM_ABILITY not valid in ability phase %d",
            ability_phase);
        ac->invalid_action = true;
      }
      break;

    case ACT_NOOP:
      // In confirmation phase, NOOP means decline
      if (ability_phase == ABILITY_PHASE_CONFIRMATION) {
        if (!azk_process_ability_decline(world)) {
          ac->invalid_action = true;
        }
      } else if (ability_phase == ABILITY_PHASE_EFFECT_SELECTION) {
        // In effect selection phase, NOOP means skip (for "up to" effects with
        // min=0)
        if (!azk_process_effect_skip(world)) {
          ac->invalid_action = true;
        }
      } else {
        cli_render_logf("[MainAction] ACT_NOOP not valid in ability phase %d",
                        ability_phase);
        ac->invalid_action = true;
      }
      break;

    case ACT_SELECT_COST_TARGET:
      if (ability_phase == ABILITY_PHASE_COST_SELECTION) {
        if (!azk_process_cost_selection(world, ac->user_action.subaction_1)) {
          ac->invalid_action = true;
        }
      } else {
        cli_render_logf(
            "[MainAction] ACT_SELECT_COST_TARGET not valid in ability phase %d",
            ability_phase);
        ac->invalid_action = true;
      }
      break;

    case ACT_SELECT_EFFECT_TARGET:
      if (ability_phase == ABILITY_PHASE_EFFECT_SELECTION) {
        if (!azk_process_effect_selection(world, ac->user_action.subaction_1)) {
          ac->invalid_action = true;
        }
      } else {
        cli_render_logf("[MainAction] ACT_SELECT_EFFECT_TARGET not valid in "
                        "ability phase %d",
                        ability_phase);
        ac->invalid_action = true;
      }
      break;

    default:
      cli_render_logf(
          "[MainAction] Action type %d not valid during ability phase",
          ac->user_action.type);
      ac->invalid_action = true;
      break;
    }
    return;
  }

  // Normal main phase handling
  switch (ac->user_action.type) {
  case ACT_PLAY_ENTITY_TO_GARDEN:
    handle_play_entity_to_garden(world, gs, ac);
    break;
  case ACT_PLAY_ENTITY_TO_ALLEY:
    handle_play_entity_to_alley(world, gs, ac);
    break;
  case ACT_GATE_PORTAL:
    handle_gate_portal(world, gs, ac);
    break;
  case ACT_ATTACH_WEAPON_FROM_HAND:
    handle_attach_weapon_from_hand(world, gs, ac);
    break;
  case ACT_ATTACK:
    handle_attack(world, gs, ac);
    break;
  case ACT_ACTIVATE_ALLEY_ABILITY:
    handle_activate_alley_ability(world, gs, ac);
    break;
  case ACT_NOOP:
    if (!azk_validate_simple_action(world, gs,
                                    gs->players[gs->active_player_index],
                                    ac->user_action.type, true)) {
      ac->invalid_action = true;
      break;
    }
    cli_render_log("[MainAction] End turn");
    // TODO: Intelligent phase transition (if player action is required, goto
    // END_TURN_ACTION)
    // TODO: Look into the possibility of not having to add another phase
    //  and instead can figure out if action is needed in END_TURN
    //  programatically through observation validation
    gs->phase = PHASE_END_TURN;
    break;
  default:
    cli_render_logf("[MainAction] Unknown main action type: %d",
                    ac->user_action.type);
    ac->invalid_action = true;
    break;
  }
}

void init_main_phase_system(ecs_world_t *world) {
  ecs_system(world, {.entity = ecs_entity(world, {.name = "MainPhaseSystem",
                                                  .add = ecs_ids(TMain)}),
                     .query.terms = {{.id = ecs_id(GameState),
                                      .src.id = ecs_id(GameState),
                                      .inout = EcsIn},
                                     {.id = ecs_id(ActionContext),
                                      .src.id = ecs_id(ActionContext)}},
                     .callback = HandleMainAction});
}
