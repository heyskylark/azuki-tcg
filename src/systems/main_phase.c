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
    // Increment entity played counter for this turn
    if (placement_type == ZONE_GARDEN) {
      gs->entities_played_garden_this_turn[gs->active_player_index]++;
    } else {
      gs->entities_played_alley_this_turn[gs->active_player_index]++;
    }

    // Queue on-play ability for processing on next loop iteration.
    // This allows deferred zone operations to flush first.
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

  // Check if attacking card has a "when attacking" ability before queueing
  const CardId *card_id = ecs_get(world, intent.attacking_card, CardId);
  if (card_id &&
      azk_has_ability_with_timing(card_id->id, ecs_id(AWhenAttacking))) {
    // Queue "when attacking" triggered ability for the attacking card
    azk_queue_triggered_effect(world, intent.attacking_card, player,
                               TIMING_TAG_WHEN_ATTACKING);
  }

  // Also check attached weapons for "when attacking" abilities
  ecs_iter_t weapon_it = ecs_children(world, intent.attacking_card);
  while (ecs_children_next(&weapon_it)) {
    for (int i = 0; i < weapon_it.count; i++) {
      ecs_entity_t weapon = weapon_it.entities[i];
      if (!ecs_has_id(world, weapon, TWeapon)) {
        continue;
      }

      const CardId *weapon_id = ecs_get(world, weapon, CardId);
      if (weapon_id &&
          azk_has_ability_with_timing(weapon_id->id, ecs_id(AWhenAttacking))) {
        azk_queue_triggered_effect(world, weapon, player,
                                   TIMING_TAG_WHEN_ATTACKING);
      }
    }
  }

  // Stay in MAIN phase to let effects resolve. The transition to response
  // window will happen after effects are processed (handled by phase gate).
  if (azk_has_queued_triggered_effects(world)) {
    cli_render_logf(
        "[MainAction] Attack declared - processing when attacking effects");
    return;
  }

  // No when-attacking ability or queue empty, proceed with normal flow
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

  // Pay IKZ cost (tap IKZ cards)
  for (int i = 0; i < intent.ikz_card_count; i++) {
    tap_card(world, intent.ikz_cards[i]);
  }

  // Trigger the main phase ability
  azk_trigger_main_ability(world, intent.card, player);

  cli_render_logf("[MainAction] Activated alley ability");
}

/**
 * Expected Action: ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY, slot_index, unused,
 * unused slot_index is 0-4 for garden, 5 for leader
 */
static void handle_activate_garden_or_leader_ability(ecs_world_t *world,
                                                      GameState *gs,
                                                      ActionContext *ac) {
  if (ac->user_action.type != ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY) {
    exit(EXIT_FAILURE);
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

  // Trigger the main phase ability
  azk_trigger_main_ability(world, intent.card, player);

  cli_render_logf("[MainAction] Activated garden/leader ability");
}

/**
 * Expected Action: ACT_PLAY_SPELL_FROM_HAND, hand_index, unused, use_ikz_token
 */
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

  cli_render_logf("[MainAction] Played spell from hand");

  // Trigger the spell's ability
  azk_trigger_spell_ability(world, intent.spell_card, player);
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

  // Trigger on-play abilities for weapons (like entities)
  azk_trigger_on_play_ability(world, intent.weapon_card, intent.player);

  // Trigger when-equipped abilities for weapons with AWhenEquipped timing
  azk_trigger_when_equipped_ability(world, intent.weapon_card, intent.player);

  cli_render_logf("[MainAction] Attach weapon");
}

void HandleMainAction(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  ActionContext *ac = ecs_field(it, ActionContext, 1);

  // Ability phase actions are now handled by AbilityResolutionPhaseSystem
  // This system should only run when ability_phase == NONE
  // (enforced by phase_gate.c pipeline selection)

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
  case ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY:
    handle_activate_garden_or_leader_ability(world, gs, ac);
    break;
  case ACT_PLAY_SPELL_FROM_HAND:
    handle_play_spell_from_hand(world, gs, ac);
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
