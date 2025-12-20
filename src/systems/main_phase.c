#include "systems/main_phase.h"
#include "components/components.h"
#include "utils/zone_util.h"
#include "utils/card_utils.h"
#include "utils/weapon_util.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "constants/game.h"
#include "utils/combat_util.h"
#include "validation/action_validation.h"

static int play_entity_to_garden_or_alley(
  ecs_world_t *world,
  GameState *gs,
  ActionContext *ac,
  ZonePlacementType placement_type
) {
  ecs_entity_t player = gs->players[gs->active_player_index];
  PlayEntityIntent intent = {0};
  if (!azk_validate_play_entity_action(
        world,
        gs,
        player,
        placement_type,
        &ac->user_action,
        true,
        &intent
      )) {
    return -1;
  }

  return summon_card_into_zone_index(world, &intent);
}

/**
 * Expected Action: ACT_PLAY_ENTITY_TO_GARDEN, hand_index, garden_index, use ikz token
*/
static void handle_play_entity_to_garden(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_GARDEN) {
    exit(EXIT_FAILURE);
  }

  int result = play_entity_to_garden_or_alley(world, gs, ac, ZONE_GARDEN);
  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  cli_render_logf("[MainAction] Played entity to garden");
}

/**
 * Expected Action: ACT_PLAY_ENTITY_TO_ALLEY, hand_index, alley_index, use ikz token
*/
static void handle_play_entity_to_alley(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_ALLEY) {
    exit(EXIT_FAILURE);
  }

  int result = play_entity_to_garden_or_alley(world, gs, ac, ZONE_ALLEY);
  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  cli_render_logf("[MainAction] Played entity to alley");
}

/**
 * Expected Action: ACT_GATE_PORTAL, alley_index, garden_index, 0
*/
static void handle_gate_portal(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  if (ac->user_action.type != ACT_GATE_PORTAL) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  GatePortalIntent intent = {0};
  if (!azk_validate_gate_portal_action(world, gs, player, &ac->user_action, true, &intent)) {
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
 * Expected Action: ACT_ATTACK, gaden_attacker_index, defender_index (opponent tapped garden entity or leader)
 * attacker_index and defender_index of 5 is the leader
*/
static void handle_attack(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  if (ac->user_action.type != ACT_ATTACK) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  AttackIntent intent = {0};
  if (!azk_validate_attack_action(world, gs, player, &ac->user_action, true, &intent)) {
    ac->invalid_action = true;
    return;
  }

  int result = attack(world, &intent);

  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  /*
  TODO: Only go to response window under these conditions:
  - has a response spell and enough IKZ
  - has an untapped entity in the garden with the defender tag
  - has any entities with effects that can be triggered on attack
  */
  // if (defender_can_respond(world, gs)) {
  //   gs->phase = PHASE_RESPONSE_WINDOW;
  //   gs->active_player_index = (gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH;
  // } else {
    gs->phase = PHASE_COMBAT_RESOLVE;
  // }

  cli_render_logf("[MainAction] Attack");
}

/**
 * Expected Action: ACT_ATTACH_WEAPON_FROM_HAND, hand_index, entity_index, use ikz token
 * entity_index of 5 is the leader
*/
static void handle_attach_weapon_from_hand(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  if (ac->user_action.type != ACT_ATTACH_WEAPON_FROM_HAND) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  AttachWeaponIntent intent = {0};
  if (!azk_validate_attach_weapon_action(world, gs, player, &ac->user_action, true, &intent)) {
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
    case ACT_NOOP:
      if (!azk_validate_simple_action(world, gs, gs->players[gs->active_player_index], ac->user_action.type, true)) {
        ac->invalid_action = true;
        break;
      }
      cli_render_log("[MainAction] End turn");
      // TODO: Intelligent phase transition (if player action is required, goto END_TURN_ACTION)
      // TODO: Look into the possibility of not having to add another phase
      //  and instead can figure out if action is needed in END_TURN programatically through observation validation
      gs->phase = PHASE_END_TURN;
      break;
    default:
      cli_render_logf("[MainAction] Unknown main action type: %d", ac->user_action.type);
      ac->invalid_action = true;
      break;
  }
}

void init_main_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "MainPhaseSystem",
      .add = ecs_ids(TMain)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState), .inout = EcsIn },
      { .id = ecs_id(ActionContext), .src.id = ecs_id(ActionContext) }
    },
    .callback = HandleMainAction
  });
}
