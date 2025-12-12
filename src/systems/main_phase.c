#include "systems/main_phase.h"
#include "components.h"
#include "abilities/ability.h"
#include "abilities/ability_runtime.h"
#include <string.h>
#include "utils/zone_util.h"
#include "utils/card_utils.h"
#include "utils/weapon_util.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "constants/game.h"
#include "utils/combat_util.h"
#include "validation/action_validation.h"

static void advance_combat_if_ready(ecs_world_t *world, GameState *gs) {
  if (gs->combat_state.attacking_card == 0 || gs->combat_state.defender_card == 0) {
    return;
  }

  ecs_entity_t attacker_owner = ecs_get_target(world, gs->combat_state.attacking_card, Rel_OwnedBy, 0);
  ecs_entity_t defender_owner = ecs_get_target(world, gs->combat_state.defender_card, Rel_OwnedBy, 0);
  ecs_assert(attacker_owner != 0 && defender_owner != 0, ECS_INVALID_PARAMETER, "Combat state cards missing owners");

  if (gs->phase == PHASE_MAIN) {
    gs->phase = PHASE_RESPONSE_WINDOW;
    gs->active_player_index = get_player_number(world, defender_owner);
    gs->response_window = 1;
    return;
  }

  if (gs->phase == PHASE_RESPONSE_WINDOW) {
    gs->phase = PHASE_COMBAT_RESOLVE;
    gs->active_player_index = get_player_number(world, attacker_owner);
    gs->response_window = 0;
  }
}

static int play_entity_to_garden_or_alley(
  ecs_world_t *world,
  GameState *gs,
  ActionContext *ac,
  AbilityContext *abctx,
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

  int summon_result = summon_card_into_zone_index(world, &intent);
  if (summon_result == 0) {
    azk_trigger_abilities_for_card(world, gs, abctx, intent.card, ABILITY_TIMING_ON_PLAY);
  }

  return summon_result;
}

/**
 * Expected Action: ACT_PLAY_ENTITY_TO_GARDEN, hand_index, garden_index, use ikz token
*/
static void handle_play_entity_to_garden(ecs_world_t *world, GameState *gs, ActionContext *ac, AbilityContext *abctx) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_GARDEN) {
    exit(EXIT_FAILURE);
  }

  int result = play_entity_to_garden_or_alley(world, gs, ac, abctx, ZONE_GARDEN);
  if (result < 0) {
    ac->invalid_action = true;
    return;
  }

  cli_render_logf("[MainAction] Played entity to garden");
}

/**
 * Expected Action: ACT_PLAY_ENTITY_TO_ALLEY, hand_index, alley_index, use ikz token
*/
static void handle_play_entity_to_alley(ecs_world_t *world, GameState *gs, ActionContext *ac, AbilityContext *abctx) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_ALLEY) {
    exit(EXIT_FAILURE);
  }

  int result = play_entity_to_garden_or_alley(world, gs, ac, abctx, ZONE_ALLEY);
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
static void handle_attack(ecs_world_t *world, GameState *gs, ActionContext *ac, AbilityContext *abctx) {
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

  azk_trigger_abilities_for_card(world, gs, abctx, intent.attacking_card, ABILITY_TIMING_ON_ATTACK);
  if (!abctx->has_pending) {
    ecs_iter_t it = ecs_children(world, intent.attacking_card);
    while (ecs_children_next(&it) && !abctx->has_pending) {
      for (int i = 0; i < it.count && !abctx->has_pending; i++) {
        ecs_entity_t child = it.entities[i];
        if (ecs_has_id(world, child, TWeapon)) {
          azk_trigger_abilities_for_card(world, gs, abctx, child, ABILITY_TIMING_ON_ATTACK);
        }
      }
    }
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
    if (!abctx->has_pending) {
      advance_combat_if_ready(world, gs);
    }
  // }

  cli_render_logf("[MainAction] Attack");
}

/**
 * Expected Action: ACT_ATTACH_WEAPON_FROM_HAND, hand_index, entity_index, use ikz token
 * entity_index of 5 is the leader
*/
static void handle_attach_weapon_from_hand(ecs_world_t *world, GameState *gs, ActionContext *ac, AbilityContext *abctx) {
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

  azk_trigger_abilities_for_card(world, gs, abctx, intent.weapon_card, ABILITY_TIMING_ON_EQUIP);

  cli_render_logf("[MainAction] Attach weapon");
}

static void handle_activate_ability(
  ecs_world_t *world,
  GameState *gs,
  ActionContext *ac,
  AbilityContext *abctx
) {
  if (ac->user_action.type != ACT_ACTIVATE_ABILITY) {
    exit(EXIT_FAILURE);
  }

  if (abctx->has_pending) {
    cli_render_log("[MainAction] Ability selection already pending");
    ac->invalid_action = true;
    return;
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  ActivateAbilityIntent intent = {0};
  if (!azk_validate_activate_ability_action(world, gs, player, &ac->user_action, true, &intent)) {
    ac->invalid_action = true;
    return;
  }

  azk_begin_or_resolve_ability(
    world,
    gs,
    abctx,
    intent.player,
    intent.source_card,
    intent.ability,
    intent.ikz_cards,
    intent.ikz_card_count,
    false
  );
  cli_render_logf("[MainAction] Activate ability");
}

static void handle_select_ability_target(
  ecs_world_t *world,
  GameState *gs,
  ActionContext *ac,
  AbilityContext *abctx
) {
  if (ac->user_action.type != ACT_SELECT_COST_TARGET && ac->user_action.type != ACT_SELECT_EFFECT_TARGET) {
    exit(EXIT_FAILURE);
  }

  ecs_entity_t player = gs->players[gs->active_player_index];
  AbilitySelectIntent intent = {0};
  if (!azk_validate_select_ability_target_action(world, gs, abctx, player, &ac->user_action, true, &intent)) {
    ac->invalid_action = true;
    return;
  }

  if (!azk_append_ability_target(abctx, intent.phase, intent.target)) {
    cli_render_log("[MainAction] Invalid ability target selection");
    ac->invalid_action = true;
    return;
  }

  azk_try_finish_ability(world, gs, abctx, false, false);

  if (!abctx->has_pending) {
    advance_combat_if_ready(world, gs);
  }
}

void HandleMainAction(ecs_iter_t *it) {
  ecs_world_t *world = ecs_get_world(it->world);
  GameState *gs = ecs_field(it, GameState, 0);
  ActionContext *ac = ecs_field(it, ActionContext, 1);
  AbilityContext *abctx = ecs_field(it, AbilityContext, 2);

  bool ability_pending_for_player =
    abctx->has_pending && (abctx->player == gs->players[gs->active_player_index]);
  if (ability_pending_for_player) {
    bool prompt_phase = abctx->phase == ABILITY_SELECTION_PROMPT;
    ActionType atype = ac->user_action.type;
    bool allowed =
      atype == ACT_NOOP ||
      atype == ACT_SELECT_COST_TARGET ||
      atype == ACT_SELECT_EFFECT_TARGET ||
      atype == ACT_ACCEPT_TRIGGERED_ABILITY;
    if (prompt_phase && atype != ACT_NOOP && atype != ACT_ACCEPT_TRIGGERED_ABILITY) {
      cli_render_log("[MainAction] Ability prompt blocks other actions");
      ac->invalid_action = true;
      return;
    }
    if (!allowed) {
      cli_render_log("[MainAction] Ability selection blocks other actions");
      ac->invalid_action = true;
      return;
    }
  }

  switch (ac->user_action.type) {
    case ACT_PLAY_ENTITY_TO_GARDEN:
      handle_play_entity_to_garden(world, gs, ac, abctx);
      break;
    case ACT_PLAY_ENTITY_TO_ALLEY:
      handle_play_entity_to_alley(world, gs, ac, abctx);
      break;
    case ACT_GATE_PORTAL:
      handle_gate_portal(world, gs, ac);
      break;
    case ACT_ATTACH_WEAPON_FROM_HAND:
      handle_attach_weapon_from_hand(world, gs, ac, abctx);
      break;
    case ACT_ATTACK:
      handle_attack(world, gs, ac, abctx);
      break;
    case ACT_ACTIVATE_ABILITY:
      handle_activate_ability(world, gs, ac, abctx);
      break;
    case ACT_SELECT_COST_TARGET:
    case ACT_SELECT_EFFECT_TARGET:
      handle_select_ability_target(world, gs, ac, abctx);
      break;
    case ACT_ACCEPT_TRIGGERED_ABILITY:
      if (!azk_validate_accept_triggered_ability_action(
            world,
            gs,
            abctx,
            gs->players[gs->active_player_index],
            &ac->user_action,
            true
          )) {
        ac->invalid_action = true;
        break;
      }
      azk_try_finish_ability(world, gs, abctx, false, false);
      if (!abctx->has_pending) {
        advance_combat_if_ready(world, gs);
      }
      break;
    case ACT_NOOP:
      if (!azk_validate_simple_action(world, gs, gs->players[gs->active_player_index], ac->user_action.type, true)) {
        ac->invalid_action = true;
        break;
      }
      if (abctx->has_pending) {
        bool decline_optional = abctx->optional &&
          (abctx->awaiting_consent || (abctx->cost_filled == 0 && abctx->effect_filled == 0));
        azk_try_finish_ability(world, gs, abctx, true, decline_optional);
        if (abctx->has_pending) {
          cli_render_log("[MainAction] Ability selection still pending");
          break;
        }
        advance_combat_if_ready(world, gs);
        cli_render_log("[MainAction] Ability window closed");
        break;
      }
      if (gs->phase == PHASE_RESPONSE_WINDOW && gs->combat_state.attacking_card != 0) {
        advance_combat_if_ready(world, gs);
        cli_render_log("[MainAction] Closing response window");
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
      { .id = ecs_id(ActionContext), .src.id = ecs_id(ActionContext) },
      { .id = ecs_id(AbilityContext), .src.id = ecs_id(AbilityContext) }
    },
    .callback = HandleMainAction
  });

  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "ResponsePhaseSystem",
      .add = ecs_ids(TResponseWindow)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState), .inout = EcsIn },
      { .id = ecs_id(ActionContext), .src.id = ecs_id(ActionContext) },
      { .id = ecs_id(AbilityContext), .src.id = ecs_id(AbilityContext) }
    },
    .callback = HandleMainAction
  });
}
