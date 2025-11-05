#include "systems/main_action.h"
#include "components.h"
#include "utils/card_utils.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"

static int play_entity_to_garden_or_alley(
  ecs_world_t *world,
  GameState *gs,
  ActionContext *ac,
  ZonePlacementType placement_type
) {
  ecs_entity_t hand_card_idx = ac->user_action.subaction_1;
  ecs_entity_t zone_card_idx = ac->user_action.subaction_2;

  ecs_entity_t hand_zone = gs->zones[gs->active_player_index].hand;
  ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand_zone);
  ecs_entity_t hand_card = hand_cards.ids[hand_card_idx];

  if (hand_card == 0) {
    exit(EXIT_FAILURE);
  }
  
  if (!is_card_type(world, hand_card, CARD_TYPE_ENTITY)) {
    cli_render_logf("Hand card %d is not an entity", hand_card);
    return -1;
  }

  return insert_card_into_zone_index(world, hand_card, gs->players[gs->active_player_index], placement_type, zone_card_idx);
}

/**
 * Expected Action: ACT_PLAY_ENTITY_TO_GARDEN, hand_index, garden_index, 0
*/
static void handle_play_entity_to_garden(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_GARDEN) {
    exit(EXIT_FAILURE);
  }

  int result = play_entity_to_garden_or_alley(world, gs, ac, ZONE_GARDEN);
  if (result < 0) {
    ac->invalid_action = true;
  }
}

/**
 * Expected Action: ACT_PLAY_ENTITY_TO_ALLEY, hand_index, alley_index, 0
*/
static void handle_play_entity_to_alley(ecs_world_t *world, GameState *gs, ActionContext *ac) {
  if (ac->user_action.type != ACT_PLAY_ENTITY_TO_ALLEY) {
    exit(EXIT_FAILURE);
  }

  int result = play_entity_to_garden_or_alley(world, gs, ac, ZONE_ALLEY);
  if (result < 0) {
    ac->invalid_action = true;
  }
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
    default:
      cli_render_logf("[MainAction] Unknown main action type: %d", ac->user_action.type);
      ac->invalid_action = true;
      break;
  }
}

void init_main_action_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "HandleMainAction",
      .add = ecs_ids(TMain)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState), .inout = EcsIn },
      { .id = ecs_id(ActionContext), .src.id = ecs_id(ActionContext) }
    },
    .callback = HandleMainAction
  });
}