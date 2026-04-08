#include "abilities/cards/azk01_034.h"

#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

static bool is_friendly_garden_card(ecs_world_t *world, const GameState *gs,
                                    uint8_t owner_num, ecs_entity_t card) {
  if (card == 0) {
    return false;
  }

  return ecs_get_target(world, card, EcsChildOf, 0) == gs->zones[owner_num].garden;
}

bool azk01_034_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || ecs_has(world, card, Frozen)) {
    return false;
  }

  const uint8_t owner_num = get_player_number(world, owner);
  if (ecs_get_target(world, card, EcsChildOf, 0) != gs->zones[owner_num].alley) {
    return false;
  }

  const ecs_entity_t attacked_card = gs->combat_state.defender_card;
  if (attacked_card == 0 || attacked_card == card) {
    return false;
  }

  return is_friendly_garden_card(world, gs, owner_num, attacked_card);
}

void azk01_034_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  if (gs == NULL) {
    return;
  }

  const ecs_entity_t kira = ctx->runtime.source_card;
  const ecs_entity_t attacked_card = gs->combat_state.defender_card;
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);

  if (kira == 0 || attacked_card == 0 || attacked_card == kira ||
      ecs_get_target(world, kira, EcsChildOf, 0) != gs->zones[owner_num].alley ||
      ecs_get_target(world, attacked_card, EcsChildOf, 0) !=
          gs->zones[owner_num].garden) {
    return;
  }

  const ZoneIndex *kira_index = ecs_get(world, kira, ZoneIndex);
  const ZoneIndex *attacked_index = ecs_get(world, attacked_card, ZoneIndex);
  if (kira_index == NULL || attacked_index == NULL) {
    return;
  }

  const int8_t alley_index = (int8_t)kira_index->index;
  const int8_t garden_index = (int8_t)attacked_index->index;

  ecs_add_pair(world, kira, EcsChildOf, gs->zones[owner_num].garden);
  ecs_set(world, kira, ZoneIndex, {.index = (uint8_t)garden_index});
  const TapState *kira_tap = ecs_get(world, kira, TapState);
  if (kira_tap != NULL && azk_card_enters_garden_tapped(world, kira) &&
      !kira_tap->tapped) {
    ecs_set(world, kira, TapState,
            {.tapped = true, .cooldown = kira_tap->cooldown});
  }

  ecs_add_pair(world, attacked_card, EcsChildOf, gs->zones[owner_num].alley);
  ecs_set(world, attacked_card, ZoneIndex, {.index = (uint8_t)alley_index});

  azk_log_card_zone_moved(world, kira, GLOG_ZONE_ALLEY, alley_index,
                          GLOG_ZONE_GARDEN, garden_index);
  azk_log_card_zone_moved(world, attacked_card, GLOG_ZONE_GARDEN, garden_index,
                          GLOG_ZONE_ALLEY, alley_index);

  gs->combat_state.defender_card = kira;

  cli_render_logf("[AZK01-034] Swapped with attacked entity and became the new attack target");
}
