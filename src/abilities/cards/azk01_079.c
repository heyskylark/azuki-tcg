#include "abilities/cards/azk01_079.h"

#include "components/components.h"
#include "utils/damage_util.h"
#include "utils/deck_utils.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

#define AZK01_079_DRAW_COUNT 2
#define AZK01_079_DAMAGE 3

void azk01_079_apply_draw(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t controller =
      ecs_get_target(world, ctx->runtime.source_card, Rel_OwnedBy, 0);
  if (controller == 0) {
    return;
  }

  (void)draw_cards_with_deckout_check(world, controller, AZK01_079_DRAW_COUNT,
                                      NULL);
}

void azk01_079_apply_damage(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t chooser_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t leader =
      find_leader_card_in_zone(world, gs->zones[chooser_num].leader);
  if (leader == 0) {
    return;
  }

  (void)deal_effect_damage(world, leader, AZK01_079_DAMAGE);
}
