#include "abilities/cards/stt03_017.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/deck_utils.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

static ecs_entity_t get_owner_leader(ecs_world_t *world, ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return find_leader_card_in_zone(world, gs->zones[owner_num].leader);
}

bool stt03_017_validate_ramp(ecs_world_t *world, ecs_entity_t card,
                             ecs_entity_t owner) {
  (void)card;
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, owner);
  return ecs_get_ordered_children(world, gs->zones[owner_num].ikz_pile).count >
         0;
}

bool stt03_017_validate_heal_target(ecs_world_t *world, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target) {
  (void)card;
  if (target == 0 || target != get_owner_leader(world, owner)) {
    return false;
  }

  const BaseStats *base = ecs_get(world, target, BaseStats);
  const CurStats *current = ecs_get(world, target, CurStats);
  return base != NULL && current != NULL && current->cur_hp < base->health;
}

void stt03_017_begin_ramp_effect(ecs_world_t *world, AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t moved[1] = {0};
  if (!move_cards_to_zone(world, gs->zones[owner_num].ikz_pile,
                          gs->zones[owner_num].ikz_area, 1, moved) ||
      moved[0] == 0) {
    return;
  }

  tap_card(world, moved[0]);
  if (ctx->effect.max_allowed > 0) {
    ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
  }
}

void stt03_017_apply_heal(ecs_world_t *world, const AbilityContext *ctx) {
  if (ctx->effect.selected_count == 0) {
    return;
  }

  ecs_entity_t leader = ctx->effect.entities[0];
  const BaseStats *base = ecs_get(world, leader, BaseStats);
  CurStats *current = ecs_get_mut(world, leader, CurStats);
  if (base == NULL || current == NULL || current->cur_hp >= base->health) {
    return;
  }

  current->cur_hp++;
  ecs_modified(world, leader, CurStats);
  azk_log_card_stat_change(world, leader, 0, 1, current->cur_atk,
                           current->cur_hp);
}

void stt03_017_draw(ecs_world_t *world, const AbilityContext *ctx) {
  (void)draw_cards_with_deckout_check(world, ctx->runtime.owner, 1, NULL);
}
