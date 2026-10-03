#include "abilities/cards/azk01_015.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/damage_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

static ecs_entity_t get_owner_leader(ecs_world_t *world, ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  return find_leader_card_in_zone(world, gs->zones[player_num].leader);
}

static ecs_entity_t find_first_tapped_ikz(ecs_world_t *world,
                                          ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t player = gs->players[player_num];
  const IKZToken *ikz_token = ecs_get(world, player, IKZToken);
  if (ikz_token && ikz_token->ikz_token != 0 &&
      is_card_tapped(world, ikz_token->ikz_token)) {
    return ikz_token->ikz_token;
  }

  ecs_entities_t cards =
      ecs_get_ordered_children(world, gs->zones[player_num].ikz_area);
  for (int32_t i = 0; i < cards.count; i++) {
    if (is_card_tapped(world, cards.ids[i])) {
      return cards.ids[i];
    }
  }

  return 0;
}

static CardElement get_mo_leader_element(ecs_world_t *world, ecs_entity_t owner) {
  return get_card_element(world, get_owner_leader(world, owner));
}

bool azk01_015_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;

  ecs_entity_t leader = get_owner_leader(world, owner);
  if (leader == 0) {
    return false;
  }

  switch (get_card_element(world, leader)) {
  case CARD_ELEMENT_WATER:
    return find_first_tapped_ikz(world, owner) != 0;
  case CARD_ELEMENT_EARTH: {
    const BaseStats *base = ecs_get(world, leader, BaseStats);
    const CurStats *cur = ecs_get(world, leader, CurStats);
    return base != NULL && cur != NULL && cur->cur_hp < base->health;
  }
  case CARD_ELEMENT_LIGHTNING:
    return !ecs_has(world, card, Charge);
  case CARD_ELEMENT_FIRE:
    return true;
  default:
    return false;
  }
}

bool azk01_015_validate_effect_target(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner, ecs_entity_t target) {
  (void)card;

  if (target == 0 || get_mo_leader_element(world, owner) != CARD_ELEMENT_FIRE) {
    return false;
  }

  return ecs_has(world, target, TLeader);
}

void azk01_015_on_cost_paid(ecs_world_t *world, AbilityContext *ctx) {
  if (get_mo_leader_element(world, ctx->runtime.owner) == CARD_ELEMENT_FIRE) {
    ctx->effect.min_required = 1;
    ctx->effect.max_allowed = 1;
    ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
    return;
  }

  azk01_015_apply_effects(world, ctx);
}

void azk01_015_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  ecs_entity_t leader = get_owner_leader(world, ctx->runtime.owner);
  if (leader == 0) {
    return;
  }

  switch (get_card_element(world, leader)) {
  case CARD_ELEMENT_WATER: {
    ecs_entity_t ikz = find_first_tapped_ikz(world, ctx->runtime.owner);
    if (ikz != 0) {
      ecs_set(world, ikz, TapState, {.tapped = false, .cooldown = false});
      azk_log_card_tap_state_changed(world, ikz, GLOG_TAP_UNTAPPED);
      azk_mark_generated_ikz_credit(world, ikz);
    }
    return;
  }
  case CARD_ELEMENT_EARTH: {
    const BaseStats *base = ecs_get(world, leader, BaseStats);
    CurStats *cur = ecs_get_mut(world, leader, CurStats);
    if (!base || !cur) {
      return;
    }
    int16_t missing_hp = (int16_t)base->health - (int16_t)cur->cur_hp;
    int8_t heal_amount =
        (int8_t)(missing_hp <= 0 ? 0 : (missing_hp < 2 ? missing_hp : 2));
    if (heal_amount > 0) {
      cur->cur_hp += heal_amount;
      ecs_modified(world, leader, CurStats);
      azk_log_card_stat_change(world, leader, 0, heal_amount, cur->cur_atk,
                               cur->cur_hp);
    }
    return;
  }
  case CARD_ELEMENT_LIGHTNING:
    apply_charge_grant(world, ctx->runtime.source_card,
                       TAG_GRANT_TICK_END_OF_TURN, 1);
    return;
  case CARD_ELEMENT_FIRE: {
    ecs_entity_t target = ctx->effect.entities[0];
    if (target != 0) {
      deal_effect_damage(world, target, 2);
    }
    return;
  }
  default:
    return;
  }
}
