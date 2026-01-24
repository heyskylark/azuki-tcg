#include "abilities/cards/stt01_015.h"

#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"

// STT01-015 "Tenraku": When Equipped; If you have 15 or more cards in your
// discard pile, this card gives an additional +1 attack.

#define STT01_015_DISCARD_THRESHOLD 15
#define STT01_015_BONUS_ATTACK 1

bool stt01_015_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;
  // Always valid - condition is checked in apply_effects
  return true;
}

void stt01_015_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  // Get the attached target (weapon is child of target entity)
  ecs_entity_t target = ecs_get_target(world, ctx->source_card, EcsChildOf, 0);
  if (target == 0) {
    cli_render_logf("[STT01-015] Weapon is not attached to any entity");
    return;
  }

  // Get player's discard pile count
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t discard = gs->zones[player_num].discard;

  ecs_entities_t discard_cards = ecs_get_ordered_children(world, discard);
  int32_t discard_count = discard_cards.count;

  cli_render_logf("[STT01-015] Checking discard pile: %d cards (threshold: %d)",
                  discard_count, STT01_015_DISCARD_THRESHOLD);

  if (discard_count < STT01_015_DISCARD_THRESHOLD) {
    cli_render_logf("[STT01-015] Condition not met, no bonus attack applied");
    return;
  }

  // Condition met - apply +1 attack to weapon's CurStats
  CurStats *weapon_stats = ecs_get_mut(world, ctx->source_card, CurStats);
  ecs_assert(weapon_stats != NULL, ECS_INVALID_PARAMETER,
             "CurStats not found for weapon");

  weapon_stats->cur_atk += STT01_015_BONUS_ATTACK;
  ecs_modified(world, ctx->source_card, CurStats);
  azk_log_card_stat_change(world, ctx->source_card, STT01_015_BONUS_ATTACK, 0,
                           weapon_stats->cur_atk, weapon_stats->cur_hp);

  // Update target entity's CurStats.cur_atk (+1)
  CurStats *target_stats = ecs_get_mut(world, target, CurStats);
  ecs_assert(target_stats != NULL, ECS_INVALID_PARAMETER,
             "CurStats not found for target entity");

  target_stats->cur_atk += STT01_015_BONUS_ATTACK;
  ecs_modified(world, target, CurStats);
  azk_log_card_stat_change(world, target, STT01_015_BONUS_ATTACK, 0,
                           target_stats->cur_atk, target_stats->cur_hp);

  cli_render_logf(
      "[STT01-015] Discard pile has %d cards, weapon gained +%d attack "
      "(total: %d)",
      discard_count, STT01_015_BONUS_ATTACK, weapon_stats->cur_atk);
}
