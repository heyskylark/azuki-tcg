#include "abilities/cards/azk01_030.h"

#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

static uint8_t untap_tapped_ikz_sources(ecs_world_t *world, ecs_entity_t owner,
                                        uint8_t max_untap) {
  if (max_untap == 0) {
    return 0;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t player = gs->players[player_num];
  uint8_t untapped = 0;

  const IKZToken *ikz_token = ecs_get(world, player, IKZToken);
  if (ikz_token && ikz_token->ikz_token != 0 && untapped < max_untap &&
      is_card_tapped(world, ikz_token->ikz_token)) {
    const TapState *ts = ecs_get(world, ikz_token->ikz_token, TapState);
    if (ts) {
      ecs_set(world, ikz_token->ikz_token, TapState,
              {.tapped = false, .cooldown = ts->cooldown});
      azk_log_card_tap_state_changed(world, ikz_token->ikz_token,
                                     GLOG_TAP_UNTAPPED);
      ++untapped;
    }
  }

  if (untapped < max_untap) {
    untapped += untap_n_ikz_cards(world, gs->zones[player_num].ikz_area,
                                  max_untap - untapped);
  }

  return untapped;
}

static uint8_t count_tapped_ikz_sources(ecs_world_t *world, ecs_entity_t owner) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, owner);
  ecs_entity_t player = gs->players[player_num];
  uint8_t count = 0;

  const IKZToken *ikz_token = ecs_get(world, player, IKZToken);
  if (ikz_token && ikz_token->ikz_token != 0 &&
      is_card_tapped(world, ikz_token->ikz_token)) {
    ++count;
  }

  ecs_entities_t ikz_cards =
      ecs_get_ordered_children(world, gs->zones[player_num].ikz_area);
  for (int32_t i = 0; i < ikz_cards.count; ++i) {
    if (is_card_tapped(world, ikz_cards.ids[i])) {
      ++count;
    }
  }

  return count;
}

bool azk01_030_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)card;
  return count_tapped_ikz_sources(world, owner) > 0;
}

void azk01_030_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  untap_tapped_ikz_sources(world, ctx->runtime.owner, 2);
}
