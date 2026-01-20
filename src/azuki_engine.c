#include "azuki/engine.h"
#include <math.h>

#include "abilities/ability_system.h"
#include "systems/phase_gate.h"
#include "utils/game_log_util.h"
#include "utils/phase_utils.h"
#include "utils/status_util.h"
#include "world.h"
#include "validation/action_enumerator.h"

AzkEngine *azk_engine_create(uint32_t seed) {
  return azk_world_init(seed);
}

AzkEngine *azk_engine_create_with_decks(
  uint32_t seed,
  const CardInfo *player0_deck,
  size_t player0_deck_count,
  const CardInfo *player1_deck,
  size_t player1_deck_count
) {
  return azk_world_init_with_decks(seed, player0_deck, player0_deck_count,
                                   player1_deck, player1_deck_count);
}

void azk_engine_destroy(AzkEngine *engine) {
  if (!engine) {
    return;
  }

  azk_world_fini(engine);
}

const GameState *azk_engine_game_state(const AzkEngine *engine) {
  if (!engine) {
    return NULL;
  }

  return ecs_singleton_get((const ecs_world_t *)engine, GameState);
}

bool azk_engine_observe(AzkEngine *engine, int8_t player_index, ObservationData *out_observation) {
  if (!engine || !out_observation) {
    return false;
  }

  *out_observation = create_observation_data(engine, player_index);
  return true;
}

bool azk_engine_requires_action(AzkEngine *engine) {
  if (!engine) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(engine, GameState);
  return gs && phase_requires_user_action(engine, gs->phase);
}

bool azk_engine_is_game_over(AzkEngine *engine) {
  if (!engine) {
    return true;
  }

  return is_game_over(engine);
}

bool azk_engine_parse_action_values(
  AzkEngine *engine,
  const int values[AZK_USER_ACTION_VALUE_COUNT],
  UserAction *out_action
) {
  if (!engine) {
    return false;
  }

  return azk_parse_user_action_values(engine, values, out_action);
}

bool azk_engine_submit_action(AzkEngine *engine, const UserAction *action) {
  if (!engine || !action) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(engine, GameState);
  if (!gs || !verify_user_action_player(gs, action)) {
    return false;
  }

  azk_store_user_action(engine, action);
  return true;
}

void azk_engine_tick(AzkEngine *engine) {
  if (!engine) {
    return;
  }

  // Process any pending passive buff updates (from observer callbacks)
  // These are queued because observer writes are deferred and not visible
  // to subsequent code in the same frame
  if (azk_has_pending_passive_buffs(engine)) {
    azk_process_passive_buff_queue(engine);
  }

  // Check for queued triggered effects to auto-process
  // Only process when no ability is currently active to avoid overwriting
  // AbilityContext mid-ability (e.g., during BOTTOM_DECK phase)
  if (azk_has_queued_triggered_effects(engine) &&
      !azk_is_in_ability_phase(engine)) {
    // Auto-process the queued effect (validates and sets up AbilityContext)
    azk_process_triggered_effect_queue(engine);
    return;
  }

  run_phase_gate_system(engine);
  ecs_progress(engine, 0);
}

bool azk_engine_was_prev_action_invalid(AzkEngine *engine) {
  if (!engine) {
    return false;
  }

  const ActionContext *ac = ecs_singleton_get(engine, ActionContext);
  ecs_assert(ac != NULL, ECS_INVALID_PARAMETER, "Action context is NULL before checking if previous action was invalid");
  return ac->invalid_action;
}

static float clamp01(float value) {
  if (value < 0.0f) {
    return 0.0f;
  }
  if (value > 1.0f) {
    return 1.0f;
  }
  return value;
}

static float compute_leader_health_ratio(ecs_world_t *world, ecs_entity_t leader_zone) {
  ecs_entities_t leader_cards = ecs_get_ordered_children(world, leader_zone);
  ecs_assert(leader_cards.count > 0, ECS_INVALID_PARAMETER, "Leader zone has no cards");

  ecs_entity_t leader_card = leader_cards.ids[0];
  const CurStats *cur_stats = ecs_get(world, leader_card, CurStats);
  ecs_assert(cur_stats != NULL, ECS_INVALID_PARAMETER, "Leader card has no CurStats component");
  const BaseStats *base_stats = ecs_get(world, leader_card, BaseStats);
  ecs_assert(base_stats != NULL, ECS_INVALID_PARAMETER, "Leader card has no BaseStats component");
  if (base_stats->health <= 0) {
    return 0.0f;
  }

  float ratio = (float)cur_stats->cur_hp / (float)base_stats->health;
  return clamp01(ratio);
}

static void compute_garden_features(
  ecs_world_t *world,
  ecs_entity_t garden_zone,
  float *out_attack_sum,
  float *out_untapped_count
) {
  float attack_sum = 0.0f;
  float untapped = 0.0f;
  ecs_entities_t cards = ecs_get_ordered_children(world, garden_zone);
  for (int32_t i = 0; i < cards.count; ++i) {
    ecs_entity_t card = cards.ids[i];
    const CurStats *cur_stats = ecs_get(world, card, CurStats);
    ecs_assert(cur_stats != NULL, ECS_INVALID_PARAMETER, "Card %d has no CurStats component", card);
    attack_sum += (float)cur_stats->cur_atk;

    const TapState *tap_state = ecs_get(world, card, TapState);
    ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER, "Card %d has no TapState component", card);

    if (!tap_state->tapped && !tap_state->cooldown) {
      untapped += 1.0f;
    }
  }
  *out_attack_sum = attack_sum;
  *out_untapped_count = untapped;
}

static float count_untapped_cards_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  float untapped = 0.0f;
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; ++i) {
    ecs_entity_t card = cards.ids[i];
    const TapState *tap_state = ecs_get(world, card, TapState);
    ecs_assert(tap_state != NULL, ECS_INVALID_PARAMETER, "Card %s has no TapState component", card);

    if (!tap_state->tapped && !tap_state->cooldown) {
      untapped += 1.0f;
    }
  }
  return untapped;
}

bool azk_engine_reward_snapshot(AzkEngine *engine, AzkRewardSnapshot *out_snapshot) {
  if (!engine || !out_snapshot) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(engine, GameState);
  if (!gs) {
    return false;
  }

  for (int player_index = 0; player_index < MAX_PLAYERS_PER_MATCH; ++player_index) {
    const PlayerZones *zones = &gs->zones[player_index];
    out_snapshot->leader_health_ratio[player_index] =
      compute_leader_health_ratio(engine, zones->leader);
    float garden_attack = 0.0f;
    float untapped_garden = 0.0f;
    compute_garden_features(engine, zones->garden, &garden_attack, &untapped_garden);
    out_snapshot->garden_attack_sum[player_index] = garden_attack;
    out_snapshot->untapped_garden_count[player_index] = untapped_garden;
    out_snapshot->untapped_ikz_count[player_index] =
      count_untapped_cards_in_zone(engine, zones->ikz_area);
  }

  return true;
}

AbilityPhase azk_engine_get_ability_phase(const AzkEngine *engine) {
  if (!engine) {
    return ABILITY_PHASE_NONE;
  }

  return azk_get_ability_phase((ecs_world_t *)engine);
}
