#include "azuki/engine.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "abilities/ability_system.h"
#include "systems/phase_gate.h"
#include "utils/game_log_util.h"
#include "utils/phase_utils.h"
#include "utils/deck_utils.h"
#include "utils/status_util.h"
#include "utils/player_util.h"
#include "world.h"
#include "validation/action_enumerator.h"

typedef struct {
  bool initialized;
  bool enabled;
  uint64_t report_every;
  uint64_t calls;
  uint64_t total_ns;
  uint64_t deck_check_ns;
  uint64_t deck_process_ns;
  uint64_t passive_check_ns;
  uint64_t passive_process_ns;
  uint64_t triggered_check_ns;
  uint64_t triggered_process_ns;
  uint64_t phase_gate_ns;
  uint64_t ecs_progress_ns;
  uint64_t ret_deck_queue;
  uint64_t ret_passive_queue;
  uint64_t ret_triggered_queue;
  uint64_t ret_phase_transition;
  uint64_t ret_progressed;
} TickProfileState;

static TickProfileState g_tick_profile = {0};

typedef enum {
  TICK_PROFILE_RETURN_NONE = 0,
  TICK_PROFILE_RETURN_DECK = 1,
  TICK_PROFILE_RETURN_PASSIVE = 2,
  TICK_PROFILE_RETURN_TRIGGERED = 3,
  TICK_PROFILE_RETURN_PHASE_CHANGE = 4,
  TICK_PROFILE_RETURN_PROGRESS = 5
} TickProfileReturnPath;

static uint64_t tick_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static bool tick_env_enabled(const char *name) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return false;
  }
  if (value[0] == '0' && value[1] == '\0') {
    return false;
  }
  return true;
}

static void init_tick_profile_if_needed(void) {
  if (g_tick_profile.initialized) {
    return;
  }

  g_tick_profile.initialized = true;
  g_tick_profile.enabled = tick_env_enabled("AZK_TICK_PROFILE");
  g_tick_profile.report_every = 50000;

  const char *report_every = getenv("AZK_TICK_PROFILE_EVERY");
  if (report_every != NULL && report_every[0] != '\0') {
    char *end_ptr = NULL;
    unsigned long long parsed = strtoull(report_every, &end_ptr, 10);
    if (end_ptr != report_every && parsed > 0ull) {
      g_tick_profile.report_every = (uint64_t)parsed;
    }
  }
}

static void maybe_report_tick_profile(void) {
  if (!g_tick_profile.enabled || g_tick_profile.calls == 0 ||
      (g_tick_profile.calls % g_tick_profile.report_every) != 0) {
    return;
  }

  const double calls = (double)g_tick_profile.calls;
  fprintf(stderr,
          "[TickProfile] calls=%" PRIu64 " avg_us=%.2f "
          "deck_check_us=%.2f deck_proc_us=%.2f "
          "passive_check_us=%.2f passive_proc_us=%.2f "
          "trigger_check_us=%.2f trigger_proc_us=%.2f "
          "phase_gate_us=%.2f ecs_progress_us=%.2f "
          "ret(deck/passive/trigger/phase/progress)=%" PRIu64 "/%" PRIu64
          "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
          g_tick_profile.calls, g_tick_profile.total_ns / calls / 1000.0,
          g_tick_profile.deck_check_ns / calls / 1000.0,
          g_tick_profile.deck_process_ns / calls / 1000.0,
          g_tick_profile.passive_check_ns / calls / 1000.0,
          g_tick_profile.passive_process_ns / calls / 1000.0,
          g_tick_profile.triggered_check_ns / calls / 1000.0,
          g_tick_profile.triggered_process_ns / calls / 1000.0,
          g_tick_profile.phase_gate_ns / calls / 1000.0,
          g_tick_profile.ecs_progress_ns / calls / 1000.0,
          g_tick_profile.ret_deck_queue, g_tick_profile.ret_passive_queue,
          g_tick_profile.ret_triggered_queue, g_tick_profile.ret_phase_transition,
          g_tick_profile.ret_progressed);
}

static void record_tick_profile(
    uint64_t total_ns, uint64_t deck_check_ns, uint64_t deck_process_ns,
    uint64_t passive_check_ns, uint64_t passive_process_ns,
    uint64_t triggered_check_ns, uint64_t triggered_process_ns,
    uint64_t phase_gate_ns, uint64_t ecs_progress_ns,
    TickProfileReturnPath return_path) {
  if (!g_tick_profile.enabled) {
    return;
  }

  g_tick_profile.calls++;
  g_tick_profile.total_ns += total_ns;
  g_tick_profile.deck_check_ns += deck_check_ns;
  g_tick_profile.deck_process_ns += deck_process_ns;
  g_tick_profile.passive_check_ns += passive_check_ns;
  g_tick_profile.passive_process_ns += passive_process_ns;
  g_tick_profile.triggered_check_ns += triggered_check_ns;
  g_tick_profile.triggered_process_ns += triggered_process_ns;
  g_tick_profile.phase_gate_ns += phase_gate_ns;
  g_tick_profile.ecs_progress_ns += ecs_progress_ns;

  switch (return_path) {
  case TICK_PROFILE_RETURN_DECK:
    g_tick_profile.ret_deck_queue++;
    break;
  case TICK_PROFILE_RETURN_PASSIVE:
    g_tick_profile.ret_passive_queue++;
    break;
  case TICK_PROFILE_RETURN_TRIGGERED:
    g_tick_profile.ret_triggered_queue++;
    break;
  case TICK_PROFILE_RETURN_PHASE_CHANGE:
    g_tick_profile.ret_phase_transition++;
    break;
  case TICK_PROFILE_RETURN_PROGRESS:
    g_tick_profile.ret_progressed++;
    break;
  default:
    break;
  }

  maybe_report_tick_profile();
}

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

bool azk_engine_observe_training(AzkEngine *engine, int8_t player_index,
                                 TrainingObservationData *out_observation) {
  if (!engine || !out_observation) {
    return false;
  }

  *out_observation = create_training_observation_data(engine, player_index);
  return true;
}

bool azk_engine_observe_training_all(
    AzkEngine *engine,
    TrainingObservationData out_observations[MAX_PLAYERS_PER_MATCH]) {
  if (!engine || !out_observations) {
    return false;
  }

  create_training_observation_data_pair(engine, out_observations);
  return true;
}

bool azk_engine_requires_action(AzkEngine *engine) {
  if (!engine) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(engine, GameState);
  if (!gs || !phase_requires_user_action(engine, gs->phase)) {
    return false;
  }

  // Keep action masking and action consumption in lockstep by not exposing
  // "requires action" while auto-processed queues still have pending work.
  if (azk_has_pending_deck_reorders(engine) ||
      azk_has_pending_passive_buffs(engine) ||
      (azk_has_queued_triggered_effects(engine) &&
       !azk_is_in_ability_phase(engine))) {
    return false;
  }

  // Mirror PhaseGate's auto-transitions so masks are generated only for states
  // where an action will actually be consumed next tick.
  if (gs->phase == PHASE_MAIN && gs->combat_state.attacking_card != 0 &&
      !azk_has_queued_triggered_effects(engine) &&
      !azk_is_in_ability_phase(engine)) {
    return false;
  }

  if (gs->phase == PHASE_RESPONSE_WINDOW &&
      !azk_has_queued_triggered_effects(engine) &&
      !azk_is_in_ability_phase(engine) &&
      !defender_can_respond(engine, gs, gs->active_player_index)) {
    return false;
  }

  return true;
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

  init_tick_profile_if_needed();
  const bool profile_enabled = g_tick_profile.enabled;
  const uint64_t tick_start_ns = profile_enabled ? tick_now_ns() : 0;
  uint64_t deck_check_ns = 0;
  uint64_t deck_process_ns = 0;
  uint64_t passive_check_ns = 0;
  uint64_t passive_process_ns = 0;
  uint64_t triggered_check_ns = 0;
  uint64_t triggered_process_ns = 0;
  uint64_t phase_gate_ns = 0;
  uint64_t ecs_progress_ns = 0;
  TickProfileReturnPath return_path = TICK_PROFILE_RETURN_NONE;

  const GameState *gs = ecs_singleton_get(engine, GameState);
  if (!gs) {
    if (profile_enabled) {
      const uint64_t total_ns = tick_now_ns() - tick_start_ns;
      record_tick_profile(total_ns, deck_check_ns, deck_process_ns,
                          passive_check_ns, passive_process_ns,
                          triggered_check_ns, triggered_process_ns,
                          phase_gate_ns, ecs_progress_ns, return_path);
    }
    return;
  }

  uint64_t section_start_ns = profile_enabled ? tick_now_ns() : 0;
  const bool has_deck_reorders = azk_has_pending_deck_reorders(engine);
  if (profile_enabled) {
    deck_check_ns += tick_now_ns() - section_start_ns;
  }

  if (has_deck_reorders) {
    section_start_ns = profile_enabled ? tick_now_ns() : 0;
    azk_process_deck_reorder_queue(engine);
    if (profile_enabled) {
      deck_process_ns += tick_now_ns() - section_start_ns;
      return_path = TICK_PROFILE_RETURN_DECK;
      const uint64_t total_ns = tick_now_ns() - tick_start_ns;
      record_tick_profile(total_ns, deck_check_ns, deck_process_ns,
                          passive_check_ns, passive_process_ns,
                          triggered_check_ns, triggered_process_ns,
                          phase_gate_ns, ecs_progress_ns, return_path);
    }
    return;
  }

  // Process any pending passive buff updates (from observer callbacks)
  // These are queued because observer writes are deferred and not visible
  // to subsequent code in the same frame
  section_start_ns = profile_enabled ? tick_now_ns() : 0;
  const bool has_passive_buffs = azk_has_pending_passive_buffs(engine);
  if (profile_enabled) {
    passive_check_ns += tick_now_ns() - section_start_ns;
  }

  if (has_passive_buffs) {
    section_start_ns = profile_enabled ? tick_now_ns() : 0;
    azk_process_passive_buff_queue(engine);
    if (profile_enabled) {
      passive_process_ns += tick_now_ns() - section_start_ns;
      return_path = TICK_PROFILE_RETURN_PASSIVE;
      const uint64_t total_ns = tick_now_ns() - tick_start_ns;
      record_tick_profile(total_ns, deck_check_ns, deck_process_ns,
                          passive_check_ns, passive_process_ns,
                          triggered_check_ns, triggered_process_ns,
                          phase_gate_ns, ecs_progress_ns, return_path);
    }
    return;
  }

  // Check for queued triggered effects to auto-process
  // Only process when no ability is currently active to avoid overwriting
  // AbilityContext mid-ability (e.g., during BOTTOM_DECK phase)
  section_start_ns = profile_enabled ? tick_now_ns() : 0;
  const bool has_queued_effects = azk_has_queued_triggered_effects(engine);
  bool in_ability_phase = false;
  if (has_queued_effects) {
    in_ability_phase = azk_is_in_ability_phase(engine);
  }
  if (profile_enabled) {
    triggered_check_ns += tick_now_ns() - section_start_ns;
  }

  if (has_queued_effects && !in_ability_phase) {
    // Auto-process the queued effect (validates and sets up AbilityContext)
    section_start_ns = profile_enabled ? tick_now_ns() : 0;
    azk_process_triggered_effect_queue(engine);
    if (profile_enabled) {
      triggered_process_ns += tick_now_ns() - section_start_ns;
      return_path = TICK_PROFILE_RETURN_TRIGGERED;
      const uint64_t total_ns = tick_now_ns() - tick_start_ns;
      record_tick_profile(total_ns, deck_check_ns, deck_process_ns,
                          passive_check_ns, passive_process_ns,
                          triggered_check_ns, triggered_process_ns,
                          phase_gate_ns, ecs_progress_ns, return_path);
    }
    return;
  }

  const Phase phase_before = gs->phase;
  section_start_ns = profile_enabled ? tick_now_ns() : 0;
  run_phase_gate_system(engine);
  if (profile_enabled) {
    phase_gate_ns += tick_now_ns() - section_start_ns;
  }

  gs = ecs_singleton_get(engine, GameState);
  if (!gs) {
    if (profile_enabled) {
      const uint64_t total_ns = tick_now_ns() - tick_start_ns;
      record_tick_profile(total_ns, deck_check_ns, deck_process_ns,
                          passive_check_ns, passive_process_ns,
                          triggered_check_ns, triggered_process_ns,
                          phase_gate_ns, ecs_progress_ns, return_path);
    }
    return;
  }

  // Phase transitions are a separate auto-progression step.
  // Returning here prevents consuming stale ActionContext.user_action.
  if (gs->phase != phase_before) {
    if (profile_enabled) {
      return_path = TICK_PROFILE_RETURN_PHASE_CHANGE;
      const uint64_t total_ns = tick_now_ns() - tick_start_ns;
      record_tick_profile(total_ns, deck_check_ns, deck_process_ns,
                          passive_check_ns, passive_process_ns,
                          triggered_check_ns, triggered_process_ns,
                          phase_gate_ns, ecs_progress_ns, return_path);
    }
    return;
  }

  section_start_ns = profile_enabled ? tick_now_ns() : 0;
  ecs_progress(engine, 0);
  if (profile_enabled) {
    ecs_progress_ns += tick_now_ns() - section_start_ns;
    return_path = TICK_PROFILE_RETURN_PROGRESS;
    const uint64_t total_ns = tick_now_ns() - tick_start_ns;
    record_tick_profile(total_ns, deck_check_ns, deck_process_ns,
                        passive_check_ns, passive_process_ns,
                        triggered_check_ns, triggered_process_ns,
                        phase_gate_ns, ecs_progress_ns, return_path);
  }
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
