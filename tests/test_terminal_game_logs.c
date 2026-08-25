#include <assert.h>
#include <stdint.h>

#include <flecs.h>

#include "components/components.h"
#include "components/game_log.h"
#include "constants/game.h"
#include "systems/mulligan_phase.h"
#include "systems/start_phase.h"
#include "utils/combat_util.h"
#include "utils/damage_util.h"
#include "utils/game_log_util.h"
#include "utils/status_util.h"
#include "world.h"

static ecs_entity_t create_zone(ecs_world_t *world, ecs_entity_t owner,
                                ecs_entity_t zone_tag) {
  ecs_entity_t zone = ecs_new(world);
  ecs_add_id(world, zone, zone_tag);
  ecs_add_id(world, zone, EcsOrderedChildren);
  ecs_add_pair(world, zone, Rel_OwnedBy, owner);
  return zone;
}

static ecs_world_t *create_test_world(void) {
  ecs_world_t *world = ecs_init();
  azk_register_components(world);
  ecs_singleton_set(world, GameStateLogContext, {0});
  ecs_singleton_set(world, PassiveBuffQueue, {0});
  ecs_singleton_set(world, ActionContext, {0});
  ecs_singleton_set(world, GameState,
                    {.winner = -1, .phase = PHASE_MAIN});

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  for (uint8_t player_index = 0; player_index < MAX_PLAYERS_PER_MATCH;
       ++player_index) {
    ecs_entity_t player = ecs_new(world);
    ecs_set(world, player, PlayerNumber, {.player_number = player_index});
    gs->players[player_index] = player;
    gs->zones[player_index].deck = create_zone(world, player, ZDeck);
    gs->zones[player_index].hand = create_zone(world, player, ZHand);
    gs->zones[player_index].leader = create_zone(world, player, ZLeader);
  }
  ecs_singleton_modified(world, GameState);
  return world;
}

static ecs_entity_t create_card(ecs_world_t *world, ecs_entity_t owner,
                                ecs_entity_t zone, CardDefId card_def_id,
                                int8_t health, bool leader) {
  ecs_entity_t card = ecs_new(world);
  ecs_set(world, card, CardId, {.id = card_def_id});
  ecs_set(world, card, CurStats, {.cur_atk = 0, .cur_hp = health});
  ecs_add_pair(world, card, Rel_OwnedBy, owner);
  ecs_add_pair(world, card, EcsChildOf, zone);
  if (leader) {
    ecs_add(world, card, TLeader);
  }
  return card;
}

static const GameStateLog *get_logs(ecs_world_t *world, uint8_t *count) {
  const GameStateLog *logs = azk_get_game_logs(world, count);
  assert(logs != NULL);
  return logs;
}

static void assert_terminal_log(const GameStateLog *log, int8_t winner,
                                GameLogEndReason reason) {
  assert(log->type == GLOG_GAME_ENDED);
  assert(log->data.game_ended.winner == winner);
  assert(log->data.game_ended.reason == reason);
}

static void test_effect_lethal_logs_death_before_game_end(void) {
  ecs_world_t *world = create_test_world();
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_entity_t source = create_card(world, gs->players[1], gs->zones[1].hand,
                                    CARD_DEF_STT01_003, 1, false);
  ecs_entity_t leader = create_card(world, gs->players[0], gs->zones[0].leader,
                                    CARD_DEF_STT01_001, 1, true);
  azk_clear_game_logs(world);

  assert(deal_effect_damage_from_source(world, source, leader, 1));
  assert(gs->winner == 1);
  assert(gs->phase == PHASE_END_MATCH);

  uint8_t count = 0;
  const GameStateLog *logs = get_logs(world, &count);
  assert(count == 3);
  assert(logs[0].type == GLOG_CARD_STAT_CHANGE);
  assert(logs[0].data.stat_change.new_hp == 0);
  assert(logs[1].type == GLOG_ENTITY_DIED);
  assert(logs[1].data.entity_died.card.card_def_id == CARD_DEF_STT01_001);
  assert(logs[1].data.entity_died.cause == GLOG_DEATH_EFFECT);
  assert_terminal_log(&logs[2], 1, GLOG_END_LEADER_DEFEATED);

  ecs_fini(world);
}

static void test_passive_health_removal_lethal_logs_terminal_sequence(void) {
  ecs_world_t *world = create_test_world();
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_entity_t source = ecs_new(world);
  ecs_entity_t leader = create_card(world, gs->players[0], gs->zones[0].leader,
                                    CARD_DEF_STT01_001, 0, true);
  apply_health_modifier(world, leader, source, 2, false);
  azk_clear_game_logs(world);

  azk_queue_passive_buff_update(world, leader, source, 0, 2, true);
  azk_process_passive_buff_queue(world);
  assert(gs->winner == 1);
  assert(gs->phase == PHASE_END_MATCH);

  uint8_t count = 0;
  const GameStateLog *logs = get_logs(world, &count);
  assert(count == 3);
  assert(logs[0].type == GLOG_CARD_STAT_CHANGE);
  assert(logs[0].data.stat_change.new_hp == 0);
  assert(logs[1].type == GLOG_ENTITY_DIED);
  assert(logs[1].data.entity_died.card.card_def_id == CARD_DEF_STT01_001);
  assert_terminal_log(&logs[2], 1, GLOG_END_LEADER_DEFEATED);

  ecs_fini(world);
}

static void test_start_phase_empty_draw_logs_deck_out(void) {
  ecs_world_t *world = create_test_world();
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->active_player_index = 0;
  gs->phase = PHASE_START_OF_TURN;
  ecs_singleton_modified(world, GameState);
  azk_clear_game_logs(world);

  DrawCard(world, gs);
  assert(gs->winner == 1);
  assert(gs->phase == PHASE_END_MATCH);

  uint8_t count = 0;
  const GameStateLog *logs = get_logs(world, &count);
  assert(count == 1);
  assert_terminal_log(&logs[0], 1, GLOG_END_DECK_OUT);

  ecs_fini(world);
}

static void test_mulligan_draw_failure_logs_deck_out_once(void) {
  ecs_world_t *world = create_test_world();
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  gs->active_player_index = 0;
  gs->phase = PHASE_PREGAME_MULLIGAN;
  ecs_singleton_modified(world, GameState);

  for (int i = 0; i < INITIAL_DRAW_COUNT; ++i) {
    create_card(world, gs->players[0], gs->zones[0].hand,
                CARD_DEF_STT01_003, 1, false);
  }
  ecs_singleton_set(world, ActionContext,
                    {.user_action = {.player = gs->players[0],
                                     .type = ACT_MULLIGAN_SHUFFLE}});
  init_mulligan_phase_system(world);
  azk_clear_game_logs(world);

  ecs_entity_t system = ecs_lookup(world, "MulliganPhaseSystem");
  assert(system != 0);
  ecs_run(world, system, 0, NULL);
  assert(gs->winner == 1);
  assert(gs->phase == PHASE_END_MATCH);

  uint8_t count = 0;
  const GameStateLog *logs = get_logs(world, &count);
  int terminal_count = 0;
  for (uint8_t i = 0; i < count; ++i) {
    if (logs[i].type == GLOG_GAME_ENDED) {
      ++terminal_count;
      assert(i == count - 1);
      assert_terminal_log(&logs[i], 1, GLOG_END_DECK_OUT);
    }
  }
  assert(terminal_count == 1);

  ecs_fini(world);
}

static void test_simultaneous_leader_defeat_sets_draw_and_logs_once(void) {
  ecs_world_t *world = create_test_world();
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_entity_t attacker =
      create_card(world, gs->players[0], gs->zones[0].leader,
                  CARD_DEF_STT01_001, 1, true);
  ecs_entity_t defender =
      create_card(world, gs->players[1], gs->zones[1].leader,
                  CARD_DEF_STT01_001, 1, true);
  ecs_set(world, attacker, CurStats, {.cur_atk = 1, .cur_hp = 1});
  ecs_set(world, defender, CurStats, {.cur_atk = 1, .cur_hp = 1});
  gs->active_player_index = 0;
  gs->combat_state.attacking_card = attacker;
  gs->combat_state.defender_card = defender;
  ecs_singleton_modified(world, GameState);
  azk_clear_game_logs(world);

  resolve_combat(world);
  assert(gs->winner == 2);
  assert(gs->phase == PHASE_END_MATCH);

  uint8_t count = 0;
  const GameStateLog *logs = get_logs(world, &count);
  int terminal_count = 0;
  int death_count = 0;
  for (uint8_t i = 0; i < count; ++i) {
    if (logs[i].type == GLOG_ENTITY_DIED) {
      ++death_count;
    } else if (logs[i].type == GLOG_GAME_ENDED) {
      ++terminal_count;
      assert(i == count - 1);
      assert_terminal_log(&logs[i], 2, GLOG_END_LEADER_DEFEATED);
    }
  }
  assert(death_count == 2);
  assert(terminal_count == 1);

  ecs_fini(world);
}


int main(void) {
  test_effect_lethal_logs_death_before_game_end();
  test_simultaneous_leader_defeat_sets_draw_and_logs_once();
  test_passive_health_removal_lethal_logs_terminal_sequence();
  test_start_phase_empty_draw_logs_deck_out();
  test_mulligan_draw_failure_logs_deck_out_once();
  return 0;
}
