#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <flecs.h>

#include "abilities/ability_system.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/game_log_util.h"
#include "world.h"

static ecs_entity_t find_player_by_pid(ecs_world_t *world, uint8_t pid_value) {
  ecs_iter_t it = ecs_each_id(world, ecs_id(PlayerId));
  while (ecs_each_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t entity = it.entities[i];
      const PlayerId *pid = ecs_get(world, entity, PlayerId);
      if (pid && pid->pid == pid_value) {
        return entity;
      }
    }
  }

  return 0;
}

static ecs_entity_t spawn_test_card(ecs_world_t *world, CardDefId id) {
  ecs_entity_t card = ecs_new(world);
  ecs_set(world, card, CardId, {.id = id, .code = NULL});
  return card;
}

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end) {
  double seconds = (double)(end->tv_sec - start->tv_sec);
  double nanos = (double)(end->tv_nsec - start->tv_nsec) / 1e9;
  return seconds + nanos;
}

int main(void) {
  const int iterations = 200000;
  const int log_reset_interval = 32;

  ecs_world_t *world = azk_world_init(123);
  assert(world != NULL);

  ecs_entity_t player = find_player_by_pid(world, 0);
  assert(player != 0);

  ecs_entity_t card = spawn_test_card(world, CARD_DEF_STT01_002);
  assert(card != 0);

  azk_clear_game_logs(world);

  struct timespec start = {0};
  struct timespec end = {0};
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i = 0; i < iterations; i++) {
    if ((i % log_reset_interval) == 0) {
      azk_clear_game_logs(world);
    }

    bool queued = azk_queue_triggered_effect(world, card, player,
                                             TIMING_TAG_ON_GATE_PORTAL);
    assert(queued);
    azk_process_triggered_effect_queue(world);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  double elapsed = elapsed_seconds(&start, &end);
  double ops = (double)iterations * 2.0;
  double ops_per_sec = ops / elapsed;
  double ns_per_op = (elapsed / ops) * 1e9;

  printf("queue_profile: iterations=%d ops=%.0f elapsed=%.6f s ops/sec=%.0f ns/op=%.1f\n",
         iterations, ops, elapsed, ops_per_sec, ns_per_op);

  azk_world_fini(world);
  return 0;
}
