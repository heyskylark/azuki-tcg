#ifndef AZUKI_VALIDATION_ACTION_ENUMERATOR_H
#define AZUKI_VALIDATION_ACTION_ENUMERATOR_H

#include <stdbool.h>
#include <stdint.h>

#include "components/components.h"
#include "validation/action_intents.h"

#define AZK_ACTION_HEAD_COUNT 4
#define AZK_ACTION_HEAD0_SIZE (ACT_MULLIGAN_SHUFFLE + 1)
#define AZK_SUBACTION_HEAD_SIZE MAX_DECK_SIZE
#define AZK_MAX_LEGAL_ACTIONS 1024

typedef struct {
  uint8_t head0_mask[AZK_ACTION_TYPE_COUNT];
  uint16_t legal_action_count;
  UserAction legal_actions[AZK_MAX_LEGAL_ACTIONS];
} AzkActionMaskSet;

bool azk_build_action_mask_for_player(
  ecs_world_t *world,
  const GameState *gs,
  int8_t player_index,
  AzkActionMaskSet *out_mask
);

#endif
