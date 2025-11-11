#ifndef AZUKI_VALIDATION_ACTION_SCHEMA_H
#define AZUKI_VALIDATION_ACTION_SCHEMA_H

#include <stddef.h>
#include <stdint.h>

#include "components.h"

#define AZK_PHASE_MASK(phase) (1u << (phase))

typedef enum {
  AZK_ACTION_PARAM_UNUSED = 0,
  AZK_ACTION_PARAM_HAND_INDEX,
  AZK_ACTION_PARAM_GARDEN_INDEX,
  AZK_ACTION_PARAM_GARDEN_OR_LEADER_INDEX,
  AZK_ACTION_PARAM_ALLEY_INDEX,
  AZK_ACTION_PARAM_BOOL,
  AZK_ACTION_PARAM_DEFENDER_INDEX
} AzkActionParamKind;

typedef struct {
  AzkActionParamKind kind;
  int min_value;
  int max_value;
} AzkActionParamSpec;

typedef struct {
  ActionType type;
  uint32_t phase_mask;
  AzkActionParamSpec params[3];
} AzkActionSpec;

const AzkActionSpec *azk_get_action_specs(size_t *out_count);

#endif
