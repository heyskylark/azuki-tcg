#include "validation/action_schema.h"

#include "constants/game.h"

#define UNUSED_PARAM { AZK_ACTION_PARAM_UNUSED, 0, 0 }

#define HAND_INDEX_PARAM { AZK_ACTION_PARAM_HAND_INDEX, 0, MAX_HAND_SIZE - 1 }
#define GARDEN_INDEX_PARAM { AZK_ACTION_PARAM_GARDEN_INDEX, 0, GARDEN_SIZE - 1 }
#define GARDEN_OR_LEADER_PARAM { AZK_ACTION_PARAM_GARDEN_OR_LEADER_INDEX, 0, GARDEN_SIZE }
#define ALLEY_INDEX_PARAM { AZK_ACTION_PARAM_ALLEY_INDEX, 0, ALLEY_SIZE - 1 }
#define BOOL_PARAM { AZK_ACTION_PARAM_BOOL, 0, 1 }

static const AzkActionSpec ACTION_SPECS[] = {
  {
    .type = ACT_NOOP,
    .phase_mask = AZK_PHASE_MASK(PHASE_PREGAME_MULLIGAN) | AZK_PHASE_MASK(PHASE_MAIN),
    .params = { UNUSED_PARAM, UNUSED_PARAM, UNUSED_PARAM }
  },
  {
    .type = ACT_PLAY_ENTITY_TO_GARDEN,
    .phase_mask = AZK_PHASE_MASK(PHASE_MAIN),
    .params = { HAND_INDEX_PARAM, GARDEN_INDEX_PARAM, BOOL_PARAM }
  },
  {
    .type = ACT_PLAY_ENTITY_TO_ALLEY,
    .phase_mask = AZK_PHASE_MASK(PHASE_MAIN),
    .params = { HAND_INDEX_PARAM, GARDEN_INDEX_PARAM, BOOL_PARAM }
  },
  {
    .type = ACT_ATTACH_WEAPON_FROM_HAND,
    .phase_mask = AZK_PHASE_MASK(PHASE_MAIN),
    .params = { HAND_INDEX_PARAM, GARDEN_OR_LEADER_PARAM, BOOL_PARAM }
  },
  {
    .type = ACT_GATE_PORTAL,
    .phase_mask = AZK_PHASE_MASK(PHASE_MAIN),
    .params = { ALLEY_INDEX_PARAM, GARDEN_INDEX_PARAM, UNUSED_PARAM }
  },
  {
    .type = ACT_ATTACK,
    .phase_mask = AZK_PHASE_MASK(PHASE_MAIN),
    .params = { GARDEN_OR_LEADER_PARAM, GARDEN_OR_LEADER_PARAM, UNUSED_PARAM }
  },
  {
    .type = ACT_MULLIGAN_SHUFFLE,
    .phase_mask = AZK_PHASE_MASK(PHASE_PREGAME_MULLIGAN),
    .params = { UNUSED_PARAM, UNUSED_PARAM, UNUSED_PARAM }
  },
  {
    .type = ACT_PLAY_SPELL_FROM_HAND,
    .phase_mask = AZK_PHASE_MASK(PHASE_RESPONSE_WINDOW),
    .params = { HAND_INDEX_PARAM, UNUSED_PARAM, BOOL_PARAM }
  },
  {
    .type = ACT_ACTIVATE_ALLEY_ABILITY,
    .phase_mask = AZK_PHASE_MASK(PHASE_MAIN),
    .params = { UNUSED_PARAM, ALLEY_INDEX_PARAM, UNUSED_PARAM }  // subaction_1=ability_index (0 for now), subaction_2=alley_slot
  },
  {
    .type = ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
    .phase_mask = AZK_PHASE_MASK(PHASE_MAIN) | AZK_PHASE_MASK(PHASE_RESPONSE_WINDOW),
    .params = { GARDEN_OR_LEADER_PARAM, UNUSED_PARAM, UNUSED_PARAM }  // subaction_1=slot_index (0-4=garden, 5=leader)
  },
  {
    .type = ACT_DECLARE_DEFENDER,
    .phase_mask = AZK_PHASE_MASK(PHASE_RESPONSE_WINDOW),
    .params = { GARDEN_INDEX_PARAM, UNUSED_PARAM, UNUSED_PARAM }
  }
};

const AzkActionSpec *azk_get_action_specs(size_t *out_count) {
  if (out_count) {
    *out_count = sizeof(ACTION_SPECS) / sizeof(ACTION_SPECS[0]);
  }
  return ACTION_SPECS;
}
