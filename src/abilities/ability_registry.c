#include "abilities/ability.h"

#include "components.h"
#include "utils/card_utils.h"
#include "utils/zone_util.h"
#include "utils/player_util.h"

static bool card_has_weapon_attached(ecs_world_t *world, ecs_entity_t card) {
  ecs_iter_t it = ecs_children(world, card);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t child = it.entities[i];
      if (ecs_has_id(world, child, TWeapon)) {
        return true;
      }
    }
  }
  return false;
}

static uint8_t count_targets_for_type(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  AbilityTargetType type
) {
  uint8_t player_number = get_player_number(world, player);
  uint8_t opponent_number = (player_number + 1) % MAX_PLAYERS_PER_MATCH;

  ecs_entity_t friendly_garden = gs->zones[player_number].garden;
  ecs_entity_t friendly_alley = gs->zones[player_number].alley;
  ecs_entity_t enemy_garden = gs->zones[opponent_number].garden;

  uint8_t count = 0;

  switch (type) {
    case ABILITY_TARGET_NONE:
      return 0;
    case ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY: {
      ecs_entities_t cards = ecs_get_ordered_children(world, friendly_garden);
      count = (uint8_t)cards.count;
      break;
    }
    case ABILITY_TARGET_FRIENDLY_ALLEY_ENTITY: {
      ecs_entities_t cards = ecs_get_ordered_children(world, friendly_alley);
      count = (uint8_t)cards.count;
      break;
    }
    case ABILITY_TARGET_FRIENDLY_ENTITY_WITH_WEAPON: {
      ecs_entities_t cards = ecs_get_ordered_children(world, friendly_garden);
      for (int32_t i = 0; i < cards.count; i++) {
        if (card_has_weapon_attached(world, cards.ids[i])) {
          count++;
        }
      }
      break;
    }
    case ABILITY_TARGET_FRIENDLY_LEADER: {
      count = 1;
      break;
    }
    case ABILITY_TARGET_ENEMY_GARDEN_ENTITY: {
      ecs_entities_t cards = ecs_get_ordered_children(world, enemy_garden);
      count = (uint8_t)cards.count;
      break;
    }
    case ABILITY_TARGET_ENEMY_LEADER: {
      count = 1; 
      break;
    }
    case ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY: {
      ecs_entities_t cards = ecs_get_ordered_children(world, enemy_garden);
      count = (uint8_t)(cards.count + 1);
      break;
    }
    case ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY: {
      ecs_entities_t garden_a = ecs_get_ordered_children(world, friendly_garden);
      ecs_entities_t garden_b = ecs_get_ordered_children(world, enemy_garden);
      count = (uint8_t)(garden_a.count + garden_b.count + 2);
      break;
    }
    default:
      break;
  }

  return count;
}

bool azk_ability_targets_available(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ecs_entity_t source_card,
  AbilityTargetRequirement req,
  AbilityTargetType required_type_override
) {
  if (req.max == 0 || req.type == ABILITY_TARGET_NONE) {
    return true;
  }

  uint8_t available = azk_ability_target_count(world, gs, player, source_card, req, required_type_override);
  if (available == 0) {
    return false;
  }

  return available >= req.min;
}

uint8_t azk_ability_target_count(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ecs_entity_t source_card,
  AbilityTargetRequirement req,
  AbilityTargetType required_type_override
) {
  if (req.max == 0 || req.type == ABILITY_TARGET_NONE) {
    return 0;
  }

  AbilityTargetType type = required_type_override != ABILITY_TARGET_NONE ? required_type_override : req.type;
  uint8_t available = count_targets_for_type(world, gs, player, type);

  // Exclude the source card itself if the selector would otherwise count it.
  if (type == ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY || type == ABILITY_TARGET_FRIENDLY_ENTITY_WITH_WEAPON) {
    ecs_entity_t owner = ecs_get_target(world, source_card, Rel_OwnedBy, 0);
    ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", source_card);
    uint8_t owner_number = get_player_number(world, owner);
    ecs_entity_t garden = gs->zones[owner_number].garden;
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
    for (int32_t i = 0; i < garden_cards.count; i++) {
      if (garden_cards.ids[i] == source_card) {
        if (available > 0) {
          available--;
        }
        break;
      }
    }
  }

  return available;
}

bool azk_ability_has_target_requirement(const AbilityTargetRequirement *req) {
  return req && req->max > 0 && req->type != ABILITY_TARGET_NONE;
}

static const AbilityDef ABILITIES[] = {
  {
    .card_def_id = CARD_DEF_STT01_001,
    .ability_index = 0,
    .name = "RaizanCharge",
    .kind = ABILITY_KIND_ACTIVATED,
    .timing = ABILITY_TIMING_MAIN,
    .source_zone = ABILITY_SOURCE_LEADER,
    .cost = { .ikz_cost = 1, .tap_self = false, .sacrifice_self = false },
    .cost_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .effect_targets = { .type = ABILITY_TARGET_FRIENDLY_ENTITY_WITH_WEAPON, .min = 1, .max = 1 },
    .once_per_turn = true,
    .optional = false
  },
  {
    .card_def_id = CARD_DEF_STT01_005,
    .ability_index = 0,
    .name = "AlpineProwlerSacrifice",
    .kind = ABILITY_KIND_ACTIVATED,
    .timing = ABILITY_TIMING_MAIN,
    .source_zone = ABILITY_SOURCE_ALLEY,
    .cost = { .ikz_cost = 0, .tap_self = false, .sacrifice_self = true },
    .cost_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .effect_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .once_per_turn = false,
    .optional = true
  },
  {
    .card_def_id = CARD_DEF_STT01_003,
    .ability_index = 0,
    .name = "CrateRatMill",
    .kind = ABILITY_KIND_TRIGGERED,
    .timing = ABILITY_TIMING_ON_PLAY,
    .source_zone = ABILITY_SOURCE_GARDEN,
    .cost = { .ikz_cost = 0, .tap_self = false, .sacrifice_self = false },
    .cost_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .effect_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .once_per_turn = false,
    .optional = false
  },
  {
    .card_def_id = CARD_DEF_STT01_006,
    .ability_index = 0,
    .name = "SilverCurrentStrike",
    .kind = ABILITY_KIND_TRIGGERED,
    .timing = ABILITY_TIMING_ON_ATTACK,
    .source_zone = ABILITY_SOURCE_GARDEN,
    .cost = { .ikz_cost = 0, .tap_self = false, .sacrifice_self = false },
    .cost_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .effect_targets = { .type = ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY, .min = 0, .max = 1 },
    .once_per_turn = true,
    .optional = false
  },
  {
    .card_def_id = CARD_DEF_STT01_012,
    .ability_index = 0,
    .name = "LightningShurikenMill",
    .kind = ABILITY_KIND_TRIGGERED,
    .timing = ABILITY_TIMING_ON_ATTACK,
    .source_zone = ABILITY_SOURCE_GARDEN,
    .cost = { .ikz_cost = 0, .tap_self = false, .sacrifice_self = false },
    .cost_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .effect_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .once_per_turn = false,
    .optional = false
  },
  {
    .card_def_id = CARD_DEF_STT01_013,
    .ability_index = 0,
    .name = "DaggerEquip",
    .kind = ABILITY_KIND_TRIGGERED,
    .timing = ABILITY_TIMING_ON_EQUIP,
    .source_zone = ABILITY_SOURCE_HAND,
    .cost = { .ikz_cost = 0, .tap_self = false, .sacrifice_self = false },
    .cost_targets = { .type = ABILITY_TARGET_FRIENDLY_LEADER, .min = 1, .max = 1 },
    .effect_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .once_per_turn = false,
    .optional = true
  },
  {
    .card_def_id = CARD_DEF_STT02_001,
    .ability_index = 0,
    .name = "ShaoReduceAttack",
    .kind = ABILITY_KIND_ACTIVATED,
    .timing = ABILITY_TIMING_RESPONSE,
    .source_zone = ABILITY_SOURCE_LEADER,
    .cost = { .ikz_cost = 1, .tap_self = false, .sacrifice_self = false },
    .cost_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .effect_targets = { .type = ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY, .min = 1, .max = 1 },
    .once_per_turn = true,
    .optional = false
  },
  {
    .card_def_id = CARD_DEF_STT02_011,
    .ability_index = 0,
    .name = "BubblemancerShield",
    .kind = ABILITY_KIND_ACTIVATED,
    .timing = ABILITY_TIMING_MAIN,
    .source_zone = ABILITY_SOURCE_GARDEN,
    .cost = { .ikz_cost = 0, .tap_self = false, .sacrifice_self = true },
    .cost_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .effect_targets = { .type = ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY, .min = 1, .max = 1 },
    .once_per_turn = false,
    .optional = true
  },
  {
    .card_def_id = CARD_DEF_STT01_017,
    .ability_index = 0,
    .name = "LightningOrb",
    .kind = ABILITY_KIND_ACTIVATED,
    .timing = ABILITY_TIMING_RESPONSE,
    .source_zone = ABILITY_SOURCE_HAND,
    .cost = { .ikz_cost = 1, .tap_self = false, .sacrifice_self = false },
    .cost_targets = { .type = ABILITY_TARGET_NONE, .min = 0, .max = 0 },
    .effect_targets = { .type = ABILITY_TARGET_ENEMY_GARDEN_ENTITY, .min = 1, .max = 2 },
    .once_per_turn = false,
    .optional = false
  }
};

const AbilityDef *azk_get_ability_defs(size_t *out_count) {
  if (out_count) {
    *out_count = sizeof(ABILITIES) / sizeof(ABILITIES[0]);
  }
  return ABILITIES;
}

const AbilityDef *azk_find_ability_def(CardDefId id, uint8_t ability_index) {
  size_t count = 0;
  const AbilityDef *defs = azk_get_ability_defs(&count);
  for (size_t i = 0; i < count; i++) {
    if (defs[i].card_def_id == id && defs[i].ability_index == ability_index) {
      return &defs[i];
    }
  }
  return NULL;
}

bool azk_card_has_activated_ability(CardDefId id) {
  size_t count = 0;
  const AbilityDef *defs = azk_get_ability_defs(&count);
  for (size_t i = 0; i < count; i++) {
    if (defs[i].card_def_id == id && defs[i].kind == ABILITY_KIND_ACTIVATED) {
      return true;
    }
  }
  return false;
}

size_t azk_collect_triggered_abilities(
  CardDefId id,
  AbilityTiming timing,
  const AbilityDef *out_defs[AZK_MAX_ABILITIES_PER_CARD]
) {
  size_t total = 0;
  size_t count = 0;
  const AbilityDef *defs = azk_get_ability_defs(&count);
  for (size_t i = 0; i < count && total < AZK_MAX_ABILITIES_PER_CARD; i++) {
    const AbilityDef *def = &defs[i];
    if (def->card_def_id != id) {
      continue;
    }
    if (def->kind != ABILITY_KIND_TRIGGERED) {
      continue;
    }
    if (def->timing != timing) {
      continue;
    }
    out_defs[total++] = def;
  }
  return total;
}
