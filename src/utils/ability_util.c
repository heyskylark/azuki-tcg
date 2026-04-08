#include "utils/ability_util.h"

#include "components/abilities.h"
#include "generated/card_defs.h"

static bool is_player_invokable_ability(ecs_world_t *world, ecs_entity_t card,
                                        const AbilityDef *def) {
  if (world == NULL || card == 0 || def == NULL) {
    return false;
  }

  if (!azk_ability_def_has_timing(def, ecs_id(AMain)) &&
      !azk_ability_def_has_timing(def, ecs_id(AResponse))) {
    return false;
  }

  const Type *type = ecs_get(world, card, Type);
  if (type == NULL) {
    const CardId *card_id = ecs_get(world, card, CardId);
    if (card_id != NULL) {
      ecs_entity_t prefab = azk_prefab_from_id(card_id->id);
      if (prefab != 0) {
        type = ecs_get(world, prefab, Type);
      }
    }
  }

  if (type == NULL) {
    return false;
  }

  return type->value == CARD_TYPE_SPELL || type->value == CARD_TYPE_ENTITY ||
         type->value == CARD_TYPE_WEAPON || type->value == CARD_TYPE_LEADER ||
         type->value == CARD_TYPE_GATE;
}

bool azk_ability_def_has_timing(const AbilityDef *def, ecs_id_t timing_tag) {
  if (def == NULL || timing_tag == 0) {
    return false;
  }

  return def->timing_tag == timing_tag || def->secondary_timing_tag == timing_tag;
}

ecs_entity_t azk_get_ability_source_card(ecs_world_t *world,
                                         ecs_entity_t ability_entity) {
  if (world == NULL || ability_entity == 0) {
    return 0;
  }

  ecs_entity_t source_card =
      ecs_get_target(world, ability_entity, Rel_AbilityOf, 0);
  if (source_card != 0) {
    return source_card;
  }

  return ecs_get(world, ability_entity, CardId) != NULL ? ability_entity : 0;
}

const AbilityDef *azk_get_ability_def_for_entity(ecs_world_t *world,
                                                 ecs_entity_t ability_entity) {
  if (world == NULL || ability_entity == 0) {
    return NULL;
  }

  const AbilityInstance *instance = ecs_get(world, ability_entity, AbilityInstance);
  if (instance == NULL) {
    const CardId *card_id = ecs_get(world, ability_entity, CardId);
    return card_id != NULL ? azk_get_ability_def(card_id->id) : NULL;
  }

  return azk_get_ability_def_at(instance->card_def_id, instance->registry_order);
}

static void insert_sorted_ability(ecs_world_t *world, ecs_entity_t ability_entity,
                                  ecs_entity_t *out_abilities,
                                  uint8_t *io_count, uint8_t out_cap) {
  if (out_abilities == NULL || io_count == NULL || *io_count >= out_cap) {
    return;
  }

  const AbilityInstance *instance = ecs_get(world, ability_entity, AbilityInstance);
  if (instance == NULL) {
    return;
  }

  uint8_t insert_at = *io_count;
  while (insert_at > 0) {
    const AbilityInstance *prev =
        ecs_get(world, out_abilities[insert_at - 1], AbilityInstance);
    if (prev == NULL || prev->registry_order <= instance->registry_order) {
      break;
    }

    out_abilities[insert_at] = out_abilities[insert_at - 1];
    insert_at--;
  }

  out_abilities[insert_at] = ability_entity;
  (*io_count)++;
}

static void insert_sorted_action_ability(ecs_world_t *world,
                                         ecs_entity_t ability_entity,
                                         ecs_entity_t *out_abilities,
                                         uint8_t *io_count, uint8_t out_cap) {
  if (out_abilities == NULL || io_count == NULL || *io_count >= out_cap) {
    return;
  }

  const AbilityInstance *instance = ecs_get(world, ability_entity, AbilityInstance);
  if (instance == NULL || instance->action_index == AZK_NO_ACTION_INDEX) {
    return;
  }

  uint8_t insert_at = *io_count;
  while (insert_at > 0) {
    const AbilityInstance *prev =
        ecs_get(world, out_abilities[insert_at - 1], AbilityInstance);
    if (prev == NULL || prev->action_index <= instance->action_index) {
      break;
    }

    out_abilities[insert_at] = out_abilities[insert_at - 1];
    insert_at--;
  }

  out_abilities[insert_at] = ability_entity;
  (*io_count)++;
}

uint8_t azk_collect_card_abilities(ecs_world_t *world, ecs_entity_t card,
                                   ecs_entity_t *out_abilities,
                                   uint8_t out_cap) {
  if (world == NULL || card == 0 || out_abilities == NULL || out_cap == 0) {
    return 0;
  }

  const CardId *card_id = ecs_get(world, card, CardId);
  uint8_t count = 0;
  ecs_iter_t it = ecs_each_id(world, ecs_pair(Rel_AbilityOf, card));
  while (ecs_each_next(&it)) {
    for (int32_t i = 0; i < it.count && count < out_cap; ++i) {
      insert_sorted_ability(world, it.entities[i], out_abilities, &count,
                            out_cap);
    }
  }

  if (card_id == NULL) {
    return count;
  }

  bool stale = count != azk_get_ability_count(card_id->id);
  for (uint8_t i = 0; i < count && !stale; ++i) {
    const AbilityInstance *instance =
        ecs_get(world, out_abilities[i], AbilityInstance);
    if (instance == NULL || instance->card_def_id != card_id->id) {
      stale = true;
    }
  }

  if (!stale) {
    return count;
  }

  return azk_sync_card_abilities(world, card, out_abilities, out_cap);
}

uint8_t azk_collect_card_timed_abilities(ecs_world_t *world, ecs_entity_t card,
                                         ecs_id_t timing_tag,
                                         ecs_entity_t *out_abilities,
                                         uint8_t out_cap) {
  if (world == NULL || card == 0 || timing_tag == 0 || out_abilities == NULL ||
      out_cap == 0) {
    return 0;
  }

  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count =
      azk_collect_card_abilities(world, card, abilities, AZK_MAX_CARD_ABILITIES);
  uint8_t count = 0;
  for (uint8_t i = 0; i < ability_count && count < out_cap; ++i) {
    if (!azk_ability_has_timing(world, abilities[i], timing_tag)) {
      continue;
    }

    out_abilities[count++] = abilities[i];
  }

  return count;
}

uint8_t azk_collect_card_action_abilities(ecs_world_t *world, ecs_entity_t card,
                                          ecs_entity_t *out_abilities,
                                          uint8_t out_cap) {
  if (world == NULL || card == 0 || out_abilities == NULL || out_cap == 0) {
    return 0;
  }

  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count =
      azk_collect_card_abilities(world, card, abilities, AZK_MAX_CARD_ABILITIES);
  uint8_t count = 0;
  for (uint8_t i = 0; i < ability_count && count < out_cap; ++i) {
    insert_sorted_action_ability(world, abilities[i], out_abilities, &count,
                                 out_cap);
  }

  return count;
}

ecs_entity_t azk_find_card_action_ability(ecs_world_t *world, ecs_entity_t card,
                                          int8_t action_index) {
  if (world == NULL || card == 0 || action_index < 0) {
    return 0;
  }

  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count = azk_collect_card_action_abilities(
      world, card, abilities, AZK_MAX_CARD_ABILITIES);
  for (uint8_t i = 0; i < ability_count; ++i) {
    const AbilityInstance *instance = ecs_get(world, abilities[i], AbilityInstance);
    if (instance != NULL && instance->action_index == action_index) {
      return abilities[i];
    }
  }

  return 0;
}

ecs_entity_t azk_find_card_ability_by_registry_order(ecs_world_t *world,
                                                     ecs_entity_t card,
                                                     uint8_t registry_order) {
  if (world == NULL || card == 0) {
    return 0;
  }

  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count =
      azk_collect_card_abilities(world, card, abilities, AZK_MAX_CARD_ABILITIES);
  for (uint8_t i = 0; i < ability_count; ++i) {
    const AbilityInstance *instance = ecs_get(world, abilities[i], AbilityInstance);
    if (instance != NULL && instance->registry_order == registry_order) {
      return abilities[i];
    }
  }

  return 0;
}

bool azk_ability_has_timing(ecs_world_t *world, ecs_entity_t ability_entity,
                            ecs_id_t timing_tag) {
  const AbilityDef *def = azk_get_ability_def_for_entity(world, ability_entity);
  return azk_ability_def_has_timing(def, timing_tag);
}

int8_t azk_determine_action_index_for_card_ability(ecs_world_t *world,
                                                   ecs_entity_t card,
                                                   uint8_t registry_order,
                                                   const AbilityDef *def) {
  if (!is_player_invokable_ability(world, card, def)) {
    return AZK_NO_ACTION_INDEX;
  }

  const CardId *card_id = ecs_get(world, card, CardId);
  if (card_id == NULL) {
    return 0;
  }

  int8_t action_index = 0;
  for (uint8_t i = 0; i < registry_order; ++i) {
    const AbilityDef *prior_def = azk_get_ability_def_at(card_id->id, i);
    if (is_player_invokable_ability(world, card, prior_def)) {
      ++action_index;
    }
  }

  return action_index;
}

AbilityInvocationMode azk_determine_invocation_mode_for_card_ability(
    ecs_world_t *world, ecs_entity_t card, const AbilityDef *def) {
  if (def == NULL) {
    return ABILITY_INVOCATION_NONE;
  }

  if (is_player_invokable_ability(world, card, def)) {
    return ABILITY_INVOCATION_PLAYER;
  }

  if (def->init_passive_observers != NULL && def->timing_tag == 0 &&
      def->secondary_timing_tag == 0) {
    return ABILITY_INVOCATION_PASSIVE;
  }

  return ABILITY_INVOCATION_TRIGGERED;
}
