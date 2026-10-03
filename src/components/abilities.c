#include "components/abilities.h"
#include "abilities/ability_registry.h"
#include "abilities/passive/passive_runtime.h"
#include "generated/card_defs.h"
#include "utils/ability_util.h"
#include "utils/status_util.h"

#include <stdio.h>

ECS_COMPONENT_DECLARE(AbilityRepeatContext);
ECS_COMPONENT_DECLARE(AbilityCostRequirements);
ECS_COMPONENT_DECLARE(AbilityEffectRequirements);
ECS_COMPONENT_DECLARE(AbilityFunctions);

ECS_TAG_DECLARE(AOnPlay);
ECS_TAG_DECLARE(AStartOfTurn);
ECS_TAG_DECLARE(AStartOfEachTurn);
ECS_TAG_DECLARE(AEndOfTurn);
ECS_TAG_DECLARE(AWhenEquipping);
ECS_TAG_DECLARE(AWhenEquipped);
ECS_TAG_DECLARE(AMain);
ECS_TAG_DECLARE(AWhenAttacking);
ECS_TAG_DECLARE(AAfterAttacking);
ECS_TAG_DECLARE(AWhenAttacked);
ECS_TAG_DECLARE(AWhenTakesDamage);
ECS_TAG_DECLARE(AWhenDealsDamage);
ECS_TAG_DECLARE(AWhenEntersGarden);
ECS_TAG_DECLARE(AResponse);
ECS_TAG_DECLARE(AAlleyOnly);
ECS_TAG_DECLARE(AGardenOnly);
ECS_TAG_DECLARE(AOnceTurn);
ECS_TAG_DECLARE(AWhenReturnedToHand);
ECS_TAG_DECLARE(AWhenDestroyed);
ECS_TAG_DECLARE(AWhenSacrificed);
ECS_TAG_DECLARE(AIgnoresCooldown);
ECS_TAG_DECLARE(AOnGatePortal);

ECS_TAG_DECLARE(Charge);
ECS_TAG_DECLARE(Defender);
ECS_TAG_DECLARE(Infiltrate);
ECS_TAG_DECLARE(Godmode);
ECS_TAG_DECLARE(SacrificeAtEndOfTurn);
ECS_TAG_DECLARE(Taunt);
ECS_TAG_DECLARE(Rooted);
ECS_TAG_DECLARE(AttrCanTargetLeaderOnly);
ECS_TAG_DECLARE(AttrCanTargetTappedAndUntappedAlley);
ECS_TAG_DECLARE(AttrGardenForceTapped);
ECS_TAG_DECLARE(AttrCountsAsIkzSource);

ECS_TAG_DECLARE(Frozen);
ECS_TAG_DECLARE(Shocked);

ECS_TAG_DECLARE(EffectImmune);

ECS_COMPONENT_DECLARE(CarapaceValue);
ECS_COMPONENT_DECLARE(CarapaceBuff);
ECS_COMPONENT_DECLARE(CardConditionCountdown);
ECS_COMPONENT_DECLARE(AttackBuff);
ECS_COMPONENT_DECLARE(HealthBuff);
ECS_COMPONENT_DECLARE(CombatDamageModifier);
ECS_COMPONENT_DECLARE(EquippedCombatModifier);
ECS_COMPONENT_DECLARE(PassiveObserverContext);
ECS_COMPONENT_DECLARE(DamageTracker);
ECS_COMPONENT_DECLARE(CopiedCardText);

static void clear_ability_tags(ecs_world_t *world, ecs_entity_t ability_entity) {
  ecs_remove(world, ability_entity, AOnPlay);
  ecs_remove(world, ability_entity, AStartOfTurn);
  ecs_remove(world, ability_entity, AStartOfEachTurn);
  ecs_remove(world, ability_entity, AEndOfTurn);
  ecs_remove(world, ability_entity, AWhenEquipping);
  ecs_remove(world, ability_entity, AWhenEquipped);
  ecs_remove(world, ability_entity, AMain);
  ecs_remove(world, ability_entity, AWhenAttacking);
  ecs_remove(world, ability_entity, AAfterAttacking);
  ecs_remove(world, ability_entity, AWhenAttacked);
  ecs_remove(world, ability_entity, AWhenTakesDamage);
  ecs_remove(world, ability_entity, AWhenDealsDamage);
  ecs_remove(world, ability_entity, AWhenEntersGarden);
  ecs_remove(world, ability_entity, AResponse);
  ecs_remove(world, ability_entity, AAlleyOnly);
  ecs_remove(world, ability_entity, AGardenOnly);
  ecs_remove(world, ability_entity, AOnceTurn);
  ecs_remove(world, ability_entity, AWhenReturnedToHand);
  ecs_remove(world, ability_entity, AWhenDestroyed);
  ecs_remove(world, ability_entity, AWhenSacrificed);
  ecs_remove(world, ability_entity, AIgnoresCooldown);
  ecs_remove(world, ability_entity, AOnGatePortal);
}

static void cleanup_attached_ability_entity(ecs_world_t *world,
                                            ecs_entity_t ability_entity,
                                            const AbilityInstance *instance) {
  if (world == NULL || ability_entity == 0 || instance == NULL) {
    return;
  }

  if (ecs_should_quit(world)) {
    // World teardown: card cleanup callbacks adjust buffs on zone entities
    // that may already be deleted. Only release the observer context, which
    // every card allocates through the passive runtime.
    if (ecs_has(world, ability_entity, PassiveObserverContext)) {
      azk_cleanup_passive_observer_context(
          world, ability_entity,
          &(PassiveObserverCleanupOptions){.free_ctx = true});
    }
    return;
  }

  const AbilityDef *ability_def =
      azk_get_ability_def_at(instance->card_def_id, instance->registry_order);
  if (ability_def != NULL && ability_def->cleanup_passive_observers != NULL) {
    ability_def->cleanup_passive_observers(world, ability_entity);
  } else if (ecs_has(world, ability_entity, PassiveObserverContext)) {
    azk_cleanup_passive_observer_context(world, ability_entity, NULL);
  }
}

static void on_remove_ability_instance(ecs_iter_t *it) {
  AbilityInstance *instances = ecs_field(it, AbilityInstance, 0);
  if (instances == NULL) {
    return;
  }

  for (int32_t i = 0; i < it->count; ++i) {
    cleanup_attached_ability_entity(it->world, it->entities[i], &instances[i]);
  }
}

void azk_register_ability_components(ecs_world_t *world) {
  ECS_COMPONENT_DEFINE(world, AbilityRepeatContext);
  ECS_COMPONENT_DEFINE(world, AbilityCostRequirements);
  ECS_COMPONENT_DEFINE(world, AbilityEffectRequirements);
  ECS_COMPONENT_DEFINE(world, AbilityFunctions);

  ECS_TAG_DEFINE(world, AOnPlay);
  ECS_TAG_DEFINE(world, AStartOfTurn);
  ECS_TAG_DEFINE(world, AStartOfEachTurn);
  ECS_TAG_DEFINE(world, AEndOfTurn);
  ECS_TAG_DEFINE(world, AWhenEquipping);
  ECS_TAG_DEFINE(world, AWhenEquipped);
  ECS_TAG_DEFINE(world, AMain);
  ECS_TAG_DEFINE(world, AWhenAttacking);
  ECS_TAG_DEFINE(world, AAfterAttacking);
  ECS_TAG_DEFINE(world, AWhenAttacked);
  ECS_TAG_DEFINE(world, AWhenTakesDamage);
  ECS_TAG_DEFINE(world, AWhenDealsDamage);
  ECS_TAG_DEFINE(world, AWhenEntersGarden);
  ECS_TAG_DEFINE(world, AResponse);
  ECS_TAG_DEFINE(world, AAlleyOnly);
  ECS_TAG_DEFINE(world, AGardenOnly);
  ECS_TAG_DEFINE(world, AOnceTurn);
  ECS_TAG_DEFINE(world, AWhenReturnedToHand);
  ECS_TAG_DEFINE(world, AWhenDestroyed);
  ECS_TAG_DEFINE(world, AWhenSacrificed);
  ECS_TAG_DEFINE(world, AIgnoresCooldown);
  ECS_TAG_DEFINE(world, AOnGatePortal);

  ECS_TAG_DEFINE(world, Charge);
  ECS_TAG_DEFINE(world, Defender);
  ECS_TAG_DEFINE(world, Infiltrate);
  ECS_TAG_DEFINE(world, Godmode);
  ECS_TAG_DEFINE(world, SacrificeAtEndOfTurn);
  ECS_TAG_DEFINE(world, Taunt);
  ECS_TAG_DEFINE(world, Rooted);
  ECS_TAG_DEFINE(world, AttrCanTargetLeaderOnly);
  ECS_TAG_DEFINE(world, AttrCanTargetTappedAndUntappedAlley);
  ECS_TAG_DEFINE(world, AttrGardenForceTapped);
  ECS_TAG_DEFINE(world, AttrCountsAsIkzSource);

  ECS_TAG_DEFINE(world, Frozen);
  ECS_TAG_DEFINE(world, Shocked);

  ECS_TAG_DEFINE(world, EffectImmune);

  ECS_COMPONENT_DEFINE(world, CarapaceValue);
  ECS_COMPONENT_DEFINE(world, CarapaceBuff);
  ECS_COMPONENT_DEFINE(world, CardConditionCountdown);
  ECS_COMPONENT_DEFINE(world, AttackBuff);
  ECS_COMPONENT_DEFINE(world, HealthBuff);
  ECS_COMPONENT_DEFINE(world, CombatDamageModifier);
  ECS_COMPONENT_DEFINE(world, EquippedCombatModifier);
  ECS_COMPONENT_DEFINE(world, PassiveObserverContext);
  ECS_COMPONENT_DEFINE(world, DamageTracker);
  ECS_COMPONENT_DEFINE(world, CopiedCardText);
  ecs_set_hooks(world, AbilityInstance,
                {.on_remove = on_remove_ability_instance});

  // Weapons like AZK01-018 store their static equip modifier on the prefab and
  // need instances to inherit that data.
  ecs_add_pair(world, ecs_id(EquippedCombatModifier), EcsOnInstantiate,
               EcsInherit);

  // Ensure CardConditionCountdown is copied to each instance on instantiation
  // (EcsOverride gives each instance its own mutable copy, unlike EcsInherit)
  ecs_add_pair(world, ecs_id(CardConditionCountdown), EcsOnInstantiate,
               EcsOverride);
}

uint8_t azk_sync_card_abilities(ecs_world_t *world, ecs_entity_t card,
                                ecs_entity_t *out_abilities,
                                uint8_t out_cap) {
  const bool was_deferred =
      ecs_is_deferred(world) && !ecs_stage_is_readonly(world);
  if (was_deferred) {
    ecs_defer_suspend(world);
  }

  const CardId* card_id = ecs_get(world, card, CardId);
  ecs_assert(card_id != 0, ECS_INVALID_PARAMETER, "CardId component not found for card %d", card);

  ecs_entity_t prefab = ecs_get_target(world, card, EcsIsA, 0);
  if (prefab != 0 && !ecs_has(world, card, EquippedCombatModifier)) {
    const EquippedCombatModifier *modifier =
        ecs_get(world, prefab, EquippedCombatModifier);
    if (modifier != NULL) {
      ecs_set_id(world, card, ecs_id(EquippedCombatModifier),
                 sizeof(EquippedCombatModifier), modifier);
    }
  }

  if (azk_can_play_as_response_from_hand(card_id->id)) {
    ecs_add(world, card, AResponse);
  } else if (ecs_has(world, card, AResponse)) {
    ecs_remove(world, card, AResponse);
  }

  const AzkCardAbilityLayout layout =
      azk_get_card_ability_layout(world, card, card_id->id);
  const uint8_t ability_count = layout.own_count + layout.copied_count;
  ecs_entity_t ability_entities[AZK_MAX_CARD_ABILITIES] = {0};
  ecs_entity_t stale_entities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t stale_count = 0;
  ecs_iter_t ability_it = ecs_each_id(world, ecs_pair(Rel_AbilityOf, card));
  while (ecs_each_next(&ability_it)) {
    for (int32_t i = 0; i < ability_it.count; ++i) {
      ecs_entity_t ability_entity = ability_it.entities[i];
      const AbilityInstance *instance =
          ecs_get(world, ability_entity, AbilityInstance);
      const int slot = azk_card_ability_slot(&layout, instance);
      if (slot < 0 || ability_entities[slot] != 0) {
        if (stale_count < AZK_MAX_CARD_ABILITIES) {
          stale_entities[stale_count++] = ability_entity;
        }
        continue;
      }

      ability_entities[slot] = ability_entity;
    }
  }

  for (uint8_t i = 0; i < stale_count; ++i) {
    ecs_delete(world, stale_entities[i]);
  }

  if (ability_count == 0) {
    if (was_deferred) {
      ecs_defer_resume(world);
    }
    return 0;
  }

  uint8_t synced_count = 0;
  int8_t next_action_index = 0;
  for (uint8_t slot = 0; slot < ability_count; ++slot) {
    const bool is_copied = slot >= layout.own_count;
    const CardDefId def_id =
        is_copied ? layout.copied_def_id : layout.own_def_id;
    const uint8_t registry_order =
        is_copied ? (uint8_t)(slot - layout.own_count) : slot;
    const AbilityDef *ability_def =
        azk_get_ability_def_at(def_id, registry_order);
    if (ability_def == NULL || !ability_def->has_ability) {
      continue;
    }

    ecs_entity_t ability_entity = ability_entities[slot];
    if (ability_entity == 0) {
      ability_entity = ecs_new(world);
      char ability_name[128];
      const char *card_name = ecs_get_name(world, card);
      snprintf(ability_name, sizeof(ability_name), "%s__ability_%u_%llu",
               card_name != NULL ? card_name : "card", (unsigned)slot,
               (unsigned long long)ability_entity);
      ecs_set_name(world, ability_entity, ability_name);
      ecs_add_pair(world, ability_entity, Rel_AbilityOf, card);
      ability_entities[slot] = ability_entity;
    }

    clear_ability_tags(world, ability_entity);

    const AbilityInstance *existing_instance =
        ecs_get(world, ability_entity, AbilityInstance);
    const bool same_ability_identity =
        existing_instance != NULL &&
        existing_instance->card_def_id == def_id &&
        existing_instance->registry_order == registry_order;
    if (existing_instance != NULL && !same_ability_identity) {
      cleanup_attached_ability_entity(world, ability_entity, existing_instance);
    }

    const AbilityRepeatContext *existing_repeat =
        ecs_get(world, ability_entity, AbilityRepeatContext);
    bool was_applied =
        same_ability_identity && existing_repeat != NULL
            ? existing_repeat->was_applied
            : false;

    const AbilityInvocationMode invocation_mode =
        azk_determine_invocation_mode_for_card_ability(world, card,
                                                       ability_def);
    int8_t action_index = AZK_NO_ACTION_INDEX;
    if (!is_copied) {
      action_index = azk_determine_action_index_for_card_ability(
          world, card, registry_order, ability_def);
      if (action_index != AZK_NO_ACTION_INDEX) {
        next_action_index = (int8_t)(action_index + 1);
      }
    } else if (invocation_mode == ABILITY_INVOCATION_PLAYER) {
      action_index = next_action_index++;
    }

    ecs_set(world, ability_entity, AbilityInstance,
            {
                .card_def_id = def_id,
                .registry_order = registry_order,
                .action_index = action_index,
                .invocation_mode = (uint8_t)invocation_mode,
            });

    if (ability_def->timing_tag != 0) {
      ecs_add_id(world, ability_entity, ability_def->timing_tag);
    }
    if (ability_def->secondary_timing_tag != 0) {
      ecs_add_id(world, ability_entity, ability_def->secondary_timing_tag);
    }

    if (ability_def->is_once_per_turn) {
      ecs_add(world, ability_entity, AOnceTurn);
      ecs_set(world, ability_entity, AbilityRepeatContext,
              {.is_once_per_turn = true, .was_applied = was_applied});
    } else if (ecs_has(world, ability_entity, AbilityRepeatContext)) {
      ecs_remove(world, ability_entity, AbilityRepeatContext);
    }

    if (ability_def->init_passive_observers != NULL &&
        !ecs_has(world, ability_entity, PassiveObserverContext)) {
      ability_def->init_passive_observers(world, ability_entity);
    }

    if (out_abilities != NULL && synced_count < out_cap) {
      out_abilities[synced_count] = ability_entity;
    }
    ++synced_count;
  }

  if (was_deferred) {
    ecs_defer_resume(world);
  }
  return synced_count;
}

void attach_ability_components(ecs_world_t* world, ecs_entity_t card) {
  (void)azk_sync_card_abilities(world, card, NULL, 0);
}

bool azk_grant_copied_card_text(ecs_world_t *world, ecs_entity_t card,
                                CardDefId source_def_id) {
  const CardId *card_id = ecs_get(world, card, CardId);
  ecs_entity_t source_prefab = azk_prefab_from_id(source_def_id);
  if (card_id == NULL || source_prefab == 0 ||
      source_def_id == card_id->id) {
    return false;
  }

  const bool was_deferred =
      ecs_is_deferred(world) && !ecs_stage_is_readonly(world);
  if (was_deferred) {
    ecs_defer_suspend(world);
  }

  ecs_set(world, card, CopiedCardText, {.card_def_id = source_def_id});
  (void)azk_sync_card_abilities(world, card, NULL, 0);

  const ecs_id_t keyword_tags[] = {
      ecs_id(Charge),
      ecs_id(Defender),
      ecs_id(Infiltrate),
      ecs_id(Godmode),
      ecs_id(AttrCanTargetLeaderOnly),
      ecs_id(AttrCanTargetTappedAndUntappedAlley),
      ecs_id(EffectImmune),
  };
  for (size_t i = 0; i < sizeof(keyword_tags) / sizeof(keyword_tags[0]); ++i) {
    if (!ecs_has_id(world, source_prefab, keyword_tags[i])) {
      continue;
    }
    if (keyword_tags[i] == ecs_id(Charge)) {
      (void)apply_charge_grant(world, card, TAG_GRANT_TICK_END_OF_TURN, 1);
    } else {
      (void)apply_timed_tag_grant(world, card, keyword_tags[i],
                                  TAG_GRANT_TICK_END_OF_TURN, 1);
    }
  }

  if (was_deferred) {
    ecs_defer_resume(world);
  }
  return true;
}

void azk_clear_copied_card_text(ecs_world_t *world, ecs_entity_t card) {
  if (card == 0 || !ecs_has(world, card, CopiedCardText)) {
    return;
  }

  const bool was_deferred =
      ecs_is_deferred(world) && !ecs_stage_is_readonly(world);
  if (was_deferred) {
    ecs_defer_suspend(world);
  }

  ecs_remove(world, card, CopiedCardText);
  (void)azk_sync_card_abilities(world, card, NULL, 0);

  if (was_deferred) {
    ecs_defer_resume(world);
  }
}
