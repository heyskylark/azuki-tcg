#include "abilities/cards/stt01_008.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/status_util.h"

// STT01-008: When equipped with a weapon card, this card has +1 attack.

// Helper to count weapons attached to an entity
static int count_attached_weapons(ecs_world_t *world, ecs_entity_t entity) {
  int count = 0;
  ecs_iter_t it = ecs_children(world, entity);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      if (ecs_has_id(world, it.entities[i], TWeapon)) {
        count++;
      }
    }
  }
  return count;
}

// Observer callback for weapon attachment/detachment
static void stt01_008_weapon_observer(ecs_iter_t *it) {
  ecs_world_t *world = it->world;

  for (int i = 0; i < it->count; i++) {
    ecs_entity_t weapon = it->entities[i];

    // Get the parent entity (the one the weapon is/was attached to)
    // For EcsOnRemove, the ChildOf relationship still exists during callback
    ecs_entity_t parent = ecs_get_target(world, weapon, EcsChildOf, 0);
    if (!parent) {
      continue;
    }

    // Verify parent is an stt01-008 card
    const CardId *card_id = ecs_get(world, parent, CardId);
    if (!card_id || card_id->id != CARD_DEF_STT01_008) {
      continue;
    }

    // Count current weapons attached
    int weapon_count = count_attached_weapons(world, parent);

    // For EcsOnRemove, the weapon being removed is still counted, so adjust
    if (it->event == EcsOnRemove) {
      weapon_count--;
    }

    // Queue the passive buff update for processing on next iteration.
    // During observer callbacks, writes are deferred and not visible to
    // subsequent code (like recalculate_attack_from_buffs). By queuing,
    // we ensure the buff is applied after deferred ops are flushed.
    if (weapon_count >= 1) {
      // Queue buff add (if not already buffed)
      cli_render_logf("[STT01-008] Weapon equipped, queuing +1 attack buff");
      azk_queue_passive_buff_update(world, parent, parent, 1, false);
    } else {
      // Queue buff remove
      cli_render_logf("[STT01-008] No weapons equipped, queuing buff removal");
      azk_queue_passive_buff_update(world, parent, parent, 0, true);
    }
  }
}

void stt01_008_init_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  // Create observer watching for TWeapon entities with ChildOf relationship to
  // this card
  ecs_entity_t observer = ecs_observer(
      world,
      {.query.terms = {{.id = ecs_pair(EcsChildOf, card)}, {.id = TWeapon}},
       .events = {EcsOnAdd, EcsOnRemove},
       .callback = stt01_008_weapon_observer});

  // Store observer ID in PassiveObserverContext for cleanup
  ecs_set(world, card, PassiveObserverContext,
          {.observers = {observer, 0, 0, 0}, .observer_count = 1});

  cli_render_logf("[STT01-008] Initialized weapon observer for card");
}

void stt01_008_cleanup_passive_observers(ecs_world_t *world,
                                         ecs_entity_t card) {
  const PassiveObserverContext *ctx =
      ecs_get(world, card, PassiveObserverContext);
  if (!ctx) {
    return;
  }

  // Delete all observers
  for (uint8_t i = 0; i < ctx->observer_count; i++) {
    if (ctx->observers[i] != 0) {
      ecs_delete(world, ctx->observers[i]);
    }
  }

  // Remove any attack buff from this card's passive effect
  if (ecs_has_pair(world, card, ecs_id(AttackBuff), card)) {
    remove_attack_modifier(world, card, card);
  }

  // Remove the PassiveObserverContext component
  ecs_remove(world, card, PassiveObserverContext);

  cli_render_logf("[STT01-008] Cleaned up weapon observer for card");
}
