#include "abilities/cards/stt01_011.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

// STT01-011 "Raizan": As long as this card is in play, the card Ikazuchi
// (STT01-016 "Raizan's Zanbato") has +5 attack instead of +4.
// Implementation: +1 attack buff to STT01-016 weapons owned by same player
// when at least one STT01-011 is in play (garden or alley).

#define STT01_011_BUFF_AMOUNT 1

// Context structure stored in observers
typedef struct {
  ecs_entity_t card;        // This STT01-011 instance
  ecs_entity_t owner;       // Owner player entity
  uint8_t player_num;       // Player number (0 or 1)
  ecs_entity_t prefab;      // Card prefab (used as buff source)
} Stt01011ObserverCtx;

// Forward declarations
static int count_stt01_011_in_play(ecs_world_t *world, uint8_t player_num);
static void update_stt01_016_buffs_for_player(ecs_world_t *world,
                                               uint8_t player_num,
                                               ecs_entity_t prefab);
static bool is_entity_in_player_garden(ecs_world_t *world, ecs_entity_t entity,
                                        uint8_t player_num);

// Count STT01-011 cards in a player's garden and alley
static int count_stt01_011_in_play(ecs_world_t *world, uint8_t player_num) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t garden = gs->zones[player_num].garden;
  ecs_entity_t alley = gs->zones[player_num].alley;

  int count = 0;

  // Check garden
  ecs_iter_t it = ecs_children(world, garden);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      const CardId *card_id = ecs_get(world, it.entities[i], CardId);
      if (card_id && card_id->id == CARD_DEF_STT01_011) {
        count++;
      }
    }
  }

  // Check alley
  it = ecs_children(world, alley);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      const CardId *card_id = ecs_get(world, it.entities[i], CardId);
      if (card_id && card_id->id == CARD_DEF_STT01_011) {
        count++;
      }
    }
  }

  return count;
}

// Check if an entity is in a player's garden
static bool is_entity_in_player_garden(ecs_world_t *world, ecs_entity_t entity,
                                        uint8_t player_num) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t garden = gs->zones[player_num].garden;

  ecs_entity_t parent = ecs_get_target(world, entity, EcsChildOf, 0);
  return parent == garden;
}

// Update buffs on all STT01-016 weapons for a player
static void update_stt01_016_buffs_for_player(ecs_world_t *world,
                                               uint8_t player_num,
                                               ecs_entity_t prefab) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t garden = gs->zones[player_num].garden;

  int stt01_011_count = count_stt01_011_in_play(world, player_num);
  bool should_have_buff = (stt01_011_count > 0);

  cli_render_logf("[STT01-011] Updating STT01-016 buffs for player %d "
                  "(STT01-011 count: %d, should_have_buff: %s)",
                  player_num, stt01_011_count, should_have_buff ? "yes" : "no");

  // Iterate through garden entities to find equipped STT01-016 weapons
  ecs_iter_t garden_it = ecs_children(world, garden);
  while (ecs_children_next(&garden_it)) {
    for (int i = 0; i < garden_it.count; i++) {
      ecs_entity_t garden_entity = garden_it.entities[i];

      // Check children of this garden entity for weapons
      ecs_iter_t weapon_it = ecs_children(world, garden_entity);
      while (ecs_children_next(&weapon_it)) {
        for (int j = 0; j < weapon_it.count; j++) {
          ecs_entity_t weapon = weapon_it.entities[j];

          // Check if this is STT01-016
          const CardId *weapon_id = ecs_get(world, weapon, CardId);
          if (!weapon_id || weapon_id->id != CARD_DEF_STT01_016) {
            continue;
          }

          // Check if weapon is owned by the same player
          ecs_entity_t weapon_owner =
              ecs_get_target(world, weapon, Rel_OwnedBy, 0);
          uint8_t weapon_owner_num = get_player_number(world, weapon_owner);
          if (weapon_owner_num != player_num) {
            continue;
          }

          bool has_buff = ecs_has_pair(world, weapon, ecs_id(AttackBuff), prefab);

          if (should_have_buff && !has_buff) {
            cli_render_logf("[STT01-011] Adding +%d buff to STT01-016 weapon %lu",
                            STT01_011_BUFF_AMOUNT, (unsigned long)weapon);
            azk_queue_passive_buff_update(world, weapon, prefab,
                                          STT01_011_BUFF_AMOUNT, 0, false);
          } else if (!should_have_buff && has_buff) {
            cli_render_logf("[STT01-011] Removing buff from STT01-016 weapon %lu",
                            (unsigned long)weapon);
            azk_queue_passive_buff_update(world, weapon, prefab, 0, 0, true);
          }
        }
      }
    }
  }
}

// Observer callback for cards entering/leaving owner's garden or alley
static void stt01_011_zone_observer(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  Stt01011ObserverCtx *ctx = it->ctx;

  if (!ctx) {
    return;
  }

  // Verify this STT01-011 card is still valid
  if (!ecs_is_valid(world, ctx->card)) {
    return;
  }

  // Check if any of the changed entities is an STT01-011
  bool stt01_011_changed = false;
  for (int i = 0; i < it->count; i++) {
    const CardId *card_id = ecs_get(world, it->entities[i], CardId);
    if (card_id && card_id->id == CARD_DEF_STT01_011) {
      stt01_011_changed = true;
      break;
    }
  }

  if (!stt01_011_changed) {
    return;
  }

  cli_render_logf("[STT01-011] Zone observer triggered (event=%s)",
                  it->event == EcsOnRemove ? "remove" : "add");

  update_stt01_016_buffs_for_player(world, ctx->player_num, ctx->prefab);
}

// Observer callback for weapons being attached (ChildOf relationship added)
static void stt01_011_weapon_attach_observer(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  Stt01011ObserverCtx *ctx = it->ctx;

  if (!ctx) {
    return;
  }

  // Verify this STT01-011 card is still valid
  if (!ecs_is_valid(world, ctx->card)) {
    return;
  }

  // Only process add events
  if (it->event != EcsOnAdd) {
    return;
  }

  for (int i = 0; i < it->count; i++) {
    ecs_entity_t weapon = it->entities[i];

    // Check if this is STT01-016
    const CardId *weapon_id = ecs_get(world, weapon, CardId);
    if (!weapon_id || weapon_id->id != CARD_DEF_STT01_016) {
      continue;
    }

    // Check if weapon is owned by the same player
    ecs_entity_t weapon_owner = ecs_get_target(world, weapon, Rel_OwnedBy, 0);
    if (!weapon_owner) {
      continue;
    }
    uint8_t weapon_owner_num = get_player_number(world, weapon_owner);
    if (weapon_owner_num != ctx->player_num) {
      continue;
    }

    // Check if the weapon's parent is in owner's garden
    ecs_entity_t parent = ecs_get_target(world, weapon, EcsChildOf, 0);
    if (!parent || !is_entity_in_player_garden(world, parent, ctx->player_num)) {
      continue;
    }

    // Check if there's at least one STT01-011 in play
    int stt01_011_count = count_stt01_011_in_play(world, ctx->player_num);
    if (stt01_011_count <= 0) {
      continue;
    }

    // Check if buff already exists
    bool has_buff = ecs_has_pair(world, weapon, ecs_id(AttackBuff), ctx->prefab);
    if (has_buff) {
      continue;
    }

    cli_render_logf("[STT01-011] Weapon attach observer: Adding +%d buff to "
                    "newly equipped STT01-016 weapon %lu",
                    STT01_011_BUFF_AMOUNT, (unsigned long)weapon);
    azk_queue_passive_buff_update(world, weapon, ctx->prefab,
                                  STT01_011_BUFF_AMOUNT, 0, false);
  }
}

void stt01_011_init_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  // Get owner's zones
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (!owner) {
    cli_render_logf("[STT01-011] Error: card has no owner");
    return;
  }

  uint8_t player_num = get_player_number(world, owner);
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t garden = gs->zones[player_num].garden;
  ecs_entity_t alley = gs->zones[player_num].alley;

  // Get card prefab (used as buff source to prevent stacking)
  ecs_entity_t prefab = ecs_get_target(world, card, EcsIsA, 0);
  if (!prefab) {
    cli_render_logf("[STT01-011] Error: card has no prefab");
    return;
  }

  // Allocate context (will be freed in cleanup)
  Stt01011ObserverCtx *ctx = ecs_os_malloc(sizeof(Stt01011ObserverCtx));
  ctx->card = card;
  ctx->owner = owner;
  ctx->player_num = player_num;
  ctx->prefab = prefab;

  // Observer 1: Watch for cards entering/leaving the owner's garden
  ecs_entity_t obs_garden = ecs_observer(
      world,
      {.query.terms = {{.id = ecs_pair(EcsChildOf, garden)},
                       {.id = ecs_id(CardId)}},
       .events = {EcsOnAdd, EcsOnRemove},
       .callback = stt01_011_zone_observer,
       .ctx = ctx});

  // Observer 2: Watch for cards entering/leaving the owner's alley
  ecs_entity_t obs_alley = ecs_observer(
      world,
      {.query.terms = {{.id = ecs_pair(EcsChildOf, alley)},
                       {.id = ecs_id(CardId)}},
       .events = {EcsOnAdd, EcsOnRemove},
       .callback = stt01_011_zone_observer,
       .ctx = ctx});

  // Observer 3: Watch for weapons being attached to any entity (wildcard)
  // This fires when TWeapon entities get a new ChildOf relationship
  ecs_entity_t obs_weapon = ecs_observer(
      world,
      {.query.terms = {{.id = ecs_pair(EcsChildOf, EcsWildcard)},
                       {.id = TWeapon}},
       .events = {EcsOnAdd},
       .callback = stt01_011_weapon_attach_observer,
       .ctx = ctx});

  // Store observer IDs and context in PassiveObserverContext for cleanup
  ecs_set(world, card, PassiveObserverContext,
          {.observers = {obs_garden, obs_alley, obs_weapon, 0},
           .observer_count = 3,
           .ctx = ctx});

  cli_render_logf("[STT01-011] Initialized observers for card %lu "
                  "(garden=%lu, alley=%lu, player=%d)",
                  (unsigned long)card, (unsigned long)garden,
                  (unsigned long)alley, player_num);
}

void stt01_011_cleanup_passive_observers(ecs_world_t *world,
                                          ecs_entity_t card) {
  const PassiveObserverContext *obs_ctx =
      ecs_get(world, card, PassiveObserverContext);
  if (!obs_ctx) {
    return;
  }

  // Get context for prefab lookup before deleting observers
  Stt01011ObserverCtx *ctx = obs_ctx->ctx;
  uint8_t player_num = 0;
  ecs_entity_t prefab = 0;

  if (ctx) {
    player_num = ctx->player_num;
    prefab = ctx->prefab;
  }

  // Delete all observers
  for (uint8_t i = 0; i < obs_ctx->observer_count; i++) {
    if (obs_ctx->observers[i] != 0) {
      ecs_delete(world, obs_ctx->observers[i]);
    }
  }

  // Free the allocated context
  if (ctx) {
    ecs_os_free(ctx);
  }

  // Check if this was the last STT01-011 in play for this player
  // If so, remove buffs from all STT01-016 weapons
  if (prefab != 0) {
    // Count remaining STT01-011 cards (excluding this one which is being removed)
    int remaining_count = count_stt01_011_in_play(world, player_num);

    // The card being cleaned up might still be counted if it's still in zone
    // So we check if count <= 1 (this card is the last one)
    // Actually, by cleanup time the card might already be removed from zone
    // So we trigger update which will correctly handle the count
    cli_render_logf("[STT01-011] Cleanup: remaining STT01-011 count = %d",
                    remaining_count);

    // Trigger update to remove buffs if no STT01-011 remain
    // Note: The card being cleaned up is likely still in zone at this point
    // so remaining_count might include it. We subtract 1 if this card is still there.
    const GameState *gs = ecs_singleton_get(world, GameState);
    ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
    bool still_in_play = (parent == gs->zones[player_num].garden ||
                          parent == gs->zones[player_num].alley);

    if (still_in_play && remaining_count <= 1) {
      // This is the last STT01-011, remove all buffs
      cli_render_logf("[STT01-011] Last STT01-011 leaving play, removing buffs");
      update_stt01_016_buffs_for_player(world, player_num, prefab);
    } else if (!still_in_play && remaining_count == 0) {
      // Card already removed from zone and no others remain
      cli_render_logf("[STT01-011] No STT01-011 in play, removing buffs");
      update_stt01_016_buffs_for_player(world, player_num, prefab);
    }
  }

  // Remove the PassiveObserverContext component
  ecs_remove(world, card, PassiveObserverContext);

  cli_render_logf("[STT01-011] Cleaned up observers for card %lu",
                  (unsigned long)card);
}
