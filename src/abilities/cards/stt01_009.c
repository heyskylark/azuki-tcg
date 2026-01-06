#include "abilities/cards/stt01_009.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

// STT01-009: If there are 6 or more weapon cards in your discard pile,
// this card has +2 attack.

#define STT01_009_WEAPON_THRESHOLD 6
#define STT01_009_BUFF_AMOUNT 2

// Context structure stored in observers
typedef struct {
  ecs_entity_t card;
} Stt01009ObserverCtx;

// Forward declarations
static void stt01_009_check_and_update_buff(ecs_world_t *world,
                                            ecs_entity_t card,
                                            bool is_removal_event);

// Observer callback for weapons entering/leaving the discard zone
static void stt01_009_discard_observer(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  Stt01009ObserverCtx *ctx = it->ctx;

  if (!ctx) {
    return;
  }

  ecs_entity_t card = ctx->card;

  // Verify card is still valid
  if (!ecs_is_valid(world, card)) {
    return;
  }

  // Verify this is still an stt01-009 card
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id || card_id->id != CARD_DEF_STT01_009) {
    return;
  }

  bool is_removal = (it->event == EcsOnRemove);

  cli_render_logf("[STT01-009] Discard observer triggered (event=%s)",
                  is_removal ? "remove" : "add");

  stt01_009_check_and_update_buff(world, card, is_removal);
}

// Observer callback for when this card changes zones
static void stt01_009_zone_change_observer(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  Stt01009ObserverCtx *ctx = it->ctx;

  if (!ctx) {
    cli_render_logf("[STT01-009] Zone observer: no context");
    return;
  }

  ecs_entity_t card = ctx->card;

  // Only process on add events (entering a zone)
  if (it->event != EcsOnAdd) {
    return;
  }

  // Verify card is still valid
  if (!ecs_is_valid(world, card)) {
    cli_render_logf("[STT01-009] Zone observer: card not valid");
    return;
  }

  // Check if the moved entity is our card
  for (int i = 0; i < it->count; i++) {
    if (it->entities[i] == card) {
      cli_render_logf("[STT01-009] Zone change observer: card %lu entered new zone",
                      (unsigned long)card);
      stt01_009_check_and_update_buff(world, card, false);
      return;
    }
  }

  // Log if our card wasn't in this batch (for debugging)
  // This is expected - the observer fires for ALL cards changing zones
}

// Shared logic to count weapons in discard and update buff
static void stt01_009_check_and_update_buff(ecs_world_t *world,
                                            ecs_entity_t card,
                                            bool is_removal_event) {
  // Verify card is still valid
  if (!ecs_is_valid(world, card)) {
    cli_render_logf("[STT01-009] Card not valid, skipping");
    return;
  }

  // Get owner first to access zones
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (!owner) {
    cli_render_logf("[STT01-009] Card has no owner, skipping");
    return;
  }

  uint8_t player_num = get_player_number(world, owner);
  const GameState *gs = ecs_singleton_get(world, GameState);

  // Check if card is in owner's garden (compare directly with zone entity)
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  ecs_entity_t garden = gs->zones[player_num].garden;

  cli_render_logf("[STT01-009] Card parent=%lu, owner garden=%lu",
                  (unsigned long)parent, (unsigned long)garden);

  if (parent != garden) {
    // Not in garden - remove any buff if present
    if (ecs_has_pair(world, card, ecs_id(AttackBuff), card)) {
      cli_render_logf("[STT01-009] Card not in garden, queuing buff removal");
      azk_queue_passive_buff_update(world, card, card, 0, 0, true);
    }
    return;
  }

  ecs_entity_t discard = gs->zones[player_num].discard;

  // Count weapons in discard
  int weapon_count = count_weapons_in_zone(world, discard);

  // For removal events, the weapon being removed is still counted
  if (is_removal_event) {
    weapon_count--;
  }

  cli_render_logf("[STT01-009] Weapons in discard: %d (threshold: %d)",
                  weapon_count, STT01_009_WEAPON_THRESHOLD);

  // Queue buff update (queue handles idempotency)
  if (weapon_count >= STT01_009_WEAPON_THRESHOLD) {
    cli_render_logf("[STT01-009] Threshold met, queuing +%d attack buff",
                    STT01_009_BUFF_AMOUNT);
    azk_queue_passive_buff_update(world, card, card, STT01_009_BUFF_AMOUNT,
                                  0, false);
  } else {
    cli_render_logf("[STT01-009] Below threshold, queuing buff removal");
    azk_queue_passive_buff_update(world, card, card, 0, 0, true);
  }
}

void stt01_009_init_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  // Get owner's zones
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (!owner) {
    cli_render_logf("[STT01-009] Error: card has no owner");
    return;
  }

  uint8_t player_num = get_player_number(world, owner);
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t discard = gs->zones[player_num].discard;
  ecs_entity_t garden = gs->zones[player_num].garden;

  // Allocate context (will be freed in cleanup)
  Stt01009ObserverCtx *ctx = ecs_os_malloc(sizeof(Stt01009ObserverCtx));
  ctx->card = card;

  // Observer 1: Watch for weapons entering/leaving the discard zone
  ecs_entity_t obs_discard = ecs_observer(
      world,
      {.query.terms = {{.id = ecs_pair(EcsChildOf, discard)}, {.id = TWeapon}},
       .events = {EcsOnAdd, EcsOnRemove},
       .callback = stt01_009_discard_observer,
       .ctx = ctx});

  // Observer 2: Watch for cards entering the owner's garden specifically
  // This fires when any card with CardId enters this player's garden
  ecs_entity_t obs_zone = ecs_observer(
      world,
      {.query.terms = {{.id = ecs_pair(EcsChildOf, garden)},
                       {.id = ecs_id(CardId)}},
       .events = {EcsOnAdd},
       .callback = stt01_009_zone_change_observer,
       .ctx = ctx});

  // Store observer IDs and context in PassiveObserverContext for cleanup
  ecs_set(world, card, PassiveObserverContext,
          {.observers = {obs_discard, obs_zone, 0, 0},
           .observer_count = 2,
           .ctx = ctx});

  cli_render_logf("[STT01-009] Initialized observers for card %lu (discard=%lu, garden=%lu)",
                  (unsigned long)card, (unsigned long)discard, (unsigned long)garden);
}

void stt01_009_cleanup_passive_observers(ecs_world_t *world,
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

  // Free the allocated context
  if (ctx->ctx) {
    ecs_os_free(ctx->ctx);
  }

  // Remove any attack buff from this card's passive effect
  if (ecs_has_pair(world, card, ecs_id(AttackBuff), card)) {
    remove_attack_modifier(world, card, card);
  }

  // Remove the PassiveObserverContext component
  ecs_remove(world, card, PassiveObserverContext);

  cli_render_logf("[STT01-009] Cleaned up observers for card");
}
