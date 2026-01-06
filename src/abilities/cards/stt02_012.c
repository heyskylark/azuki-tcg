#include "abilities/cards/stt02_012.h"

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/status_util.h"

// STT02-012: If the number of entities in your garden is 2 or more than
// the number of entities in your opponent's garden, this card has +1 attack
// and +1 health.

#define STT02_012_GARDEN_THRESHOLD 2
#define STT02_012_ATK_BUFF 1
#define STT02_012_HP_BUFF 1

// Context structure stored in observers
typedef struct {
  ecs_entity_t card;
  uint8_t owner_player_num;
} Stt02012ObserverCtx;

// Forward declarations
static void stt02_012_check_and_update_buff(ecs_world_t *world,
                                            ecs_entity_t card,
                                            uint8_t owner_player_num,
                                            bool is_removal_event,
                                            bool is_player_garden);

// Count entities (cards with TEntity tag) in a zone
static int count_entities_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  int count = 0;
  for (int32_t i = 0; i < cards.count; i++) {
    if (ecs_has_id(world, cards.ids[i], TEntity)) {
      count++;
    }
  }
  return count;
}

// Observer callback for entities entering/leaving the player's garden
static void stt02_012_player_garden_observer(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  Stt02012ObserverCtx *ctx = it->ctx;

  if (!ctx) {
    return;
  }

  ecs_entity_t card = ctx->card;

  // Verify card is still valid
  if (!ecs_is_valid(world, card)) {
    return;
  }

  // Verify this is still an stt02-012 card
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id || card_id->id != CARD_DEF_STT02_012) {
    return;
  }

  bool is_removal = (it->event == EcsOnRemove);

  cli_render_logf("[STT02-012] Player garden observer triggered (event=%s)",
                  is_removal ? "remove" : "add");

  stt02_012_check_and_update_buff(world, card, ctx->owner_player_num,
                                  is_removal, true);
}

// Observer callback for entities entering/leaving the opponent's garden
static void stt02_012_opponent_garden_observer(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  Stt02012ObserverCtx *ctx = it->ctx;

  if (!ctx) {
    return;
  }

  ecs_entity_t card = ctx->card;

  // Verify card is still valid
  if (!ecs_is_valid(world, card)) {
    return;
  }

  // Verify this is still an stt02-012 card
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id || card_id->id != CARD_DEF_STT02_012) {
    return;
  }

  bool is_removal = (it->event == EcsOnRemove);

  cli_render_logf("[STT02-012] Opponent garden observer triggered (event=%s)",
                  is_removal ? "remove" : "add");

  stt02_012_check_and_update_buff(world, card, ctx->owner_player_num,
                                  is_removal, false);
}

// Shared logic to check garden counts and update buff
static void stt02_012_check_and_update_buff(ecs_world_t *world,
                                            ecs_entity_t card,
                                            uint8_t owner_player_num,
                                            bool is_removal_event,
                                            bool is_player_garden) {
  // Verify card is still valid
  if (!ecs_is_valid(world, card)) {
    cli_render_logf("[STT02-012] Card not valid, skipping");
    return;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t opponent_num = (owner_player_num == 0) ? 1 : 0;

  // Check if card is in owner's garden (buff only applies when in garden)
  ecs_entity_t parent = ecs_get_target(world, card, EcsChildOf, 0);
  ecs_entity_t owner_garden = gs->zones[owner_player_num].garden;

  if (parent != owner_garden) {
    // Not in garden - remove any buff if present
    if (ecs_has_pair(world, card, ecs_id(AttackBuff), card) ||
        ecs_has_pair(world, card, ecs_id(HealthBuff), card)) {
      cli_render_logf("[STT02-012] Card not in garden, queuing buff removal");
      azk_queue_passive_buff_update(world, card, card, 0, 0, true);
    }
    return;
  }

  ecs_entity_t opponent_garden = gs->zones[opponent_num].garden;

  // Count entities in both gardens
  int player_count = count_entities_in_zone(world, owner_garden);
  int opponent_count = count_entities_in_zone(world, opponent_garden);

  // Adjust count for removal events (the entity being removed is still counted)
  if (is_removal_event) {
    if (is_player_garden) {
      player_count--;
    } else {
      opponent_count--;
    }
  }

  int difference = player_count - opponent_count;

  cli_render_logf("[STT02-012] Garden counts: player=%d, opponent=%d, diff=%d (threshold=%d)",
                  player_count, opponent_count, difference, STT02_012_GARDEN_THRESHOLD);

  // Queue buff update (queue handles idempotency)
  if (difference >= STT02_012_GARDEN_THRESHOLD) {
    cli_render_logf("[STT02-012] Threshold met, queuing +%d/+%d buff",
                    STT02_012_ATK_BUFF, STT02_012_HP_BUFF);
    azk_queue_passive_buff_update(world, card, card, STT02_012_ATK_BUFF,
                                  STT02_012_HP_BUFF, false);
  } else {
    cli_render_logf("[STT02-012] Below threshold, queuing buff removal");
    azk_queue_passive_buff_update(world, card, card, 0, 0, true);
  }
}

void stt02_012_init_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  // Get owner's zones
  ecs_entity_t owner = ecs_get_target(world, card, Rel_OwnedBy, 0);
  if (!owner) {
    cli_render_logf("[STT02-012] Error: card has no owner");
    return;
  }

  uint8_t owner_player_num = get_player_number(world, owner);
  uint8_t opponent_num = (owner_player_num == 0) ? 1 : 0;
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_entity_t player_garden = gs->zones[owner_player_num].garden;
  ecs_entity_t opponent_garden = gs->zones[opponent_num].garden;

  // Allocate context (will be freed in cleanup)
  Stt02012ObserverCtx *ctx = ecs_os_malloc(sizeof(Stt02012ObserverCtx));
  ctx->card = card;
  ctx->owner_player_num = owner_player_num;

  // Observer 1: Watch for entities entering/leaving the player's garden
  // Use ecs_id(CardId) to match all cards, filter by TEntity in callback
  ecs_entity_t obs_player = ecs_observer(
      world,
      {.query.terms = {{.id = ecs_pair(EcsChildOf, player_garden)},
                       {.id = ecs_id(CardId)}},
       .events = {EcsOnAdd, EcsOnRemove},
       .callback = stt02_012_player_garden_observer,
       .ctx = ctx});

  // Observer 2: Watch for entities entering/leaving the opponent's garden
  ecs_entity_t obs_opponent = ecs_observer(
      world,
      {.query.terms = {{.id = ecs_pair(EcsChildOf, opponent_garden)},
                       {.id = ecs_id(CardId)}},
       .events = {EcsOnAdd, EcsOnRemove},
       .callback = stt02_012_opponent_garden_observer,
       .ctx = ctx});

  // Store observer IDs and context in PassiveObserverContext for cleanup
  ecs_set(world, card, PassiveObserverContext,
          {.observers = {obs_player, obs_opponent, 0, 0},
           .observer_count = 2,
           .ctx = ctx});

  cli_render_logf("[STT02-012] Initialized observers for card %lu "
                  "(player_garden=%lu, opponent_garden=%lu)",
                  (unsigned long)card, (unsigned long)player_garden,
                  (unsigned long)opponent_garden);

  // Note: Initial state check happens when the card itself enters the garden
  // via the player_garden observer
}

void stt02_012_cleanup_passive_observers(ecs_world_t *world,
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

  // Remove any health buff from this card's passive effect
  if (ecs_has_pair(world, card, ecs_id(HealthBuff), card)) {
    remove_health_modifier(world, card, card);
  }

  // Remove the PassiveObserverContext component
  ecs_remove(world, card, PassiveObserverContext);

  cli_render_logf("[STT02-012] Cleaned up observers for card");
}
