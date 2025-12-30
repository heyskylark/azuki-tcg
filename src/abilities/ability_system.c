#include "abilities/ability_system.h"

#include "abilities/ability_registry.h"
#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

// Forward declaration of timing tag constant
#define TIMING_TAG_ON_PLAY_FWD 0

bool azk_trigger_on_play_ability(ecs_world_t *world, ecs_entity_t card,
                                 ecs_entity_t owner) {
  // Get card ID
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id) {
    return false;
  }

  // Check if card has an ability
  if (!azk_has_ability(card_id->id)) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def || !def->has_ability) {
    return false;
  }

  // Check if it's an OnPlay ability
  if (def->timing_tag != ecs_id(AOnPlay)) {
    return false;
  }

  // Queue the effect for processing on next loop iteration
  // This is necessary because during ecs_progress(), zone changes (ChildOf)
  // are deferred and not visible yet. By queuing, we ensure the card is
  // in the correct zone when validation runs.
  return azk_queue_triggered_effect(world, card, owner, TIMING_TAG_ON_PLAY_FWD);
}

bool azk_process_ability_confirmation(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_CONFIRMATION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Move to next phase based on requirements
  if (def->cost_req.min > 0) {
    ctx->phase = ABILITY_PHASE_COST_SELECTION;
    cli_render_logf("[Ability] Confirmed, selecting cost targets");
  } else if (def->on_cost_paid) {
    // No cost targets to select, but has on_cost_paid callback
    // (e.g., STT02-003 which has selection phase without cost)
    if (def->apply_costs) {
      def->apply_costs(world, ctx);
      cli_render_logf("[Ability] Applied costs (no cost targets needed)");
    }
    def->on_cost_paid(world, ctx);
    cli_render_logf("[Ability] Called on_cost_paid callback");
    // on_cost_paid sets the next phase (SELECTION_PICK or BOTTOM_DECK)
    // Check if we're done or need further processing
    if (ctx->phase == ABILITY_PHASE_NONE) {
      azk_clear_ability_context(world);
      return true;
    }
  } else if (def->effect_req.min > 0) {
    // No cost targets to select, but still apply costs (e.g., sacrifice self,
    // draw cards)
    if (def->apply_costs) {
      def->apply_costs(world, ctx);
      cli_render_logf("[Ability] Applied costs (no cost targets needed)");
    }
    ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
    cli_render_logf("[Ability] Confirmed, selecting effect targets");
  } else {
    // No targets needed - apply costs and effects immediately
    if (def->apply_costs) {
      def->apply_costs(world, ctx);
    }
    if (def->apply_effects) {
      def->apply_effects(world, ctx);
    }
    azk_clear_ability_context(world);
    cli_render_logf("[Ability] Confirmed and applied ability with no targets");
    return true;
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_ability_decline(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_CONFIRMATION) {
    return false;
  }

  if (!ctx->is_optional) {
    // Can't decline non-optional abilities
    return false;
  }

  cli_render_logf("[Ability] Declined optional ability");
  azk_clear_ability_context(world);
  return true;
}

bool azk_process_cost_selection(ecs_world_t *world, int target_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_COST_SELECTION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
  if (!card_id) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    return false;
  }

  // Get the target entity based on cost type
  ecs_entity_t target = 0;
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);

  switch (def->cost_req.type) {
  case ABILITY_TARGET_FRIENDLY_HAND:
  case ABILITY_TARGET_FRIENDLY_HAND_WEAPON: {
    ecs_entity_t hand = gs->zones[player_num].hand;
    ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
    if (target_index >= 0 && target_index < hand_cards.count) {
      target = hand_cards.ids[target_index];
    }
    break;
  }
  case ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY: {
    ecs_entity_t garden = gs->zones[player_num].garden;
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
    for (int i = 0; i < garden_cards.count; i++) {
      const ZoneIndex *zi = ecs_get(world, garden_cards.ids[i], ZoneIndex);
      if (zi && zi->index == target_index) {
        target = garden_cards.ids[i];
        break;
      }
    }
    break;
  }
  default:
    break;
  }

  if (target == 0) {
    cli_render_logf("[Ability] Invalid cost target index %d", target_index);
    return false;
  }

  // Validate the target
  if (def->validate_cost_target &&
      !def->validate_cost_target(world, ctx->source_card, ctx->owner, target)) {
    cli_render_logf("[Ability] Cost target validation failed");
    return false;
  }

  // Add target to context
  if (ctx->cost_filled >= MAX_ABILITY_SELECTION) {
    cli_render_logf("[Ability] Too many cost targets");
    return false;
  }

  ctx->cost_targets[ctx->cost_filled] = target;
  ctx->cost_filled++;

  cli_render_logf("[Ability] Added cost target %d (%d/%d)", target_index,
                  ctx->cost_filled, ctx->cost_expected);

  // Check if we have enough targets
  if (ctx->cost_filled >= ctx->cost_expected) {
    // Apply costs
    if (def->apply_costs) {
      def->apply_costs(world, ctx);
      cli_render_logf("[Ability] Applied costs");
    }

    // Call on_cost_paid callback if defined (for multi-step abilities)
    if (def->on_cost_paid) {
      def->on_cost_paid(world, ctx);
      cli_render_logf("[Ability] Called on_cost_paid callback");
      // on_cost_paid may have set up selection phase - check if we should
      // continue
      if (ctx->phase == ABILITY_PHASE_SELECTION_PICK ||
          ctx->phase == ABILITY_PHASE_BOTTOM_DECK) {
        ecs_singleton_modified(world, AbilityContext);
        return true;
      }
    }

    // Move to effect selection or apply effects
    // Use max > 0 (not min > 0) to enter effect selection for "up to" effects
    if (def->effect_req.max > 0) {
      ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
      cli_render_logf("[Ability] Moving to effect selection");
    } else {
      // No effect targets possible - apply effects and finish
      if (def->apply_effects) {
        def->apply_effects(world, ctx);
        cli_render_logf("[Ability] Applied effects");
      }
      azk_clear_ability_context(world);
      return true;
    }
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_effect_selection(ecs_world_t *world, int target_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_EFFECT_SELECTION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
  if (!card_id) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    return false;
  }

  // Get the target entity based on effect type
  ecs_entity_t target = 0;
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);

  switch (def->effect_req.type) {
  case ABILITY_TARGET_FRIENDLY_HAND: {
    ecs_entity_t hand = gs->zones[player_num].hand;
    ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
    if (target_index >= 0 && target_index < hand_cards.count) {
      target = hand_cards.ids[target_index];
    }
    break;
  }
  case ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY: {
    ecs_entity_t garden = gs->zones[player_num].garden;
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
    for (int i = 0; i < garden_cards.count; i++) {
      const ZoneIndex *zi = ecs_get(world, garden_cards.ids[i], ZoneIndex);
      if (zi && zi->index == target_index) {
        target = garden_cards.ids[i];
        break;
      }
    }
    break;
  }
  case ABILITY_TARGET_ENEMY_GARDEN_ENTITY: {
    uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
    ecs_entity_t garden = gs->zones[enemy_num].garden;
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
    if (target_index >= 0 && target_index < garden_cards.count) {
      target = garden_cards.ids[target_index];
    }
    break;
  }
  case ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY: {
    // Index encoding: 0-4 = opponent garden slots (by ZoneIndex), 5 = opponent leader
    uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
    if (target_index < GARDEN_SIZE) {
      // Garden entity by zone index
      ecs_entity_t garden = gs->zones[enemy_num].garden;
      ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
      for (int i = 0; i < garden_cards.count; i++) {
        const ZoneIndex *zi = ecs_get(world, garden_cards.ids[i], ZoneIndex);
        if (zi && zi->index == target_index) {
          target = garden_cards.ids[i];
          break;
        }
      }
    } else if (target_index == GARDEN_SIZE) {
      // Leader
      target = find_leader_card_in_zone(world, gs->zones[enemy_num].leader);
    }
    break;
  }
  case ABILITY_TARGET_ANY_GARDEN_ENTITY: {
    // Index encoding: 0-4 = self garden, 5-9 = opponent garden
    uint8_t target_player_num;
    int garden_index;
    if (target_index < GARDEN_SIZE) {
      target_player_num = player_num;
      garden_index = target_index;
    } else {
      target_player_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
      garden_index = target_index - GARDEN_SIZE;
    }
    ecs_entity_t garden = gs->zones[target_player_num].garden;
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
    // Find card at the specific zone index
    for (int i = 0; i < garden_cards.count; i++) {
      const ZoneIndex *zi = ecs_get(world, garden_cards.ids[i], ZoneIndex);
      if (zi && zi->index == garden_index) {
        target = garden_cards.ids[i];
        break;
      }
    }
    break;
  }
  default:
    break;
  }

  if (target == 0) {
    cli_render_logf("[Ability] Invalid effect target index %d", target_index);
    return false;
  }

  // Validate the target
  if (def->validate_effect_target &&
      !def->validate_effect_target(world, ctx->source_card, ctx->owner,
                                   target)) {
    cli_render_logf("[Ability] Effect target validation failed");
    return false;
  }

  // Add target to context
  if (ctx->effect_filled >= MAX_ABILITY_SELECTION) {
    cli_render_logf("[Ability] Too many effect targets");
    return false;
  }

  ctx->effect_targets[ctx->effect_filled] = target;
  ctx->effect_filled++;

  cli_render_logf("[Ability] Added effect target %d (%d/%d)", target_index,
                  ctx->effect_filled, ctx->effect_expected);

  // Check if we have enough targets
  if (ctx->effect_filled >= ctx->effect_expected) {
    // Apply effects and finish
    if (def->apply_effects) {
      def->apply_effects(world, ctx);
      cli_render_logf("[Ability] Applied effects");
    }
    azk_clear_ability_context(world);
  } else {
    ecs_singleton_modified(world, AbilityContext);
  }

  return true;
}

bool azk_process_effect_skip(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_EFFECT_SELECTION) {
    return false;
  }

  // Can only skip if minimum is 0 ("up to" effects)
  if (ctx->effect_min > 0) {
    cli_render_logf(
        "[Ability] Cannot skip effect selection - minimum targets required");
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Apply effects with no targets (effect_filled == 0)
  if (def->apply_effects) {
    def->apply_effects(world, ctx);
    cli_render_logf("[Ability] Applied effects (skipped target selection)");
  }

  azk_clear_ability_context(world);
  return true;
}

bool azk_process_selection_pick(ecs_world_t *world, int selection_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // Validate index is in range
  if (selection_index < 0 || selection_index >= ctx->selection_count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection_count);
    return false;
  }

  ecs_entity_t target = ctx->selection_cards[selection_index];
  if (target == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Validate the selection target if validation function exists
  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->source_card, ctx->owner,
                                      target)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  // Store the picked card in effect_targets (reusing the array)
  if (ctx->selection_picked < MAX_ABILITY_SELECTION) {
    ctx->effect_targets[ctx->selection_picked] = target;
  }
  ctx->selection_picked++;

  // Mark this slot as picked by setting to 0
  ctx->selection_cards[selection_index] = 0;

  cli_render_logf("[Ability] Picked selection %d (%d/%d)", selection_index,
                  ctx->selection_picked, ctx->selection_pick_max);

  // Check if we've picked enough
  if (ctx->selection_picked >= ctx->selection_pick_max) {
    // Call on_selection_complete callback
    if (def->on_selection_complete) {
      def->on_selection_complete(world, ctx);
      cli_render_logf("[Ability] Called on_selection_complete callback");
    }

    // After selection complete, should be in BOTTOM_DECK or done
    if (ctx->phase != ABILITY_PHASE_BOTTOM_DECK &&
        ctx->phase != ABILITY_PHASE_NONE) {
      // Move to bottom deck phase if there are remaining cards
      int remaining = 0;
      for (int i = 0; i < ctx->selection_count; i++) {
        if (ctx->selection_cards[i] != 0) {
          remaining++;
        }
      }
      if (remaining > 0) {
        ctx->phase = ABILITY_PHASE_BOTTOM_DECK;
      } else {
        azk_clear_ability_context(world);
        return true;
      }
    }
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_selection_to_alley(ecs_world_t *world, int selection_index,
                                    int alley_slot_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // Validate selection index is in range
  if (selection_index < 0 || selection_index >= ctx->selection_count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection_count);
    return false;
  }

  // Validate alley slot index is in range
  if (alley_slot_index < 0 || alley_slot_index >= ALLEY_SIZE) {
    cli_render_logf("[Ability] Invalid alley slot index %d", alley_slot_index);
    return false;
  }

  ecs_entity_t target = ctx->selection_cards[selection_index];
  if (target == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Verify the ability allows selecting to alley
  if (!def->can_select_to_alley) {
    cli_render_logf("[Ability] This ability does not allow selecting to alley");
    return false;
  }

  // Verify the target is an entity card
  const Type *target_type = ecs_get(world, target, Type);
  if (!target_type || target_type->value != CARD_TYPE_ENTITY) {
    cli_render_logf("[Ability] Only entity cards can be selected to alley");
    return false;
  }

  // Validate the selection target if validation function exists
  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->source_card, ctx->owner,
                                      target)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  // Get game state and zones
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t alley = gs->zones[player_num].alley;

  // Check for displaced card at the target slot
  ecs_entity_t displaced_card = 0;
  ecs_entities_t alley_cards = ecs_get_ordered_children(world, alley);
  for (int32_t i = 0; i < alley_cards.count; i++) {
    const ZoneIndex *zi = ecs_get(world, alley_cards.ids[i], ZoneIndex);
    if (zi && zi->index == alley_slot_index) {
      displaced_card = alley_cards.ids[i];
      break;
    }
  }

  // Handle displaced card - only allow replacement if alley is full
  if (displaced_card != 0) {
    bool alley_full = alley_cards.count >= ALLEY_SIZE;
    if (!alley_full) {
      // Reject: slot occupied but alley has empty slots available
      cli_render_logf("[Ability] Alley slot %d is already occupied",
                      alley_slot_index);
      return false;
    }
    // Alley is full - forced replacement allowed
    discard_card(world, displaced_card);
    cli_render_logf("[Ability] Displaced card from alley slot %d",
                    alley_slot_index);
  }

  // Move the selected card to the alley slot
  ecs_add_pair(world, target, EcsChildOf, alley);
  ecs_set(world, target, ZoneIndex, {.index = (uint8_t)alley_slot_index});

  // Reset tap state for newly placed card
  ecs_set(world, target, TapState, {.tapped = false, .cooldown = false});

  // Track that an entity was played to alley this turn (for abilities like
  // STT02-005)
  GameState *gs_mut = ecs_singleton_get_mut(world, GameState);
  gs_mut->entities_played_alley_this_turn[player_num]++;
  ecs_singleton_modified(world, GameState);

  // Queue on-play ability for the played entity (if it has one)
  // Will be processed after current ability completes (including bottom deck)
  azk_trigger_on_play_ability(world, target, ctx->owner);

  cli_render_logf("[Ability] Selected card to alley slot %d", alley_slot_index);

  // Don't store in effect_targets - the card is already moved to alley.
  // Storing it would cause on_selection_complete to incorrectly move it to hand
  // due to Flecs deferred operations (parent check sees old value).
  // Just increment the pick count.
  ctx->selection_picked++;

  // Mark this slot as picked by setting to 0
  ctx->selection_cards[selection_index] = 0;

  // Check if we've picked enough
  if (ctx->selection_picked >= ctx->selection_pick_max) {
    // Call on_selection_complete callback
    if (def->on_selection_complete) {
      def->on_selection_complete(world, ctx);
      cli_render_logf("[Ability] Called on_selection_complete callback");
    }

    // After selection complete, should be in BOTTOM_DECK or done
    if (ctx->phase != ABILITY_PHASE_BOTTOM_DECK &&
        ctx->phase != ABILITY_PHASE_NONE) {
      // Move to bottom deck phase if there are remaining cards
      int remaining = 0;
      for (int i = 0; i < ctx->selection_count; i++) {
        if (ctx->selection_cards[i] != 0) {
          remaining++;
        }
      }
      if (remaining > 0) {
        ctx->phase = ABILITY_PHASE_BOTTOM_DECK;
      } else {
        azk_clear_ability_context(world);
        return true;
      }
    }
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_skip_selection(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // For "up to" effects - allow skipping even if we haven't picked any
  // This is different from effect selection where min determines if we can skip

  const CardId *card_id = ecs_get(world, ctx->source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  cli_render_logf("[Ability] Skipped selection pick");

  // Call on_selection_complete callback (even with no picks)
  if (def->on_selection_complete) {
    def->on_selection_complete(world, ctx);
    cli_render_logf("[Ability] Called on_selection_complete callback");
  }

  // Check if there are remaining cards to bottom deck
  int remaining = 0;
  for (int i = 0; i < ctx->selection_count; i++) {
    if (ctx->selection_cards[i] != 0) {
      remaining++;
    }
  }

  if (remaining > 0) {
    ctx->phase = ABILITY_PHASE_BOTTOM_DECK;
    ecs_singleton_modified(world, AbilityContext);
  } else {
    azk_clear_ability_context(world);
  }

  return true;
}

bool azk_process_bottom_deck(ecs_world_t *world, int selection_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_BOTTOM_DECK) {
    return false;
  }

  // Validate index is in range
  if (selection_index < 0 || selection_index >= ctx->selection_count) {
    cli_render_logf("[Ability] Invalid bottom deck index %d", selection_index);
    return false;
  }

  ecs_entity_t card = ctx->selection_cards[selection_index];
  if (card == 0) {
    cli_render_logf("[Ability] Selection slot %d already empty",
                    selection_index);
    return false;
  }

  // Get player for deck access
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t deck = gs->zones[player_num].deck;

  // Move card from selection zone to bottom of deck
  ecs_add_pair(world, card, EcsChildOf, deck);

  // Reorder to put at bottom (position 0)
  ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
  int32_t count = deck_cards.count;

  if (count > 1) {
    ecs_entity_t *new_order = ecs_os_malloc_n(ecs_entity_t, count);
    int card_idx = -1;
    for (int32_t i = 0; i < count; i++) {
      if (deck_cards.ids[i] == card) {
        card_idx = i;
        break;
      }
    }

    if (card_idx >= 0) {
      new_order[0] = card;
      int dest = 1;
      for (int32_t i = 0; i < count; i++) {
        if (i != card_idx) {
          new_order[dest++] = deck_cards.ids[i];
        }
      }
      ecs_set_child_order(world, deck, new_order, count);
    }
    ecs_os_free(new_order);
  }

  // Mark slot as empty
  ctx->selection_cards[selection_index] = 0;

  cli_render_logf("[Ability] Bottom decked card from slot %d", selection_index);

  // Check if there are remaining cards
  int remaining = 0;
  for (int i = 0; i < ctx->selection_count; i++) {
    if (ctx->selection_cards[i] != 0) {
      remaining++;
    }
  }

  if (remaining == 0) {
    cli_render_logf("[Ability] All cards bottom decked, ability complete");
    azk_clear_ability_context(world);
  } else {
    ecs_singleton_modified(world, AbilityContext);
  }

  return true;
}

bool azk_process_bottom_deck_all(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->phase != ABILITY_PHASE_BOTTOM_DECK) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t deck = gs->zones[player_num].deck;

  // Bottom deck all remaining cards in order (0, 1, 2, ...)
  for (int i = 0; i < ctx->selection_count; i++) {
    ecs_entity_t card = ctx->selection_cards[i];
    if (card == 0) {
      continue;
    }

    // Move card to deck
    ecs_add_pair(world, card, EcsChildOf, deck);

    // Reorder to put at bottom
    ecs_entities_t deck_cards = ecs_get_ordered_children(world, deck);
    int32_t count = deck_cards.count;

    if (count > 1) {
      ecs_entity_t *new_order = ecs_os_malloc_n(ecs_entity_t, count);
      int card_idx = -1;
      for (int32_t j = 0; j < count; j++) {
        if (deck_cards.ids[j] == card) {
          card_idx = j;
          break;
        }
      }

      if (card_idx >= 0) {
        new_order[0] = card;
        int dest = 1;
        for (int32_t j = 0; j < count; j++) {
          if (j != card_idx) {
            new_order[dest++] = deck_cards.ids[j];
          }
        }
        ecs_set_child_order(world, deck, new_order, count);
      }
      ecs_os_free(new_order);
    }

    ctx->selection_cards[i] = 0;
  }

  cli_render_logf(
      "[Ability] Bottom decked all remaining cards, ability complete");
  azk_clear_ability_context(world);
  return true;
}

bool azk_is_in_ability_phase(ecs_world_t *world) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return ctx && ctx->phase != ABILITY_PHASE_NONE;
}

AbilityPhase azk_get_ability_phase(ecs_world_t *world) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return ctx ? ctx->phase : ABILITY_PHASE_NONE;
}

void azk_clear_ability_context(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  ctx->phase = ABILITY_PHASE_NONE;
  ctx->source_card = 0;
  ctx->owner = 0;
  ctx->is_optional = false;
  ctx->cost_min = 0;
  ctx->cost_expected = 0;
  ctx->cost_filled = 0;
  ctx->effect_min = 0;
  ctx->effect_expected = 0;
  ctx->effect_filled = 0;

  for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
    ctx->cost_targets[i] = 0;
    ctx->effect_targets[i] = 0;
  }

  // Clear selection zone tracking
  ctx->selection_count = 0;
  ctx->selection_picked = 0;
  ctx->selection_pick_max = 0;
  for (int i = 0; i < MAX_SELECTION_ZONE_SIZE; i++) {
    ctx->selection_cards[i] = 0;
  }

  ecs_singleton_modified(world, AbilityContext);
}

bool azk_trigger_main_ability(ecs_world_t *world, ecs_entity_t card,
                              ecs_entity_t owner) {
  // Check if card is frozen (frozen cards cannot activate abilities)
  if (ecs_has(world, card, Frozen)) {
    cli_render_logf("[Ability] Card is frozen and cannot activate abilities");
    return false;
  }

  // Get card ID
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id) {
    return false;
  }

  // Check if card has an ability
  if (!azk_has_ability(card_id->id)) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def || !def->has_ability) {
    return false;
  }

  // Check if it's a Main phase ability
  if (def->timing_tag != ecs_id(AMain)) {
    return false;
  }

  // Validate the ability can be activated
  if (def->validate && !def->validate(world, card, owner)) {
    cli_render_logf("[Ability] Main ability validation failed");
    return false;
  }

  // Get ability context singleton
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  // Set up the ability context
  ctx->source_card = card;
  ctx->owner = owner;
  ctx->is_optional = def->is_optional;
  ctx->cost_min = def->cost_req.min;
  ctx->cost_expected = def->cost_req.max;
  ctx->cost_filled = 0;
  ctx->effect_min = def->effect_req.min;
  ctx->effect_expected = def->effect_req.max;
  ctx->effect_filled = 0;

  // Clear target arrays
  for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
    ctx->cost_targets[i] = 0;
    ctx->effect_targets[i] = 0;
  }

  // Main abilities are triggered by player action, so skip confirmation phase
  // (player already opted in by taking the action)
  if (def->cost_req.min > 0) {
    ctx->phase = ABILITY_PHASE_COST_SELECTION;
    cli_render_logf("[Ability] Triggered main ability, selecting cost targets");
  } else if (def->effect_req.min > 0) {
    // No cost targets needed, apply costs (if any) and go to effect selection
    if (def->apply_costs) {
      def->apply_costs(world, ctx);
    }
    ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
    cli_render_logf(
        "[Ability] Triggered main ability, selecting effect targets");
  } else {
    // No targets needed - apply immediately
    if (def->apply_costs) {
      def->apply_costs(world, ctx);
    }
    if (def->apply_effects) {
      def->apply_effects(world, ctx);
    }
    ctx->phase = ABILITY_PHASE_NONE;
    cli_render_logf("[Ability] Applied main ability with no targets");
  }
  ecs_singleton_modified(world, AbilityContext);
  return ctx->phase != ABILITY_PHASE_NONE;
}

bool azk_trigger_spell_ability(ecs_world_t *world, ecs_entity_t spell_card,
                               ecs_entity_t owner) {
  // Get card ID
  const CardId *card_id = ecs_get(world, spell_card, CardId);
  if (!card_id) {
    return false;
  }

  // Check if card has an ability
  if (!azk_has_ability(card_id->id)) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def || !def->has_ability) {
    return false;
  }

  // Get ability context singleton
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  // Set up the ability context
  ctx->source_card = spell_card;
  ctx->owner = owner;
  ctx->is_optional = false; // Spells are already cast, not optional
  ctx->cost_min = def->cost_req.min;
  ctx->cost_expected = def->cost_req.max;
  ctx->cost_filled = 0;
  ctx->effect_min = def->effect_req.min;
  ctx->effect_expected = def->effect_req.max;
  ctx->effect_filled = 0;

  // Clear target arrays
  for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
    ctx->cost_targets[i] = 0;
    ctx->effect_targets[i] = 0;
  }

  // Spells skip confirmation phase - go straight to cost or effect selection
  if (def->cost_req.min > 0) {
    ctx->phase = ABILITY_PHASE_COST_SELECTION;
    cli_render_logf("[Ability] Spell triggered, selecting cost targets");
  } else if (def->effect_req.min > 0) {
    ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
    cli_render_logf("[Ability] Spell triggered, selecting effect targets");
  } else {
    // No targets needed - apply immediately
    if (def->apply_effects) {
      def->apply_effects(world, ctx);
    }
    ctx->phase = ABILITY_PHASE_NONE;
    cli_render_logf("[Ability] Applied spell with no targets");
    return false; // No further action required
  }

  ecs_singleton_modified(world, AbilityContext);
  return ctx->phase != ABILITY_PHASE_NONE;
}

bool azk_trigger_leader_response_ability(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner) {
  // Get card ID
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id) {
    return false;
  }

  // Check if card has an ability
  if (!azk_has_ability(card_id->id)) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def || !def->has_ability) {
    return false;
  }

  // Get ability context singleton
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  // Set up the ability context
  ctx->source_card = card;
  ctx->owner = owner;
  ctx->is_optional = false; // Response abilities are already activated
  ctx->cost_min = def->cost_req.min;
  ctx->cost_expected = def->cost_req.max;
  ctx->cost_filled = 0;
  ctx->effect_min = def->effect_req.min;
  ctx->effect_expected = def->effect_req.max;
  ctx->effect_filled = 0;

  // Clear target arrays
  for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
    ctx->cost_targets[i] = 0;
    ctx->effect_targets[i] = 0;
  }

  // Leader response abilities skip confirmation (already activated)
  // Go straight to cost or effect selection
  if (def->cost_req.min > 0) {
    ctx->phase = ABILITY_PHASE_COST_SELECTION;
    cli_render_logf("[Ability] Leader response triggered, selecting cost targets");
  } else if (def->effect_req.min > 0) {
    ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
    cli_render_logf("[Ability] Leader response triggered, selecting effect targets");
  } else {
    // No targets needed - apply immediately
    if (def->apply_effects) {
      def->apply_effects(world, ctx);
    }
    ctx->phase = ABILITY_PHASE_NONE;
    cli_render_logf("[Ability] Applied leader response with no targets");
    return false; // No further action required
  }

  ecs_singleton_modified(world, AbilityContext);
  return ctx->phase != ABILITY_PHASE_NONE;
}

// Timing tag constants for queue indexing
#define TIMING_TAG_ON_PLAY 0
#define TIMING_TAG_START_OF_TURN 1
#define TIMING_TAG_END_OF_TURN 2
#define TIMING_TAG_WHEN_EQUIPPING 3
#define TIMING_TAG_WHEN_EQUIPPED 4
#define TIMING_TAG_WHEN_ATTACKING 5
#define TIMING_TAG_WHEN_ATTACKED 6
#define TIMING_TAG_WHEN_RETURNED_TO_HAND 7
#define TIMING_TAG_ON_GATE_PORTAL 8

bool azk_queue_triggered_effect(ecs_world_t *world, ecs_entity_t card,
                                ecs_entity_t owner, uint8_t timing_tag) {
  TriggeredEffectQueue *queue =
      ecs_singleton_get_mut(world, TriggeredEffectQueue);

  if (queue->count >= MAX_TRIGGERED_EFFECT_QUEUE) {
    cli_render_logf("[Ability] Triggered effect queue full, cannot queue");
    return false;
  }

  queue->effects[queue->count].source_card = card;
  queue->effects[queue->count].owner = owner;
  queue->effects[queue->count].timing_tag = timing_tag;
  queue->count++;

  cli_render_logf("[Ability] Queued triggered effect (tag=%d, count=%d)",
                  timing_tag, queue->count);
  ecs_singleton_modified(world, TriggeredEffectQueue);
  return true;
}

bool azk_has_queued_triggered_effects(ecs_world_t *world) {
  const TriggeredEffectQueue *queue =
      ecs_singleton_get(world, TriggeredEffectQueue);
  return queue && queue->count > 0;
}

// Helper to get the expected timing tag ecs_id for a given tag index
static ecs_id_t get_timing_tag_id(uint8_t tag_index) {
  switch (tag_index) {
  case TIMING_TAG_ON_PLAY:
    return ecs_id(AOnPlay);
  case TIMING_TAG_START_OF_TURN:
    return ecs_id(AStartOfTurn);
  case TIMING_TAG_END_OF_TURN:
    return ecs_id(AEndOfTurn);
  case TIMING_TAG_WHEN_EQUIPPING:
    return ecs_id(AWhenEquipping);
  case TIMING_TAG_WHEN_EQUIPPED:
    return ecs_id(AWhenEquipped);
  case TIMING_TAG_WHEN_ATTACKING:
    return ecs_id(AWhenAttacking);
  case TIMING_TAG_WHEN_ATTACKED:
    return ecs_id(AWhenAttacked);
  case TIMING_TAG_WHEN_RETURNED_TO_HAND:
    return ecs_id(AWhenReturnedToHand);
  case TIMING_TAG_ON_GATE_PORTAL:
    return ecs_id(AOnGatePortal);
  default:
    return 0;
  }
}

bool azk_process_triggered_effect_queue(ecs_world_t *world) {
  TriggeredEffectQueue *queue =
      ecs_singleton_get_mut(world, TriggeredEffectQueue);

  if (!queue || queue->count == 0) {
    return false;
  }

  // Pop first effect (FIFO)
  PendingTriggeredEffect effect = queue->effects[0];

  // Shift remaining effects
  for (uint8_t i = 0; i < queue->count - 1; i++) {
    queue->effects[i] = queue->effects[i + 1];
  }
  queue->count--;
  ecs_singleton_modified(world, TriggeredEffectQueue);

  cli_render_logf("[Ability] Processing queued effect (tag=%d, remaining=%d)",
                  effect.timing_tag, queue->count);

  // Now process the effect - card should be in correct zone after deferred ops
  // flushed
  ecs_entity_t card = effect.source_card;
  ecs_entity_t owner = effect.owner;

  // Switch active player to ability owner if different
  // This ensures the correct player has control to confirm/decline
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  uint8_t owner_player_num = get_player_number(world, owner);
  if (gs->active_player_index != owner_player_num) {
    cli_render_logf(
        "[Ability] Switching control to player %d for triggered ability",
        owner_player_num);
    gs->active_player_index = owner_player_num;
    ecs_singleton_modified(world, GameState);
  }

  // Get card ID
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id) {
    cli_render_logf("[Ability] Queued effect: card no longer valid");
    return false;
  }

  // Check if card has an ability
  if (!azk_has_ability(card_id->id)) {
    cli_render_logf("[Ability] Queued effect: card has no ability");
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def || !def->has_ability) {
    cli_render_logf("[Ability] Queued effect: no ability definition");
    return false;
  }

  // Check if it's the correct timing tag
  ecs_id_t expected_tag = get_timing_tag_id(effect.timing_tag);
  if (def->timing_tag != expected_tag) {
    cli_render_logf("[Ability] Queued effect: timing tag mismatch");
    return false;
  }

  // Validate the ability can be activated (card is now in correct zone)
  if (def->validate && !def->validate(world, card, owner)) {
    cli_render_logf("[Ability] Queued effect: validation failed");
    return false;
  }

  // Get ability context singleton
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  // Set up the ability context
  ctx->source_card = card;
  ctx->owner = owner;
  ctx->is_optional = def->is_optional;
  ctx->cost_min = def->cost_req.min;
  ctx->cost_expected = def->cost_req.max;
  ctx->cost_filled = 0;
  ctx->effect_min = def->effect_req.min;
  ctx->effect_expected = def->effect_req.max;
  ctx->effect_filled = 0;

  // Clear target arrays
  for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
    ctx->cost_targets[i] = 0;
    ctx->effect_targets[i] = 0;
  }

  if (def->is_optional) {
    // Optional ability - enter confirmation phase
    ctx->phase = ABILITY_PHASE_CONFIRMATION;
    cli_render_logf(
        "[Ability] Triggered optional ability, waiting for confirmation");
    ecs_singleton_modified(world, AbilityContext);
    return true;
  } else {
    // Non-optional ability - skip confirmation, go straight to cost selection
    if (def->cost_req.min > 0) {
      ctx->phase = ABILITY_PHASE_COST_SELECTION;
      cli_render_logf(
          "[Ability] Triggered mandatory ability, selecting cost targets");
    } else if (def->effect_req.min > 0) {
      ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
      cli_render_logf(
          "[Ability] Triggered mandatory ability, selecting effect targets");
    } else {
      // No targets needed - apply immediately
      if (def->apply_effects) {
        def->apply_effects(world, ctx);
      }
      ctx->phase = ABILITY_PHASE_NONE;
      cli_render_logf("[Ability] Applied mandatory ability with no targets");
    }
    ecs_singleton_modified(world, AbilityContext);
    return ctx->phase != ABILITY_PHASE_NONE;
  }
}

void azk_trigger_return_to_hand_observers(ecs_world_t *world,
                                          ecs_entity_t bounced_card) {
  (void)bounced_card; // May be used in future for filtering

  const GameState *gs = ecs_singleton_get(world, GameState);

  // Scan BOTH players' gardens for cards with AWhenReturnedToHand ability
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
    ecs_entity_t player = gs->players[p];
    ecs_entity_t garden = gs->zones[p].garden;
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

    for (int32_t i = 0; i < garden_cards.count; i++) {
      ecs_entity_t card = garden_cards.ids[i];

      // Check if card has AWhenReturnedToHand timing tag
      if (!ecs_has(world, card, AWhenReturnedToHand)) {
        continue;
      }

      // Get card ID and ability def
      const CardId *card_id = ecs_get(world, card, CardId);
      if (!card_id || !azk_has_ability(card_id->id)) {
        continue;
      }

      const AbilityDef *def = azk_get_ability_def(card_id->id);
      if (!def || def->timing_tag != ecs_id(AWhenReturnedToHand)) {
        continue;
      }

      // Validate the ability can be activated
      if (def->validate && !def->validate(world, card, player)) {
        continue;
      }

      // Queue the triggered effect
      azk_queue_triggered_effect(world, card, player,
                                 TIMING_TAG_WHEN_RETURNED_TO_HAND);
      cli_render_logf("[Ability] Queued return-to-hand observer for card %s",
                      ecs_get_name(world, card));
    }
  }
}

void azk_trigger_gate_portal_ability(ecs_world_t *world, ecs_entity_t gate_card,
                                     ecs_entity_t portaled_card,
                                     ecs_entity_t owner) {
  // Verify this is actually a gate card
  ecs_assert(is_card_type(world, gate_card, CARD_TYPE_GATE),
             ECS_INVALID_PARAMETER,
             "Gate portal ability triggered on non-gate card %llu",
             (unsigned long long)gate_card);

  const CardId *card_id = ecs_get(world, gate_card, CardId);
  ecs_assert(card_id != NULL, ECS_INVALID_PARAMETER,
             "Gate card %llu has no CardId component",
             (unsigned long long)gate_card);

  // All gate cards must have a registered ability
  ecs_assert(azk_has_ability(card_id->id), ECS_INVALID_PARAMETER,
             "Gate card %llu (def %d) has no registered ability",
             (unsigned long long)gate_card, card_id->id);

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  ecs_assert(def != NULL && def->timing_tag == ecs_id(AOnGatePortal),
             ECS_INVALID_PARAMETER,
             "Gate card %llu ability has wrong timing tag",
             (unsigned long long)gate_card);

  // Validate can still fail (e.g., conditional effects)
  if (def->validate && !def->validate(world, gate_card, owner)) {
    return;
  }

  // Set up context with portaled card info
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);
  ctx->source_card = gate_card;
  ctx->owner = owner;
  ctx->is_optional = def->is_optional;
  ctx->effect_targets[0] = portaled_card; // Store portaled card for effect
  ctx->effect_filled = 1;

  // Apply effect immediately (non-optional, no targets needed)
  if (def->apply_effects) {
    def->apply_effects(world, ctx);
  }
  ctx->phase = ABILITY_PHASE_NONE;
  ecs_singleton_modified(world, AbilityContext);

  cli_render_logf("[Ability] Applied gate portal ability for %s",
                  ecs_get_name(world, gate_card));
}
