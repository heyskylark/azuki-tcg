#include "abilities/ability_system.h"

#include "abilities/core/ability_context.h"
#include "abilities/core/ability_flow.h"
#include "abilities/core/ability_runtime.h"
#include "abilities/ability_registry.h"
#include "abilities/targeting/ability_targeting.h"
#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/weapon_util.h"
#include "utils/zone_util.h"

// Forward declaration of timing tag constant
#define TIMING_TAG_ON_PLAY_FWD 0

static uint8_t ability_selection_remaining_count(const AbilityContext *ctx) {
  return azk_count_remaining_selection_cards(ctx);
}

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

bool azk_trigger_when_equipped_ability(ecs_world_t *world, ecs_entity_t card,
                                       ecs_entity_t owner) {
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id) {
    return false;
  }

  if (!azk_has_ability(card_id->id)) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def || !def->has_ability) {
    return false;
  }

  // Check if it's an AWhenEquipped ability
  if (def->timing_tag != ecs_id(AWhenEquipped)) {
    return false;
  }

  // Queue for processing after deferred ops (ChildOf) flush
  return azk_queue_triggered_effect(world, card, owner, TIMING_TAG_WHEN_EQUIPPED);
}

bool azk_process_ability_confirmation(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_CONFIRMATION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  if (def->cost_req.min > 0) {
    uint8_t available_cost_targets = azk_count_ability_target_choices(
        world, def, ABILITY_TARGET_SCOPE_COST, ctx->runtime.source_card,
        ctx->runtime.owner);
    if (available_cost_targets < def->cost_req.min) {
      cli_render_logf("[Ability] Confirmed ability has no valid cost targets "
                      "(available=%u, required_min=%u), skipping",
                      (unsigned)available_cost_targets,
                      (unsigned)def->cost_req.min);
      azk_clear_ability_context(world);
      return true;
    }
    if (ctx->cost.max_allowed > available_cost_targets) {
      ctx->cost.max_allowed = available_cost_targets;
    }
  }

  if (!azk_enter_initial_phase(world, ctx, def,
                               &(AbilityInitialPhaseOptions){
                                   .select_effects_when_max_positive = true,
                                   .apply_costs_before_effect_selection = true,
                               })) {
    azk_clear_ability_context(world);
    cli_render_logf("[Ability] Confirmed and applied ability with no targets");
    return true;
  }

  azk_log_ability_initial_phase_entry(
      ctx->runtime.phase, "[Ability] Confirmed, selecting cost targets",
      "[Ability] Confirmed, selecting effect targets",
      "[Ability] Confirmed, started selection flow");

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_ability_decline(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_CONFIRMATION) {
    return false;
  }

  if (!ctx->runtime.is_optional) {
    // Can't decline non-optional abilities
    return false;
  }

  cli_render_logf("[Ability] Declined optional ability");
  azk_clear_ability_context(world);
  return true;
}

bool azk_process_cost_selection(ecs_world_t *world, int target_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_COST_SELECTION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    return false;
  }

  ecs_entity_t target = azk_resolve_ability_target_choice_entity(
      world, def, ABILITY_TARGET_SCOPE_COST, ctx->runtime.owner, target_index);

  if (target == 0) {
    cli_render_logf("[Ability] Invalid cost target index %d", target_index);
    return false;
  }

  // Validate the target
  if (def->validate_cost_target &&
      !def->validate_cost_target(world, ctx->runtime.source_card,
                                 ctx->runtime.owner, target)) {
    cli_render_logf("[Ability] Cost target validation failed");
    return false;
  }

  // Add target to context
  if (ctx->cost.selected_count >= MAX_ABILITY_SELECTION) {
    cli_render_logf("[Ability] Too many cost targets");
    return false;
  }

  ctx->cost.entities[ctx->cost.selected_count] = target;
  ctx->cost.selected_count++;

  cli_render_logf("[Ability] Added cost target %d (%d/%d)", target_index,
                  ctx->cost.selected_count, ctx->cost.max_allowed);

  // Check if we have enough targets
  if (ctx->cost.selected_count >= ctx->cost.max_allowed) {
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
      if (ctx->runtime.phase == ABILITY_PHASE_SELECTION_PICK ||
          ctx->runtime.phase == ABILITY_PHASE_BOTTOM_DECK) {
        ecs_singleton_modified(world, AbilityContext);
        return true;
      }
    }

    // Move to effect selection or apply effects
    // Use max > 0 (not min > 0) to enter effect selection for "up to" effects
    if (def->effect_req.max > 0) {
      ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
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

  if (ctx->runtime.phase != ABILITY_PHASE_EFFECT_SELECTION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    return false;
  }

  ecs_entity_t target = azk_resolve_ability_target_choice_entity(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, ctx->runtime.owner,
      target_index);

  if (target == 0) {
    cli_render_logf("[Ability] Invalid effect target index %d", target_index);
    return false;
  }

  // Validate the target
  if (def->validate_effect_target &&
      !def->validate_effect_target(world, ctx->runtime.source_card,
                                   ctx->runtime.owner, target)) {
    cli_render_logf("[Ability] Effect target validation failed");
    return false;
  }

  // Add target to context
  if (ctx->effect.selected_count >= MAX_ABILITY_SELECTION) {
    cli_render_logf("[Ability] Too many effect targets");
    return false;
  }

  ctx->effect.entities[ctx->effect.selected_count] = target;
  ctx->effect.selected_count++;

  cli_render_logf("[Ability] Added effect target %d (%d/%d)", target_index,
                  ctx->effect.selected_count, ctx->effect.max_allowed);

  // Check if we have enough targets
  if (ctx->effect.selected_count >= ctx->effect.max_allowed) {
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

  if (ctx->runtime.phase != ABILITY_PHASE_EFFECT_SELECTION) {
    return false;
  }

  // Can only skip if minimum is 0 ("up to" effects)
  if (ctx->effect.min_required > 0) {
    cli_render_logf(
        "[Ability] Cannot skip effect selection - minimum targets required");
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Apply effects with no targets (effect.selected_count == 0)
  if (def->apply_effects) {
    def->apply_effects(world, ctx);
    cli_render_logf("[Ability] Applied effects (skipped target selection)");
  }

  azk_clear_ability_context(world);
  return true;
}

bool azk_process_selection_pick(ecs_world_t *world, int selection_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // Validate index is in range
  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection.count);
    return false;
  }

  ecs_entity_t target = ctx->selection.cards[selection_index];
  if (target == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Safety check: verify ability allows adding to hand
  // If ability has special selection modes but can_select_to_hand is false, reject
  bool has_special_selection = def->can_select_to_alley || def->can_select_to_equip;
  if (has_special_selection && !def->can_select_to_hand) {
    cli_render_logf("[Ability] Selection pick not allowed - must use equip or alley action");
    return false;
  }

  // Validate the selection target if validation function exists
  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->runtime.source_card,
                                      ctx->runtime.owner, target)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  // Store hand-bound selection picks separately from effect targets.
  if (ctx->selection.picked_count < MAX_ABILITY_SELECTION) {
    ctx->selection.picked_cards[ctx->selection.picked_count] = target;
  }
  ctx->selection.picked_count++;

  // Mark this slot as picked by setting to 0
  ctx->selection.cards[selection_index] = 0;

  cli_render_logf("[Ability] Picked selection %d (%d/%d)", selection_index,
                  ctx->selection.picked_count, ctx->selection.pick_max);

  // Check if we've picked enough
  if (ctx->selection.picked_count >= ctx->selection.pick_max) {
    // Call on_selection_complete callback
    if (def->on_selection_complete) {
      def->on_selection_complete(world, ctx);
      cli_render_logf("[Ability] Called on_selection_complete callback");
    }

    // After selection complete, should be in BOTTOM_DECK or done
    if (ctx->runtime.phase != ABILITY_PHASE_BOTTOM_DECK &&
        ctx->runtime.phase != ABILITY_PHASE_NONE) {
      // Move to bottom deck phase if there are remaining cards
      uint8_t remaining = ability_selection_remaining_count(ctx);
      if (remaining > 0) {
        ctx->runtime.phase = ABILITY_PHASE_BOTTOM_DECK;
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

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // Validate selection index is in range
  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection.count);
    return false;
  }

  // Validate alley slot index is in range
  if (alley_slot_index < 0 || alley_slot_index >= ALLEY_SIZE) {
    cli_render_logf("[Ability] Invalid alley slot index %d", alley_slot_index);
    return false;
  }

  ecs_entity_t target = ctx->selection.cards[selection_index];
  if (target == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
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
      !def->validate_selection_target(world, ctx->runtime.source_card,
                                      ctx->runtime.owner, target)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  // Get game state and zones
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t alley = gs->zones[player_num].alley;
  ecs_entity_t selection_zone = gs->zones[player_num].selection;

  int8_t from_index =
      azk_get_card_index_in_zone(world, target, selection_zone);

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

  // Log selection -> alley movement
  azk_log_card_zone_moved(world, target, GLOG_ZONE_SELECTION, from_index,
                          GLOG_ZONE_ALLEY, (int8_t)alley_slot_index);

  // Reset tap state for newly placed card
  ecs_set(world, target, TapState, {.tapped = false, .cooldown = false});

  azk_debug_validate_zone_indices(world, alley, ALLEY_SIZE);

  // Track that an entity was played to alley this turn (for abilities like
  // STT02-005)
  GameState *gs_mut = ecs_singleton_get_mut(world, GameState);
  gs_mut->entities_played_alley_this_turn[player_num]++;
  ecs_singleton_modified(world, GameState);

  // Queue on-play ability for the played entity (if it has one)
  // Will be processed after current ability completes (including bottom deck)
  azk_trigger_on_play_ability(world, target, ctx->runtime.owner);

  cli_render_logf("[Ability] Selected card to alley slot %d", alley_slot_index);

  // Preserve pick index progression while keeping hand-bound picks separate.
  if (ctx->selection.picked_count < MAX_ABILITY_SELECTION) {
    ctx->selection.picked_cards[ctx->selection.picked_count] = 0;
  }
  ctx->selection.picked_count++;

  // Mark this slot as picked by setting to 0
  ctx->selection.cards[selection_index] = 0;

  // Check if we've picked enough
  if (ctx->selection.picked_count >= ctx->selection.pick_max) {
    // Call on_selection_complete callback
    if (def->on_selection_complete) {
      def->on_selection_complete(world, ctx);
      cli_render_logf("[Ability] Called on_selection_complete callback");
    }

    // After selection complete, should be in BOTTOM_DECK or done
    if (ctx->runtime.phase != ABILITY_PHASE_BOTTOM_DECK &&
        ctx->runtime.phase != ABILITY_PHASE_NONE) {
      // Move to bottom deck phase if there are remaining cards
      uint8_t remaining = ability_selection_remaining_count(ctx);
      if (remaining > 0) {
        ctx->runtime.phase = ABILITY_PHASE_BOTTOM_DECK;
      } else {
        azk_clear_ability_context(world);
        return true;
      }
    }
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_selection_to_equip(ecs_world_t *world, int selection_index,
                                    int entity_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // Validate selection index is in range
  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection.count);
    return false;
  }

  // Validate entity index (0-4 for garden, 5 for leader)
  if (entity_index < 0 || entity_index > GARDEN_SIZE) {
    cli_render_logf("[Ability] Invalid entity index %d", entity_index);
    return false;
  }

  ecs_entity_t weapon = ctx->selection.cards[selection_index];
  if (weapon == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = azk_get_ability_def(card_id->id);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Verify the ability allows selecting to equip
  if (!def->can_select_to_equip) {
    cli_render_logf("[Ability] This ability does not allow selecting to equip");
    return false;
  }

  // Verify the weapon is a weapon card
  const Type *weapon_type = ecs_get(world, weapon, Type);
  if (!weapon_type || weapon_type->value != CARD_TYPE_WEAPON) {
    cli_render_logf("[Ability] Only weapon cards can be selected to equip");
    return false;
  }

  // Validate the selection target if validation function exists
  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->runtime.source_card,
                                      ctx->runtime.owner, weapon)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  // Get game state and find target entity
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t selection_zone = gs->zones[player_num].selection;
  ecs_entity_t target_entity = 0;

  if (entity_index < GARDEN_SIZE) {
    target_entity = find_card_in_zone_index(world, gs->zones[player_num].garden,
                                            entity_index);
  } else {
    // entity_index == GARDEN_SIZE means leader
    target_entity =
        find_leader_card_in_zone(world, gs->zones[player_num].leader);
  }

  if (target_entity == 0) {
    cli_render_logf("[Ability] No entity at slot %d", entity_index);
    return false;
  }

  // Get weapon and target stats
  const CurStats *weapon_stats = ecs_get(world, weapon, CurStats);
  const CurStats *target_stats = ecs_get(world, target_entity, CurStats);
  if (!weapon_stats || !target_stats) {
    cli_render_logf("[Ability] Missing stats for weapon or target");
    return false;
  }

  int8_t from_index = azk_get_card_index_in_zone(world, weapon, selection_zone);

  // Log selection -> equipped movement before the deferred reparent changes
  // the visible parent chain.
  azk_log_card_zone_moved(world, weapon, GLOG_ZONE_SELECTION, from_index,
                          GLOG_ZONE_EQUIPPED, -1);

  // Attach weapon to target (ChildOf relationship)
  ecs_add_pair(world, weapon, EcsChildOf, target_entity);

  // Apply weapon attack bonus directly (because ChildOf is deferred)
  apply_weapon_attack_bonus(world, target_entity, weapon_stats->cur_atk);

  cli_render_logf("[Ability] Equipped weapon (+%d attack) to entity at slot %d",
                  weapon_stats->cur_atk, entity_index);

  // Trigger weapon abilities (on-play and when-equipped)
  azk_trigger_on_play_ability(world, weapon, ctx->runtime.owner);
  azk_trigger_when_equipped_ability(world, weapon, ctx->runtime.owner);

  // Mark this slot as picked by setting to 0
  ctx->selection.cards[selection_index] = 0;
  if (ctx->selection.picked_count < MAX_ABILITY_SELECTION) {
    ctx->selection.picked_cards[ctx->selection.picked_count] = 0;
  }
  ctx->selection.picked_count++;

  // Check if we've picked enough
  if (ctx->selection.picked_count >= ctx->selection.pick_max) {
    // Call on_selection_complete callback
    if (def->on_selection_complete) {
      def->on_selection_complete(world, ctx);
      cli_render_logf("[Ability] Called on_selection_complete callback");
    }

    // For discard-based selection, no bottom deck phase needed
    // Just clear the ability context
    if (ctx->runtime.phase != ABILITY_PHASE_NONE) {
      azk_clear_ability_context(world);
      return true;
    }
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_skip_selection(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // For "up to" effects - allow skipping even if we haven't picked any
  // This is different from effect selection where min determines if we can skip

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
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
  uint8_t remaining = ability_selection_remaining_count(ctx);

  if (remaining > 0) {
    ctx->runtime.phase = ABILITY_PHASE_BOTTOM_DECK;
    ecs_singleton_modified(world, AbilityContext);
  } else {
    azk_clear_ability_context(world);
  }

  return true;
}

bool azk_process_bottom_deck(ecs_world_t *world, int selection_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_BOTTOM_DECK) {
    return false;
  }

  // Validate index is in range
  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid bottom deck index %d", selection_index);
    return false;
  }

  ecs_entity_t card = ctx->selection.cards[selection_index];
  if (card == 0) {
    cli_render_logf("[Ability] Selection slot %d already empty",
                    selection_index);
    return false;
  }

  // Move card from selection zone to bottom of deck (with log emission)
  move_selection_to_deck_bottom(world, ctx->runtime.owner, card);

  // Mark slot as empty
  ctx->selection.cards[selection_index] = 0;

  cli_render_logf("[Ability] Bottom decked card from slot %d", selection_index);

  // Check if there are remaining cards
  uint8_t remaining = ability_selection_remaining_count(ctx);

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

  if (ctx->runtime.phase != ABILITY_PHASE_BOTTOM_DECK) {
    return false;
  }

  // Bottom deck all remaining cards in order (0, 1, 2, ...)
  for (int i = 0; i < ctx->selection.count; i++) {
    ecs_entity_t card = ctx->selection.cards[i];
    if (card == 0) {
      continue;
    }

    // Move card to bottom of deck (with log emission)
    move_selection_to_deck_bottom(world, ctx->runtime.owner, card);

    ctx->selection.cards[i] = 0;
  }

  cli_render_logf(
      "[Ability] Bottom decked all remaining cards, ability complete");
  azk_clear_ability_context(world);
  return true;
}

bool azk_is_in_ability_phase(ecs_world_t *world) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return ctx && ctx->runtime.phase != ABILITY_PHASE_NONE;
}

AbilityPhase azk_get_ability_phase(ecs_world_t *world) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return ctx ? ctx->runtime.phase : ABILITY_PHASE_NONE;
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

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, card, owner);
  if (def->cost_req.min > 0 && available_cost_targets < def->cost_req.min) {
    cli_render_logf("[Ability] Main ability has no valid cost targets "
                    "(available=%u, required_min=%u)",
                    (unsigned)available_cost_targets,
                    (unsigned)def->cost_req.min);
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

  return azk_begin_ability(
      world, card, owner, def,
      &(AbilityBeginOptions){
          .is_optional = def->is_optional,
          .available_cost_targets = available_cost_targets,
          .select_effects_when_max_positive = false,
          .apply_costs_before_effect_selection = true,
          .applied_log = "[Ability] Applied main ability with no targets",
          .cost_selection_log =
              "[Ability] Triggered main ability, selecting cost targets",
          .effect_selection_log =
              "[Ability] Triggered main ability, selecting effect targets",
          .selection_log =
              "[Ability] Triggered main ability, started selection flow",
      });
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

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, spell_card, owner);
  if (def->cost_req.min > 0 && available_cost_targets < def->cost_req.min) {
    cli_render_logf("[Ability] Spell has no valid cost targets "
                    "(available=%u, required_min=%u)",
                    (unsigned)available_cost_targets,
                    (unsigned)def->cost_req.min);
    return false;
  }

  uint8_t available_effect_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, spell_card, owner);

  return azk_begin_ability(
      world, spell_card, owner, def,
      &(AbilityBeginOptions){
          .is_optional = false,
          .available_cost_targets = available_cost_targets,
          .available_effect_targets = available_effect_targets,
          .clamp_effect_expected_to_available = true,
          .select_effects_when_max_positive = false,
          .apply_costs_before_effect_selection = false,
          .applied_log = "[Ability] Applied spell with no targets",
          .cost_selection_log =
              "[Ability] Spell triggered, selecting cost targets",
          .effect_selection_log =
              "[Ability] Spell triggered, selecting effect targets",
          .selection_log = "[Ability] Spell triggered, started selection flow",
      });
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

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, card, owner);
  if (def->cost_req.min > 0 && available_cost_targets < def->cost_req.min) {
    cli_render_logf("[Ability] Leader response has no valid cost targets "
                    "(available=%u, required_min=%u)",
                    (unsigned)available_cost_targets,
                    (unsigned)def->cost_req.min);
    return false;
  }

  return azk_begin_ability(
      world, card, owner, def,
      &(AbilityBeginOptions){
          .is_optional = false,
          .available_cost_targets = available_cost_targets,
          .select_effects_when_max_positive = false,
          .apply_costs_before_effect_selection = false,
          .applied_log = "[Ability] Applied leader response with no targets",
          .cost_selection_log =
              "[Ability] Leader response triggered, selecting cost targets",
          .effect_selection_log =
              "[Ability] Leader response triggered, selecting effect targets",
          .selection_log =
              "[Ability] Leader response triggered, started selection flow",
      });
}

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

  // Log effect queued (ability_index=0 as default, timing_tag for trigger type)
  azk_log_effect_queued(world, card, 0, timing_tag);

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

  // Log effect enabled (ability is now being processed)
  azk_log_effect_enabled(world, effect.source_card, 0);

  cli_render_logf("[Ability] Processing queued effect (tag=%d, remaining=%d)",
                  effect.timing_tag, queue->count);

  // Now process the effect - card should be in correct zone after deferred ops
  // flushed
  ecs_entity_t card = effect.source_card;
  ecs_entity_t owner = effect.owner;

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

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, card, owner);
  if (def->cost_req.min > 0 && available_cost_targets < def->cost_req.min) {
    cli_render_logf("[Ability] Queued effect has no valid cost targets "
                    "(available=%u, required_min=%u), skipping",
                    (unsigned)available_cost_targets,
                    (unsigned)def->cost_req.min);
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

  return azk_begin_ability(
      world, card, owner, def,
      &(AbilityBeginOptions){
          .is_optional = def->is_optional,
          .enter_confirmation_when_optional = true,
          .transfer_control_on_user_input = true,
          .available_cost_targets = available_cost_targets,
          .select_effects_when_max_positive = true,
          .apply_costs_before_effect_selection = true,
          .clear_context_on_immediate_resolve = true,
          .confirmation_log =
              "[Ability] Triggered optional ability, waiting for confirmation",
          .applied_log = "[Ability] Applied mandatory ability with no targets",
          .cost_selection_log =
              "[Ability] Triggered mandatory ability, selecting cost targets",
          .effect_selection_log =
              "[Ability] Triggered mandatory ability, selecting effect targets",
          .selection_log =
              "[Ability] Triggered mandatory ability, started selection flow",
      });
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

  bool is_active = azk_begin_ability(
      world, gate_card, owner, def,
      &(AbilityBeginOptions){
          .is_optional = def->is_optional,
          .enter_confirmation_when_optional = true,
          .available_cost_targets = azk_count_ability_target_choices(
              world, def, ABILITY_TARGET_SCOPE_COST, gate_card, owner),
          .initial_scratch =
              {
                  .kind = ABILITY_SCRATCH_GATE_PORTAL,
                  .data.gate_portal =
                      {
                          .portaled_card = portaled_card,
                      },
              },
          .confirmation_log =
              "[Ability] Gate portal triggered optional ability, waiting for "
              "confirmation",
          .cost_selection_log =
              "[Ability] Gate portal ability, selecting cost targets",
          .effect_selection_log =
              "[Ability] Gate portal ability, selecting effect targets",
          .selection_log =
              "[Ability] Gate portal triggered multi-step ability",
      });

  if (!is_active) {
    cli_render_logf("[Ability] Applied gate portal ability for %s",
                    ecs_get_name(world, gate_card));
  }
}
