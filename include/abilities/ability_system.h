#ifndef AZUKI_ABILITY_SYSTEM_H
#define AZUKI_ABILITY_SYSTEM_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

// Trigger an on-play ability for a card that was just played
// Returns true if an ability was triggered (player needs to confirm/decline)
// Returns false if no ability, ability invalid, or ability is non-optional and
// auto-executes
bool azk_trigger_on_play_ability(ecs_world_t *world, ecs_entity_t card,
                                 ecs_entity_t owner);

// Process ability confirmation (ACT_CONFIRM_ABILITY)
// Transitions from CONFIRMATION to COST_SELECTION (or EFFECT_SELECTION if no
// cost) Returns true on success, false if no ability pending
bool azk_process_ability_confirmation(ecs_world_t *world);

// Process ability decline (ACT_NOOP during confirmation phase)
// Clears the ability context and returns to normal play
bool azk_process_ability_decline(ecs_world_t *world);

// Process cost target selection (ACT_SELECT_COST_TARGET)
// target_index is the index from the user action (e.g., hand index)
// Returns true if target is valid and added, false otherwise
bool azk_process_cost_selection(ecs_world_t *world, int target_index);

// Process effect target selection (ACT_SELECT_EFFECT_TARGET)
// Returns true if target is valid and added, false otherwise
bool azk_process_effect_selection(ecs_world_t *world, int target_index);

// Skip effect target selection (ACT_NOOP during effect selection when min=0)
// For "up to" effects where the player can choose to not select any targets
// Returns true if skipping is valid (min=0), false otherwise
bool azk_process_effect_skip(ecs_world_t *world);

// Process selection pick (ACT_SELECT_FROM_SELECTION)
// selection_index is the index into the selection zone
// Returns true if selection is valid, false otherwise
bool azk_process_selection_pick(ecs_world_t *world, int selection_index);

// Process selection to alley (ACT_SELECT_TO_ALLEY)
// selection_index is the index into the selection zone
// alley_slot_index is the target alley slot (0-4)
// Moves an entity card directly to alley without IKZ cost
// Returns true if successful, false otherwise
bool azk_process_selection_to_alley(ecs_world_t *world, int selection_index,
                                    int alley_slot_index);

// Skip selection pick (ACT_NOOP during selection pick when selection_pick_max >
// 0 allows) For "up to" effects where the player can choose to not select any
// cards Returns true if skipping is valid, false otherwise
bool azk_process_skip_selection(ecs_world_t *world);

// Process bottom deck action (ACT_BOTTOM_DECK_CARD)
// selection_index is the index into the selection zone
// Returns true if valid, false otherwise
bool azk_process_bottom_deck(ecs_world_t *world, int selection_index);

// Process bottom deck all action (ACT_BOTTOM_DECK_ALL)
// Bottom decks all remaining selection cards in their current order
// Returns true if valid, false otherwise
bool azk_process_bottom_deck_all(ecs_world_t *world);

// Check if we're currently in an ability sub-phase
bool azk_is_in_ability_phase(ecs_world_t *world);

// Get current ability phase
AbilityPhase azk_get_ability_phase(ecs_world_t *world);

// Clear ability context (call after ability completes or is cancelled)
void azk_clear_ability_context(ecs_world_t *world);

// Trigger a spell ability after the spell has been cast
// Unlike on-play abilities, spells skip confirmation (already committed to
// casting) Returns true if ability requires target selection, false if
// auto-executes
bool azk_trigger_spell_ability(ecs_world_t *world, ecs_entity_t spell_card,
                               ecs_entity_t owner);

// Trigger a main phase ability (AMain tag) for a card in the alley
// These are optional abilities that can be activated during the main phase
// Returns true if ability requires confirmation/selection, false if
// auto-executes or no ability
bool azk_trigger_main_ability(ecs_world_t *world, ecs_entity_t card,
                              ecs_entity_t owner);

// Queue a triggered effect for processing on next game loop
// This is used for timing-based triggers (on play, on equip, etc.) where
// deferred zone operations haven't flushed yet
// timing_tag is the index of the timing tag (e.g., AOnPlay)
// Returns true if successfully queued, false if queue is full
bool azk_queue_triggered_effect(ecs_world_t *world, ecs_entity_t card,
                                ecs_entity_t owner, uint8_t timing_tag);

// Check if there are pending triggered effects in the queue
bool azk_has_queued_triggered_effects(ecs_world_t *world);

// Process the next triggered effect in the queue
// Validates the ability and sets up AbilityContext
// Returns true if an ability requires user input, false otherwise
bool azk_process_triggered_effect_queue(ecs_world_t *world);

#endif // AZUKI_ABILITY_SYSTEM_H
