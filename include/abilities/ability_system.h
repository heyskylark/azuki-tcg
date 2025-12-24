#ifndef AZUKI_ABILITY_SYSTEM_H
#define AZUKI_ABILITY_SYSTEM_H

#include <stdbool.h>
#include <flecs.h>

#include "components/components.h"

// Trigger an on-play ability for a card that was just played
// Returns true if an ability was triggered (player needs to confirm/decline)
// Returns false if no ability, ability invalid, or ability is non-optional and auto-executes
bool azk_trigger_on_play_ability(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner);

// Process ability confirmation (ACT_CONFIRM_ABILITY)
// Transitions from CONFIRMATION to COST_SELECTION (or EFFECT_SELECTION if no cost)
// Returns true on success, false if no ability pending
bool azk_process_ability_confirmation(ecs_world_t* world);

// Process ability decline (ACT_NOOP during confirmation phase)
// Clears the ability context and returns to normal play
bool azk_process_ability_decline(ecs_world_t* world);

// Process cost target selection (ACT_SELECT_COST_TARGET)
// target_index is the index from the user action (e.g., hand index)
// Returns true if target is valid and added, false otherwise
bool azk_process_cost_selection(ecs_world_t* world, int target_index);

// Process effect target selection (ACT_SELECT_EFFECT_TARGET)
// Returns true if target is valid and added, false otherwise
bool azk_process_effect_selection(ecs_world_t* world, int target_index);

// Check if we're currently in an ability sub-phase
bool azk_is_in_ability_phase(ecs_world_t* world);

// Get current ability phase
AbilityPhase azk_get_ability_phase(ecs_world_t* world);

// Clear ability context (call after ability completes or is cancelled)
void azk_clear_ability_context(ecs_world_t* world);

// Trigger a spell ability after the spell has been cast
// Unlike on-play abilities, spells skip confirmation (already committed to casting)
// Returns true if ability requires target selection, false if auto-executes
bool azk_trigger_spell_ability(ecs_world_t* world, ecs_entity_t spell_card, ecs_entity_t owner);

#endif // AZUKI_ABILITY_SYSTEM_H
