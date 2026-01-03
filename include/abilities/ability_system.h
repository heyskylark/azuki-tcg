#ifndef AZUKI_ABILITY_SYSTEM_H
#define AZUKI_ABILITY_SYSTEM_H

#include <flecs.h>
#include <stdbool.h>

#include "components/components.h"

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

// Trigger a leader response ability (AResponse tag) during response window
// Similar to spell abilities but for leader cards with response timing
// Returns true if ability requires target selection, false if auto-executes
bool azk_trigger_leader_response_ability(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner);

// Trigger observer abilities when a card is returned to hand from garden/alley
// Scans both players' gardens for cards with AWhenReturnedToHand timing tag
// and queues triggered effects for valid observers
void azk_trigger_return_to_hand_observers(ecs_world_t *world,
                                          ecs_entity_t bounced_card);

// Trigger when-equipped ability for a weapon that was just attached
// Returns true if ability requires user input, false otherwise
bool azk_trigger_when_equipped_ability(ecs_world_t *world, ecs_entity_t card,
                                       ecs_entity_t owner);

// Trigger gate card's portal ability after successfully portaling an entity
// gate_card: the gate card that was used to portal
// portaled_card: the entity card that was moved from alley to garden
// owner: the player who owns the gate card
// All gate cards must have a registered ability with AOnGatePortal timing tag
void azk_trigger_gate_portal_ability(ecs_world_t *world, ecs_entity_t gate_card,
                                     ecs_entity_t portaled_card,
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
