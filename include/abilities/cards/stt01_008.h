#ifndef AZUKI_ABILITY_STT01_008_H
#define AZUKI_ABILITY_STT01_008_H

#include <flecs.h>

// STT01-008: When equipped with a weapon card, this card has +1 attack.
// This is a passive observer-based ability that triggers on weapon attachment.

// Initialize passive observers for this card
// Creates an observer that watches for weapon ChildOf attachment/detachment
void stt01_008_init_passive_observers(ecs_world_t *world, ecs_entity_t card);

// Cleanup passive observers and remove any buffs
void stt01_008_cleanup_passive_observers(ecs_world_t *world, ecs_entity_t card);

#endif // AZUKI_ABILITY_STT01_008_H
