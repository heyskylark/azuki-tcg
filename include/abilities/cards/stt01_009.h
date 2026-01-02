#ifndef AZUKI_ABILITY_STT01_009_H
#define AZUKI_ABILITY_STT01_009_H

#include <flecs.h>

// STT01-009: If there are 6 or more weapon cards in your discard pile,
// this card has +2 attack.
// This is a passive observer-based ability that watches the discard zone.

// Initialize passive observers for this card
// Creates observers that watch for weapons in owner's discard and zone changes
void stt01_009_init_passive_observers(ecs_world_t *world, ecs_entity_t card);

// Cleanup passive observers and remove any buffs
void stt01_009_cleanup_passive_observers(ecs_world_t *world, ecs_entity_t card);

#endif // AZUKI_ABILITY_STT01_009_H
