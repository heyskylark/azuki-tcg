#ifndef AZUKI_ABILITY_STT01_011_H
#define AZUKI_ABILITY_STT01_011_H

#include <flecs.h>

// STT01-011 "Raizan": As long as this card is in play, the card Ikazuchi
// (STT01-016 "Raizan's Zanbato") has +5 attack instead of +4.
// This is a passive observer-based ability that buffs STT01-016 weapons
// owned by the same player when STT01-011 is in play.

// Initialize passive observers for this card
// Creates observers that watch for:
// 1. STT01-011 entering/leaving owner's garden or alley
// 2. STT01-016 weapons being attached to owner's garden entities
void stt01_011_init_passive_observers(ecs_world_t *world, ecs_entity_t card);

// Cleanup passive observers and remove any buffs
void stt01_011_cleanup_passive_observers(ecs_world_t *world, ecs_entity_t card);

#endif // AZUKI_ABILITY_STT01_011_H
