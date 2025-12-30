#ifndef AZUKI_ABILITIES_STT02_013_H
#define AZUKI_ABILITIES_STT02_013_H

#include <flecs.h>

#include "components/components.h"

// STT02-013: "[On Play] Look at the top 3 cards of your deck, reveal up to 1
// 2 cost or less water type card and add it to your hand, then bottom deck
// the rest in any order. You may play the card to the alley if it is an entity."

// Validate if ability can be activated (requires 3+ cards in deck)
bool stt02_013_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner);

// Called after confirmation: move top 3 cards from deck to selection zone
void stt02_013_on_cost_paid(ecs_world_t *world, AbilityContext *ctx);

// Validate selection target - must be <=2 cost AND water element
bool stt02_013_validate_selection_target(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         ecs_entity_t target);

// Called after selection pick is complete: transition to bottom deck phase
void stt02_013_on_selection_complete(ecs_world_t *world, AbilityContext *ctx);

#endif // AZUKI_ABILITIES_STT02_013_H
