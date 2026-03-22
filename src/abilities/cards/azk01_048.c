#include "abilities/cards/azk01_048.h"

#include "components/abilities.h"

void azk01_048_init_passive_observers(ecs_world_t *world, ecs_entity_t card) {
  ecs_set(world, card, CarapaceValue, {.amount = 1});
}
