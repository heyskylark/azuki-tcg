#include "abilities/cards/azk01_048.h"

#include "components/abilities.h"
#include "utils/ability_util.h"

void azk01_048_init_passive_observers(ecs_world_t *world,
                                      ecs_entity_t ability_entity) {
  ecs_entity_t card = azk_get_ability_source_card(world, ability_entity);
  if (card == 0) {
    return;
  }

  ecs_set(world, card, CarapaceValue, {.amount = 1});
}

void azk01_048_cleanup_passive_observers(ecs_world_t *world,
                                         ecs_entity_t ability_entity) {
  ecs_entity_t card = azk_get_ability_source_card(world, ability_entity);
  if (card != 0 && ecs_has(world, card, CarapaceValue)) {
    ecs_remove(world, card, CarapaceValue);
  }
}
