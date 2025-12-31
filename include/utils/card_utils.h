#ifndef AZUKI_UTILS_CARD_UTILS_H
#define AZUKI_UTILS_CARD_UTILS_H

#include "generated/card_defs.h"
#include <flecs.h>

bool is_card_type(ecs_world_t *world, ecs_entity_t card, CardType type);
void discard_card(ecs_world_t *world, ecs_entity_t card);
void return_card_to_hand(ecs_world_t *world, ecs_entity_t card);
/**
 * Check if a card can be tapped.
 * Returns false if card is already tapped or on cooldown (unless ignored).
 * @param world The ECS world
 * @param card The card entity to check
 * @param ignore_cooldown If true, cooldown does not prevent tapping
 * @return true if the card can be tapped
 */
bool can_tap_card(ecs_world_t *world, ecs_entity_t card, bool ignore_cooldown);

/**
 * Tap a card. Caller should validate with can_tap_card() first.
 * @param world The ECS world
 * @param card The card entity to tap
 */
void tap_card(ecs_world_t *world, ecs_entity_t card);

void set_card_to_cooldown(ecs_world_t *world, ecs_entity_t card);
bool is_card_tapped(ecs_world_t *world, ecs_entity_t card);
bool is_card_cooldown(ecs_world_t *world, ecs_entity_t card);

/**
 * Check if a card is a weapon card.
 * @param world The ECS world
 * @param card The card entity to check
 * @return true if the card is a weapon type
 */
bool is_weapon_card(ecs_world_t *world, ecs_entity_t card);

/**
 * Count weapon cards in a zone.
 * @param world The ECS world
 * @param zone The zone entity to search
 * @return Number of weapon cards in the zone
 */
int count_weapons_in_zone(ecs_world_t *world, ecs_entity_t zone);

/**
 * Check if a card has a specific subtype tag.
 * @param world The ECS world
 * @param card The card entity to check
 * @param subtype_tag The subtype tag ID (e.g., ecs_id(TSubtype_Watercrafting))
 * @return true if the card has the subtype
 */
bool has_subtype(ecs_world_t *world, ecs_entity_t card, ecs_id_t subtype_tag);

/**
 * Check if a card has the Watercrafting subtype.
 * @param world The ECS world
 * @param card The card entity to check
 * @return true if the card has the Watercrafting subtype
 */
bool is_watercrafting_card(ecs_world_t *world, ecs_entity_t card);

/**
 * Check if a card has the Water element.
 * @param world The ECS world
 * @param card The card entity to check
 * @return true if the card has the Water element
 */
bool is_water_element_card(ecs_world_t *world, ecs_entity_t card);

/**
 * Count cards with a specific subtype in a zone.
 * @param world The ECS world
 * @param zone The zone entity to search
 * @param subtype_tag The subtype tag ID
 * @return Number of cards with the subtype in the zone
 */
int count_subtype_in_zone(ecs_world_t *world, ecs_entity_t zone,
                          ecs_id_t subtype_tag);

/**
 * Check if an entity has a weapon equipped (attached as child).
 * @param world The ECS world
 * @param entity The entity to check (garden entity or leader)
 * @return true if the entity has at least one weapon attached
 */
bool has_equipped_weapon(ecs_world_t *world, ecs_entity_t entity);

#endif
