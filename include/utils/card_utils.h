#ifndef AZUKI_UTILS_CARD_UTILS_H
#define AZUKI_UTILS_CARD_UTILS_H

#include "generated/card_defs.h"
#include <flecs.h>

bool is_card_type(ecs_world_t *world, ecs_entity_t card, CardType type);
void discard_card(ecs_world_t *world, ecs_entity_t card);
void sacrifice_card(ecs_world_t *world, ecs_entity_t card);
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

/** Mark an IKZ source recovered or created by an effect for reward attribution. */
void azk_mark_generated_ikz_credit(ecs_world_t *world, ecs_entity_t card);

/** Clear unused generated-IKZ attribution before the natural turn refresh. */
void azk_clear_generated_ikz_credit(ecs_world_t *world, ecs_entity_t card);

void set_card_to_cooldown(ecs_world_t *world, ecs_entity_t card);
bool is_card_tapped(ecs_world_t *world, ecs_entity_t card);
bool is_card_cooldown(ecs_world_t *world, ecs_entity_t card);
bool azk_card_has_godmode_in_play(ecs_world_t *world, ecs_entity_t card);
bool azk_card_enters_garden_tapped(ecs_world_t *world, ecs_entity_t card);
bool azk_card_cannot_be_untapped(ecs_world_t *world, ecs_entity_t card);
bool azk_card_can_only_attack_leaders(ecs_world_t *world, ecs_entity_t card);
bool azk_card_can_attack_opponent_alley(ecs_world_t *world, ecs_entity_t card);
bool azk_card_counts_as_ikz_source(ecs_world_t *world, ecs_entity_t card);
bool azk_can_play_card_from_hand_during_response_window(ecs_world_t *world,
                                                        ecs_entity_t card);
int8_t azk_get_effective_card_play_cost(ecs_world_t *world, ecs_entity_t player,
                                        ecs_entity_t card);
void discard_card_for_replacement(ecs_world_t *world, ecs_entity_t card);

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
 * Check if a card has the Normal element.
 * @param world The ECS world
 * @param card The card to inspect
 * @return true if the card has the Normal element
 */
bool is_normal_element_card(ecs_world_t *world, ecs_entity_t card);

/**
 * Get a card's element enum value.
 * @param world The ECS world
 * @param card The card to inspect
 * @return The card element value, or CARD_ELEMENT_NORMAL if missing
 */
CardElement get_card_element(ecs_world_t *world, ecs_entity_t card);

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
