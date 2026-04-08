#ifndef AZUKI_ABILITY_REGISTRY_H
#define AZUKI_ABILITY_REGISTRY_H

#include <flecs.h>
#include <stdbool.h>

#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"

typedef struct {
  bool has_ability;
  bool is_optional;
  bool is_once_per_turn;    // If true, ability can only be used once per turn
  bool can_play_as_response_from_hand; // If true, attach card-level AResponse for hand-play during response
  bool can_select_to_garden; // If true, entity cards can be selected to garden
  bool can_select_to_alley; // If true, entity cards can be selected to alley
  bool can_select_to_equip; // If true, weapon cards can be selected to equip
  bool selection_to_equip_is_reequip; // If true, selection equip moves an existing weapon
  bool can_select_to_hand;  // If true, cards can be selected to add to hand (default true)
  bool selection_pick_is_optional; // If true, ACT_NOOP may skip selection pick
  bool can_topdeck_selection; // If true, remaining selection cards may be placed on top
  bool clear_selection_if_still_active; // If true, do not auto-bottom-deck leftovers
  int8_t ikz_cost;         // IKZ cost for activating ability (0 = no cost)
  AbilityCostRequirements cost_req;
  AbilityEffectRequirements effect_req;
  ecs_id_t timing_tag; // AOnPlay, AStartOfTurn, etc. (0 if none)
  ecs_id_t secondary_timing_tag; // Optional second timing tag for dual-timing cards

  // Function pointers for ability logic
  bool (*validate)(ecs_world_t *, ecs_entity_t card, ecs_entity_t owner);
  bool (*validate_cost_target)(ecs_world_t *, ecs_entity_t card,
                               ecs_entity_t owner, ecs_entity_t target);
  bool (*validate_effect_target)(ecs_world_t *, ecs_entity_t card,
                                 ecs_entity_t owner, ecs_entity_t target);
  void (*apply_costs)(ecs_world_t *, const AbilityContext *);
  void (*apply_effects)(ecs_world_t *, const AbilityContext *);

  // Sub-effect callbacks for multi-step abilities (reveal/selection effects)
  void (*on_cost_paid)(ecs_world_t *, AbilityContext *);
  bool (*validate_selection_target)(ecs_world_t *, ecs_entity_t card,
                                    ecs_entity_t owner, ecs_entity_t target);
  void (*on_selection_complete)(ecs_world_t *, AbilityContext *);

  // Observer-based passive abilities (initialized on the owned ability entity)
  void (*init_passive_observers)(ecs_world_t *, ecs_entity_t ability_entity);
  void (*cleanup_passive_observers)(ecs_world_t *, ecs_entity_t ability_entity);
} AbilityDef;

// Get ability definition for a card
const AbilityDef *azk_get_ability_def(CardDefId id);

// Get number of registered abilities for a card.
uint8_t azk_get_ability_count(CardDefId id);

// Get a registered ability by deterministic registry order.
const AbilityDef *azk_get_ability_def_at(CardDefId id, uint8_t registry_order);

// Register additional ability defs after the primary registry entry.
bool azk_set_additional_card_abilities(CardDefId id, const AbilityDef *defs,
                                       uint8_t count);

// Get a static equipped combat modifier spec for a card, if it has one.
const EquippedCombatModifier *azk_get_equipped_combat_modifier_spec(
    CardDefId id);

// Check if card has an ability
bool azk_has_ability(CardDefId id);

// Check if card has an ability with a specific timing tag
bool azk_has_ability_with_timing(CardDefId id, ecs_id_t timing_tag);

// Check card-level response-play metadata.
bool azk_can_play_as_response_from_hand(CardDefId id);

// Initialize ability registry (call after ability tags are registered)
void azk_init_ability_registry(ecs_world_t *world);

#endif // AZUKI_ABILITY_REGISTRY_H
