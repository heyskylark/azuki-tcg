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
  bool can_select_to_alley; // If true, entity cards can be selected to alley
  bool can_select_to_equip; // If true, weapon cards can be selected to equip
  bool can_select_to_hand;  // If true, cards can be selected to add to hand (default true)
  int8_t ikz_cost;         // IKZ cost for activating ability (0 = no cost)
  AbilityCostRequirements cost_req;
  AbilityEffectRequirements effect_req;
  ecs_id_t timing_tag; // AOnPlay, AStartOfTurn, etc. (0 if none)

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

  // Observer-based passive abilities (initialized when card is instantiated)
  void (*init_passive_observers)(ecs_world_t *, ecs_entity_t card);
  void (*cleanup_passive_observers)(ecs_world_t *, ecs_entity_t card);
} AbilityDef;

// Get ability definition for a card
const AbilityDef *azk_get_ability_def(CardDefId id);

// Check if card has an ability
bool azk_has_ability(CardDefId id);

// Check if card has an ability with a specific timing tag
bool azk_has_ability_with_timing(CardDefId id, ecs_id_t timing_tag);

// Initialize ability registry (call after ability tags are registered)
void azk_init_ability_registry(ecs_world_t *world);

#endif // AZUKI_ABILITY_REGISTRY_H
