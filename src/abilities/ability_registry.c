#include "abilities/ability_registry.h"

#include "abilities/cards/st01_007.h"
#include "components/abilities.h"

// Static registry table - most entries are empty (no ability)
// Entries are populated in azk_init_ability_registry() after tags are registered
static AbilityDef kAbilityRegistry[CARD_DEF_COUNT] = {0};

// Flag to track if registry has been initialized
static bool kRegistryInitialized = false;

const AbilityDef* azk_get_ability_def(CardDefId id) {
    if ((size_t)id >= CARD_DEF_COUNT) {
        return NULL;
    }
    return &kAbilityRegistry[id];
}

bool azk_has_ability(CardDefId id) {
    if ((size_t)id >= CARD_DEF_COUNT) {
        return false;
    }
    return kAbilityRegistry[id].has_ability;
}

void azk_init_ability_registry(ecs_world_t* world) {
    (void)world;

    if (kRegistryInitialized) {
        return;
    }

    // ST01-007 "Alley Guy": On Play; You may discard 1:Draw 1
    kAbilityRegistry[CARD_DEF_STT01_007] = (AbilityDef){
        .has_ability = true,
        .is_optional = true,
        .cost_req = {
            .type = ABILITY_TARGET_FRIENDLY_HAND,
            .min = 1,
            .max = 1
        },
        .effect_req = {
            .type = ABILITY_TARGET_NONE,
            .min = 0,
            .max = 0
        },
        .timing_tag = ecs_id(AOnPlay),
        .validate = st01_007_validate,
        .validate_cost_target = st01_007_validate_cost_target,
        .validate_effect_target = NULL,
        .apply_costs = st01_007_apply_costs,
        .apply_effects = st01_007_apply_effects,
    };

    kRegistryInitialized = true;
}
