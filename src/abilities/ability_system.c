#include "abilities/ability_system.h"

#include "abilities/ability_registry.h"
#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"

bool azk_trigger_on_play_ability(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    // Get card ID
    const CardId* card_id = ecs_get(world, card, CardId);
    if (!card_id) {
        return false;
    }

    // Check if card has an ability
    if (!azk_has_ability(card_id->id)) {
        return false;
    }

    const AbilityDef* def = azk_get_ability_def(card_id->id);
    if (!def || !def->has_ability) {
        return false;
    }

    // Check if it's an OnPlay ability
    if (def->timing_tag != ecs_id(AOnPlay)) {
        return false;
    }

    // Validate the ability can be activated
    if (def->validate && !def->validate(world, card, owner)) {
        cli_render_logf("[Ability] ST01-007 ability validation failed (not enough cards)");
        return false;
    }

    // Get ability context singleton
    AbilityContext* ctx = ecs_singleton_get_mut(world, AbilityContext);

    // Set up the ability context
    ctx->source_card = card;
    ctx->owner = owner;
    ctx->is_optional = def->is_optional;
    ctx->cost_min = def->cost_req.min;
    ctx->cost_expected = def->cost_req.max;
    ctx->cost_filled = 0;
    ctx->effect_min = def->effect_req.min;
    ctx->effect_expected = def->effect_req.max;
    ctx->effect_filled = 0;

    // Clear target arrays
    for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
        ctx->cost_targets[i] = 0;
        ctx->effect_targets[i] = 0;
    }

    if (def->is_optional) {
        // Optional ability - enter confirmation phase
        ctx->phase = ABILITY_PHASE_CONFIRMATION;
        cli_render_logf("[Ability] Triggered optional ability, waiting for confirmation");
        ecs_singleton_modified(world, AbilityContext);
        return true;
    } else {
        // Non-optional ability - skip confirmation, go straight to cost selection
        if (def->cost_req.min > 0) {
            ctx->phase = ABILITY_PHASE_COST_SELECTION;
            cli_render_logf("[Ability] Triggered mandatory ability, selecting cost targets");
        } else if (def->effect_req.min > 0) {
            ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
            cli_render_logf("[Ability] Triggered mandatory ability, selecting effect targets");
        } else {
            // No targets needed - apply immediately
            if (def->apply_effects) {
                def->apply_effects(world, ctx);
            }
            ctx->phase = ABILITY_PHASE_NONE;
            cli_render_logf("[Ability] Applied mandatory ability with no targets");
        }
        ecs_singleton_modified(world, AbilityContext);
        return ctx->phase != ABILITY_PHASE_NONE;
    }
}

bool azk_process_ability_confirmation(ecs_world_t* world) {
    AbilityContext* ctx = ecs_singleton_get_mut(world, AbilityContext);

    if (ctx->phase != ABILITY_PHASE_CONFIRMATION) {
        return false;
    }

    const CardId* card_id = ecs_get(world, ctx->source_card, CardId);
    if (!card_id) {
        azk_clear_ability_context(world);
        return false;
    }

    const AbilityDef* def = azk_get_ability_def(card_id->id);
    if (!def) {
        azk_clear_ability_context(world);
        return false;
    }

    // Move to next phase based on requirements
    if (def->cost_req.min > 0) {
        ctx->phase = ABILITY_PHASE_COST_SELECTION;
        cli_render_logf("[Ability] Confirmed, selecting cost targets");
    } else if (def->effect_req.min > 0) {
        // No cost targets to select, but still apply costs (e.g., sacrifice self, draw cards)
        if (def->apply_costs) {
            def->apply_costs(world, ctx);
            cli_render_logf("[Ability] Applied costs (no cost targets needed)");
        }
        ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
        cli_render_logf("[Ability] Confirmed, selecting effect targets");
    } else {
        // No targets needed - apply costs and effects immediately
        if (def->apply_costs) {
            def->apply_costs(world, ctx);
        }
        if (def->apply_effects) {
            def->apply_effects(world, ctx);
        }
        azk_clear_ability_context(world);
        cli_render_logf("[Ability] Confirmed and applied ability with no targets");
        return true;
    }

    ecs_singleton_modified(world, AbilityContext);
    return true;
}

bool azk_process_ability_decline(ecs_world_t* world) {
    AbilityContext* ctx = ecs_singleton_get_mut(world, AbilityContext);

    if (ctx->phase != ABILITY_PHASE_CONFIRMATION) {
        return false;
    }

    if (!ctx->is_optional) {
        // Can't decline non-optional abilities
        return false;
    }

    cli_render_logf("[Ability] Declined optional ability");
    azk_clear_ability_context(world);
    return true;
}

bool azk_process_cost_selection(ecs_world_t* world, int target_index) {
    AbilityContext* ctx = ecs_singleton_get_mut(world, AbilityContext);

    if (ctx->phase != ABILITY_PHASE_COST_SELECTION) {
        return false;
    }

    const CardId* card_id = ecs_get(world, ctx->source_card, CardId);
    if (!card_id) {
        return false;
    }

    const AbilityDef* def = azk_get_ability_def(card_id->id);
    if (!def) {
        return false;
    }

    // Get the target entity based on cost type
    ecs_entity_t target = 0;
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, ctx->owner);

    switch (def->cost_req.type) {
        case ABILITY_TARGET_FRIENDLY_HAND: {
            ecs_entity_t hand = gs->zones[player_num].hand;
            ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
            if (target_index >= 0 && target_index < hand_cards.count) {
                target = hand_cards.ids[target_index];
            }
            break;
        }
        // Add other target types as needed
        default:
            break;
    }

    if (target == 0) {
        cli_render_logf("[Ability] Invalid cost target index %d", target_index);
        return false;
    }

    // Validate the target
    if (def->validate_cost_target && !def->validate_cost_target(world, ctx->source_card, ctx->owner, target)) {
        cli_render_logf("[Ability] Cost target validation failed");
        return false;
    }

    // Add target to context
    if (ctx->cost_filled >= MAX_ABILITY_SELECTION) {
        cli_render_logf("[Ability] Too many cost targets");
        return false;
    }

    ctx->cost_targets[ctx->cost_filled] = target;
    ctx->cost_filled++;

    cli_render_logf("[Ability] Added cost target %d (%d/%d)", target_index, ctx->cost_filled, ctx->cost_expected);

    // Check if we have enough targets
    if (ctx->cost_filled >= ctx->cost_expected) {
        // Apply costs
        if (def->apply_costs) {
            def->apply_costs(world, ctx);
            cli_render_logf("[Ability] Applied costs");
        }

        // Move to effect selection or apply effects
        if (def->effect_req.min > 0) {
            ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
            cli_render_logf("[Ability] Moving to effect selection");
        } else {
            // Apply effects and finish
            if (def->apply_effects) {
                def->apply_effects(world, ctx);
                cli_render_logf("[Ability] Applied effects");
            }
            azk_clear_ability_context(world);
            return true;
        }
    }

    ecs_singleton_modified(world, AbilityContext);
    return true;
}

bool azk_process_effect_selection(ecs_world_t* world, int target_index) {
    AbilityContext* ctx = ecs_singleton_get_mut(world, AbilityContext);

    if (ctx->phase != ABILITY_PHASE_EFFECT_SELECTION) {
        return false;
    }

    const CardId* card_id = ecs_get(world, ctx->source_card, CardId);
    if (!card_id) {
        return false;
    }

    const AbilityDef* def = azk_get_ability_def(card_id->id);
    if (!def) {
        return false;
    }

    // Get the target entity based on effect type
    ecs_entity_t target = 0;
    const GameState* gs = ecs_singleton_get(world, GameState);
    uint8_t player_num = get_player_number(world, ctx->owner);

    switch (def->effect_req.type) {
        case ABILITY_TARGET_FRIENDLY_HAND: {
            ecs_entity_t hand = gs->zones[player_num].hand;
            ecs_entities_t hand_cards = ecs_get_ordered_children(world, hand);
            if (target_index >= 0 && target_index < hand_cards.count) {
                target = hand_cards.ids[target_index];
            }
            break;
        }
        case ABILITY_TARGET_ENEMY_GARDEN_ENTITY: {
            uint8_t enemy_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
            ecs_entity_t garden = gs->zones[enemy_num].garden;
            ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
            if (target_index >= 0 && target_index < garden_cards.count) {
                target = garden_cards.ids[target_index];
            }
            break;
        }
        case ABILITY_TARGET_ANY_GARDEN_ENTITY: {
            // Index encoding: 0-4 = self garden, 5-9 = opponent garden
            uint8_t target_player_num;
            int garden_index;
            if (target_index < GARDEN_SIZE) {
                target_player_num = player_num;
                garden_index = target_index;
            } else {
                target_player_num = (player_num + 1) % MAX_PLAYERS_PER_MATCH;
                garden_index = target_index - GARDEN_SIZE;
            }
            ecs_entity_t garden = gs->zones[target_player_num].garden;
            ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);
            // Find card at the specific zone index
            for (int i = 0; i < garden_cards.count; i++) {
                const ZoneIndex* zi = ecs_get(world, garden_cards.ids[i], ZoneIndex);
                if (zi && zi->index == garden_index) {
                    target = garden_cards.ids[i];
                    break;
                }
            }
            break;
        }
        default:
            break;
    }

    if (target == 0) {
        cli_render_logf("[Ability] Invalid effect target index %d", target_index);
        return false;
    }

    // Validate the target
    if (def->validate_effect_target && !def->validate_effect_target(world, ctx->source_card, ctx->owner, target)) {
        cli_render_logf("[Ability] Effect target validation failed");
        return false;
    }

    // Add target to context
    if (ctx->effect_filled >= MAX_ABILITY_SELECTION) {
        cli_render_logf("[Ability] Too many effect targets");
        return false;
    }

    ctx->effect_targets[ctx->effect_filled] = target;
    ctx->effect_filled++;

    cli_render_logf("[Ability] Added effect target %d (%d/%d)", target_index, ctx->effect_filled, ctx->effect_expected);

    // Check if we have enough targets
    if (ctx->effect_filled >= ctx->effect_expected) {
        // Apply effects and finish
        if (def->apply_effects) {
            def->apply_effects(world, ctx);
            cli_render_logf("[Ability] Applied effects");
        }
        azk_clear_ability_context(world);
    } else {
        ecs_singleton_modified(world, AbilityContext);
    }

    return true;
}

bool azk_is_in_ability_phase(ecs_world_t* world) {
    const AbilityContext* ctx = ecs_singleton_get(world, AbilityContext);
    return ctx && ctx->phase != ABILITY_PHASE_NONE;
}

AbilityPhase azk_get_ability_phase(ecs_world_t* world) {
    const AbilityContext* ctx = ecs_singleton_get(world, AbilityContext);
    return ctx ? ctx->phase : ABILITY_PHASE_NONE;
}

void azk_clear_ability_context(ecs_world_t* world) {
    AbilityContext* ctx = ecs_singleton_get_mut(world, AbilityContext);

    ctx->phase = ABILITY_PHASE_NONE;
    ctx->source_card = 0;
    ctx->owner = 0;
    ctx->is_optional = false;
    ctx->cost_min = 0;
    ctx->cost_expected = 0;
    ctx->cost_filled = 0;
    ctx->effect_min = 0;
    ctx->effect_expected = 0;
    ctx->effect_filled = 0;

    for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
        ctx->cost_targets[i] = 0;
        ctx->effect_targets[i] = 0;
    }

    ecs_singleton_modified(world, AbilityContext);
}

bool azk_trigger_main_ability(ecs_world_t* world, ecs_entity_t card, ecs_entity_t owner) {
    // Get card ID
    const CardId* card_id = ecs_get(world, card, CardId);
    if (!card_id) {
        return false;
    }

    // Check if card has an ability
    if (!azk_has_ability(card_id->id)) {
        return false;
    }

    const AbilityDef* def = azk_get_ability_def(card_id->id);
    if (!def || !def->has_ability) {
        return false;
    }

    // Check if it's a Main phase ability
    if (def->timing_tag != ecs_id(AMain)) {
        return false;
    }

    // Validate the ability can be activated
    if (def->validate && !def->validate(world, card, owner)) {
        cli_render_logf("[Ability] Main ability validation failed");
        return false;
    }

    // Get ability context singleton
    AbilityContext* ctx = ecs_singleton_get_mut(world, AbilityContext);

    // Set up the ability context
    ctx->source_card = card;
    ctx->owner = owner;
    ctx->is_optional = def->is_optional;
    ctx->cost_min = def->cost_req.min;
    ctx->cost_expected = def->cost_req.max;
    ctx->cost_filled = 0;
    ctx->effect_min = def->effect_req.min;
    ctx->effect_expected = def->effect_req.max;
    ctx->effect_filled = 0;

    // Clear target arrays
    for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
        ctx->cost_targets[i] = 0;
        ctx->effect_targets[i] = 0;
    }

    // Main abilities are triggered by player action, so skip confirmation phase
    // (player already opted in by taking the action)
    if (def->cost_req.min > 0) {
        ctx->phase = ABILITY_PHASE_COST_SELECTION;
        cli_render_logf("[Ability] Triggered main ability, selecting cost targets");
    } else if (def->effect_req.min > 0) {
        // No cost targets needed, apply costs (if any) and go to effect selection
        if (def->apply_costs) {
            def->apply_costs(world, ctx);
        }
        ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
        cli_render_logf("[Ability] Triggered main ability, selecting effect targets");
    } else {
        // No targets needed - apply immediately
        if (def->apply_costs) {
            def->apply_costs(world, ctx);
        }
        if (def->apply_effects) {
            def->apply_effects(world, ctx);
        }
        ctx->phase = ABILITY_PHASE_NONE;
        cli_render_logf("[Ability] Applied main ability with no targets");
    }
    ecs_singleton_modified(world, AbilityContext);
    return ctx->phase != ABILITY_PHASE_NONE;
}

bool azk_trigger_spell_ability(ecs_world_t* world, ecs_entity_t spell_card, ecs_entity_t owner) {
    // Get card ID
    const CardId* card_id = ecs_get(world, spell_card, CardId);
    if (!card_id) {
        return false;
    }

    // Check if card has an ability
    if (!azk_has_ability(card_id->id)) {
        return false;
    }

    const AbilityDef* def = azk_get_ability_def(card_id->id);
    if (!def || !def->has_ability) {
        return false;
    }

    // Get ability context singleton
    AbilityContext* ctx = ecs_singleton_get_mut(world, AbilityContext);

    // Set up the ability context
    ctx->source_card = spell_card;
    ctx->owner = owner;
    ctx->is_optional = false;  // Spells are already cast, not optional
    ctx->cost_min = def->cost_req.min;
    ctx->cost_expected = def->cost_req.max;
    ctx->cost_filled = 0;
    ctx->effect_min = def->effect_req.min;
    ctx->effect_expected = def->effect_req.max;
    ctx->effect_filled = 0;

    // Clear target arrays
    for (int i = 0; i < MAX_ABILITY_SELECTION; i++) {
        ctx->cost_targets[i] = 0;
        ctx->effect_targets[i] = 0;
    }

    // Spells skip confirmation phase - go straight to cost or effect selection
    if (def->cost_req.min > 0) {
        ctx->phase = ABILITY_PHASE_COST_SELECTION;
        cli_render_logf("[Ability] Spell triggered, selecting cost targets");
    } else if (def->effect_req.min > 0) {
        ctx->phase = ABILITY_PHASE_EFFECT_SELECTION;
        cli_render_logf("[Ability] Spell triggered, selecting effect targets");
    } else {
        // No targets needed - apply immediately
        if (def->apply_effects) {
            def->apply_effects(world, ctx);
        }
        ctx->phase = ABILITY_PHASE_NONE;
        cli_render_logf("[Ability] Applied spell with no targets");
        return false;  // No further action required
    }

    ecs_singleton_modified(world, AbilityContext);
    return ctx->phase != ABILITY_PHASE_NONE;
}
