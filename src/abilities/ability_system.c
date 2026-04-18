#include "abilities/ability_system.h"

#include "abilities/core/ability_context.h"
#include "abilities/core/ability_flow.h"
#include "abilities/core/ability_runtime.h"
#include "abilities/selection/ability_selection.h"
#include "abilities/selection/ability_selection_helpers.h"
#include "abilities/ability_registry.h"
#include "abilities/targeting/ability_targeting.h"
#include "components/abilities.h"
#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/game_log_util.h"
#include "utils/player_util.h"
#include "utils/ability_util.h"
#include "utils/weapon_util.h"
#include "utils/zone_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declaration of timing tag constant
#define TIMING_TAG_ON_PLAY_FWD 0

static ecs_id_t get_timing_tag_id(uint8_t tag_index);

static bool stt04_017_debug_enabled(void) {
  static bool initialized = false;
  static bool enabled = false;
  if (!initialized) {
    const char *raw = getenv("AZK_DEBUG_STT04_017");
    enabled = raw != NULL &&
              (strcmp(raw, "1") == 0 || strcmp(raw, "true") == 0 ||
               strcmp(raw, "TRUE") == 0 || strcmp(raw, "yes") == 0 ||
               strcmp(raw, "on") == 0);
    initialized = true;
  }
  return enabled;
}

static const char *debug_card_type_name(CardType type) {
  switch (type) {
  case CARD_TYPE_ENTITY:
    return "ENTITY";
  case CARD_TYPE_SPELL:
    return "SPELL";
  case CARD_TYPE_WEAPON:
    return "WEAPON";
  case CARD_TYPE_GATE:
    return "GATE";
  case CARD_TYPE_LEADER:
    return "LEADER";
  case CARD_TYPE_IKZ:
    return "IKZ";
  case CARD_TYPE_EXTRA_IKZ:
    return "EXTRA_IKZ";
  default:
    return "UNKNOWN";
  }
}

static void debug_dump_stt04_017_cost_failure(ecs_world_t *world,
                                              const AbilityContext *ctx,
                                              const AbilityDef *def,
                                              int target_index,
                                              ecs_entity_t resolved_target,
                                              const char *reason) {
  if (!world || !ctx || !def || !stt04_017_debug_enabled()) {
    return;
  }

  const CardId *source_card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (source_card_id == NULL || source_card_id->id != CARD_DEF_STT04_017) {
    return;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL || ctx->runtime.owner == 0) {
    return;
  }

  const uint8_t owner_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t garden_zone = gs->zones[owner_num].garden;
  const ecs_entity_t direct_zone_match =
      find_card_in_zone_index(world, garden_zone, target_index);
  const bool resolved_valid =
      resolved_target != 0 &&
      (!def->validate_cost_target ||
       def->validate_cost_target(world, ctx->runtime.source_card,
                                 ctx->runtime.owner, resolved_target));

  fprintf(stderr,
          "[STT04-017 DEBUG] reason=%s owner_player=%u target_index=%d "
          "resolved_target=%llu direct_zone_match=%llu selected=%u/%u "
          "phase=%d source_card=%llu source_ability=%llu\n",
          reason != NULL ? reason : "unknown", (unsigned)owner_num, target_index,
          (unsigned long long)resolved_target,
          (unsigned long long)direct_zone_match, (unsigned)ctx->cost.selected_count,
          (unsigned)ctx->cost.max_allowed, (int)ctx->runtime.phase,
          (unsigned long long)ctx->runtime.source_card,
          (unsigned long long)ctx->runtime.source_ability);

  for (uint8_t i = 0; i < ctx->cost.selected_count; ++i) {
    const ecs_entity_t selected = ctx->cost.entities[i];
    const CardId *selected_card_id = ecs_get(world, selected, CardId);
    const Type *selected_type = ecs_get(world, selected, Type);
    const ZoneIndex *selected_zone_index = ecs_get(world, selected, ZoneIndex);
    const EcsIdentifier *selected_name = ecs_get(world, selected, EcsIdentifier);
    fprintf(stderr,
            "  selected[%u]: entity=%llu code=%s name=%s zone_index=%d type=%s\n",
            (unsigned)i, (unsigned long long)selected,
            selected_card_id != NULL ? selected_card_id->code : "<missing>",
            selected_name != NULL ? selected_name->value : "<unnamed>",
            selected_zone_index != NULL ? (int)selected_zone_index->index : -1,
            selected_type != NULL ? debug_card_type_name(selected_type->value)
                                  : "<missing>");
  }

  AbilityTargetChoice choices[AZK_MAX_ABILITY_TARGET_CHOICES];
  const int choice_count = azk_collect_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, ctx->runtime.source_card,
      ctx->runtime.owner, choices, AZK_MAX_ABILITY_TARGET_CHOICES);
  fprintf(stderr, "  enumerated_choices=%d resolved_valid=%d\n", choice_count,
          resolved_valid ? 1 : 0);
  for (int i = 0; i < choice_count; ++i) {
    const ecs_entity_t entity = choices[i].entity;
    const CardId *choice_card_id = ecs_get(world, entity, CardId);
    const Type *choice_type = ecs_get(world, entity, Type);
    const ZoneIndex *choice_zone_index = ecs_get(world, entity, ZoneIndex);
    const EcsIdentifier *choice_name = ecs_get(world, entity, EcsIdentifier);
    const bool choice_valid =
        !def->validate_cost_target ||
        def->validate_cost_target(world, ctx->runtime.source_card,
                                  ctx->runtime.owner, entity);
    fprintf(stderr,
            "  choice[%d]: action_index=%d entity=%llu code=%s name=%s "
            "zone_index=%d type=%s valid=%d%s\n",
            i, choices[i].action_index, (unsigned long long)entity,
            choice_card_id != NULL ? choice_card_id->code : "<missing>",
            choice_name != NULL ? choice_name->value : "<unnamed>",
            choice_zone_index != NULL ? (int)choice_zone_index->index : -1,
            choice_type != NULL ? debug_card_type_name(choice_type->value)
                                : "<missing>",
            choice_valid ? 1 : 0,
            choices[i].action_index == target_index ? " <requested>" : "");
  }

  ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden_zone);
  fprintf(stderr, "  garden_cards=%d zone=%llu\n", (int)garden_cards.count,
          (unsigned long long)garden_zone);
  for (int32_t i = 0; i < garden_cards.count; ++i) {
    const ecs_entity_t card = garden_cards.ids[i];
    const CardId *card_id = ecs_get(world, card, CardId);
    const Type *type = ecs_get(world, card, Type);
    const ZoneIndex *zone_index = ecs_get(world, card, ZoneIndex);
    const EcsIdentifier *name = ecs_get(world, card, EcsIdentifier);
    fprintf(stderr,
            "  garden[%d]: entity=%llu code=%s name=%s zone_index=%d type=%s\n",
            (int)i, (unsigned long long)card,
            card_id != NULL ? card_id->code : "<missing>",
            name != NULL ? name->value : "<unnamed>",
            zone_index != NULL ? (int)zone_index->index : -1,
            type != NULL ? debug_card_type_name(type->value) : "<missing>");
  }
}

static AbilitySelectionCompletionMode get_selection_completion_mode(
    const AbilityDef *def) {
  if (def && def->clear_selection_if_still_active) {
    return AZK_SELECTION_COMPLETION_CLEAR_IF_STILL_ACTIVE;
  }

  return AZK_SELECTION_COMPLETION_ALLOW_BOTTOM_DECK;
}

static const AbilityDef *get_context_ability_def(ecs_world_t *world,
                                                 const AbilityContext *ctx) {
  if (ctx == NULL) {
    return NULL;
  }

  return azk_get_ability_def_for_entity(world, ctx->runtime.source_ability);
}

static int8_t get_action_index_for_ability(ecs_world_t *world,
                                           ecs_entity_t ability_entity) {
  const AbilityInstance *instance =
      ecs_get(world, ability_entity, AbilityInstance);
  return instance != NULL ? instance->action_index : AZK_NO_ACTION_INDEX;
}

static void apply_deferred_costs_if_needed(ecs_world_t *world,
                                           AbilityContext *ctx,
                                           const AbilityDef *def) {
  if (!ctx || !def || !def->apply_costs ||
      ctx->runtime.apply_costs_before_effect_selection ||
      ctx->runtime.costs_applied) {
    return;
  }

  const bool was_deferred =
      ecs_is_deferred(world) && !ecs_stage_is_readonly(world);
  if (was_deferred) {
    ecs_defer_suspend(world);
  }
  def->apply_costs(world, ctx);
  ctx->runtime.costs_applied = true;
  if (was_deferred) {
    ecs_defer_resume(world);
  }
  cli_render_logf("[Ability] Applied deferred costs");
}

bool azk_trigger_on_play_ability(ecs_world_t *world, ecs_entity_t card,
                                 ecs_entity_t owner) {
  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count = azk_collect_card_timed_abilities(
      world, card, ecs_id(AOnPlay), abilities, AZK_MAX_CARD_ABILITIES);
  bool queued_any = false;
  for (uint8_t i = 0; i < ability_count; ++i) {
    if (azk_queue_triggered_effect(world, abilities[i], owner,
                                   TIMING_TAG_ON_PLAY_FWD)) {
      queued_any = true;
    }
  }

  // Queue the effect for processing on next loop iteration.
  return queued_any;
}

bool azk_trigger_when_equipped_ability(ecs_world_t *world, ecs_entity_t card,
                                       ecs_entity_t owner) {
  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count = azk_collect_card_timed_abilities(
      world, card, ecs_id(AWhenEquipped), abilities, AZK_MAX_CARD_ABILITIES);
  bool queued_any = false;
  for (uint8_t i = 0; i < ability_count; ++i) {
    if (azk_queue_triggered_effect(world, abilities[i], owner,
                                   TIMING_TAG_WHEN_EQUIPPED)) {
      queued_any = true;
    }
  }

  return queued_any;
}

bool azk_trigger_enter_garden_ability(ecs_world_t *world, ecs_entity_t card,
                                      ecs_entity_t owner) {
  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count = azk_collect_card_timed_abilities(
      world, card, ecs_id(AWhenEntersGarden), abilities, AZK_MAX_CARD_ABILITIES);
  bool queued_any = false;
  for (uint8_t i = 0; i < ability_count; ++i) {
    if (azk_queue_triggered_effect(world, abilities[i], owner,
                                   TIMING_TAG_WHEN_ENTERS_GARDEN)) {
      queued_any = true;
    }
  }

  return queued_any;
}

static bool queue_timing_abilities_in_zone(ecs_world_t *world,
                                           ecs_entity_t owner,
                                           ecs_entity_t zone,
                                           uint8_t timing_tag,
                                           ecs_id_t timing_tag_id) {
  bool queued_any = false;
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);

  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
    uint8_t ability_count = azk_collect_card_timed_abilities(
        world, card, timing_tag_id, abilities, AZK_MAX_CARD_ABILITIES);
    for (uint8_t j = 0; j < ability_count; ++j) {
      if (azk_queue_triggered_effect(world, abilities[j], owner, timing_tag)) {
        queued_any = true;
      }
    }
  }

  return queued_any;
}

bool azk_trigger_end_of_turn_abilities(ecs_world_t *world) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!gs) {
    return false;
  }

  const uint8_t active_player_index = gs->active_player_index;
  const ecs_entity_t owner = gs->players[active_player_index];
  const ecs_id_t timing_tag_id = ecs_id(AEndOfTurn);
  bool queued_any = false;

  queued_any |= queue_timing_abilities_in_zone(
      world, owner, gs->zones[active_player_index].garden,
      TIMING_TAG_END_OF_TURN, timing_tag_id);
  queued_any |= queue_timing_abilities_in_zone(
      world, owner, gs->zones[active_player_index].leader,
      TIMING_TAG_END_OF_TURN, timing_tag_id);
  queued_any |= queue_timing_abilities_in_zone(
      world, owner, gs->zones[active_player_index].alley,
      TIMING_TAG_END_OF_TURN, timing_tag_id);

  return queued_any;
}

bool azk_trigger_start_of_turn_abilities(ecs_world_t *world) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!gs) {
    return false;
  }

  const uint8_t active_player_index = gs->active_player_index;
  const ecs_entity_t owner = gs->players[active_player_index];
  const ecs_id_t timing_tag_id = ecs_id(AStartOfTurn);
  bool queued_any = false;

  queued_any |= queue_timing_abilities_in_zone(
      world, owner, gs->zones[active_player_index].garden,
      TIMING_TAG_START_OF_TURN, timing_tag_id);
  queued_any |= queue_timing_abilities_in_zone(
      world, owner, gs->zones[active_player_index].leader,
      TIMING_TAG_START_OF_TURN, timing_tag_id);
  queued_any |= queue_timing_abilities_in_zone(
      world, owner, gs->zones[active_player_index].alley,
      TIMING_TAG_START_OF_TURN, timing_tag_id);

  return queued_any;
}

bool azk_trigger_start_of_each_turn_abilities(ecs_world_t *world) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  if (!gs) {
    return false;
  }

  bool queued_any = false;
  const ecs_id_t timing_tag_id = ecs_id(AStartOfEachTurn);
  for (uint8_t player_num = 0; player_num < MAX_PLAYERS_PER_MATCH; ++player_num) {
    ecs_entity_t owner = gs->players[player_num];
    queued_any |= queue_timing_abilities_in_zone(
        world, owner, gs->zones[player_num].garden,
        TIMING_TAG_START_OF_EACH_TURN, timing_tag_id);
    queued_any |= queue_timing_abilities_in_zone(
        world, owner, gs->zones[player_num].leader,
        TIMING_TAG_START_OF_EACH_TURN, timing_tag_id);
    queued_any |= queue_timing_abilities_in_zone(
        world, owner, gs->zones[player_num].alley,
        TIMING_TAG_START_OF_EACH_TURN, timing_tag_id);
  }

  return queued_any;
}

bool azk_process_ability_confirmation(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_CONFIRMATION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  if (def->cost_req.min > 0) {
    uint8_t available_cost_targets = azk_count_ability_target_choices(
        world, def, ABILITY_TARGET_SCOPE_COST, ctx->runtime.source_card,
        ctx->runtime.owner);
    if (available_cost_targets < def->cost_req.min) {
      cli_render_logf("[Ability] Confirmed ability has no valid cost targets "
                      "(available=%u, required_min=%u), skipping",
                      (unsigned)available_cost_targets,
                      (unsigned)def->cost_req.min);
      azk_clear_ability_context(world);
      return true;
    }
    if (ctx->cost.max_allowed > available_cost_targets) {
      ctx->cost.max_allowed = available_cost_targets;
    }
  }

  if (!azk_enter_initial_phase(world, ctx, def,
                               &(AbilityInitialPhaseOptions){
                                   .select_effects_when_max_positive = true,
                                   .apply_costs_before_effect_selection = true,
                               })) {
    azk_clear_ability_context(world);
    cli_render_logf("[Ability] Confirmed and applied ability with no targets");
    return true;
  }

  azk_log_ability_initial_phase_entry(
      ctx->runtime.phase, "[Ability] Confirmed, selecting cost targets",
      "[Ability] Confirmed, selecting effect targets",
      "[Ability] Confirmed, started selection flow");

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_ability_decline(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_CONFIRMATION) {
    return false;
  }

  if (!ctx->runtime.is_optional) {
    // Can't decline non-optional abilities
    return false;
  }

  cli_render_logf("[Ability] Declined optional ability");
  azk_clear_ability_context(world);
  return true;
}

static bool finish_cost_selection(ecs_world_t *world, AbilityContext *ctx,
                                  const AbilityDef *def) {
  if (!ctx || !def) {
    return false;
  }

  if (def->apply_costs) {
    const bool was_deferred =
        ecs_is_deferred(world) && !ecs_stage_is_readonly(world);
    if (was_deferred) {
      ecs_defer_suspend(world);
    }
    def->apply_costs(world, ctx);
    ctx->runtime.costs_applied = true;
    if (was_deferred) {
      ecs_defer_resume(world);
    }
    cli_render_logf("[Ability] Applied costs");
  }

  if (def->on_cost_paid) {
    def->on_cost_paid(world, ctx);
    cli_render_logf("[Ability] Called on_cost_paid callback");
    if (ctx->runtime.phase == ABILITY_PHASE_SELECTION_PICK ||
        ctx->runtime.phase == ABILITY_PHASE_BOTTOM_DECK) {
      ecs_singleton_modified(world, AbilityContext);
      return true;
    }
  }

  if (ctx->effect.max_allowed > 0) {
    if (!azk_prepare_effect_selection_after_costs(world, ctx, def)) {
      if (def->apply_effects) {
        def->apply_effects(world, ctx);
        cli_render_logf(
            "[Ability] Applied effects after costs removed all valid targets");
      }
      azk_clear_ability_context(world);
      return true;
    }
    ctx->runtime.phase = ABILITY_PHASE_EFFECT_SELECTION;
    cli_render_logf("[Ability] Moving to effect selection");
    ecs_singleton_modified(world, AbilityContext);
    return true;
  }

  if (def->apply_effects) {
    def->apply_effects(world, ctx);
    cli_render_logf("[Ability] Applied effects");
  }

  azk_clear_ability_context(world);
  return true;
}

bool azk_process_cost_selection(ecs_world_t *world, int target_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_COST_SELECTION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    return false;
  }

  ecs_entity_t target = azk_resolve_ability_target_choice_entity(
      world, def, ABILITY_TARGET_SCOPE_COST, ctx->runtime.owner, target_index);

  if (target == 0) {
    debug_dump_stt04_017_cost_failure(world, ctx, def, target_index, target,
                                      "resolve_failed");
    cli_render_logf("[Ability] Invalid cost target index %d", target_index);
    return false;
  }

  // Validate the target
  if (def->validate_cost_target &&
      !def->validate_cost_target(world, ctx->runtime.source_card,
                                 ctx->runtime.owner, target)) {
    debug_dump_stt04_017_cost_failure(world, ctx, def, target_index, target,
                                      "validate_failed");
    cli_render_logf("[Ability] Cost target validation failed");
    return false;
  }

  // Add target to context
  if (ctx->cost.selected_count >= MAX_ABILITY_SELECTION) {
    cli_render_logf("[Ability] Too many cost targets");
    return false;
  }

  ctx->cost.entities[ctx->cost.selected_count] = target;
  ctx->cost.selected_count++;

  cli_render_logf("[Ability] Added cost target %d (%d/%d)", target_index,
                  ctx->cost.selected_count, ctx->cost.max_allowed);

  if (ctx->cost.selected_count >= ctx->cost.max_allowed) {
    return finish_cost_selection(world, ctx, def);
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_cost_skip(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_COST_SELECTION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  if (ctx->cost.selected_count < ctx->cost.min_required) {
    cli_render_logf(
        "[Ability] Cannot finish cost selection before minimum is reached");
    return false;
  }

  return finish_cost_selection(world, ctx, def);
}

bool azk_process_effect_selection(ecs_world_t *world, int target_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_EFFECT_SELECTION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    return false;
  }

  ecs_entity_t target = azk_resolve_ability_target_choice_entity(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, ctx->runtime.owner,
      target_index);

  if (target == 0) {
    cli_render_logf("[Ability] Invalid effect target index %d", target_index);
    return false;
  }

  // Validate the target
  if (def->validate_effect_target &&
      !def->validate_effect_target(world, ctx->runtime.source_card,
                                   ctx->runtime.owner, target)) {
    cli_render_logf("[Ability] Effect target validation failed");
    return false;
  }

  // Add target to context
  if (ctx->effect.selected_count >= MAX_ABILITY_SELECTION) {
    cli_render_logf("[Ability] Too many effect targets");
    return false;
  }

  ctx->effect.entities[ctx->effect.selected_count] = target;
  ctx->effect.selected_count++;

  cli_render_logf("[Ability] Added effect target %d (%d/%d)", target_index,
                  ctx->effect.selected_count, ctx->effect.max_allowed);

  // Check if we have enough targets
  if (ctx->effect.selected_count >= ctx->effect.max_allowed) {
    // Apply effects and finish
    AbilityPhase phase_before_effects = ctx->runtime.phase;
    apply_deferred_costs_if_needed(world, ctx, def);
    if (def->apply_effects) {
      def->apply_effects(world, ctx);
      cli_render_logf("[Ability] Applied effects");
    }
    if (ctx->runtime.phase != phase_before_effects &&
        ctx->runtime.phase != ABILITY_PHASE_NONE) {
      ecs_singleton_modified(world, AbilityContext);
    } else {
      azk_clear_ability_context(world);
    }
  } else {
    ecs_singleton_modified(world, AbilityContext);
  }

  return true;
}

bool azk_process_effect_skip(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_EFFECT_SELECTION) {
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  if (ctx->effect.selected_count < ctx->effect.min_required) {
    uint8_t remaining_choices = azk_count_ability_target_choices(
        world, def, ABILITY_TARGET_SCOPE_EFFECT, ctx->runtime.source_card,
        ctx->runtime.owner);
    if (remaining_choices > 0) {
      cli_render_logf(
          "[Ability] Cannot finish effect selection before minimum is reached");
      return false;
    }
    cli_render_logf(
        "[Ability] No valid effect targets remain; resolving ability");
  }

  apply_deferred_costs_if_needed(world, ctx, def);
  if (def->apply_effects) {
    AbilityPhase phase_before_effects = ctx->runtime.phase;
    def->apply_effects(world, ctx);
    cli_render_logf("[Ability] Applied effects (finished target selection)");
    if (ctx->runtime.phase != phase_before_effects &&
        ctx->runtime.phase != ABILITY_PHASE_NONE) {
      ecs_singleton_modified(world, AbilityContext);
      return true;
    }
  }

  azk_clear_ability_context(world);
  return true;
}

bool azk_process_selection_pick(ecs_world_t *world, int selection_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // Validate index is in range
  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection.count);
    return false;
  }

  ecs_entity_t target = ctx->selection.cards[selection_index];
  if (target == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Safety check: verify ability allows adding to hand
  // If ability has special selection modes but can_select_to_hand is false, reject
  bool has_special_selection = def->can_select_to_garden ||
                               def->can_select_to_alley ||
                               def->can_select_to_equip;
  if (has_special_selection && !def->can_select_to_hand) {
    cli_render_logf("[Ability] Selection pick not allowed - must use equip or alley action");
    return false;
  }

  // Validate the selection target if validation function exists
  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->runtime.source_card,
                                      ctx->runtime.owner, target)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  azk_record_selection_pick(ctx, selection_index, target);

  cli_render_logf("[Ability] Picked selection %d (%d/%d)", selection_index,
                  ctx->selection.picked_count, ctx->selection.pick_max);

  // Check if we've picked enough
  if (ctx->selection.picked_count >= ctx->selection.pick_max) {
    return azk_finish_selection_resolution(
        world, ctx, def, get_selection_completion_mode(def));
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_selection_to_garden(ecs_world_t *world, int selection_index,
                                     int garden_slot_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection.count);
    return false;
  }

  if (garden_slot_index < 0 || garden_slot_index >= GARDEN_SIZE) {
    cli_render_logf("[Ability] Invalid garden slot index %d", garden_slot_index);
    return false;
  }

  ecs_entity_t target = ctx->selection.cards[selection_index];
  if (target == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  if (!def->can_select_to_garden) {
    cli_render_logf("[Ability] This ability does not allow selecting to garden");
    return false;
  }

  const Type *target_type = ecs_get(world, target, Type);
  if (!target_type || target_type->value != CARD_TYPE_ENTITY) {
    cli_render_logf("[Ability] Only entity cards can be selected to garden");
    return false;
  }

  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->runtime.source_card,
                                      ctx->runtime.owner, target)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  const ecs_entity_t garden = gs->zones[player_num].garden;
  ecs_entity_t displaced_card =
      find_card_in_zone_index(world, garden, garden_slot_index);
  const ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

  if (displaced_card != 0 && garden_cards.count < GARDEN_SIZE) {
    cli_render_logf("[Ability] Garden slot %d is already occupied",
                    garden_slot_index);
    return false;
  }

  PlayEntityIntent intent = {
      .player = ctx->runtime.owner,
      .card = target,
      .placement_type = ZONE_GARDEN,
      .target_zone = garden,
      .zone_index = garden_slot_index,
      .displaced_card = displaced_card,
  };

  if (summon_card_into_zone_index(world, &intent) < 0) {
    cli_render_logf("[Ability] Failed to place selected card into garden");
    return false;
  }

  GameState *gs_mut = ecs_singleton_get_mut(world, GameState);
  gs_mut->entities_played_garden_this_turn[player_num]++;
  gs_mut->cards_played_this_turn[player_num]++;
  gs_mut->next_card_play_cost_reduction[player_num] = 0;
  ecs_singleton_modified(world, GameState);

  azk_trigger_on_play_ability(world, target, ctx->runtime.owner);

  cli_render_logf("[Ability] Selected card to garden slot %d",
                  garden_slot_index);

  azk_record_selection_pick(ctx, selection_index, target);

  if (ctx->selection.picked_count >= ctx->selection.pick_max) {
    return azk_finish_selection_resolution(
        world, ctx, def, get_selection_completion_mode(def));
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_selection_to_alley(ecs_world_t *world, int selection_index,
                                    int alley_slot_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // Validate selection index is in range
  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection.count);
    return false;
  }

  // Validate alley slot index is in range
  if (alley_slot_index < 0 || alley_slot_index >= ALLEY_SIZE) {
    cli_render_logf("[Ability] Invalid alley slot index %d", alley_slot_index);
    return false;
  }

  ecs_entity_t target = ctx->selection.cards[selection_index];
  if (target == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Verify the ability allows selecting to alley
  if (!def->can_select_to_alley) {
    cli_render_logf("[Ability] This ability does not allow selecting to alley");
    return false;
  }

  // Verify the target is an entity card
  const Type *target_type = ecs_get(world, target, Type);
  if (!target_type || target_type->value != CARD_TYPE_ENTITY) {
    cli_render_logf("[Ability] Only entity cards can be selected to alley");
    return false;
  }

  // Validate the selection target if validation function exists
  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->runtime.source_card,
                                      ctx->runtime.owner, target)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  // Get game state and zones
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t alley = gs->zones[player_num].alley;
  ecs_entity_t selection_zone = gs->zones[player_num].selection;

  int8_t from_index =
      azk_get_card_index_in_zone(world, target, selection_zone);

  // Check for displaced card at the target slot
  ecs_entity_t displaced_card = 0;
  ecs_entities_t alley_cards = ecs_get_ordered_children(world, alley);
  for (int32_t i = 0; i < alley_cards.count; i++) {
    const ZoneIndex *zi = ecs_get(world, alley_cards.ids[i], ZoneIndex);
    if (zi && zi->index == alley_slot_index) {
      displaced_card = alley_cards.ids[i];
      break;
    }
  }

  // Handle displaced card - only allow replacement if alley is full
  if (displaced_card != 0) {
    bool alley_full = alley_cards.count >= ALLEY_SIZE;
    if (!alley_full) {
      // Reject: slot occupied but alley has empty slots available
      cli_render_logf("[Ability] Alley slot %d is already occupied",
                      alley_slot_index);
      return false;
    }
    // Alley is full - forced replacement allowed
    discard_card_for_replacement(world, displaced_card);
    cli_render_logf("[Ability] Displaced card from alley slot %d",
                    alley_slot_index);
  }

  // Move the selected card to the alley slot
  ecs_add_pair(world, target, EcsChildOf, alley);
  ecs_set(world, target, ZoneIndex, {.index = (uint8_t)alley_slot_index});

  // Log selection -> alley movement
  azk_log_card_zone_moved(world, target, GLOG_ZONE_SELECTION, from_index,
                          GLOG_ZONE_ALLEY, (int8_t)alley_slot_index);

  // Reset tap state for newly placed card
  ecs_set(world, target, TapState, {.tapped = false, .cooldown = false});

  azk_debug_validate_zone_indices(world, alley, ALLEY_SIZE);

  // Track that an entity was played to alley this turn (for abilities like
  // STT02-005)
  GameState *gs_mut = ecs_singleton_get_mut(world, GameState);
  gs_mut->entities_played_alley_this_turn[player_num]++;
  gs_mut->cards_played_this_turn[player_num]++;
  gs_mut->next_card_play_cost_reduction[player_num] = 0;
  ecs_singleton_modified(world, GameState);

  // Queue on-play ability for the played entity (if it has one)
  // Will be processed after current ability completes (including bottom deck)
  azk_trigger_on_play_ability(world, target, ctx->runtime.owner);

  cli_render_logf("[Ability] Selected card to alley slot %d", alley_slot_index);

  azk_record_selection_pick(ctx, selection_index, target);

  // Check if we've picked enough
  if (ctx->selection.picked_count >= ctx->selection.pick_max) {
    return azk_finish_selection_resolution(
        world, ctx, def, get_selection_completion_mode(def));
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_can_select_to_equip(ecs_world_t *world, int selection_index,
                             int entity_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx == NULL || ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    return false;
  }

  if (entity_index < 0 || entity_index > GARDEN_SIZE) {
    return false;
  }

  const ecs_entity_t weapon = ctx->selection.cards[selection_index];
  if (weapon == 0) {
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (def == NULL || !def->can_select_to_equip) {
    return false;
  }

  const Type *weapon_type = ecs_get(world, weapon, Type);
  if (weapon_type == NULL || weapon_type->value != CARD_TYPE_WEAPON) {
    return false;
  }

  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->runtime.source_card,
                                      ctx->runtime.owner, weapon)) {
    return false;
  }

  const GameState *gs = ecs_singleton_get(world, GameState);
  if (gs == NULL) {
    return false;
  }

  const uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t target_entity = 0;
  if (entity_index < GARDEN_SIZE) {
    target_entity = find_card_in_zone_index(world, gs->zones[player_num].garden,
                                            entity_index);
  } else {
    target_entity =
        find_leader_card_in_zone(world, gs->zones[player_num].leader);
  }

  if (target_entity == 0) {
    return false;
  }

  const CurStats *weapon_stats = ecs_get(world, weapon, CurStats);
  const CurStats *target_stats = ecs_get(world, target_entity, CurStats);
  if (weapon_stats == NULL || target_stats == NULL) {
    return false;
  }

  const ReequipOrigin *reequip_origin = ecs_get(world, weapon, ReequipOrigin);
  if (def->selection_to_equip_is_reequip && reequip_origin != NULL &&
      reequip_origin->previous_host == target_entity) {
    return false;
  }

  return true;
}

bool azk_process_selection_to_equip(ecs_world_t *world, int selection_index,
                                    int entity_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // Validate selection index is in range
  if (selection_index < 0 || selection_index >= ctx->selection.count) {
    cli_render_logf("[Ability] Invalid selection index %d (count=%d)",
                    selection_index, ctx->selection.count);
    return false;
  }

  // Validate entity index (0-4 for garden, 5 for leader)
  if (entity_index < 0 || entity_index > GARDEN_SIZE) {
    cli_render_logf("[Ability] Invalid entity index %d", entity_index);
    return false;
  }

  ecs_entity_t weapon = ctx->selection.cards[selection_index];
  if (weapon == 0) {
    cli_render_logf("[Ability] Selection slot %d is empty", selection_index);
    return false;
  }

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  // Verify the ability allows selecting to equip
  if (!def->can_select_to_equip) {
    cli_render_logf("[Ability] This ability does not allow selecting to equip");
    return false;
  }

  // Verify the weapon is a weapon card
  const Type *weapon_type = ecs_get(world, weapon, Type);
  if (!weapon_type || weapon_type->value != CARD_TYPE_WEAPON) {
    cli_render_logf("[Ability] Only weapon cards can be selected to equip");
    return false;
  }

  // Validate the selection target if validation function exists
  if (def->validate_selection_target &&
      !def->validate_selection_target(world, ctx->runtime.source_card,
                                      ctx->runtime.owner, weapon)) {
    cli_render_logf("[Ability] Selection target validation failed");
    return false;
  }

  // Get game state and find target entity
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->runtime.owner);
  ecs_entity_t selection_zone = gs->zones[player_num].selection;
  ecs_entity_t target_entity = 0;

  if (entity_index < GARDEN_SIZE) {
    target_entity = find_card_in_zone_index(world, gs->zones[player_num].garden,
                                            entity_index);
  } else {
    // entity_index == GARDEN_SIZE means leader
    target_entity =
        find_leader_card_in_zone(world, gs->zones[player_num].leader);
  }

  if (target_entity == 0) {
    cli_render_logf("[Ability] No entity at slot %d", entity_index);
    return false;
  }

  // Get weapon and target stats
  const CurStats *weapon_stats = ecs_get(world, weapon, CurStats);
  const CurStats *target_stats = ecs_get(world, target_entity, CurStats);
  if (!weapon_stats || !target_stats) {
    cli_render_logf("[Ability] Missing stats for weapon or target");
    return false;
  }

  const ReequipOrigin *reequip_origin =
      ecs_get(world, weapon, ReequipOrigin);
  if (def->selection_to_equip_is_reequip && reequip_origin != NULL &&
      reequip_origin->previous_host == target_entity) {
    cli_render_logf("[Ability] Re-equipped weapon must move to a different host");
    return false;
  }

  int8_t from_index = azk_get_card_index_in_zone(world, weapon, selection_zone);

  // Log selection -> equipped movement before the deferred reparent changes
  // the visible parent chain.
  azk_log_card_zone_moved(world, weapon, GLOG_ZONE_SELECTION, from_index,
                          GLOG_ZONE_EQUIPPED, -1);

  // Attach weapon to target (ChildOf relationship)
  ecs_add_pair(world, weapon, EcsChildOf, target_entity);

  // Apply weapon attack bonus directly (because ChildOf is deferred)
  apply_weapon_attack_bonus(world, target_entity, weapon_stats->cur_atk);
  apply_weapon_combat_modifier_if_any(world, weapon, target_entity);

  cli_render_logf("[Ability] Equipped weapon (+%d attack) to entity at slot %d",
                  weapon_stats->cur_atk, entity_index);

  // Trigger weapon abilities (when-equipped always, on-play only for new equips)
  if (!def->selection_to_equip_is_reequip) {
    azk_trigger_on_play_ability(world, weapon, ctx->runtime.owner);
  }
  azk_trigger_when_equipped_ability(world, weapon, ctx->runtime.owner);
  azk_trigger_when_equipped_ability(world, target_entity, ctx->runtime.owner);

  if (def->selection_to_equip_is_reequip) {
    ecs_remove(world, weapon, ReequipOrigin);
  } else {
    GameState *gs_mut = ecs_singleton_get_mut(world, GameState);
    gs_mut->cards_played_this_turn[player_num]++;
    gs_mut->next_card_play_cost_reduction[player_num] = 0;
    ecs_singleton_modified(world, GameState);
  }

  azk_record_selection_pick(ctx, selection_index, weapon);

  // Check if we've picked enough
  if (ctx->selection.picked_count >= ctx->selection.pick_max) {
    return azk_finish_selection_resolution(
        world, ctx, def, AZK_SELECTION_COMPLETION_CLEAR_IF_STILL_ACTIVE);
  }

  ecs_singleton_modified(world, AbilityContext);
  return true;
}

bool azk_process_skip_selection(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_SELECTION_PICK) {
    return false;
  }

  // For "up to" effects - allow skipping even if we haven't picked any
  // This is different from effect selection where min determines if we can skip

  const CardId *card_id = ecs_get(world, ctx->runtime.source_card, CardId);
  if (!card_id) {
    azk_clear_ability_context(world);
    return false;
  }

  const AbilityDef *def = get_context_ability_def(world, ctx);
  if (!def) {
    azk_clear_ability_context(world);
    return false;
  }

  if (!def->selection_pick_is_optional) {
    cli_render_logf("[Ability] Selection pick cannot be skipped");
    return false;
  }

  cli_render_logf("[Ability] Skipped selection pick");
  return azk_finish_selection_resolution(
      world, ctx, def, get_selection_completion_mode(def));
}

bool azk_process_bottom_deck(ecs_world_t *world, int selection_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_BOTTOM_DECK) {
    return false;
  }

  return azk_bottom_deck_selection_card(world, ctx, selection_index);
}

bool azk_process_top_deck(ecs_world_t *world, int selection_index) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_BOTTOM_DECK) {
    return false;
  }

  return azk_top_deck_selection_card(world, ctx, selection_index);
}

bool azk_process_bottom_deck_all(ecs_world_t *world) {
  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);

  if (ctx->runtime.phase != ABILITY_PHASE_BOTTOM_DECK) {
    return false;
  }

  return azk_bottom_deck_all_selection_cards(world, ctx);
}

bool azk_is_in_ability_phase(ecs_world_t *world) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return ctx && ctx->runtime.phase != ABILITY_PHASE_NONE;
}

AbilityPhase azk_get_ability_phase(ecs_world_t *world) {
  const AbilityContext *ctx = ecs_singleton_get(world, AbilityContext);
  return ctx ? ctx->runtime.phase : ABILITY_PHASE_NONE;
}

bool azk_trigger_main_ability(ecs_world_t *world, ecs_entity_t card,
                              ecs_entity_t owner, int8_t action_index) {
  // Check if card is frozen (frozen cards cannot activate abilities)
  if (ecs_has(world, card, Frozen)) {
    cli_render_logf("[Ability] Card is frozen and cannot activate abilities");
    return false;
  }

  ecs_entity_t ability_entity =
      azk_find_card_action_ability(world, card, action_index);
  if (ability_entity == 0 ||
      !azk_ability_has_timing(world, ability_entity, ecs_id(AMain))) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def_for_entity(world, ability_entity);
  if (!def || !def->has_ability) {
    return false;
  }

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, card, owner);
  if (def->cost_req.min > 0 && available_cost_targets < def->cost_req.min) {
    cli_render_logf("[Ability] Main ability has no valid cost targets "
                    "(available=%u, required_min=%u)",
                    (unsigned)available_cost_targets,
                    (unsigned)def->cost_req.min);
    return false;
  }

  // Validate the ability can be activated
  if (def->validate && !def->validate(world, card, owner)) {
    cli_render_logf("[Ability] Main ability validation failed");
    return false;
  }

  return azk_begin_ability(
      world, ability_entity, owner, def,
      &(AbilityBeginOptions){
          .is_optional = def->is_optional,
          .available_cost_targets = available_cost_targets,
          .select_effects_when_max_positive = true,
          .apply_costs_before_effect_selection = true,
          .clear_context_on_immediate_resolve = true,
          .applied_log = "[Ability] Applied main ability with no targets",
          .cost_selection_log =
              "[Ability] Triggered main ability, selecting cost targets",
          .effect_selection_log =
              "[Ability] Triggered main ability, selecting effect targets",
          .selection_log =
              "[Ability] Triggered main ability, started selection flow",
      });
}

bool azk_trigger_spell_ability(ecs_world_t *world, ecs_entity_t spell_card,
                               ecs_entity_t owner, int8_t action_index) {
  ecs_entity_t ability_entity =
      azk_find_card_action_ability(world, spell_card, action_index);
  if (ability_entity == 0) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def_for_entity(world, ability_entity);
  if (!def || !def->has_ability) {
    return false;
  }

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, spell_card, owner);
  if (def->cost_req.min > 0 && available_cost_targets < def->cost_req.min) {
    cli_render_logf("[Ability] Spell has no valid cost targets "
                    "(available=%u, required_min=%u)",
                    (unsigned)available_cost_targets,
                    (unsigned)def->cost_req.min);
    return false;
  }

  uint8_t available_effect_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, spell_card, owner);

  return azk_begin_ability(
      world, ability_entity, owner, def,
      &(AbilityBeginOptions){
          .is_optional = false,
          .available_cost_targets = available_cost_targets,
          .available_effect_targets = available_effect_targets,
          .clamp_effect_expected_to_available = true,
          .select_effects_when_max_positive = true,
          .apply_costs_before_effect_selection = false,
          .clear_context_on_immediate_resolve = true,
          .applied_log = "[Ability] Applied spell with no targets",
          .cost_selection_log =
              "[Ability] Spell triggered, selecting cost targets",
          .effect_selection_log =
              "[Ability] Spell triggered, selecting effect targets",
          .selection_log = "[Ability] Spell triggered, started selection flow",
      });
}

bool azk_trigger_leader_response_ability(ecs_world_t *world, ecs_entity_t card,
                                         ecs_entity_t owner,
                                         int8_t action_index) {
  ecs_entity_t ability_entity =
      azk_find_card_action_ability(world, card, action_index);
  if (ability_entity == 0 ||
      !azk_ability_has_timing(world, ability_entity, ecs_id(AResponse))) {
    return false;
  }

  const AbilityDef *def = azk_get_ability_def_for_entity(world, ability_entity);
  if (!def || !def->has_ability) {
    return false;
  }

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, card, owner);
  if (def->cost_req.min > 0 && available_cost_targets < def->cost_req.min) {
    cli_render_logf("[Ability] Leader response has no valid cost targets "
                    "(available=%u, required_min=%u)",
                    (unsigned)available_cost_targets,
                    (unsigned)def->cost_req.min);
    return false;
  }

  return azk_begin_ability(
      world, ability_entity, owner, def,
      &(AbilityBeginOptions){
          .is_optional = false,
          .available_cost_targets = available_cost_targets,
          .select_effects_when_max_positive = true,
          .apply_costs_before_effect_selection = false,
          .clear_context_on_immediate_resolve = true,
          .applied_log = "[Ability] Applied leader response with no targets",
          .cost_selection_log =
              "[Ability] Leader response triggered, selecting cost targets",
          .effect_selection_log =
              "[Ability] Leader response triggered, selecting effect targets",
          .selection_log =
              "[Ability] Leader response triggered, started selection flow",
      });
}

bool azk_queue_triggered_effect(ecs_world_t *world, ecs_entity_t ability_entity,
                                ecs_entity_t owner, uint8_t timing_tag) {
  if (ability_entity == 0 ||
      ecs_get(world, ability_entity, AbilityInstance) == NULL) {
    return false;
  }

  ecs_entity_t card = azk_get_ability_source_card(world, ability_entity);

  if (ability_entity == 0 || card == 0) {
    return false;
  }

  if (ecs_has(world, ability_entity, AOnceTurn)) {
    const AbilityRepeatContext *repeat_ctx =
        ecs_get(world, ability_entity, AbilityRepeatContext);
    if (repeat_ctx && repeat_ctx->was_applied) {
      return false;
    }
  }

  TriggeredEffectQueue *queue =
      ecs_singleton_get_mut(world, TriggeredEffectQueue);

  if (queue->count >= MAX_TRIGGERED_EFFECT_QUEUE) {
    cli_render_logf("[Ability] Triggered effect queue full, cannot queue");
    return false;
  }

  queue->effects[queue->count].ability_entity = ability_entity;
  queue->effects[queue->count].source_card = card;
  queue->effects[queue->count].owner = owner;
  queue->effects[queue->count].action_index =
      get_action_index_for_ability(world, ability_entity);
  const AbilityInstance *instance =
      ecs_get(world, ability_entity, AbilityInstance);
  queue->effects[queue->count].registry_order =
      instance != NULL ? instance->registry_order : 0;
  queue->effects[queue->count].timing_tag = timing_tag;
  queue->count++;

  azk_log_effect_queued(world, card,
                        queue->effects[queue->count - 1].action_index,
                        timing_tag);

  cli_render_logf("[Ability] Queued triggered effect (tag=%d, count=%d)",
                  timing_tag, queue->count);
  ecs_singleton_modified(world, TriggeredEffectQueue);
  return true;
}

bool azk_has_queued_triggered_effects(ecs_world_t *world) {
  const TriggeredEffectQueue *queue =
      ecs_singleton_get(world, TriggeredEffectQueue);
  return queue && queue->count > 0;
}

// Helper to get the expected timing tag ecs_id for a given tag index
static ecs_id_t get_timing_tag_id(uint8_t tag_index) {
  switch (tag_index) {
  case TIMING_TAG_ON_PLAY:
    return ecs_id(AOnPlay);
  case TIMING_TAG_START_OF_TURN:
    return ecs_id(AStartOfTurn);
  case TIMING_TAG_START_OF_EACH_TURN:
    return ecs_id(AStartOfEachTurn);
  case TIMING_TAG_END_OF_TURN:
    return ecs_id(AEndOfTurn);
  case TIMING_TAG_WHEN_EQUIPPING:
    return ecs_id(AWhenEquipping);
  case TIMING_TAG_WHEN_EQUIPPED:
    return ecs_id(AWhenEquipped);
  case TIMING_TAG_WHEN_ATTACKING:
    return ecs_id(AWhenAttacking);
  case TIMING_TAG_AFTER_ATTACKING:
    return ecs_id(AAfterAttacking);
  case TIMING_TAG_WHEN_ATTACKED:
    return ecs_id(AWhenAttacked);
  case TIMING_TAG_WHEN_TAKES_DAMAGE:
    return ecs_id(AWhenTakesDamage);
  case TIMING_TAG_WHEN_DEALS_DAMAGE:
    return ecs_id(AWhenDealsDamage);
  case TIMING_TAG_WHEN_RETURNED_TO_HAND:
    return ecs_id(AWhenReturnedToHand);
  case TIMING_TAG_WHEN_DESTROYED:
    return ecs_id(AWhenDestroyed);
  case TIMING_TAG_WHEN_SACRIFICED:
    return ecs_id(AWhenSacrificed);
  case TIMING_TAG_ON_GATE_PORTAL:
    return ecs_id(AOnGatePortal);
  case TIMING_TAG_WHEN_ENTERS_GARDEN:
    return ecs_id(AWhenEntersGarden);
  default:
    return 0;
  }
}

bool azk_process_triggered_effect_queue(ecs_world_t *world) {
  TriggeredEffectQueue *queue =
      ecs_singleton_get_mut(world, TriggeredEffectQueue);

  if (!queue || queue->count == 0) {
    return false;
  }

  // Pop first effect (FIFO)
  PendingTriggeredEffect effect = queue->effects[0];

  // Shift remaining effects
  for (uint8_t i = 0; i < queue->count - 1; i++) {
    queue->effects[i] = queue->effects[i + 1];
  }
  queue->count--;
  ecs_singleton_modified(world, TriggeredEffectQueue);

  // Log effect enabled (ability is now being processed)
  azk_log_effect_enabled(world, effect.source_card, effect.action_index);

  cli_render_logf("[Ability] Processing queued effect (tag=%d, remaining=%d)",
                  effect.timing_tag, queue->count);

  // Now process the effect - card should be in correct zone after deferred ops
  // flushed
  ecs_entity_t ability_entity = effect.ability_entity;
  ecs_entity_t card = effect.source_card;
  ecs_entity_t owner = effect.owner;
  if (ability_entity == 0 || card == 0) {
    cli_render_logf("[Ability] Queued effect: source ability is no longer valid");
    return false;
  }

  // Get card ID
  const CardId *card_id = ecs_get(world, card, CardId);
  if (!card_id) {
    cli_render_logf("[Ability] Queued effect: card no longer valid");
    return false;
  }

  const AbilityDef *def = azk_get_ability_def_for_entity(world, ability_entity);
  if (!def || !def->has_ability) {
    cli_render_logf("[Ability] Queued effect: no ability definition");
    return false;
  }

  if (ecs_has(world, ability_entity, AOnceTurn)) {
    const AbilityRepeatContext *repeat_ctx =
        ecs_get(world, ability_entity, AbilityRepeatContext);
    if (repeat_ctx && repeat_ctx->was_applied) {
      cli_render_logf("[Ability] Queued effect: once-per-turn already used");
      return false;
    }
  }

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, card, owner);
  if (def->cost_req.min > 0 && available_cost_targets < def->cost_req.min) {
    cli_render_logf("[Ability] Queued effect has no valid cost targets "
                    "(available=%u, required_min=%u), skipping",
                    (unsigned)available_cost_targets,
                    (unsigned)def->cost_req.min);
    return false;
  }

  // Check if it's the correct timing tag
  ecs_id_t expected_tag = get_timing_tag_id(effect.timing_tag);
  if (expected_tag == 0 ||
      !azk_ability_has_timing(world, ability_entity, expected_tag)) {
    cli_render_logf("[Ability] Queued effect: timing tag mismatch");
    return false;
  }

  // Validate the ability can be activated (card is now in correct zone)
  if (def->validate && !def->validate(world, card, owner)) {
    cli_render_logf("[Ability] Queued effect: validation failed");
    return false;
  }

  uint8_t available_effect_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, card, owner);
  if (def->effect_req.min > 0 &&
      available_effect_targets < def->effect_req.min) {
    cli_render_logf("[Ability] Queued effect has no valid effect targets "
                    "(available=%u, required_min=%u), skipping",
                    (unsigned)available_effect_targets,
                    (unsigned)def->effect_req.min);
    return false;
  }

  return azk_begin_ability(
      world, ability_entity, owner, def,
      &(AbilityBeginOptions){
          .is_optional = def->is_optional,
          .enter_confirmation_when_optional = true,
          .transfer_control_on_user_input = true,
          .available_cost_targets = available_cost_targets,
          .available_effect_targets = available_effect_targets,
          .clamp_effect_expected_to_available = true,
          .select_effects_when_max_positive = true,
          .apply_costs_before_effect_selection = true,
          .clear_context_on_immediate_resolve = true,
          .confirmation_log =
              "[Ability] Triggered optional ability, waiting for confirmation",
          .applied_log = "[Ability] Applied mandatory ability with no targets",
          .cost_selection_log =
              "[Ability] Triggered mandatory ability, selecting cost targets",
          .effect_selection_log =
              "[Ability] Triggered mandatory ability, selecting effect targets",
          .selection_log =
              "[Ability] Triggered mandatory ability, started selection flow",
      });
}

void azk_trigger_return_to_hand_observers(ecs_world_t *world,
                                          ecs_entity_t bounced_card) {
  (void)bounced_card; // May be used in future for filtering

  const GameState *gs = ecs_singleton_get(world, GameState);

  // Scan BOTH players' gardens for cards with AWhenReturnedToHand ability
  for (int p = 0; p < MAX_PLAYERS_PER_MATCH; p++) {
    ecs_entity_t player = gs->players[p];
    ecs_entity_t garden = gs->zones[p].garden;
    ecs_entities_t garden_cards = ecs_get_ordered_children(world, garden);

    for (int32_t i = 0; i < garden_cards.count; i++) {
      ecs_entity_t card = garden_cards.ids[i];
      ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
      uint8_t ability_count = azk_collect_card_timed_abilities(
          world, card, ecs_id(AWhenReturnedToHand), abilities,
          AZK_MAX_CARD_ABILITIES);
      for (uint8_t j = 0; j < ability_count; ++j) {
        const AbilityDef *def =
            azk_get_ability_def_for_entity(world, abilities[j]);
        if (def != NULL && def->validate && !def->validate(world, card, player)) {
          continue;
        }

        azk_queue_triggered_effect(world, abilities[j], player,
                                   TIMING_TAG_WHEN_RETURNED_TO_HAND);
        cli_render_logf("[Ability] Queued return-to-hand observer for card %s",
                        ecs_get_name(world, card));
      }
    }
  }
}

void azk_trigger_gate_portal_ability(ecs_world_t *world, ecs_entity_t gate_card,
                                     ecs_entity_t portaled_card,
                                     uint8_t garden_index,
                                     ecs_entity_t owner) {
  // Verify this is actually a gate card
  ecs_assert(is_card_type(world, gate_card, CARD_TYPE_GATE),
             ECS_INVALID_PARAMETER,
             "Gate portal ability triggered on non-gate card %llu",
             (unsigned long long)gate_card);

  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  uint8_t ability_count = azk_collect_card_timed_abilities(
      world, gate_card, ecs_id(AOnGatePortal), abilities,
      AZK_MAX_CARD_ABILITIES);
  ecs_entity_t ability_entity = ability_count > 0 ? abilities[0] : 0;
  const AbilityDef *def = azk_get_ability_def_for_entity(world, ability_entity);
  ecs_assert(def != NULL && azk_ability_def_has_timing(def, ecs_id(AOnGatePortal)),
             ECS_INVALID_PARAMETER,
             "Gate card %llu has no registered portal ability",
             (unsigned long long)gate_card);

  const AbilityScratchState initial_scratch = {
      .kind = ABILITY_SCRATCH_GATE_PORTAL,
      .data.gate_portal =
          {
              .portaled_card = portaled_card,
              .garden_index = garden_index,
          },
  };

  AbilityContext *ctx = ecs_singleton_get_mut(world, AbilityContext);
  ecs_assert(ctx != NULL, ECS_INVALID_PARAMETER,
             "AbilityContext singleton missing for gate portal ability");
  const AbilityContext saved_ctx = *ctx;

  // Gate portal abilities may validate and count targets from scratch state.
  azk_reset_ability_context_state(ctx);
  ctx->runtime.source_ability = ability_entity;
  ctx->runtime.source_card = gate_card;
  ctx->runtime.owner = owner;
  const AbilityInstance *instance = ecs_get(world, ability_entity, AbilityInstance);
  if (instance != NULL) {
    ctx->runtime.action_index = instance->action_index;
    ctx->runtime.registry_order = instance->registry_order;
  }
  ctx->scratch = initial_scratch;

  // Validate can still fail (e.g., conditional effects)
  if (def->validate && !def->validate(world, gate_card, owner)) {
    *ctx = saved_ctx;
    ecs_singleton_modified(world, AbilityContext);
    return;
  }

  uint8_t available_cost_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_COST, gate_card, owner);
  uint8_t available_effect_targets = azk_count_ability_target_choices(
      world, def, ABILITY_TARGET_SCOPE_EFFECT, gate_card, owner);

  *ctx = saved_ctx;

  bool is_active = azk_begin_ability(
      world, ability_entity, owner, def,
      &(AbilityBeginOptions){
          .is_optional = def->is_optional,
          .enter_confirmation_when_optional = true,
          .available_cost_targets = available_cost_targets,
          .available_effect_targets = available_effect_targets,
          .clamp_effect_expected_to_available = true,
          .select_effects_when_max_positive = true,
          .apply_costs_before_effect_selection = true,
          .initial_scratch = initial_scratch,
          .confirmation_log =
              "[Ability] Gate portal triggered optional ability, waiting for "
              "confirmation",
          .clear_context_on_immediate_resolve = true,
          .cost_selection_log =
              "[Ability] Gate portal ability, selecting cost targets",
          .effect_selection_log =
              "[Ability] Gate portal ability, selecting effect targets",
          .selection_log =
              "[Ability] Gate portal triggered multi-step ability",
      });

  if (!is_active) {
    cli_render_logf("[Ability] Applied gate portal ability for %s",
                    ecs_get_name(world, gate_card));
  }
}
