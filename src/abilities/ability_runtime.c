#include "abilities/ability_runtime.h"

#include <string.h>

#include "utils/card_utils.h"
#include "utils/zone_util.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "abilities/ability.h"

static bool card_has_weapon_attached(ecs_world_t *world, ecs_entity_t card) {
  ecs_iter_t it = ecs_children(world, card);
  while (ecs_children_next(&it)) {
    for (int i = 0; i < it.count; i++) {
      ecs_entity_t child = it.entities[i];
      if (ecs_has_id(world, child, TWeapon)) {
        return true;
      }
    }
  }
  return false;
}

uint16_t azk_make_ability_uid(const AbilityDef *def) {
  return (uint16_t)((def->card_def_id << 8) | def->ability_index);
}

bool azk_is_once_per_turn_used(ecs_world_t *world, ecs_entity_t card, uint8_t ability_index) {
  const AbilityUsage *usage = ecs_get(world, card, AbilityUsage);
  if (usage == NULL) {
    return false;
  }
  return (usage->once_per_turn_mask & (1u << ability_index)) != 0;
}

void azk_mark_once_per_turn_used(ecs_world_t *world, ecs_entity_t card, uint8_t ability_index) {
  AbilityUsage *usage = ecs_get_mut(world, card, AbilityUsage);
  if (usage == NULL) {
    return;
  }
  usage->once_per_turn_mask |= (1u << ability_index);
}

static void reset_once_per_turn_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    AbilityUsage *usage = ecs_get_mut(world, card, AbilityUsage);
    if (usage) {
      usage->once_per_turn_mask = 0;
    }
    ecs_iter_t child_it = ecs_children(world, card);
    while (ecs_children_next(&child_it)) {
      for (int j = 0; j < child_it.count; j++) {
        ecs_entity_t child = child_it.entities[j];
        AbilityUsage *child_usage = ecs_get_mut(world, child, AbilityUsage);
        if (child_usage) {
          child_usage->once_per_turn_mask = 0;
        }
      }
    }
  }
}

void azk_reset_once_per_turn_for_player(ecs_world_t *world, ecs_entity_t player) {
  uint8_t player_number = get_player_number(world, player);
  const GameState *gs = ecs_singleton_get(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  reset_once_per_turn_in_zone(world, gs->zones[player_number].garden);
  reset_once_per_turn_in_zone(world, gs->zones[player_number].alley);
  reset_once_per_turn_in_zone(world, gs->zones[player_number].hand);
  reset_once_per_turn_in_zone(world, gs->zones[player_number].leader);
  reset_once_per_turn_in_zone(world, gs->zones[player_number].gate);
}

static bool ability_target_already_selected(ecs_entity_t target, const ecs_entity_t *buffer, uint8_t count) {
  for (uint8_t i = 0; i < count; i++) {
    if (buffer[i] == target) {
      return true;
    }
  }
  return false;
}

static uint8_t clamp_expected(uint8_t requested) {
  return requested > AZK_MAX_ABILITY_SELECTIONS ? AZK_MAX_ABILITY_SELECTIONS : requested;
}

static bool collect_ikz_payment_for_trigger(
  ecs_world_t *world,
  ecs_entity_t ikz_zone,
  uint8_t ikz_cost,
  ecs_entity_t out_cards[AZK_MAX_IKZ_PAYMENT],
  uint8_t *out_count
) {
  if (ikz_cost == 0) {
    *out_count = 0;
    return true;
  }

  ecs_assert(ikz_cost <= AZK_MAX_IKZ_PAYMENT, ECS_INVALID_PARAMETER, "IKZ cost %d exceeds buffer", ikz_cost);

  uint8_t count = 0;
  if (get_tappable_ikz_cards(world, ikz_zone, ikz_cost, &count, out_cards, false) < 0) {
    return false;
  }

  if (count < ikz_cost) {
    return false;
  }

  *out_count = ikz_cost;
  return true;
}

bool azk_begin_or_resolve_ability(
  ecs_world_t *world,
  GameState *gs,
  AbilityContext *actx,
  ecs_entity_t player,
  ecs_entity_t source_card,
  const AbilityDef *def,
  const ecs_entity_t *ikz_cards,
  uint8_t ikz_card_count,
  bool from_trigger
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(actx != NULL, ECS_INVALID_PARAMETER, "AbilityContext is null");
  ecs_assert(def != NULL, ECS_INVALID_PARAMETER, "AbilityDef is null");

  uint8_t cost_expected = 0;
  uint8_t cost_min = 0;
  if (azk_ability_has_target_requirement(&def->cost_targets)) {
    uint8_t cost_available = azk_ability_target_count(
      world,
      gs,
      player,
      source_card,
      def->cost_targets,
      ABILITY_TARGET_NONE
    );
    uint8_t capped = clamp_expected(def->cost_targets.max);
    cost_expected = cost_available > 0
      ? (cost_available < capped ? cost_available : capped)
      : 0;
    cost_min = def->cost_targets.min > cost_expected ? cost_expected : def->cost_targets.min;
  }

  uint8_t effect_expected = 0;
  uint8_t effect_min = 0;
  if (azk_ability_has_target_requirement(&def->effect_targets)) {
    uint8_t effect_available = azk_ability_target_count(
      world,
      gs,
      player,
      source_card,
      def->effect_targets,
      ABILITY_TARGET_NONE
    );
    uint8_t capped = clamp_expected(def->effect_targets.max);
    effect_expected = effect_available > 0
      ? (effect_available < capped ? effect_available : capped)
      : 0;
    effect_min = def->effect_targets.min > effect_expected ? effect_expected : def->effect_targets.min;
  }

  bool needs_cost_selection = cost_expected > 0;
  bool needs_effect_selection = effect_expected > 0;
  bool requires_prompt = from_trigger && def->optional;

  if (!needs_cost_selection && !needs_effect_selection && !requires_prompt) {
    AbilityContext temp = {
      .has_pending = false,
      .from_trigger = from_trigger,
      .optional = def->optional,
      .awaiting_consent = false,
      .source_card = source_card,
      .player = player,
      .ability_uid = azk_make_ability_uid(def),
      .phase = ABILITY_SELECTION_NONE,
      .cost_min = cost_min,
      .cost_expected = cost_expected,
      .effect_min = effect_min,
      .effect_expected = effect_expected,
      .ikz_card_count = ikz_card_count
    };
    memcpy(temp.ikz_cards, ikz_cards, sizeof(ecs_entity_t) * ikz_card_count);
    azk_resolve_ability(world, gs, &temp, def);
    return false;
  }

  azk_clear_ability_context(actx);
  actx->has_pending = true;
  actx->from_trigger = from_trigger;
  actx->optional = def->optional;
  actx->awaiting_consent = requires_prompt;
  actx->source_card = source_card;
  actx->player = player;
  actx->ability_uid = azk_make_ability_uid(def);
  actx->cost_expected = cost_expected;
  actx->cost_min = cost_min;
  actx->effect_expected = effect_expected;
  actx->effect_min = effect_min;
  actx->phase = requires_prompt
    ? ABILITY_SELECTION_PROMPT
    : (needs_cost_selection ? ABILITY_SELECTION_COST : ABILITY_SELECTION_EFFECT);
  actx->ikz_card_count = ikz_card_count;
  memcpy(actx->ikz_cards, ikz_cards, sizeof(ecs_entity_t) * ikz_card_count);

  // Return true to signal the caller that selections or consent are required.
  return true;
}

bool azk_append_ability_target(
  AbilityContext *actx,
  AbilitySelectionPhase phase,
  ecs_entity_t target
) {
  if (actx == NULL || !actx->has_pending) {
    return false;
  }

  ecs_entity_t *buffer = NULL;
  uint8_t *filled = NULL;
  uint8_t expected = 0;

  switch (phase) {
    case ABILITY_SELECTION_COST:
      buffer = actx->cost_targets;
      filled = &actx->cost_filled;
      expected = actx->cost_expected;
      break;
    case ABILITY_SELECTION_EFFECT:
      buffer = actx->effect_targets;
      filled = &actx->effect_filled;
      expected = actx->effect_expected;
      break;
    default:
      return false;
  }

  if (*filled >= expected || *filled >= AZK_MAX_ABILITY_SELECTIONS) {
    return false;
  }

  if (ability_target_already_selected(target, buffer, *filled)) {
    return false;
  }

  buffer[*filled] = target;
  (*filled)++;
  return true;
}

void azk_try_finish_ability(
  ecs_world_t *world,
  GameState *gs,
  AbilityContext *actx,
  bool manual_finish,
  bool user_declined
) {
  if (actx == NULL || !actx->has_pending) {
    return;
  }

  uint8_t card_def_id = (uint8_t)(actx->ability_uid >> 8);
  uint8_t ability_index = (uint8_t)(actx->ability_uid & 0xFF);
  const AbilityDef *def = azk_find_ability_def(card_def_id, ability_index);
  if (def == NULL) {
    azk_clear_ability_context(actx);
    return;
  }

  if (user_declined) {
    if (actx->optional) {
      cli_render_log("[Ability] Optional ability declined");
      azk_clear_ability_context(actx);
    }
    return;
  }

  if (actx->awaiting_consent) {
    actx->awaiting_consent = false;
    if (actx->cost_expected == 0 && actx->effect_expected == 0) {
      actx->phase = ABILITY_SELECTION_NONE;
      azk_resolve_ability(world, gs, actx, def);
      return;
    }
    if (actx->cost_expected > 0) {
      actx->phase = ABILITY_SELECTION_COST;
      return;
    }
    actx->phase = ABILITY_SELECTION_EFFECT;
    return;
  }

  bool cost_done = actx->cost_expected == 0 || actx->cost_filled >= actx->cost_expected;
  if (!cost_done && manual_finish && actx->cost_filled >= actx->cost_min) {
    cost_done = true;
  }

  if (!cost_done) {
    return;
  }

  if (actx->phase == ABILITY_SELECTION_COST && actx->effect_expected > 0) {
    actx->phase = ABILITY_SELECTION_EFFECT;
    // Do not fall through to resolve until effect picks are done.
    return;
  }

  bool effect_done = actx->effect_expected == 0 || actx->effect_filled >= actx->effect_expected;
  if (!effect_done && manual_finish && actx->effect_filled >= actx->effect_min) {
    effect_done = true;
  }

  if (!effect_done) {
    return;
  }

  azk_resolve_ability(world, gs, actx, def);
}

bool azk_validate_target_against_req(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ecs_entity_t source_card,
  ecs_entity_t target,
  AbilityTargetRequirement req,
  AbilityTargetType required_type_override
) {
  if (req.max == 0 || req.type == ABILITY_TARGET_NONE) {
    return true;
  }

  AbilityTargetType type = required_type_override != ABILITY_TARGET_NONE ? required_type_override : req.type;
  ecs_entity_t target_owner = ecs_get_target(world, target, Rel_OwnedBy, 0);
  ecs_assert(target_owner != 0, ECS_INVALID_PARAMETER, "Target %d has no owner", target);
  uint8_t target_owner_number = get_player_number(world, target_owner);
  uint8_t player_number = get_player_number(world, player);
  bool is_friendly = target_owner_number == player_number;

  switch (type) {
    case ABILITY_TARGET_FRIENDLY_GARDEN_ENTITY:
      return is_friendly && ecs_has_id(world, target, TEntity);
    case ABILITY_TARGET_FRIENDLY_ALLEY_ENTITY: {
      if (!is_friendly) return false;
      ecs_entity_t alley = gs->zones[player_number].alley;
      return ecs_get_target(world, target, EcsChildOf, 0) == alley;
    }
    case ABILITY_TARGET_FRIENDLY_ENTITY_WITH_WEAPON:
      return is_friendly && ecs_has_id(world, target, TEntity) && card_has_weapon_attached(world, target);
    case ABILITY_TARGET_FRIENDLY_LEADER:
      return is_friendly && ecs_has_id(world, target, TLeader);
    case ABILITY_TARGET_ENEMY_GARDEN_ENTITY:
      return !is_friendly && ecs_has_id(world, target, TEntity);
    case ABILITY_TARGET_ENEMY_LEADER:
      return !is_friendly && ecs_has_id(world, target, TLeader);
    case ABILITY_TARGET_ENEMY_LEADER_OR_GARDEN_ENTITY:
      return !is_friendly && (ecs_has_id(world, target, TEntity) || ecs_has_id(world, target, TLeader));
    case ABILITY_TARGET_ANY_LEADER_OR_GARDEN_ENTITY:
      return ecs_has_id(world, target, TEntity) || ecs_has_id(world, target, TLeader);
    case ABILITY_TARGET_NONE:
    default:
      return true;
  }
}

int azk_resolve_ability(
  ecs_world_t *world,
  GameState *gs,
  AbilityContext *actx,
  const AbilityDef *def
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(actx != NULL, ECS_INVALID_PARAMETER, "AbilityContext is null");
  ecs_assert(def != NULL, ECS_INVALID_PARAMETER, "AbilityDef is null");

  // Apply IKZ payments
  for (uint8_t i = 0; i < actx->ikz_card_count; i++) {
    set_card_to_tapped(world, actx->ikz_cards[i]);
  }

  // Apply tap cost
  if (def->cost.tap_self) {
    set_card_to_tapped(world, actx->source_card);
  }

  // Apply sacrifice cost
  if (def->cost.sacrifice_self) {
    discard_card(world, actx->source_card);
  }

  // Mark once-per-turn usage
  if (def->once_per_turn) {
    azk_mark_once_per_turn_used(world, actx->source_card, def->ability_index);
  }

  // NOTE: Effects are stubs for now; just log the selection set.
  cli_render_logf(
    "[Ability] Resolved %s with %u cost target(s) and %u effect target(s)",
    def->name ? def->name : "unknown",
    actx->cost_filled,
    actx->effect_filled
  );

  azk_clear_ability_context(actx);
  return 0;
}

void azk_clear_ability_context(AbilityContext *actx) {
  if (!actx) return;
  memset(actx, 0, sizeof(*actx));
}

bool azk_trigger_abilities_for_card(
  ecs_world_t *world,
  GameState *gs,
  AbilityContext *actx,
  ecs_entity_t source_card,
  AbilityTiming timing
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState is null");
  ecs_assert(actx != NULL, ECS_INVALID_PARAMETER, "AbilityContext is null");

  const CardId *card_id = ecs_get(world, source_card, CardId);
  if (card_id == NULL) {
    return false;
  }

  const AbilityDef *matching[AZK_MAX_ABILITIES_PER_CARD] = {0};
  size_t def_count = azk_collect_triggered_abilities(card_id->id, timing, matching);
  if (def_count == 0) {
    return false;
  }

  if (actx->has_pending) {
    cli_render_log("[Ability] Pending selection; delaying triggered ability");
    return false;
  }

  ecs_entity_t owner = ecs_get_target(world, source_card, Rel_OwnedBy, 0);
  ecs_assert(owner != 0, ECS_INVALID_PARAMETER, "Card %d has no owner", source_card);

  bool triggered_any = false;

  for (size_t i = 0; i < def_count; i++) {
    const AbilityDef *def = matching[i];
    if (def == NULL) continue;

    if (def->once_per_turn && azk_is_once_per_turn_used(world, source_card, def->ability_index)) {
      continue;
    }

    if (def->cost.tap_self && (is_card_tapped(world, source_card) || is_card_cooldown(world, source_card))) {
      continue;
    }

    // Validate target availability before prompting the player.
    if (azk_ability_has_target_requirement(&def->cost_targets) &&
        !azk_ability_targets_available(world, gs, owner, source_card, def->cost_targets, ABILITY_TARGET_NONE)) {
      continue;
    }
    if (azk_ability_has_target_requirement(&def->effect_targets) &&
        !azk_ability_targets_available(world, gs, owner, source_card, def->effect_targets, ABILITY_TARGET_NONE)) {
      continue;
    }

    ecs_entity_t ikz_cards[AZK_MAX_IKZ_PAYMENT] = {0};
    uint8_t ikz_count = 0;
    ecs_entity_t ikz_zone = gs->zones[get_player_number(world, owner)].ikz_area;
    if (!collect_ikz_payment_for_trigger(world, ikz_zone, def->cost.ikz_cost, ikz_cards, &ikz_count)) {
      continue;
    }

    bool pending = azk_begin_or_resolve_ability(
      world,
      gs,
      actx,
      owner,
      source_card,
      def,
      ikz_cards,
      ikz_count,
      true
    );

    triggered_any = true;

    if (pending) {
      break;
    }
  }

  return triggered_any;
}
