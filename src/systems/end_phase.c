#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "systems/end_phase.h"
#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/entity_util.h"
#include "utils/game_log_util.h"
#include "utils/status_util.h"
#include "utils/card_utils.h"
#include "utils/zone_util.h"

static void discard_end_of_turn_marked_cards_in_zone(ecs_world_t *world,
                                                     ecs_entity_t zone) {
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int32_t i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    if (!ecs_has(world, card, SacrificeAtEndOfTurn) ||
        ecs_has(world, card, TLeader)) {
      continue;
    }

    discard_equipped_weapon_cards(world, card);
    sacrifice_card(world, card);
  }
}

static void clear_spent_or_expiring_ikz_token(ecs_world_t *world,
                                              ecs_entity_t player) {
  IKZToken *ikz_token = ecs_get_mut(world, player, IKZToken);
  if (ikz_token == NULL || ikz_token->ikz_token == 0) {
    return;
  }

  const TapState *tap_state = ecs_get(world, ikz_token->ikz_token, TapState);
  const bool spent = tap_state != NULL && tap_state->tapped;
  if (!spent && !ikz_token->expires_eot) {
    return;
  }

  ecs_delete(world, ikz_token->ikz_token);
  *ikz_token = (IKZToken){0};
  ecs_modified(world, player, IKZToken);
}

void HandleEndPhase(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);
  uint8_t ending_player_index = gs->active_player_index;
  uint8_t next_player_index =
      (ending_player_index + 1) % MAX_PLAYERS_PER_MATCH;

  if (!gs->end_of_turn_abilities_queued) {
    bool queued_any = azk_trigger_end_of_turn_abilities(world);
    gs->end_of_turn_abilities_queued = true;
    if (queued_any) {
      cli_render_log("[EndPhase] Queued active player's end-of-turn abilities");
      return;
    }
  }

  discard_end_of_turn_marked_cards_in_zone(world, gs->zones[ending_player_index].garden);
  discard_end_of_turn_marked_cards_in_zone(world, gs->zones[ending_player_index].alley);
  discard_end_of_turn_marked_cards_in_zone(world, gs->zones[next_player_index].garden);
  discard_end_of_turn_marked_cards_in_zone(world, gs->zones[next_player_index].alley);
  clear_spent_or_expiring_ikz_token(world, gs->players[ending_player_index]);
  clear_spent_or_expiring_ikz_token(world, gs->players[next_player_index]);

  ecs_entity_t active_player_garden_zone = gs->zones[ending_player_index].garden;
  ecs_entities_t active_player_garden_cards =
      ecs_get_ordered_children(world, active_player_garden_zone);

  cli_render_logf("[EndPhase] Resetting entity health for active player's garden cards");
  for (int32_t i = 0; i < active_player_garden_cards.count; i++) {
    ecs_entity_t garden_card = active_player_garden_cards.ids[i];

    reset_entity_health(world, garden_card);
    discard_equipped_weapon_cards(world, garden_card);
  }

  // Expire end-of-turn attack modifiers for both players
  expire_eot_attack_modifiers_in_zone(world, gs->zones[ending_player_index].garden);
  expire_eot_attack_modifiers_in_zone(world, gs->zones[ending_player_index].leader);
  expire_eot_combat_damage_modifiers_in_zone(world,
                                             gs->zones[ending_player_index].garden);
  expire_eot_combat_damage_modifiers_in_zone(world,
                                             gs->zones[ending_player_index].leader);
  expire_eot_combat_damage_modifiers_in_zone(world,
                                             gs->zones[ending_player_index].alley);
  expire_eot_carapace_modifiers_in_zone(world, gs->zones[ending_player_index].garden);
  expire_eot_carapace_modifiers_in_zone(world, gs->zones[ending_player_index].leader);
  expire_eot_carapace_modifiers_in_zone(world, gs->zones[ending_player_index].alley);
  expire_eot_health_modifiers_in_zone(world, gs->zones[ending_player_index].garden);
  expire_eot_health_modifiers_in_zone(world, gs->zones[ending_player_index].leader);
  tick_end_of_turn_effects_for_player(world, ending_player_index);

  ecs_entity_t defending_player_garden_zone = gs->zones[next_player_index].garden;
  ecs_entities_t defending_player_garden_cards = ecs_get_ordered_children(world, defending_player_garden_zone);

  cli_render_logf("[EndPhase] Resetting entity health for defending player's garden cards");
  for (int32_t i = 0; i < defending_player_garden_cards.count; i++) {
    ecs_entity_t garden_card = defending_player_garden_cards.ids[i];

    reset_entity_health(world, garden_card);
  }

  // Expire end-of-turn attack modifiers for defending player
  expire_eot_attack_modifiers_in_zone(world, gs->zones[next_player_index].garden);
  expire_eot_attack_modifiers_in_zone(world, gs->zones[next_player_index].leader);
  expire_eot_combat_damage_modifiers_in_zone(world,
                                             gs->zones[next_player_index].garden);
  expire_eot_combat_damage_modifiers_in_zone(world,
                                             gs->zones[next_player_index].leader);
  expire_eot_combat_damage_modifiers_in_zone(world,
                                             gs->zones[next_player_index].alley);
  expire_eot_carapace_modifiers_in_zone(world, gs->zones[next_player_index].garden);
  expire_eot_carapace_modifiers_in_zone(world, gs->zones[next_player_index].leader);
  expire_eot_carapace_modifiers_in_zone(world, gs->zones[next_player_index].alley);
  expire_eot_health_modifiers_in_zone(world, gs->zones[next_player_index].garden);
  expire_eot_health_modifiers_in_zone(world, gs->zones[next_player_index].leader);
  tick_end_of_turn_effects_for_player(world, next_player_index);

  ecs_entity_t leader_card = find_leader_card_in_zone(world, gs->zones[ending_player_index].leader);
  discard_equipped_weapon_cards(world, leader_card);

  // Log turn ended before transitioning
  azk_log_turn_ended(world, ending_player_index, gs->turn_number);

  gs->end_of_turn_abilities_queued = false;
  gs->phase = PHASE_START_OF_TURN;
  gs->active_player_index = next_player_index;

  cli_render_log("[EndPhase] End phase");
}

void init_end_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "EndPhaseSystem",
      .add = ecs_ids(TEndTurn)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) },
    },
    .callback = HandleEndPhase
  });
}
