#include "systems/start_phase.h"
#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "components/components.h"
#include "utils/cli_rendering_util.h"
#include "utils/deck_utils.h"
#include "utils/game_log_util.h"
#include "utils/ability_util.h"
#include "utils/status_util.h"
#include "utils/zone_util.h"

void DrawCard(ecs_world_t *world, GameState *gs) {
  ecs_entity_t deck_zone = gs->zones[gs->active_player_index].deck;
  ecs_entity_t hand_zone = gs->zones[gs->active_player_index].hand;

  ecs_entity_t out_cards[1] = {0};
  if (!move_cards_to_zone(world, deck_zone, hand_zone, 1, out_cards)) {
    cli_render_log("[DrawCard] No cards in deck");

    gs->winner = (gs->active_player_index + 1) % 2;

    return;
  }

  cli_render_logf("[DrawCard] Drew card %s", ecs_get_name(world, out_cards[0]));
}

void GrantIKZ(ecs_world_t *world, GameState *gs) {
  ecs_entity_t ikz_pile_zone = gs->zones[gs->active_player_index].ikz_pile;
  ecs_entity_t ikz_area_zone = gs->zones[gs->active_player_index].ikz_area;

  ecs_entity_t out_card[1] = {0};
  if (!move_cards_to_zone(world, ikz_pile_zone, ikz_area_zone, 1, out_card)) {
    return;
  }

  cli_render_log("[GrantIKZ] IKZ granted");
}

static void reset_once_per_turn_in_zone(ecs_world_t *world, ecs_entity_t zone) {
  ecs_entity_t abilities[AZK_MAX_CARD_ABILITIES] = {0};
  ecs_entities_t cards = ecs_get_ordered_children(world, zone);
  for (int i = 0; i < cards.count; i++) {
    ecs_entity_t card = cards.ids[i];
    uint8_t ability_count =
        azk_collect_card_abilities(world, card, abilities, AZK_MAX_CARD_ABILITIES);
    for (uint8_t j = 0; j < ability_count; ++j) {
      if (!ecs_has(world, abilities[j], AOnceTurn)) {
        continue;
      }

      ecs_set(world, abilities[j], AbilityRepeatContext,
              {
                  .is_once_per_turn = true,
                  .was_applied = false
              });
    }

    ecs_iter_t child_it = ecs_children(world, card);
    while (ecs_children_next(&child_it)) {
      for (int j = 0; j < child_it.count; ++j) {
        ecs_entity_t child = child_it.entities[j];
        uint8_t child_ability_count = azk_collect_card_abilities(
            world, child, abilities, AZK_MAX_CARD_ABILITIES);
        for (uint8_t k = 0; k < child_ability_count; ++k) {
          if (!ecs_has(world, abilities[k], AOnceTurn)) {
            continue;
          }

          ecs_set(world, abilities[k], AbilityRepeatContext,
                  {
                      .is_once_per_turn = true,
                      .was_applied = false
                  });
        }
      }
    }
  }
}

static void ResetOnceTurnAbilities(ecs_world_t *world, GameState *gs) {
  // Reset once-per-turn abilities for BOTH players at start of each turn
  for (int i = 0; i < MAX_PLAYERS_PER_MATCH; i++) {
    reset_once_per_turn_in_zone(world, gs->zones[i].garden);
    reset_once_per_turn_in_zone(world, gs->zones[i].alley);
    reset_once_per_turn_in_zone(world, gs->zones[i].leader);
  }
}

static void UntapAllCards(ecs_world_t *world, GameState *gs) {
  ecs_entity_t garden_zone = gs->zones[gs->active_player_index].garden;
  untap_all_cards_in_zone(world, garden_zone);

  ecs_entity_t alley_zone = gs->zones[gs->active_player_index].alley;
  untap_all_cards_in_zone(world, alley_zone);

  ecs_entity_t ikz_area_zone = gs->zones[gs->active_player_index].ikz_area;
  untap_all_cards_in_zone(world, ikz_area_zone);

  ecs_entity_t leader_zone = gs->zones[gs->active_player_index].leader;
  untap_all_cards_in_zone(world, leader_zone);

  ecs_entity_t gate_zone = gs->zones[gs->active_player_index].gate;
  untap_all_cards_in_zone(world, gate_zone);

  cli_render_log("[UntapAllCards] Untapped all cards in zones");
}

static void handle_phase_transition(ecs_world_t *world, GameState *gs) {
  if (is_game_over(world)) {
    gs->phase = PHASE_END_MATCH;
  } else {
    gs->phase = PHASE_MAIN;
  }
}

static bool should_draw_start_of_turn_card(const GameState *gs) {
  return gs->turn_number > 1;
}

void StartPhase(ecs_iter_t *it) {
  ecs_world_t *world = it->world;
  GameState *gs = ecs_field(it, GameState, 0);

  // Increment turn number
  gs->turn_number++;

  // Log turn started
  azk_log_turn_started(world, gs->active_player_index, gs->turn_number);

  // Tick down status effect durations for BOTH players before untap
  tick_status_effects_for_player(world, 0);
  tick_status_effects_for_player(world, 1);

  // Reset entities played this turn counters
  for (int i = 0; i < MAX_PLAYERS_PER_MATCH; i++) {
    gs->entities_played_garden_this_turn[i] = 0;
    gs->entities_played_alley_this_turn[i] = 0;
    gs->cards_played_this_turn[i] = 0;
    gs->discarded_cards_this_turn[i] = 0;
    gs->entities_returned_to_hand_this_turn[i] = 0;
    gs->next_card_play_cost_reduction[i] = 0;
  }

  // Reset once-per-turn abilities for both players
  ResetOnceTurnAbilities(world, gs);

  UntapAllCards(world, gs);
  if (should_draw_start_of_turn_card(gs)) {
    DrawCard(world, gs);
  } else {
    cli_render_log("[StartPhase] Skipping opening draw for the starting player");
  }
  GrantIKZ(world, gs);
  azk_trigger_start_of_turn_abilities(world);
  azk_trigger_start_of_each_turn_abilities(world);

  handle_phase_transition(world, gs);

  cli_render_log("[StartPhase] Start phase");
}

void init_start_phase_system(ecs_world_t *world) {
  ecs_system(world, {
    .entity = ecs_entity(world, {
      .name = "StartPhaseSystem",
      .add = ecs_ids(TStartOfTurn)
    }),
    .query.terms = {
      { .id = ecs_id(GameState), .src.id = ecs_id(GameState) },
    },
    .callback = StartPhase
  });
}
