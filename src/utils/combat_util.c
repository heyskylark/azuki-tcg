#include "utils/combat_util.h"
#include "abilities/ability_registry.h"
#include "abilities/ability_system.h"
#include "components/abilities.h"
#include "components/components.h"
#include "utils/card_utils.h"
#include "utils/cli_rendering_util.h"
#include "utils/damage_util.h"
#include "utils/game_log_util.h"
#include "utils/status_util.h"

static void trigger_lightning_kanabo_if_present(ecs_world_t *world,
                                                ecs_entity_t dealer,
                                                ecs_entity_t recipient,
                                                int8_t damage_dealt) {
  if (damage_dealt <= 0 || dealer == 0 || recipient == 0) {
    return;
  }

  ecs_iter_t child_it = ecs_children(world, dealer);
  while (ecs_children_next(&child_it)) {
    for (int i = 0; i < child_it.count; ++i) {
      ecs_entity_t weapon = child_it.entities[i];
      const CardId *weapon_id = ecs_get(world, weapon, CardId);
      if (weapon_id == NULL || weapon_id->id != CARD_DEF_AZK01_044) {
        continue;
      }

      if (ecs_has(world, weapon, AOnceTurn)) {
        const AbilityRepeatContext *repeat_ctx =
            ecs_get(world, weapon, AbilityRepeatContext);
        if (repeat_ctx && repeat_ctx->was_applied) {
          continue;
        }

        ecs_set(world, weapon, AbilityRepeatContext,
                {.is_once_per_turn = true, .was_applied = true});
      }

      apply_shocked(world, recipient, 2);
    }
  }
}

static int8_t calculate_combat_damage(ecs_world_t *world, ecs_entity_t dealer,
                                      ecs_entity_t recipient,
                                      int8_t base_damage) {
  int16_t damage = base_damage;
  damage += get_total_outgoing_combat_damage_modifier(world, dealer);
  damage += get_total_incoming_combat_damage_modifier(world, recipient);
  damage -= get_total_carapace_value(world, recipient);

  if (damage < 0) {
    damage = 0;
  } else if (damage > INT8_MAX) {
    damage = INT8_MAX;
  }

  return (int8_t)damage;
}

int attack(
  ecs_world_t *world,
  const AttackIntent *intent
) {
  ecs_assert(world != NULL, ECS_INVALID_PARAMETER, "World is null");
  ecs_assert(intent != NULL, ECS_INVALID_PARAMETER, "AttackIntent is null");

  GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  gs->last_combat = (LastCombatResult){0};
  tap_card(world, intent->attacking_card);

  CombatState combat_state = {
    .attacking_card = intent->attacking_card,
    .defender_card = intent->defending_card,
    .defender_intercepted = false,
  };
  gs->combat_state = combat_state;

  return 0;
}

void resolve_combat(ecs_world_t *world) {
  GameState *gs = ecs_singleton_get_mut(world, GameState);
  ecs_assert(gs != NULL, ECS_INVALID_PARAMETER, "GameState singleton missing");

  ecs_assert(gs->combat_state.attacking_card != 0, ECS_INVALID_PARAMETER, "Combat state attacking card not set");
  ecs_assert(gs->combat_state.defender_card != 0, ECS_INVALID_PARAMETER, "Combat state defender card not set");

  CurStats *attacking_card_cur_stats = ecs_get_mut(world, gs->combat_state.attacking_card, CurStats);
  ecs_assert(attacking_card_cur_stats != NULL, ECS_INVALID_PARAMETER, "Attacking card cur stats not found");
  CurStats *defender_card_cur_stats = ecs_get_mut(world, gs->combat_state.defender_card, CurStats);
  ecs_assert(defender_card_cur_stats != NULL, ECS_INVALID_PARAMETER, "Defender card cur stats not found");
  const bool attacker_is_leader =
      ecs_has(world, gs->combat_state.attacking_card, TLeader);
  const bool defender_is_leader =
      ecs_has(world, gs->combat_state.defender_card, TLeader);
  ecs_entity_t defender_parent =
      ecs_get_target(world, gs->combat_state.defender_card, EcsChildOf, 0);

  // If defender is frozen, no damage is dealt by either party
  // (Attacker can never be frozen due to attack validation)
  bool defender_frozen = ecs_has(world, gs->combat_state.defender_card, Frozen);
  int8_t attacker_damage = 0;
  int8_t defender_damage = 0;
  if (defender_frozen) {
    cli_render_log("[Combat] Defender is frozen - no damage dealt");
  } else {
    attacker_damage = calculate_combat_damage(
        world, gs->combat_state.defender_card, gs->combat_state.attacking_card,
        defender_card_cur_stats->cur_atk);
    defender_damage = calculate_combat_damage(
        world, gs->combat_state.attacking_card, gs->combat_state.defender_card,
        attacking_card_cur_stats->cur_atk);

    attacking_card_cur_stats->cur_hp -= attacker_damage;
    defender_card_cur_stats->cur_hp -= defender_damage;

    trigger_lightning_kanabo_if_present(world, gs->combat_state.attacking_card,
                                        gs->combat_state.defender_card,
                                        defender_damage);
    trigger_lightning_kanabo_if_present(world, gs->combat_state.defender_card,
                                        gs->combat_state.attacking_card,
                                        attacker_damage);

    azk_record_damage_event(world, gs->combat_state.defender_card,
                            gs->combat_state.attacking_card, attacker_damage,
                            false);
    azk_record_damage_event(world, gs->combat_state.attacking_card,
                            gs->combat_state.defender_card, defender_damage,
                            false);

    // Log combat damage (attacker deals defender_damage to defender, takes attacker_damage)
    azk_log_combat_damage(world, gs->combat_state.attacking_card,
                          gs->combat_state.defender_card,
                          defender_damage, attacker_damage,
                          attacker_damage, defender_damage);
  }

  bool attacking_leader_defeated = false;
  bool defender_leader_defeated = false;
  bool attacker_destroyed = false;
  bool defender_destroyed = false;
  // Capture destruction outcomes before discard_card resets CurStats.
  if (attacking_card_cur_stats->cur_hp <= 0) {
    if (azk_card_has_godmode_in_play(world, gs->combat_state.attacking_card)) {
      attacking_card_cur_stats->cur_hp = 1;
    } else if (attacker_is_leader) {
      attacking_leader_defeated = true;
      // Log entity died (leader defeated by combat)
      azk_log_entity_died(world, gs->combat_state.attacking_card,
                          GLOG_DEATH_COMBAT);
    } else {
      attacker_destroyed = true;
      // Log entity died before discarding (combat death)
      azk_log_entity_died(world, gs->combat_state.attacking_card,
                          GLOG_DEATH_COMBAT);
      discard_card(world, gs->combat_state.attacking_card);
    }
  }

  if (defender_card_cur_stats->cur_hp <= 0) {
    if (azk_card_has_godmode_in_play(world, gs->combat_state.defender_card)) {
      defender_card_cur_stats->cur_hp = 1;
    } else if (defender_is_leader) {
      defender_leader_defeated = true;
      // Log entity died (leader defeated by combat)
      azk_log_entity_died(world, gs->combat_state.defender_card,
                          GLOG_DEATH_COMBAT);
    } else {
      defender_destroyed = true;
      // Log entity died before discarding (combat death)
      azk_log_entity_died(world, gs->combat_state.defender_card,
                          GLOG_DEATH_COMBAT);
      discard_card(world, gs->combat_state.defender_card);
    }
  }

  if (attacking_leader_defeated && defender_leader_defeated) {
    gs->winner = 2;
    // Log game ended as draw
    azk_log_game_ended(world, 2, GLOG_END_LEADER_DEFEATED);
  } else if (attacking_leader_defeated) {
    gs->winner = (gs->active_player_index + 1) % 2;
    // Log game ended
    azk_log_game_ended(world, gs->winner, GLOG_END_LEADER_DEFEATED);
  } else if (defender_leader_defeated) {
    gs->winner = gs->active_player_index;
    // Log game ended
    azk_log_game_ended(world, gs->winner, GLOG_END_LEADER_DEFEATED);
  }

  gs->last_combat = (LastCombatResult){
      .attacker = gs->combat_state.attacking_card,
      .defender = gs->combat_state.defender_card,
      .defender_was_leader = defender_is_leader,
      .defender_was_garden_entity =
          defender_parent == gs->zones[(gs->active_player_index + 1) % MAX_PLAYERS_PER_MATCH].garden,
      .defender_destroyed = defender_destroyed,
      .attacker_destroyed = attacker_destroyed,
      .damage_to_defender = defender_damage,
      .damage_to_attacker = attacker_damage,
  };

  // TODO: Resolve "after attacking" or "when attacked" effects that trigger from the outcome
}
