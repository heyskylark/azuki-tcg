#ifndef AZUKI_VALIDATION_ACTION_VALIDATION_H
#define AZUKI_VALIDATION_ACTION_VALIDATION_H

#include <stdbool.h>
#include <flecs.h>

#include "validation/action_intents.h"

bool azk_validate_play_entity_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ZonePlacementType placement_type,
  const UserAction *action,
  bool log_errors,
  PlayEntityIntent *out_intent
);

bool azk_validate_gate_portal_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors,
  GatePortalIntent *out_intent
);

bool azk_validate_attack_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors,
  AttackIntent *out_intent
);

bool azk_validate_attach_weapon_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  const UserAction *action,
  bool log_errors,
  AttachWeaponIntent *out_intent
);

bool azk_validate_simple_action(
  ecs_world_t *world,
  const GameState *gs,
  ecs_entity_t player,
  ActionType type,
  bool log_errors
);

#endif
