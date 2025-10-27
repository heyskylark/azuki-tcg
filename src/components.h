#ifndef AZUKI_ECS_COMPONENTS_H
#define AZUKI_ECS_COMPONENTS_H

#include <stdint.h>
#include <flecs.h>

typedef enum {
  PHASE_PREGAME_MULLIGAN_P0 = 0,
  PHASE_PREGAME_MULLIGAN_P1 = 1,
  PHASE_START_OF_TURN = 2,
  PHASE_MAIN = 3,
  PHASE_COMBAT_DECLARED = 4,
  PHASE_RESPONSE_WINDOW = 5,
  PHASE_COMBAT_RESOLVE = 6,
  PHASE_END_TURN = 7,
  PHASE_END_MATCH = 8
} Phase;

typedef enum {
  ACT_NOOP = 0,
  ACT_PLAY_ENTITY_TO_GARDEN = 1,
  ACT_PLAY_ENTITY_TO_ALLEY = 2,
  /* 3-5 reserved for weapons/spells once those flows are online */
  ACT_ATTACK = 6,
  ACT_DECLARE_DEFENDER = 7,
  /* 8-9 reserved for gate portal + ability activation */
  ACT_END_TURN = 10,
  ACT_MULLIGAN_KEEP = 11,
  ACT_MULLIGAN_SHUFFLE = 12
} ActionType;

typedef struct { uint32_t seed; int8_t active; uint8_t phase; uint8_t response_window; int8_t winner; } GameState;
typedef struct { uint8_t pid; } PlayerId;
typedef struct { ecs_entity_t player; } Owner;
typedef struct { int8_t attack, health; } BaseStats;
typedef struct { int8_t cur_atk, cur_hp; } CurStats;
typedef struct { uint8_t tapped, cooldown; } TapState;
typedef struct { uint8_t element; } Element;
typedef struct { uint8_t gate_points; } GatePoints;
typedef struct { int8_t ikz_cost; } IKZCost;

extern ECS_COMPONENT_DECLARE(GameState);
extern ECS_COMPONENT_DECLARE(PlayerId);
extern ECS_COMPONENT_DECLARE(Owner);
extern ECS_COMPONENT_DECLARE(BaseStats);
extern ECS_COMPONENT_DECLARE(CurStats);
extern ECS_COMPONENT_DECLARE(TapState);
extern ECS_COMPONENT_DECLARE(Element);
extern ECS_COMPONENT_DECLARE(GatePoints);
extern ECS_COMPONENT_DECLARE(IKZCost);

extern ECS_ENTITY_DECLARE(Rel_InZone);
extern ECS_ENTITY_DECLARE(Rel_OwnedBy);

/* Card Type Tags */
extern ECS_TAG_DECLARE(TLeader);
extern ECS_TAG_DECLARE(TGate);
extern ECS_TAG_DECLARE(TEntity);
extern ECS_TAG_DECLARE(TWeapon);
extern ECS_TAG_DECLARE(TSpell);
extern ECS_TAG_DECLARE(TIKZ);

/* Board Zone Tags */
extern ECS_TAG_DECLARE(ZDeck);
extern ECS_TAG_DECLARE(ZHand);
extern ECS_TAG_DECLARE(ZGarden);
extern ECS_TAG_DECLARE(ZAlley);
extern ECS_TAG_DECLARE(ZIKZPileTag);
extern ECS_TAG_DECLARE(ZIKZAreaTag);
extern ECS_TAG_DECLARE(ZDiscard);

void azk_register_components(ecs_world_t *world);

#endif
