#include "components.h"
#include "generated/card_defs.h"

ECS_COMPONENT_DECLARE(ActionContext);
ECS_COMPONENT_DECLARE(GameState);
ECS_COMPONENT_DECLARE(PlayerNumber);
ECS_COMPONENT_DECLARE(PlayerId);
ECS_COMPONENT_DECLARE(ZoneIndex);
ECS_COMPONENT_DECLARE(IKZToken);

ECS_ENTITY_DECLARE(Rel_OwnedBy);

ECS_TAG_DECLARE(ZDeck);
ECS_TAG_DECLARE(ZHand);
ECS_TAG_DECLARE(ZLeader);
ECS_TAG_DECLARE(ZGate);
ECS_TAG_DECLARE(ZGarden);
ECS_TAG_DECLARE(ZAlley);
ECS_TAG_DECLARE(ZIKZPileTag);
ECS_TAG_DECLARE(ZIKZAreaTag);
ECS_TAG_DECLARE(ZDiscard);
  
ECS_TAG_DECLARE(TMulligan);
ECS_TAG_DECLARE(TStartOfTurn);
ECS_TAG_DECLARE(TMain);
ECS_TAG_DECLARE(TResponseWindow);
ECS_TAG_DECLARE(TCombatResolve);
ECS_TAG_DECLARE(TEndTurn);
ECS_TAG_DECLARE(TEndMatch);

void azk_register_components(ecs_world_t *world) {
  ECS_COMPONENT_DEFINE(world, ActionContext);
  ECS_COMPONENT_DEFINE(world, GameState);
  ECS_COMPONENT_DEFINE(world, PlayerNumber);
  ECS_COMPONENT_DEFINE(world, PlayerId);
  ECS_COMPONENT_DEFINE(world, ZoneIndex);
  ECS_COMPONENT_DEFINE(world, IKZToken);
  
  {
    ecs_entity_desc_t desc = {
      .name = "Rel_OwnedBy",
      .add = (ecs_id_t[]){
        EcsRelationship,
        EcsAcyclic,
        0
      }
    };
    Rel_OwnedBy = ecs_entity_init(world, &desc);
    ecs_assert(Rel_OwnedBy != 0, ECS_INVALID_PARAMETER, "failed to create entity Rel_OwnedBy");
    ecs_id(Rel_OwnedBy) = Rel_OwnedBy;
  }

  ECS_TAG_DEFINE(world, ZDeck);
  ECS_TAG_DEFINE(world, ZHand);
  ECS_TAG_DEFINE(world, ZLeader);
  ECS_TAG_DEFINE(world, ZGate);
  ECS_TAG_DEFINE(world, ZGarden);
  ECS_TAG_DEFINE(world, ZAlley);
  ECS_TAG_DEFINE(world, ZIKZPileTag);
  ECS_TAG_DEFINE(world, ZIKZAreaTag);
  ECS_TAG_DEFINE(world, ZDiscard);

  ECS_TAG_DEFINE(world, TMulligan);
  ECS_TAG_DEFINE(world, TStartOfTurn);
  ECS_TAG_DEFINE(world, TMain);
  ECS_TAG_DEFINE(world, TResponseWindow);
  ECS_TAG_DEFINE(world, TCombatResolve);
  ECS_TAG_DEFINE(world, TEndTurn);
  ECS_TAG_DEFINE(world, TEndMatch);

  azk_register_card_def_resources(world);
}
