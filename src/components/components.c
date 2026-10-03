#include "components/components.h"
#include "abilities/ability_registry.h"
#include "components/abilities.h"
#include "components/game_log.h"
#include "generated/card_defs.h"

ECS_COMPONENT_DECLARE(ActionContext);
ECS_COMPONENT_DECLARE(AbilityContext);
ECS_COMPONENT_DECLARE(AbilityInstance);
ECS_COMPONENT_DECLARE(GameState);
ECS_COMPONENT_DECLARE(PlayerNumber);
ECS_COMPONENT_DECLARE(PlayerId);
ECS_COMPONENT_DECLARE(ZoneIndex);
ECS_COMPONENT_DECLARE(IKZToken);
ECS_COMPONENT_DECLARE(ReequipOrigin);
ECS_COMPONENT_DECLARE(PendingDamageRedirectQueue);
ECS_COMPONENT_DECLARE(STT03BobuState);
ECS_COMPONENT_DECLARE(STT03MiharuState);
ECS_COMPONENT_DECLARE(STT04KuraiState);
ECS_COMPONENT_DECLARE(TriggeredEffectQueue);
ECS_COMPONENT_DECLARE(PassiveBuffQueue);
ECS_COMPONENT_DECLARE(DeckReorderQueue);
ECS_COMPONENT_DECLARE(PhaseGateCache);

ECS_ENTITY_DECLARE(Rel_OwnedBy);
ECS_ENTITY_DECLARE(Rel_AbilityOf);

ECS_TAG_DECLARE(ZDeck);
ECS_TAG_DECLARE(ZHand);
ECS_TAG_DECLARE(ZLeader);
ECS_TAG_DECLARE(ZGate);
ECS_TAG_DECLARE(ZGarden);
ECS_TAG_DECLARE(ZAlley);
ECS_TAG_DECLARE(ZIKZPileTag);
ECS_TAG_DECLARE(ZIKZAreaTag);
ECS_TAG_DECLARE(ZDiscard);
ECS_TAG_DECLARE(ZSelection);
ECS_TAG_DECLARE(RewardGeneratedIKZCredit);
  
ECS_TAG_DECLARE(TMulligan);
ECS_TAG_DECLARE(TStartOfTurn);
ECS_TAG_DECLARE(TMain);
ECS_TAG_DECLARE(TResponseWindow);
ECS_TAG_DECLARE(TCombatResolve);
ECS_TAG_DECLARE(TEndTurnAction);
ECS_TAG_DECLARE(TEndTurn);
ECS_TAG_DECLARE(TEndMatch);
ECS_TAG_DECLARE(TAbilityResolution);

static void on_owned_card_id_set(ecs_iter_t *it) {
  for (int32_t i = 0; i < it->count; ++i) {
    ecs_entity_t entity = it->entities[i];
    if (ecs_get_target(it->world, entity, Rel_OwnedBy, 0) == 0) {
      continue;
    }

    attach_ability_components(it->world, entity);
  }
}

void azk_register_components(ecs_world_t *world) {
  ECS_COMPONENT_DEFINE(world, ActionContext);
  ECS_COMPONENT_DEFINE(world, AbilityContext);
  ECS_COMPONENT_DEFINE(world, AbilityInstance);
  ECS_COMPONENT_DEFINE(world, GameState);
  ECS_COMPONENT_DEFINE(world, PlayerNumber);
  ECS_COMPONENT_DEFINE(world, PlayerId);
  ECS_COMPONENT_DEFINE(world, ZoneIndex);
  ECS_COMPONENT_DEFINE(world, IKZToken);
  ECS_COMPONENT_DEFINE(world, ReequipOrigin);
  ECS_COMPONENT_DEFINE(world, PendingDamageRedirectQueue);
  ECS_COMPONENT_DEFINE(world, STT03BobuState);
  ECS_COMPONENT_DEFINE(world, STT03MiharuState);
  ECS_COMPONENT_DEFINE(world, STT04KuraiState);
  ECS_COMPONENT_DEFINE(world, TriggeredEffectQueue);
  ECS_COMPONENT_DEFINE(world, PassiveBuffQueue);
  ECS_COMPONENT_DEFINE(world, DeckReorderQueue);
  ECS_COMPONENT_DEFINE(world, PhaseGateCache);

  // Initialize TriggeredEffectQueue singleton
  ecs_singleton_set(world, TriggeredEffectQueue, {.count = 0});
  ecs_singleton_set(world, PendingDamageRedirectQueue, {.count = 0});

  // Initialize PassiveBuffQueue singleton
  ecs_singleton_set(world, PassiveBuffQueue, {.count = 0});
  ecs_singleton_set(world, DeckReorderQueue, {.count = 0});

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

  {
    ecs_entity_desc_t desc = {
      .name = "Rel_AbilityOf",
      .add = (ecs_id_t[]){
        EcsRelationship,
        EcsAcyclic,
        0
      }
    };
    Rel_AbilityOf = ecs_entity_init(world, &desc);
    ecs_assert(Rel_AbilityOf != 0, ECS_INVALID_PARAMETER,
               "failed to create entity Rel_AbilityOf");
    ecs_id(Rel_AbilityOf) = Rel_AbilityOf;
    ecs_add_pair(world, Rel_AbilityOf, EcsOnDeleteTarget, EcsDelete);
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
  ECS_TAG_DEFINE(world, ZSelection);
  ECS_TAG_DEFINE(world, RewardGeneratedIKZCredit);

  ECS_TAG_DEFINE(world, TMulligan);
  ECS_TAG_DEFINE(world, TStartOfTurn);
  ECS_TAG_DEFINE(world, TMain);
  ECS_TAG_DEFINE(world, TResponseWindow);
  ECS_TAG_DEFINE(world, TCombatResolve);
  ECS_TAG_DEFINE(world, TEndTurnAction);
  ECS_TAG_DEFINE(world, TEndTurn);
  ECS_TAG_DEFINE(world, TEndMatch);
  ECS_TAG_DEFINE(world, TAbilityResolution);

  azk_register_ability_components(world);
  azk_register_card_def_resources(world);
  azk_init_ability_registry(world);
  azk_register_game_log_components(world);

  ecs_observer(world,
               {
                   .entity = ecs_entity(world, {.name = "OnOwnedCardIdSet"}),
                   .query.terms =
                       {
                           {
                               .id = ecs_id(CardId),
                           },
                           {
                               .id = ecs_pair(Rel_OwnedBy, EcsWildcard),
                           },
                       },
                   .events = {EcsOnSet},
                   .callback = on_owned_card_id_set,
               });
}
