#include "abilities/cards/stt02_002.h"

#include "components/components.h"
#include "generated/card_defs.h"
#include "utils/cli_rendering_util.h"
#include "utils/player_util.h"
#include "utils/zone_util.h"

// STT02-002 "Hydromancy": On Gate Portal; untap IKZ up to portaled card's gate
// points

bool stt02_002_validate(ecs_world_t *world, ecs_entity_t card,
                        ecs_entity_t owner) {
  (void)world;
  (void)card;
  (void)owner;

  // Always valid - effect just happens after portal
  return true;
}

void stt02_002_apply_effects(ecs_world_t *world, const AbilityContext *ctx) {
  const GameState *gs = ecs_singleton_get(world, GameState);
  uint8_t player_num = get_player_number(world, ctx->owner);
  ecs_entity_t ikz_area = gs->zones[player_num].ikz_area;

  // Get gate points from the portaled card (stored in ctx->effect_targets[0])
  ecs_entity_t portaled_card = ctx->effect_targets[0];
  const GatePoints *gp = ecs_get(world, portaled_card, GatePoints);

  if (gp && gp->gate_points > 0) {
    uint8_t untapped = untap_n_ikz_cards(world, ikz_area, gp->gate_points);
    cli_render_logf("[GatePortal] Hydromancy untapped %u IKZ", untapped);
  }
}
