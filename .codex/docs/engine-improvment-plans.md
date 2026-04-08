# Engine Improvement Plans

This file tracks engine code paths that are still card-specific and need a more
extendible abstraction later.

## Kira Attack Swap Abstraction

- Current generic hook:
  `src/systems/main_phase.c`
  `queue_kira_attack_redirect_if_present(...)`
- Current card-specific resolution:
  `src/abilities/cards/azk01_034.c`
- Current hard-coded behavior:
  When a Garden defender is attacked, the engine scans the defending player's
  Alley for `AZK01-034`, then queues its `AWhenAttacked` effect.
- Why a simple tag is not enough yet:
  A future `ATTACK_SWAPPABLE`-style tag would still need:
  source-zone predicates
  defender replacement semantics
  combat-state mutation rules
  optional targeting / multiple eligible swaps
- Likely future abstraction:
  An attack-redirection or defender-rewrite hook with:
  trigger zone filters
  replacement priority rules
  a standard way to mutate `GameState.combat_state.defender_card`

## Pekiro Damage Redirect Abstraction

- Current code path:
  `src/utils/damage_util.c`
  `maybe_queue_pekiro_redirect(...)`
  `deal_effect_damage_from_source_internal(...)`
- Current hard-coded behavior:
  Before ability/spell damage is applied, `AZK01-062` gets a special redirect
  queue and prompt flow.
- Why it stays as-is for now:
  This is effectively a replacement / redirect effect, not just a triggered
  observer.
- Likely future abstraction:
  A pre-damage interception layer that carries:
  damage source
  damage amount
  damage kind
  replacement / redirect arbitration

## Bobu Destroy Or Sacrifice Event Abstraction

- Current state setup:
  `src/abilities/cards/stt03_001.c`
- Current generic hook:
  `src/utils/card_utils.c`
  `maybe_trigger_bobu_state(...)`
  `maybe_trigger_special_destroy_observers(...)`
  `discard_card_internal(...)`
- Current hard-coded behavior:
  `STT03-001` installs temporary leader state, then the discard path consumes it
  when the first qualifying Earth entity in Garden or Alley is destroyed or
  sacrificed.
- Why it stays as-is for now:
  The current engine only exposes the discard reason in the discard path.
  Passive observers do not receive that reason directly.
- Likely future abstraction:
  A leave-play event payload with:
  card
  owner
  source zone
  reason (`destroy`, `sacrifice`, `replacement`, `bounce`, etc.)

## Miharu Destroy Observer

- Current generic hook:
  `src/utils/card_utils.c`
  `maybe_trigger_miharu_state(...)`
  `maybe_trigger_special_destroy_observers(...)`
  `discard_card_internal(...)`
- Existing validation rules embedded there:
  only on destroy
  only while the destroyed card left the owner's Garden
  only on the opponent's turn
  once per turn via `STT03MiharuState.last_heal_turn`
  must still work if Miharu itself is the destroyed card
- Why the current passive observer system is not enough:
  Passive observers currently watch ECS relationship changes like `EcsChildOf`
  add/remove. They do not know whether a Garden leave was caused by destroy,
  sacrifice, return-to-hand, or replacement.
- Additional lifecycle issue to think through:
  `AbilityDef.cleanup_passive_observers` exists in the registry shape, but there
  is no central discard / return-to-hand path that invokes those cleanup hooks
  today. Any garden-scoped observer design needs an explicit lifecycle model.
- Likely future abstraction:
  A discard / leave-play event bus with reason data that fires before the card's
  observable state is fully cleared.

## Kurai Destroy Observer

- Current generic hook:
  `src/utils/card_utils.c`
  `maybe_trigger_kurai_state(...)`
  `maybe_trigger_special_destroy_observers(...)`
  `discard_card_internal(...)`
- Existing validation rules embedded there:
  only on destroy
  only when an opposing Garden entity is destroyed
  once per turn via `STT04KuraiState.last_untap_turn`
  untap preserves cooldown state
- Why the current passive observer system is not enough:
  Same limitation as Miharu: zone-remove observers do not know the discard
  reason, so they cannot distinguish destroy from other ways a card leaves
  Garden.
- Likely future abstraction:
  A typed destroy event or leave-play event payload that cards can subscribe to
  with owner / opponent / zone filters.
