# STT03/STT04 Card Notes

Working notes for the remaining STT03 and STT04 implementation batch.

## STT03

- `STT03-001` Bobu
  - `[Once/Turn][Main] Pay 1 IKZ: Until the start of your next turn, the first time an Earth entity in your Garden or Alley is destroyed or sacrificed, you may heal 1 to your leader.`
- `STT03-002` Stonehaven Gate
  - `Portal an untapped entity from the Alley into the Garden, then you may give an entity in your Garden with base health equal to or less than the Gate Power of the portaled entity Defender until the start of your next turn.`
- `STT03-003` Koyama Farm Potter
  - `[On Play] Look at the top 5 cards of your deck, you may reveal up to 1 Verdant subtype card and add it to your hand, then bottom deck the rest in any order.`
- `STT03-004` Sloth Scarecrow
  - `[Main] You may sacrifice this card: Heal 1 to your leader.`
- `STT03-005` Wobbly Cabbage Cart
  - `[When Destroyed] You may destroy an entity with 1 health in your opponent's Garden.`
- `STT03-006` Cactus Farmer
  - `[When Destroyed] Draw 1, then discard 1.`
- `STT03-007` Koyama Farm Caretaker
  - `[In Garden Only Ability] Treat this card also as an IKZ card.`
- `STT03-008` Midnight Courier
  - Vanilla.
- `STT03-009` Warding Totem
  - `[On Play] Add 1 IKZ from your IKZ pile to your IKZ area; the IKZ enters the field tapped.`
- `STT03-010` Shroommancer
  - `[Once/Turn] Whenever this card attacks and destroys an entity in your opponent's Garden, you may heal 1 to your leader.`
- `STT03-011` Koyama Farm Plowman
  - `[On Play] If this card is played in the Garden, you may destroy an entity with a base health of 2 or less in your opponent's Garden.`
- `STT03-012` Miharu of the White Bloom
  - `[On Play] You may play an entity with a cost of 2 or less from your hand into your Garden.`
  - `[In Garden Only Ability][Once/Turn] On your opponent's turn, whenever another entity in your Garden or this entity is destroyed, you may heal 2 to your leader.`
- `STT03-013` Stone Masked Ancient
  - `[In Garden Only Ability] Taunt.`
  - `When this card enters the Garden, you may tap it.`
- `STT03-014` Sandcoil Python
  - `[On Play] All entities in your opponent's Garden with a base attack of 3 or less become Rooted.`
- `STT03-015` Jar of Beans
  - `[Main] Heal 3 to your leader. If you have 7 or more IKZ in your IKZ area, heal 5 instead.`
- `STT03-016` Quicksand
  - `[Main] Destroy all entities with 2 health or less in your opponent's Garden.`

## STT04

- `STT04-001` Zero
  - `[Once/Turn][Main] Deal 1 damage to this card: Deal 1 damage to an entity in your Garden or Alley, then give that entity +1 attack until the end of the turn.`
- `STT04-002` Ragefire Gate
  - `Portal an untapped entity from the Alley into the Garden, then you may give an entity in your Garden that took damage this turn +Attack equal to the Gate Power of the portaled entity until the end of the turn.`
- `STT04-003` Cinderwake Seer
  - `[Start of Each Turn] You must deal 1 damage to this card.`
- `STT04-004` Fanatic Kindler
  - `[On Play] You may sacrifice this card: Deal 1 damage to an entity in any Garden.`
- `STT04-005` Ruby
  - `[On Play] Look at the top 5 cards of your deck, reveal up to 1 Pyreskin subtype card and add it to your hand, then bottom deck the rest in any order.`
- `STT04-006` Wolf Cub
  - Vanilla.
- `STT04-007` Enraged Howler
  - `[Once/Turn] Whenever this entity takes damage, it gets +1 attack until the end of turn.`
- `STT04-008` Lady Emberheart
  - `[Once/Turn] On your turn, after this card attacks an entity, you may untap this card.`
- `STT04-009` Cinderwake Ritualist
  - `[In Garden Only Ability][Once/Turn] Whenever this card takes damage from card effects, you may also deal that damage to another leader or entity in any Garden, capped at 2 damage.`
- `STT04-010` Reckless Tinkerer
  - `[On Play] You must deal 1 damage to this card.`
- `STT04-011` Scorchland Raven
  - Vanilla.
- `STT04-012` Spiteful Raider
  - `[Once/Turn] Whenever this card takes damage, deal 1 damage to a leader or an entity in any Garden.`
- `STT04-013` Kurai the Volcano
  - `[Once/Turn] Whenever an entity in your opponent's Garden is destroyed, untap this card.`
- `STT04-014` Scorchveil Shinobi, Suzuka
  - `[On Play] If your leader has the Scorchweaver subtype, deal 1 damage to all entities in each player's Garden, then give all other entities in your Garden +1 attack until the end of the turn.`
- `STT04-015` Detonation Pact
  - `[Main] Deal 1 damage to your leader: Deal 2 damage to your opponent's leader.`
- `STT04-016` Collateral Burst
  - `[Main] Deal 1 damage to an entity in your Garden: Deal up to 2 damage to a leader or an entity in your opponent's Garden.`
- `STT04-017` Wrath of Sinder
  - `[Main] Sacrifice any number of entities in your Garden: Deal damage to an entity or a leader in any Garden equal to the number of entities sacrificed.`

## Implementation Notes

- `STT03-007` is already supported in `azk_card_counts_as_ikz_source(...)`.
- `Until the start of your next turn` can use `TAG_GRANT_TICK_START_OF_TURN` with `remaining_ticks = 2`, because status ticking currently runs for both players at each turn start.
- `STT03-001`, `STT03-012`, and `STT04-013` need destroy-event hooks that are broader than the existing self-only destroy/sacrifice trigger path.
- `STT04-001` needs friendly `Garden or Alley` entity targeting, which is not fully wired in the current ability target collector.
