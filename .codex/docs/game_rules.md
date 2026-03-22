# Azuki TCG Rules Reference

Status: OCR-derived Markdown from `.codex/docs/game_rules.pdf`

Last updated: 2026-03-20

Use this as the primary in-repo rules reference when implementing new cards and mechanics. This file is a cleaned transcription and summary of the PDF, organized for engineering use.

If engine behavior conflicts with this document, flag the discrepancy before implementing new cards that depend on that rule.

## Win Conditions

- Primary win condition: if your opponent's Leader's Health reaches `0`, you win.
- Alternate win condition: if your opponent must draw from their deck and cannot because their deck is empty, you win.

## Field Layout

### Garden

- Leaders start face-up in the Garden and cannot be removed.
- Only entities and Leaders in your Garden can attack entities and Leaders in the opponent's Garden.
- Leaders can always be targeted by attacks, whether tapped or untapped.
- Garden entities can only be targeted by attacks while tapped.
- Garden entities can always be targeted by spells and abilities, whether tapped or untapped.
- Garden max size: `5` entities, not counting the Leader.
- If the Garden is full and a new entity enters, an existing entity must be replaced and put into the discard pile.
- Replacing does not count as being destroyed, removed, or sacrificed.
- A card with `Godmode` can still be chosen as the replacement target.

### Alley

- Alley entities cannot be targeted by attacks or abilities unless a card effect explicitly says otherwise.
- Alley max size: `5` entities.
- If the Alley is full and a new entity is played there, an existing entity must be replaced and put into the discard pile.
- Replacing does not count as being destroyed, removed, or sacrificed.
- If a Garden/Alley swap happens while both rows are full, no replacement is needed because the swap resolves simultaneously.
- If a player forces their opponent to replace an entity on the field, the opponent chooses which entity to replace.

### Gate

- The Gate is a permanent on the board.
- Gates have no Health and cannot be attacked or destroyed.
- Your Gate must match your Leader's element.
- Tapping the Gate portals an entity from the Alley into the Garden.
- Gate abilities scale based on the portaled entity's Gate Power.

### Other Zones

- `IKZ Pile`: IKZ you cannot use yet.
- `IKZ Area`: IKZ you can tap and spend.
- `Main Deck`
- `Discard Pile`

## Card Types

### Leader

- Each player has one permanent Leader in the Garden.
- Leader Health starts at the value printed on the card.
- Leader element determines legal deck construction.
- Neutral cards can be used with any Leader.

### IKZ

- IKZ is a colorless resource used to pay costs.

### Entities

- Entities can be played in the Garden or Alley during the Main Phase.

### Spells

- `[Main]` spells can only be played during the Main Phase.
- `[Response]` spells can be played when the opponent declares an attack.
- Multiple `[Response]` spells can be played during the same attack.
- The active player cannot play `[Response]` cards.

### Weapons

- Weapons can be attached to Leaders or Garden entities.
- Weapons cannot be attached to Alley cards.
- Weapons go to the discard pile at end of turn.
- Multiple weapons can be attached to the same card.
- If an entity with attached weapons leaves the field, its weapons go to the discard pile.
- A weapon's `[On Play]` ability triggers when it enters the field.
- Moving a weapon from one entity to another does not retrigger its `[On Play]` ability.

## Phases

### Game Start

- Draw `7` cards.
- Mulligan rule: once per game, you may put your whole hand on the bottom of the deck, draw `7` new cards, then shuffle.

### Start of Turn

- Draw `1` card each turn.
- The player going first does not draw on their first turn.
- Gain `1` IKZ each turn, up to `10`.
- Untap all tapped cards.
- If you have multiple start-of-turn effects, you choose their order.
- If both players have start-of-turn effects, the active player chooses the order.

### Main Phase

- Playing cards, attacking, and using abilities happen during the Main Phase.
- If an ability does not specify `[Response]`, treat it as a `[Main]` ability.
- Within the Main Phase, actions can be taken in any order if legal.
- Example from the rules: you may attack, then play a card, then attack again.

### Response Window

- `[Response]` cards can be played when the opponent declares an attack.
- `[Response]` abilities from entities already on the field can also be activated then.
- The attacking player does not get a response window during their own attack.

### End of Turn

- If you have multiple end-of-turn effects, you choose their order.
- If both players have end-of-turn effects, the active player chooses the order.
- At end of turn, entity stats reset to original values unless another active effect still modifies them.

## Playing Entities, Using Gates, and Cooldown

- You play entities by paying their IKZ cost.
- You can play entities into either the Garden or the Alley.
- You can use the Gate once per turn because it must tap to activate.
- If multiple entities are in the Alley, only one can be portaled per turn.
- Entities have `Cooldown` on the turn they enter the Garden.
- A card on Cooldown cannot tap to attack or use tap abilities that turn.
- Cooldown applies whether the entity entered the Garden by being played there directly or by being portaled from the Alley.
- Cooldown still applies even if the entity spent earlier turns in the Alley.
- `Defender` is explicitly not affected by Cooldown.
- You may portal an entity into the Garden on the same turn it was played to the Alley, but you are not required to.

## Combat

- Combat damage is simultaneous.
- Attacker and defender deal damage to each other at the same time.
- If a Leader has no Attack, an attacking entity takes no damage from that Leader.
- If a weapon-equipped Leader attacks into an entity, the Leader takes permanent damage.
- Entity Health resets at end of turn; Leader damage is permanent.
- An end-of-turn reset is not healing.
- When a Leader reaches `0` Health, that player loses.

### Combat Order

1. Declare attacker and target.
2. Resolve `[When Attacking]` effects.
3. Enter the `[Response]` window.
4. Declare defender.
5. Resolve damage.
6. Resolve `[After Attacking]` effects.

## Turn-One and Resource Rules

- Players start the game with `7` cards and `0` IKZ.
- First player: gains `1` IKZ and does not draw on their first turn.
- Second player: gains `1` IKZ and an expendable IKZ token.
- IKZ tokens are single-use temporary resources.
- The second player's IKZ token does not expire if unused, but once spent it is gone for the rest of the game.
- The second player cannot access that token until their first turn.
- You may attack on your first turn if you play an entity with `Charge` or your Leader has a weapon equipped.
- After the first round, each player gains `1` IKZ and draws `1` card each turn.

## Hand Size

- There is no maximum hand size.

## Card Resolution Order

- Cards fully resolve before the next card's effects start resolving.
- The PDF example says that if one card causes another effect source to enter play, the first card fully resolves before the second source resolves its own effect.

## Card Effects and Persistence

- All text on cards counts as a card effect.
- A card that cannot be affected by a certain kind of effect can still be targeted by that effect; the effect simply fizzles.
- Example from the PDF: a `Godmode` card can be targeted by sacrifice or destroy effects, but the effect does not affect that card.
- Example from the PDF: a card that cannot take damage from card effects can still be targeted by damage effects, but takes no damage from that effect.
- When a card leaves the field, all its stats and effects reset.
- If a card leaves the field and later re-enters, its `[Once/Turn]` effects can be used again.

## Stats

### Leaders

- Leader Health is loss-critical; reaching `0` loses the game.
- There is no overheal beyond printed max Health.

### Entities

- `IKZ Cost`: top-left number; cost to play the card.
- `Gate Power`: beneath IKZ cost; used by the Gate ability when that entity is portaled.
- `Attack`: top number in the bottom-right.
- `Health`: bottom number in the bottom-right.
- If an entity's Health reaches `0`, it is destroyed and sent to the discard pile.
- There is no overheal beyond printed max Health.

## Card Effects: Timing and Zone Restrictions

- Zone restrictions matter.
- Example from the PDF: "In Garden Only Ability" means the ability can only be activated while the card is in the Garden.
- `Once per Turn` effects may only be activated once during a turn.
- If the card leaves the field and re-enters, that once-per-turn limit refreshes.
- Timing windows define when effects can be activated.
- Example from the PDF: `[On Play]` means only when the card is played onto the field.

## Keyword and Condition Glossary

These entries include player-facing rule clarifications added after the OCR pass so card implementation can rely on explicit semantics instead of partial PDF wording.

### Keyword Abilities

- `Defender`: when an opponent declares an attack targeting your Leader or another entity, you may tap an untapped entity with `Defender` in your Garden to redirect the attack to that card.
- `Infiltrate`: when a card with this ability attacks, the defending player cannot activate `Defender` during that attack.
- `Carapace N`: damage taken from all sources is reduced by `N`.
- `Carapace` stacks.
- `Godmode`: the card cannot leave the field from taking damage or from card effects.
- `Godmode` does not stop the card from being targeted.
- `Godmode` does not stop replacement when a row is full.

### Negative Conditions

- `Frozen`: the card's abilities are disabled, and it cannot attack or be damaged.
- `Shocked`: the card does not untap during its next untap step.

## Card Wording Rules

- Text before `:` is the activation cost.
- Text after `:` is the resulting effect.
- Paying the cost is optional unless the card says `MUST`.
- If you do not pay the cost, you do not get the effect.
- You may choose to play a card with an `[On Play]` effect and decline to activate that effect.
- You may also attack without activating a `[When Attacking]` effect.
- If a card says `MUST` in the cost, then paying that cost and resolving the effect is mandatory.
- If you do pay an activation cost, you must resolve the resulting effect.

## Deck Construction

- Main deck size is exactly `50` cards.
- Up to `4` copies of a card with the same `card_id`.
- Different `card_id` values are different cards even if they share a name.
- Two cards with the same name but different `card_id` and text are separate cards.

## Elements

- The Leader sets the deck's element.
- All cards in the deck, including the Gate, must match the Leader's element.
- Neutral cards can be used with any Leader.
- Listed elements:
  - Fire
  - Water
  - Earth
  - Lightning
  - Neutral

## Implementation-Specific Notes

- `Carapace` stacks.
- Example from the PDF: `Carapace 1` becomes `Carapace 2` if `Oathstone` is played on it.
- `Godmode` and `Carapace` were partially described in the PDF and are clarified above.
- `Frozen` and `Shocked` were clarified here because the OCR-derived PDF text did not fully define them.
