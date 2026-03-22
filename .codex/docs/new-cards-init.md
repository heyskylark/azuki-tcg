# New Cards Init

Status: Working notes

Last updated: 2026-03-20

## Scope

This doc covers how to add new cards to the repo before and during implementation.

## Required Rules References

Before implementing new cards, keyword mechanics, or effect text semantics, check these docs first:

- [game_rules.md](./game_rules.md) for the primary rules reference derived from `game_rules.pdf`
- [azuki_tcg_guide.md](./azuki_tcg_guide.md) for the quick-start / field-layout reference derived from `azuki_tcg_guide.pdf`

Use these docs to confirm:

- targeting rules by zone
- attack / response timing windows
- cooldown, replacement, and combat timing behavior
- card wording semantics like `cost : effect`, optional activation, and `MUST`
- player-facing keyword/mechanic rules such as `Infiltrate`, `Carapace`, `Godmode`, `Frozen`, and `Shocked`

For cards using those mechanics, verify the implementation matches these rules:

- `Infiltrate`: attacking with this card disables the defending player's `Defender` response for that attack
- `Carapace N`: reduce damage from all sources by `N`; Carapace stacks
- `Godmode`: the card cannot leave the field from damage or card effects, but it can still be targeted and still be replaced when a row is full
- `Frozen`: the card's abilities are disabled, and it cannot attack or be damaged
- `Shocked`: the card does not untap during its next untap step

If repo behavior appears to conflict with those references, call out the discrepancy before implementing the new card.

Assumptions for this workflow:

- new cards are not being added to the built-in starter decks
- card metadata is stored in both the C engine source data and the database
- custom card behavior is implemented in the C engine
- we want to batch card seed migrations when possible to avoid accumulating lots of tiny migrations

## Core Rules

- Do not hand-edit generated card definitions in `src/generated/` or `include/generated/`.
- Always regenerate generated card defs from `scripts/azuki-card-defs.jsonl` via `scripts/generate_card_defs.py`.
- Treat `CardDefId` values as unstable until the JSONL batch order is finalized.
- Prefer appending new cards in a stable set/order block so existing numeric IDs do not shift unexpectedly.
- Do not touch starter deck definitions in `src/world.c` or `packages/backend-core/src/services/DeckService/constants/index.ts` for normal new-card work.
- Do not edit old committed migrations to add more cards later.
- For card DB inserts, prefer one new custom seed migration per batch of new cards instead of one migration per card.

## Best Order

This is the recommended order because it reduces churn and keeps the C engine, TS mapping, and DB in sync.

### 1. Finalize the card batch

Before touching code:

- decide which cards belong in this batch
- decide their final `card_id` values
- decide their final order in `scripts/azuki-card-defs.jsonl`
- collect metadata for each card:
  - name
  - rarity
  - element
  - type
  - attack / health / gate points / IKZ cost
  - keywords
  - subtypes
  - effect text
  - image key

Why this goes first:

- `CardDefId` values are assigned by generated enum order
- changing the JSONL order later will shift numeric IDs and force follow-up fixes in TS mappings, tests, and debug tools

### Card Image Quick Read

When a new card is coming from an unfinished image, read the frame carefully:

- top-left number = IKZ cost
- left-side badge number = gate power
- black diamond at bottom right = attack
- white diamond at bottom right = defence / health
- top-right badge = card type
- text box = effect text
- subtype line sits above the artist / set footer

Useful filename hints:

- image keys often encode set, card code, display name, type shorthand, and rarity
- example: `_E_UC_` usually means `ENTITY` + `UC`

Use image-derived metadata only as a hint when the user has not provided the value explicitly. If the image and user input conflict, prefer the explicit user input and call out the mismatch.

### 2. Add raw engine card data

Update:

- `scripts/azuki-card-defs.jsonl`

This is the source of truth for engine-level card metadata such as:

- card code
- type
- element
- base stats
- IKZ cost
- keywords
- subtypes

If the new cards need a new keyword shape or new enum value, update the generator first:

- `scripts/generate_card_defs.py`

Relevant generator areas:

- keyword mapping and keyword-with-data support
- allowed fields per card type
- type / element / rarity enum orders

### 3. Regenerate card defs immediately

Run the generator after the JSONL update, before doing follow-up code changes.

Command:

```bash
python3 scripts/generate_card_defs.py scripts/azuki-card-defs.jsonl
```

This updates:

- `include/generated/card_defs.h`
- `src/generated/card_defs.c`

After regeneration, use the generated enum as the canonical source for new numeric `CardDefId` values.

### 4. Update TS card ID mapping from generated output

Update:

- `packages/backend-core/src/services/cardMapperService.ts`

This file currently duplicates:

- the `CardDefId` enum
- the `cardCode -> defId` map
- the reverse `defId -> cardCode` map

Anything that initializes the engine from DB decks depends on this mapping being correct.

This also affects:

- deck loading into the websocket engine
- deck API responses that include `cardDefId`
- debug draw

### 5. Stage DB card metadata in one named batched custom migration

Create one new named custom SQL migration for the whole batch of new cards.

Do this in batches, not per card, when practical.

Recommended command:

```bash
bunx drizzle-kit generate --custom --name=<card-batch-name>
```

Recommended rule:

- when you start a batch of new cards, create one named custom migration for that batch
- keep adding that batch's card inserts to the same new custom migration file while the batch is still in progress
- do not run `db:migrate` for that new migration until the card batch is done
- once the batch is finalized and the migration is committed, do not go back and keep editing it for future unrelated batches

For card batches, the migration should insert rows into `cards` with:

- `card_code`
- `name`
- `rarity`
- `special_rarity`
- `element`
- `card_type`
- `attack`
- `health`
- `gate_points`
- `ikz_cost`
- `keywords`
- `subtypes`
- `effect_text`
- `flavor_text`
- `image_url`

Relevant files:

- `packages/backend-core/src/drizzle/schemas/cards.ts`
- `packages/backend-core/drizzle/0001_seed-cards.sql`

Notes:

- this is a data migration, so a custom SQL migration is appropriate
- batching these inserts reduces migration count and helps keep full DB rebuilds faster
- the intent is to avoid repeated partial card migrations while a batch is still being designed

### 6. Implement custom C engine behavior only for cards that need it

If a card is vanilla and only needs base stats, keywords, and metadata, stop here on the engine side.

If the card has custom behavior:

- add a header in `include/abilities/cards/`
- add a source file in `src/abilities/cards/`
- register the ability in `src/abilities/ability_registry.c`

Typical things to wire:

- `validate`
- `validate_cost_target`
- `validate_effect_target`
- `apply_costs`
- `apply_effects`
- selection callbacks
- passive observer init / cleanup

Relevant files:

- `include/abilities/ability_registry.h`
- `src/abilities/ability_registry.c`
- `src/components/abilities.c`
- `src/world.c`

Notes:

- `src/world.c` already attaches ability metadata to instantiated cards
- CMake does not need manual source registration because `src/*.c` is globbed

### 7. Extend engine infrastructure only if the mechanic does not fit current systems

Only do this if the new card introduces a new mechanic beyond the current engine surface.

Examples:

- new target type
- new timing model
- new keyword tag
- new status effect
- new observable state that must reach the client

Possible update points:

- `include/components/abilities.h`
- `src/components/abilities.c`
- `include/abilities/targeting/ability_targeting.h`
- `src/abilities/targeting/ability_targeting.c`
- `include/utils/observation_util.h`
- `src/utils/observation_util.c`
- `apps/websocket/native/src/addon.c`
- `apps/websocket/src/engine/types/index.ts`
- `apps/web/src/lib/game/abilityTargeting.ts`
- `apps/web/src/types/game.ts`
- `apps/web/src/components/game/cards/Card3D.tsx`

If the mechanic fits existing patterns, do not expand these surfaces unnecessarily.

### 8. Verify the web path

Normal cards do not need a separate frontend registry entry.

The web app gets card metadata from:

- deck API responses
- snapshot card metadata
- image keys from the DB

Relevant files:

- `packages/backend-core/src/services/DeckService/index.ts`
- `packages/backend-core/src/services/cardMetadataService.ts`
- `apps/web/src/app/api/decks/[deckId]/route.ts`
- `apps/web/src/app/(main)/rooms/[id]/components/InMatchView.tsx`
- `apps/web/src/contexts/AssetContext.tsx`
- `apps/web/src/types/game.ts`

Frontend code changes are only needed if the new cards introduce:

- new visible statuses
- new target semantics
- new interaction patterns
- new render cues

## Decision Tree

### Vanilla card

Required:

1. add to `scripts/azuki-card-defs.jsonl`
2. regenerate card defs
3. update `packages/backend-core/src/services/cardMapperService.ts`
4. add the card to the next batched custom card seed migration
5. rebuild and verify

### Card with custom existing-style effect

Required:

1. add to JSONL
2. regenerate
3. update TS card mapping
4. add DB row in the current batched migration
5. add C ability header/source
6. register in `src/abilities/ability_registry.c`
7. add tests
8. rebuild and verify

### Card with new mechanic

Required:

1. do all steps above
2. extend engine infrastructure only where the mechanic truly requires it
3. thread new state through observation, native serialization, websocket TS types, and client rendering

## Testing Checklist

At minimum, verify:

- generated `CardDefId` values match the TS mapping
- the DB row exists and deck APIs return the correct metadata
- the websocket service can load decks with the new card
- the native binding still builds
- the engine can instantiate the new card correctly

Useful places to update or add checks:

- `tests/test_world.c`
- `apps/websocket/scripts/test-engine.ts`

For effect cards, add focused ability tests rather than relying only on world creation.

## Rebuild Checklist

If C engine files changed:

1. regenerate card defs if needed
2. rebuild the native module
3. rebuild the websocket service

Commands:

```bash
bun ws build:native
bun ws build
```

If DB migrations changed:

```bash
bun core db:migrate
```

If schema files changed, generate migrations first:

```bash
bun core db:generate
```

## Current Non-Goals

This workflow intentionally does not include:

- starter deck updates
- automatic generation of TS `CardDefId` mappings from the C header
- consolidation of existing historical card seed migrations

Those can be tackled later if the duplication becomes the main source of friction.
