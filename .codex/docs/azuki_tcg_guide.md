# Azuki TCG Quick Start Guide Reference

Status: OCR-derived Markdown from `.codex/docs/azuki_tcg_guide.pdf`

Last updated: 2026-03-20

This PDF is a one-page visual quick-start sheet. The source layout is dense, so this Markdown is a cleaned summary rather than a line-by-line transcription. Use [game_rules.md](./game_rules.md) as the authoritative rules reference when wording or timing details matter.

## What This Guide Is Good For

- field layout
- parts of a card
- high-level turn flow
- common action/timing vocabulary
- quick onboarding context when reading new card designs

## Field Setup Summary

- Put your `Leader` in the Garden.
- Put your matching `Gate` in play.
- Put your `50-card` deck face-down in the deck area.
- Keep a `Discard Pile` for cards that leave play.
- Track Leader Health with a `d20`.
- Separate IKZ into:
  - `IKZ Pile`: future IKZ not yet available
  - `IKZ Area`: IKZ currently available to tap and spend

## Field Zones

### Garden

- Front row.
- Entities attack from the Garden.
- Garden has a max size of `5` entities.
- An entity that enters the Garden has `Cooldown` that turn and cannot attack or use tap abilities.
- Your Leader always occupies the Garden.

### Alley

- Back row.
- Alley has a max size of `5` entities.
- Cards are often staged here before being portaled by the Gate.

### Deck / Discard / IKZ

- `Deck Area`: your main deck.
- `Discard Pile`: face-up pile for cards that leave play.
- `IKZ Pile`: reserved IKZ not yet active.
- `IKZ Area`: active IKZ you may tap to pay card costs.

## Card Types Shown in the Guide

- `Leader`
- `Gate`
- `Entity`
- `Weapon`
- `Spell`
- `IKZ`

## Parts of a Card

The guide visually labels common card fields. For implementation work, the main takeaways are:

- IKZ cost
- attack
- health
- gate power on entities
- card type
- effect text
- card name and element

## Core Player Actions

- play cards by paying IKZ and placing them into the proper zone
- attack with Garden cards that are allowed to attack
- use the Gate to portal Alley entities into the Garden
- activate card abilities when their timing window allows it

## Ability / Wording Reminders

- The guide repeats the normal `cost : effect` card-text pattern.
- Costs must be paid to get the effect.
- Timing words like `[On Play]`, start of turn, end of turn, and response matter.
- The quick-start glossary is consistent with the main rules PDF, but the main rules doc should win if wording is ambiguous.

## Turn Flow Summary

This guide presents a quick turn-order overview rather than full formal rules.

### Start of Turn

- untap cards
- draw / gain IKZ according to turn rules
- resolve start-of-turn effects

### Main Phase

- attack
- play cards
- activate abilities
- use the Gate

### End of Turn

- resolve end-of-turn effects
- reset temporary entity damage and temporary status/state as applicable

For exact timing and attack/response sequencing, use [game_rules.md](./game_rules.md).

## Why This Matters for New Card Work

Use this guide when you need fast orientation on:

- whether a card belongs in the Garden vs Alley
- what visible stats/parts a card exposes
- how the Gate/portal pattern is explained to players
- what basic timing vocabulary the player-facing materials use

Use [game_rules.md](./game_rules.md) when implementing exact behavior.
