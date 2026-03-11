# TCG Client State Refactor

Status: Proposed

Last updated: 2026-03-11

## Goal

Make the C engine the only source of truth for client-visible game state while preserving:

- smooth step-based animations
- deterministic replay support
- compatibility with the current AI / training action interface

The immediate bug that motivated this work was ordered-zone drift between engine truth and client truth when the client reconstructed hand state from logs whose `toIndex` values were computed before Flecs committed deferred structural changes.

## Current Problem

The current stack has three separate concerns coupled together too tightly:

1. The engine mutates ECS state while Flecs may still be in deferred mode.
2. The engine writes step logs inline during mutation.
3. The client uses those logs as authoritative state updates.

This is fragile for ordered zones like `HAND` and `SELECTION`.

Relevant code paths:

- Engine logs are authored inline in `src/utils/game_log_util.c`.
- The native bridge clears logs, ticks the engine, and returns logs in `apps/websocket/native/src/addon.c`.
- The websocket service sends `GAME_LOG_BATCH` after action resolution in `apps/websocket/src/engine/actionResolutionService.ts`.
- The client applies those logs incrementally in `apps/web/src/lib/game/logProcessor.ts`.
- The client builds outgoing actions from the action mask by index in `apps/web/src/lib/game/actionValidation.ts`.
- The engine validates actions from raw tuple indices in `src/validation/action_validation.c`.

This creates two classes of risk:

- state-update correctness risk: the client reconstructs the wrong final zone order
- request correctness risk: the client may submit the wrong action if its displayed index order drifts

## Design Principles

1. The engine remains authoritative.
2. Step logs are for animation and replay playback, not the sole source of truth for private ordered zones.
3. AI and training can stay index-based internally.
4. The web protocol should eventually stop depending on raw hand indices for action submission.
5. Replay should have a canonical source of truth separate from derived animation logs.

## Refactor Order

This project should be tackled in the following order.

### Phase 1A: Two-Phase Step-Log Finalization

Objective:
- make step logs robust against Flecs deferred structural updates

Approach:
- capture move intent and `fromIndex` before deferred changes are committed
- finalize `toIndex` only after the relevant world state has been committed
- use committed world state to resolve final ordered-zone placement

Why first:
- this directly addresses the class of bug that already occurred
- it is lower risk than changing the client action protocol at the same time
- it improves animation logs even if later phases are delayed

Important note:
- if a single action contains multiple visible substeps, log finalization should happen at stable substep boundaries or sync points, not only at the very end of the entire action
- otherwise intermediate visible locations may be lost for logs like `DECK -> SELECTION -> HAND`

### Phase 1B: Authoritative Owner-Zone Patch After Action

Objective:
- ensure the client ends every action with the exact committed engine truth for private ordered zones

Approach:
- after the engine has fully resolved the action and Flecs has committed state, build an observation-derived patch for the owning player
- patch only the zones that are most vulnerable to drift and most sensitive for gameplay, starting with:
  - `HAND`
  - `SELECTION`
  - owner-visible counts tied to those zones where needed

The model is:

- step logs drive animations
- owner-zone patch corrects final truth

This is intentionally similar to a partial snapshot, but limited to the owner-sensitive zones that are most dangerous to reconstruct from logs alone.

### Phase 2: Add Stable Owner-Visible Card Instance Identity

Objective:
- give the client a stable way to refer to a physical card instance that is not "whatever is at index N"

Approach:
- add a new match-scoped card instance identifier for owner-visible cards
- do not reuse the current `cardId` / `cardCode` field for this, because the current field is effectively card-definition identity, not unique per physical copy

Initial scope:

- owner hand cards
- owner selection cards
- public board cards if useful for tooling and replay

Do not expose stable identity for hidden opponent cards in a way that leaks continuity across hidden zone moves.

### Phase 3: Add Server-Issued Action IDs to the Web Protocol

Objective:
- stop requiring the browser to manufacture the authoritative action tuple from local index state

Approach:
- keep the engine-native indexed action tuple and current action mask for AI / training
- decorate each legal action row sent to the browser with:
  - `actionId`
  - `stateVersion`
  - `sourceInstanceId` when relevant
- make the browser submit `actionId` instead of the raw 4-int tuple
- server resolves `actionId -> canonical indexed action tuple` just before validation/execution

This isolates the browser from index drift without changing the AI policy head.

### Phase 4: Migrate Browser Interaction Logic to Instance ID / Action ID

Objective:
- stop routing browser interactions through displayed hand indices

Approach:
- current browser action helpers in `apps/web/src/lib/game/actionValidation.ts` search the action mask by `handIndex` and return raw tuples
- these helpers should be migrated to:
  - map UI interaction to `instanceId`
  - look up matching legal actions by `sourceInstanceId`
  - submit the chosen `actionId`

This is the phase that actually removes request correctness dependence on client-side ordering.

### Phase 5: Add Persistent Action Journal and Replay Metadata

Objective:
- make deterministic replay rely on canonical action history rather than derived logs

Approach:
- store a persistent match action journal
- store enough replay metadata to recreate the initial engine state

This should be additive to existing log persistence, not a replacement.

### Phase 6: Re-evaluate Long-Term Need for Log-Derived State

Objective:
- reduce client dependence on `GAME_LOG_BATCH` as an authoritative reducer

Likely outcome:
- logs remain for animation and replay playback
- committed observation patches or snapshots remain authoritative for sensitive zones

## Detailed Technical Spec

## 1. Two-Phase Step-Log Finalization

### Problem Being Solved

Inline log creation during deferred Flecs mutation can observe stale ordered-zone state.

That makes fields like:

- `fromIndex`
- `toIndex`
- zone order assumptions

unsafe if they are treated as finalized before commit.

### Target Model

During resolution, the engine records an incomplete semantic step:

```c
typedef struct {
  uint32_t seq;
  uint8_t player;
  const char *instance_id;
  GameLogZone from_zone;
  int8_t from_index;
  GameLogZone to_zone;
  GameLogMoveCause cause;
} PendingZoneMoveLog;
```

At finalize time, after the relevant commit point:

- `toIndex` is resolved from committed world state
- post-move metadata is resolved from committed world state
- visibility redaction is applied after finalization, per viewer

### Key Rule

`fromIndex` must be captured before commit.

It cannot reliably be reconstructed from the final world if the card has already left the source zone.

### Expected Benefit

- animation logs remain accurate even when the world used deferred structural changes
- this directly reduces the risk of incorrect hand / selection insertion order on the client

## 2. Authoritative Owner-Zone Patch

### Definition

After action resolution completes, the server sends the owning player an authoritative committed patch for private ordered zones.

Example shape:

```ts
interface OwnerZonePatch {
  hand?: SnapshotHandCard[];
  selectionCards?: SnapshotSelectionCard[];
}
```

The final wire shape could attach this to `GAME_LOG_BATCH`:

```ts
interface GameLogBatchMessage {
  type: "GAME_LOG_BATCH";
  batchNumber: number;
  logs: unknown[];
  stateContext: SnapshotStateContext;
  actionMask?: SnapshotActionMask | null;
  ownerPatch?: OwnerZonePatch;
}
```

### Client Behavior

Client receives `GAME_LOG_BATCH`:

1. apply finalized logs for animation / transition intent
2. overwrite owner-sensitive zones from `ownerPatch`
3. replace `actionMask`

This means:

- logs can still drive the animation layer
- owner patch becomes final truth for ordered private zones

### Why This Matters

Even after Phase 1A, logs are still a derived artifact.

The owner patch provides a committed observation-based correction layer so the client never remains drifted if a step log is imperfect.

### Initial Scope

Start with:

- `HAND`
- `SELECTION`

Possible expansion later:

- owner-visible discard ordering if the UI ever depends on it
- other ordered private zones if they become interactive

## 3. Stable Card Instance Identity

### Problem Being Solved

The current owner-facing snapshot field named `cardId` is effectively card code / definition identity, not unique per physical card copy.

That is not sufficient to disambiguate:

- duplicate copies of the same card in hand
- replay references to a specific card instance
- browser-to-server action routing detached from indices

### Target Model

Add a new owner-visible match-scoped instance id:

```ts
type CardInstanceId = string;

interface SnapshotHandCard {
  instanceId: CardInstanceId;
  cardId: string | null;
  cardDefId: number;
  type: string;
  ikzCost: number;
}
```

Requirements:

- stable for the life of that physical card instance in the match
- deterministic and serializable
- not leaked for hidden opponent cards

### Recommended Internal Source

This can be derived from:

- a dedicated match-scoped network id component, or
- a deterministic per-match instance id allocated when cards are instantiated

Do not use raw ECS entity ids as the public protocol unless that choice is deliberate and documented.

## 4. Server-Issued Action IDs

### Problem Being Solved

The browser currently builds outgoing actions by searching an indexed action mask and returning the matching raw 4-int tuple.

That means displayed local order can affect submitted intent.

### Target Model

The engine still enumerates indexed legal actions.

The websocket layer decorates each legal action row with:

- `actionId`
- `stateVersion`
- `sourceInstanceId`

Example:

```ts
interface SnapshotActionMask {
  primaryActionMask: boolean[];
  legalActionCount: number;
  legalPrimary: number[];
  legalSub1: number[];
  legalSub2: number[];
  legalSub3: number[];

  stateVersion: number;
  legalActionIds: string[];
  legalSourceInstanceIds: Array<string | null>;
}
```

Client submission becomes:

```ts
interface GameActionMessage {
  type: "GAME_ACTION";
  actionId: string;
  stateVersion: number;
}
```

Server behavior:

- resolve `actionId -> legal action row -> indexed action tuple`
- verify `stateVersion` matches the currently active state
- submit the resolved tuple to the engine

### Compatibility

The AI model can continue using the current 4-int action tuple internally.

No policy-head change is required for this phase.

## 5. Replay Architecture

### Canonical Replay Source

Replay should be driven by:

- match setup metadata
- deterministic engine seed / starting state
- ordered action journal

Suggested replay metadata:

- `roomId` or `matchId`
- `rngSeed`
- `player0DeckId`
- `player1DeckId`
- starting player
- engine / rules version

Suggested action journal fields:

- `matchId`
- `actionSequence`
- `playerSlot`
- `stateVersionBefore`
- `stateVersionAfter`
- resolved indexed action tuple
- `sourceInstanceId` if applicable
- timestamp

### Role Of Finalized Logs

Finalized logs still matter for:

- animation playback
- spectator playback
- step-by-step debugging

But they should be treated as derived replay output, not the canonical replay input.

### Database Impact

The existing `game_logs` table is useful but not sufficient as the full replay backbone.

Current constraints:

- it stores derived logs only
- it is keyed by `room_id`
- it is deleted if the room row is deleted because of `ON DELETE CASCADE`
- it does not capture full replay metadata or action intent history

Therefore:

- existing `game_logs` can remain for step playback
- a new action-journal table should be added for canonical replay
- replay metadata should be archived with match results or a new replay metadata table

## Why This Order Is Recommended

### Why Phase 1A First

The current bug originated from an incorrect `toIndex` in a move log.

Improving step-log finalization first:

- directly fixes the observed failure mode
- validates the animation pipeline under committed state
- avoids stacking a protocol migration on top of an unresolved log problem

### Why Phase 1B Immediately After

Even perfect-looking logs are still derived.

An authoritative owner patch ensures:

- the client ends each action in committed engine truth
- any remaining blind spots in finalized logs do not leave private ordered zones drifted

### Why Action IDs Later

Action ids solve a different problem:

- not "did the client reconstruct final state correctly?"
- but "did the browser submit the exact intended move?"

That is important, but it is cleaner to tackle after the animation/state pipeline has been stabilized.

## Current / Planned Task Tracker

Legend:

- `[ ]` not started
- `[-]` in progress
- `[x]` done

### Phase 1A: Two-Phase Step-Log Finalization

- [ ] Define pending step-log data model for ordered-zone moves
- [ ] Capture `fromIndex` pre-commit for ordered-zone move logs
- [ ] Finalize `toIndex` post-commit from committed world state
- [ ] Apply the same approach to `HAND`
- [ ] Apply the same approach to `SELECTION`
- [ ] Review whether any other ordered zones need the same treatment
- [ ] Add focused regression tests for multi-card ordered-zone moves in one action
- [ ] Add focused regression tests for multi-step moves like `DECK -> SELECTION -> HAND`

### Phase 1B: Authoritative Owner-Zone Patch

- [ ] Define `ownerPatch` payload shape
- [ ] Populate `ownerPatch.hand` from committed observation
- [ ] Populate `ownerPatch.selectionCards` from committed observation
- [ ] Send owner patch with `GAME_LOG_BATCH`
- [ ] Update client reducer to apply owner patch after log application
- [ ] Verify that hand/selection drift cannot persist after a batch is processed

### Phase 2: Stable Card Instance Identity

- [ ] Define engine-side card instance identity source
- [ ] Expose owner-visible `instanceId` in hand snapshots
- [ ] Expose owner-visible `instanceId` in selection snapshots
- [ ] Decide whether board cards should also expose public instance ids
- [ ] Verify hidden opponent zones do not leak stable hidden-card continuity

### Phase 3: Server-Issued Action IDs

- [ ] Add `stateVersion` to owner-facing action data
- [ ] Add `legalActionIds` to websocket action mask
- [ ] Add `legalSourceInstanceIds` to websocket action mask
- [ ] Accept `actionId` in `GAME_ACTION`
- [ ] Keep legacy tuple submission during migration
- [ ] Add state-version mismatch handling

### Phase 4: Browser Migration

- [ ] Refactor hand interaction helpers away from `handIndex -> tuple`
- [ ] Refactor selection interaction helpers away from index-only submission
- [ ] Submit `actionId` from board/hand/selection interactions
- [ ] Keep action mask as the legality source
- [ ] Remove browser dependence on displayed hand order for action submission

### Phase 5: Replay Data Model

- [ ] Define action-journal table schema
- [ ] Persist action journal for every accepted action
- [ ] Persist replay metadata needed for deterministic reconstruction
- [ ] Decide whether finalized logs remain stored in `game_logs` or move to a replay-scoped table
- [ ] Design replay loader from journal
- [ ] Design fast playback from finalized step logs

### Validation / Exit Criteria

- [ ] Reproduce the original return-to-hand ordering scenario and confirm client truth stays aligned
- [ ] Confirm the client can still animate stepwise zone transitions
- [ ] Confirm owner hand ordering matches the engine after every accepted action
- [ ] Confirm browser-submitted actions no longer depend on local hand index ordering
- [ ] Confirm AI / training path remains compatible with indexed engine actions

## Open Questions

- Should owner patches be sent for every accepted action, or only when sensitive ordered zones changed?
- Should log finalization happen once per engine tick, or at finer-grained stable substeps inside a tick?
- Should replay artifacts be tied to `match_results` instead of `rooms`, given current `room_id` cascade behavior on `game_logs`?
- Should stable public card instance identity be exposed for board cards, or only owner-private zones initially?

## Recommended Immediate Next Step

Start with Phase 1A and Phase 1B together:

1. two-phase log finalization for ordered/private zone moves
2. authoritative owner-zone patch for hand and selection

That is the smallest meaningful architecture improvement that:

- addresses the known bug class
- preserves current AI / training behavior
- does not yet require a web action protocol migration
