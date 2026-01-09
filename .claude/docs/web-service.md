# Web Service

Web service consists of:
- **C Engine**: Core game logic (Flecs ECS-based, deterministic)
- **WebSocket Server**: uWebSockets.js for real-time game communication
- **API Server**: Next.js API routes for REST endpoints (auth, rooms, decks, etc.)
- **Web Client**: Next.js frontend with React Three Fiber for 3D game rendering

---

# Tech Stack

## Next.js (API + Frontend)

Next.js serves as both the HTTP API server and web frontend in a single codebase.

**API Routes** (`/app/api/`):
- Handle all REST endpoints (auth, users, decks, rooms, cards, matches)
- Server-side only, no client bundle bloat
- TypeScript with Zod validation
- Prisma ORM for PostgreSQL access

**Frontend Pages**:
- Landing page, login/register
- Deck builder with card search and drag-drop
- Room lobby and matchmaking UI
- Match history and replay viewer
- User profile and settings

**Why Next.js**:
- Single codebase for API + frontend reduces complexity
- Server Components for fast initial page loads
- API routes provide type-safe backend without separate Express server
- Easy deployment (Vercel, Docker, self-hosted)

## uWebSockets.js (WebSocket Server)

Separate process from Next.js for real-time game communication.

**Why separate from Next.js**:
- Next.js API routes are request/response based, not suited for persistent WebSocket connections
- uWebSockets.js is the fastest WebSocket implementation for Node.js
- Can scale independently from HTTP API
- Maintains IPC connection to C engine

**Deployment**:
- Runs as separate Node.js process alongside Next.js
- Shares PostgreSQL database for room/user lookups
- JWT validation for WebSocket authentication (tokens issued by Next.js API)

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Browser                        │
├─────────────────────────────────────────────────────────────┤
│  Next.js Frontend (React)     │  React Three Fiber (Game)   │
│  - Pages/Components           │  - 3D Card Rendering        │
│  - Deck Builder               │  - Board State              │
│  - Room UI                    │  - Animations/Effects       │
└──────────────┬────────────────┴──────────────┬──────────────┘
               │ HTTP/REST                      │ WebSocket
               ▼                                ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   Next.js API Routes     │    │   uWebSockets.js Server      │
│   - /api/auth/*          │    │   - Real-time game messages  │
│   - /api/rooms/*         │    │   - GAME_LOG_BATCH           │
│   - /api/decks/*         │    │   - GAME_SNAPSHOT            │
│   - /api/cards/*         │    │   - Room state updates       │
└──────────────┬───────────┘    └──────────────┬───────────────┘
               │                                │ IPC (Unix Socket)
               │                                ▼
               │                 ┌──────────────────────────────┐
               │                 │      C Engine Process        │
               │                 │   - Flecs ECS worlds         │
               │                 │   - Game logic               │
               │                 │   - Action validation        │
               ▼                 └──────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                      PostgreSQL                              │
│  users, decks, cards, rooms, match_results, game_logs       │
└─────────────────────────────────────────────────────────────┘
```

## React Three Fiber (Game Client)

3D game rendering within the Next.js React application.

**Core Libraries**:
- `@react-three/fiber` - React renderer for Three.js
- `@react-three/drei` - Useful helpers (OrbitControls, Text, etc.)
- `@react-three/postprocessing` - Visual effects (bloom, vignette)

**Game Scene Components**:
```
<Canvas>
  <GameBoard>
    <PlayerArea position="bottom">
      <LeaderCard />
      <GateCard />
      <GardenSlots />  {/* 5 entity slots */}
      <AlleySlots />   {/* 5 entity slots */}
      <HandCards />
      <IKZPool />
    </PlayerArea>
    <PlayerArea position="top">
      {/* Opponent's board - mirrored */}
    </PlayerArea>
    <CombatZone />     {/* Active combat visualization */}
  </GameBoard>
  <CardDetailOverlay />
  <ActionMaskUI />     {/* Valid action highlights */}
</Canvas>
```

**State Management**:
- Zustand store for client game state (built from GAME_LOG_BATCH)
- React Query for API data (decks, cards, user info)
- WebSocket messages update Zustand store directly

**Animation Approach**:
- Card movements animated via `@react-spring/three`
- Combat effects via particle systems
- State changes trigger animations based on log type

---

# Architecture

## C Engine <-> WebSocket Integration

**Recommended Approach: Separate Process + IPC (Unix Domain Sockets or Named Pipes)**

For handling ~1000 concurrent games with lightweight game state, we recommend running the C engine as a separate long-running process that communicates with the Node.js WebSocket server via IPC.

**Why IPC over other approaches:**
- **Process isolation**: A crash in one game world doesn't bring down the WS server or other games
- **Memory safety**: C engine runs in its own address space
- **Scalability**: Can spawn multiple engine processes across CPU cores
- **Simpler debugging**: Can restart engine process without restarting WS server

**Implementation options:**
1. **Single engine process with multiplexed worlds** - One C process manages all game worlds internally, communicates via a single IPC channel. Simpler but less isolated.
2. **Pool of engine worker processes** - N worker processes, each handling M games. Node.js routes messages to the appropriate worker. Better isolation, more complex routing.
3. **One process per game** (not recommended for 1000 games due to process overhead)

**Recommended: Option 1 for V0, migrate to Option 2 if scaling issues arise.**

### IPC Message Format

**Format: JSON** (human-readable, easy debugging; can migrate to binary format like MessagePack if perf issues arise)

Actions are simple numerical tuples: `[head0, head1, head2, head3]` (4 int8 values).
All card references use engine numerical IDs (`CardDefId` enum values), not database UUIDs.

```
Node.js -> C Engine:
{
    "type": "ACTION",
    "world_id": "uuid",
    "player_id": "uuid",
    "action": [0, 0, 0, 0]
}

{
    "type": "CREATE_WORLD",
    "world_id": "uuid",
    "rng_seed": 12345,
    "player0_deck": [1, 2, 3, ...],  // CardDefId array
    "player1_deck": [4, 5, 6, ...]
}

{
    "type": "DESTROY_WORLD",
    "world_id": "uuid"
}
```

```
C Engine -> Node.js:
{
    "type": "LOG_BATCH",
    "world_id": "uuid",
    "state_context": {
        "phase": "MAIN",
        "ability_subphase": "NONE",
        "active_player": 0,
        "turn_number": 5
    },
    "logs": [...],
    "action_mask": {...}  // null if not active player's turn
}

{
    "type": "ERROR",
    "world_id": "uuid",
    "code": "INVALID_ACTION",
    "message": "..."
}
```

### World Isolation & Crash Handling

- Each game world is a separate `ecs_world_t*` instance within the C process
- Worlds are stored in a hash map keyed by `world_id`
- On action error, return error response but don't crash the process
- Use signal handlers (SIGSEGV, SIGABRT) to catch fatal errors and log the offending world_id
- If the engine process crashes, the WS server should:
  1. Mark all active games as ABORTED in the database
  2. Notify connected clients of the error
  3. Restart the engine process
  4. (Future) Restore games from persisted state

---

# Data Models

## Enums

```typescript
// User status
enum UserStatus { ACTIVE, DELETED, BANNED }

// User type
enum UserType { HUMAN, AI }

// Deck status
enum DeckStatus { COMPLETE, IN_PROGRESS, DELETED }

// Room status
enum RoomStatus {
    WAITING_FOR_PLAYERS,
    DECK_SELECTION,
    READY_CHECK,
    STARTING,
    IN_MATCH,
    COMPLETED,
    ABORTED
}

// Room type
enum RoomType { PRIVATE, MATCH_MAKING }

// Match result type
enum WinType { WIN, DRAW, ABANDON, FORFEIT, TIMEOUT }

// Card rarity (matches C engine)
enum CardRarity { L, G, C, UC, R, SR, IKZ }

// Card element (matches C engine)
enum CardElement { NORMAL, LIGHTNING, WATER, EARTH, FIRE }

// Card type (matches C engine)
enum CardType { LEADER, GATE, ENTITY, WEAPON, SPELL, IKZ, EXTRA_IKZ }
```

## Tables

```json
// users
{
    id: uuid,
    username: string,          // unique
    password_hash: string,     // bcrypt hash
    type: UserType,
    status: UserStatus,
    model_key?: string,        // AI model path (null for humans)
    created_at: Date,
    updated_at: Date
}

// decks
// Each user when created will also have STT01 and STT02 starter decks auto-added (they can't be deleted).
// Deck element is derived from the leader/gate card element (not stored separately).
{
    id: uuid,
    name: string,
    user_id: uuid,             // users fk relation
    leader_card_id: uuid,      // cards fk relation (required)
    gate_card_id: uuid,        // cards fk relation (required)
    status: DeckStatus,
    is_system_deck: bool,      // for STT01/STT02 starter decks, use PSQL CHECK constraint to prevent deletion
    created_at: Date,
    updated_at: Date
}

// deck_card_junctions
{
    id: uuid,
    deck_id: uuid,             // cascade delete
    card_id: uuid,             // cards fk relation
    quantity: int,             // number of copies (1-4 typically)
    created_at: Date,
    updated_at: Date
}

// cards (pre-generated from card_defs, seeded to DB)
{
    id: uuid,
    engine_id: int,            // CardDefId from C engine (0-35 currently)
    card_code: string,         // e.g., "STT01-001", "IKZ-001"
    name: string,              // e.g., "Raizan", "Lightning Shuriken"
    rarity: CardRarity,        // L, G, C, UC, R, SR, IKZ
    element: CardElement,      // NORMAL, LIGHTNING, WATER, EARTH, FIRE
    card_type: CardType,       // LEADER, GATE, ENTITY, WEAPON, SPELL, IKZ, EXTRA_IKZ
    attack?: int,              // null for spells, gates, IKZ
    health?: int,              // null for spells, weapons, gates, IKZ
    gate_points?: int,         // entities only (0-4 typically)
    ikz_cost?: int,            // null for leaders, gates, IKZ cards
    keywords: string[],        // ["charge", "defender", "infiltrate", "godmode", "effect_immune"]
    subtypes: string[],        // ["Black Jade", "Steelborn", "Raizan", ...] (max 3)
    effect_text?: string,      // card effect description from CSV
    flavor_text?: string,      // flavor text from CSV
    image_url?: string,        // card art URL
    created_at: Date,
    updated_at: Date
}

// rooms
{
    id: uuid,
    status: RoomStatus,
    type: RoomType,
    password_hash?: string,    // bcrypt hash for private rooms
    world_id?: string,         // C engine world identifier
    rng_seed?: int,            // seed for deterministic replays
    player0_id?: uuid,         // users fk relation
    player0_deck_id?: uuid,    // decks fk relation
    player0_ready: boolean,
    player1_id?: uuid,         // users fk relation
    player1_deck_id?: uuid,    // decks fk relation
    player1_ready: boolean,
    deck_selection_deadline?: Date,  // timeout timestamp
    created_at: Date,
    updated_at: Date
}

// match_results
{
    id: uuid,
    room_id: uuid,             // rooms fk relation
    player0_id: uuid,          // users fk relation
    player1_id: uuid,          // users fk relation
    winner_id?: uuid,          // null for draw
    win_type: WinType,
    total_turns: int,
    duration_seconds: int,
    created_at: Date,
    updated_at: Date
}

// game_logs (for replays and debugging)
{
    id: uuid,
    room_id: uuid,             // rooms fk relation
    batch_number: int,         // which action batch this belongs to
    sequence_number: int,      // order within batch
    log_type: string,          // GameLogType value (e.g., "CARD_ZONE_MOVED")
    player: int?,              // 0, 1, or null (for game-level events like GAME_ENDED)
    log_data: json,            // full log payload
    created_at: Date
}
```

### Deck Validation Rules

Validated client-side for UX, then re-validated server-side on deck save:

- **Deck size**: Exactly 40 main deck cards (excluding leader and gate)
- **Leader**: Exactly 1 leader card
- **Gate**: Exactly 1 gate card (must match leader element)
- **Copy limit**: Maximum 4 copies of any non-leader/gate card
- **Element restriction**: All deck cards must match the leader/gate element OR be NORMAL element. No mixing of non-NORMAL elements.

---

# Authentication

## JWT Authentication Flow

1. User registers/logs in via HTTP API
2. Server returns access token (short-lived, 15min) and refresh token (long-lived, 7 days)
3. Access token included in `Authorization: Bearer <token>` header for HTTP requests
4. Refresh token used to obtain new access tokens

## WebSocket Authentication

1. Client requests room join via HTTP `POST /rooms/:roomId/join` with JWT
2. Server validates JWT, checks room access, returns short-lived `joinToken` (5 min expiry)
3. Client connects to WebSocket with `joinToken` as query param or in initial message
4. WS server validates `joinToken`, associates connection with user and room

```
HTTP: POST /rooms/:roomId/join
Headers: Authorization: Bearer <jwt>
Body: { "password": "room_password" }  // if private room
Response: {
    "joinToken": "...",
    "wsChannel": "/rooms/:roomId",
    "expiresAt": "2024-01-01T12:05:00Z"
}
```

---

# WebSocket Protocol

## Connection Flow

```
Client                          Server
   |                               |
   |-- WS Connect + joinToken ---->|
   |                               | (validate joinToken)
   |<---- CONNECTION_ACK ----------|
   |                               |
   |<---- ROOM_STATE --------------|  (current room state)
   |                               |
```

## Message Types

### Client -> Server

```typescript
// Select deck for the match
{ type: "SELECT_DECK", deckId: "uuid" }

// Signal ready to start
{ type: "READY", ready: boolean }

// Submit game action (during match)
{ type: "GAME_ACTION", action: [number, number, number, number] }

// Resign/forfeit the match
{ type: "FORFEIT" }

// Heartbeat/ping
{ type: "PING" }
```

### Server -> Client

```typescript
// Connection acknowledged
{ type: "CONNECTION_ACK", playerId: "uuid", playerSlot: 0 | 1 }

// Room state update (pre-match)
{
    type: "ROOM_STATE",
    status: RoomStatus,
    players: [
        { id: "uuid", username: "...", deckSelected: boolean, ready: boolean },
        { ... } | null
    ],
    deckSelectionDeadline?: string  // ISO timestamp
}

// Full game state snapshot (for reconnection/spectator join only)
{
    type: "GAME_SNAPSHOT",
    state_context: {
        phase: string,
        ability_subphase: string,
        active_player: 0 | 1,
        turn_number: number
    },
    players: [
        {
            leader: { card_id: string, zone_index: number, cur_hp: number, cur_atk: number, keywords: string[], status_effects: object[] },
            gate: { card_id: string, zone_index: number, tapped: boolean, cooldown: number },
            garden: [...],      // 5 slots with card state or null
            alley: [...],       // 5 slots with card state or null
            ikz_pool: [...],    // tapped IKZ cards
            hand_count: number,
            deck_count: number,
            discard_count: number
        },
        { ... }
    ],
    your_hand: [...],           // only for the reconnecting player
    combat_stack: [...],
    action_mask: {...} | null   // if it's your turn
}

// Game over
{
    type: "GAME_OVER",
    winnerId: "uuid" | null,
    winType: WinType,
    reason: string
}

// Error
{ type: "ERROR", code: string, message: string }

// Heartbeat response
{ type: "PONG" }
```

---

# Game State Logging

Instead of sending full observations on every update, the server sends incremental **game state logs** that describe what changed. This reduces bandwidth and enables replay functionality.

## Log Type Enumeration

```c
typedef enum {
    // Zone & Card State
    GLOG_CARD_ZONE_MOVED,           // Card moved between zones
    GLOG_CARD_STAT_CHANGE,          // Attack/health delta
    GLOG_CARD_TAP_STATE_CHANGED,    // Tapped, untapped, or cooldown

    // Status Effects
    GLOG_STATUS_EFFECT_APPLIED,     // Frozen/Shocked/EffectImmune added
    GLOG_STATUS_EFFECT_EXPIRED,     // Status effect removed

    // Combat (granular)
    GLOG_COMBAT_DECLARED,           // Attacker + target
    GLOG_DEFENDER_DECLARED,         // Intercepting entity
    GLOG_COMBAT_DAMAGE,             // Damage dealt to each combatant
    GLOG_ENTITY_DIED,               // Entity HP <= 0, going to discard

    // Abilities
    GLOG_EFFECT_QUEUED,             // Triggered ability enters queue
    GLOG_CARD_EFFECT_ENABLED,       // Ability actually fires/resolves

    // Deck Operations
    GLOG_DECK_SHUFFLED,             // Deck order randomized (mulligan, effects)

    // Game Flow
    GLOG_TURN_STARTED,              // Turn begins for a player
    GLOG_TURN_ENDED,                // Turn ends
    GLOG_GAME_ENDED,                // Winner determined
} GameLogType;
```

## WebSocket Batch Format

Logs are sent in batches with game state context:

```json
{
    "type": "GAME_LOG_BATCH",
    "state_context": {
        "phase": "MAIN",
        "ability_subphase": "NONE",  // or CONFIRMATION, COST_SELECTION, EFFECT_SELECTION
        "active_player": 0,
        "turn_number": 5
    },
    "logs": [
        { "type": "CARD_ZONE_MOVED", ... },
        { "type": "CARD_TAP_STATE_CHANGED", ... }
    ],
    "action_mask": null  // Only present when phase requires input
}
```

### Batch Timing
- **Send after each user action is processed** - immediate feedback
- User submits action → engine processes → batch sent with results
- Auto-triggered effects included in same batch as triggering action

### Action Mask Delivery
- **Sent only** when transitioning to phases requiring input: MAIN, RESPONSE_WINDOW, PREGAME_MULLIGAN
- **Format**: Multi-head mask matching RL training format
- **Recipient**: Active player only (inactive player receives null mask)

## Visibility & Per-Player Batches

**Separate batches are sent to each player.** Each player receives their own batch with opponent's private info redacted.

Example: Player 0 draws a card
- Player 0 batch: `{ card_id: "STT01-005", from_zone: "DECK", from_index: 0, to_zone: "HAND", to_index: 5, metadata: {...} }`
- Player 1 batch: `{ card_id: "HIDDEN", from_zone: "DECK", from_index: null, to_zone: "HAND", to_index: null, metadata: null }`

| Log Type | Owner Sees | Opponent Sees |
|----------|------------|---------------|
| CARD_ZONE_MOVED (Deck→Hand) | Full (card_id, metadata, indices) | Redacted (card_id="HIDDEN", metadata=null, indices=null) |
| CARD_ZONE_MOVED (Deck→Selection) | Full | Redacted |
| CARD_ZONE_MOVED (Selection→Hand/Alley) | Full | Full (card is revealed to both players) |
| CARD_ZONE_MOVED (Selection→BottomDeck) | Full | Redacted (card never revealed) |
| CARD_ZONE_MOVED (Hand→Board) | Full | Full (card is now public on board) |
| CARD_ZONE_MOVED (Board→anywhere) | Full | Full |
| All other log types | Full | Full |

**Redaction rule**: When `card_id` is `"HIDDEN"`, `metadata` MUST also be `null` to prevent stat inference.

**Future spectator support**: Spectators will receive both players' unredacted logs (same as database storage).

## Log Type Structures

### CARD_ZONE_MOVED
```json
// Full version (owner sees, or public move like Hand→Board)
{
    "type": "CARD_ZONE_MOVED",
    "player": 0,
    "card_id": "STT01-005",
    "from_zone": "HAND",
    "from_index": 3,
    "to_zone": "GARDEN",
    "to_index": 2,
    "metadata": {
        "cur_atk": 3,
        "cur_hp": 4,
        "tapped": false,
        "cooldown": false,
        "keywords": ["defender"],
        "attached_weapons": []
    }
}

// Redacted version (opponent sees for Deck→Hand, Deck→Selection, etc.)
{
    "type": "CARD_ZONE_MOVED",
    "player": 0,
    "card_id": "HIDDEN",
    "from_zone": "DECK",
    "from_index": null,
    "to_zone": "HAND",
    "to_index": null,
    "metadata": null
}
```

### CARD_TAP_STATE_CHANGED
```json
{
    "type": "CARD_TAP_STATE_CHANGED",
    "player": 0,
    "card_id": "STT01-005",
    "zone": "GARDEN",
    "zone_index": 2,
    "new_state": "TAPPED"  // TAPPED, UNTAPPED, or COOLDOWN
}
```

### CARD_STAT_CHANGE
```json
{
    "type": "CARD_STAT_CHANGE",
    "player": 0,
    "card_id": "STT01-005",
    "zone": "GARDEN",
    "zone_index": 2,
    "atk_delta": -2,
    "hp_delta": 0,
    "new_atk": 1,
    "new_hp": 4
}
```

### STATUS_EFFECT_APPLIED
```json
{
    "type": "STATUS_EFFECT_APPLIED",
    "player": 0,
    "card_id": "STT01-005",
    "zone": "GARDEN",
    "zone_index": 2,
    "effect": "FROZEN",
    "duration": 2  // -1 for permanent
}
```

### STATUS_EFFECT_EXPIRED
```json
{
    "type": "STATUS_EFFECT_EXPIRED",
    "player": 0,
    "card_id": "STT01-005",
    "zone": "GARDEN",
    "zone_index": 2,
    "effect": "FROZEN"
}
```

### COMBAT_DECLARED
```json
{
    "type": "COMBAT_DECLARED",
    "attacker": {
        "player": 0,
        "card_id": "STT01-005",
        "zone": "GARDEN",
        "zone_index": 2
    },
    "target": {
        "player": 1,
        "card_id": "STT02-001",
        "zone": "LEADER",
        "zone_index": 0
    }
}
```

### DEFENDER_DECLARED
```json
{
    "type": "DEFENDER_DECLARED",
    "defender": {
        "player": 1,
        "card_id": "STT02-008",
        "zone": "GARDEN",
        "zone_index": 1
    }
}
```

### COMBAT_DAMAGE
```json
{
    "type": "COMBAT_DAMAGE",
    "attacker": {
        "player": 0,
        "card_id": "STT01-005",
        "zone": "GARDEN",
        "zone_index": 2,
        "damage_dealt": 3,
        "damage_taken": 2
    },
    "defender": {
        "player": 1,
        "card_id": "STT02-001",
        "zone": "LEADER",
        "zone_index": 0,
        "damage_dealt": 2,
        "damage_taken": 3
    }
}
```

### ENTITY_DIED
```json
{
    "type": "ENTITY_DIED",
    "player": 0,
    "card_id": "STT01-005",
    "zone": "GARDEN",
    "zone_index": 2,
    "cause": "COMBAT"  // COMBAT, ABILITY, or EFFECT
}
```

### EFFECT_QUEUED
```json
{
    "type": "EFFECT_QUEUED",
    "player": 0,
    "card_id": "STT01-011",
    "zone": "GARDEN",
    "zone_index": 0,
    "ability_index": 0,
    "trigger_reason": "ON_PLAY"  // ON_PLAY, WHEN_EQUIPPED, WHEN_ATTACKING, etc.
}
```

### CARD_EFFECT_ENABLED
```json
{
    "type": "CARD_EFFECT_ENABLED",
    "player": 0,
    "card_id": "STT01-011",
    "zone": "GARDEN",
    "zone_index": 0,
    "ability_index": 0,
    "ability_name": "Steelborn Strike"  // optional, for client display
}
```

### DECK_SHUFFLED
```json
{
    "type": "DECK_SHUFFLED",
    "player": 0,
    "reason": "MULLIGAN"  // MULLIGAN or EFFECT
}
```

### TURN_STARTED
```json
{
    "type": "TURN_STARTED",
    "player": 0,
    "turn_number": 5
}
```

### TURN_ENDED
```json
{
    "type": "TURN_ENDED",
    "player": 0,
    "turn_number": 5
}
```

### GAME_ENDED
```json
{
    "type": "GAME_ENDED",
    "winner": 0,  // 0, 1, or 2 for draw
    "reason": "LEADER_DEFEATED"  // LEADER_DEFEATED, DECK_OUT, CONCEDE
}
```

## Data Flow

```
Player Action → WebSocket Server → IPC → C Engine
                                         ↓
                              Process action, generate logs
                                         ↓
                              IPC Response (logs + state_context + mask?)
                                         ↓
                              WebSocket Server splits per-player
                                         ↓
                    ┌────────────────────┴────────────────────┐
                    ↓                                         ↓
            Player 0 Batch                            Player 1 Batch
            (full info for P0,                        (full info for P1,
             redacted for P1)                          redacted for P0)
```

## Implementation Notes

### C Engine Changes
1. Add `GameStateLog` struct and enum to `include/components/`
2. Add `GameStateLogContext` singleton to ECS world
3. Add `azk_log_*()` helper functions for each log type
4. Instrument existing systems to call log helpers at state-change points
5. Clear logs at start of each action processing

### WebSocket Server Changes
1. After IPC response, extract logs from engine
2. Clone logs array for each player
3. Apply visibility redaction to opponent's private info
4. Serialize and send batches

### Database Storage (for replays)
- Store full logs (no redaction) with game_id
- Store action_mask at each decision point (optional, for analysis)
- Replay: seed + full logs = deterministic reconstruction

### Selection Zone Visibility
- Active player sees card_id when cards enter Selection zone
- Opponent sees "HIDDEN" until card moves to Hand/Alley (then revealed)
- Cards moved to bottom of deck remain hidden to opponent

### IKZ Payment
- Individual `CARD_TAP_STATE_CHANGED` per IKZ card tapped
- Playing a 3-cost entity generates 3 separate tap logs

### Triggered Effect Queue
- `EFFECT_QUEUED` fires when ability enters the triggered effect queue
- `CARD_EFFECT_ENABLED` fires when ability actually resolves
- Helps debug complex trigger chains (on-play → when-equipped → when-attacking)

---

# Room Lifecycle

## State Machine

```
                  +---> COMPLETED
                  |
WAITING_FOR_PLAYERS --> DECK_SELECTION --> READY_CHECK --> STARTING --> IN_MATCH
        |                    |                  |              |            |
        |                    v                  v              v            v
        +---------------> ABORTED <------------+<-------------+<-----------+
```

## State Transitions

| From | To | Trigger |
|------|-----|---------|
| WAITING_FOR_PLAYERS | DECK_SELECTION | Player 1 joins room |
| DECK_SELECTION | READY_CHECK | Both players select decks |
| DECK_SELECTION | ABORTED | Deck selection timeout (2 min) |
| READY_CHECK | STARTING | Both players ready |
| READY_CHECK | DECK_SELECTION | Player un-readies |
| STARTING | IN_MATCH | Engine world created, player slots assigned via coin flip |
| IN_MATCH | COMPLETED | Game ends normally (win/draw) |
| IN_MATCH | ABORTED | Disconnect timeout (1 min) without reconnect |
| IN_MATCH | COMPLETED | Player forfeits |
| * | ABORTED | Room creator cancels (pre-match only) |

## Player Assignment

Player slot assignment (player0 vs player1) is **not** determined by room creation order. When transitioning from STARTING to IN_MATCH:
1. Server performs a coin flip using the room's `rng_seed`
2. Players are randomly assigned to player0/player1 slots
3. player0 always acts first in the game

## Mulligan Phase

Mulligan decisions are handled as regular game actions (same 4-head action format). The game starts in `PREGAME_MULLIGAN_P0` phase, each player submits their mulligan action in turn, then proceeds to `START_OF_TURN`.

## Timeouts

- **Deck selection**: 2 minutes from room becoming full
- **Reconnection**: 1 minute from disconnect
- **Turn timer**: (V1) TBD, not in V0

## Rate Limiting

To prevent abuse/DDoS:
- **Actions**: Max 10 actions per second per player (soft limit, excess queued)
- **WebSocket messages**: Max 30 messages per second per connection
- **Room creation**: Max 5 rooms per user per minute
- **API requests**: Standard rate limiting per IP/user

## Reconnection Flow

1. Player disconnects (WS connection closes)
2. Server marks player as disconnected, starts 1 min timer
3. Player reconnects via HTTP `POST /rooms/:roomId/rejoin` with JWT
4. Server validates:
   - User is player0 or player1 in the room
   - Room is still IN_MATCH
   - Reconnection timeout hasn't expired
5. Server issues new `joinToken`
6. Player connects to WS with new `joinToken`
7. Client shows loading state while waiting for game data
8. Server sends `GAME_SNAPSHOT` to reconnected player
9. Client exits loading state, player can resume actions

---

# API Endpoints

## Authentication

```
POST /auth/register
Body: { username, password }
Response: { userId, accessToken, refreshToken }

POST /auth/login
Body: { username, password }
Response: { userId, accessToken, refreshToken }

POST /auth/refresh
Body: { refreshToken }
Response: { accessToken, refreshToken }

POST /auth/logout
Headers: Authorization: Bearer <token>
Response: { success: true }
```

## Users

```
GET /users/me
Headers: Authorization: Bearer <token>
Response: { id, username, type, status, createdAt }

PATCH /users/me
Headers: Authorization: Bearer <token>
Body: { username?, password? }
Response: { id, username, ... }

DELETE /users/me
Headers: Authorization: Bearer <token>
Response: { success: true }
```

## Cards

```
GET /cards
Query: ?limit=20&offset=0&element=LIGHTNING&type=ENTITY&rarity=SR&search=Raizan&cost_min=3&cost_max=5&subtype=BlackJade
Response: {
    cards: [...],
    total: number,
    limit: number,
    offset: number
}

GET /cards/:id
Response: { id, engineId, cardCode, name, ... }
```

## Decks

```
GET /decks
Headers: Authorization: Bearer <token>
Response: { decks: [...] }

GET /decks/:id
Headers: Authorization: Bearer <token>
Response: {
    id, name, status, isSystemDeck,
    element: CardElement,     // derived from leader/gate
    leader: { ... },
    gate: { ... },
    cards: [{ card: {...}, quantity: number }, ...]
}

POST /decks
Headers: Authorization: Bearer <token>
Body: {
    name,
    leaderCardId,
    gateCardId,
    cards: [{ cardId, quantity }, ...]
}
Response: { id, ... }
// Server validates deck rules before saving (element derived from leader)

PATCH /decks/:id
Headers: Authorization: Bearer <token>
Body: { name?, leaderCardId?, gateCardId?, cards? }
Response: { id, ... }
// Cannot modify is_system_deck decks

DELETE /decks/:id
Headers: Authorization: Bearer <token>
Response: { success: true }
// Cannot delete is_system_deck decks
```

## Rooms

```
GET /rooms
Headers: Authorization: Bearer <token>
Query: ?status=WAITING_FOR_PLAYERS
Response: { rooms: [...] }

GET /rooms/:id
Headers: Authorization: Bearer <token>
Response: { id, status, type, players: [...], ... }

POST /rooms
Headers: Authorization: Bearer <token>
Body: { password?: string }
Response: { id, joinToken, wsChannel }
// Creator joins room; player slot (0 or 1) assigned randomly when match starts

POST /rooms/:id/join
Headers: Authorization: Bearer <token>
Body: { password?: string }
Response: { joinToken, wsChannel, expiresAt }

POST /rooms/:id/rejoin
Headers: Authorization: Bearer <token>
Response: { joinToken, wsChannel, expiresAt }
// Only valid if user is already a player in an IN_MATCH room

DELETE /rooms/:id
Headers: Authorization: Bearer <token>
Response: { success: true }
// Only room creator can delete, only pre-match
```

## Match History

```
GET /matches
Headers: Authorization: Bearer <token>
Query: ?limit=20&offset=0
Response: {
    matches: [
        { id, opponentUsername, winnerId, winType, totalTurns, duration, createdAt },
        ...
    ],
    total, limit, offset
}

GET /matches/:id
Headers: Authorization: Bearer <token>
Response: {
    id, roomId, players: [...], winnerId, winType,
    totalTurns, duration, createdAt
}

GET /matches/:id/log
Headers: Authorization: Bearer <token>
Response: {
    rngSeed: number,
    events: [
        { sequenceNumber, eventType, playerId, actionData, eventData, createdAt },
        ...
    ]
}
// For replay functionality - includes rng_seed for deterministic replay
```

---

# Implementation Notes

## V0 Scope

**In scope:**
- Private password-protected rooms only
- 2-player matches
- Basic deck builder with validation
- Real-time gameplay via WebSocket
- Game logging for future replays
- Reconnection within 1 minute
- Basic match history
- STT01/STT02 starter decks auto-created for new users

**Out of scope (future):**
- Public matchmaking queue
- Spectator mode
- Turn timers / chess clock
- ELO/ranking system
- Friends list
- Game state persistence for crash recovery
- AI opponents

## Database

PostgreSQL recommended for:
- JSONB support for flexible event logging
- Strong consistency for game state
- Good indexing for card search
- CHECK constraints for is_system_deck protection

## Client Behavior

- **Server-authoritative**: Client does not predict game state; all updates come from engine via WebSocket
- **Action masking**: Client uses `legalActions` masks to grey out/disable invalid actions in UI
- **No client-side validation bypass**: If user somehow sends invalid action, engine rejects it and returns error
- **Hand visibility**: Client sees same information as RL training model (own hand, opponent hand count only)

## Deployment Considerations

- Single server for V0 (Node.js WS server + C engine process)
- Redis for session storage and rate limiting (optional for V0)
- NGINX for TLS termination and static asset serving
- Consider container deployment (Docker) for engine isolation

---

# AI Players (Future)

- Eventually users can play against trained RL models
- `users.type = AI` distinguishes AI from human players
- `users.model_key` points to the model checkpoint path
- AI player runs in a separate process, receives observations, returns actions
- Room can be created with AI as player1: `POST /rooms { aiOpponent: "model_key" }`
