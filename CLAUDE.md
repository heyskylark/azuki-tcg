# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Azuki TCG is a reinforcement learning environment for a custom trading card game, built with:
- **C Core Engine**: Flecs ECS-based game logic with deterministic simulation
- **Python Bindings**: PufferLib/PettingZoo integration for RL training
- **Multi-head Policy**: LSTM-based neural network with 4 action heads

The project trains competitive agents via self-play using PPO on a multi-agent card game environment.

## Development Environment

### Docker Setup (Recommended)

```bash
PROJECT=~/git/azuki-tcg
PUFFER=~/git/rl/SkyPufferLib

docker run -d --name puffertank-dev \
  --gpus all \
  --network host \
  --ipc host \
  --restart unless-stopped \
  -v "$PROJECT":/workspace \
  -v "$PUFFER":/ext/SkyPufferLib \
  -v "$HOME/.cache/pip":/root/.cache/pip \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/npm":/root/.npm \
  -w /workspace \
  pufferai/puffertank:3.0 bash -lc "sleep infinity"

docker exec -it puffertank-dev bash
```

### System Dependencies

- **Linux**: `sudo apt install libncurses-dev` (Ubuntu/Debian), `sudo dnf install ncurses-devel` (Fedora), `sudo pacman -S ncurses` (Arch)
- **macOS**: `brew install ncurses`
- **Windows**: Use MSYS2 (`pacman -S mingw-w64-x86_64-ncurses`)

## Build Commands

### Building the C Environment

```bash
# Debug build (from repository root)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j

# Release build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Build only the Python binding target
cmake -S . -B build && cmake --build build --target azuki_puffer_env
```

### Cleaning Build Artifacts

```bash
cmake --build build --target clean
```

### Running Tests

```bash
# Build and run C unit tests
cmake --build build
ctest --test-dir build

# Or run specific test directly
./build/world_tests
```

## Running RL Training

```bash
# From repository root
PYTHONPATH=build/python/src:python/src:$PYTHONPATH uv run --active python/src/train.py \
  --config python/config/azuki.ini \
  --train.device cuda \
  --train.total-timesteps 1_000_000
```

**Important**: The `PYTHONPATH` must include `build/python/src` to import the compiled `binding` module.

### Training Configuration

Training parameters are configured in `python/config/azuki.ini`. Key parameters:
- `vec.num_envs`: Number of parallel environments
- `train.device`: `cuda` or `cpu`
- `train.total_timesteps`: Total training steps
- `train.batch_size`: PPO batch size
- `train.bptt_horizon`: LSTM truncation length

Command-line overrides use dot notation: `--train.device cuda --vec.num_envs 12`

## Architecture Overview

### C Core Engine (`src/`, `include/`)

The game engine uses **Flecs ECS** (Entity Component System) architecture:

- **`world.c/h`**: ECS world initialization and core game state singleton
- **`azuki_engine.c`**: High-level API wrapping ECS operations
- **`components.h`**: ECS component definitions (cards, players, phases)
- **`systems/`**: ECS systems for game phases (draw, main, combat, end turn)
- **`queries/`**: Reusable ECS queries for common entity lookups
- **`validation/`**: Action validation and legal move enumeration
- **`utils/`**: Phase transitions, card utilities, RNG helpers
- **`generated/`**: Auto-generated card definitions from JSONL

**Key Concepts**:
- Game state is stored in ECS components, primarily in a `GameState` singleton
- Systems process entities in phases (pregame, main, combat, response, end turn)
- The engine is **deterministic** with explicit RNG state for reproducible training
- All game logic runs in C for performance; Python only handles RL orchestration

### Python Bindings (`python/src/`)

- **`binding.c`**: C extension module exporting PufferLib-compatible interface
- **`tcg.h`**: C header defining observation/action layouts
- **`tcg.py`**: PettingZoo AEC wrapper around the C binding
- **`observation.py`**: Observation space encoding utilities
- **`action.py`**: Action space and multi-head action utilities
- **`train.py`**: PufferLib training entrypoint
- **`policy/tcg_policy.py`**: Multi-head LSTM policy network

### Multi-Head Action Space

Actions are represented as 4 discrete heads:
1. **Head 0** (13 actions): Action type (NOOP, PLAY_ENTITY, ATTACK, END_TURN, etc.)
2. **Head 1** (16 values): Hand index / slot index / ability index
3. **Head 2** (8 values): Target kind/slot combinations
4. **Head 3** (8 values): Auxiliary parameters (gate slot, replacement flag)

Legal action masks are provided per-head via `env.infos[agent]["action_mask"]`.

### Turn-Based AEC Flow

- Two players alternate as `agent_selection`
- During **response windows** (e.g., defender declaring blockers), control switches to the non-active player
- The `tcg.py` wrapper handles agent switching and mask propagation

## Card Generation Pipeline

Card definitions are stored as structured data and code-generated:

```bash
# Generate card definitions from JSONL
python3 scripts/generate_card_defs.py path/to/cards.jsonl \
  -o src/generated/card_defs.c \
  --header include/generated/card_defs.h
```

Card schema is documented in `.codex/docs/cards.schema.md`. Generated files are included in the CMake build automatically.

## Testing Strategy

### C Unit Tests (`tests/`)

Located in `tests/test_world.c` and run via CTest:
```bash
cmake --build build
ctest --test-dir build --verbose
```

### Python Integration Tests

```bash
# From python/src/
pytest python/src/test.py -v
```

### Key Invariants to Test

- **Determinism**: Same seed produces identical game outcomes
- **Action Masking**: No illegal actions should pass validation
- **Zone Consistency**: Cards never occupy invalid zones or duplicate slots
- **Resource Tracking**: IKZ (mana) taps/untaps correctly
- **Combat Resolution**: Damage, death, and weapon attachment logic

## Important Development Notes

### Observation Space

Observations are flat `float32` vectors (`AZK_OBS_LEN` elements) containing:
1. Phase one-hot encoding
2. Leader stats (attack, health, keywords) for both players
3. Gate features (tapped, cooldown)
4. Garden/Alley slots (5 each per player): card stats, tapped, cooldown, keywords
5. Hand encoding (top-K cards with type/cost/element)
6. IKZ (mana) pool state
7. Combat stack summary

See `python/src/observation.py` for encoding details and `.codex/docs/azuki-observations.md` for specification.

### Phase System

Game phases follow a state machine:
```
PREGAME_MULLIGAN_P0 → PREGAME_MULLIGAN_P1 → START_OF_TURN → MAIN →
COMBAT_DECLARED → RESPONSE_WINDOW → COMBAT_RESOLVE → END_TURN
```

- **Main Phase**: Active player plays cards, activates abilities, attacks
- **Response Window**: Defender can play response spells or declare defenders
- **Combat Resolve**: Simultaneous damage, death triggers

Systems in `src/systems/` correspond to these phases.

### ECS Component-System Model

When modifying game logic:
1. **Define components** in `include/components/components.h` (structs with game state)
2. **Create systems** in `src/systems/` that query and modify components
3. **Register systems** in `world.c` initialization with correct phase ordering
4. **Update queries** in `src/queries/` if new entity lookups are needed

Flecs uses a single `ecs_world_t*` (aliased as `AzkEngine*`) for all operations.

### PufferLib Integration

The C binding implements the PufferLib `env_` interface:
- `env_init`: Initialize environment with numpy buffer pointers
- `env_step`: Execute one agent action, update observations/rewards/terminals
- `env_reset`: Reset game state to initial conditions
- `env_get`: Return metadata (obs/action sizes, agent count)

The binding does **not** manage Python memory—PufferLib allocates buffers and passes pointers.

## Documentation

Comprehensive design documents are in `.codex/docs/`:
- `azuki-env-tech-spec.md`: Full C engine specification
- `azuki-training-spec.md`: RL training pipeline details
- `azuki-observations.md`: Observation space layout
- `azuki-product-spec.md`: Game rules and mechanics
- `cards.schema.md`: Card definition format

Refer to these for architectural decisions and implementation details.

## Common Gotchas

- **Build path**: Python training requires `build/python/src` in `PYTHONPATH` to import `binding`
- **Seed management**: Both C engine and Python RNG must be seeded for reproducibility
- **Action masking**: Policies must apply masks before argmax to prevent illegal moves
- **Response windows**: Agent selection switches to defender during combat; ensure policy handles role correctly
- **ECS assertions**: Debug builds use `ecs_assert`; enable them when diagnosing engine bugs

## Web Service (Node.js/TypeScript)

The project includes a web service for online play, consisting of a WebSocket server and a Next.js web application.

### Node.js Version

**Requires Node.js 22 LTS** (uWebSockets.js doesn't support Node 24+).

```bash
# Using nvm (recommended)
nvm use 22

# Or check .nvmrc file
cat .nvmrc
```

### Yarn Workspaces

The web service uses Yarn workspaces for monorepo management:

```bash
# Install all dependencies from root
yarn install

# Run commands in specific workspaces
yarn ws <command>      # @azuki/websocket
yarn web <command>     # @azuki/web
yarn core <command>    # @tcg/backend-core
```

### Running the Web Service

**Docker Compose (Recommended):**

```bash
# Start all services (db, migrations, ws, web)
yarn dev

# Start with file watching for hot reload
yarn dev:watch

# Stop all services
yarn dev:down

# Stop and reset database
yarn dev:clean
```

This starts:
- PostgreSQL on port 5432
- Migrations (runs once, then exits)
- WebSocket server on port 3001
- Next.js web app on port 3000

**Manual (without Docker):**

```bash
# Start database only
docker compose up db -d

# Run migrations
yarn core db:migrate

# WebSocket server (port 3001)
yarn ws dev

# Next.js web app (port 3000)
yarn web dev

# Build backend-core package (required before other builds)
yarn core build
```

### Native Module (Engine Wrapper)

The WebSocket server uses a **native Node.js addon** to interface with the C game engine. This must be built before running the server.

**Build Order:**
1. C Engine Library (`libazuki_lib`)
2. Native Module (`apps/websocket/native/`)
3. TypeScript code

**Development Build:**
```bash
# 1. Build C engine (from repository root)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j

# 2. Build native module
yarn ws build:native

# 3. Now you can run the dev server
yarn ws dev
```

**Production Build:**
```bash
# Build everything for production
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
yarn ws build:all
yarn ws start
```

**When to Rebuild:**
- C engine changes (`src/`, `include/`): Rebuild C engine + native module
- Native wrapper changes (`apps/websocket/native/`): Rebuild native module only
- TypeScript changes: Automatic with `yarn ws dev`

### Web Service Architecture

- **WebSocket Server** (`apps/websocket`): uWebSockets.js server for real-time game communication
  - Handles WebSocket connections, game actions, room management
  - **Native module** wraps C engine for in-process game logic
  - Winston logging

- **Web App** (`apps/web`): Next.js 15 with App Router
  - API routes for REST endpoints (auth, rooms, decks)
  - React frontend with Tailwind CSS
  - React Three Fiber for 3D game rendering (future)

- **Backend Core** (`packages/backend-core`): Shared TypeScript types, constants, utilities, database
  - Enums matching web-service.md spec (RoomStatus, CardType, etc.)
  - WebSocket message types
  - Game constants
  - Drizzle ORM schemas and database connection

### Database Migrations

Drizzle ORM is used for database schema management. Migrations are stored in `packages/backend-core/drizzle/`.

**Schema Changes (modifying .ts schema files):**
```bash
# After modifying any schema file in packages/backend-core/src/drizzle/schemas/
yarn core db:generate
```
This auto-generates a migration SQL file. **Never manually edit generated migration files.**

**Custom Migrations (seed data, manual SQL):**
```bash
# Create an empty custom migration file
cd packages/backend-core
npx drizzle-kit generate --custom --name=seed-cards
```
This creates an empty `.sql` file you can populate with custom SQL (e.g., INSERT statements for seed data).

**Important:**
- Schema changes go through `db:generate` - don't add them to custom migrations
- Custom migrations are for non-schema operations: seed data, data transformations, manual indexes
- Never modify existing migration files that have been committed/run

### Import Conventions

**All TypeScript code must use absolute path aliases. No relative imports.**

| Alias | Resolves To | Usage |
|-------|-------------|-------|
| `@/*` | `./src/*` | Internal imports within the same package |
| `@tcg/backend-core/*` | `packages/backend-core/dist/*` | Imports from backend-core package (apps only, requires `yarn core build` first) |

**Examples:**

```typescript
// In apps/websocket or apps/web - import directly from the source file
import logger from "@/logger";
import { WebSocketService } from "@/services/WebSocketService";
import { RoomStatus } from "@tcg/backend-core/types";  // Defined directly in types/index.ts
import { TokenType } from "@tcg/backend-core/types/auth";  // Defined in types/auth.ts
import db from "@tcg/backend-core/database";

// In packages/backend-core
import { isDefined } from "@/utils";
import { users } from "@/drizzle/schemas/users";
```

**Do NOT use:**
- Relative imports: `../constants/index.js`
- Package imports for internal modules: `@tcg/backend-core` (use `@tcg/backend-core/*` instead)

### No Barrel Exports

**Do NOT create index.ts files that only re-export from other files.** Import directly from the source file where code is defined.

```typescript
// BAD - barrel export file (index.ts that only re-exports)
export { withErrorHandler } from "./withErrorHandler";
export { withAuth } from "./withAuth";
export * from "./auth";  // re-exporting everything from another file

// GOOD - import directly from source files
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth } from "@/lib/hof/withAuth";
import { TokenType } from "@tcg/backend-core/types/auth";
```

**Exceptions:**
- `packages/backend-core/src/drizzle/schemas/index.ts` - Required by Drizzle ORM for schema aggregation
- Files that define exports directly (not just re-export) are fine to import from

### Backend Services Convention

**All services use functional programming style (exported functions, not classes).**

```typescript
// Good - functional service
export async function findUserByEmail(email: string, database = db) { ... }
export async function createUser(params: CreateUserParams, database = db) { ... }

// Bad - class-based service
export class UserService {
  async findUserByEmail(email: string) { ... }
}
```

Services accept an optional `database` parameter for transaction support.

### Service Directory Structure

**For services with constants, configuration data, or multiple helper functions, use a directory structure:**

```
services/
├── userService.ts              # Simple service (single file)
└── DeckService/                # Complex service (directory)
    ├── index.ts                # Service functions (main exports)
    └── constants/
        └── index.ts            # Constants and configuration data
```

**When to use directory structure:**
- Service has significant constant data (e.g., starter deck configurations)
- Service has multiple internal helper functions
- Constants may be reused elsewhere or need separate maintenance

**When to use single file:**
- Simple CRUD operations
- Few or no constants
- Small, focused service

**Example directory-based service:**
```typescript
// services/DeckService/constants/index.ts
export interface StarterCardInfo { cardCode: string; quantity: number; }
export const starterDecks: StarterDeckConfig[] = [...];

// services/DeckService/index.ts
import { starterDecks } from "@/services/DeckService/constants";
export async function addStarterDecks(userId: string, database = db) { ... }
```

### Input Validation Convention

**All API request bodies must be validated with Zod using `.strict()` mode.**

```typescript
import { z } from "zod";

// Email uses Zod's built-in .email() validation
const emailSchema = z
  .string()
  .email("Invalid email format")
  .transform((e) => e.toLowerCase().trim());

// Request schema with .strict() to reject unknown keys
const requestSchema = z.object({
  email: emailSchema,
  password: z.string().min(8).max(128),
}).strict();

// Usage in route handler
const data = requestSchema.parse(await request.json());
```

### TypeScript Type Safety

**Avoid using `as` type assertions and `!` non-null assertions.** Instead, use runtime checks or Zod validation to verify values.

```typescript
// BAD - using 'as' to force a type
const room = result as RoomData;
const value = payload["key"] as string;

// BAD - using '!' non-null assertion
const room = results[0]!;

// GOOD - use proper null checks
const room = results[0];
if (!room) {
  throw new Error("Failed to create room");
}

// GOOD - use .then() to get proper type inference from Drizzle
const room = await database
  .select()
  .from(Rooms)
  .where(eq(Rooms.id, roomId))
  .limit(1)
  .then((results) => results[0]);

// GOOD - use Zod to validate unknown payloads
const payloadSchema = z.object({
  roomId: z.string(),
  playerSlot: z.union([z.literal(0), z.literal(1)]),
});
const result = payloadSchema.safeParse(payload);
if (!result.success) {
  throw new InvalidTokenError();
}
const { roomId, playerSlot } = result.data;
```

**Why:** Type assertions (`as`) bypass TypeScript's type checking and can hide bugs. Use runtime validation to ensure values match expected types.

### API Route Patterns

**Use Higher-Order Functions (HOFs) for consistent error handling and authentication.**

```typescript
// Public route
export const POST = withErrorHandler(handler);

// Protected route
export const GET = withErrorHandler(withAuth(handler));
```

- `withErrorHandler`: Catches errors, returns appropriate HTTP status codes
- `withAuth`: Validates JWT, fetches user, passes to handler as `request.user`

### Error Classes Convention

**Custom errors extend `ApiError` base class with HTTP status code.**

Located in `packages/backend-core/src/errors/`.

```typescript
export class ApiError extends Error {
  public readonly status: number;
  constructor(message: string, status: number) { ... }
}

// Specific errors
export class ValidationError extends ApiError { ... }  // 400
export class UnauthorizedError extends ApiError { ... } // 401
export class EmailAlreadyExistsError extends ApiError { ... } // 409
```

### Web Service Documentation

See `.claude/docs/web-service.md` for comprehensive documentation including:
- Architecture diagrams
- WebSocket protocol
- API endpoints
- Database schema
- Game state logging format

## Project Structure Summary

```
azuki-tcg/
├── src/              # C game engine source
├── include/          # C headers
├── tests/            # C unit tests
├── python/
│   ├── src/          # Python bindings, policy, training
│   └── config/       # Training config files
├── apps/
│   ├── websocket/    # uWebSockets.js server
│   │   ├── native/   # Native Node.js addon (C++ wrapper for C engine)
│   │   │   ├── binding.gyp
│   │   │   └── src/
│   │   └── src/
│   │       ├── server.ts
│   │       ├── engine/       # Engine wrapper (TypeScript)
│   │       ├── handlers/
│   │       ├── services/
│   │       ├── state/
│   │       ├── logger/
│   │       ├── constants/
│   │       └── utils/
│   └── web/          # Next.js web application
│       └── src/app/
├── packages/
│   └── backend-core/ # Backend TypeScript code (types, db, utils)
│       └── src/
│           ├── types/
│           ├── constants/
│           ├── utils/
│           ├── database/
│           ├── drizzle/
│           ├── errors/
│           └── services/
├── scripts/          # Card generation utilities
├── docker/           # Dockerfiles for web services
├── .codex/docs/      # Design documentation (C engine)
├── .claude/docs/     # Design documentation (web service)
├── build/            # CMake build output (gitignored)
├── docker-compose.yml # Web service orchestration
├── package.json      # Yarn workspaces root
├── tsconfig.base.json # Shared TypeScript config
└── .nvmrc            # Node.js version (22)
```

When making changes:
- C engine (`src/`, `include/`): Rebuild with cmake, run C unit tests, rebuild native module
- Native module (`apps/websocket/native/`): Rebuild with `yarn ws build:native`
- Web service TypeScript: Ensure TypeScript compiles without errors, test WebSocket connections
- Python bindings: Ensure Python integration tests pass
