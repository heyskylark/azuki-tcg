# @azuki/websocket

WebSocket server for Azuki TCG real-time game communication using uWebSockets.js.

## Requirements

- **[Bun](https://bun.sh)** - JavaScript/TypeScript runtime and package manager
- **CMake 3.16+** (for building the C engine)
- **C compiler** (gcc, clang, or MSVC)
- **Python 3** (for node-gyp)
- **Node.js** (for node-gyp header compatibility when building native modules)

## Architecture

The WebSocket server uses a **native Node.js addon** to interface with the C game engine. This provides:
- Direct in-process communication (no IPC overhead)
- Multi-world management (multiple concurrent games)
- Full game state access for observations and action validation

```
┌─────────────────────────────────────────────────────────┐
│                 WebSocket Server (Bun)                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Native Module (C)                  │    │
│  │         apps/websocket/native/                  │    │
│  │  ┌───────────────────────────────────────────┐  │    │
│  │  │           C Game Engine                   │  │    │
│  │  │   libazuki_lib (Flecs ECS)                │  │    │
│  │  └───────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Build Order

The build has three stages that must be completed in order:

1. **C Engine Library** - Core game logic (libazuki_lib)
2. **Native Module** - N-API addon wrapping the C engine
3. **TypeScript** - WebSocket server code

## Development Setup

### Step 1: Build the C Engine

From the repository root:

```bash
# Debug build (faster compilation, includes debug symbols)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j

# Or Release build (optimized)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

This creates `build/libazuki_lib.a` (or `.so`/`.dll` depending on platform).

### Step 2: Build the Native Module

```bash
# From repository root
bun ws build:native

# Or from apps/websocket directory
cd apps/websocket
bun run build:native
```

This runs `node-gyp rebuild` which:
- Compiles the C wrapper code in `native/src/`
- Links against `libazuki_lib` and `libflecs`
- Outputs `native/build/Release/azuki_engine.node`

### Step 3: Run Development Server

```bash
# From repository root (starts all services via Docker Compose)
bun dev

# Or run just the WebSocket server locally
bun ws dev
```

### Full Development Workflow

```bash
# 1. Build C engine (only needed once, or after engine changes)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j

# 2. Build native module (only needed once, or after native/ changes)
bun ws build:native

# 3. Start development (with hot reload for TypeScript)
bun dev
```

## Native Debugging (Docker + GDB + Sanitizers)

This repo includes a debug Dockerfile and compose override to help catch native
segfaults with `gdb` and sanitizer instrumentation. Sanitizers are **development
only** and should not be used in production builds.

### Run the stack with a debug WebSocket container

Open two terminals:

```bash
# Terminal 1: run the rest of the stack (db, migrate, web)
bun dev:infra
```

```bash
# Terminal 2: run the WebSocket service under gdb (with sanitizers enabled)
bun ws debug
```

`gdb` will auto-run the server. When it crashes, use:

```gdb
bt
thread apply all bt
```

Notes:
- `bun ws debug` uses `docker-compose.debug.yml` + `docker/ws.debug.Dockerfile`.
- The debug image builds the C engine and native addon with ASan/UBSan and debug symbols.
- The debug image is cached; rebuild only when C/C++ sources change:

```bash
bun ws debug:build
```
- If ASan builds are too slow, use the no-sanitizer debug image:

```bash
bun ws debug:nosan:build
bun ws debug:nosan
```
- If you need to tweak sanitizer behavior, set `ASAN_OPTIONS` / `UBSAN_OPTIONS` in your shell before running `bun ws debug`.

## Production Build

### Option 1: Docker (Recommended)

```bash
# Build and start all services
docker compose up --build

# Or build just the WebSocket service
docker compose build ws
```

The Dockerfile handles all build steps including the C engine and native module.

### Option 2: Manual Build

```bash
# 1. Build C engine (Release mode for production)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 2. Build everything (native module + TypeScript)
bun ws build:all

# 3. Start server
bun ws start
```

### Build Scripts

| Script | Description |
|--------|-------------|
| `bun ws dev` | Start development server with hot reload |
| `bun ws build` | Build TypeScript only |
| `bun ws build:native` | Build native module only |
| `bun ws build:all` | Build native module + TypeScript |
| `bun ws start` | Start production server |

## Troubleshooting

### Native Module Build Fails

**Missing C engine library:**
```
error: cannot find -lazuki_lib
```
Solution: Build the C engine first (Step 1 above).

**Wrong runtime version:**
```
Error: The module was compiled against a different Node.js version
```
Solution: Rebuild the native module after switching runtimes:
```bash
bun ws build:native
```

**Missing build tools:**
```
gyp ERR! find Python
gyp ERR! stack Error: Could not find any Python installation
```
Solution: Install Python 3 and ensure it's in your PATH.

### Server Crashes on Startup

**Native binding not found:**
```
Error: Native binding not found. Run 'bun run build:native' first.
```
Solution: Build the native module (Step 2 above).

### Rebuild After Changes

| Changed | Rebuild Command |
|---------|-----------------|
| C engine code (`src/`, `include/`) | `cmake --build build -j && bun ws build:native` |
| Native wrapper (`native/src/`) | `bun ws build:native` |
| TypeScript (`apps/websocket/src/`) | Automatic with `bun ws dev` |

## Health Check

```bash
curl http://localhost:3001/health
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3001` | Server port |
| `LOG_LEVEL` | `info` | Logging level (debug, info, warn, error) |
| `NODE_ENV` | - | Set to `development` for colorized logs |
| `DATABASE_URL` | - | PostgreSQL connection string |
| `JWT_SECRET` | - | Secret for JWT token signing |
| `JWT_ISSUER` | - | JWT issuer identifier |

## Project Structure

```
apps/websocket/
├── native/                 # Native Node.js addon (C using N-API)
│   ├── binding.gyp         # node-gyp build configuration
│   └── src/
│       └── addon.c         # Module entry point + world management
├── src/
│   ├── server.ts           # Main entry point
│   ├── engine/             # Engine wrapper (TypeScript)
│   │   ├── EngineBinding.ts    # Native module loader
│   │   ├── WorldManager.ts     # Multi-game routing
│   │   ├── gameActionHandler.ts # Action processing
│   │   ├── snapshotGenerator.ts # Game state snapshots
│   │   ├── logProcessor.ts     # Log redaction
│   │   └── types/              # Engine type definitions
│   ├── handlers/           # WebSocket message handlers
│   ├── services/           # WebSocketService
│   ├── state/              # Room/connection registries
│   ├── utils/              # Helper utilities
│   ├── logger/             # Winston logging setup
│   └── constants/          # Configuration constants
└── dist/                   # Compiled output
```

## Import Conventions

Use absolute path aliases (no relative imports):

```typescript
// Internal imports use @/ - import directly from the file, not barrel exports
import { UserData } from "@/constants";
import logger from "@/logger";
import { WebSocketService } from "@/services/WebSocketService";

// Engine imports
import { createGameWorld } from "@/engine/WorldManager";
import { generateSnapshot } from "@/engine/snapshotGenerator";

// Shared package imports use @tcg/backend-core/*
import { RoomStatus } from "@tcg/backend-core/types";
```

## Dependencies

- **uWebSockets.js** - High-performance WebSocket server
- **N-API** - Node.js native API for addons (C)
- **winston** - Logging library
- **@tcg/backend-core** - Shared types, database, and services
