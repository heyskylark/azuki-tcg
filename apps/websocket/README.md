# @azuki/websocket

WebSocket server for Azuki TCG real-time game communication using uWebSockets.js.

## Requirements

- **Node.js 20 LTS** (uWebSockets.js doesn't support Node 24+)
- Yarn 1.x

```bash
# Check Node version
node --version  # Should be v20.x.x

# Switch to Node 20 using nvm
nvm use 20
```

## Installation

From the repository root:

```bash
yarn install
```

Or install this package specifically:

```bash
yarn workspace @azuki/websocket install
```

## Development

```bash
# From repository root
yarn ws dev

# Or from this directory
yarn dev
```

The server starts on `http://localhost:3001` by default.

## Production

```bash
# Build
yarn ws build

# Start
yarn ws start
```

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

## Project Structure

```
src/
├── server.ts       # Main entry point
├── clients/        # External client connections (Redis, C engine IPC)
├── services/       # WebSocketService, message handlers
├── utils/          # Helper utilities
├── logger/         # Winston logging setup
├── errors/         # Custom error classes
└── constants/      # Configuration constants
```

## Import Conventions

Use absolute path aliases (no relative imports):

```typescript
// Internal imports use @/ - import directly from the file, not barrel exports
import { UserData } from "@/constants";
import logger from "@/logger";
import { WebSocketService } from "@/services/WebSocketService";

// Shared package imports use @tcg/backend-core/*
import { RoomStatus } from "@tcg/backend-core/types";
```

## Dependencies

- **uWebSockets.js** - High-performance WebSocket server
- **winston** - Logging library
- **@azuki/shared** - Shared types and constants
