# WebSocket Server Integration Tests

This directory contains integration tests for the Azuki TCG WebSocket server, focusing on the end-to-end flow from user registration through game simulation.

## Overview

The test framework provides:
- **HTTP + WebSocket testing** via `TestClient` utility
- **Database isolation** with per-suite reset
- **Engine integration testing** for WebSocket ↔ C Engine layer
- **CLI test runner** for rapid manual iteration

## Quick Start

### Prerequisites

1. **Start the test database:**
   ```bash
   # From repository root
   docker compose -f docker-compose.test.yml up db-test -d
   ```

2. **Run migrations on test database:**
   ```bash
   DATABASE_URL=postgres://azuki:azuki@localhost:5433/azuki_test yarn core db:migrate
   ```

3. **Ensure services are running:**
   - Next.js web app on port 3000
   - WebSocket server on port 3001

### Running Tests

```bash
# Run all tests
yarn ws test

# Run tests in watch mode
yarn ws test:watch

# Run tests once (no watch)
yarn ws test:run

# Run CLI integration test
yarn ws test:integration

# Run CLI test with verbose output
yarn ws test:integration --verbose

# Run with test coverage
yarn ws test:coverage
```

## Test Structure

```
tests/
├── integration/
│   ├── auth.test.ts              # Authentication: register, login, logout
│   ├── rooms.test.ts             # Room management: create, join, close
│   ├── decks.test.ts             # Deck operations: get, select
│   ├── websocket.test.ts         # WebSocket: connect, messages, state
│   ├── engine-integration.test.ts # Engine: actions, logs, state sync
│   └── game-flow.test.ts         # Full end-to-end game flow
├── utils/
│   ├── TestClient.ts             # HTTP + WebSocket client wrapper
│   ├── fixtures.ts               # Test data: users, decks, actions
│   ├── helpers.ts                # Common utilities
│   └── database.ts               # Database reset/cleanup
├── setup.ts                      # Global test setup
└── README.md                     # This file
```

## TestClient Usage

The `TestClient` class provides a unified interface for HTTP requests and WebSocket operations:

```typescript
import { TestClient } from "@test/utils/TestClient";

const client = new TestClient();

// Authentication
await client.register("username", "email@test.com", "password");
await client.login("email@test.com", "password");

// Room operations
const { room } = await client.createRoom();
const { joinToken } = await client.joinRoom(room.id);

// WebSocket
await client.connectWebSocket(joinToken);
const ack = await client.waitForMessage("CONNECTION_ACK");

await client.sendMessage("SELECT_DECK", { deckId: "..." });
const state = await client.waitForMessage("ROOM_STATE");

client.disconnect();
```

## Test Database

Tests use a dedicated PostgreSQL database (`azuki_test`) running on port 5433 to avoid interfering with development data.

### Docker Compose Configuration

```yaml
# docker-compose.test.yml
services:
  db-test:
    image: postgres:16-alpine
    ports:
      - "5433:5432"
    environment:
      POSTGRES_DB: azuki_test
```

### Database Reset

Tests call `resetTestDatabase()` in `beforeEach` hooks to ensure clean state:

```typescript
beforeEach(async () => {
  await resetTestDatabase();
});
```

## Environment Variables

Create `apps/websocket/.env.test` or set these variables:

```env
NODE_ENV=test
DATABASE_URL=postgres://azuki:azuki@localhost:5433/azuki_test
API_URL=http://localhost:3000
WS_URL=ws://localhost:3001
JWT_SECRET=test-jwt-secret-for-testing-purposes-only-32chars
JWT_ISSUER=azuki-tcg-test
LOG_LEVEL=warn
ENGINE_DEBUG=true
```

## Writing New Tests

### Integration Test Template

```typescript
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { TestClient } from "@test/utils/TestClient";
import { generateUniqueTestUser } from "@test/utils/fixtures";
import { resetTestDatabase } from "@test/utils/database";

describe("Feature Name", () => {
  let client: TestClient;

  beforeEach(async () => {
    await resetTestDatabase();
    client = new TestClient();
  });

  afterEach(() => {
    client.disconnect();
  });

  it("should do something", async () => {
    const user = generateUniqueTestUser();
    await client.register(user.username, user.email, user.password);

    // Test logic...
    expect(result).toBe(expected);
  });
});
```

### Testing WebSocket Messages

```typescript
// Wait for specific message type
const roomState = await client.waitForMessage("ROOM_STATE", 5000);

// Send message
await client.sendMessage("READY", { ready: true });

// Get all queued messages without waiting
const messages = client.getQueuedMessages();

// Clear message queue
client.clearMessageQueue();
```

### Testing Game Actions

```typescript
import { extractValidActionFromMask } from "@test/utils/helpers";

// Get action mask from game message
const snapshot = await client.waitForMessage("GAME_SNAPSHOT");
const actionMask = snapshot.actionMask;

// Extract valid action from mask
const validAction = extractValidActionFromMask(actionMask);

// Send action
await client.sendMessage("GAME_ACTION", { action: validAction });
```

## Debugging

### Verbose Logging

```bash
# CLI runner with verbose output
yarn ws test:integration --verbose
```

### Vitest Debug Mode

```bash
# Run specific test file
yarn ws test auth.test.ts

# Run tests matching pattern
yarn ws test -t "should register"
```

### Common Issues

1. **Connection refused**: Ensure web app (3000) and WebSocket server (3001) are running
2. **Test database errors**: Run migrations on test database
3. **Timeout errors**: Increase timeout in `waitForMessage()` calls
4. **Cookie issues**: Clear cookies between tests with `client.clearCookies()`

## CI/CD Integration

The tests are designed to run in CI environments. Example GitHub Actions workflow:

```yaml
name: Integration Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: azuki
          POSTGRES_USER: azuki
          POSTGRES_DB: azuki_test
        ports:
          - 5433:5432
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - run: yarn install
      - run: yarn core build
      - run: yarn core db:migrate
        env:
          DATABASE_URL: postgres://azuki:azuki@localhost:5433/azuki_test
      - run: yarn ws test:run
        env:
          DATABASE_URL: postgres://azuki:azuki@localhost:5433/azuki_test
          JWT_SECRET: test-secret-for-ci
```
