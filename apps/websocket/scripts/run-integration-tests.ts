#!/usr/bin/env tsx
/**
 * CLI Integration Test Runner
 *
 * A standalone script for running integration tests manually.
 * Useful for rapid iteration and debugging without the full test runner.
 *
 * Usage:
 *   yarn ws test:integration
 *   yarn ws test:integration --verbose
 *   yarn ws test:integration --engine-debug
 */

import WebSocket from "ws";

// Test environment configuration
process.env.NODE_ENV = "test";

if (!process.env.JWT_SECRET) {
  process.env.JWT_SECRET = "test-jwt-secret-for-testing-purposes-only-32chars";
}

if (!process.env.JWT_ISSUER) {
  process.env.JWT_ISSUER = "azuki-tcg-test";
}

// Configuration
const API_URL = process.env.API_URL ?? "http://localhost:3000";
const WS_URL = process.env.WS_URL ?? "ws://localhost:3001";
const VERBOSE = process.argv.includes("--verbose");
const ENGINE_DEBUG = process.argv.includes("--engine-debug");

// Test state
let testsPassed = 0;
let testsFailed = 0;

// Logging utilities
function log(message: string, ...args: unknown[]): void {
  console.log(message, ...args);
}

function logVerbose(message: string, ...args: unknown[]): void {
  if (VERBOSE) {
    console.log(`  [VERBOSE] ${message}`, ...args);
  }
}

function logSuccess(message: string): void {
  console.log(`  ✅ ${message}`);
  testsPassed++;
}

function logFailure(message: string, error?: unknown): void {
  console.log(`  ❌ ${message}`);
  if (error) {
    console.error(`     Error:`, error);
  }
  testsFailed++;
}

// Simple test client
class SimpleTestClient {
  private cookies: Map<string, string> = new Map();
  private ws: WebSocket | null = null;
  private messageQueue: Array<Record<string, unknown>> = [];

  async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const url = `${API_URL}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (this.cookies.size > 0) {
      headers["Cookie"] = Array.from(this.cookies.entries())
        .map(([name, value]) => `${name}=${value}`)
        .join("; ");
    }

    logVerbose(`${method} ${path}`);

    const response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    // Parse cookies
    const setCookieHeaders = response.headers.getSetCookie();
    for (const setCookie of setCookieHeaders) {
      const [cookiePart] = setCookie.split(";");
      if (cookiePart) {
        const [name, value] = cookiePart.split("=");
        if (name && value !== undefined) {
          this.cookies.set(name.trim(), value.trim());
        }
      }
    }

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(
        `HTTP ${response.status}: ${(errorBody as { message?: string }).message ?? response.statusText}`
      );
    }

    return response.json() as Promise<T>;
  }

  async connectWebSocket(token: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `${WS_URL}?token=${encodeURIComponent(token)}`;
      this.ws = new WebSocket(url);

      this.ws.on("open", () => {
        logVerbose("WebSocket connected");
        resolve();
      });

      this.ws.on("message", (data: WebSocket.Data) => {
        try {
          const message = JSON.parse(data.toString());
          logVerbose(`WS received: ${message.type}`);
          this.messageQueue.push(message);
        } catch (error) {
          console.error("Failed to parse WS message:", error);
        }
      });

      this.ws.on("error", (error: Error) => {
        reject(error);
      });
    });
  }

  async sendMessage(type: string, payload: Record<string, unknown>): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected");
    }
    const message = { type, ...payload };
    logVerbose(`WS send: ${type}`);
    this.ws.send(JSON.stringify(message));
  }

  async waitForMessage(type?: string, timeoutMs = 5000): Promise<Record<string, unknown>> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const idx = this.messageQueue.findIndex((m) => !type || m.type === type);
      if (idx !== -1) {
        const msg = this.messageQueue[idx];
        if (msg) {
          this.messageQueue.splice(idx, 1);
          return msg;
        }
      }
      await new Promise((r) => setTimeout(r, 100));
    }

    throw new Error(`Timeout waiting for message${type ? ` of type "${type}"` : ""}`);
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Generate unique test user
function generateUser(): { username: string; email: string; password: string } {
  const id = Date.now() + Math.random().toString(36).slice(2);
  return {
    username: `testUser_${id}`,
    email: `test_${id}@test.com`,
    password: "testpass123!",
  };
}

// Test functions
async function testUserRegistration(): Promise<boolean> {
  log("\n📝 Testing User Registration...");
  const client = new SimpleTestClient();

  try {
    const user = generateUser();
    const response = await client.request<{ user: { id: string; username: string } }>(
      "POST",
      "/api/users/auth/sign-up",
      user
    );

    if (response.user?.id && response.user?.username === user.username) {
      logSuccess("User registration successful");
      return true;
    } else {
      logFailure("User registration returned unexpected response");
      return false;
    }
  } catch (error) {
    logFailure("User registration failed", error);
    return false;
  }
}

async function testRoomCreation(): Promise<boolean> {
  log("\n🏠 Testing Room Creation...");
  const client = new SimpleTestClient();

  try {
    const user = generateUser();
    await client.request("POST", "/api/users/auth/sign-up", user);

    const response = await client.request<{ room: { id: string; status: string } }>(
      "POST",
      "/api/rooms",
      {}
    );

    if (response.room?.id && response.room?.status === "WAITING_FOR_PLAYERS") {
      logSuccess("Room creation successful");
      return true;
    } else {
      logFailure("Room creation returned unexpected response");
      return false;
    }
  } catch (error) {
    logFailure("Room creation failed", error);
    return false;
  }
}

async function testWebSocketConnection(): Promise<boolean> {
  log("\n🔌 Testing WebSocket Connection...");
  const client = new SimpleTestClient();

  try {
    const user = generateUser();
    await client.request("POST", "/api/users/auth/sign-up", user);

    const { room } = await client.request<{ room: { id: string } }>("POST", "/api/rooms", {});

    const { joinToken } = await client.request<{ joinToken: string }>(
      "POST",
      `/api/rooms/${room.id}/join`,
      {}
    );

    await client.connectWebSocket(joinToken);

    const ack = await client.waitForMessage("CONNECTION_ACK");
    client.disconnect();

    if (ack.type === "CONNECTION_ACK" && ack.playerSlot !== undefined) {
      logSuccess("WebSocket connection successful");
      return true;
    } else {
      logFailure("WebSocket connection returned unexpected response");
      return false;
    }
  } catch (error) {
    logFailure("WebSocket connection failed", error);
    return false;
  }
}

async function testFullGameFlow(): Promise<boolean> {
  log("\n🎮 Testing Full Game Flow...");
  const client1 = new SimpleTestClient();
  const client2 = new SimpleTestClient();

  try {
    // Register users
    log("  1️⃣ Registering users...");
    const user1 = generateUser();
    const user2 = generateUser();
    await client1.request("POST", "/api/users/auth/sign-up", user1);
    await client2.request("POST", "/api/users/auth/sign-up", user2);
    logSuccess("Users registered");

    // Create room
    log("  2️⃣ Creating room...");
    const { room } = await client1.request<{ room: { id: string } }>("POST", "/api/rooms", {});
    logSuccess(`Room created: ${room.id}`);

    // Join room
    log("  3️⃣ Joining room...");
    const join1 = await client1.request<{ joinToken: string; playerSlot: number }>(
      "POST",
      `/api/rooms/${room.id}/join`,
      {}
    );
    const join2 = await client2.request<{ joinToken: string; playerSlot: number }>(
      "POST",
      `/api/rooms/${room.id}/join`,
      {}
    );
    logSuccess(`Both players joined (slots: ${join1.playerSlot}, ${join2.playerSlot})`);

    // Connect WebSockets
    log("  4️⃣ Connecting WebSockets...");
    await client1.connectWebSocket(join1.joinToken);
    await client2.connectWebSocket(join2.joinToken);

    await client1.waitForMessage("CONNECTION_ACK");
    await client2.waitForMessage("CONNECTION_ACK");

    await client1.waitForMessage("ROOM_STATE");
    await client2.waitForMessage("ROOM_STATE");
    logSuccess("WebSockets connected");

    // Start game
    log("  5️⃣ Starting game...");
    await client1.sendMessage("START_GAME", {});

    const state1 = await client1.waitForMessage("ROOM_STATE");
    await client2.waitForMessage("ROOM_STATE");

    if (state1.status !== "DECK_SELECTION") {
      logFailure(`Expected DECK_SELECTION, got ${state1.status}`);
      return false;
    }
    logSuccess("Game started (DECK_SELECTION)");

    // Get decks
    log("  6️⃣ Getting decks...");
    const decks1 = await client1.request<{ decks: Array<{ id: string }> }>("GET", "/api/decks");
    const decks2 = await client2.request<{ decks: Array<{ id: string }> }>("GET", "/api/decks");

    if (decks1.decks.length === 0 || decks2.decks.length === 0) {
      log("    ⚠️ No starter decks available, skipping deck selection");
      client1.disconnect();
      client2.disconnect();
      logSuccess("Flow completed (partial - no decks)");
      return true;
    }

    // Select decks
    log("  7️⃣ Selecting decks...");
    await client1.sendMessage("SELECT_DECK", { deckId: decks1.decks[0]?.id });
    await client2.sendMessage("SELECT_DECK", { deckId: decks2.decks[0]?.id });

    await client1.waitForMessage("ROOM_STATE");
    await client2.waitForMessage("ROOM_STATE");
    logSuccess("Decks selected");

    // Ready up
    log("  8️⃣ Readying up...");
    await client1.sendMessage("READY", { ready: true });
    await client2.sendMessage("READY", { ready: true });

    // Wait for game to start
    log("  9️⃣ Waiting for game to start...");
    let gameStarted = false;
    const maxWait = 15000;
    const startTime = Date.now();

    while (!gameStarted && Date.now() - startTime < maxWait) {
      try {
        const msg = await client1.waitForMessage(undefined, 1000);
        if (
          (msg.type === "ROOM_STATE" && msg.status === "IN_MATCH") ||
          msg.type === "GAME_SNAPSHOT" ||
          msg.type === "GAME_LOG_BATCH"
        ) {
          gameStarted = true;
          logSuccess(`Game started (received ${msg.type})`);
        }
      } catch {
        // Continue
      }
    }

    if (!gameStarted) {
      logFailure("Game did not start within timeout");
    }

    // Cleanup
    client1.disconnect();
    client2.disconnect();

    return gameStarted;
  } catch (error) {
    logFailure("Full game flow failed", error);
    client1.disconnect();
    client2.disconnect();
    return false;
  }
}

// Main runner
async function main(): Promise<void> {
  console.log("🚀 Starting Integration Tests\n");
  console.log(`   API URL: ${API_URL}`);
  console.log(`   WS URL:  ${WS_URL}`);
  console.log(`   Verbose: ${VERBOSE}`);
  console.log(`   Engine Debug: ${ENGINE_DEBUG}`);

  const startTime = Date.now();

  // Run tests
  await testUserRegistration();
  await testRoomCreation();
  await testWebSocketConnection();
  await testFullGameFlow();

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);

  // Summary
  console.log("\n" + "=".repeat(50));
  console.log("📊 Test Summary");
  console.log("=".repeat(50));
  console.log(`   Passed: ${testsPassed}`);
  console.log(`   Failed: ${testsFailed}`);
  console.log(`   Duration: ${duration}s`);
  console.log("=".repeat(50));

  if (testsFailed > 0) {
    console.log("\n❌ Some tests failed!");
    process.exit(1);
  } else {
    console.log("\n✅ All tests passed!");
    process.exit(0);
  }
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
