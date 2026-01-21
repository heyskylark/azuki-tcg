import WebSocket from "ws";

/**
 * API response types
 */
interface UserResponse {
  id: string;
  username: string;
  displayName: string;
  email: string;
}

interface AuthResponse {
  user: UserResponse;
}

interface RoomResponse {
  id: string;
  status: string;
  type: string;
  hasPassword: boolean;
  player0Id: string | null;
  player1Id: string | null;
  createdAt: string;
}

interface CreateRoomResponse {
  room: RoomResponse;
}

interface JoinRoomResponse {
  joinToken: string;
  playerSlot: 0 | 1;
  isNewJoin: boolean;
}

interface DeckResponse {
  id: string;
  name: string;
  leaderCardId: string;
  gateCardId: string;
  status: string;
  createdAt: string;
  updatedAt: string;
}

interface GetDecksResponse {
  decks: DeckResponse[];
}

interface WebSocketMessage {
  type: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

/**
 * TestClient provides a unified HTTP + WebSocket client for integration testing.
 * Handles cookie management for authentication and WebSocket message queuing.
 */
export class TestClient {
  private apiUrl: string;
  private wsUrl: string;
  private cookies: Map<string, string> = new Map();
  private ws: WebSocket | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private messageWaiters: Array<{
    type: string | null;
    resolve: (message: WebSocketMessage) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }> = [];

  constructor(apiUrl?: string, wsUrl?: string) {
    this.apiUrl = apiUrl ?? process.env.API_URL ?? "http://localhost:3000";
    this.wsUrl = wsUrl ?? process.env.WS_URL ?? "ws://localhost:3001";
  }

  // ============================================
  // HTTP Request Helper
  // ============================================

  private async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    const url = `${this.apiUrl}${path}`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    // Include cookies in request
    if (this.cookies.size > 0) {
      headers["Cookie"] = Array.from(this.cookies.entries())
        .map(([name, value]) => `${name}=${value}`)
        .join("; ");
    }

    const response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    // Parse Set-Cookie headers
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
      const error = new Error(
        `HTTP ${response.status}: ${errorBody.message ?? response.statusText}`
      ) as Error & { status: number; body: unknown };
      error.status = response.status;
      error.body = errorBody;
      throw error;
    }

    return response.json() as Promise<T>;
  }

  // ============================================
  // Authentication
  // ============================================

  /**
   * Register a new user.
   */
  async register(
    username: string,
    email: string,
    password: string
  ): Promise<AuthResponse> {
    return this.request<AuthResponse>("POST", "/api/users/auth/sign-up", {
      username,
      email,
      password,
    });
  }

  /**
   * Login with email and password.
   */
  async login(email: string, password: string): Promise<AuthResponse> {
    return this.request<AuthResponse>("POST", "/api/users/auth/login", {
      email,
      password,
    });
  }

  /**
   * Logout the current user.
   */
  async logout(): Promise<void> {
    await this.request<void>("POST", "/api/users/auth/logout", {});
    this.cookies.clear();
  }

  /**
   * Refresh authentication tokens.
   */
  async refreshTokens(): Promise<AuthResponse> {
    return this.request<AuthResponse>("POST", "/api/users/auth/refresh", {});
  }

  // ============================================
  // Rooms
  // ============================================

  /**
   * Create a new room.
   */
  async createRoom(password?: string): Promise<CreateRoomResponse> {
    return this.request<CreateRoomResponse>("POST", "/api/rooms", { password });
  }

  /**
   * Join a room.
   */
  async joinRoom(roomId: string, password?: string): Promise<JoinRoomResponse> {
    return this.request<JoinRoomResponse>(
      "POST",
      `/api/rooms/${roomId}/join`,
      { password }
    );
  }

  /**
   * Get room details.
   */
  async getRoom(roomId: string): Promise<{ room: RoomResponse }> {
    return this.request<{ room: RoomResponse }>("GET", `/api/rooms/${roomId}`);
  }

  // ============================================
  // Decks
  // ============================================

  /**
   * Get user's decks (including starter decks).
   */
  async getDecks(): Promise<GetDecksResponse> {
    return this.request<GetDecksResponse>("GET", "/api/decks");
  }

  /**
   * Get starter decks (convenience method).
   */
  async getStarterDecks(): Promise<DeckResponse[]> {
    const response = await this.getDecks();
    // Starter decks typically have specific naming
    return response.decks.filter(
      (deck) =>
        deck.name.includes("Starter") ||
        deck.name.includes("STT") ||
        deck.status === "COMPLETE"
    );
  }

  // ============================================
  // WebSocket
  // ============================================

  /**
   * Connect to WebSocket server with join token.
   */
  async connectWebSocket(joinToken: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `${this.wsUrl}?token=${encodeURIComponent(joinToken)}`;

      this.ws = new WebSocket(url);

      this.ws.on("open", () => {
        resolve();
      });

      this.ws.on("message", (data: WebSocket.Data) => {
        try {
          const message = JSON.parse(data.toString()) as WebSocketMessage;
          this.handleIncomingMessage(message);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      });

      this.ws.on("error", (error: Error) => {
        reject(error);
      });

      this.ws.on("close", (code: number, reason: Buffer) => {
        // Reject any pending waiters
        for (const waiter of this.messageWaiters) {
          clearTimeout(waiter.timeout);
          waiter.reject(
            new Error(
              `WebSocket closed (code: ${code}, reason: ${reason.toString()})`
            )
          );
        }
        this.messageWaiters = [];
      });
    });
  }

  private handleIncomingMessage(message: WebSocketMessage): void {
    // Check if any waiter is waiting for this message type
    const waiterIndex = this.messageWaiters.findIndex(
      (w) => w.type === null || w.type === message.type
    );

    if (waiterIndex !== -1) {
      const waiter = this.messageWaiters[waiterIndex];
      if (waiter) {
        this.messageWaiters.splice(waiterIndex, 1);
        clearTimeout(waiter.timeout);
        waiter.resolve(message);
      }
    } else {
      // Queue the message for later retrieval
      this.messageQueue.push(message);
    }
  }

  /**
   * Send a message over WebSocket.
   */
  async sendMessage(type: string, payload: Record<string, unknown>): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket is not connected");
    }

    const message = { type, ...payload };
    this.ws.send(JSON.stringify(message));
  }

  /**
   * Wait for a specific message type or any message.
   */
  async waitForMessage(
    type?: string,
    timeoutMs: number = 10000
  ): Promise<WebSocketMessage> {
    // Check if message is already in queue
    const queueIndex = this.messageQueue.findIndex(
      (m) => !type || m.type === type
    );

    if (queueIndex !== -1) {
      const message = this.messageQueue[queueIndex];
      if (message) {
        this.messageQueue.splice(queueIndex, 1);
        return message;
      }
    }

    // Wait for message
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        const waiterIndex = this.messageWaiters.findIndex(
          (w) => w.resolve === resolve
        );
        if (waiterIndex !== -1) {
          this.messageWaiters.splice(waiterIndex, 1);
        }
        reject(
          new Error(
            `Timeout waiting for message${type ? ` of type "${type}"` : ""}`
          )
        );
      }, timeoutMs);

      this.messageWaiters.push({
        type: type ?? null,
        resolve,
        reject,
        timeout,
      });
    });
  }

  /**
   * Get all queued messages without waiting.
   */
  getQueuedMessages(): WebSocketMessage[] {
    const messages = [...this.messageQueue];
    this.messageQueue = [];
    return messages;
  }

  /**
   * Clear the message queue.
   */
  clearMessageQueue(): void {
    this.messageQueue = [];
  }

  /**
   * Check if WebSocket is connected.
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Disconnect WebSocket.
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.messageQueue = [];
    // Clear all waiters
    for (const waiter of this.messageWaiters) {
      clearTimeout(waiter.timeout);
      waiter.reject(new Error("Client disconnected"));
    }
    this.messageWaiters = [];
  }

  // ============================================
  // State Management
  // ============================================

  /**
   * Get current cookies.
   */
  getCookies(): Map<string, string> {
    return new Map(this.cookies);
  }

  /**
   * Set a cookie manually.
   */
  setCookie(name: string, value: string): void {
    this.cookies.set(name, value);
  }

  /**
   * Clear all cookies.
   */
  clearCookies(): void {
    this.cookies.clear();
  }

  /**
   * Get access token from cookies.
   */
  getAccessToken(): string | undefined {
    return this.cookies.get("access_token");
  }

  /**
   * Get refresh token from cookies.
   */
  getRefreshToken(): string | undefined {
    return this.cookies.get("refresh_token");
  }
}
