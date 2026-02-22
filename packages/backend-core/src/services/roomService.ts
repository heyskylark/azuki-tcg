import { createHash } from "node:crypto";
import { and, eq, ne, notInArray, or } from "drizzle-orm";
import { SignJWT } from "jose";
import { uuidv7 } from "uuidv7";
import * as bcrypt from "bcryptjs";
import db, { type IDatabase, type ITransaction } from "@core/database";
import { Rooms } from "@core/drizzle/schemas/rooms";
import { Decks } from "@core/drizzle/schemas/decks";
import { Users } from "@core/drizzle/schemas/users";
import { JwtTokens } from "@core/drizzle/schemas/jwt_tokens";
import { RoomStatus, RoomType, DeckStatus, UserStatus, UserType } from "@core/types";
import { TokenType, type AuthConfig } from "@core/types/auth";
import { JOIN_TOKEN_EXPIRY_SECONDS } from "@core/constants/auth";
import { addStarterDecks } from "@core/services/DeckService";
import {
  RoomNotFoundError,
  NotRoomOwnerError,
  InvalidRoomStatusError,
  RoomFullError,
  InvalidRoomPasswordError,
  UserAlreadyInRoomError,
  RoomClosedError,
} from "@core/errors";

type Database = IDatabase | ITransaction;

export type RoomData = typeof Rooms.$inferSelect;

export interface CreateRoomParams {
  creatorId: string;
  password?: string;
  type?: RoomType;
  aiModelKey?: string;
}

export interface CreateRoomResult {
  room: RoomData;
}

function buildAiUsername(modelKey: string): string {
  const normalizedKey = modelKey.trim();
  const hash = createHash("sha256").update(normalizedKey).digest("hex");
  const sanitizedLabel = normalizedKey
    .split("/")
    .at(-1)
    ?.replace(/[^a-zA-Z0-9_]/g, "")
    .toLowerCase()
    .slice(0, 24) ?? "model";

  const label = sanitizedLabel.length > 0 ? sanitizedLabel : "model";
  return `ai_${label}_${hash.slice(0, 10)}`;
}

async function getOrCreateAiUserForModel(
  modelKey: string,
  database: Database
): Promise<{ id: string }> {
  const normalizedModelKey = modelKey.trim();
  if (normalizedModelKey.length === 0) {
    throw new Error("aiModelKey must not be empty");
  }

  const existing = await database
    .select({ id: Users.id })
    .from(Users)
    .where(and(eq(Users.type, UserType.AI), eq(Users.modelKey, normalizedModelKey)))
    .limit(1)
    .then((results) => results[0]);

  if (existing) {
    return { id: existing.id };
  }

  const username = buildAiUsername(normalizedModelKey);

  try {
    const created = await database
      .insert(Users)
      .values({
        username,
        displayName: "AI Opponent",
        passwordHash: uuidv7(),
        type: UserType.AI,
        status: UserStatus.ACTIVE,
        modelKey: normalizedModelKey,
      })
      .returning({ id: Users.id })
      .then((results) => results[0]);

    if (created) {
      return { id: created.id };
    }
  } catch {
    // Fall through to lookup path (handles races on deterministic username).
  }

  const resolved = await database
    .select({ id: Users.id })
    .from(Users)
    .where(and(eq(Users.type, UserType.AI), eq(Users.modelKey, normalizedModelKey)))
    .limit(1)
    .then((results) => results[0]);

  if (!resolved) {
    throw new Error("Failed to create AI user");
  }

  return { id: resolved.id };
}

async function getOrCreateAiDeckId(
  userId: string,
  database: Database
): Promise<string> {
  const existingDeck = await database
    .select({ id: Decks.id })
    .from(Decks)
    .where(and(eq(Decks.userId, userId), ne(Decks.status, DeckStatus.DELETED)))
    .orderBy(Decks.createdAt)
    .limit(1)
    .then((results) => results[0]);

  if (existingDeck) {
    return existingDeck.id;
  }

  await addStarterDecks(userId, database);

  const starterDeck = await database
    .select({ id: Decks.id })
    .from(Decks)
    .where(and(eq(Decks.userId, userId), ne(Decks.status, DeckStatus.DELETED)))
    .orderBy(Decks.createdAt)
    .limit(1)
    .then((results) => results[0]);

  if (!starterDeck) {
    throw new Error("Failed to create AI starter deck");
  }

  return starterDeck.id;
}

export async function createRoom(
  params: CreateRoomParams,
  config: AuthConfig,
  database: Database = db
): Promise<CreateRoomResult> {
  const existingRoom = await findActiveRoomForUser(params.creatorId, database);
  if (existingRoom) {
    throw new UserAlreadyInRoomError();
  }

  const passwordHash = params.password
    ? await bcrypt.hash(params.password, config.saltRounds)
    : null;

  let aiUserId: string | null = null;
  let aiDeckId: string | null = null;
  if (params.aiModelKey) {
    const aiUser = await getOrCreateAiUserForModel(params.aiModelKey, database);
    aiUserId = aiUser.id;
    aiDeckId = await getOrCreateAiDeckId(aiUser.id, database);
  }

  const room = await database
    .insert(Rooms)
    .values({
      status: RoomStatus.WAITING_FOR_PLAYERS,
      type: params.type ?? RoomType.PRIVATE,
      passwordHash,
      player0Id: params.creatorId,
      player1Id: aiUserId,
      player1DeckId: aiDeckId,
    })
    .returning()
    .then((results) => results[0]);

  if (!room) {
    throw new Error("Failed to create room");
  }

  return { room };
}

export async function findRoomById(
  roomId: string,
  database: Database = db
): Promise<RoomData | null> {
  const room = await database
    .select()
    .from(Rooms)
    .where(eq(Rooms.id, roomId))
    .limit(1)
    .then((results) => results[0]);

  return room ?? null;
}

const INACTIVE_ROOM_STATUSES = [RoomStatus.COMPLETED, RoomStatus.ABORTED, RoomStatus.CLOSED];

export async function findActiveRoomForUser(
  userId: string,
  database: Database = db
): Promise<RoomData | null> {
  const room = await database
    .select()
    .from(Rooms)
    .where(
      and(
        or(eq(Rooms.player0Id, userId), eq(Rooms.player1Id, userId)),
        notInArray(Rooms.status, INACTIVE_ROOM_STATUSES)
      )
    )
    .limit(1)
    .then((results) => results[0]);

  return room ?? null;
}

export async function closeRoom(
  roomId: string,
  userId: string,
  database: Database = db
): Promise<RoomData> {
  const room = await findRoomById(roomId, database);

  if (!room) {
    throw new RoomNotFoundError();
  }

  if (room.player0Id !== userId) {
    throw new NotRoomOwnerError();
  }

  const allowedStatuses = [RoomStatus.WAITING_FOR_PLAYERS, RoomStatus.DECK_SELECTION];
  if (!allowedStatuses.includes(room.status)) {
    throw new InvalidRoomStatusError(
      "Room can only be closed in WAITING_FOR_PLAYERS or DECK_SELECTION status"
    );
  }

  const updatedRoom = await database
    .update(Rooms)
    .set({ status: RoomStatus.CLOSED })
    .where(eq(Rooms.id, roomId))
    .returning()
    .then((results) => results[0]);

  if (!updatedRoom) {
    throw new Error("Failed to update room");
  }

  return updatedRoom;
}

export interface UpdateRoomParams {
  password?: string | null;
}

export async function updateRoom(
  roomId: string,
  userId: string,
  params: UpdateRoomParams,
  config: AuthConfig,
  database: Database = db
): Promise<RoomData> {
  const room = await findRoomById(roomId, database);

  if (!room) {
    throw new RoomNotFoundError();
  }

  if (room.player0Id !== userId) {
    throw new NotRoomOwnerError();
  }

  if (room.status !== RoomStatus.WAITING_FOR_PLAYERS) {
    throw new InvalidRoomStatusError("Room can only be updated in WAITING_FOR_PLAYERS status");
  }

  const updates: Partial<typeof Rooms.$inferInsert> = {};

  if (params.password !== undefined) {
    updates.passwordHash = params.password
      ? await bcrypt.hash(params.password, config.saltRounds)
      : null;
  }

  if (Object.keys(updates).length === 0) {
    return room;
  }

  const updatedRoom = await database
    .update(Rooms)
    .set(updates)
    .where(eq(Rooms.id, roomId))
    .returning()
    .then((results) => results[0]);

  if (!updatedRoom) {
    throw new Error("Failed to update room");
  }

  return updatedRoom;
}

export interface JoinRoomResult {
  joinToken: string;
  playerSlot: 0 | 1;
  isNewJoin: boolean;
}

export async function joinRoom(
  roomId: string,
  userId: string,
  password: string | undefined,
  config: AuthConfig,
  database: Database = db
): Promise<JoinRoomResult> {
  const room = await findRoomById(roomId, database);

  if (!room) {
    throw new RoomNotFoundError();
  } else if (INACTIVE_ROOM_STATUSES.includes(room.status)) {
    throw new RoomClosedError();
  }

  if (room.player0Id === userId) {
    const joinToken = await createJoinToken(roomId, userId, 0, config, database);
    return { joinToken, playerSlot: 0, isNewJoin: false };
  }

  if (room.player1Id === userId) {
    const joinToken = await createJoinToken(roomId, userId, 1, config, database);
    return { joinToken, playerSlot: 1, isNewJoin: false };
  }

  // User is not already in this room - check if they're in another active room
  const existingRoom = await findActiveRoomForUser(userId, database);
  if (existingRoom) {
    throw new UserAlreadyInRoomError();
  }

  if (room.status !== RoomStatus.WAITING_FOR_PLAYERS) {
    throw new InvalidRoomStatusError("Room is not accepting new players");
  }

  if (room.passwordHash) {
    if (!password) {
      throw new InvalidRoomPasswordError();
    }
    const isValid = await bcrypt.compare(password, room.passwordHash);
    if (!isValid) {
      throw new InvalidRoomPasswordError();
    }
  }

  if (room.player1Id !== null) {
    throw new RoomFullError();
  }

  await database.update(Rooms).set({ player1Id: userId }).where(eq(Rooms.id, roomId));

  const joinToken = await createJoinToken(roomId, userId, 1, config, database);

  return { joinToken, playerSlot: 1, isNewJoin: true };
}

export async function updateRoomStatus(
  roomId: string,
  status: RoomStatus,
  additionalUpdates?: Partial<typeof Rooms.$inferInsert>,
  database: Database = db
): Promise<RoomData> {
  const updates = {
    status,
    ...additionalUpdates,
  };

  const updatedRoom = await database
    .update(Rooms)
    .set(updates)
    .where(eq(Rooms.id, roomId))
    .returning()
    .then((results) => results[0]);

  if (!updatedRoom) {
    throw new RoomNotFoundError();
  }

  return updatedRoom;
}

export async function updatePlayerDeck(
  roomId: string,
  playerSlot: 0 | 1,
  deckId: string,
  database: Database = db
): Promise<RoomData> {
  const updates = playerSlot === 0 ? { player0DeckId: deckId } : { player1DeckId: deckId };

  const updatedRoom = await database
    .update(Rooms)
    .set(updates)
    .where(eq(Rooms.id, roomId))
    .returning()
    .then((results) => results[0]);

  if (!updatedRoom) {
    throw new RoomNotFoundError();
  }

  return updatedRoom;
}

export async function updatePlayerReady(
  roomId: string,
  playerSlot: 0 | 1,
  ready: boolean,
  database: Database = db
): Promise<RoomData> {
  const updates = playerSlot === 0 ? { player0Ready: ready } : { player1Ready: ready };

  const updatedRoom = await database
    .update(Rooms)
    .set(updates)
    .where(eq(Rooms.id, roomId))
    .returning()
    .then((results) => results[0]);

  if (!updatedRoom) {
    throw new RoomNotFoundError();
  }

  return updatedRoom;
}

export async function removePlayer1FromRoom(
  roomId: string,
  database: Database = db
): Promise<RoomData> {
  const updatedRoom = await database
    .update(Rooms)
    .set({ player1Id: null })
    .where(eq(Rooms.id, roomId))
    .returning()
    .then((results) => results[0]);

  if (!updatedRoom) {
    throw new RoomNotFoundError();
  }

  return updatedRoom;
}

export async function verifyDeckOwnership(
  userId: string,
  deckId: string,
  database: Database = db
): Promise<boolean> {
  const deck = await database
    .select({ userId: Decks.userId })
    .from(Decks)
    .where(eq(Decks.id, deckId))
    .limit(1)
    .then((results) => results[0]);

  if (!deck) {
    return false;
  }

  return deck.userId === userId;
}

async function createJoinToken(
  roomId: string,
  userId: string,
  playerSlot: 0 | 1,
  config: AuthConfig,
  database: Database = db
): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  const jti = uuidv7();
  const secretKey = new TextEncoder().encode(config.jwtSecret);

  const token = await new SignJWT({
    roomId,
    playerSlot,
  })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuer(config.jwtIssuer)
    .setSubject(userId)
    .setAudience(config.jwtIssuer)
    .setExpirationTime(now + JOIN_TOKEN_EXPIRY_SECONDS)
    .setIssuedAt(now)
    .setJti(jti)
    .sign(secretKey);

  await database.insert(JwtTokens).values({
    jti,
    userId,
    tokenType: TokenType.JOIN,
    expiresAt: new Date((now + JOIN_TOKEN_EXPIRY_SECONDS) * 1000),
  });

  return token;
}
