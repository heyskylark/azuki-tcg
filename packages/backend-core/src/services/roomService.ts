import { and, eq, notInArray, or } from "drizzle-orm";
import { SignJWT } from "jose";
import { uuidv7 } from "uuidv7";
import * as bcrypt from "bcryptjs";
import db, { type IDatabase, type ITransaction } from "@core/database";
import { Rooms } from "@core/drizzle/schemas/rooms";
import { Decks } from "@core/drizzle/schemas/decks";
import { JwtTokens } from "@core/drizzle/schemas/jwt_tokens";
import { RoomStatus, RoomType } from "@core/types";
import { TokenType, type AuthConfig } from "@core/types/auth";
import { JOIN_TOKEN_EXPIRY_SECONDS } from "@core/constants/auth";
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
}

export interface CreateRoomResult {
  room: RoomData;
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

  const room = await database
    .insert(Rooms)
    .values({
      status: RoomStatus.WAITING_FOR_PLAYERS,
      type: params.type ?? RoomType.PRIVATE,
      passwordHash,
      player0Id: params.creatorId,
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
