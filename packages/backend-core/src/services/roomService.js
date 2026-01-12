import { and, eq, notInArray, or } from "drizzle-orm";
import { SignJWT } from "jose";
import { uuidv7 } from "uuidv7";
import * as bcrypt from "bcrypt";
import db from "@/database";
import { Rooms } from "@/drizzle/schemas/rooms";
import { Decks } from "@/drizzle/schemas/decks";
import { JwtTokens } from "@/drizzle/schemas/jwt_tokens";
import { RoomStatus, RoomType } from "@/types";
import { TokenType } from "@/types/auth";
import { JOIN_TOKEN_EXPIRY_SECONDS } from "@/constants/auth";
import { RoomNotFoundError, NotRoomOwnerError, InvalidRoomStatusError, RoomFullError, InvalidRoomPasswordError, UserAlreadyInRoomError, } from "@/errors";
export async function createRoom(params, config, database = db) {
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
    const joinToken = await createJoinToken(room.id, params.creatorId, 0, config, database);
    return { room, joinToken };
}
export async function findRoomById(roomId, database = db) {
    const room = await database
        .select()
        .from(Rooms)
        .where(eq(Rooms.id, roomId))
        .limit(1)
        .then((results) => results[0]);
    return room ?? null;
}
const INACTIVE_ROOM_STATUSES = [
    RoomStatus.COMPLETED,
    RoomStatus.ABORTED,
    RoomStatus.CLOSED,
];
export async function findActiveRoomForUser(userId, database = db) {
    const room = await database
        .select()
        .from(Rooms)
        .where(and(or(eq(Rooms.player0Id, userId), eq(Rooms.player1Id, userId)), notInArray(Rooms.status, INACTIVE_ROOM_STATUSES)))
        .limit(1)
        .then((results) => results[0]);
    return room ?? null;
}
export async function closeRoom(roomId, userId, database = db) {
    const room = await findRoomById(roomId, database);
    if (!room) {
        throw new RoomNotFoundError();
    }
    if (room.player0Id !== userId) {
        throw new NotRoomOwnerError();
    }
    const allowedStatuses = [RoomStatus.WAITING_FOR_PLAYERS, RoomStatus.DECK_SELECTION];
    if (!allowedStatuses.includes(room.status)) {
        throw new InvalidRoomStatusError("Room can only be closed in WAITING_FOR_PLAYERS or DECK_SELECTION status");
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
export async function updateRoom(roomId, userId, params, config, database = db) {
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
    const updates = {};
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
export async function joinRoom(roomId, userId, password, config, database = db) {
    const room = await findRoomById(roomId, database);
    if (!room) {
        throw new RoomNotFoundError();
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
    await database
        .update(Rooms)
        .set({ player1Id: userId })
        .where(eq(Rooms.id, roomId));
    const joinToken = await createJoinToken(roomId, userId, 1, config, database);
    return { joinToken, playerSlot: 1, isNewJoin: true };
}
export async function updateRoomStatus(roomId, status, additionalUpdates, database = db) {
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
export async function updatePlayerDeck(roomId, playerSlot, deckId, database = db) {
    const updates = playerSlot === 0
        ? { player0DeckId: deckId }
        : { player1DeckId: deckId };
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
export async function updatePlayerReady(roomId, playerSlot, ready, database = db) {
    const updates = playerSlot === 0
        ? { player0Ready: ready }
        : { player1Ready: ready };
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
export async function verifyDeckOwnership(userId, deckId, database = db) {
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
async function createJoinToken(roomId, userId, playerSlot, config, database = db) {
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
//# sourceMappingURL=roomService.js.map