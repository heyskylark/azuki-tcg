import { type IDatabase, type ITransaction } from "@/database";
import { Rooms } from "@/drizzle/schemas/rooms";
import { RoomStatus, RoomType } from "@/types";
import { type AuthConfig } from "@/types/auth";
type Database = IDatabase | ITransaction;
export type RoomData = typeof Rooms.$inferSelect;
export interface CreateRoomParams {
    creatorId: string;
    password?: string;
    type?: RoomType;
}
export interface CreateRoomResult {
    room: RoomData;
    joinToken: string;
}
export declare function createRoom(params: CreateRoomParams, config: AuthConfig, database?: Database): Promise<CreateRoomResult>;
export declare function findRoomById(roomId: string, database?: Database): Promise<RoomData | null>;
export declare function findActiveRoomForUser(userId: string, database?: Database): Promise<RoomData | null>;
export declare function closeRoom(roomId: string, userId: string, database?: Database): Promise<RoomData>;
export interface UpdateRoomParams {
    password?: string | null;
}
export declare function updateRoom(roomId: string, userId: string, params: UpdateRoomParams, config: AuthConfig, database?: Database): Promise<RoomData>;
export interface JoinRoomResult {
    joinToken: string;
    playerSlot: 0 | 1;
    isNewJoin: boolean;
}
export declare function joinRoom(roomId: string, userId: string, password: string | undefined, config: AuthConfig, database?: Database): Promise<JoinRoomResult>;
export declare function updateRoomStatus(roomId: string, status: RoomStatus, additionalUpdates?: Partial<typeof Rooms.$inferInsert>, database?: Database): Promise<RoomData>;
export declare function updatePlayerDeck(roomId: string, playerSlot: 0 | 1, deckId: string, database?: Database): Promise<RoomData>;
export declare function updatePlayerReady(roomId: string, playerSlot: 0 | 1, ready: boolean, database?: Database): Promise<RoomData>;
export declare function verifyDeckOwnership(userId: string, deckId: string, database?: Database): Promise<boolean>;
export {};
//# sourceMappingURL=roomService.d.ts.map