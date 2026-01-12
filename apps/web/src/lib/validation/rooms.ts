import { z } from "zod";

const ROOM_PASSWORD_MIN_LENGTH = 4;
const ROOM_PASSWORD_MAX_LENGTH = 64;

const roomPasswordSchema = z
  .string()
  .min(
    ROOM_PASSWORD_MIN_LENGTH,
    `Password must be at least ${ROOM_PASSWORD_MIN_LENGTH} characters`
  )
  .max(
    ROOM_PASSWORD_MAX_LENGTH,
    `Password must be at most ${ROOM_PASSWORD_MAX_LENGTH} characters`
  );

export const createRoomSchema = z
  .object({
    password: roomPasswordSchema.optional(),
  })
  .strict();

export const updateRoomSchema = z
  .object({
    password: z.union([roomPasswordSchema, z.null()]).optional(),
  })
  .strict();

export const joinRoomSchema = z
  .object({
    password: z.string().optional(),
  })
  .strict();

export type CreateRoomInput = z.infer<typeof createRoomSchema>;
export type UpdateRoomInput = z.infer<typeof updateRoomSchema>;
export type JoinRoomInput = z.infer<typeof joinRoomSchema>;
