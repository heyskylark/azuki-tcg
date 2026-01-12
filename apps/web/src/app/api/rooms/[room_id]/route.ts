import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { env } from "@/lib/env";
import { updateRoomSchema } from "@/lib/validation/rooms";
import {
  closeRoom,
  updateRoom,
  findRoomById,
} from "@tcg/backend-core/services/roomService";
import { RoomNotFoundError } from "@tcg/backend-core/errors";
import type { AuthConfig } from "@tcg/backend-core/types/auth";

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

interface RouteContext {
  params: Promise<{ room_id: string }>;
}

async function deleteHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { room_id: roomId } = await context.params;

  const room = await closeRoom(roomId, request.user.id);

  return NextResponse.json({
    room: {
      id: room.id,
      status: room.status,
    },
  });
}

async function patchHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { room_id: roomId } = await context.params;
  const body = await request.json();
  const updates = updateRoomSchema.parse(body);

  const room = await updateRoom(roomId, request.user.id, updates, authConfig);

  return NextResponse.json({
    room: {
      id: room.id,
      status: room.status,
      type: room.type,
      hasPassword: room.passwordHash !== null,
      player0Id: room.player0Id,
      player1Id: room.player1Id,
      updatedAt: room.updatedAt,
    },
  });
}

async function getHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { room_id: roomId } = await context.params;

  const room = await findRoomById(roomId);

  if (!room) {
    throw new RoomNotFoundError();
  }

  return NextResponse.json({
    room: {
      id: room.id,
      status: room.status,
      type: room.type,
      hasPassword: room.passwordHash !== null,
      player0Id: room.player0Id,
      player1Id: room.player1Id,
      createdAt: room.createdAt,
      updatedAt: room.updatedAt,
    },
  });
}

export const GET = withErrorHandler(withAuth<RouteContext>(getHandler));
export const DELETE = withErrorHandler(withAuth<RouteContext>(deleteHandler));
export const PATCH = withErrorHandler(withAuth<RouteContext>(patchHandler));
