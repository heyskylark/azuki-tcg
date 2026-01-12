import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { findActiveRoomForUser } from "@tcg/backend-core/services/roomService";

async function handler(request: AuthenticatedRequest): Promise<NextResponse> {
  const room = await findActiveRoomForUser(request.user.id);

  if (!room) {
    return NextResponse.json({ room: null });
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

export const GET = withErrorHandler(withAuth(handler));
