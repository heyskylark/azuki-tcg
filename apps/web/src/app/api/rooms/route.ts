import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { env } from "@/lib/env";
import { createRoomSchema } from "@/lib/validation/rooms";
import { createRoom } from "@tcg/backend-core/services/roomService";
import type { AuthConfig } from "@tcg/backend-core/types/auth";

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

async function postHandler(request: AuthenticatedRequest): Promise<NextResponse> {
  const body = await request.json();
  const { password, aiModelKey } = createRoomSchema.parse(body);

  const { room } = await createRoom(
    {
      creatorId: request.user.id,
      password,
      aiModelKey,
    },
    authConfig
  );

  return NextResponse.json(
    {
      room: {
        id: room.id,
        status: room.status,
        type: room.type,
        hasPassword: room.passwordHash !== null,
        player0Id: room.player0Id,
        player1Id: room.player1Id,
        createdAt: room.createdAt,
      },
    },
    { status: 201 }
  );
}

export const POST = withErrorHandler(withAuth(postHandler));
