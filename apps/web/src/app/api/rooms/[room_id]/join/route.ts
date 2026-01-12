import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { env } from "@/lib/env";
import { joinRoomSchema } from "@/lib/validation/rooms";
import { joinRoom } from "@tcg/backend-core/services/roomService";
import type { AuthConfig } from "@tcg/backend-core/types/auth";

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

interface RouteContext {
  params: Promise<{ room_id: string }>;
}

async function postHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { room_id: roomId } = await context.params;
  const body = await request.json();
  const { password } = joinRoomSchema.parse(body);

  const result = await joinRoom(roomId, request.user.id, password, authConfig);

  return NextResponse.json({
    joinToken: result.joinToken,
    playerSlot: result.playerSlot,
    isNewJoin: result.isNewJoin,
  });
}

export const POST = withErrorHandler(withAuth<RouteContext>(postHandler));
