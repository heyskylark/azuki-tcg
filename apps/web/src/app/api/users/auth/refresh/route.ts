import { NextRequest, NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { setAuthCookies, getRefreshToken } from "@/lib/cookies";
import { env } from "@/lib/env";
import { getUserWithEmail } from "@tcg/backend-core/services/userService";
import { refreshTokens } from "@tcg/backend-core/services/authService";
import {
  UnauthorizedError,
  UserBannedError,
  UserDeletedError,
} from "@tcg/backend-core/errors";
import { UserStatus } from "@tcg/backend-core/types";
import type { AuthConfig } from "@tcg/backend-core/types/auth";
import { verifyRefreshToken } from "@tcg/backend-core/services/authService";

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

async function handler(_request: NextRequest): Promise<NextResponse> {
  const refreshToken = await getRefreshToken();

  if (!refreshToken) {
    throw new UnauthorizedError("No refresh token provided");
  }

  const payload = await verifyRefreshToken(refreshToken, authConfig);

  const user = await getUserWithEmail(payload.sub);
  if (!user) {
    throw new UnauthorizedError("User not found");
  }

  if (user.status === UserStatus.BANNED) {
    throw new UserBannedError();
  }
  if (user.status === UserStatus.DELETED) {
    throw new UserDeletedError();
  }

  const tokens = await refreshTokens(
    refreshToken,
    {
      id: user.id,
      username: user.username,
      email: user.email,
      status: user.status,
    },
    authConfig
  );

  await setAuthCookies(tokens);

  return NextResponse.json({
    user: {
      id: user.id,
      username: user.username,
      email: user.email,
    },
  });
}

export const POST = withErrorHandler(handler);
