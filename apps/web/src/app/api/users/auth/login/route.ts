import { NextRequest, NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { setAuthCookies } from "@/lib/cookies";
import { env } from "@/lib/env";
import { loginSchema } from "@/lib/validation/auth";
import { findUserByEmail } from "@tcg/backend-core/services/userService";
import { verifyPassword, createTokens } from "@tcg/backend-core/services/authService";
import {
  InvalidCredentialsError,
  UserBannedError,
  UserDeletedError,
} from "@tcg/backend-core/errors";
import { UserStatus } from "@tcg/backend-core/types";
import type { AuthConfig } from "@tcg/backend-core/types/auth";

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

async function handler(request: NextRequest): Promise<NextResponse> {
  const body = await request.json();
  const { email, password } = loginSchema.parse(body);

  const user = await findUserByEmail(email);
  if (!user) {
    throw new InvalidCredentialsError();
  }

  if (user.status === UserStatus.BANNED) {
    throw new UserBannedError();
  }
  if (user.status === UserStatus.DELETED) {
    throw new UserDeletedError();
  }

  const isValidPassword = await verifyPassword(password, user.passwordHash);
  if (!isValidPassword) {
    throw new InvalidCredentialsError();
  }

  const tokens = await createTokens(
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
