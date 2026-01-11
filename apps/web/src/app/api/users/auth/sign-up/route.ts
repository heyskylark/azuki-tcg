import { NextRequest, NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { setAuthCookies } from "@/lib/cookies";
import { env } from "@/lib/env";
import { signUpSchema } from "@/lib/validation/auth";
import { createUser } from "@tcg/backend-core/services/userService";
import { hashPassword, createTokens } from "@tcg/backend-core/services/authService";
import { UserStatus } from "@tcg/backend-core/types";
import type { AuthConfig } from "@tcg/backend-core/types/auth";

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

async function handler(request: NextRequest): Promise<NextResponse> {
  const body = await request.json();
  const { username, email, password } = signUpSchema.parse(body);

  const passwordHash = await hashPassword(password, authConfig.saltRounds);

  const user = await createUser({
    username,
    email,
    passwordHash,
  });

  const tokens = await createTokens(
    {
      id: user.id,
      username: user.username,
      email: user.email,
      status: UserStatus.ACTIVE,
    },
    authConfig
  );

  await setAuthCookies(tokens);

  return NextResponse.json(
    {
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
      },
    },
    { status: 201 }
  );
}

export const POST = withErrorHandler(handler);
