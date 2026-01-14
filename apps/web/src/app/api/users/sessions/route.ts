import { NextResponse } from "next/server";
import { verifyAccessToken } from "@tcg/backend-core/services/authService";
import type { AuthConfig } from "@tcg/backend-core/types/auth";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { getAccessToken } from "@/lib/cookies";
import { env } from "@/lib/env";

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

async function handler(): Promise<NextResponse> {
  const accessToken = await getAccessToken();

  if (!accessToken) {
    return NextResponse.json({ valid: false });
  }

  try {
    await verifyAccessToken(accessToken, authConfig);
    return NextResponse.json({ valid: true });
  } catch {
    return NextResponse.json({ valid: false });
  }
}

export const GET = withErrorHandler(handler);
