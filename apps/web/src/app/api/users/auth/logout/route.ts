import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { clearAuthCookies } from "@/lib/cookies";
import { revokeAllUserTokens } from "@tcg/backend-core/services/authService";

async function handler(request: AuthenticatedRequest): Promise<NextResponse> {
  await revokeAllUserTokens(request.user.id);
  await clearAuthCookies();

  return NextResponse.json({ success: true });
}

export const POST = withErrorHandler(withAuth(handler));
