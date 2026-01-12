import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";

async function handler(request: AuthenticatedRequest): Promise<NextResponse> {
  return NextResponse.json({
    user: {
      id: request.user.id,
      username: request.user.username,
      displayName: request.user.displayName,
      email: request.user.email,
    },
  });
}

export const GET = withErrorHandler(withAuth(handler));
