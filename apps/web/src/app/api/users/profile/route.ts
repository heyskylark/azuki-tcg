import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { updateProfileSchema } from "@/lib/validation/profile";
import { updateUserDisplayName } from "@tcg/backend-core/services/userService";

async function patchHandler(
  request: AuthenticatedRequest
): Promise<NextResponse> {
  const body = await request.json();
  const { displayName } = updateProfileSchema.parse(body);

  const result = await updateUserDisplayName(request.user.id, displayName);

  return NextResponse.json({
    user: {
      id: request.user.id,
      displayName: result.displayName,
    },
  });
}

export const PATCH = withErrorHandler(withAuth(patchHandler));
