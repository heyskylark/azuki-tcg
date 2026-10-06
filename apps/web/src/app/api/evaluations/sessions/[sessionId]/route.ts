import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { getHumanEvaluationSession } from "@tcg/backend-core/services/humanEvaluationService";

interface RouteContext {
  params: Promise<{ sessionId: string }>;
}

async function getHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { sessionId } = await context.params;
  const session = await getHumanEvaluationSession(sessionId, request.user.id);
  return NextResponse.json({ session });
}

export const GET = withErrorHandler(withAuth<RouteContext>(getHandler));
