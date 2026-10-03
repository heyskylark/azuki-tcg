import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { getHumanEvaluationMatch } from "@tcg/backend-core/services/humanEvaluationService";

interface RouteContext {
  params: Promise<{ matchId: string }>;
}

async function getHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { matchId } = await context.params;
  const match = await getHumanEvaluationMatch(matchId, request.user.id);
  return NextResponse.json({ match });
}

export const GET = withErrorHandler(withAuth<RouteContext>(getHandler));
