import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { getHumanEvaluationReview } from "@tcg/backend-core/services/humanEvaluationService";

interface RouteContext {
  params: Promise<{ matchId: string }>;
}

async function getHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { matchId } = await context.params;
  const review = await getHumanEvaluationReview(matchId, request.user.id);
  return NextResponse.json({ review });
}

export const GET = withErrorHandler(withAuth<RouteContext>(getHandler));
