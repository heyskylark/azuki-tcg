import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { emptyHumanEvaluationBodySchema } from "@/lib/validation/humanEvaluations";
import { claimNextHumanEvaluationMatch } from "@tcg/backend-core/services/humanEvaluationService";

interface RouteContext {
  params: Promise<{ sessionId: string }>;
}

async function postHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  emptyHumanEvaluationBodySchema.parse(await request.json());
  const { sessionId } = await context.params;
  const match = await claimNextHumanEvaluationMatch(sessionId, request.user.id);
  return NextResponse.json({ match });
}

export const POST = withErrorHandler(withAuth<RouteContext>(postHandler));
