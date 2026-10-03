import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { emptyHumanEvaluationBodySchema } from "@/lib/validation/humanEvaluations";
import { revealHumanEvaluationSession } from "@tcg/backend-core/services/humanEvaluationService";

interface RouteContext {
  params: Promise<{ sessionId: string }>;
}

async function postHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  emptyHumanEvaluationBodySchema.parse(await request.json());
  const { sessionId } = await context.params;
  const session = await revealHumanEvaluationSession(sessionId, request.user.id);
  return NextResponse.json({ session });
}

export const POST = withErrorHandler(withAuth<RouteContext>(postHandler));
