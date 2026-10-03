import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { humanEvaluationAnnotationSchema } from "@/lib/validation/humanEvaluations";
import { annotateHumanEvaluationMatch } from "@tcg/backend-core/services/humanEvaluationService";

interface RouteContext {
  params: Promise<{ matchId: string }>;
}

async function postHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const input = humanEvaluationAnnotationSchema.parse(await request.json());
  const { matchId } = await context.params;
  const annotation = await annotateHumanEvaluationMatch(matchId, request.user.id, input);
  return NextResponse.json({ annotation }, { status: 201 });
}

export const POST = withErrorHandler(withAuth<RouteContext>(postHandler));
