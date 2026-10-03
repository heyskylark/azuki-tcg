import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { createHumanEvaluationSessionSchema } from "@/lib/validation/humanEvaluations";
import { createHumanEvaluationSession } from "@tcg/backend-core/services/humanEvaluationService";

async function postHandler(request: AuthenticatedRequest): Promise<NextResponse> {
  const input = createHumanEvaluationSessionSchema.parse(await request.json());
  const session = await createHumanEvaluationSession({ reviewerId: request.user.id, ...input });
  return NextResponse.json({ session }, { status: 201 });
}

export const POST = withErrorHandler(withAuth(postHandler));
