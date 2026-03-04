import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { listSelectableAiModels } from "@tcg/backend-core/services/aiModelService";

async function getHandler(_request: AuthenticatedRequest): Promise<NextResponse> {
  const models = await listSelectableAiModels();

  return NextResponse.json(
    {
      models: models.map((model) => ({
        id: model.id,
        displayName: model.displayName,
      })),
    },
    { status: 200 }
  );
}

export const GET = withErrorHandler(withAuth(getHandler));
