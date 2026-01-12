import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { getUserDecks } from "@tcg/backend-core/services/DeckService";

async function handler(request: AuthenticatedRequest): Promise<NextResponse> {
  const decks = await getUserDecks(request.user.id);

  return NextResponse.json({ decks });
}

export const GET = withErrorHandler(withAuth(handler));
