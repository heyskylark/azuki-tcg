import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { getDeckWithCards } from "@tcg/backend-core/services/DeckService";
import { DeckNotFoundError } from "@tcg/backend-core/errors";

interface RouteContext {
  params: Promise<{ deckId: string }>;
}

async function handler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { deckId } = await context.params;

  const deck = await getDeckWithCards(deckId);

  if (!deck) {
    throw new DeckNotFoundError();
  }

  return NextResponse.json({ deck });
}

export const GET = withErrorHandler(withAuth<RouteContext>(handler));
