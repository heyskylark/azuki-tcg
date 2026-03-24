import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { copyDeckSchema } from "@/lib/validation/decks";
import { copyDeck } from "@tcg/backend-core/services/DeckService";

interface RouteContext {
  params: Promise<{ deckId: string }>;
}

async function postHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { deckId } = await context.params;
  const body = await request.json();
  const input = copyDeckSchema.parse(body);

  const deck = await copyDeck(deckId, request.user.id, input.name);

  return NextResponse.json({ deck }, { status: 201 });
}

export const POST = withErrorHandler(withAuth<RouteContext>(postHandler));
