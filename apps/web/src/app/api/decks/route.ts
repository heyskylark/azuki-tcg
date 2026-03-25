import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { createOrUpdateDeckSchema } from "@/lib/validation/decks";
import { createDeck, getUserDecks } from "@tcg/backend-core/services/DeckService";

async function getHandler(request: AuthenticatedRequest): Promise<NextResponse> {
  const decks = await getUserDecks(request.user.id);

  return NextResponse.json({ decks });
}

async function postHandler(request: AuthenticatedRequest): Promise<NextResponse> {
  const body = await request.json();
  const input = createOrUpdateDeckSchema.parse(body);

  const deck = await createDeck(request.user.id, input);

  return NextResponse.json({ deck }, { status: 201 });
}

export const GET = withErrorHandler(withAuth(getHandler));
export const POST = withErrorHandler(withAuth(postHandler));
