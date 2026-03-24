import { NextResponse } from "next/server";
import { withErrorHandler } from "@/lib/hof/withErrorHandler";
import { withAuth, type AuthenticatedRequest } from "@/lib/hof/withAuth";
import { createOrUpdateDeckSchema } from "@/lib/validation/decks";
import {
  getDeckWithCards,
  softDeleteDeck,
  updateDeck,
} from "@tcg/backend-core/services/DeckService";
import { DeckNotFoundError } from "@tcg/backend-core/errors";

interface RouteContext {
  params: Promise<{ deckId: string }>;
}

async function getHandler(
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

async function putHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { deckId } = await context.params;
  const body = await request.json();
  const input = createOrUpdateDeckSchema.parse(body);

  const deck = await updateDeck(deckId, request.user.id, input);

  return NextResponse.json({ deck });
}

async function deleteHandler(
  request: AuthenticatedRequest,
  context: RouteContext
): Promise<NextResponse> {
  const { deckId } = await context.params;

  const deck = await softDeleteDeck(deckId, request.user.id);

  return NextResponse.json({ deck });
}

export const GET = withErrorHandler(withAuth<RouteContext>(getHandler));
export const PUT = withErrorHandler(withAuth<RouteContext>(putHandler));
export const DELETE = withErrorHandler(withAuth<RouteContext>(deleteHandler));
