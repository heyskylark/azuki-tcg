import { z } from "zod";
import { Suspense } from "react";
import { notFound, redirect } from "next/navigation";
import { getServerUser } from "@/lib/auth/getServerUser";
import { findRoomById } from "@tcg/backend-core/services/roomService";
import { getDeckBuilderCards } from "@tcg/backend-core/services/DeckService";
import { getHumanEvaluationMatchByRoomId } from "@tcg/backend-core/services/humanEvaluationService";
import { cardCodeToDefId } from "@tcg/backend-core/services/cardMapperService";
import { RoomClient, type RoomData } from "./RoomClient";
import { RoomSkeleton } from "./RoomSkeleton";
import type { RoomEvaluationMetadata } from "@tcg/backend-core/types/ws";
import type { DeckCard } from "@/types/game";

interface PageProps {
  params: Promise<{ id: string }>;
}

const uuidSchema = z.uuid();

/**
 * Blind-safe card list for evaluation rooms.
 *
 * The opponent's generated deck must never be fetched before reveal, and snapshot
 * metadata only covers cards that are already visible, so the client is given the
 * entire playable catalog instead. It is identical for every match and therefore
 * says nothing about the deck the model drafted.
 */
async function loadBlindCardCatalog(): Promise<DeckCard[]> {
  const cards = await getDeckBuilderCards();

  return cards.flatMap((card) => {
    const cardDefId = cardCodeToDefId(card.cardCode);

    if (cardDefId === null) {
      return [];
    }

    return [
      {
        cardCode: card.cardCode,
        cardDefId,
        imageKey: card.imageKey,
        name: card.name,
        cardType: card.cardType,
        attack: card.attack,
        health: card.health,
        ikzCost: card.ikzCost,
        quantity: 1,
      },
    ];
  });
}

async function RoomContent({ roomId }: { roomId: string }) {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  const parsedId = uuidSchema.safeParse(roomId);
  if (!parsedId.success) {
    notFound();
  }

  const room = await findRoomById(parsedId.data);

  if (!room || !room.player0Id) {
    notFound();
  }

  const isOwner = room.player0Id === user.id;
  const isRoomMember = isOwner || room.player1Id === user.id;

  // Player ids stay on the server: the client needs only these booleans, and the
  // AI's user id would otherwise be a stable cross-match identifier in an
  // evaluation room.
  const roomData: RoomData = {
    id: room.id,
    status: room.status,
    type: room.type,
    hasPassword: room.passwordHash !== null,
    isInRoom: isRoomMember,
    isOwner,
    createdAt: room.createdAt.toISOString(),
    updatedAt: room.updatedAt.toISOString(),
  };

  const evaluationMatch = isRoomMember ? await getHumanEvaluationMatchByRoomId(room.id) : null;

  // Only the blind fields cross into the client bundle: never the model key,
  // checkpoint hash, seeds, or the generated deck id.
  const evaluation: RoomEvaluationMetadata | null = evaluationMatch
    ? {
        matchId: evaluationMatch.matchId,
        sessionId: evaluationMatch.sessionId,
        ordinal: evaluationMatch.ordinal,
        totalMatches: evaluationMatch.totalMatches,
      }
    : null;

  const evaluationCardCatalog = evaluation ? await loadBlindCardCatalog() : null;

  return (
    <RoomClient
      initialRoom={roomData}
      user={user}
      evaluation={evaluation}
      evaluationCardCatalog={evaluationCardCatalog}
    />
  );
}

export default async function RoomPage({ params }: PageProps) {
  const { id: roomId } = await params;

  return (
    <Suspense fallback={<RoomSkeleton />}>
      <RoomContent roomId={roomId} />
    </Suspense>
  );
}
