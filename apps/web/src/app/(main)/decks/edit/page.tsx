import { notFound, redirect } from "next/navigation";
import { DeckBuilderForm } from "@/components/decks/DeckBuilderForm";
import { getServerUser } from "@/lib/auth/getServerUser";
import {
  getDeckBuilderCards,
  getEditableDeckForUser,
} from "@tcg/backend-core/services/DeckService";

interface EditDeckPageProps {
  searchParams: Promise<{ deckId?: string }>;
}

export default async function EditDeckPage({ searchParams }: EditDeckPageProps) {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  const { deckId } = await searchParams;

  if (deckId == null) {
    notFound();
  }

  const [availableCards, editableDeck] = await Promise.all([
    getDeckBuilderCards(),
    getEditableDeckForUser(deckId, user.id),
  ]);

  if (editableDeck == null || editableDeck.isSystemDeck) {
    notFound();
  }

  return <DeckBuilderForm availableCards={availableCards} initialDeck={editableDeck} mode="edit" />;
}
