import { redirect } from "next/navigation";
import { getServerUser } from "@/lib/auth/getServerUser";
import { getUserDecks, getDeckWithCards } from "@tcg/backend-core/services/DeckService";
import GameTestClient from "@/app/dev/game-test/GameTestClient";
import type { DeckCard } from "@/types/game";

/**
 * Development test page for the 3D game renderer.
 * Protected route - requires authentication.
 * Uses real deck data from the database.
 */
export default async function GameTestPage() {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  // Fetch user's decks
  const decks = await getUserDecks(user.id);

  if (decks.length === 0) {
    return (
      <div className="h-screen w-screen bg-black flex items-center justify-center">
        <div className="text-center text-white">
          <h1 className="text-2xl font-bold mb-4">No Decks Found</h1>
          <p className="text-gray-400 mb-4">
            You need at least one deck to test the game renderer.
          </p>
          <a href="/decks" className="text-blue-400 hover:underline">
            Go to Decks
          </a>
        </div>
      </div>
    );
  }

  // Use the first deck for testing
  const firstDeck = decks[0];
  const deckWithCards = await getDeckWithCards(firstDeck.id);

  if (!deckWithCards || deckWithCards.cards.length === 0) {
    return (
      <div className="h-screen w-screen bg-black flex items-center justify-center">
        <div className="text-center text-white">
          <h1 className="text-2xl font-bold mb-4">Empty Deck</h1>
          <p className="text-gray-400 mb-4">
            The deck &quot;{firstDeck.name}&quot; has no cards.
          </p>
          <a href="/decks" className="text-blue-400 hover:underline">
            Go to Decks
          </a>
        </div>
      </div>
    );
  }

  // Convert to DeckCard format expected by the client
  const deckCards: DeckCard[] = deckWithCards.cards.map((card) => ({
    cardCode: card.cardCode,
    cardDefId: card.cardDefId,
    imageKey: card.imageKey,
    name: card.name,
    cardType: card.cardType,
    attack: card.attack,
    health: card.health,
    ikzCost: card.ikzCost,
    quantity: card.quantity,
  }));

  return <GameTestClient deckCards={deckCards} deckName={deckWithCards.name} />;
}
