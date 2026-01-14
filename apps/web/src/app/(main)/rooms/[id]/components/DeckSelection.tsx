"use client";

import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { useDecks } from "@/hooks/useDecks";
import { useCountdown } from "@/hooks/useCountdown";
import { useRoom } from "@/contexts/RoomContext";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { PlayerCard } from "@/app/(main)/rooms/[id]/components/PlayerCard";
import type { RoomStateMessage } from "@tcg/backend-core/types/ws";

interface DeckSelectionProps {
  roomState: RoomStateMessage;
  userId: string;
  playerSlot: 0 | 1;
  selectedDeckId: string | null;
  onSelectDeck: (deckId: string) => void;
  onReady: (ready: boolean) => void;
}

export function DeckSelection({
  roomState,
  userId,
  playerSlot,
  selectedDeckId,
  onSelectDeck,
  onReady,
}: DeckSelectionProps) {
  const router = useRouter();
  const { leave } = useRoom();
  const { decks, isLoading: isLoadingDecks, error: decksError } = useDecks();
  const { secondsRemaining } = useCountdown(roomState.deckSelectionDeadline);

  const handleLeaveRoom = () => {
    if (!confirm("Are you sure you want to leave? The game will be aborted.")) {
      return;
    }
    leave();
    router.push("/dashboard");
  };

  const currentPlayer = roomState.players[playerSlot];
  const opponentSlot = playerSlot === 0 ? 1 : 0;
  const opponentPlayer = roomState.players[opponentSlot];

  const isReady = currentPlayer?.ready ?? false;
  const hasDeckSelected = currentPlayer?.deckSelected ?? false;

  const handleDeckClick = (deckId: string) => {
    if (isReady) {
      return; // Can't change deck while ready
    }
    onSelectDeck(deckId);
  };

  const handleReadyClick = () => {
    onReady(!isReady);
  };

  return (
    <div className="space-y-6">
      {/* Countdown timer */}
      {roomState.deckSelectionDeadline && (
        <div className="text-center">
          <p className="text-sm text-muted-foreground">
            Time remaining to select deck:
          </p>
          <p className="text-2xl font-bold">
            {Math.floor(secondsRemaining / 60)}:
            {(secondsRemaining % 60).toString().padStart(2, "0")}
          </p>
        </div>
      )}

      {/* Player cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <PlayerCard
          player={roomState.players[0]}
          isCurrentUser={roomState.players[0]?.id === userId}
          playerLabel="Player 1 (Host)"
          showDeckStatus
          showReadyStatus
        />
        <PlayerCard
          player={roomState.players[1]}
          isCurrentUser={roomState.players[1]?.id === userId}
          playerLabel="Player 2"
          showDeckStatus
          showReadyStatus
        />
      </div>

      {/* Deck selection section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Select Your Deck</h3>
          {isReady && (
            <span className="text-sm text-muted-foreground">
              Unready to change deck
            </span>
          )}
        </div>

        {decksError && (
          <Alert variant="destructive">
            <AlertDescription>{decksError}</AlertDescription>
          </Alert>
        )}

        {isLoadingDecks ? (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <Card key={i} className="animate-pulse">
                <CardContent className="pt-4">
                  <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                  <div className="h-3 bg-muted rounded w-1/2" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : decks.length === 0 ? (
          <Alert>
            <AlertDescription>
              You don&apos;t have any decks. Create a deck first to play.
            </AlertDescription>
          </Alert>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {decks.map((deck) => (
              <Card
                key={deck.id}
                className={cn(
                  "cursor-pointer transition-all hover:border-primary",
                  selectedDeckId === deck.id && "border-primary ring-2 ring-primary",
                  isReady && "opacity-50 cursor-not-allowed"
                )}
                onClick={() => handleDeckClick(deck.id)}
              >
                <CardContent className="pt-4">
                  <p className="font-medium">{deck.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {deck.cardCount} cards
                    {deck.isSystemDeck && " (Starter)"}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Ready and Leave buttons */}
      <div className="flex justify-center gap-4">
        <Button
          size="lg"
          onClick={handleReadyClick}
          disabled={!hasDeckSelected && !isReady}
          variant={isReady ? "secondary" : "default"}
        >
          {isReady ? "Cancel Ready" : "Ready"}
        </Button>
        <Button
          size="lg"
          variant="outline"
          onClick={handleLeaveRoom}
        >
          Leave Room
        </Button>
      </div>
    </div>
  );
}
