"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  createEvaluationSession,
  GAMES_PER_MODEL_OPTIONS,
  type EvaluationGamesPerModel,
} from "@/lib/evaluations/client";
import type { DeckSummary } from "@tcg/backend-core/types/deck";

/** gate + leader + 50 main deck cards + 10 auto-added IKZ cards. */
const PLAYABLE_DECK_CARD_COUNT = 62;

const MODE_DESCRIPTIONS: Record<EvaluationGamesPerModel, string> = {
  8: "Eight games per model.",
  16: "Sixteen games per model.",
};

interface EvaluationStartFormProps {
  decks: DeckSummary[];
}

export function EvaluationStartForm({ decks }: EvaluationStartFormProps) {
  const router = useRouter();
  const playableDecks = decks.filter((deck) => deck.cardCount >= PLAYABLE_DECK_CARD_COUNT);
  const [humanDeckId, setHumanDeckId] = useState<string | null>(
    playableDecks.length === 1 ? playableDecks[0].id : null
  );
  const [gamesPerModel, setGamesPerModel] = useState<EvaluationGamesPerModel>(8);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    if (humanDeckId === null) {
      setError("Choose the deck you will play for every match in this session.");
      return;
    }

    setIsStarting(true);

    try {
      const session = await createEvaluationSession({ humanDeckId, gamesPerModel });
      router.push(`/evaluations/${session.id}`);
    } catch (startError) {
      setIsStarting(false);
      setError(
        startError instanceof Error ? startError.message : "Failed to start evaluation session"
      );
    }
  };

  if (playableDecks.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No complete deck to evaluate with</CardTitle>
          <CardDescription>
            An evaluation session locks one of your decks in for every match, so it needs a finished
            50-card deck with a gate and a leader.
          </CardDescription>
        </CardHeader>
        <CardFooter>
          <Button asChild>
            <Link href="/decks/new">Build a deck</Link>
          </Button>
        </CardFooter>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Start a session</CardTitle>
        <CardDescription>
          You pick your deck once. Each opponent prepares its own deck before its match, and which
          opponent you face is assigned by the server.
        </CardDescription>
      </CardHeader>
      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-6">
          {error ? (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <fieldset disabled={isStarting}>
            <legend className="text-sm font-medium">Your deck for the whole session</legend>
            <p className="text-muted-foreground mt-1 text-xs">
              It cannot change once the session starts — a fixed human deck is what makes the model
              comparison meaningful.
            </p>
            <ul className="mt-3 grid gap-2 sm:grid-cols-2">
              {playableDecks.map((deck) => (
                <li key={deck.id}>
                  <label className="block">
                    <input
                      type="radio"
                      name="humanDeckId"
                      value={deck.id}
                      checked={humanDeckId === deck.id}
                      onChange={() => setHumanDeckId(deck.id)}
                      className="peer sr-only"
                    />
                    <span className="hover:bg-accent peer-checked:border-primary peer-checked:bg-accent peer-focus-visible:border-ring peer-focus-visible:ring-ring/50 flex cursor-pointer items-start justify-between gap-3 rounded-lg border px-4 py-3 transition-colors peer-focus-visible:ring-[3px]">
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-medium">{deck.name}</span>
                        <span className="text-muted-foreground text-xs tabular-nums">
                          {deck.cardCount} cards
                        </span>
                      </span>
                      {deck.isSystemDeck ? <Badge variant="outline">Starter</Badge> : null}
                    </span>
                  </label>
                </li>
              ))}
            </ul>
          </fieldset>

          <fieldset disabled={isStarting} className="border-t pt-5">
            <legend className="text-sm font-medium">Games per model</legend>
            <p className="text-muted-foreground mt-1 text-xs">
              Every enabled model gets the same quota, the same deck-assignment plan, and a balanced
              share of the starting-player advantage.
            </p>
            <div className="mt-3 flex gap-2">
              {GAMES_PER_MODEL_OPTIONS.map((option) => (
                <label key={option} className="flex-1">
                  <input
                    type="radio"
                    name="gamesPerModel"
                    value={option}
                    checked={gamesPerModel === option}
                    onChange={() => setGamesPerModel(option)}
                    className="peer sr-only"
                  />
                  <span className="hover:bg-accent peer-checked:border-primary peer-checked:bg-accent peer-focus-visible:border-ring peer-focus-visible:ring-ring/50 flex cursor-pointer flex-col gap-1 rounded-lg border px-4 py-3 transition-colors peer-focus-visible:ring-[3px]">
                    <span className="text-sm font-medium tabular-nums">{option} games</span>
                    <span className="text-muted-foreground text-xs">
                      {MODE_DESCRIPTIONS[option]}
                    </span>
                  </span>
                </label>
              ))}
            </div>
            <p className="text-muted-foreground mt-3 text-xs">
              The session length is decided server-side from the models currently enrolled in
              evaluation. You will see the total, never the roster.
            </p>
          </fieldset>
        </CardContent>
        <CardFooter className="mt-6 border-t">
          <Button type="submit" disabled={isStarting}>
            {isStarting ? "Starting session..." : "Start blind session"}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}
