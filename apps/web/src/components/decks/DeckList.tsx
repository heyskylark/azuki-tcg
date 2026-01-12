"use client";

import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Deck {
  id: string;
  name: string;
  isSystemDeck: boolean;
  cardCount: number;
}

interface DeckListProps {
  decks: Deck[];
}

export function DeckList({ decks }: DeckListProps) {
  if (decks.length === 0) {
    return (
      <Card>
        <CardHeader className="py-8 text-center">
          <p className="text-muted-foreground">
            You don&apos;t have any decks yet.
          </p>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      {decks.map((deck) => (
        <Card key={deck.id}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">{deck.name}</CardTitle>
              {deck.isSystemDeck && (
                <Badge variant="secondary">Starter</Badge>
              )}
            </div>
            <CardDescription>
              {deck.cardCount} cards
            </CardDescription>
          </CardHeader>
        </Card>
      ))}
    </div>
  );
}
