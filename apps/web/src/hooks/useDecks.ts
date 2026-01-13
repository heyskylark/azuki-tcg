"use client";

import { useState, useEffect, useCallback } from "react";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";

export interface DeckSummary {
  id: string;
  name: string;
  isSystemDeck: boolean;
  cardCount: number;
}

interface UseDecksReturn {
  decks: DeckSummary[];
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useDecks(): UseDecksReturn {
  const [decks, setDecks] = useState<DeckSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDecks = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await authenticatedFetch("/api/decks");

      if (!response.ok) {
        throw new Error("Failed to fetch decks");
      }

      const data = await response.json();
      setDecks(data.decks);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch decks");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDecks();
  }, [fetchDecks]);

  return {
    decks,
    isLoading,
    error,
    refetch: fetchDecks,
  };
}
