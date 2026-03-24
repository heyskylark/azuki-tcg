"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Copy, Pencil, Trash2, X } from "lucide-react";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
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
import { Input } from "@/components/ui/input";

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
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [copyDeckId, setCopyDeckId] = useState<string | null>(null);
  const [copyDeckName, setCopyDeckName] = useState("");
  const [isCopying, setIsCopying] = useState(false);
  const [deletingDeckId, setDeletingDeckId] = useState<string | null>(null);
  const copySourceDeck = decks.find((deck) => deck.id === copyDeckId) ?? null;

  if (decks.length === 0) {
    return (
      <Card>
        <CardHeader className="py-8 text-center">
          <p className="text-muted-foreground">You don&apos;t have any decks yet.</p>
        </CardHeader>
      </Card>
    );
  }

  const handleOpenCopyModal = (deck: Deck) => {
    setCopyDeckId(deck.id);
    setCopyDeckName(`${deck.name} Copy`);
    setError(null);
  };

  const handleCopyDeck = async () => {
    if (copySourceDeck == null) {
      return;
    }

    setIsCopying(true);
    setError(null);

    const response = await authenticatedFetch(`/api/decks/${copySourceDeck.id}/copy`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: copyDeckName,
      }),
    });

    let responseBody: { message?: string } | null = null;

    try {
      responseBody = await response.json();
    } catch {
      responseBody = null;
    }

    if (!response.ok) {
      setError(responseBody?.message ?? "Failed to copy deck");
      setIsCopying(false);
      return;
    }

    setCopyDeckId(null);
    setCopyDeckName("");
    setIsCopying(false);
    router.refresh();
  };

  const handleDeleteDeck = async (deck: Deck) => {
    if (!confirm(`Delete "${deck.name}"? You can’t edit a deleted deck later.`)) {
      return;
    }

    setDeletingDeckId(deck.id);
    setError(null);

    const response = await authenticatedFetch(`/api/decks/${deck.id}`, {
      method: "DELETE",
    });

    let responseBody: { message?: string } | null = null;

    try {
      responseBody = await response.json();
    } catch {
      responseBody = null;
    }

    if (!response.ok) {
      setError(responseBody?.message ?? "Failed to delete deck");
      setDeletingDeckId(null);
      return;
    }

    setDeletingDeckId(null);
    router.refresh();
  };

  return (
    <>
      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {decks.map((deck) => {
          const isDeleting = deletingDeckId === deck.id;

          return (
            <Card key={deck.id}>
              <CardHeader>
                <div className="flex items-center justify-between gap-3">
                  <CardTitle className="text-lg">{deck.name}</CardTitle>
                  {deck.isSystemDeck && <Badge variant="secondary">Starter</Badge>}
                </div>
                <CardDescription>{deck.cardCount} cards</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground text-sm">
                  {deck.isSystemDeck
                    ? "Starter decks can be copied but not edited."
                    : "User decks can be copied, edited, or soft deleted."}
                </p>
              </CardContent>
              <CardFooter className="flex flex-wrap gap-2">
                <Button type="button" variant="outline" onClick={() => handleOpenCopyModal(deck)}>
                  <Copy />
                  Copy
                </Button>
                {!deck.isSystemDeck && (
                  <Button asChild variant="outline">
                    <Link href={`/decks/edit?deckId=${deck.id}`}>
                      <Pencil />
                      Edit
                    </Link>
                  </Button>
                )}
                {!deck.isSystemDeck && (
                  <Button
                    type="button"
                    variant="destructive"
                    onClick={() => handleDeleteDeck(deck)}
                    disabled={isDeleting}
                  >
                    <Trash2 />
                    {isDeleting ? "Deleting..." : "Delete"}
                  </Button>
                )}
              </CardFooter>
            </Card>
          );
        })}
      </div>

      {copySourceDeck != null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
          <div className="bg-background w-full max-w-md rounded-xl border p-6 shadow-lg">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold">Copy Deck</h2>
                <p className="text-muted-foreground mt-2 text-sm">
                  Create a new editable copy of {copySourceDeck.name}.
                </p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => {
                  setCopyDeckId(null);
                  setCopyDeckName("");
                }}
                disabled={isCopying}
                aria-label="Close copy deck modal"
              >
                <X />
              </Button>
            </div>

            <div className="space-y-4">
              <Input
                value={copyDeckName}
                onChange={(event) => setCopyDeckName(event.target.value)}
                placeholder="New deck name"
                maxLength={120}
              />

              <div className="flex justify-end gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setCopyDeckId(null);
                    setCopyDeckName("");
                  }}
                  disabled={isCopying}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  onClick={handleCopyDeck}
                  disabled={isCopying || copyDeckName.trim().length === 0}
                >
                  {isCopying ? "Copying..." : "Create Copy"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
