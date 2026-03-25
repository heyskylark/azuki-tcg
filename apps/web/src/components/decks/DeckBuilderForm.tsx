"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Minus, Plus, X } from "lucide-react";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import { cn } from "@/lib/utils";
import { buildImageUrl } from "@/types/game";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { CardElement, CardType } from "@tcg/backend-core/types/cards";
import type { DeckBuilderCard, EditableDeck } from "@tcg/backend-core/types/deck";

type PickerMode = "gate" | "leader" | "main";

interface DeckBuilderFormProps {
  availableCards: DeckBuilderCard[];
  initialDeck?: EditableDeck;
  mode: "create" | "edit";
}

interface InitialBuilderState {
  gateId: string | null;
  leaderId: string | null;
  mainDeckCounts: Record<string, number>;
}

type PreviewSource = "picker" | "selected-slot" | "selected-main";

interface PreviewState {
  card: DeckBuilderCard;
  source: PreviewSource;
}

const MAIN_DECK_CARD_LIMIT = 50;

function isAllowedDeckElement(cardElement: CardElement, gateElement: CardElement): boolean {
  return cardElement === CardElement.NORMAL || cardElement === gateElement;
}

function isMainDeckCardType(cardType: CardType): boolean {
  return (
    cardType === CardType.ENTITY || cardType === CardType.SPELL || cardType === CardType.WEAPON
  );
}

function buildCardLookup(cards: DeckBuilderCard[]): Map<string, DeckBuilderCard> {
  const lookup = new Map<string, DeckBuilderCard>();

  for (const card of cards) {
    lookup.set(card.id, card);
  }

  return lookup;
}

function buildInitialState(
  availableCards: DeckBuilderCard[],
  initialDeck?: EditableDeck
): InitialBuilderState {
  if (initialDeck == null) {
    return {
      gateId: null,
      leaderId: null,
      mainDeckCounts: {},
    };
  }

  const cardsById = buildCardLookup(availableCards);
  let gateId: string | null = null;
  let leaderId: string | null = null;
  const mainDeckCounts: Record<string, number> = {};

  for (const entry of initialDeck.cards) {
    const card = cardsById.get(entry.cardId);

    if (card == null) {
      continue;
    }

    if (card.cardType === CardType.GATE) {
      gateId = entry.cardId;
      continue;
    }

    if (card.cardType === CardType.LEADER) {
      leaderId = entry.cardId;
      continue;
    }

    if (isMainDeckCardType(card.cardType)) {
      mainDeckCounts[entry.cardId] = entry.quantity;
    }
  }

  return {
    gateId,
    leaderId,
    mainDeckCounts,
  };
}

function getMainDeckCount(mainDeckCounts: Record<string, number>): number {
  return Object.values(mainDeckCounts).reduce((total, quantity) => total + quantity, 0);
}

function getCardLabel(card: DeckBuilderCard): string {
  return `${card.name} (${card.cardCode})`;
}

function getFuzzyScore(text: string, query: string): number | null {
  const normalizedText = text.toLowerCase();
  const normalizedQuery = query.trim().toLowerCase();

  if (normalizedQuery.length === 0) {
    return 0;
  }

  const directMatchIndex = normalizedText.indexOf(normalizedQuery);
  if (directMatchIndex !== -1) {
    return 10_000 - directMatchIndex;
  }

  let searchIndex = 0;
  let score = 0;
  let firstMatchIndex = -1;
  let consecutiveMatches = 0;

  for (const character of normalizedQuery) {
    const matchIndex = normalizedText.indexOf(character, searchIndex);

    if (matchIndex === -1) {
      return null;
    }

    if (firstMatchIndex === -1) {
      firstMatchIndex = matchIndex;
    }

    if (matchIndex === searchIndex) {
      consecutiveMatches += 1;
    } else {
      consecutiveMatches = 1;
    }

    score += 5 + consecutiveMatches;

    const previousCharacter = normalizedText[matchIndex - 1];
    if (
      matchIndex === 0 ||
      previousCharacter === " " ||
      previousCharacter === "-" ||
      previousCharacter === "_"
    ) {
      score += 8;
    }

    searchIndex = matchIndex + 1;
  }

  return score - Math.max(firstMatchIndex, 0);
}

function getCardSearchScore(card: DeckBuilderCard, query: string): number | null {
  const nameScore = getFuzzyScore(card.name, query);
  const codeScore = getFuzzyScore(card.cardCode, query);

  if (nameScore == null && codeScore == null) {
    return null;
  }

  return Math.max(nameScore ?? Number.NEGATIVE_INFINITY, codeScore ?? Number.NEGATIVE_INFINITY);
}

function isCardSelectedInPicker(
  card: DeckBuilderCard,
  selectedGateId: string | null,
  selectedLeaderId: string | null,
  mainDeckCounts: Record<string, number>
): boolean {
  if (card.cardType === CardType.GATE) {
    return selectedGateId === card.id;
  }

  if (card.cardType === CardType.LEADER) {
    return selectedLeaderId === card.id;
  }

  if (isMainDeckCardType(card.cardType)) {
    return (mainDeckCounts[card.id] ?? 0) > 0;
  }

  return false;
}

function DeckSlotButton({
  title,
  description,
  card,
  onClick,
  onPreview,
  disabled = false,
}: {
  title: string;
  description: string;
  card: DeckBuilderCard | null;
  onClick: () => void;
  onPreview: (card: DeckBuilderCard, source: PreviewSource) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "w-full rounded-xl border p-4 text-left transition-colors",
        "hover:border-primary disabled:cursor-not-allowed disabled:opacity-60"
      )}
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium">{title}</p>
          <p className="text-muted-foreground text-sm">{description}</p>
        </div>
        {card != null && <Badge variant="secondary">{card.element}</Badge>}
      </div>
      {card == null ? (
        <div className="text-muted-foreground rounded-lg border border-dashed px-4 py-8 text-sm">
          Nothing selected
        </div>
      ) : (
        <div className="flex items-center gap-3">
          <img
            src={buildImageUrl(card.imageKey)}
            alt={card.name}
            className="h-24 w-16 cursor-zoom-in rounded-md border object-cover"
            onClick={(event) => {
              event.stopPropagation();
              onPreview(card, "selected-slot");
            }}
          />
          <div className="min-w-0">
            <p className="truncate font-medium">{card.name}</p>
            <p className="text-muted-foreground text-sm">{card.cardCode}</p>
            <p className="text-muted-foreground text-sm">{card.cardType}</p>
          </div>
        </div>
      )}
    </button>
  );
}

function CardPickerRow({
  card,
  mode,
  quantity,
  onAdd,
  onRemove,
  onPreview,
  onSelect,
  disableAdd,
}: {
  card: DeckBuilderCard;
  mode: PickerMode;
  quantity: number;
  onAdd: () => void;
  onRemove: () => void;
  onPreview: (card: DeckBuilderCard, source: PreviewSource) => void;
  onSelect: () => void;
  disableAdd: boolean;
}) {
  const isPrimaryDisabled = mode === "main" ? disableAdd : false;
  const primaryActionLabel =
    mode === "main" ? `Add ${getCardLabel(card)}` : `Select ${getCardLabel(card)}`;

  const handlePrimaryAction = () => {
    if (isPrimaryDisabled) {
      return;
    }

    if (mode === "main") {
      onAdd();
      return;
    }

    onSelect();
  };

  return (
    <div
      role="button"
      tabIndex={isPrimaryDisabled ? -1 : 0}
      aria-label={primaryActionLabel}
      onClick={handlePrimaryAction}
      onKeyDown={(event) => {
        if (event.key !== "Enter" && event.key !== " ") {
          return;
        }

        event.preventDefault();
        handlePrimaryAction();
      }}
      className={cn(
        "flex items-center gap-3 rounded-xl border p-3 transition-colors",
        isPrimaryDisabled ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:border-primary"
      )}
    >
      <img
        src={buildImageUrl(card.imageKey)}
        alt={card.name}
        className="h-24 w-16 cursor-zoom-in rounded-md border object-cover"
        onClick={(event) => {
          event.stopPropagation();
          onPreview(card, "picker");
        }}
      />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <p className="font-medium">{card.name}</p>
          <Badge variant="secondary">{card.element}</Badge>
          <Badge variant="outline">{card.cardType}</Badge>
        </div>
        <p className="text-muted-foreground text-sm">{card.cardCode}</p>
        <p className="text-muted-foreground text-sm">
          {card.attack != null && card.health != null
            ? `${card.attack}/${card.health}`
            : "Support card"}
          {card.ikzCost != null ? ` • ${card.ikzCost} IKZ` : ""}
        </p>
      </div>
      {mode === "main" ? (
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="icon-sm"
            onClick={(event) => {
              event.stopPropagation();
              onRemove();
            }}
            disabled={quantity === 0}
            aria-label={`Remove ${getCardLabel(card)}`}
          >
            <Minus />
          </Button>
          <div className="w-10 text-center text-sm font-medium">{quantity}</div>
          <Button
            type="button"
            variant="outline"
            size="icon-sm"
            onClick={(event) => {
              event.stopPropagation();
              onAdd();
            }}
            disabled={disableAdd}
            aria-label={`Add ${getCardLabel(card)}`}
          >
            <Plus />
          </Button>
        </div>
      ) : (
        <p className="text-muted-foreground text-sm">Click anywhere on the row to select</p>
      )}
    </div>
  );
}

function CardPreviewOverlay({
  previewState,
  onClose,
  onSelect,
  onAdd,
  onRemove,
  actionDisabled,
  selectedQuantity,
}: {
  previewState: PreviewState;
  onClose: () => void;
  onSelect: () => void;
  onAdd: () => void;
  onRemove: () => void;
  actionDisabled: boolean;
  selectedQuantity: number;
}) {
  const { card, source } = previewState;
  const showAdjuster = source === "selected-main" && isMainDeckCardType(card.cardType);
  const showSelectedStatus =
    source === "selected-slot" || (source === "selected-main" && !showAdjuster);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
      onClick={onClose}
    >
      <div
        className="bg-background relative flex max-h-full w-full max-w-6xl flex-col gap-4 overflow-hidden rounded-2xl border p-4 md:flex-row md:p-6"
        onClick={(event) => event.stopPropagation()}
      >
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="absolute right-4 top-4 z-10"
          onClick={onClose}
          aria-label="Close card preview"
        >
          <X />
        </Button>
        <div className="flex min-h-0 flex-1 items-center justify-center">
          <img
            src={buildImageUrl(card.imageKey)}
            alt={card.name}
            className="max-h-[85vh] w-auto max-w-full object-contain"
          />
        </div>
        <div className="flex w-full flex-col gap-3 md:max-w-sm">
          <div>
            <p className="text-2xl font-semibold">{card.name}</p>
            <p className="text-muted-foreground mt-1 text-sm">{card.cardCode}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">{card.element}</Badge>
            <Badge variant="outline">{card.cardType}</Badge>
          </div>
          <div className="text-muted-foreground space-y-2 text-sm">
            <p>
              Stats:{" "}
              {card.attack != null && card.health != null
                ? `${card.attack}/${card.health}`
                : "Support card"}
            </p>
            <p>IKZ Cost: {card.ikzCost ?? 0}</p>
          </div>
          <p className="text-muted-foreground text-sm">
            Click outside the card or press the close button to return to the deck editor.
          </p>
          {showAdjuster ? (
            <div className="flex items-center gap-3">
              <Button type="button" variant="outline" size="icon-sm" onClick={onRemove}>
                <Minus />
              </Button>
              <div className="min-w-12 text-center text-sm font-medium">{selectedQuantity}</div>
              <Button
                type="button"
                variant="outline"
                size="icon-sm"
                onClick={onAdd}
                disabled={actionDisabled}
              >
                <Plus />
              </Button>
            </div>
          ) : showSelectedStatus ? (
            <div className="text-muted-foreground text-sm font-medium">Selected Card</div>
          ) : (
            <Button onClick={onSelect} disabled={actionDisabled}>
              Select Card
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

export function DeckBuilderForm({ availableCards, initialDeck, mode }: DeckBuilderFormProps) {
  const router = useRouter();
  const initialState = buildInitialState(availableCards, initialDeck);
  const [deckName, setDeckName] = useState(initialDeck?.name ?? "");
  const [selectedGateId, setSelectedGateId] = useState<string | null>(initialState.gateId);
  const [selectedLeaderId, setSelectedLeaderId] = useState<string | null>(initialState.leaderId);
  const [mainDeckCounts, setMainDeckCounts] = useState<Record<string, number>>(
    initialState.mainDeckCounts
  );
  const [pickerMode, setPickerMode] = useState<PickerMode>(
    initialState.gateId == null ? "gate" : initialState.leaderId == null ? "leader" : "main"
  );
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [previewState, setPreviewState] = useState<PreviewState | null>(null);
  const [pickerSearch, setPickerSearch] = useState("");
  const cardsById = buildCardLookup(availableCards);
  const selectedGate = selectedGateId == null ? null : (cardsById.get(selectedGateId) ?? null);
  const selectedLeader =
    selectedLeaderId == null ? null : (cardsById.get(selectedLeaderId) ?? null);
  const gateOptions = availableCards.filter((card) => card.cardType === CardType.GATE);
  const leaderOptions =
    selectedGate == null
      ? []
      : availableCards.filter(
          (card) =>
            card.cardType === CardType.LEADER &&
            isAllowedDeckElement(card.element, selectedGate.element)
        );
  const mainDeckOptions =
    selectedGate == null
      ? []
      : availableCards.filter(
          (card) =>
            isMainDeckCardType(card.cardType) &&
            isAllowedDeckElement(card.element, selectedGate.element)
        );

  const selectedMainDeckCards = Object.entries(mainDeckCounts)
    .map(([cardId, quantity]) => {
      const card = cardsById.get(cardId);

      if (card == null) {
        return null;
      }

      return {
        card,
        quantity,
      };
    })
    .filter((entry) => entry != null)
    .sort((leftEntry, rightEntry) => leftEntry.card.name.localeCompare(rightEntry.card.name));

  const mainDeckCount = getMainDeckCount(mainDeckCounts);
  const canSubmit =
    deckName.trim().length > 0 &&
    selectedGate != null &&
    selectedLeader != null &&
    mainDeckCount === MAIN_DECK_CARD_LIMIT &&
    !isSubmitting;

  const pickerCards =
    pickerMode === "gate" ? gateOptions : pickerMode === "leader" ? leaderOptions : mainDeckOptions;
  const filteredPickerCards = pickerCards
    .filter(
      (card) => !isCardSelectedInPicker(card, selectedGateId, selectedLeaderId, mainDeckCounts)
    )
    .map((card) => ({
      card,
      score: getCardSearchScore(card, pickerSearch),
    }))
    .filter((entry) => entry.score != null)
    .sort((leftEntry, rightEntry) => {
      const scoreDifference = (rightEntry.score ?? 0) - (leftEntry.score ?? 0);

      if (scoreDifference !== 0) {
        return scoreDifference;
      }

      return leftEntry.card.name.localeCompare(rightEntry.card.name);
    })
    .map((entry) => entry.card);
  const previewCard = previewState?.card ?? null;
  const previewActionDisabled =
    previewCard == null
      ? true
      : previewCard.cardType === CardType.LEADER
        ? selectedGate == null || !isAllowedDeckElement(previewCard.element, selectedGate.element)
        : isMainDeckCardType(previewCard.cardType)
          ? selectedGate == null ||
            !isAllowedDeckElement(previewCard.element, selectedGate.element) ||
            mainDeckCount >= MAIN_DECK_CARD_LIMIT
          : false;
  const previewSelectedQuantity =
    previewCard == null || !isMainDeckCardType(previewCard.cardType)
      ? 0
      : (mainDeckCounts[previewCard.id] ?? 0);

  const openPreview = (card: DeckBuilderCard, source: PreviewSource) => {
    setPreviewState({ card, source });
  };

  const handleGateSelect = (gateId: string) => {
    const nextGate = cardsById.get(gateId);

    if (nextGate == null) {
      return;
    }

    const shouldResetDeck = selectedGate != null && selectedGate.element !== nextGate.element;

    setSelectedGateId(gateId);
    setPickerMode("leader");
    setPickerSearch("");
    setError(null);

    if (shouldResetDeck) {
      setSelectedLeaderId(null);
      setMainDeckCounts({});
      return;
    }

    if (selectedLeader != null && !isAllowedDeckElement(selectedLeader.element, nextGate.element)) {
      setSelectedLeaderId(null);
    }
  };

  const handleLeaderSelect = (leaderId: string) => {
    setSelectedLeaderId(leaderId);
    setPickerMode("main");
    setPickerSearch("");
    setError(null);
  };

  const handleAddMainDeckCard = (cardId: string) => {
    if (mainDeckCount >= MAIN_DECK_CARD_LIMIT) {
      return;
    }

    setMainDeckCounts((currentCounts) => ({
      ...currentCounts,
      [cardId]: (currentCounts[cardId] ?? 0) + 1,
    }));
    setPickerSearch("");
    setError(null);
  };

  const handleRemoveMainDeckCard = (cardId: string) => {
    setMainDeckCounts((currentCounts) => {
      const nextQuantity = (currentCounts[cardId] ?? 0) - 1;

      if (nextQuantity <= 0) {
        const { [cardId]: _removedCard, ...remainingCounts } = currentCounts;
        return remainingCounts;
      }

      return {
        ...currentCounts,
        [cardId]: nextQuantity,
      };
    });
  };

  const handlePreviewPrimaryAction = () => {
    if (previewCard == null || previewActionDisabled) {
      return;
    }

    if (previewCard.cardType === CardType.GATE) {
      handleGateSelect(previewCard.id);
      setPreviewState(null);
      return;
    }

    if (previewCard.cardType === CardType.LEADER) {
      handleLeaderSelect(previewCard.id);
      setPreviewState(null);
      return;
    }

    handleAddMainDeckCard(previewCard.id);
    setPreviewState(null);
  };

  const handlePreviewAdd = () => {
    if (previewCard == null) {
      return;
    }

    handleAddMainDeckCard(previewCard.id);
  };

  const handlePreviewRemove = () => {
    if (previewCard == null) {
      return;
    }

    handleRemoveMainDeckCard(previewCard.id);

    if (previewSelectedQuantity <= 1) {
      setPreviewState(null);
    }
  };

  const handleSubmit = async () => {
    if (!canSubmit) {
      setError("Deck name, gate, leader, and 50 main deck cards are required");
      return;
    }

    if (mode === "edit" && initialDeck == null) {
      setError("Deck could not be loaded");
      return;
    }

    if (selectedGateId == null || selectedLeaderId == null) {
      setError("Deck name, gate, leader, and 50 main deck cards are required");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const cardCounts: Record<string, number> = {
      ...mainDeckCounts,
      [selectedGateId]: 1,
      [selectedLeaderId]: 1,
    };
    const updateDeckId = initialDeck?.id ?? null;

    const response = await authenticatedFetch(
      mode === "create" ? "/api/decks" : `/api/decks/${updateDeckId}`,
      {
        method: mode === "create" ? "POST" : "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: deckName,
          cardCounts,
        }),
      }
    );

    let responseBody: { message?: string } | null = null;

    try {
      responseBody = await response.json();
    } catch {
      responseBody = null;
    }

    if (!response.ok) {
      setError(responseBody?.message ?? "Failed to save deck");
      setIsSubmitting(false);
      return;
    }

    router.push("/decks");
    router.refresh();
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold">{mode === "create" ? "Create Deck" : "Edit Deck"}</h1>
          <p className="text-muted-foreground mt-2">
            Pick a gate, match a leader, and build a 50-card main deck.
          </p>
        </div>
        <div className="flex gap-3">
          <Button asChild variant="outline">
            <Link href="/decks">Cancel</Link>
          </Button>
          <Button onClick={handleSubmit} disabled={!canSubmit}>
            {isSubmitting ? "Saving..." : mode === "create" ? "Create Deck" : "Save Changes"}
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Deck Details</CardTitle>
          <CardDescription>Name your deck before saving it.</CardDescription>
        </CardHeader>
        <CardContent>
          <Input
            value={deckName}
            onChange={(event) => setDeckName(event.target.value)}
            placeholder="Deck name"
            maxLength={120}
          />
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Core Cards</CardTitle>
              <CardDescription>
                Start with a gate, then pick a leader that matches it.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <DeckSlotButton
                title="Gate"
                description="Choose your gate first."
                card={selectedGate}
                onClick={() => setPickerMode("gate")}
                onPreview={openPreview}
              />
              <DeckSlotButton
                title="Leader"
                description={
                  selectedGate == null
                    ? "Select a gate to unlock leader choices."
                    : "Pick a leader that matches the gate element."
                }
                card={selectedLeader}
                onClick={() => setPickerMode("leader")}
                onPreview={openPreview}
                disabled={selectedGate == null}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Main Deck</CardTitle>
              <CardDescription>
                {mainDeckCount} / {MAIN_DECK_CARD_LIMIT} cards selected. IKZ cards are added
                automatically.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                type="button"
                variant={pickerMode === "main" ? "default" : "outline"}
                onClick={() => setPickerMode("main")}
                disabled={selectedGate == null}
              >
                Open Main Deck Picker
              </Button>

              {selectedMainDeckCards.length === 0 ? (
                <div className="text-muted-foreground rounded-xl border border-dashed px-4 py-8 text-sm">
                  No main deck cards selected yet.
                </div>
              ) : (
                <div className="space-y-3">
                  {selectedMainDeckCards.map((entry) => (
                    <div
                      key={entry.card.id}
                      className="flex items-center gap-3 rounded-xl border p-3"
                    >
                      <img
                        src={buildImageUrl(entry.card.imageKey)}
                        alt={entry.card.name}
                        className="h-20 w-14 cursor-zoom-in rounded-md border object-cover"
                        onClick={() => openPreview(entry.card, "selected-main")}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium">{entry.card.name}</p>
                        <p className="text-muted-foreground text-sm">{entry.card.cardCode}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          type="button"
                          variant="outline"
                          size="icon-sm"
                          onClick={() => handleRemoveMainDeckCard(entry.card.id)}
                          aria-label={`Remove ${getCardLabel(entry.card)}`}
                        >
                          <Minus />
                        </Button>
                        <div className="w-10 text-center text-sm font-medium">{entry.quantity}</div>
                        <Button
                          type="button"
                          variant="outline"
                          size="icon-sm"
                          onClick={() => handleAddMainDeckCard(entry.card.id)}
                          disabled={mainDeckCount >= MAIN_DECK_CARD_LIMIT}
                          aria-label={`Add ${getCardLabel(entry.card)}`}
                        >
                          <Plus />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>
              {pickerMode === "gate"
                ? "Gate Picker"
                : pickerMode === "leader"
                  ? "Leader Picker"
                  : "Main Deck Picker"}
            </CardTitle>
            <CardDescription>
              {pickerMode === "gate"
                ? "Choose from every gate in the system."
                : pickerMode === "leader"
                  ? selectedGate == null
                    ? "Pick a gate before choosing a leader."
                    : "Only leaders that match the selected gate element are shown."
                  : selectedGate == null
                    ? "Pick a gate before adding main deck cards."
                    : "Only NORMAL cards and cards that match the gate element are shown."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant={pickerMode === "gate" ? "default" : "outline"}
                onClick={() => setPickerMode("gate")}
              >
                Gates
              </Button>
              <Button
                type="button"
                variant={pickerMode === "leader" ? "default" : "outline"}
                onClick={() => setPickerMode("leader")}
                disabled={selectedGate == null}
              >
                Leaders
              </Button>
              <Button
                type="button"
                variant={pickerMode === "main" ? "default" : "outline"}
                onClick={() => setPickerMode("main")}
                disabled={selectedGate == null}
              >
                Main Deck
              </Button>
            </div>
            <Input
              value={pickerSearch}
              onChange={(event) => setPickerSearch(event.target.value)}
              placeholder="Fuzzy search by card name or card code"
            />

            {filteredPickerCards.length === 0 ? (
              <div className="text-muted-foreground rounded-xl border border-dashed px-4 py-12 text-sm">
                {pickerCards.length === 0
                  ? "Pick a gate to unlock this picker."
                  : pickerSearch.trim().length > 0
                    ? "No cards match the current search."
                    : "No cards match the current picker."}
              </div>
            ) : (
              <div className="max-h-[70vh] space-y-3 overflow-y-auto pr-1">
                {filteredPickerCards.map((card) => (
                  <CardPickerRow
                    key={card.id}
                    card={card}
                    mode={pickerMode}
                    quantity={mainDeckCounts[card.id] ?? 0}
                    onAdd={() => handleAddMainDeckCard(card.id)}
                    onRemove={() => handleRemoveMainDeckCard(card.id)}
                    onPreview={openPreview}
                    onSelect={() => {
                      if (pickerMode === "gate") {
                        handleGateSelect(card.id);
                        return;
                      }

                      handleLeaderSelect(card.id);
                    }}
                    disableAdd={mainDeckCount >= MAIN_DECK_CARD_LIMIT}
                  />
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      {previewState != null && (
        <CardPreviewOverlay
          previewState={previewState}
          onClose={() => setPreviewState(null)}
          onSelect={handlePreviewPrimaryAction}
          onAdd={handlePreviewAdd}
          onRemove={handlePreviewRemove}
          actionDisabled={previewActionDisabled}
          selectedQuantity={previewSelectedQuantity}
        />
      )}
    </div>
  );
}
