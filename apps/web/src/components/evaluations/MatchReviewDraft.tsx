import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { buildImageUrl } from "@/types/game";
import type { DeckBuilderCard } from "@tcg/backend-core/types/deck";
import {
  HumanEvaluationDeckSource,
  type HumanEvaluationReview,
} from "@tcg/backend-core/types/humanEvaluations";

interface MatchReviewDraftProps {
  draft: HumanEvaluationReview["draft"];
  assignment: HumanEvaluationReview["assignment"];
  cardsByCode: Record<string, DeckBuilderCard>;
}

interface DeckMetrics {
  totalCards: number;
  distinctCards: number;
  unknownCodes: string[];
  typeCounts: Map<string, number>;
  elementCounts: Map<string, number>;
  costCounts: Map<number, number>;
  copiesHistogram: Map<number, number>;
  topCards: Array<{ code: string; count: number; card: DeckBuilderCard | undefined }>;
}

function computeDeckMetrics(
  cardCounts: Record<string, number>,
  cardsByCode: Record<string, DeckBuilderCard>
): DeckMetrics {
  const typeCounts = new Map<string, number>();
  const elementCounts = new Map<string, number>();
  const costCounts = new Map<number, number>();
  const copiesHistogram = new Map<number, number>();
  const unknownCodes: string[] = [];
  let totalCards = 0;

  for (const [code, count] of Object.entries(cardCounts)) {
    totalCards += count;
    copiesHistogram.set(count, (copiesHistogram.get(count) ?? 0) + 1);

    const card = cardsByCode[code];
    if (!card) {
      unknownCodes.push(code);
      continue;
    }

    typeCounts.set(card.cardType, (typeCounts.get(card.cardType) ?? 0) + count);
    elementCounts.set(card.element, (elementCounts.get(card.element) ?? 0) + count);

    if (card.ikzCost !== null) {
      costCounts.set(card.ikzCost, (costCounts.get(card.ikzCost) ?? 0) + count);
    }
  }

  const topCards = Object.entries(cardCounts)
    .map(([code, count]) => ({ code, count, card: cardsByCode[code] }))
    .sort((left, right) => right.count - left.count || left.code.localeCompare(right.code))
    .slice(0, 8);

  return {
    totalCards,
    distinctCards: Object.keys(cardCounts).length,
    unknownCodes,
    typeCounts,
    elementCounts,
    costCounts,
    copiesHistogram,
    topCards,
  };
}

function CardThumb({
  code,
  card,
  caption,
}: {
  code: string;
  card: DeckBuilderCard | undefined;
  caption?: string;
}) {
  return (
    <div className="flex items-center gap-3">
      {card ? (
        <img
          src={buildImageUrl(card.imageKey)}
          alt={card.name}
          loading="lazy"
          className="h-20 w-14 shrink-0 rounded-md border object-cover"
        />
      ) : (
        <div className="bg-muted text-muted-foreground flex h-20 w-14 shrink-0 items-center justify-center rounded-md border text-[10px]">
          no art
        </div>
      )}
      <div className="min-w-0">
        <p className="truncate text-sm font-medium">{card?.name ?? "Unknown card"}</p>
        <p className="text-muted-foreground font-mono text-xs">{code}</p>
        {caption ? <p className="text-muted-foreground mt-1 text-xs">{caption}</p> : null}
      </div>
    </div>
  );
}

function DistributionBars({
  title,
  entries,
  total,
}: {
  title: string;
  entries: Array<[string, number]>;
  total: number;
}) {
  return (
    <div>
      <p className="text-muted-foreground text-xs font-medium">{title}</p>
      <ul className="mt-2 space-y-1.5">
        {entries.map(([label, count]) => (
          <li key={label} className="flex items-center gap-3">
            <span className="w-28 shrink-0 truncate text-xs capitalize">
              {label.toLowerCase().replace(/_/g, " ")}
            </span>
            <span className="bg-secondary h-1.5 min-w-0 flex-1 overflow-hidden rounded-full">
              <span
                className="bg-primary block h-full rounded-full"
                style={{ width: total === 0 ? "0%" : `${(count / total) * 100}%` }}
              />
            </span>
            <span className="text-muted-foreground w-8 shrink-0 text-right text-xs tabular-nums">
              {count}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function MatchReviewDraft({ draft, assignment, cardsByCode }: MatchReviewDraftProps) {
  const gateCard = cardsByCode[assignment.gateCardCode];
  const leaderCard = cardsByCode[assignment.leaderCardCode];
  const isPremade = assignment.deckSource === HumanEvaluationDeckSource.PREMADE;

  if (draft === null) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{isPremade ? "Premade deck" : "Drafted deck"}</CardTitle>
          <CardDescription>
            No deck artifact was stored for this match — deck preparation never completed, which is
            why the match is recorded as a technical abort.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-2">
          <CardThumb code={assignment.gateCardCode} card={gateCard} caption="Assigned gate" />
          <CardThumb code={assignment.leaderCardCode} card={leaderCard} caption="Assigned leader" />
        </CardContent>
      </Card>
    );
  }

  const metrics = computeDeckMetrics(draft.cardCounts, cardsByCode);
  const sortedCosts = [...metrics.costCounts.entries()].sort((left, right) => left[0] - right[0]);
  const sortedCopies = [...metrics.copiesHistogram.entries()].sort(
    (left, right) => left[0] - right[0]
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>{isPremade ? "Premade deck" : "Drafted deck"}</CardTitle>
        <CardDescription>
          {isPremade ? (
            <>
              Fixed tournament deck <span className="font-mono">{assignment.premadeDeckSlug}</span>{" "}
              from the model&apos;s curated deck pool; no draft took place.
            </>
          ) : (
            "Built by the model before the match from the assigned gate and leader, one card at a time out of server-offered candidates."
          )}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-8">
        <div className="grid gap-4 sm:grid-cols-2">
          <CardThumb code={assignment.gateCardCode} card={gateCard} caption="Assigned gate" />
          <CardThumb code={assignment.leaderCardCode} card={leaderCard} caption="Assigned leader" />
        </div>

        <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <dt className="text-muted-foreground text-xs">Main deck cards</dt>
            <dd className="text-lg font-semibold tabular-nums">{metrics.totalCards}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Distinct cards</dt>
            <dd className="text-lg font-semibold tabular-nums">{metrics.distinctCards}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Draft picks recorded</dt>
            <dd className="text-lg font-semibold tabular-nums">{draft.picks.length}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Copies profile</dt>
            <dd className="text-sm tabular-nums">
              {sortedCopies.length === 0
                ? "—"
                : sortedCopies.map(([copies, cards]) => `${cards}×${copies}-of`).join(" · ")}
            </dd>
          </div>
        </dl>

        <div className="grid gap-6 lg:grid-cols-3">
          <DistributionBars
            title="Card types"
            entries={[...metrics.typeCounts.entries()].sort((left, right) => right[1] - left[1])}
            total={metrics.totalCards}
          />
          <DistributionBars
            title="Elements"
            entries={[...metrics.elementCounts.entries()].sort((left, right) => right[1] - left[1])}
            total={metrics.totalCards}
          />
          <DistributionBars
            title="IKZ cost curve"
            entries={sortedCosts.map(([cost, count]): [string, number] => [`Cost ${cost}`, count])}
            total={metrics.totalCards}
          />
        </div>

        {metrics.unknownCodes.length > 0 ? (
          <p className="text-muted-foreground text-xs">
            {metrics.unknownCodes.length} drafted card code(s) are not in the current playable
            catalog: <span className="font-mono">{metrics.unknownCodes.join(", ")}</span>
          </p>
        ) : null}

        <div>
          <p className="text-muted-foreground text-xs font-medium">
            {isPremade ? "Most-included cards" : "Most-drafted cards"}
          </p>
          <ul className="mt-3 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {metrics.topCards.map((entry) => (
              <li key={entry.code}>
                <CardThumb
                  code={entry.code}
                  card={entry.card}
                  caption={`${entry.count} cop${entry.count === 1 ? "y" : "ies"}`}
                />
              </li>
            ))}
          </ul>
        </div>

        <dl className="grid gap-3 border-t pt-6 sm:grid-cols-3">
          <div className="min-w-0">
            <dt className="text-muted-foreground text-xs">Deck hash</dt>
            <dd className="font-mono text-xs break-all">{draft.deckHash}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-muted-foreground text-xs">Catalog hash</dt>
            <dd className="font-mono text-xs break-all">{draft.catalogHash}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-muted-foreground text-xs">Checkpoint sha256</dt>
            <dd className="font-mono text-xs break-all">{draft.checkpointSha256}</dd>
          </div>
        </dl>

        {isPremade ? null : (
          <details className="border-t pt-6">
            <summary className="cursor-pointer text-sm font-medium">
              Draft order and offered candidates ({draft.picks.length} picks)
            </summary>
            <ol className="mt-4 space-y-2">
              {draft.picks.map((pick) => (
                <li
                  key={pick.ordinal}
                  className="flex flex-wrap items-baseline gap-x-3 gap-y-2 rounded-md border px-3 py-2"
                >
                  <span className="text-muted-foreground w-10 shrink-0 text-xs tabular-nums">
                    #{pick.ordinal}
                  </span>
                  <span className="flex flex-wrap gap-1.5">
                    {pick.candidateCardCodes.map((candidateCode, candidateIndex) => {
                      const isSelected = candidateIndex === pick.selectedIndex;

                      return (
                        <Badge
                          key={`${pick.ordinal}-${candidateIndex}-${candidateCode}`}
                          variant={isSelected ? "default" : "outline"}
                          className="font-mono text-[11px]"
                          title={cardsByCode[candidateCode]?.name ?? candidateCode}
                        >
                          {cardsByCode[candidateCode]?.name ?? candidateCode}
                        </Badge>
                      );
                    })}
                  </span>
                  <span className="text-muted-foreground ml-auto text-xs">
                    picked index {pick.selectedIndex} ·{" "}
                    <span className="font-mono">{pick.selectedCardCode}</span>
                  </span>
                </li>
              ))}
            </ol>
          </details>
        )}

        <details>
          <summary className="cursor-pointer text-sm font-medium">
            Final deck in {isPremade ? "deck list" : "draft"} order (
            {draft.orderedMainCardCodes.length} cards)
          </summary>
          <ol className="mt-3 flex flex-wrap gap-1.5">
            {draft.orderedMainCardCodes.map((code, index) => (
              <li key={`${index}-${code}`}>
                <Badge variant="outline" className="font-mono text-[11px]">
                  <span className="text-muted-foreground mr-1 tabular-nums">{index + 1}</span>
                  {cardsByCode[code]?.name ?? code}
                </Badge>
              </li>
            ))}
          </ol>
        </details>
      </CardContent>
    </Card>
  );
}
