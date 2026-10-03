import { Suspense } from "react";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { z } from "zod";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { MatchReviewAnnotation } from "@/components/evaluations/MatchReviewAnnotation";
import { MatchReviewDraft } from "@/components/evaluations/MatchReviewDraft";
import { MatchReviewTimeline } from "@/components/evaluations/MatchReviewTimeline";
import { EVALUATION_PROTOCOL } from "@/lib/evaluations/client";
import { getServerUser } from "@/lib/auth/getServerUser";
import { getDeckBuilderCards } from "@tcg/backend-core/services/DeckService";
import { getHumanEvaluationReview } from "@tcg/backend-core/services/humanEvaluationService";
import { ApiError } from "@tcg/backend-core/errors";
import type { DeckBuilderCard } from "@tcg/backend-core/types/deck";
import type { HumanEvaluationReview } from "@tcg/backend-core/types/humanEvaluations";

interface PageProps {
  params: Promise<{ sessionId: string; matchId: string }>;
}

const uuidSchema = z.uuid();

type ReviewLoadResult =
  | {
      status: "ready";
      review: HumanEvaluationReview;
      cards: DeckBuilderCard[];
    }
  | { status: "missing" }
  | { status: "unavailable"; message: string };

async function loadReview(matchId: string, reviewerId: string): Promise<ReviewLoadResult> {
  try {
    const [review, cards] = await Promise.all([
      getHumanEvaluationReview(matchId, reviewerId),
      getDeckBuilderCards(),
    ]);

    return { status: "ready", review, cards };
  } catch (loadError) {
    if (loadError instanceof ApiError && loadError.status === 404) {
      return { status: "missing" };
    }

    if (loadError instanceof ApiError && loadError.status === 403) {
      return { status: "unavailable", message: loadError.message };
    }

    throw loadError;
  }
}

export default async function EvaluationMatchReviewPage({ params }: PageProps) {
  const { sessionId, matchId } = await params;

  return (
    <>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Frozen match review</h1>
          <p className="text-muted-foreground mt-2">
            The draft, every decision, and the annotation you wrote before the reveal.
          </p>
        </div>
        <Button variant="outline" asChild>
          <Link href={`/evaluations/${sessionId}`}>Back to session</Link>
        </Button>
      </div>
      <Suspense fallback={<ReviewSkeleton />}>
        <ReviewContent sessionId={sessionId} matchId={matchId} />
      </Suspense>
    </>
  );
}

function ReviewSkeleton() {
  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-56" />
          <Skeleton className="mt-2 h-4 w-80" />
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-4">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-40" />
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    </div>
  );
}

async function ReviewContent({ sessionId, matchId }: { sessionId: string; matchId: string }) {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  const parsedSessionId = uuidSchema.safeParse(sessionId);
  const parsedMatchId = uuidSchema.safeParse(matchId);

  if (!parsedSessionId.success || !parsedMatchId.success) {
    notFound();
  }

  const loaded = await loadReview(parsedMatchId.data, user.id);

  if (loaded.status === "missing") {
    notFound();
  }

  if (loaded.status === "unavailable") {
    return (
      <Alert>
        <AlertTitle>This match is still blind</AlertTitle>
        <AlertDescription>
          <p>
            {loaded.message} A match review only opens after every match in the session has an
            annotation and the session has been revealed.
          </p>
          <Button size="sm" variant="outline" className="mt-2" asChild>
            <Link href={`/evaluations/${parsedSessionId.data}`}>Back to session</Link>
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  const { review, cards } = loaded;

  if (review.sessionId !== parsedSessionId.data) {
    notFound();
  }

  const cardsByCode = Object.fromEntries(cards.map((card) => [card.cardCode, card]));

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle>Match {review.ordinal}</CardTitle>
                <Badge variant="outline" className="font-mono text-xs">
                  {EVALUATION_PROTOCOL}
                </Badge>
              </div>
              <CardDescription className="mt-2">
                Played against{" "}
                <span className="text-foreground font-medium">{review.model.displayName}</span>
              </CardDescription>
            </div>
            {review.result ? (
              <div className="text-right">
                <p className="text-muted-foreground text-xs">Result</p>
                <p className="text-sm font-medium capitalize">
                  {review.result.winType.toLowerCase().replace(/_/g, " ")}
                </p>
                <p className="text-muted-foreground text-xs tabular-nums">
                  {review.result.totalTurns} turns · {review.result.durationSeconds}s
                </p>
              </div>
            ) : (
              <Badge variant="destructive">No result recorded</Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="min-w-0">
              <dt className="text-muted-foreground text-xs">Model key</dt>
              <dd className="font-mono text-xs break-all">{review.model.modelKey}</dd>
            </div>
            <div className="min-w-0">
              <dt className="text-muted-foreground text-xs">Checkpoint sha256</dt>
              <dd className="font-mono text-xs break-all">{review.model.checkpointSha256}</dd>
            </div>
            <div className="min-w-0">
              <dt className="text-muted-foreground text-xs">Seating</dt>
              <dd className="text-xs tabular-nums">
                AI in slot {review.assignment.aiSlot} · player {review.assignment.startingPlayer}{" "}
                started
              </dd>
            </div>
            <div className="min-w-0">
              <dt className="text-muted-foreground text-xs">Battle seed</dt>
              <dd className="font-mono text-xs tabular-nums">{review.assignment.battleSeed}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <MatchReviewDraft
        draft={review.draft}
        assignment={review.assignment}
        cardsByCode={cardsByCode}
      />

      <MatchReviewAnnotation
        annotation={review.annotation}
        actualModelDisplayName={review.model.displayName}
      />

      <MatchReviewTimeline
        actions={review.actions}
        gameLogs={review.gameLogs}
        annotation={review.annotation}
      />
    </div>
  );
}
