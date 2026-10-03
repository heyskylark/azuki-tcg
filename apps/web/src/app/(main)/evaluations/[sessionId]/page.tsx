import { Suspense } from "react";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { EvaluationSessionDashboard } from "@/components/evaluations/EvaluationSessionDashboard";
import { EvaluationSessionsSkeleton } from "@/components/evaluations/EvaluationSessionsSkeleton";
import { getServerUser } from "@/lib/auth/getServerUser";
import { getUserDecks } from "@tcg/backend-core/services/DeckService";
import { getHumanEvaluationSession } from "@tcg/backend-core/services/humanEvaluationService";
import { ApiError } from "@tcg/backend-core/errors";
import type { DeckSummary } from "@tcg/backend-core/types/deck";
import type { HumanEvaluationSessionSummary } from "@tcg/backend-core/types/humanEvaluations";

interface PageProps {
  params: Promise<{ sessionId: string }>;
}

const uuidSchema = z.uuid();

/** Returns null when the session does not exist or is not this reviewer's. */
async function loadSession(
  sessionId: string,
  reviewerId: string
): Promise<{ session: HumanEvaluationSessionSummary; decks: DeckSummary[] } | null> {
  try {
    const [session, decks] = await Promise.all([
      getHumanEvaluationSession(sessionId, reviewerId),
      getUserDecks(reviewerId),
    ]);

    return { session, decks };
  } catch (loadError) {
    if (loadError instanceof ApiError && (loadError.status === 404 || loadError.status === 403)) {
      return null;
    }

    throw loadError;
  }
}

export default async function EvaluationSessionPage({ params }: PageProps) {
  const { sessionId } = await params;

  return (
    <>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Evaluation session</h1>
          <p className="text-muted-foreground mt-2">
            One deck, a fixed schedule of anonymous opponents, an annotation for every match.
          </p>
        </div>
        <Button variant="outline" asChild>
          <Link href="/evaluations">All sessions</Link>
        </Button>
      </div>
      <Suspense fallback={<EvaluationSessionsSkeleton />}>
        <EvaluationSessionContent sessionId={sessionId} />
      </Suspense>
    </>
  );
}

async function EvaluationSessionContent({ sessionId }: { sessionId: string }) {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  const parsedSessionId = uuidSchema.safeParse(sessionId);

  if (!parsedSessionId.success) {
    notFound();
  }

  const loaded = await loadSession(parsedSessionId.data, user.id);

  if (loaded === null) {
    notFound();
  }

  const { session, decks } = loaded;
  const humanDeck = decks.find((deck) => deck.id === session.humanDeckId);

  return (
    <EvaluationSessionDashboard
      initialSession={session}
      humanDeckName={humanDeck?.name ?? "Deleted deck"}
    />
  );
}
