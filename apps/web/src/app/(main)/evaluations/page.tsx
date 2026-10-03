import { Suspense } from "react";
import Link from "next/link";
import { redirect } from "next/navigation";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EvaluationStartForm } from "@/components/evaluations/EvaluationStartForm";
import { EvaluationSessionsSkeleton } from "@/components/evaluations/EvaluationSessionsSkeleton";
import { EVALUATION_PROTOCOL } from "@/lib/evaluations/client";
import { getServerUser } from "@/lib/auth/getServerUser";
import { getUserDecks } from "@tcg/backend-core/services/DeckService";
import { listHumanEvaluationSessionsForReviewer } from "@tcg/backend-core/services/humanEvaluationService";
import type { DeckSummary } from "@tcg/backend-core/types/deck";
import type { HumanEvaluationSessionSummary } from "@tcg/backend-core/types/humanEvaluations";

export default function EvaluationsPage() {
  return (
    <>
      <div className="mb-8 max-w-3xl">
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold">Blind evaluation</h1>
          <Badge variant="outline" className="font-mono text-xs">
            {EVALUATION_PROTOCOL}
          </Badge>
        </div>
        <p className="text-muted-foreground mt-2">
          Play a fixed deck against a scheduled series of anonymous opponents, rate every match
          before you learn who you played, and only then reveal the assignment.
        </p>
      </div>
      <Suspense fallback={<EvaluationSessionsSkeleton />}>
        <EvaluationsContent />
      </Suspense>
    </>
  );
}

async function EvaluationsContent() {
  const user = await getServerUser();

  if (!user) {
    redirect("/login");
  }

  let decks: DeckSummary[] = [];
  let sessions: HumanEvaluationSessionSummary[] = [];
  let error: string | null = null;

  try {
    [decks, sessions] = await Promise.all([
      getUserDecks(user.id),
      listHumanEvaluationSessionsForReviewer(user.id),
    ]);
  } catch (loadError) {
    error = loadError instanceof Error ? loadError.message : "Failed to load evaluation data";
  }

  const activeSessions = sessions.filter((session) => session.revealedAt === null);
  const revealedSessions = sessions.filter((session) => session.revealedAt !== null);

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <div className="space-y-8">
        {error ? (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {activeSessions.length > 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>Sessions in progress</CardTitle>
              <CardDescription>
                Finish an open session before starting another — mixing sessions dilutes the
                comparison and makes fatigue harder to read out of the data.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {activeSessions.map((session) => (
                <div
                  key={session.id}
                  className="flex flex-wrap items-center justify-between gap-4 rounded-lg border px-4 py-3"
                >
                  <div className="min-w-0">
                    <p className="text-sm font-medium tabular-nums">
                      {session.completedMatches} of {session.totalMatches} matches played ·{" "}
                      {session.annotatedMatches} annotated
                    </p>
                    <p className="text-muted-foreground mt-1 text-xs">
                      Started {new Date(session.createdAt).toLocaleString()} ·{" "}
                      {session.gamesPerModel} games per model
                    </p>
                  </div>
                  <Button asChild size="sm">
                    <Link href={`/evaluations/${session.id}`}>Resume session</Link>
                  </Button>
                </div>
              ))}
            </CardContent>
          </Card>
        ) : null}

        <EvaluationStartForm decks={decks} />
      </div>

      <div className="space-y-8">
        <Card>
          <CardHeader>
            <CardTitle>How to keep it blind</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-muted-foreground space-y-3 text-sm">
              <li>
                <span className="text-foreground font-medium">
                  Opponents are labelled, not named.
                </span>{" "}
                Labels are stable inside one session and carry no ordering information about
                training progress.
              </li>
              <li>
                <span className="text-foreground font-medium">Annotate immediately.</span> Each
                match asks for ratings the moment it ends, before any identity exists in the UI.
              </li>
              <li>
                <span className="text-foreground font-medium">Reveal is one-way.</span> Once you
                reveal a session its annotations are frozen and cannot be edited.
              </li>
              <li>
                <span className="text-foreground font-medium">Stop when you are tired.</span>{" "}
                Sessions resume exactly where you left them. A rushed sixteenth game is worse data
                than no game.
              </li>
            </ul>
          </CardContent>
        </Card>

        {revealedSessions.length > 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>Revealed sessions</CardTitle>
              <CardDescription>Frozen results, open for inspection.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {revealedSessions.map((session) => (
                <Link
                  key={session.id}
                  href={`/evaluations/${session.id}`}
                  className="hover:bg-accent flex items-center justify-between gap-4 rounded-md border px-3 py-2 transition-colors"
                >
                  <span className="min-w-0">
                    <span className="block text-sm font-medium tabular-nums">
                      {session.totalMatches} matches · {session.gamesPerModel}/model
                    </span>
                    <span className="text-muted-foreground text-xs">
                      Revealed{" "}
                      {session.revealedAt === null
                        ? "—"
                        : new Date(session.revealedAt).toLocaleDateString()}
                    </span>
                  </span>
                  <Badge variant="secondary">Revealed</Badge>
                </Link>
              ))}
            </CardContent>
          </Card>
        ) : null}
      </div>
    </div>
  );
}
