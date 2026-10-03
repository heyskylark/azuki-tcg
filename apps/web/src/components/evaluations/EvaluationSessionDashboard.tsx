"use client";

import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Check, Lock, X } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MatchAnnotationForm } from "@/components/evaluations/MatchAnnotationForm";
import { MatchOutcomeBadge, MatchStatusBadge } from "@/components/evaluations/MatchStatusBadges";
import {
  claimNextEvaluationMatch,
  EVALUATION_PROTOCOL,
  fetchEvaluationSession,
  needsAnnotation,
  revealEvaluationSession,
  type EvaluationSessionMatch,
  type EvaluationSessionSummary,
} from "@/lib/evaluations/client";
import { HumanEvaluationMatchStatus } from "@tcg/backend-core/types/humanEvaluations";

interface EvaluationSessionDashboardProps {
  initialSession: EvaluationSessionSummary;
  humanDeckName: string;
}

interface RevealedModelRecord {
  displayName: string;
  checkpointSha256: string;
  wins: number;
  losses: number;
  draws: number;
  aborted: number;
}

export function EvaluationSessionDashboard({
  initialSession,
  humanDeckName,
}: EvaluationSessionDashboardProps) {
  const router = useRouter();
  const [session, setSession] = useState(initialSession);
  const [annotatingMatchId, setAnnotatingMatchId] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<"next" | "reveal" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isRevealed = session.revealedAt !== null;

  const openMatch = session.matches.find(
    (match) =>
      match.roomId !== null &&
      (match.status === HumanEvaluationMatchStatus.CLAIMED ||
        match.status === HumanEvaluationMatchStatus.IN_PROGRESS)
  );
  const unannotatedMatch = session.matches.find(needsAnnotation);
  const nextScheduledMatch = session.matches.find(
    (match) => match.status === HumanEvaluationMatchStatus.SCHEDULED
  );
  const abortedCount = session.matches.filter(
    (match) => match.status === HumanEvaluationMatchStatus.ABORTED
  ).length;

  const totalMatches = Math.max(1, session.totalMatches);
  const playedProgress = Math.round((session.completedMatches / totalMatches) * 100);
  const annotatedProgress = Math.round((session.annotatedMatches / totalMatches) * 100);

  const refresh = useCallback(async () => {
    try {
      setSession(await fetchEvaluationSession(session.id));
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : "Failed to refresh session");
    }
  }, [session.id]);

  const handleStartNext = async () => {
    setError(null);
    setPendingAction("next");

    try {
      const match = await claimNextEvaluationMatch(session.id);
      router.push(`/rooms/${match.roomId}`);
    } catch (nextError) {
      setPendingAction(null);
      setError(nextError instanceof Error ? nextError.message : "Failed to open the next match");
    }
  };

  const handleReveal = async () => {
    setError(null);
    setPendingAction("reveal");

    try {
      setSession(await revealEvaluationSession(session.id));
    } catch (revealError) {
      setError(revealError instanceof Error ? revealError.message : "Failed to reveal");
    } finally {
      setPendingAction(null);
    }
  };

  const revealedRecords = useMemo(() => {
    const records = new Map<string, RevealedModelRecord>();

    for (const match of session.matches) {
      if (match.revealed === null) {
        continue;
      }

      const key = match.revealed.checkpointSha256;
      const record = records.get(key) ?? {
        displayName: match.revealed.modelDisplayName,
        checkpointSha256: match.revealed.checkpointSha256,
        wins: 0,
        losses: 0,
        draws: 0,
        aborted: 0,
      };

      if (match.outcome === "WIN") {
        record.wins += 1;
      } else if (match.outcome === "LOSS") {
        record.losses += 1;
      } else if (match.outcome === "DRAW") {
        record.draws += 1;
      } else if (match.outcome === "ABORTED") {
        record.aborted += 1;
      }

      records.set(key, record);
    }

    return [...records.values()].sort((left, right) => right.wins - left.wins);
  }, [session.matches]);

  const annotationTarget =
    annotatingMatchId === null
      ? null
      : (session.matches.find((match) => match.matchId === annotatingMatchId) ?? null);

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle>Session progress</CardTitle>
                <Badge variant="outline" className="font-mono text-xs">
                  {EVALUATION_PROTOCOL}
                </Badge>
                {isRevealed ? <Badge variant="secondary">Revealed</Badge> : null}
              </div>
              <CardDescription className="mt-2">
                Playing <span className="text-foreground font-medium">{humanDeckName}</span> ·{" "}
                {session.gamesPerModel} games per model · started{" "}
                {new Date(session.createdAt).toLocaleString()}
              </CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {openMatch ? (
                <Button asChild>
                  <Link href={`/rooms/${openMatch.roomId}`}>Resume match {openMatch.ordinal}</Link>
                </Button>
              ) : nextScheduledMatch ? (
                <Button
                  onClick={handleStartNext}
                  disabled={pendingAction !== null || unannotatedMatch !== undefined}
                >
                  {pendingAction === "next"
                    ? "Opening room..."
                    : `Start match ${nextScheduledMatch.ordinal} of ${session.totalMatches}`}
                </Button>
              ) : null}
              <Button variant="outline" onClick={refresh} disabled={pendingAction !== null}>
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="grid gap-6 sm:grid-cols-2">
          <ProgressMeter
            label="Matches played"
            current={session.completedMatches}
            total={session.totalMatches}
            percent={playedProgress}
          />
          <ProgressMeter
            label="Matches annotated"
            current={session.annotatedMatches}
            total={session.totalMatches}
            percent={annotatedProgress}
          />
        </CardContent>
      </Card>

      {error ? (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {unannotatedMatch ? (
        <Alert>
          <AlertTitle>Match {unannotatedMatch.ordinal} still needs your annotation</AlertTitle>
          <AlertDescription>
            <p>
              Annotations are collected while the opponent is still anonymous, so the next match
              stays locked until this one is rated.
            </p>
            <Button
              size="sm"
              className="mt-2"
              onClick={() => setAnnotatingMatchId(unannotatedMatch.matchId)}
            >
              Annotate match {unannotatedMatch.ordinal}
            </Button>
          </AlertDescription>
        </Alert>
      ) : null}

      {abortedCount > 0 ? (
        <Alert>
          <AlertTitle>
            {abortedCount === 1
              ? "One match ended in a technical abort"
              : `${abortedCount} matches ended in a technical abort`}
          </AlertTitle>
          <AlertDescription>
            An abort means the room was torn down before a result was recorded — a disconnect, an
            engine failure, or a deck that could not be generated. It is not scored as a win or a
            loss, but it still needs an annotation so the protocol stays complete: say what you saw
            before it broke.
          </AlertDescription>
        </Alert>
      ) : null}

      {annotationTarget ? (
        <Card>
          <CardHeader>
            <CardTitle>Annotate match {annotationTarget.ordinal}</CardTitle>
            <CardDescription>
              {annotationTarget.opponentLabel} ·{" "}
              {annotationTarget.status === HumanEvaluationMatchStatus.ABORTED
                ? "Technical abort — rate what you managed to see"
                : "Rate the match before any identity is revealed"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <MatchAnnotationForm
              matchId={annotationTarget.matchId}
              opponentLabel={annotationTarget.opponentLabel}
              onSubmitted={() => {
                setAnnotatingMatchId(null);
                void refresh();
              }}
              onCancel={() => setAnnotatingMatchId(null)}
            />
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Schedule</CardTitle>
          <CardDescription>
            {isRevealed
              ? "Every match with its assigned checkpoint, frozen at reveal."
              : "Opponent labels are stable within this session and reveal nothing about which checkpoint you are facing."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-muted-foreground border-b text-left text-xs">
                  <th scope="col" className="py-2 pr-4 font-medium">
                    #
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Opponent
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Status
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Result
                  </th>
                  <th scope="col" className="py-2 pr-4 font-medium">
                    Annotation
                  </th>
                  {isRevealed ? (
                    <th scope="col" className="py-2 pr-4 font-medium">
                      Checkpoint
                    </th>
                  ) : null}
                  <th scope="col" className="py-2 font-medium">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {session.matches.map((match) => (
                  <tr key={match.matchId} className="border-b last:border-0">
                    <td className="py-3 pr-4 tabular-nums">{match.ordinal}</td>
                    <td className="py-3 pr-4 font-medium">{match.opponentLabel}</td>
                    <td className="py-3 pr-4">
                      <MatchStatusBadge status={match.status} />
                    </td>
                    <td className="py-3 pr-4">
                      <MatchOutcomeBadge outcome={match.outcome} />
                    </td>
                    <td className="py-3 pr-4">
                      {match.hasAnnotation ? (
                        <span className="inline-flex items-center gap-1.5 text-xs text-emerald-700">
                          <Check className="size-3.5" />
                          Saved
                        </span>
                      ) : needsAnnotation(match) ? (
                        <span className="inline-flex items-center gap-1.5 text-xs text-amber-700">
                          <X className="size-3.5" />
                          Required
                        </span>
                      ) : (
                        <span className="text-muted-foreground text-xs">—</span>
                      )}
                    </td>
                    {isRevealed ? (
                      <td className="py-3 pr-4">
                        {match.revealed === null ? (
                          <span className="text-muted-foreground text-xs">—</span>
                        ) : (
                          <span className="block">
                            <span className="block text-sm">{match.revealed.modelDisplayName}</span>
                            <span
                              className="text-muted-foreground font-mono text-[11px]"
                              title={match.revealed.checkpointSha256}
                            >
                              {match.revealed.checkpointSha256.slice(0, 12)}
                            </span>
                          </span>
                        )}
                      </td>
                    ) : null}
                    <td className="py-3">
                      <MatchRowAction
                        match={match}
                        sessionId={session.id}
                        isRevealed={isRevealed}
                        onAnnotate={() => setAnnotatingMatchId(match.matchId)}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {isRevealed ? (
        <Card>
          <CardHeader>
            <CardTitle>Revealed assignment</CardTitle>
            <CardDescription>
              Your record against each checkpoint in this session. Aborted matches are counted
              separately and never scored.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-muted-foreground border-b text-left text-xs">
                    <th scope="col" className="py-2 pr-4 font-medium">
                      Model
                    </th>
                    <th scope="col" className="py-2 pr-4 font-medium">
                      Checkpoint sha256
                    </th>
                    <th scope="col" className="py-2 pr-4 font-medium">
                      You won
                    </th>
                    <th scope="col" className="py-2 pr-4 font-medium">
                      You lost
                    </th>
                    <th scope="col" className="py-2 pr-4 font-medium">
                      Drawn
                    </th>
                    <th scope="col" className="py-2 font-medium">
                      Aborted
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {revealedRecords.map((record) => (
                    <tr key={record.checkpointSha256} className="border-b last:border-0">
                      <td className="py-3 pr-4 font-medium">{record.displayName}</td>
                      <td className="text-muted-foreground py-3 pr-4 font-mono text-xs break-all">
                        {record.checkpointSha256}
                      </td>
                      <td className="py-3 pr-4 tabular-nums">{record.wins}</td>
                      <td className="py-3 pr-4 tabular-nums">{record.losses}</td>
                      <td className="py-3 pr-4 tabular-nums">{record.draws}</td>
                      <td className="py-3 tabular-nums">{record.aborted}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lock className="size-4" />
              Reveal the assignment
            </CardTitle>
            <CardDescription>
              Reveal is irreversible and only unlocks once every scheduled match has been played and
              annotated.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <ul className="space-y-2 text-sm">
              <RequirementRow
                met={session.completedMatches >= session.totalMatches}
                label={`All ${session.totalMatches} matches played`}
                detail={`${session.completedMatches} of ${session.totalMatches} finished`}
              />
              <RequirementRow
                met={session.annotatedMatches >= session.totalMatches}
                label={`All ${session.totalMatches} matches annotated`}
                detail={`${session.annotatedMatches} of ${session.totalMatches} annotated`}
              />
            </ul>
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={handleReveal}
                disabled={!session.revealReady || pendingAction !== null}
              >
                {pendingAction === "reveal" ? "Revealing..." : "Reveal checkpoints"}
              </Button>
              {session.revealReady ? (
                <p className="text-muted-foreground text-xs">
                  Your annotations are already stored and will not change.
                </p>
              ) : (
                <p className="text-muted-foreground text-xs">
                  Locked until the protocol is complete.
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Protocol</CardTitle>
          <CardDescription>
            What makes this session usable as evidence — and what quietly ruins it.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 md:grid-cols-2">
          <ul className="text-muted-foreground space-y-3 text-sm">
            <li>
              <span className="text-foreground font-medium">The schedule is fixed.</span> Every
              model gets the same number of games, the same gate and leader contexts, and a balanced
              share of going first. You cannot pick or re-roll an opponent.
            </li>
            <li>
              <span className="text-foreground font-medium">One deck, all session.</span> Your deck
              is the constant that makes the comparison mean something.
            </li>
            <li>
              <span className="text-foreground font-medium">Annotate at the table.</span> Ratings
              are captured the moment a match ends, while the opponent is still just a label.
            </li>
          </ul>
          <ul className="text-muted-foreground space-y-3 text-sm">
            <li>
              <span className="text-foreground font-medium">Fatigue is a real bias.</span> Long runs
              drift toward generous ratings and lazier play. Stop at any point; the session resumes
              exactly here.
            </li>
            <li>
              <span className="text-foreground font-medium">Forfeits count as losses.</span> If a
              match becomes unplayable, forfeit and say so in the annotation rather than stalling.
            </li>
            <li>
              <span className="text-foreground font-medium">Reveal is the end.</span> After reveal,
              annotations are frozen and each match opens a full draft-and-decision review.
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}

function ProgressMeter({
  label,
  current,
  total,
  percent,
}: {
  label: string;
  current: number;
  total: number;
  percent: number;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-muted-foreground text-sm tabular-nums">
          {current} / {total}
        </span>
      </div>
      <div
        role="progressbar"
        aria-valuenow={current}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-label={label}
        className="bg-secondary mt-2 h-1.5 w-full overflow-hidden rounded-full"
      >
        <div
          className="bg-primary h-full rounded-full transition-[width] duration-500"
          style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
        />
      </div>
    </div>
  );
}

function RequirementRow({ met, label, detail }: { met: boolean; label: string; detail: string }) {
  return (
    <li className="flex items-start gap-2">
      {met ? (
        <Check className="mt-0.5 size-4 shrink-0 text-emerald-600" />
      ) : (
        <X className="text-muted-foreground mt-0.5 size-4 shrink-0" />
      )}
      <span>
        <span className={met ? "font-medium" : "text-muted-foreground font-medium"}>{label}</span>
        <span className="text-muted-foreground block text-xs tabular-nums">{detail}</span>
      </span>
    </li>
  );
}

function MatchRowAction({
  match,
  sessionId,
  isRevealed,
  onAnnotate,
}: {
  match: EvaluationSessionMatch;
  sessionId: string;
  isRevealed: boolean;
  onAnnotate: () => void;
}) {
  if (needsAnnotation(match)) {
    return (
      <Button size="sm" variant="outline" onClick={onAnnotate}>
        Annotate
      </Button>
    );
  }

  if (
    match.roomId !== null &&
    (match.status === HumanEvaluationMatchStatus.CLAIMED ||
      match.status === HumanEvaluationMatchStatus.IN_PROGRESS)
  ) {
    return (
      <Button size="sm" variant="outline" asChild>
        <Link href={`/rooms/${match.roomId}`}>Resume</Link>
      </Button>
    );
  }

  if (
    isRevealed &&
    (match.status === HumanEvaluationMatchStatus.COMPLETED ||
      match.status === HumanEvaluationMatchStatus.ABORTED)
  ) {
    return (
      <Button size="sm" variant="ghost" asChild>
        <Link href={`/evaluations/${sessionId}/matches/${match.matchId}`}>Review</Link>
      </Button>
    );
  }

  return null;
}
