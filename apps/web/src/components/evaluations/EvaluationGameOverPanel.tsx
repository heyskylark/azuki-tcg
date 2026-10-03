"use client";

import { useEffect, useState } from "react";
import { Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MatchAnnotationForm } from "@/components/evaluations/MatchAnnotationForm";
import { useRoom } from "@/contexts/RoomContext";
import { EVALUATION_PROTOCOL, fetchEvaluationMatch } from "@/lib/evaluations/client";
import type { RoomEvaluationMetadata } from "@tcg/backend-core/types/ws";

interface EvaluationGameOverPanelProps {
  outcome: "win" | "lose" | "draw";
  reason?: string | null;
  evaluation: RoomEvaluationMetadata;
  onContinue: () => void;
}

const OUTCOME_TITLES: Record<"win" | "lose" | "draw", string> = {
  win: "You won",
  lose: "You lost",
  draw: "Draw",
};

export function EvaluationGameOverPanel({
  outcome,
  reason,
  evaluation,
  onContinue,
}: EvaluationGameOverPanelProps) {
  const { activeRoom, roomState } = useRoom();
  const [isSaved, setIsSaved] = useState(false);

  const opponentSlot = activeRoom?.playerSlot === 0 ? 1 : 0;
  const opponentLabel = roomState?.players[opponentSlot]?.username ?? "Your opponent";
  const isLastMatch = evaluation.ordinal >= evaluation.totalMatches;

  // A reload or a replayed game-over must not offer a second annotation for a
  // match that is already annotated: the API rejects it and the reviewer would
  // lose the typed text.
  useEffect(() => {
    let isCancelled = false;

    const checkExistingAnnotation = async () => {
      try {
        const match = await fetchEvaluationMatch(evaluation.matchId);

        if (!isCancelled && match.hasAnnotation) {
          setIsSaved(true);
        }
      } catch {
        // The form stays available; submitting surfaces any real error.
      }
    };

    void checkExistingAnnotation();

    return () => {
      isCancelled = true;
    };
  }, [evaluation.matchId]);

  return (
    <div className="fixed inset-0 z-[100] overflow-y-auto bg-black/85 p-4 backdrop-blur-sm sm:p-8">
      <div className="mx-auto w-full max-w-4xl">
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <CardTitle className="text-2xl">{OUTCOME_TITLES[outcome]}</CardTitle>
                  <Badge variant="secondary">
                    Match {evaluation.ordinal} of {evaluation.totalMatches}
                  </Badge>
                  <Badge variant="outline" className="font-mono text-xs">
                    {EVALUATION_PROTOCOL}
                  </Badge>
                </div>
                <CardDescription className="mt-2">
                  {reason ? `${reason} · ` : null}
                  {opponentLabel} stays anonymous. Rate the match now — this is the only point in
                  the protocol where your impression is still uncontaminated.
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {isSaved ? (
              <div className="space-y-4">
                <p className="flex items-center gap-2 text-sm font-medium text-emerald-700">
                  <Check className="size-4" />
                  Annotation saved and frozen.
                </p>
                <p className="text-muted-foreground text-sm">
                  {isLastMatch
                    ? "That was the last scheduled match. You can reveal the assignment from the session page once every annotation is in."
                    : "The next opponent is already scheduled. Take a break first if you need one — the session resumes where you left it."}
                </p>
                <Button onClick={onContinue}>
                  {isLastMatch ? "Go to reveal" : "Back to session"}
                </Button>
              </div>
            ) : (
              <MatchAnnotationForm
                matchId={evaluation.matchId}
                opponentLabel={opponentLabel}
                submitLabel="Save annotation and continue"
                onSubmitted={() => setIsSaved(true)}
              />
            )}
          </CardContent>
        </Card>
        {isSaved ? null : (
          <p className="mt-4 text-center text-xs text-white/70">
            The next match stays locked until this annotation is saved.
          </p>
        )}
      </div>
    </div>
  );
}
