"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EVALUATION_PROTOCOL } from "@/lib/evaluations/client";
import type { RoomEvaluationMetadata, RoomStateMessage } from "@tcg/backend-core/types/ws";

interface EvaluationRoomLobbyProps {
  evaluation: RoomEvaluationMetadata;
  roomState: RoomStateMessage;
  playerSlot: 0 | 1;
}

export function EvaluationRoomLobby({
  evaluation,
  roomState,
  playerSlot,
}: EvaluationRoomLobbyProps) {
  const you = roomState.players[playerSlot];
  const opponent = roomState.players[playerSlot === 0 ? 1 : 0];
  const isOpponentDrafting = opponent !== null && !opponent.deckSelected;
  const isStarting = roomState.status === "STARTING" || roomState.status === "READY_CHECK";

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle>
                  Match {evaluation.ordinal} of {evaluation.totalMatches}
                </CardTitle>
                <Badge variant="outline" className="font-mono text-xs">
                  {EVALUATION_PROTOCOL}
                </Badge>
                <Badge variant="secondary">Blind</Badge>
              </div>
              <CardDescription className="mt-2">
                Your deck is already locked in for this session. The opponent drafts its own 50
                cards before the game starts — you will not be told which checkpoint it is until the
                whole session is annotated and revealed.
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" asChild>
              <Link href={`/evaluations/${evaluation.sessionId}`}>Session</Link>
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-lg border px-4 py-3">
            <p className="text-muted-foreground text-xs">You</p>
            <p className="mt-1 text-sm font-medium">{you?.username ?? "You"}</p>
            <p className="text-muted-foreground mt-2 text-xs">
              {you?.deckSelected
                ? "Session deck locked in · ready"
                : "Waiting for the server to assign your deck"}
            </p>
          </div>
          <div className="rounded-lg border px-4 py-3">
            <p className="text-muted-foreground text-xs">Opponent</p>
            <p className="mt-1 text-sm font-medium">{opponent?.username ?? "Assigning..."}</p>
            <p className="text-muted-foreground mt-2 flex items-center gap-2 text-xs">
              {isOpponentDrafting ? (
                <>
                  <span className="border-primary size-3 shrink-0 animate-spin rounded-full border-2 border-t-transparent" />
                  Drafting its deck, one card at a time
                </>
              ) : isStarting ? (
                "Deck built · starting the match"
              ) : (
                "Deck built"
              )}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Before you start</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="text-muted-foreground space-y-2 text-sm">
            <li>
              Play it as a real game. Conceding early or stalling makes the match unusable as
              evidence.
            </li>
            <li>
              The moment it ends you will be asked to rate the opponent — the ratings are collected
              while it is still anonymous.
            </li>
            <li>
              If you are too tired to play properly, leave now and resume from the session page. A
              rushed match is worse than a missing one.
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
