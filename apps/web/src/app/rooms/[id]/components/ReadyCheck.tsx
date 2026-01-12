"use client";

import { useCountdown } from "@/hooks/useCountdown";
import { Button } from "@/components/ui/button";
import { PlayerCard } from "@/app/rooms/[id]/components/PlayerCard";
import type { RoomStateMessage } from "@tcg/backend-core/types/ws";

interface ReadyCheckProps {
  roomState: RoomStateMessage;
  userId: string;
  playerSlot: 0 | 1;
  onUnready: () => void;
}

export function ReadyCheck({
  roomState,
  userId,
  playerSlot,
  onUnready,
}: ReadyCheckProps) {
  const { secondsRemaining } = useCountdown(roomState.readyCountdownEnd);

  return (
    <div className="space-y-6">
      {/* Countdown display */}
      <div className="text-center py-8">
        <p className="text-sm text-muted-foreground mb-2">Game starting in</p>
        <div className="text-6xl font-bold text-primary">
          {secondsRemaining}
        </div>
        <p className="text-sm text-muted-foreground mt-4">
          Both players are ready!
        </p>
      </div>

      {/* Player cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <PlayerCard
          player={roomState.players[0]}
          isCurrentUser={roomState.players[0]?.id === userId}
          playerLabel="Player 1 (Host)"
          showReadyStatus
        />
        <PlayerCard
          player={roomState.players[1]}
          isCurrentUser={roomState.players[1]?.id === userId}
          playerLabel="Player 2"
          showReadyStatus
        />
      </div>

      {/* Unready button */}
      <div className="flex justify-center">
        <Button variant="secondary" size="lg" onClick={onUnready}>
          Cancel (Go back to deck selection)
        </Button>
      </div>
    </div>
  );
}
