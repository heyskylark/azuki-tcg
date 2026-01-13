"use client";

import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { PlayerInfo } from "@tcg/backend-core/types/ws";

interface PlayerCardProps {
  player: PlayerInfo | null;
  isCurrentUser: boolean;
  playerLabel: string;
  showDeckStatus?: boolean;
  showReadyStatus?: boolean;
  className?: string;
}

export function PlayerCard({
  player,
  isCurrentUser,
  playerLabel,
  showDeckStatus = false,
  showReadyStatus = false,
  className,
}: PlayerCardProps) {
  const isEmpty = !player;

  return (
    <Card className={cn("relative", className)}>
      <CardContent className="pt-4">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm text-muted-foreground">{playerLabel}</p>
          {player && (
            <div
              className={cn(
                "h-2 w-2 rounded-full",
                player.connected ? "bg-green-500" : "bg-red-500"
              )}
              title={player.connected ? "Connected" : "Disconnected"}
            />
          )}
        </div>

        <p className="font-medium">
          {isEmpty ? (
            <span className="text-muted-foreground">Waiting...</span>
          ) : (
            <>
              {player.username}
              {isCurrentUser && (
                <span className="text-muted-foreground ml-1">(You)</span>
              )}
            </>
          )}
        </p>

        {player && (showDeckStatus || showReadyStatus) && (
          <div className="flex gap-2 mt-2">
            {showDeckStatus && (
              <Badge variant={player.deckSelected ? "default" : "outline"}>
                {player.deckSelected ? "Deck Selected" : "No Deck"}
              </Badge>
            )}
            {showReadyStatus && (
              <Badge variant={player.ready ? "default" : "secondary"}>
                {player.ready ? "Ready" : "Not Ready"}
              </Badge>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
