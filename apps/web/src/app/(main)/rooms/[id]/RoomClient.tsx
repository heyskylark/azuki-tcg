"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useRoom } from "@/contexts/RoomContext";
import { AssetProvider } from "@/contexts/AssetContext";
import { GameStateProvider } from "@/contexts/GameStateContext";
import { GameBridge } from "@/components/game/GameBridge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ConnectionIndicator } from "@/app/(main)/rooms/[id]/components/ConnectionIndicator";
import { PasswordPrompt } from "@/app/(main)/rooms/[id]/components/PasswordPrompt";
import { WaitingForPlayers } from "@/app/(main)/rooms/[id]/components/WaitingForPlayers";
import { DeckSelection } from "@/app/(main)/rooms/[id]/components/DeckSelection";
import { ReadyCheck } from "@/app/(main)/rooms/[id]/components/ReadyCheck";
import { InMatchView } from "@/app/(main)/rooms/[id]/components/InMatchView";
import type { AuthenticatedUser } from "@tcg/backend-core/types/auth";

export interface RoomData {
  id: string;
  status: string;
  type: string;
  hasPassword: boolean;
  player0Id: string;
  player1Id: string | null;
  createdAt: string;
  updatedAt: string;
}

interface RoomClientProps {
  initialRoom: RoomData;
  user: AuthenticatedUser;
}

const INACTIVE_ROOM_STATUSES = ["COMPLETED", "CLOSED", "ABORTED"];

export function RoomClient({ initialRoom, user }: RoomClientProps) {
  const {
    activeRoom,
    roomState,
    connectionStatus,
    error,
    join,
    send,
    clearError,
  } = useRoom();

  const isInRoom =
    initialRoom.player0Id === user.id || initialRoom.player1Id === user.id;
  const isOwner = initialRoom.player0Id === user.id;
  const isRoomInactive = INACTIVE_ROOM_STATUSES.includes(initialRoom.status);
  const playerSlot = activeRoom?.playerSlot ?? null;

  const [needsPassword, setNeedsPassword] = useState(false);
  const [isJoining, setIsJoining] = useState(false);
  const hasAttemptedJoin = useRef(false);

  // Auto-join on mount if conditions are right
  useEffect(() => {
    if (hasAttemptedJoin.current) {
      return;
    }

    // Don't re-process if already showing password prompt
    if (needsPassword) {
      return;
    }

    // Don't join inactive rooms
    if (isRoomInactive) {
      return;
    }

    // If already connected to this room, don't rejoin
    if (activeRoom?.id === initialRoom.id && connectionStatus === "connected") {
      return;
    }

    // If already connecting or joining, wait
    if (connectionStatus === "connecting" || isJoining) {
      return;
    }

    const doJoin = async () => {
      hasAttemptedJoin.current = true;

      if (isInRoom) {
        // User is already in the room, auto-join without password
        setIsJoining(true);
        await join(initialRoom.id);
        setIsJoining(false);
      } else if (!initialRoom.hasPassword) {
        // Room doesn't require password, auto-join
        setIsJoining(true);
        await join(initialRoom.id);
        setIsJoining(false);
      } else {
        // Room requires password and user is not in room
        setNeedsPassword(true);
      }
    };

    doJoin();
  }, [isInRoom, initialRoom.hasPassword, initialRoom.id, join, connectionStatus, activeRoom?.id, isRoomInactive, isJoining, needsPassword]);

  // Derive connection state for UI
  type ConnectionState = "idle" | "joining" | "connecting" | "connected" | "error" | "inactive";
  const connectionState: ConnectionState = isRoomInactive
    ? "inactive"
    : isJoining
      ? "joining"
      : connectionStatus === "connecting"
        ? "connecting"
        : connectionStatus === "connected" && activeRoom?.id === initialRoom.id
          ? "connected"
          : connectionStatus === "error" || error
            ? "error"
            : "idle";

  // Track locally selected deck (before server confirmation)
  const [selectedDeckId, setSelectedDeckId] = useState<string | null>(null);

  const handleSelectDeck = useCallback(
    (deckId: string) => {
      setSelectedDeckId(deckId);
      send({ type: "SELECT_DECK", deckId });
    },
    [send]
  );

  const handleReady = useCallback(
    (ready: boolean) => {
      send({ type: "READY", ready });
    },
    [send]
  );

  const handleJoinWithPassword = useCallback(
    async (password: string) => {
      setIsJoining(true);
      const success = await join(initialRoom.id, password);
      setIsJoining(false);
      if (success) {
        setNeedsPassword(false);
      }
    },
    [join, initialRoom.id]
  );

  // Determine current display status
  const displayStatus = roomState?.status ?? initialRoom.status;

  // Render password prompt if needed
  if (needsPassword) {
    return (
      <PasswordPrompt
        roomId={initialRoom.id}
        onSubmit={handleJoinWithPassword}
        isLoading={connectionState === "joining"}
        error={error}
      />
    );
  }

  // Render loading state while joining/connecting
  if (connectionState === "joining" || connectionState === "connecting") {
    return (
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Connecting to Room...</CardTitle>
          <CardDescription>Please wait while we connect you to the game.</CardDescription>
        </CardHeader>
        <CardContent className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
        </CardContent>
      </Card>
    );
  }

  // Render error state
  if (connectionState === "error" && !roomState) {
    return (
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Connection Error</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert variant="destructive">
            <AlertDescription>{error || "Failed to connect to room"}</AlertDescription>
          </Alert>
          <Button onClick={() => join(initialRoom.id)}>Try Again</Button>
        </CardContent>
      </Card>
    );
  }

  // Render inactive room state (no WebSocket connection needed)
  if (connectionState === "inactive") {
    return (
      <Card className="max-w-4xl mx-auto">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Room</CardTitle>
              <CardDescription>Room ID: {initialRoom.id}</CardDescription>
            </div>
            <Badge variant="outline">
              {initialRoom.status.replace(/_/g, " ")}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="text-2xl font-bold mb-4">Room Ended</div>
            <p className="text-muted-foreground">
              This room is no longer active ({initialRoom.status.toLowerCase()}).
            </p>
            <Button className="mt-4" onClick={() => window.location.href = "/dashboard"}>
              Back to Dashboard
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Render phase-specific content
  const renderContent = () => {
    // Show error if present (but still connected)
    const errorAlert = error && (
      <Alert variant="destructive" className="mb-4">
        <AlertDescription className="flex items-center justify-between">
          {error}
          <Button variant="ghost" size="sm" onClick={clearError}>
            Dismiss
          </Button>
        </AlertDescription>
      </Alert>
    );

    // If we don't have room state yet, show loading
    if (!roomState) {
      return (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
        </div>
      );
    }

    switch (displayStatus) {
      case "WAITING_FOR_PLAYERS":
        return (
          <>
            {errorAlert}
            <WaitingForPlayers
              roomId={initialRoom.id}
              roomState={roomState}
              userId={user.id}
              isOwner={isOwner}
              hasPassword={initialRoom.hasPassword}
            />
          </>
        );

      case "DECK_SELECTION":
        if (playerSlot === null) {
          return <div>Loading player info...</div>;
        }
        return (
          <>
            {errorAlert}
            <DeckSelection
              roomState={roomState}
              userId={user.id}
              playerSlot={playerSlot}
              selectedDeckId={selectedDeckId}
              onSelectDeck={handleSelectDeck}
              onReady={handleReady}
            />
          </>
        );

      case "READY_CHECK":
        if (playerSlot === null) {
          return <div>Loading player info...</div>;
        }
        return (
          <>
            {errorAlert}
            <ReadyCheck
              roomState={roomState}
              userId={user.id}
              playerSlot={playerSlot}
              onUnready={() => handleReady(false)}
            />
          </>
        );

      case "STARTING":
        return (
          <div className="text-center py-8">
            <div className="text-4xl font-bold text-primary mb-4">Game Starting!</div>
            <p className="text-muted-foreground">Initializing game world...</p>
          </div>
        );

      case "IN_MATCH":
        // Handled separately - returns full-screen view
        return null;

      case "COMPLETED":
      case "ABORTED":
      case "CLOSED":
        return (
          <div className="text-center py-8">
            <div className="text-2xl font-bold mb-4">Room Ended</div>
            <p className="text-muted-foreground">
              This room is no longer active ({displayStatus.toLowerCase()}).
            </p>
            <Button className="mt-4" onClick={() => window.location.href = "/dashboard"}>
              Back to Dashboard
            </Button>
          </div>
        );

      default:
        return (
          <div className="text-center py-8">
            <p className="text-muted-foreground">Unknown room state: {displayStatus}</p>
          </div>
        );
    }
  };

  // For IN_MATCH status, render full-screen view without Card wrapper
  if (displayStatus === "IN_MATCH" && roomState) {
    if (playerSlot === null) {
      return <div>Loading player info...</div>;
    }
    return (
      <AssetProvider>
        <GameStateProvider>
          <GameBridge playerSlot={playerSlot}>
            <InMatchView />
          </GameBridge>
        </GameStateProvider>
      </AssetProvider>
    );
  }

  return (
    <Card className="max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Room</CardTitle>
            <CardDescription>Room ID: {initialRoom.id}</CardDescription>
          </div>
          <div className="flex items-center gap-4">
            <ConnectionIndicator status={connectionStatus} />
            <div className="flex gap-2">
              <Badge
                variant={
                  displayStatus === "WAITING_FOR_PLAYERS"
                    ? "default"
                    : displayStatus === "DECK_SELECTION" || displayStatus === "READY_CHECK"
                    ? "secondary"
                    : "outline"
                }
              >
                {displayStatus.replace(/_/g, " ")}
              </Badge>
              {initialRoom.hasPassword && <Badge variant="outline">Private</Badge>}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>{renderContent()}</CardContent>
    </Card>
  );
}
