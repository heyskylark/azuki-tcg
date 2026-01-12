"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { PlayerCard } from "@/app/rooms/[id]/components/PlayerCard";
import type { RoomStateMessage } from "@tcg/backend-core/types/ws";

interface WaitingForPlayersProps {
  roomId: string;
  roomState: RoomStateMessage;
  userId: string;
  isOwner: boolean;
  hasPassword: boolean;
}

export function WaitingForPlayers({
  roomId,
  roomState,
  userId,
  isOwner,
  hasPassword,
}: WaitingForPlayersProps) {
  const router = useRouter();
  const [newPassword, setNewPassword] = useState("");
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUpdatePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsUpdating(true);
    setError(null);

    try {
      const response = await authenticatedFetch(`/api/rooms/${roomId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          password: newPassword.trim() || null,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || "Failed to update room");
      }

      setNewPassword("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update room");
    } finally {
      setIsUpdating(false);
    }
  };

  const handleCloseRoom = async () => {
    if (!confirm("Are you sure you want to close this room?")) {
      return;
    }

    setIsUpdating(true);
    setError(null);

    try {
      const response = await authenticatedFetch(`/api/rooms/${roomId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || "Failed to close room");
      }

      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to close room");
      setIsUpdating(false);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="text-center py-4">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <svg
            className="w-8 h-8 text-muted-foreground animate-spin"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        </div>
        <h3 className="text-lg font-semibold">Waiting for opponent...</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Share the room link with a friend to start playing.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <PlayerCard
          player={roomState.players[0]}
          isCurrentUser={roomState.players[0]?.id === userId}
          playerLabel="Player 1 (Host)"
        />
        <PlayerCard
          player={roomState.players[1]}
          isCurrentUser={roomState.players[1]?.id === userId}
          playerLabel="Player 2"
        />
      </div>

      {isOwner && (
        <>
          <Separator />
          <div className="space-y-4">
            <h3 className="font-semibold">Room Settings</h3>
            <form onSubmit={handleUpdatePassword} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="newPassword">
                  {hasPassword ? "Change Password" : "Set Password"}
                </Label>
                <div className="flex gap-2">
                  <Input
                    id="newPassword"
                    type="password"
                    placeholder={
                      hasPassword
                        ? "New password or leave empty to remove"
                        : "Set a password"
                    }
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    disabled={isUpdating}
                  />
                  <Button type="submit" disabled={isUpdating}>
                    {isUpdating ? "Updating..." : "Update"}
                  </Button>
                </div>
              </div>
            </form>
            <Button
              variant="destructive"
              onClick={handleCloseRoom}
              disabled={isUpdating}
            >
              Close Room
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
