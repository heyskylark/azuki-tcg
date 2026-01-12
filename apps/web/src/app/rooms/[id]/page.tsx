"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import { Navbar } from "@/components/layout/Navbar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

interface Room {
  id: string;
  status: string;
  type: string;
  hasPassword: boolean;
  player0Id: string;
  player1Id: string | null;
  createdAt: string;
  updatedAt: string;
}

export default function RoomPage() {
  const params = useParams();
  const router = useRouter();
  const { user } = useAuth();
  const roomId = params.id as string;

  const [room, setRoom] = useState<Room | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newPassword, setNewPassword] = useState("");
  const [isUpdating, setIsUpdating] = useState(false);

  const fetchRoom = useCallback(async () => {
    try {
      const response = await authenticatedFetch(`/api/rooms/${roomId}`);
      if (!response.ok) {
        throw new Error("Failed to fetch room");
      }
      const data = await response.json();
      setRoom(data.room);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load room");
    } finally {
      setIsLoading(false);
    }
  }, [roomId]);

  useEffect(() => {
    fetchRoom();
  }, [fetchRoom]);

  const isOwner = room && user && room.player0Id === user.id;

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

      await fetchRoom();
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

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Card className="max-w-2xl mx-auto">
            <CardHeader>
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-4 w-32 mt-2" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-24 w-full" />
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (!room) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Alert variant="destructive">
            <AlertDescription>Room not found</AlertDescription>
          </Alert>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <Card className="max-w-2xl mx-auto">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Room</CardTitle>
              <div className="flex gap-2">
                <Badge variant={room.status === "WAITING_FOR_PLAYERS" ? "default" : "secondary"}>
                  {room.status.replace(/_/g, " ")}
                </Badge>
                {room.hasPassword && <Badge variant="outline">Private</Badge>}
              </div>
            </div>
            <CardDescription>Room ID: {room.id}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-4">
              <h3 className="font-semibold">Players</h3>
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardContent className="pt-4">
                    <p className="text-sm text-muted-foreground">Player 1 (Host)</p>
                    <p className="font-medium">
                      {room.player0Id === user?.id ? "You" : room.player0Id}
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-4">
                    <p className="text-sm text-muted-foreground">Player 2</p>
                    <p className="font-medium">
                      {room.player1Id
                        ? room.player1Id === user?.id
                          ? "You"
                          : room.player1Id
                        : "Waiting..."}
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>

            {isOwner && room.status === "WAITING_FOR_PLAYERS" && (
              <>
                <Separator />
                <div className="space-y-4">
                  <h3 className="font-semibold">Room Settings</h3>
                  <form onSubmit={handleUpdatePassword} className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="newPassword">
                        {room.hasPassword ? "Change Password" : "Set Password"}
                      </Label>
                      <div className="flex gap-2">
                        <Input
                          id="newPassword"
                          type="password"
                          placeholder={room.hasPassword ? "New password or leave empty to remove" : "Set a password"}
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

            <Separator />
            <div className="flex justify-between items-center">
              <p className="text-sm text-muted-foreground">
                No live updates yet. Use the button to refresh.
              </p>
              <Button variant="outline" onClick={fetchRoom}>
                Refresh
              </Button>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
