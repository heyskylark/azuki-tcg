"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
import { Navbar } from "@/components/layout/Navbar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface ActiveRoom {
  id: string;
  status: string;
}

export default function CreateRoomPage() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeRoom, setActiveRoom] = useState<ActiveRoom | null>(null);
  const [isCheckingActiveRoom, setIsCheckingActiveRoom] = useState(true);

  useEffect(() => {
    async function checkActiveRoom() {
      try {
        const response = await authenticatedFetch("/api/users/rooms/active");
        if (response.ok) {
          const data = await response.json();
          setActiveRoom(data.room);
        }
      } catch {
        // Ignore errors - just allow room creation
      } finally {
        setIsCheckingActiveRoom(false);
      }
    }
    checkActiveRoom();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const body: { password?: string } = {};
      if (password.trim()) {
        body.password = password;
      }

      const response = await authenticatedFetch("/api/rooms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || "Failed to create room");
      }

      const data = await response.json();
      router.push(`/rooms/${data.room.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create room");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-md mx-auto">
          <Card>
            <CardHeader>
              <CardTitle>Create Room</CardTitle>
              <CardDescription>
                {activeRoom
                  ? "You already have an active room."
                  : "Start a new game room. Add a password to make it private."}
              </CardDescription>
            </CardHeader>
            {isCheckingActiveRoom ? (
              <CardContent>
                <p className="text-sm text-muted-foreground">Loading...</p>
              </CardContent>
            ) : activeRoom ? (
              <>
                <CardContent>
                  <Alert>
                    <AlertDescription>
                      You are already in an active room. You can only be in one
                      room at a time.
                    </AlertDescription>
                  </Alert>
                </CardContent>
                <CardFooter>
                  <Button
                    className="w-full"
                    onClick={() => router.push(`/rooms/${activeRoom.id}`)}
                  >
                    Go to Room
                  </Button>
                </CardFooter>
              </>
            ) : (
              <form onSubmit={handleSubmit}>
                <CardContent className="space-y-4">
                  {error && (
                    <Alert variant="destructive">
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}
                  <div className="space-y-2">
                    <Label htmlFor="password">Password (optional)</Label>
                    <Input
                      id="password"
                      type="password"
                      placeholder="Leave empty for public room"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={isLoading}
                    />
                    <p className="text-sm text-muted-foreground">
                      Share this password with your friend to let them join.
                    </p>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button type="submit" className="w-full" disabled={isLoading}>
                    {isLoading ? "Creating..." : "Create Room"}
                  </Button>
                </CardFooter>
              </form>
            )}
          </Card>
        </div>
      </main>
    </div>
  );
}
