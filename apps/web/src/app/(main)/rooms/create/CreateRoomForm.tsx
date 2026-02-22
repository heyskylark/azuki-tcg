"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { authenticatedFetch } from "@/lib/api/authenticatedFetch";
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

export function CreateRoomForm() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [aiModelKey, setAiModelKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const body: { password?: string; aiModelKey?: string } = {};
      if (password.trim()) {
        body.password = password;
      }
      if (aiModelKey.trim()) {
        body.aiModelKey = aiModelKey.trim();
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
      setIsLoading(false);
      setError(err instanceof Error ? err.message : "Failed to create room");
    }
  };

  return (
    <div className="max-w-md mx-auto">
      <Card>
        <CardHeader>
          <CardTitle>Create Room</CardTitle>
          <CardDescription>
            Start a new game room. Add a password to make it private.
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit} className="space-y-6">
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
            <div className="space-y-2">
              <Label htmlFor="aiModelKey">AI Model Key (optional)</Label>
              <Input
                id="aiModelKey"
                type="text"
                placeholder="e.g. s3://bucket/models/policy.pt"
                value={aiModelKey}
                onChange={(e) => setAiModelKey(e.target.value)}
                disabled={isLoading}
              />
              <p className="text-sm text-muted-foreground">
                Set this to create a room against an AI opponent.
              </p>
            </div>
          </CardContent>
          <CardFooter>
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? "Creating..." : "Create Room"}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
