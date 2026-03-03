"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { z } from "zod";
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

const aiModelSchema = z
  .object({
    id: z.string().uuid(),
    displayName: z.string().trim().min(1),
  })
  .strict();

const aiModelsResponseSchema = z
  .object({
    models: z.array(aiModelSchema),
  })
  .strict();

const errorResponseSchema = z
  .object({
    message: z.string(),
  })
  .strict();

type AiModelOption = z.infer<typeof aiModelSchema>;

export function CreateRoomForm() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [aiModelId, setAiModelId] = useState("");
  const [aiModels, setAiModels] = useState<AiModelOption[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const loadAiModels = async () => {
      setIsLoadingModels(true);
      setModelLoadError(null);

      try {
        const response = await authenticatedFetch("/api/ai-models");
        const payload: unknown = await response.json();

        if (!response.ok) {
          const parsedError = errorResponseSchema.safeParse(payload);
          const message = parsedError.success
            ? parsedError.data.message
            : "Failed to load AI models";
          throw new Error(message);
        }

        const parsed = aiModelsResponseSchema.parse(payload);
        if (cancelled) {
          return;
        }

        setAiModels(parsed.models);
      } catch (err) {
        if (cancelled) {
          return;
        }

        setAiModels([]);
        setAiModelId("");
        setModelLoadError(err instanceof Error ? err.message : "Failed to load AI models");
      } finally {
        if (!cancelled) {
          setIsLoadingModels(false);
        }
      }
    };

    void loadAiModels();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const body: { password?: string; aiModelId?: string } = {};
      if (password.trim()) {
        body.password = password;
      }
      if (aiModelId) {
        body.aiModelId = aiModelId;
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
              <Label htmlFor="aiModelId">AI Opponent (optional)</Label>
              <select
                id="aiModelId"
                value={aiModelId}
                onChange={(e) => setAiModelId(e.target.value)}
                disabled={isLoading || isLoadingModels}
                className="border-input bg-transparent focus-visible:border-ring focus-visible:ring-ring/50 h-9 w-full rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50"
              >
                <option value="">None (Player vs Player)</option>
                {aiModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.displayName}
                  </option>
                ))}
              </select>
              <p className="text-sm text-muted-foreground">
                {isLoadingModels
                  ? "Loading available AI models..."
                  : "Choose an AI model to create a room against an AI opponent."}
              </p>
              {modelLoadError && (
                <p className="text-sm text-destructive">{modelLoadError}</p>
              )}
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
