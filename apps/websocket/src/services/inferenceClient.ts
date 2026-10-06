import { z } from "zod";
import type { ActionTuple } from "@/engine/types";
import { env } from "@/env";

const inferResponseSchema = z
  .object({
    action: z.tuple([z.number().int(), z.number().int(), z.number().int(), z.number().int()]),
    device: z.string(),
  })
  .strict();

const errorResponseSchema = z
  .object({
    error: z.string(),
  })
  .strict();
const sessionStatusResponseSchema = z
  .object({
    active: z.boolean(),
  })
  .strict();

const generateDeckRequestSchema = z
  .object({
    modelKey: z.string().min(1),
    sessionKey: z.string().min(1),
    draftSeed: z.number().int().nonnegative().max(0xffffffff),
    aiSlot: z.union([z.literal(0), z.literal(1)]),
    gateCardCode: z.string().min(1),
    leaderCardCode: z.string().min(1),
  })
  .strict();

const deckPickSchema = z
  .object({
    ordinal: z.number().int().nonnegative(),
    candidateCardCodes: z.array(z.string().min(1)).min(1),
    selectedIndex: z.number().int().nonnegative(),
    selectedCardCode: z.string().min(1),
  })
  .strict();

const generatedDeckSchema = z
  .object({
    gateCardCode: z.string().min(1),
    leaderCardCode: z.string().min(1),
    orderedMainCardCodes: z.array(z.string().min(1)).length(50),
    cardCounts: z.record(z.string(), z.number().int().positive()),
    picks: z.array(deckPickSchema).length(50),
    deckHash: z.string().regex(/^[0-9a-f]{64}$/),
    catalogHash: z.string().regex(/^[0-9a-f]{64}$/),
    checkpointSha256: z.string().regex(/^[0-9a-f]{64}$/),
  })
  .strict()
  .superRefine((artifact, context) => {
    const countTotal = Object.values(artifact.cardCounts).reduce(
      (total, count) => total + count,
      0
    );
    if (countTotal !== 50) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "cardCounts must total 50 cards",
        path: ["cardCounts"],
      });
    }
    const orderedCounts = new Map<string, number>();
    for (const cardCode of artifact.orderedMainCardCodes) {
      orderedCounts.set(cardCode, (orderedCounts.get(cardCode) ?? 0) + 1);
    }
    if (
      Object.keys(artifact.cardCounts).length !== orderedCounts.size ||
      [...orderedCounts].some(([cardCode, count]) => artifact.cardCounts[cardCode] !== count)
    ) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "cardCounts must match orderedMainCardCodes",
        path: ["cardCounts"],
      });
    }

    for (const [index, pick] of artifact.picks.entries()) {
      if (pick.ordinal !== index + 1) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          message: "pick ordinals must be contiguous and one-based",
          path: ["picks", index, "ordinal"],
        });
      }
      if (pick.selectedCardCode !== artifact.orderedMainCardCodes[index]) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          message: "draft picks must match orderedMainCardCodes",
          path: ["picks", index, "selectedCardCode"],
        });
      }
      if (
        pick.selectedIndex >= pick.candidateCardCodes.length ||
        pick.candidateCardCodes[pick.selectedIndex] !== pick.selectedCardCode
      ) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          message: "selected pick must match its candidate list",
          path: ["picks", index],
        });
      }
    }
  });

export interface GenerateDeckParams {
  modelKey: string;
  sessionKey: string;
  draftSeed: number;
  aiSlot: 0 | 1;
  gateCardCode: string;
  leaderCardCode: string;
}

export type GeneratedDeckArtifact = z.infer<typeof generatedDeckSchema>;

interface InferActionParams {
  modelKey: string;
  sessionKey: string;
  observationPacked: Uint8Array;
  resetSession?: boolean;
  requireSession?: boolean;
}

function inferenceHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (env.INFERENCE_SHARED_SECRET) {
    headers["Authorization"] = `Bearer ${env.INFERENCE_SHARED_SECRET}`;
  }
  return headers;
}

function withTimeoutAbort(timeoutMs: number): {
  controller: AbortController;
  cancel: () => void;
} {
  const controller = new AbortController();
  const timeout = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  return {
    controller,
    cancel: () => clearTimeout(timeout),
  };
}

export async function inferAction(params: InferActionParams): Promise<ActionTuple> {
  const { controller, cancel } = withTimeoutAbort(env.INFERENCE_TIMEOUT_MS);
  try {
    const response = await fetch(`${env.INFERENCE_URL}/infer`, {
      method: "POST",
      headers: inferenceHeaders(),
      body: JSON.stringify({
        modelKey: params.modelKey,
        sessionKey: params.sessionKey,
        observationBase64: Buffer.from(params.observationPacked).toString("base64"),
        resetSession: params.resetSession ?? false,
        requireSession: params.requireSession ?? false,
      }),
      signal: controller.signal,
    });

    const payload = await response.json();

    if (!response.ok) {
      const parsedError = errorResponseSchema.safeParse(payload);
      console.error(env.INFERENCE_URL);
      const message = parsedError.success
        ? parsedError.data.error
        : `Inference request to ${env.INFERENCE_URL} failed with status ${response.status}`;
      throw new Error(message);
    }

    const parsed = inferResponseSchema.parse(payload);
    const [a0, a1, a2, a3] = parsed.action;
    return [a0, a1, a2, a3] satisfies ActionTuple;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Inference request timed out after ${env.INFERENCE_TIMEOUT_MS}ms`);
    }
    throw error;
  } finally {
    cancel();
  }
}

export async function generateDeck(params: GenerateDeckParams): Promise<GeneratedDeckArtifact> {
  const request = generateDeckRequestSchema.parse(params);
  const { controller, cancel } = withTimeoutAbort(env.INFERENCE_TIMEOUT_MS);
  try {
    const response = await fetch(`${env.INFERENCE_URL}/deck/generate`, {
      method: "POST",
      headers: inferenceHeaders(),
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    const payload = await response.json();

    if (!response.ok) {
      const parsedError = errorResponseSchema.safeParse(payload);
      const message = parsedError.success
        ? parsedError.data.error
        : `Deck generation request failed with status ${response.status}`;
      throw new Error(message);
    }

    return generatedDeckSchema.parse(payload);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Deck generation request timed out after ${env.INFERENCE_TIMEOUT_MS}ms`);
    }
    throw error;
  } finally {
    cancel();
  }
}

export async function hasInferenceSession(sessionKey: string): Promise<boolean> {
  const { controller, cancel } = withTimeoutAbort(env.INFERENCE_TIMEOUT_MS);
  try {
    const response = await fetch(`${env.INFERENCE_URL}/session/status`, {
      method: "POST",
      headers: inferenceHeaders(),
      body: JSON.stringify({ sessionKey }),
      signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok) {
      const parsedError = errorResponseSchema.safeParse(payload);
      throw new Error(
        parsedError.success
          ? parsedError.data.error
          : `Inference session status failed with status ${response.status}`
      );
    }
    return sessionStatusResponseSchema.parse(payload).active;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Inference session status timed out after ${env.INFERENCE_TIMEOUT_MS}ms`);
    }
    throw error;
  } finally {
    cancel();
  }
}

export async function endInferenceSession(sessionKey: string): Promise<void> {
  const { controller, cancel } = withTimeoutAbort(env.INFERENCE_TIMEOUT_MS);
  try {
    await fetch(`${env.INFERENCE_URL}/session/end`, {
      method: "POST",
      headers: inferenceHeaders(),
      body: JSON.stringify({ sessionKey }),
      signal: controller.signal,
    });
  } catch {
    // Ignore session cleanup failures; room lifecycle should not block on this.
  } finally {
    cancel();
  }
}
