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

interface InferActionParams {
  modelKey: string;
  sessionKey: string;
  observationPacked: Uint8Array;
  resetSession?: boolean;
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
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        modelKey: params.modelKey,
        sessionKey: params.sessionKey,
        observationBase64: Buffer.from(params.observationPacked).toString("base64"),
        resetSession: params.resetSession ?? false,
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

export async function endInferenceSession(sessionKey: string): Promise<void> {
  const { controller, cancel } = withTimeoutAbort(env.INFERENCE_TIMEOUT_MS);
  try {
    await fetch(`${env.INFERENCE_URL}/session/end`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ sessionKey }),
      signal: controller.signal,
    });
  } catch {
    // Ignore session cleanup failures; room lifecycle should not block on this.
  } finally {
    cancel();
  }
}
