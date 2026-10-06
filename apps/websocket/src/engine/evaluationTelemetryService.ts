import { createHash } from "node:crypto";
import { recordHumanEvaluationAction } from "@tcg/backend-core/services/humanEvaluationService";
import { getPlayerObservationBySlot } from "@/engine/WorldManager";
import { getRoomChannel } from "@/state/RoomRegistry";
import type { ActionTuple } from "@/engine/types";

interface DecisionContext {
  observation: unknown;
  legalActionMask: unknown;
  stateHash: string;
  receivedAt: Date;
}

export function captureEvaluationDecisionContext(
  roomId: string,
  actorSlot: 0 | 1,
  receivedAt: Date
): DecisionContext | null {
  const channel = getRoomChannel(roomId);
  if (!channel?.evaluation) {
    return null;
  }

  const observation = getPlayerObservationBySlot(roomId, actorSlot);
  const serializedObservation = JSON.stringify(observation);
  return {
    observation,
    legalActionMask: observation?.actionMask ?? null,
    stateHash: createHash("sha256").update(serializedObservation).digest("hex"),
    receivedAt,
  };
}

interface PersistDecisionParams {
  roomId: string;
  actorSlot: 0 | 1;
  actorSource: "HUMAN" | "AI";
  action: ActionTuple;
  accepted: boolean;
  error: string | null;
  context: DecisionContext | null;
  resolvedAt: Date;
}

export async function persistEvaluationDecision(params: PersistDecisionParams): Promise<void> {
  if (!params.context) {
    return;
  }

  await recordHumanEvaluationAction({
    roomId: params.roomId,
    actorSlot: params.actorSlot,
    actorSource: params.actorSource,
    action: params.action,
    accepted: params.accepted,
    error: params.error,
    observation: params.context.observation,
    legalActionMask: params.context.legalActionMask,
    stateHash: params.context.stateHash,
    receivedAt: params.context.receivedAt,
    resolvedAt: params.resolvedAt,
  });
}
