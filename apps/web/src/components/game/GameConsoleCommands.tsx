"use client";

import { useEffect } from "react";
import type { DebugDrawMessage, DebugIkzMessage } from "@tcg/backend-core/types/ws";
import { useRoom } from "@/contexts/RoomContext";

declare global {
  interface Window {
    azkDebugDraw?: (cardCode: string) => void;
    azkDebugIKZ?: (count: number) => void;
  }
}

const DEBUG_ACTIONS_ENABLED =
  process.env.NEXT_PUBLIC_AZK_DEBUG_ACTIONS_ENABLED === "true" ||
  process.env.NEXT_PUBLIC_AZK_DEBUG_ACTIONS_ENABLED === "1";

export function GameConsoleCommands() {
  const { send } = useRoom();

  useEffect(() => {
    if (!DEBUG_ACTIONS_ENABLED) {
      return;
    }

    const previousDebugDraw = window.azkDebugDraw;
    const previousDebugIKZ = window.azkDebugIKZ;

    window.azkDebugDraw = (cardCode: string) => {
      if (typeof cardCode !== "string") {
        console.error("[AzukiDebug] azkDebugDraw(cardCode) requires a string card code");
        return;
      }

      const normalizedCardCode = cardCode.trim().toUpperCase();
      if (normalizedCardCode.length === 0) {
        console.error("[AzukiDebug] Card code cannot be empty");
        return;
      }

      const message = {
        type: "DEBUG_DRAW",
        cardCode: normalizedCardCode,
      } satisfies DebugDrawMessage;

      send(message);
      console.info(`[AzukiDebug] Requested debug draw for ${normalizedCardCode}`);
    };

    window.azkDebugIKZ = (count: number) => {
      if (!Number.isInteger(count) || count < 0) {
        console.error("[AzukiDebug] azkDebugIKZ(count) requires a non-negative integer");
        return;
      }

      const message = {
        type: "DEBUG_IKZ",
        count,
      } satisfies DebugIkzMessage;

      send(message);
      console.info(`[AzukiDebug] Requested debug IKZ grant for ${count}`);
    };

    console.info("[AzukiDebug] Command available: azkDebugDraw('STT02-007')");
    console.info("[AzukiDebug] Command available: azkDebugIKZ(3)");

    return () => {
      if (previousDebugDraw) {
        window.azkDebugDraw = previousDebugDraw;
      } else {
        delete window.azkDebugDraw;
      }

      if (previousDebugIKZ) {
        window.azkDebugIKZ = previousDebugIKZ;
      } else {
        delete window.azkDebugIKZ;
      }
    };
  }, [send]);

  return null;
}
