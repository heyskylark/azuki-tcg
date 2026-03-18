"use client";

import { useEffect } from "react";
import type { DebugDrawMessage } from "@tcg/backend-core/types/ws";
import { useRoom } from "@/contexts/RoomContext";

declare global {
  interface Window {
    azkDebugDraw?: (cardCode: string) => void;
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

    console.info("[AzukiDebug] Command available: azkDebugDraw('STT02-007')");

    return () => {
      if (previousDebugDraw) {
        window.azkDebugDraw = previousDebugDraw;
      } else {
        delete window.azkDebugDraw;
      }
    };
  }, [send]);

  return null;
}
