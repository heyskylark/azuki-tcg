"use client";

import { useEffect, type ReactNode } from "react";
import type {
  GameSnapshotMessage,
  GameLogBatchMessage,
} from "@tcg/backend-core/types/ws";
import { useRoom } from "@/contexts/RoomContext";
import { useAssets } from "@/contexts/AssetContext";
import { useGameState } from "@/contexts/GameStateContext";

interface GameBridgeProps {
  playerSlot: 0 | 1;
  children: ReactNode;
}

export function GameBridge({ playerSlot, children }: GameBridgeProps) {
  const { onGameMessage } = useRoom();
  const { preloadCardsByMetadata } = useAssets();
  const { setCardMappings, processSnapshot, processLogBatch } = useGameState();

  useEffect(() => {
    const unsubscribe = onGameMessage((message) => {
      if (message.type === "GAME_SNAPSHOT") {
        const snapshot = message as GameSnapshotMessage;
        const cardMetadata = snapshot.cardMetadata;

        if (cardMetadata && Object.keys(cardMetadata).length > 0) {
          void (async () => {
            try {
              const mappings = await preloadCardsByMetadata(cardMetadata);
              setCardMappings(mappings);
              processSnapshot(snapshot, playerSlot, mappings);
            } catch {
              processSnapshot(snapshot, playerSlot);
            }
          })();
          return;
        }

        processSnapshot(snapshot, playerSlot);
        return;
      }

      if (message.type === "GAME_LOG_BATCH") {
        processLogBatch(message as GameLogBatchMessage, playerSlot);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [
    onGameMessage,
    preloadCardsByMetadata,
    processSnapshot,
    processLogBatch,
    setCardMappings,
    playerSlot,
  ]);

  return <>{children}</>;
}
