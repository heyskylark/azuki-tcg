/**
 * TypeScript wrapper around the native module.
 * Handles loading the native binding and provides type-safe access.
 */

import { createRequire } from "module";
import type {
  ActionResult,
  ActionTuple,
  CreateWorldResult,
  DeckCardEntry,
  GameLog,
  ObservationData,
  StateContext,
} from "@/engine/types";

export interface DebugLogEntry {
  level: "INFO" | "WARN" | "ERROR";
  message: string;
}

const require = createRequire(import.meta.url);

interface NativeBinding {
  createWorld(seed: number): CreateWorldResult;
  createWorldWithDecks(
    seed: number,
    player0Deck: DeckCardEntry[],
    player1Deck: DeckCardEntry[]
  ): CreateWorldResult;
  destroyWorld(worldId: string): void;
  submitAction(
    worldId: string,
    playerIndex: number,
    action: ActionTuple
  ): ActionResult;
  getObservation(worldId: string, playerIndex: number): ObservationData;
  getGameState(worldId: string): StateContext;
  getGameLogs(worldId: string): GameLog[];
  requiresAction(worldId: string): boolean;
  isGameOver(worldId: string): boolean;
  getActivePlayer(worldId: string): number;
  setDebugLogging(enabled: boolean): void;
  getDebugLogs(): DebugLogEntry[];
  clearDebugLogs(): void;
}

let nativeBinding: NativeBinding | null = null;

/**
 * Load the native binding module.
 * Must be called after the native module is built.
 */
export function loadNativeBinding(): NativeBinding {
  if (nativeBinding) {
    return nativeBinding;
  }

  try {
    // Try to load the native module from the build output directory
    // Path is relative to compiled location (dist/engine/) not source (src/engine/)
    const binding: NativeBinding = require("../../native/build/Release/azuki_engine.node");
    nativeBinding = binding;
    return binding;
  } catch (error) {
    // If Release build not found, try Debug
    try {
      const binding: NativeBinding = require("../../native/build/Debug/azuki_engine.node");
      nativeBinding = binding;
      return binding;
    } catch {
      throw new Error(
        "Native binding not found. Run 'yarn build:native' first. Original error: " +
          (error as Error).message
      );
    }
  }
}

/**
 * Get the native binding, throwing if not loaded.
 */
export function getNativeBinding(): NativeBinding {
  if (!nativeBinding) {
    return loadNativeBinding();
  }
  return nativeBinding;
}

/**
 * Check if the native binding is loaded.
 */
export function isNativeBindingLoaded(): boolean {
  return nativeBinding !== null;
}

// Re-export types from engine/types
export type {
  ActionResult,
  ActionTuple,
  CreateWorldResult,
  DeckCardEntry,
  GameLog,
  ObservationData,
  StateContext,
};
