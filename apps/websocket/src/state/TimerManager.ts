import logger from "@/logger";
import {
  READY_COUNTDOWN_MS,
  DISCONNECT_GRACE_MS,
} from "@/constants";

interface RoomTimers {
  readyCountdown: NodeJS.Timeout | null;
  deckSelectionDeadline: NodeJS.Timeout | null;
  disconnectGrace: Map<0 | 1, NodeJS.Timeout>;
}

const roomTimers = new Map<string, RoomTimers>();

function getOrCreateTimers(roomId: string): RoomTimers {
  let timers = roomTimers.get(roomId);
  if (!timers) {
    timers = {
      readyCountdown: null,
      deckSelectionDeadline: null,
      disconnectGrace: new Map(),
    };
    roomTimers.set(roomId, timers);
  }
  return timers;
}

export function startReadyCountdown(
  roomId: string,
  onComplete: () => Promise<void>
): void {
  const timers = getOrCreateTimers(roomId);

  if (timers.readyCountdown) {
    clearTimeout(timers.readyCountdown);
  }

  logger.info("Starting ready countdown", { roomId, durationMs: READY_COUNTDOWN_MS });

  timers.readyCountdown = setTimeout(async () => {
    timers.readyCountdown = null;
    try {
      await onComplete();
    } catch (error) {
      logger.error("Error in ready countdown callback", { roomId, error });
    }
  }, READY_COUNTDOWN_MS);
}

export function cancelReadyCountdown(roomId: string): boolean {
  const timers = roomTimers.get(roomId);
  if (!timers?.readyCountdown) {
    return false;
  }

  clearTimeout(timers.readyCountdown);
  timers.readyCountdown = null;
  logger.info("Cancelled ready countdown", { roomId });
  return true;
}

export function isReadyCountdownActive(roomId: string): boolean {
  const timers = roomTimers.get(roomId);
  return timers?.readyCountdown !== null;
}

export function startDeckSelectionTimeout(
  roomId: string,
  deadline: Date,
  onTimeout: () => Promise<void>
): void {
  const timers = getOrCreateTimers(roomId);

  if (timers.deckSelectionDeadline) {
    clearTimeout(timers.deckSelectionDeadline);
  }

  const msUntilDeadline = deadline.getTime() - Date.now();
  if (msUntilDeadline <= 0) {
    logger.warn("Deck selection deadline already passed", { roomId, deadline });
    onTimeout().catch((error) => {
      logger.error("Error in deck selection timeout callback", { roomId, error });
    });
    return;
  }

  logger.info("Starting deck selection timeout", { roomId, deadline, msUntilDeadline });

  timers.deckSelectionDeadline = setTimeout(async () => {
    timers.deckSelectionDeadline = null;
    try {
      await onTimeout();
    } catch (error) {
      logger.error("Error in deck selection timeout callback", { roomId, error });
    }
  }, msUntilDeadline);
}

export function cancelDeckSelectionTimeout(roomId: string): boolean {
  const timers = roomTimers.get(roomId);
  if (!timers?.deckSelectionDeadline) {
    return false;
  }

  clearTimeout(timers.deckSelectionDeadline);
  timers.deckSelectionDeadline = null;
  logger.info("Cancelled deck selection timeout", { roomId });
  return true;
}

export function startDisconnectGrace(
  roomId: string,
  playerSlot: 0 | 1,
  onTimeout: () => Promise<void>
): void {
  const timers = getOrCreateTimers(roomId);

  const existingTimer = timers.disconnectGrace.get(playerSlot);
  if (existingTimer) {
    clearTimeout(existingTimer);
  }

  logger.info("Starting disconnect grace period", { roomId, playerSlot, durationMs: DISCONNECT_GRACE_MS });

  const timer = setTimeout(async () => {
    timers.disconnectGrace.delete(playerSlot);
    try {
      await onTimeout();
    } catch (error) {
      logger.error("Error in disconnect grace timeout callback", { roomId, playerSlot, error });
    }
  }, DISCONNECT_GRACE_MS);

  timers.disconnectGrace.set(playerSlot, timer);
}

export function cancelDisconnectGrace(roomId: string, playerSlot: 0 | 1): boolean {
  const timers = roomTimers.get(roomId);
  if (!timers) {
    return false;
  }

  const timer = timers.disconnectGrace.get(playerSlot);
  if (!timer) {
    return false;
  }

  clearTimeout(timer);
  timers.disconnectGrace.delete(playerSlot);
  logger.info("Cancelled disconnect grace period", { roomId, playerSlot });
  return true;
}

export function clearAllTimersForRoom(roomId: string): void {
  const timers = roomTimers.get(roomId);
  if (!timers) {
    return;
  }

  if (timers.readyCountdown) {
    clearTimeout(timers.readyCountdown);
  }

  if (timers.deckSelectionDeadline) {
    clearTimeout(timers.deckSelectionDeadline);
  }

  for (const timer of timers.disconnectGrace.values()) {
    clearTimeout(timer);
  }

  roomTimers.delete(roomId);
  logger.info("Cleared all timers for room", { roomId });
}

export function getReadyCountdownEnd(roomId: string): Date | null {
  const timers = roomTimers.get(roomId);
  if (!timers?.readyCountdown) {
    return null;
  }
  return new Date(Date.now() + READY_COUNTDOWN_MS);
}
