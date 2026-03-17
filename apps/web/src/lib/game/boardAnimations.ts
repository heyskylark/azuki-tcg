"use client";

import type { CardMapping, GameState, ResolvedCard, ResolvedHandCard } from "@/types/game";
import { buildImageUrl } from "@/types/game";
import type { CardZoneMovedData, ProcessedGameLog, ZoneMovedMetadata } from "@/types/gameLogs";

type BoardAnimationSide = "my" | "opponent";

interface DeckAnimationAnchor {
  zone: "DECK";
  side: BoardAnimationSide;
}

interface DiscardAnimationAnchor {
  zone: "DISCARD";
  side: BoardAnimationSide;
}

interface HandAnimationAnchor {
  zone: "HAND";
  side: BoardAnimationSide;
  index: number;
  handCount: number;
}

interface BoardZoneAnimationAnchor {
  zone: "GARDEN" | "ALLEY";
  side: BoardAnimationSide;
  index: number;
}

export type BoardAnimationAnchor =
  | DeckAnimationAnchor
  | DiscardAnimationAnchor
  | HandAnimationAnchor
  | BoardZoneAnimationAnchor;

export interface BoardAnimationCard {
  cardCode: string;
  imageUrl: string;
  name: string;
  attack: number | null;
  health: number | null;
  tapped: boolean;
  cooldown: boolean;
  isFrozen: boolean;
  isShocked: boolean;
  isEffectImmune: boolean;
  hasCharge: boolean;
  hasDefender: boolean;
  hasInfiltrate: boolean;
  showStats: boolean;
}

export interface BoardMoveAnimation {
  id: string;
  kind: "draw" | "play" | "portal" | "discard";
  card: BoardAnimationCard;
  from: BoardAnimationAnchor;
  to: BoardAnimationAnchor;
  durationMs: number;
  hiddenTargetKey: string | null;
  startedAtMs: number;
}

const DRAW_ANIMATION_MS = 420;
const ZONE_MOVE_ANIMATION_MS = 360;

export function buildBoardAnimationKey(
  side: BoardAnimationSide,
  zone: "HAND" | "GARDEN" | "ALLEY",
  index: number
): string {
  return `${side}:${zone}:${index}`;
}

export function getBoardAnimationKeyForAnchor(anchor: BoardAnimationAnchor): string | null {
  if (anchor.zone === "DECK" || anchor.zone === "DISCARD") {
    return null;
  }

  return buildBoardAnimationKey(anchor.side, anchor.zone, anchor.index);
}

function resolveCardMapping(
  cardDefId: number | null,
  cardDefIdMap: Map<number, CardMapping>
): CardMapping | null {
  if (cardDefId === null) {
    return null;
  }

  return cardDefIdMap.get(cardDefId) ?? null;
}

function buildCardFromMapping(
  mapping: CardMapping,
  metadata: ZoneMovedMetadata | null,
  showStats: boolean
): BoardAnimationCard {
  return {
    cardCode: mapping.cardCode,
    imageUrl: buildImageUrl(mapping.imageKey),
    name: mapping.name,
    attack: metadata?.curAtk ?? mapping.attack,
    health: metadata?.curHp ?? mapping.health,
    tapped: metadata?.tapped ?? false,
    cooldown: metadata?.cooldown ?? false,
    isFrozen: metadata?.isFrozen ?? false,
    isShocked: false,
    isEffectImmune: metadata?.isEffectImmune ?? false,
    hasCharge: metadata?.hasCharge ?? false,
    hasDefender: metadata?.hasDefender ?? false,
    hasInfiltrate: metadata?.hasInfiltrate ?? false,
    showStats,
  };
}

function buildCardFromResolvedHandCard(
  card: ResolvedHandCard,
  metadata: ZoneMovedMetadata | null,
  showStats: boolean
): BoardAnimationCard {
  return {
    cardCode: card.cardCode,
    imageUrl: card.imageUrl,
    name: card.name,
    attack: metadata?.curAtk ?? null,
    health: metadata?.curHp ?? null,
    tapped: metadata?.tapped ?? false,
    cooldown: metadata?.cooldown ?? false,
    isFrozen: metadata?.isFrozen ?? false,
    isShocked: false,
    isEffectImmune: metadata?.isEffectImmune ?? false,
    hasCharge: metadata?.hasCharge ?? false,
    hasDefender: metadata?.hasDefender ?? false,
    hasInfiltrate: metadata?.hasInfiltrate ?? false,
    showStats,
  };
}

function buildCardFromResolvedBoardCard(
  card: ResolvedCard,
  metadata: ZoneMovedMetadata | null,
  showStats: boolean
): BoardAnimationCard {
  return {
    cardCode: card.cardCode,
    imageUrl: card.imageUrl,
    name: card.name,
    attack: metadata?.curAtk ?? card.curAtk,
    health: metadata?.curHp ?? card.curHp,
    tapped: metadata?.tapped ?? card.tapped,
    cooldown: metadata?.cooldown ?? card.cooldown,
    isFrozen: metadata?.isFrozen ?? card.isFrozen,
    isShocked: card.isShocked,
    isEffectImmune: metadata?.isEffectImmune ?? card.isEffectImmune,
    hasCharge: metadata?.hasCharge ?? card.hasCharge,
    hasDefender: metadata?.hasDefender ?? card.hasDefender,
    hasInfiltrate: metadata?.hasInfiltrate ?? card.hasInfiltrate,
    showStats,
  };
}

function buildAnimationCard(
  state: GameState,
  data: CardZoneMovedData,
  isMyCard: boolean,
  cardDefIdMap: Map<number, CardMapping>,
  showStats: boolean
): BoardAnimationCard | null {
  const sourceBoard = isMyCard ? state.myBoard : state.opponentBoard;
  const mapping = resolveCardMapping(data.card.cardDefId, cardDefIdMap);

  if (data.fromZone === "HAND" && isMyCard) {
    const handCard = state.myHand[data.fromIndex];
    if (handCard) {
      return buildCardFromResolvedHandCard(handCard, data.metadata, showStats);
    }
  }

  if (data.fromZone === "ALLEY") {
    const alleyCard = sourceBoard.alley[data.fromIndex];
    if (alleyCard) {
      return buildCardFromResolvedBoardCard(alleyCard, data.metadata, showStats);
    }
  }

  if (data.fromZone === "GARDEN") {
    const gardenCard = sourceBoard.garden[data.fromIndex];
    if (gardenCard) {
      return buildCardFromResolvedBoardCard(gardenCard, data.metadata, showStats);
    }
  }

  if (mapping) {
    return buildCardFromMapping(mapping, data.metadata, showStats);
  }

  return null;
}

function buildHandAnchor(
  side: BoardAnimationSide,
  index: number,
  handCount: number
): HandAnimationAnchor {
  return {
    zone: "HAND",
    side,
    index,
    handCount,
  };
}

function buildBoardAnchor(
  zone: "GARDEN" | "ALLEY",
  side: BoardAnimationSide,
  index: number
): BoardZoneAnimationAnchor {
  return {
    zone,
    side,
    index,
  };
}

function buildDeckAnchor(side: BoardAnimationSide): DeckAnimationAnchor {
  return {
    zone: "DECK",
    side,
  };
}

function buildDiscardAnchor(side: BoardAnimationSide): DiscardAnimationAnchor {
  return {
    zone: "DISCARD",
    side,
  };
}

function buildZoneMoveAnimation(
  state: GameState,
  data: CardZoneMovedData,
  playerSlot: 0 | 1,
  cardDefIdMap: Map<number, CardMapping>
): BoardMoveAnimation | null {
  const isMyCard = data.card.player === playerSlot;
  const side: BoardAnimationSide = isMyCard ? "my" : "opponent";

  if (data.fromZone === "DECK" && data.toZone === "HAND" && isMyCard) {
    const card = buildAnimationCard(state, data, isMyCard, cardDefIdMap, false);
    if (!card) {
      return null;
    }

    const destinationIndex =
      data.toIndex >= 0 && data.toIndex < state.myHand.length ? data.toIndex : state.myHand.length;
    const destination = buildHandAnchor(side, destinationIndex, state.myHand.length + 1);

    return {
      id: `anim:${data.card.player}:${data.fromZone}:${data.fromIndex}:${data.toZone}:${data.toIndex}:${Date.now()}`,
      kind: "draw",
      card,
      from: buildDeckAnchor(side),
      to: destination,
      durationMs: DRAW_ANIMATION_MS,
      hiddenTargetKey: getBoardAnimationKeyForAnchor(destination),
      startedAtMs: performance.now(),
    };
  }

  if (
    data.fromZone === "HAND" &&
    !isMyCard &&
    (data.toZone === "GARDEN" || data.toZone === "ALLEY")
  ) {
    const card = buildAnimationCard(state, data, isMyCard, cardDefIdMap, true);
    if (!card) {
      return null;
    }

    const source = buildHandAnchor(side, data.fromIndex, Math.max(data.fromIndex + 1, 1));
    const destination = buildBoardAnchor(data.toZone, side, data.toIndex);

    return {
      id: `anim:${data.card.player}:${data.fromZone}:${data.fromIndex}:${data.toZone}:${data.toIndex}:${Date.now()}`,
      kind: "play",
      card,
      from: source,
      to: destination,
      durationMs: ZONE_MOVE_ANIMATION_MS,
      hiddenTargetKey: getBoardAnimationKeyForAnchor(destination),
      startedAtMs: performance.now(),
    };
  }

  if (
    data.toZone === "DISCARD" &&
    ((data.fromZone === "HAND" && isMyCard) ||
      data.fromZone === "GARDEN" ||
      data.fromZone === "ALLEY")
  ) {
    const showStats = data.fromZone !== "HAND";
    const card = buildAnimationCard(state, data, isMyCard, cardDefIdMap, showStats);
    if (!card) {
      return null;
    }

    const source =
      data.fromZone === "HAND"
        ? buildHandAnchor(side, data.fromIndex, state.myHand.length)
        : buildBoardAnchor(data.fromZone, side, data.fromIndex);

    return {
      id: `anim:${data.card.player}:${data.fromZone}:${data.fromIndex}:${data.toZone}:${data.toIndex}:${Date.now()}`,
      kind: "discard",
      card,
      from: source,
      to: buildDiscardAnchor(side),
      durationMs: ZONE_MOVE_ANIMATION_MS,
      hiddenTargetKey: null,
      startedAtMs: performance.now(),
    };
  }

  if (!isMyCard && data.fromZone === "ALLEY" && data.toZone === "GARDEN") {
    const card = buildAnimationCard(state, data, isMyCard, cardDefIdMap, true);
    if (!card) {
      return null;
    }

    const source = buildBoardAnchor("ALLEY", side, data.fromIndex);
    const destination = buildBoardAnchor("GARDEN", side, data.toIndex);

    return {
      id: `anim:${data.card.player}:${data.fromZone}:${data.fromIndex}:${data.toZone}:${data.toIndex}:${Date.now()}`,
      kind: "portal",
      card,
      from: source,
      to: destination,
      durationMs: ZONE_MOVE_ANIMATION_MS,
      hiddenTargetKey: getBoardAnimationKeyForAnchor(destination),
      startedAtMs: performance.now(),
    };
  }

  return null;
}

export function buildBoardMoveAnimationForLog(
  state: GameState,
  log: ProcessedGameLog,
  playerSlot: 0 | 1,
  cardDefIdMap: Map<number, CardMapping>
): BoardMoveAnimation | null {
  if (log.type !== "ZONE_MOVED") {
    return null;
  }

  return buildZoneMoveAnimation(state, log.data, playerSlot, cardDefIdMap);
}
