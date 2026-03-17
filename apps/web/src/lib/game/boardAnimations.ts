"use client";

import type {
  CardMapping,
  GameState,
  ResolvedCard,
  ResolvedHandCard,
  ResolvedLeader,
} from "@/types/game";
import { buildImageUrl } from "@/types/game";
import type {
  CardZoneMovedData,
  CombatDamageData,
  LogCardRef,
  ProcessedGameLog,
  ZoneMovedMetadata,
} from "@/types/gameLogs";

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

interface LeaderAnimationAnchor {
  zone: "LEADER";
  side: BoardAnimationSide;
  index: 0;
}

export type BoardAnimationAnchor =
  | DeckAnimationAnchor
  | DiscardAnimationAnchor
  | HandAnimationAnchor
  | BoardZoneAnimationAnchor
  | LeaderAnimationAnchor;

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
  kind: "draw" | "play" | "portal" | "discard";
  id: string;
  card: BoardAnimationCard;
  from: BoardAnimationAnchor;
  to: BoardAnimationAnchor;
  durationMs: number;
  hiddenSlotKeys: string[];
  startedAtMs: number;
}

export interface BoardDamageNumberAnimation {
  id: string;
  value: number;
  anchor: BoardAnimationAnchor;
  color: string;
}

export interface BoardCombatAnimation {
  kind: "combat";
  id: string;
  attacker: BoardAnimationCard;
  defender: BoardAnimationCard;
  attackerFrom: BoardAnimationAnchor;
  attackerTo: BoardAnimationAnchor;
  defenderAnchor: BoardAnimationAnchor;
  damageNumbers: BoardDamageNumberAnimation[];
  durationMs: number;
  hiddenSlotKeys: string[];
  startedAtMs: number;
}

export type BoardAnimation = BoardMoveAnimation | BoardCombatAnimation;

const DRAW_ANIMATION_MS = 420;
const ZONE_MOVE_ANIMATION_MS = 360;
const COMBAT_ANIMATION_MS = 700;

export function buildBoardAnimationKey(
  side: BoardAnimationSide,
  zone: "HAND" | "GARDEN" | "ALLEY" | "LEADER",
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

function buildCardFromResolvedLeader(
  leader: ResolvedLeader,
  showStats: boolean
): BoardAnimationCard {
  return {
    cardCode: leader.cardCode,
    imageUrl: leader.imageUrl,
    name: leader.name,
    attack: leader.curAtk,
    health: leader.curHp,
    tapped: leader.tapped,
    cooldown: leader.cooldown,
    isFrozen: leader.isFrozen,
    isShocked: leader.isShocked,
    isEffectImmune: leader.isEffectImmune,
    hasCharge: leader.hasCharge,
    hasDefender: leader.hasDefender,
    hasInfiltrate: leader.hasInfiltrate,
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

function buildCombatCard(
  state: GameState,
  card: LogCardRef,
  playerSlot: 0 | 1,
  cardDefIdMap: Map<number, CardMapping>,
  showStats: boolean
): BoardAnimationCard | null {
  const isMyCard = card.player === playerSlot;
  const board = isMyCard ? state.myBoard : state.opponentBoard;

  switch (card.zone) {
    case "LEADER":
      return buildCardFromResolvedLeader(board.leader, showStats);

    case "GARDEN": {
      const gardenCard = board.garden[card.zoneIndex];
      return gardenCard ? buildCardFromResolvedBoardCard(gardenCard, null, showStats) : null;
    }

    case "ALLEY": {
      const alleyCard = board.alley[card.zoneIndex];
      return alleyCard ? buildCardFromResolvedBoardCard(alleyCard, null, showStats) : null;
    }

    default: {
      const mapping = resolveCardMapping(card.cardDefId, cardDefIdMap);
      return mapping ? buildCardFromMapping(mapping, null, showStats) : null;
    }
  }
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

function buildLeaderAnchor(side: BoardAnimationSide): LeaderAnimationAnchor {
  return {
    zone: "LEADER",
    side,
    index: 0,
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

function buildCombatAnchor(card: LogCardRef, playerSlot: 0 | 1): BoardAnimationAnchor | null {
  const side: BoardAnimationSide = card.player === playerSlot ? "my" : "opponent";

  switch (card.zone) {
    case "LEADER":
      return buildLeaderAnchor(side);

    case "GARDEN":
    case "ALLEY":
      return buildBoardAnchor(card.zone, side, card.zoneIndex);

    default:
      return null;
  }
}

function createAnimationId(prefix: string): string {
  return `${prefix}:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`;
}

function getHiddenSlotKeys(anchors: BoardAnimationAnchor[]): string[] {
  return Array.from(
    new Set(
      anchors
        .map((anchor) => getBoardAnimationKeyForAnchor(anchor))
        .filter((key): key is string => key !== null)
    )
  );
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
      kind: "draw",
      id: createAnimationId(
        `move:${data.card.player}:${data.fromZone}:${data.fromIndex}:${data.toZone}:${data.toIndex}`
      ),
      card,
      from: buildDeckAnchor(side),
      to: destination,
      durationMs: DRAW_ANIMATION_MS,
      hiddenSlotKeys: getHiddenSlotKeys([destination]),
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
      kind: "play",
      id: createAnimationId(
        `move:${data.card.player}:${data.fromZone}:${data.fromIndex}:${data.toZone}:${data.toIndex}`
      ),
      card,
      from: source,
      to: destination,
      durationMs: ZONE_MOVE_ANIMATION_MS,
      hiddenSlotKeys: getHiddenSlotKeys([destination]),
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
      kind: "discard",
      id: createAnimationId(
        `move:${data.card.player}:${data.fromZone}:${data.fromIndex}:${data.toZone}:${data.toIndex}`
      ),
      card,
      from: source,
      to: buildDiscardAnchor(side),
      durationMs: ZONE_MOVE_ANIMATION_MS,
      hiddenSlotKeys: [],
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
      kind: "portal",
      id: createAnimationId(
        `move:${data.card.player}:${data.fromZone}:${data.fromIndex}:${data.toZone}:${data.toIndex}`
      ),
      card,
      from: source,
      to: destination,
      durationMs: ZONE_MOVE_ANIMATION_MS,
      hiddenSlotKeys: getHiddenSlotKeys([destination]),
      startedAtMs: performance.now(),
    };
  }

  return null;
}

function buildCombatAnimation(
  state: GameState,
  data: CombatDamageData,
  playerSlot: 0 | 1,
  cardDefIdMap: Map<number, CardMapping>
): BoardCombatAnimation | null {
  const attackerFrom = buildCombatAnchor(data.attacker, playerSlot);
  const attackerTo = buildCombatAnchor(data.defender, playerSlot);
  const defenderAnchor = buildCombatAnchor(data.defender, playerSlot);

  if (!attackerFrom || !attackerTo || !defenderAnchor) {
    return null;
  }

  const attacker = buildCombatCard(state, data.attacker, playerSlot, cardDefIdMap, true);
  const defender = buildCombatCard(state, data.defender, playerSlot, cardDefIdMap, true);

  if (!attacker || !defender) {
    return null;
  }

  // Keep the attacker upright during the lunge and reveal the tapped card after
  // the overlay finishes.
  attacker.tapped = false;

  const damageNumbers: BoardDamageNumberAnimation[] = [];
  if (data.attackerDamageTaken > 0) {
    damageNumbers.push({
      id: createAnimationId("dmg:attacker"),
      value: data.attackerDamageTaken,
      anchor: attackerFrom,
      color: "#ffb86c",
    });
  }
  if (data.defenderDamageTaken > 0) {
    damageNumbers.push({
      id: createAnimationId("dmg:defender"),
      value: data.defenderDamageTaken,
      anchor: defenderAnchor,
      color: "#ff7f7f",
    });
  }

  return {
    kind: "combat",
    id: createAnimationId(
      `combat:${data.attacker.player}:${data.attacker.zone}:${data.attacker.zoneIndex}:${data.defender.player}:${data.defender.zone}:${data.defender.zoneIndex}`
    ),
    attacker,
    defender,
    attackerFrom,
    attackerTo,
    defenderAnchor,
    damageNumbers,
    durationMs: COMBAT_ANIMATION_MS,
    hiddenSlotKeys: getHiddenSlotKeys([attackerFrom, defenderAnchor]),
    startedAtMs: performance.now(),
  };
}

export function buildBoardAnimationForLog(
  state: GameState,
  log: ProcessedGameLog,
  playerSlot: 0 | 1,
  cardDefIdMap: Map<number, CardMapping>
): BoardAnimation | null {
  switch (log.type) {
    case "ZONE_MOVED":
      return buildZoneMoveAnimation(state, log.data, playerSlot, cardDefIdMap);

    case "COMBAT_DAMAGE":
      return buildCombatAnimation(state, log.data, playerSlot, cardDefIdMap);

    default:
      return null;
  }
}
