"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Text } from "@react-three/drei";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { Card3D, EmptyCardSlot, CARD_WIDTH, CARD_HEIGHT } from "@/components/game/cards/Card3D";
import { LeaderAttackDisplay, LeaderHealthDisplay } from "@/components/game/cards/CardStats";
import { DraggableHandCard } from "@/components/game/cards/DraggableHandCard";
import { DraggableAlleyCard } from "@/components/game/cards/DraggableAlleyCard";
import { AttackDragOverlay } from "@/components/game/attack/AttackDragOverlay";
import { useDragStore } from "@/stores/dragStore";
import {
  findValidAlleyAbilityAction,
  findValidAttackAction,
  findValidAction,
  findValidGardenOrLeaderAbilityAction,
  findValidGateAction,
  findValidWeaponAttachAction,
  getActivatableAlleySlots,
  getActivatableGardenOrLeaderSlots,
  getValidAttackers,
  getValidAttackTargetsForAttacker,
  getValidCostTargets,
  getValidEffectTargets,
  getValidGateSourceAlleySlots,
  buildCostTargetAction,
  buildEffectTargetAction,
} from "@/lib/game/actionValidation";
import { buildAbilityTargetMaps } from "@/lib/game/abilityTargeting";
import type {
  ResolvedPlayerBoard,
  ResolvedCard,
  ResolvedLeader,
  ResolvedGate,
  ResolvedHandCard,
  ResolvedIkz,
} from "@/types/game";
import type { SnapshotActionMask } from "@tcg/backend-core/types/ws";

// Board layout constants
const SLOT_SPACING = 1.8;
const IKZ_SPACING = 1.2;  // Tighter spacing for IKZ cards (up to 10)
const GARDEN_SLOTS = 5;
const ALLEY_SLOTS = 5;

// Z positions (depth into screen) - positive Z is toward player
// Card height is 2.0, gap of ~2.1 between rows
const MY_GARDEN_Z = 1.5;
const MY_ALLEY_Z = 3.6;   // Same gap as alley-to-IKZ
const MY_IKZ_Z = 5.7;     // Same gap as garden-to-alley
const MY_HAND_Z = 7.4;    // Hand slightly overlaps IKZ area

const OPP_GARDEN_Z = -1.5;
const OPP_ALLEY_Z = -3.6;  // Same gap as alley-to-IKZ
const OPP_IKZ_Z = -5.7;    // Same gap as garden-to-alley

// X positions - right side for leader/gate/deck/discard, left side for IKZ pile
const RIGHT_SIDE_X = 6;
const DECK_X = RIGHT_SIDE_X;
const DISCARD_X = RIGHT_SIDE_X + 1.8;
const IKZ_PILE_X = -6;

/**
 * Board surface - the green felt table.
 */
function BoardSurface() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
      <planeGeometry args={[22, 20]} />
      <meshStandardMaterial color="#1a472a" />
    </mesh>
  );
}

/**
 * Render a row of garden/alley slots with cards or empty slots.
 * Supports drag-and-drop for empty slots when not opponent's row.
 * Supports ability target highlighting during EFFECT_SELECTION phase.
 * Supports weapon attachment targeting on entities.
 * For player's alley, renders DraggableAlleyCard for cards that can be gated.
 */
function CardRow({
  cards,
  basePosition,
  zone,
  isOpponent = false,
  actionMask,
  onDropToSlot,
  onWeaponAttachToSlot,
  attackableSlots,
  attackTargets,
  onAttackStart,
  abilityTargets,
  onAbilityTargetClick,
  activatableSlots,
  onActivateAbility,
}: {
  cards: (ResolvedCard | null)[];
  basePosition: [number, number, number];
  zone: "garden" | "alley";
  isOpponent?: boolean;
  actionMask?: SnapshotActionMask | null;
  onDropToSlot?: (zone: "garden" | "alley", slotIndex: number) => void;
  onWeaponAttachToSlot?: (zone: "garden" | "alley" | "leader", slotIndex: number) => void;
  attackableSlots?: Set<number>;
  attackTargets?: Set<number>;
  onAttackStart?: (attackerIndex: number, position: [number, number, number], pointerPosition: [number, number, number]) => void;
  abilityTargets?: Map<number, number>;
  onAbilityTargetClick?: (targetIndex: number) => void;
  activatableSlots?: Set<number>;
  onActivateAbility?: (slotIndex: number) => void;
}) {
  const [baseX, baseY, baseZ] = basePosition;

  // Get valid drop slots from drag store
  const validGardenSlots = useDragStore((state) => state.validGardenSlots);
  const validAlleySlots = useDragStore((state) => state.validAlleySlots);
  const validWeaponAttachTargets = useDragStore((state) => state.validWeaponAttachTargets);
  const dragPhase = useDragStore((state) => state.dragPhase);
  const dragSourceType = useDragStore((state) => state.dragSourceType);

  const isDragging = dragPhase === "pickup" || dragPhase === "dragging";
  const isWeaponDrag = isDragging && dragSourceType === "weapon";
  const validSlots = zone === "garden" ? validGardenSlots : validAlleySlots;

  // For player's alley, compute which cards can be gated
  const gateableAlleySlots =
    !isOpponent && zone === "alley"
      ? getValidGateSourceAlleySlots(actionMask ?? null)
      : new Set<number>();

  return (
    <group>
      {cards.map((card, index) => {
        const x = (index - Math.floor(GARDEN_SLOTS / 2)) * SLOT_SPACING + baseX;
        const position: [number, number, number] = [x, baseY, baseZ];

        const abilityTargetIndex = abilityTargets?.get(index);
        const isAbilityTarget = abilityTargetIndex !== undefined;
        const canActivateAbility =
          !isOpponent && activatableSlots ? activatableSlots.has(index) : false;

        if (card) {
          // For player's alley cards that can be gated, use DraggableAlleyCard
          if (!isOpponent && zone === "alley" && gateableAlleySlots.has(index)) {
            return (
              <DraggableAlleyCard
                key={`alley-card-${index}-${card.cardCode}`}
                card={card}
                alleyIndex={index}
                position={position}
                actionMask={actionMask ?? null}
                isAbilityActivatable={canActivateAbility}
                onAbilityActivate={
                  canActivateAbility && onActivateAbility
                    ? () => onActivateAbility(index)
                    : undefined
                }
              />
            );
          }

          // Check if this entity is a valid weapon attachment target
          // Weapon targets: 0-4 for garden slots (only player's garden, not alley)
          const isWeaponAttachTarget =
            !isOpponent &&
            zone === "garden" &&
            isWeaponDrag &&
            validWeaponAttachTargets.has(index);

          const isAttackSource =
            !isOpponent &&
            zone === "garden" &&
            attackableSlots?.has(index);

          const isAttackTarget =
            isOpponent &&
            zone === "garden" &&
            attackTargets?.has(index);

          return (
            <Card3D
              key={`card-${index}-${card.cardCode}`}
              cardCode={card.cardCode}
              imageUrl={card.imageUrl}
              name={card.name}
              attack={card.curAtk}
              health={card.curHp}
              position={position}
              tapped={card.tapped}
              cooldown={card.cooldown}
              isFrozen={card.isFrozen}
              isShocked={card.isShocked}
              hasCharge={card.hasCharge}
              hasDefender={card.hasDefender}
              hasInfiltrate={card.hasInfiltrate}
              showStats={true}
              isAbilityTarget={isAbilityTarget}
              isWeaponTarget={isWeaponAttachTarget}
              isAttackTarget={isAttackTarget}
              isAbilityActivatable={canActivateAbility}
              onAbilityActivate={
                canActivateAbility && onActivateAbility
                  ? () => onActivateAbility(index)
                  : undefined
              }
              onAbilityTargetClick={
                isAbilityTarget && onAbilityTargetClick && abilityTargetIndex !== undefined
                  ? () => onAbilityTargetClick(abilityTargetIndex)
                  : undefined
              }
              onWeaponTargetClick={
                isWeaponAttachTarget && onWeaponAttachToSlot
                  ? () => onWeaponAttachToSlot("garden", index)
                  : undefined
              }
              onPointerDown={
                isAttackSource && onAttackStart
                  ? (event) => {
                      event.stopPropagation();
                      onAttackStart(
                        index,
                        position,
                        [event.point.x, event.point.y, event.point.z]
                      );
                    }
                  : undefined
              }
            />
          );
        }

        // Only show drop targets for player's slots (not opponent)
        const isValidDropTarget = !isOpponent && isDragging && validSlots.has(index);

        return (
          <EmptyCardSlot
            key={`empty-${index}`}
            position={position}
            label={`${isOpponent ? "O" : "G"}${index}`}
            slotIndex={index}
            zone={zone}
            isValidDropTarget={isValidDropTarget}
            onDrop={
              isValidDropTarget && onDropToSlot
                ? () => onDropToSlot(zone, index)
                : undefined
            }
          />
        );
      })}
    </group>
  );
}

/**
 * Leader card with health display.
 * Supports weapon attachment when leader is slot 5 in weapon targets.
 */
function LeaderCard({
  leader,
  position,
  isWeaponTarget = false,
  onWeaponTargetClick,
  isAbilityTarget = false,
  onAbilityTargetClick,
  isAbilityActivatable = false,
  onAbilityActivate,
  isAttackSource = false,
  isAttackTarget = false,
  onAttackStart,
}: {
  leader: ResolvedLeader;
  position: [number, number, number];
  isWeaponTarget?: boolean;
  onWeaponTargetClick?: () => void;
  isAbilityTarget?: boolean;
  onAbilityTargetClick?: () => void;
  isAbilityActivatable?: boolean;
  onAbilityActivate?: () => void;
  isAttackSource?: boolean;
  isAttackTarget?: boolean;
  onAttackStart?: (pointerPosition: [number, number, number]) => void;
}) {
  return (
    <Card3D
      cardCode={leader.cardCode}
      imageUrl={leader.imageUrl}
      name={leader.name}
      attack={leader.curAtk}
      health={leader.curHp}
      position={position}
      tapped={leader.tapped}
      cooldown={leader.cooldown}
      hasCharge={leader.hasCharge}
      hasDefender={leader.hasDefender}
      hasInfiltrate={leader.hasInfiltrate}
      showStats={false}
      isWeaponTarget={isWeaponTarget}
      onWeaponTargetClick={onWeaponTargetClick}
      isAbilityTarget={isAbilityTarget}
      onAbilityTargetClick={onAbilityTargetClick}
      isAbilityActivatable={isAbilityActivatable}
      onAbilityActivate={onAbilityActivate}
      isAttackTarget={isAttackTarget}
      onPointerDown={
        isAttackSource && onAttackStart
          ? (event) => {
              event.stopPropagation();
              onAttackStart([event.point.x, event.point.y, event.point.z]);
            }
          : undefined
      }
    >
      {/* Large health display above leader */}
      <LeaderHealthDisplay
        currentHp={leader.curHp}
        position={[0, 0.2, -1.2]}
      />

      {/* Small attack badge next to health */}
      <LeaderAttackDisplay
        currentAtk={leader.curAtk}
        position={[-0.6, 0.2, -1.2]}
      />
    </Card3D>
  );
}

/**
 * Gate card display.
 */
function GateCard({
  gate,
  position,
}: {
  gate: ResolvedGate;
  position: [number, number, number];
}) {
  return (
    <Card3D
      cardCode={gate.cardCode}
      imageUrl={gate.imageUrl}
      name={gate.name}
      position={position}
      tapped={gate.tapped}
      cooldown={gate.cooldown}
      showStats={false}
    />
  );
}

/**
 * IKZ pool display - IKZ cards spread in a row.
 */
function IkzPool({
  ikzArea,
  position,
}: {
  ikzArea: ResolvedIkz[];
  position: [number, number, number];
}) {
  const cardCount = ikzArea.length;

  return (
    <group position={position}>
      {ikzArea.map((ikz, index) => {
        // Center the row of cards with tighter spacing
        const x = (index - (cardCount - 1) / 2) * IKZ_SPACING;
        // Slight height increase per card to prevent z-fighting when overlapping
        const y = index * 0.01;
        return (
          <Card3D
            key={`ikz-${index}-${ikz.cardCode}`}
            cardCode={ikz.cardCode}
            imageUrl={ikz.imageUrl}
            name={ikz.name}
            position={[x, y, 0]}
            tapped={ikz.tapped}
            cooldown={ikz.cooldown}
            showStats={false}
          />
        );
      })}
    </group>
  );
}

/**
 * Deck stack representation.
 */
function DeckStack({
  count,
  position,
  label,
}: {
  count: number;
  position: [number, number, number];
  label: string;
}) {
  const stackHeight = 0.1 + count * 0.005;
  return (
    <group position={position}>
      {/* Stack of cards */}
      <mesh position={[0, stackHeight / 2, 0]}>
        <boxGeometry args={[CARD_WIDTH * 0.8, stackHeight, CARD_HEIGHT * 0.8]} />
        <meshStandardMaterial color="#2a2a4e" />
      </mesh>
      {/* Count text */}
      <Text
        position={[0, stackHeight + 0.05, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.25}
        color="white"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {count}
      </Text>
    </group>
  );
}

/**
 * Discard pile representation.
 */
function DiscardPile({
  count,
  position,
}: {
  count: number;
  position: [number, number, number];
}) {
  const stackHeight = Math.max(0.02, count * 0.003);
  return (
    <group position={position}>
      {/* Pile of cards - slightly messier than deck */}
      <mesh position={[0, stackHeight / 2, 0]}>
        <boxGeometry args={[CARD_WIDTH * 0.8, stackHeight, CARD_HEIGHT * 0.8]} />
        <meshStandardMaterial color="#3a2a2e" />
      </mesh>
      {/* Count text */}
      <Text
        position={[0, stackHeight + 0.05, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.25}
        color="#aa8888"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {count}
      </Text>
    </group>
  );
}

/**
 * IKZ pile stack representation.
 */
function IkzPileStack({
  count,
  position,
}: {
  count: number;
  position: [number, number, number];
}) {
  const stackHeight = 0.1 + count * 0.005;
  return (
    <group position={position}>
      {/* Stack of IKZ cards */}
      <mesh position={[0, stackHeight / 2, 0]}>
        <boxGeometry args={[CARD_WIDTH * 0.8, stackHeight, CARD_HEIGHT * 0.8]} />
        <meshStandardMaterial color="#2a4a2e" />
      </mesh>
      {/* Count text */}
      <Text
        position={[0, stackHeight + 0.05, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.25}
        color="#88ff88"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {count}
      </Text>
    </group>
  );
}

/**
 * Player's hand display - fan layout at bottom of screen.
 * Uses DraggableHandCard for drag-and-drop card playing.
 */
function HandDisplay({
  cards,
  position,
  actionMask,
  abilityTargets,
  onAbilityTargetClick,
}: {
  cards: ResolvedHandCard[];
  position: [number, number, number];
  actionMask: SnapshotActionMask | null;
  abilityTargets?: Map<number, number>;
  onAbilityTargetClick?: (targetIndex: number) => void;
}) {
  const cardCount = cards.length;
  const draggedCardIndex = useDragStore((state) => state.draggedCardIndex);
  const dragPhase = useDragStore((state) => state.dragPhase);

  return (
    <group position={position}>
      {cards.map((card, index) => {
        // Fan layout calculation
        const centerOffset = (cardCount - 1) / 2;
        const normalizedIndex = index - centerOffset;
        const x = normalizedIndex * (SLOT_SPACING * 0.7);
        // Slight height increase per card to prevent z-fighting when overlapping
        const y = index * 0.01;
        const z = Math.abs(normalizedIndex) * 0.1; // Slight arc
        const rotationY = normalizedIndex * -0.05; // Fan angle

        // Hide the card in hand if it's being dragged
        const isBeingDragged = draggedCardIndex === index && dragPhase !== "idle";
        if (isBeingDragged) {
          return null;
        }

        const abilityTargetIndex = abilityTargets?.get(index);
        return (
          <DraggableHandCard
            key={`hand-${index}-${card.cardCode}`}
            card={card}
            handIndex={index}
            position={[x, y, z]}
            rotation={[0, rotationY, 0]}
            actionMask={actionMask}
            abilityTargetIndex={abilityTargetIndex}
            onAbilityTargetClick={onAbilityTargetClick}
          />
        );
      })}
    </group>
  );
}

/**
 * One player's side of the board.
 */
function PlayerArea({
  board,
  hand,
  isOpponent,
  actionMask,
  onDropToSlot,
  onWeaponAttachToSlot,
  attackableSlots,
  attackTargets,
  onAttackStart,
  isLeaderWeaponTarget,
  abilityTargets,
  onAbilityTargetClick,
  activatableGardenSlots,
  activatableAlleySlots,
  canActivateLeaderAbility,
  onActivateGardenOrLeaderAbility,
  onActivateAlleyAbility,
}: {
  board: ResolvedPlayerBoard;
  hand?: ResolvedHandCard[];
  isOpponent: boolean;
  actionMask?: SnapshotActionMask | null;
  onDropToSlot?: (zone: "garden" | "alley", slotIndex: number) => void;
  onWeaponAttachToSlot?: (zone: "garden" | "alley" | "leader", slotIndex: number) => void;
  attackableSlots?: Set<number>;
  attackTargets?: Set<number>;
  onAttackStart?: (attackerIndex: number, position: [number, number, number], pointerPosition: [number, number, number]) => void;
  isLeaderWeaponTarget?: boolean;
  abilityTargets?: {
    gardenTargets?: Map<number, number>;
    handTargets?: Map<number, number>;
    leaderTargetIndex?: number | null;
  };
  onAbilityTargetClick?: (targetIndex: number) => void;
  activatableGardenSlots?: Set<number>;
  activatableAlleySlots?: Set<number>;
  canActivateLeaderAbility?: boolean;
  onActivateGardenOrLeaderAbility?: (slotIndex: number) => void;
  onActivateAlleyAbility?: (alleyIndex: number) => void;
}) {
  const gardenZ = isOpponent ? OPP_GARDEN_Z : MY_GARDEN_Z;
  const alleyZ = isOpponent ? OPP_ALLEY_Z : MY_ALLEY_Z;
  const ikzZ = isOpponent ? OPP_IKZ_Z : MY_IKZ_Z;

  const isLeaderAttackSource = !isOpponent && attackableSlots?.has(5);
  const isLeaderAttackTarget = isOpponent && attackTargets?.has(5);
  const leaderTargetIndex = abilityTargets?.leaderTargetIndex;
  const isLeaderAbilityTarget =
    leaderTargetIndex !== null && leaderTargetIndex !== undefined;
  const isLeaderAbilityActivatable = !isOpponent && !!canActivateLeaderAbility;

  return (
    <group>
      {/* Leader - right side, same row as garden */}
      <LeaderCard
        leader={board.leader}
        position={[RIGHT_SIDE_X, 0, gardenZ]}
        isWeaponTarget={isLeaderWeaponTarget}
        onWeaponTargetClick={
          isLeaderWeaponTarget && onWeaponAttachToSlot
            ? () => onWeaponAttachToSlot("leader", 5)
            : undefined
        }
        isAbilityTarget={isLeaderAbilityTarget}
        onAbilityTargetClick={
          isLeaderAbilityTarget && onAbilityTargetClick && leaderTargetIndex !== null && leaderTargetIndex !== undefined
            ? () => onAbilityTargetClick(leaderTargetIndex)
            : undefined
        }
        isAbilityActivatable={isLeaderAbilityActivatable}
        onAbilityActivate={
          isLeaderAbilityActivatable && onActivateGardenOrLeaderAbility
            ? () => onActivateGardenOrLeaderAbility(5)
            : undefined
        }
        isAttackSource={isLeaderAttackSource}
        isAttackTarget={isLeaderAttackTarget}
        onAttackStart={
          isLeaderAttackSource && onAttackStart
            ? (pointerPosition) => onAttackStart(5, [RIGHT_SIDE_X, 0, gardenZ], pointerPosition)
            : undefined
        }
      />

      {/* Gate - right side, same row as alley */}
      <GateCard gate={board.gate} position={[RIGHT_SIDE_X, 0, alleyZ]} />

      {/* Garden (front row) */}
      <CardRow
        cards={board.garden}
        basePosition={[0, 0, gardenZ]}
        zone="garden"
        isOpponent={isOpponent}
        actionMask={actionMask}
        onDropToSlot={onDropToSlot}
        onWeaponAttachToSlot={onWeaponAttachToSlot}
        attackableSlots={attackableSlots}
        attackTargets={attackTargets}
        onAttackStart={onAttackStart}
        abilityTargets={abilityTargets?.gardenTargets}
        onAbilityTargetClick={onAbilityTargetClick}
        activatableSlots={!isOpponent ? activatableGardenSlots : undefined}
        onActivateAbility={!isOpponent ? onActivateGardenOrLeaderAbility : undefined}
      />

      {/* Alley (back row) */}
      <CardRow
        cards={board.alley}
        basePosition={[0, 0, alleyZ]}
        zone="alley"
        isOpponent={isOpponent}
        actionMask={actionMask}
        onDropToSlot={onDropToSlot}
        onWeaponAttachToSlot={onWeaponAttachToSlot}
        attackTargets={attackTargets}
        activatableSlots={!isOpponent ? activatableAlleySlots : undefined}
        onActivateAbility={!isOpponent ? onActivateAlleyAbility : undefined}
      />

      {/* Deck - below the gate */}
      <DeckStack
        count={board.deckCount}
        position={[DECK_X, 0, ikzZ]}
        label={isOpponent ? "Opp Deck" : "My Deck"}
      />

      {/* Discard pile - to the right of deck */}
      <DiscardPile
        count={board.discardCount}
        position={[DISCARD_X, 0, ikzZ]}
      />

      {/* IKZ Pile - to the left of IKZ area */}
      <IkzPileStack
        count={board.ikzPileCount}
        position={[IKZ_PILE_X, 0, ikzZ]}
      />

      {/* IKZ Pool - spread out below alley */}
      <IkzPool
        ikzArea={board.ikzArea}
        position={[0, 0, ikzZ]}
      />

      {/* Hand (only for player, not opponent) - slightly elevated to overlap IKZ */}
      {!isOpponent && hand && hand.length > 0 && (
        <HandDisplay
          cards={hand}
          position={[0, 0.05, MY_HAND_Z]}
          actionMask={actionMask ?? null}
          abilityTargets={abilityTargets?.handTargets}
          onAbilityTargetClick={onAbilityTargetClick}
        />
      )}
    </group>
  );
}

/**
 * Main game board component.
 * Renders the complete game state with both players' boards.
 * Handles drag-and-drop card playing via WebSocket.
 */
export function Board() {
  const { gameState } = useGameState();
  const { send } = useRoom();

  // Drag store actions
  const draggedCardIndex = useDragStore((state) => state.draggedCardIndex);
  const dragPhase = useDragStore((state) => state.dragPhase);
  const dragSourceType = useDragStore((state) => state.dragSourceType);
  const validWeaponAttachTargets = useDragStore((state) => state.validWeaponAttachTargets);
  const drop = useDragStore((state) => state.drop);
  const setOnDropCallback = useDragStore((state) => state.setOnDropCallback);

  const isInAbilityPhase =
    gameState?.abilitySubphase !== undefined &&
    gameState.abilitySubphase !== "NONE";

  const validAttackers = useMemo(
    () => getValidAttackers(gameState?.actionMask ?? null),
    [gameState?.actionMask]
  );

  const [attackDrag, setAttackDrag] = useState<{
    active: boolean;
    attackerIndex: number | null;
    attackerPosition: [number, number, number];
    initialPointerPosition: [number, number, number];
    validTargets: Set<number>;
  }>({
    active: false,
    attackerIndex: null,
    attackerPosition: [0, 0, 0],
    initialPointerPosition: [0, 0, 0],
    validTargets: new Set<number>(),
  });

  const abilityActivationEnabled =
    dragPhase === "idle" && !attackDrag.active && !isInAbilityPhase;

  const abilityActivationSlots = useMemo(() => {
    if (!abilityActivationEnabled) {
      return { gardenSlots: new Set<number>(), leaderActive: false };
    }
    const slots = getActivatableGardenOrLeaderSlots(gameState?.actionMask ?? null);
    const gardenSlots = new Set<number>();
    let leaderActive = false;
    slots.forEach((slot) => {
      if (slot === 5) {
        leaderActive = true;
      } else if (slot >= 0 && slot < 5) {
        gardenSlots.add(slot);
      }
    });
    return { gardenSlots, leaderActive };
  }, [abilityActivationEnabled, gameState?.actionMask]);

  const activatableAlleySlots = useMemo(() => {
    if (!abilityActivationEnabled) {
      return new Set<number>();
    }
    return getActivatableAlleySlots(gameState?.actionMask ?? null);
  }, [abilityActivationEnabled, gameState?.actionMask]);

  const cancelAttackDrag = useCallback(() => {
    setAttackDrag({
      active: false,
      attackerIndex: null,
      attackerPosition: [0, 0, 0],
      initialPointerPosition: [0, 0, 0],
      validTargets: new Set<number>(),
    });
  }, []);

  const startAttackDrag = useCallback(
    (
      attackerIndex: number,
      attackerPosition: [number, number, number],
      pointerPosition: [number, number, number]
    ) => {
      if (!gameState?.actionMask) return;
      if (dragPhase !== "idle" || attackDrag.active || isInAbilityPhase) return;
      if (!validAttackers.has(attackerIndex)) return;

      const validTargets = getValidAttackTargetsForAttacker(
        gameState.actionMask,
        attackerIndex
      );
      if (validTargets.size === 0) return;

      setAttackDrag({
        active: true,
        attackerIndex,
        attackerPosition,
        initialPointerPosition: pointerPosition,
        validTargets,
      });
    },
    [attackDrag.active, dragPhase, gameState?.actionMask, isInAbilityPhase, validAttackers]
  );

  // Handle dropping a hand card on a slot
  const handleDropToSlot = useCallback(
    (zone: "garden" | "alley" | "leader", slotIndex: number) => {
      // Leader zone not valid for regular entity drops
      if (zone === "leader") return;

      if (draggedCardIndex === null || !gameState?.actionMask) {
        return;
      }

      // Find the valid action tuple for this drop
      const action = findValidAction(
        gameState.actionMask,
        draggedCardIndex,
        zone,
        slotIndex
      );

      if (action) {
        // Send the game action via WebSocket
        send({
          type: "GAME_ACTION",
          action,
        });

        // Complete the drop - clear drag state
        drop();
      }
    },
    [draggedCardIndex, gameState?.actionMask, send, drop]
  );

  // Handle gate drop - moving an alley card to garden
  const handleGateDropToGarden = useCallback(
    (zone: "garden" | "alley" | "leader", gardenIndex: number) => {
      console.log("[Board] handleGateDropToGarden called:", { zone, gardenIndex });

      // Gate action only targets garden
      if (zone !== "garden") {
        console.log("[Board] Gate drop ignored - zone is not garden");
        return;
      }

      const currentSourceAlleyIndex = useDragStore.getState().sourceAlleyIndex;
      console.log("[Board] sourceAlleyIndex:", currentSourceAlleyIndex);

      if (currentSourceAlleyIndex === null || !gameState?.actionMask) {
        console.log("[Board] Gate drop failed - missing sourceAlleyIndex or actionMask");
        return;
      }

      // Find the valid gate action tuple for this drop
      const action = findValidGateAction(
        gameState.actionMask,
        currentSourceAlleyIndex,
        gardenIndex
      );

      console.log("[Board] findValidGateAction result:", action);

      if (action) {
        // Send the game action via WebSocket
        console.log("[Board] Sending GAME_ACTION:", action);
        send({
          type: "GAME_ACTION",
          action,
        });

        // Complete the drop - clear drag state
        drop();
      } else {
        console.log("[Board] No valid gate action found");
      }
    },
    [gameState?.actionMask, send, drop]
  );

  // Handle weapon attachment drop - attaching a weapon from hand to an entity
  const handleWeaponAttachDrop = useCallback(
    (zone: "garden" | "alley" | "leader", entitySlot: number) => {
      console.log("[Board] handleWeaponAttachDrop called:", { zone, entitySlot });

      // Weapon attach targets are: garden (0-4) or leader (5)
      // Convert zone + slotIndex to entitySlot
      let targetSlot: number;
      if (zone === "leader") {
        targetSlot = 5; // Leader is slot 5
      } else if (zone === "garden") {
        targetSlot = entitySlot; // Garden slots 0-4
      } else {
        console.log("[Board] Weapon attach ignored - invalid zone:", zone);
        return;
      }

      const currentDraggedCardIndex = useDragStore.getState().draggedCardIndex;
      console.log("[Board] draggedCardIndex:", currentDraggedCardIndex);

      if (currentDraggedCardIndex === null || !gameState?.actionMask) {
        console.log("[Board] Weapon attach failed - missing draggedCardIndex or actionMask");
        return;
      }

      // Find the valid weapon attach action tuple for this drop
      const action = findValidWeaponAttachAction(
        gameState.actionMask,
        currentDraggedCardIndex,
        targetSlot
      );

      console.log("[Board] findValidWeaponAttachAction result:", action);

      if (action) {
        // Send the game action via WebSocket
        console.log("[Board] Sending GAME_ACTION:", action);
        send({
          type: "GAME_ACTION",
          action,
        });

        // Complete the drop - clear drag state
        drop();
      } else {
        console.log("[Board] No valid weapon attach action found");
      }
    },
    [gameState?.actionMask, send, drop]
  );

  const handleActivateGardenOrLeaderAbility = useCallback(
    (slotIndex: number) => {
      if (!abilityActivationEnabled || !gameState?.actionMask) return;

      const action = findValidGardenOrLeaderAbilityAction(
        gameState.actionMask,
        slotIndex
      );

      if (action) {
        send({
          type: "GAME_ACTION",
          action,
        });
      }
    },
    [abilityActivationEnabled, gameState?.actionMask, send]
  );

  const handleActivateAlleyAbility = useCallback(
    (alleyIndex: number) => {
      if (!abilityActivationEnabled || !gameState?.actionMask) return;

      const action = findValidAlleyAbilityAction(
        gameState.actionMask,
        alleyIndex
      );

      if (action) {
        send({
          type: "GAME_ACTION",
          action,
        });
      }
    },
    [abilityActivationEnabled, gameState?.actionMask, send]
  );

  const handleAttackCommit = useCallback(
    (targetIndex: number) => {
      if (!gameState?.actionMask) {
        cancelAttackDrag();
        return;
      }

      const attackerIndex = attackDrag.attackerIndex;
      if (attackerIndex === null) {
        cancelAttackDrag();
        return;
      }

      const action = findValidAttackAction(
        gameState.actionMask,
        attackerIndex,
        targetIndex
      );

      if (action) {
        send({
          type: "GAME_ACTION",
          action,
        });
      }

      cancelAttackDrag();
    },
    [attackDrag.attackerIndex, cancelAttackDrag, gameState?.actionMask, send]
  );

  // Register the appropriate drop callback with the drag store when dragging
  useEffect(() => {
    if (dragPhase === "pickup" || dragPhase === "dragging") {
      // Use appropriate handler based on drag source type
      let callback: (zone: "garden" | "alley" | "leader", slotIndex: number) => void;
      if (dragSourceType === "alley") {
        callback = handleGateDropToGarden;
      } else if (dragSourceType === "weapon") {
        callback = handleWeaponAttachDrop;
      } else {
        callback = handleDropToSlot;
      }
      console.log("[Board] Setting drop callback:", {
        dragPhase,
        dragSourceType,
      });
      setOnDropCallback(callback);
    }
    return () => {
      // Don't clear callback here - let drop() or reset() handle it
    };
  }, [dragPhase, dragSourceType, handleDropToSlot, handleGateDropToGarden, handleWeaponAttachDrop, setOnDropCallback]);

  useEffect(() => {
    if (attackDrag.active && !gameState?.actionMask) {
      cancelAttackDrag();
    }
  }, [attackDrag.active, cancelAttackDrag, gameState?.actionMask]);

  useEffect(() => {
    if (attackDrag.active && isInAbilityPhase) {
      cancelAttackDrag();
    }
  }, [attackDrag.active, cancelAttackDrag, isInAbilityPhase]);

  const isInCostSelection = gameState?.abilitySubphase === "COST_SELECTION";
  const isInEffectSelection = gameState?.abilitySubphase === "EFFECT_SELECTION";

  const abilityTargets = useMemo(() => {
    if (!gameState) return null;
    if (!isInCostSelection && !isInEffectSelection) return null;
    const targetType = isInCostSelection
      ? gameState.abilityCostTargetType
      : gameState.abilityEffectTargetType;
    if (targetType === undefined || targetType === null) return null;
    const targetIndices = isInCostSelection
      ? getValidCostTargets(gameState.actionMask ?? null)
      : getValidEffectTargets(gameState.actionMask ?? null);
    return buildAbilityTargetMaps({
      targetType,
      targetIndices,
      gameState,
    });
  }, [gameState, isInCostSelection, isInEffectSelection]);

  const handleAbilityTargetClick = useCallback(
    (targetIndex: number) => {
      if (!isInCostSelection && !isInEffectSelection) return;
      const action = isInCostSelection
        ? buildCostTargetAction(targetIndex)
        : buildEffectTargetAction(targetIndex);
      send({
        type: "GAME_ACTION",
        action,
      });
    },
    [isInCostSelection, isInEffectSelection, send]
  );

  // Check if leader is a valid weapon attachment target
  // Leader is slot 5 in the weapon targets
  const isDraggingWeapon = (dragPhase === "pickup" || dragPhase === "dragging") && dragSourceType === "weapon";
  const isLeaderWeaponTarget = isDraggingWeapon && validWeaponAttachTargets.has(5);

  if (!gameState) {
    return (
      <group>
        <BoardSurface />
        <Text
          position={[0, 1, 0]}
          fontSize={0.5}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          Waiting for game state...
        </Text>
      </group>
    );
  }

  return (
    <group>
      {/* Board surface */}
      <BoardSurface />

      {/* My area (bottom) */}
      <PlayerArea
        board={gameState.myBoard}
        hand={gameState.myHand}
        isOpponent={false}
        actionMask={gameState.actionMask}
        onDropToSlot={handleDropToSlot}
        onWeaponAttachToSlot={handleWeaponAttachDrop}
        attackableSlots={validAttackers}
        onAttackStart={startAttackDrag}
        isLeaderWeaponTarget={isLeaderWeaponTarget}
        activatableGardenSlots={abilityActivationSlots.gardenSlots}
        activatableAlleySlots={activatableAlleySlots}
        canActivateLeaderAbility={abilityActivationSlots.leaderActive}
        onActivateGardenOrLeaderAbility={handleActivateGardenOrLeaderAbility}
        onActivateAlleyAbility={handleActivateAlleyAbility}
        abilityTargets={
          abilityTargets
            ? {
                gardenTargets: abilityTargets.selfGarden,
                handTargets: abilityTargets.hand,
                leaderTargetIndex: abilityTargets.selfLeader,
              }
            : undefined
        }
        onAbilityTargetClick={abilityTargets ? handleAbilityTargetClick : undefined}
      />

      {/* Opponent area (top) */}
      <PlayerArea
        board={gameState.opponentBoard}
        isOpponent={true}
        attackTargets={attackDrag.active ? attackDrag.validTargets : undefined}
        abilityTargets={
          abilityTargets
            ? {
                gardenTargets: abilityTargets.opponentGarden,
                leaderTargetIndex: abilityTargets.opponentLeader,
              }
            : undefined
        }
        onAbilityTargetClick={abilityTargets ? handleAbilityTargetClick : undefined}
      />

      <AttackDragOverlay
        isActive={attackDrag.active}
        attackerPosition={attackDrag.attackerPosition}
        initialPointerPosition={attackDrag.initialPointerPosition}
        validTargets={attackDrag.validTargets}
        onCommit={handleAttackCommit}
        onCancel={cancelAttackDrag}
      />
    </group>
  );
}
