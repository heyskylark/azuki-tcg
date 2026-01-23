"use client";

import { useCallback, useEffect } from "react";
import { Text } from "@react-three/drei";
import { useGameState } from "@/contexts/GameStateContext";
import { useRoom } from "@/contexts/RoomContext";
import { Card3D, EmptyCardSlot, CARD_WIDTH, CARD_HEIGHT } from "@/components/game/cards/Card3D";
import { LeaderHealthDisplay } from "@/components/game/cards/CardStats";
import { DraggableHandCard } from "@/components/game/cards/DraggableHandCard";
import { DraggableAlleyCard } from "@/components/game/cards/DraggableAlleyCard";
import { useDragStore } from "@/stores/dragStore";
import {
  findValidAction,
  findValidGateAction,
  getValidEffectTargets,
  getValidGateSourceAlleySlots,
  buildEffectTargetAction,
} from "@/lib/game/actionValidation";
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
 * For player's alley, renders DraggableAlleyCard for cards that can be gated.
 */
function CardRow({
  cards,
  basePosition,
  zone,
  isOpponent = false,
  actionMask,
  onDropToSlot,
  validEffectTargets,
  onEffectTargetClick,
  targetIndexOffset,
}: {
  cards: (ResolvedCard | null)[];
  basePosition: [number, number, number];
  zone: "garden" | "alley";
  isOpponent?: boolean;
  actionMask?: SnapshotActionMask | null;
  onDropToSlot?: (zone: "garden" | "alley", slotIndex: number) => void;
  validEffectTargets?: Set<number>;
  onEffectTargetClick?: (targetIndex: number) => void;
  targetIndexOffset?: number;
}) {
  const [baseX, baseY, baseZ] = basePosition;

  // Get valid drop slots from drag store
  const validGardenSlots = useDragStore((state) => state.validGardenSlots);
  const validAlleySlots = useDragStore((state) => state.validAlleySlots);
  const dragPhase = useDragStore((state) => state.dragPhase);

  const isDragging = dragPhase === "pickup" || dragPhase === "dragging";
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

        // Calculate target index for ability targeting
        const targetIndex = (targetIndexOffset ?? 0) + index;
        const isAbilityTarget = validEffectTargets?.has(targetIndex) ?? false;

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
              />
            );
          }

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
              showStats={true}
              isAbilityTarget={isAbilityTarget}
              onAbilityTargetClick={
                isAbilityTarget && onEffectTargetClick
                  ? () => onEffectTargetClick(targetIndex)
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
 */
function LeaderCard({
  leader,
  position,
}: {
  leader: ResolvedLeader;
  position: [number, number, number];
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
      showStats={false}
    >
      {/* Large health display above leader */}
      <LeaderHealthDisplay
        currentHp={leader.curHp}
        position={[0, 0.2, -1.2]}
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
}: {
  cards: ResolvedHandCard[];
  position: [number, number, number];
  actionMask: SnapshotActionMask | null;
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

        return (
          <DraggableHandCard
            key={`hand-${index}-${card.cardCode}`}
            card={card}
            handIndex={index}
            position={[x, y, z]}
            rotation={[0, rotationY, 0]}
            actionMask={actionMask}
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
  validEffectTargets,
  onEffectTargetClick,
  gardenTargetOffset,
  alleyTargetOffset,
}: {
  board: ResolvedPlayerBoard;
  hand?: ResolvedHandCard[];
  isOpponent: boolean;
  actionMask?: SnapshotActionMask | null;
  onDropToSlot?: (zone: "garden" | "alley", slotIndex: number) => void;
  validEffectTargets?: Set<number>;
  onEffectTargetClick?: (targetIndex: number) => void;
  gardenTargetOffset?: number;
  alleyTargetOffset?: number;
}) {
  const gardenZ = isOpponent ? OPP_GARDEN_Z : MY_GARDEN_Z;
  const alleyZ = isOpponent ? OPP_ALLEY_Z : MY_ALLEY_Z;
  const ikzZ = isOpponent ? OPP_IKZ_Z : MY_IKZ_Z;

  return (
    <group>
      {/* Leader - right side, same row as garden */}
      <LeaderCard leader={board.leader} position={[RIGHT_SIDE_X, 0, gardenZ]} />

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
        validEffectTargets={validEffectTargets}
        onEffectTargetClick={onEffectTargetClick}
        targetIndexOffset={gardenTargetOffset}
      />

      {/* Alley (back row) */}
      <CardRow
        cards={board.alley}
        basePosition={[0, 0, alleyZ]}
        zone="alley"
        isOpponent={isOpponent}
        actionMask={actionMask}
        onDropToSlot={onDropToSlot}
        validEffectTargets={validEffectTargets}
        onEffectTargetClick={onEffectTargetClick}
        targetIndexOffset={alleyTargetOffset}
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
        />
      )}
    </group>
  );
}

// Target index offsets for ability targeting
// These map slot indices to the target index used by the C engine
// My Garden: 0-4, Opponent Garden: 5-9, My Alley: 10-14, Opponent Alley: 15-19
const MY_GARDEN_TARGET_OFFSET = 0;
const OPP_GARDEN_TARGET_OFFSET = 5;
const MY_ALLEY_TARGET_OFFSET = 10;
const OPP_ALLEY_TARGET_OFFSET = 15;

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
  const drop = useDragStore((state) => state.drop);
  const setOnDropCallback = useDragStore((state) => state.setOnDropCallback);

  // Handle dropping a hand card on a slot
  const handleDropToSlot = useCallback(
    (zone: "garden" | "alley", slotIndex: number) => {
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
    (zone: "garden" | "alley", gardenIndex: number) => {
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

  // Register the appropriate drop callback with the drag store when dragging
  useEffect(() => {
    if (dragPhase === "pickup" || dragPhase === "dragging") {
      // Use gate handler for alley drags, regular handler for hand drags
      const callback = dragSourceType === "alley" ? handleGateDropToGarden : handleDropToSlot;
      console.log("[Board] Setting drop callback:", {
        dragPhase,
        dragSourceType,
        isGateHandler: dragSourceType === "alley",
      });
      setOnDropCallback(callback);
    }
    return () => {
      // Don't clear callback here - let drop() or reset() handle it
    };
  }, [dragPhase, dragSourceType, handleDropToSlot, handleGateDropToGarden, setOnDropCallback]);

  // Check if we're in effect selection phase and compute valid targets
  const isInEffectSelection = gameState?.abilitySubphase === "EFFECT_SELECTION";
  const validEffectTargetsArray = isInEffectSelection
    ? getValidEffectTargets(gameState?.actionMask ?? null)
    : [];
  const validEffectTargets = new Set(validEffectTargetsArray);

  // Handle clicking an effect target
  const handleEffectTargetClick = useCallback(
    (targetIndex: number) => {
      if (!isInEffectSelection) return;
      send({
        type: "GAME_ACTION",
        action: buildEffectTargetAction(targetIndex),
      });
    },
    [isInEffectSelection, send]
  );

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
        validEffectTargets={isInEffectSelection ? validEffectTargets : undefined}
        onEffectTargetClick={isInEffectSelection ? handleEffectTargetClick : undefined}
        gardenTargetOffset={MY_GARDEN_TARGET_OFFSET}
        alleyTargetOffset={MY_ALLEY_TARGET_OFFSET}
      />

      {/* Opponent area (top) */}
      <PlayerArea
        board={gameState.opponentBoard}
        isOpponent={true}
        validEffectTargets={isInEffectSelection ? validEffectTargets : undefined}
        onEffectTargetClick={isInEffectSelection ? handleEffectTargetClick : undefined}
        gardenTargetOffset={OPP_GARDEN_TARGET_OFFSET}
        alleyTargetOffset={OPP_ALLEY_TARGET_OFFSET}
      />
    </group>
  );
}
