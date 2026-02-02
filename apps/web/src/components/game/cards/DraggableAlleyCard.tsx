"use client";

import { useRef, useCallback } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { Card3D, CARD_WIDTH, CARD_HEIGHT } from "@/components/game/cards/Card3D";
import { useDragStore } from "@/stores/dragStore";
import { useGameState } from "@/contexts/GameStateContext";
import {
  getValidGateSourceAlleySlots,
  getValidGateTargetGardenSlots,
} from "@/lib/game/actionValidation";
import type { SnapshotActionMask } from "@tcg/backend-core/types/ws";
import type { ResolvedCard } from "@/types/game";

// Z threshold for determining if cursor has left alley zone into garden
// MY_ALLEY_Z = 3.6, MY_GARDEN_Z = 1.5, threshold is midpoint
const ALLEY_TO_GARDEN_Z_THRESHOLD = 2.5;

// Constants for garden slot calculation (must match DraggedCard.tsx)
const SLOT_SPACING = 1.8;
const NUM_SLOTS = 5;

interface DraggableAlleyCardProps {
  card: ResolvedCard;
  alleyIndex: number;
  position: [number, number, number];
  actionMask: SnapshotActionMask | null;
  isAbilityActivatable?: boolean;
  onAbilityActivate?: () => void;
}

/**
 * An alley card that can be dragged to gate into the garden.
 * Handles pickup, drag detection, and initiates the drag state for gate actions.
 */
export function DraggableAlleyCard({
  card,
  alleyIndex,
  position,
  actionMask,
  isAbilityActivatable = false,
  onAbilityActivate,
}: DraggableAlleyCardProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const { camera } = useThree();
  const pendingDragRef = useRef<{
    startX: number;
    startY: number;
    alleyPosition: [number, number, number];
  } | null>(null);

  const dragPhase = useDragStore((state) => state.dragPhase);
  const sourceAlleyIndex = useDragStore((state) => state.sourceAlleyIndex);
  const startAlleyPickup = useDragStore((state) => state.startAlleyPickup);
  const startDragging = useDragStore((state) => state.startDragging);
  const updateTargetPosition = useDragStore((state) => state.updateTargetPosition);
  const setHoveredSlot = useDragStore((state) => state.setHoveredSlot);
  const startReturning = useDragStore((state) => state.startReturning);

  const { gameState } = useGameState();

  // Check if we're in an ability phase - disable dragging during abilities
  const isInAbilityPhase =
    gameState?.abilitySubphase !== undefined &&
    gameState.abilitySubphase !== "NONE";

  // Check if this card can be gated (also disabled during ability phases)
  const canGate =
    !isInAbilityPhase &&
    getValidGateSourceAlleySlots(actionMask).has(alleyIndex);

  // Check if this specific card is being dragged
  const isBeingDragged =
    sourceAlleyIndex === alleyIndex && dragPhase !== "idle";

  // Project pointer to XZ plane at given Y height
  const projectToXZPlane = useCallback(
    (clientX: number, clientY: number, targetY: number): THREE.Vector3 => {
      const rect = document.querySelector("canvas")?.getBoundingClientRect();
      if (!rect) return new THREE.Vector3(0, targetY, 0);

      const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);

      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -targetY);
      const intersection = new THREE.Vector3();
      raycaster.ray.intersectPlane(plane, intersection);

      return intersection;
    },
    [camera]
  );

  const handlePointerDown = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (dragPhase !== "idle") return;

      e.stopPropagation();

      if (!groupRef.current) return;

      const worldPos = new THREE.Vector3();
      groupRef.current.getWorldPosition(worldPos);
      const alleyPosition: [number, number, number] = [
        worldPos.x,
        worldPos.y,
        worldPos.z,
      ];

      if (canGate && e.clientX !== undefined && e.clientY !== undefined) {
        pendingDragRef.current = {
          startX: e.clientX,
          startY: e.clientY,
          alleyPosition,
        };
        return;
      }

      if (!canGate && isAbilityActivatable && onAbilityActivate) {
        onAbilityActivate();
      }
    },
    [
      canGate,
      dragPhase,
      isAbilityActivatable,
      onAbilityActivate,
    ]
  );

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (pendingDragRef.current && canGate && dragPhase === "idle") {
        if (e.clientX === undefined || e.clientY === undefined) return;
        const dx = e.clientX - pendingDragRef.current.startX;
        const dy = e.clientY - pendingDragRef.current.startY;
        if (Math.hypot(dx, dy) < 6) {
          return;
        }

        const alleyPosition = pendingDragRef.current.alleyPosition;
        pendingDragRef.current = null;

        const validGardenSlots = getValidGateTargetGardenSlots(
          actionMask,
          alleyIndex
        );

        console.log("[DraggableAlleyCard] Starting pickup:", {
          alleyIndex,
          validGardenSlots: [...validGardenSlots],
        });

        startAlleyPickup(
          alleyIndex,
          card.cardCode,
          alleyPosition,
          validGardenSlots
        );

        const intersection = projectToXZPlane(e.clientX, e.clientY, 3);
        updateTargetPosition([intersection.x, 3, intersection.z]);
      }

      if (dragPhase !== "pickup" || sourceAlleyIndex !== alleyIndex) return;

      e.stopPropagation();

      if (e.clientX === undefined) return;

      // Project to XZ plane
      const targetY = dragPhase === "pickup" ? 3 : 0.5;
      const intersection = projectToXZPlane(e.clientX, e.clientY, targetY);

      updateTargetPosition([intersection.x, targetY, intersection.z]);

      // Check if cursor has moved into garden zone - transition to dragging
      if (intersection.z < ALLEY_TO_GARDEN_Z_THRESHOLD) {
        // Calculate the garden slot index from cursor X position
        // Slots are centered at x = (index - 2) * SLOT_SPACING
        // So index = round(x / SLOT_SPACING) + 2
        const slotIndex =
          Math.round(intersection.x / SLOT_SPACING) + Math.floor(NUM_SLOTS / 2);

        console.log("[DraggableAlleyCard] Transitioning to dragging:", {
          calculatedSlotIndex: slotIndex,
          intersectionX: intersection.x,
          intersectionZ: intersection.z,
        });

        // Update hovered slot to garden zone BEFORE transitioning to dragging
        // This fixes a race condition where releasing immediately after entering
        // the garden zone would see stale hoveredZone/hoveredSlotIndex values
        if (slotIndex >= 0 && slotIndex < NUM_SLOTS) {
          setHoveredSlot("garden", slotIndex);
        }

        startDragging();
      }
    },
    [
      canGate,
      dragPhase,
      sourceAlleyIndex,
      alleyIndex,
      actionMask,
      card.cardCode,
      projectToXZPlane,
      updateTargetPosition,
      setHoveredSlot,
      startDragging,
      startAlleyPickup,
    ]
  );

  const handlePointerUp = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (pendingDragRef.current) {
        pendingDragRef.current = null;
        if (isAbilityActivatable && onAbilityActivate) {
          e.stopPropagation();
          onAbilityActivate();
        }
        return;
      }

      if (dragPhase !== "pickup" || sourceAlleyIndex !== alleyIndex) return;

      e.stopPropagation();

      // Released while still in pickup phase - animate back to alley
      startReturning();
    },
    [
      dragPhase,
      sourceAlleyIndex,
      alleyIndex,
      startReturning,
      isAbilityActivatable,
      onAbilityActivate,
    ]
  );

  // Don't render if this card is being dragged (DraggedCard renders it instead)
  if (isBeingDragged) {
    return null;
  }

  return (
    <group
      ref={groupRef}
      position={position}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      <Card3D
        cardCode={card.cardCode}
        imageUrl={card.imageUrl}
        name={card.name}
        attack={card.curAtk}
        health={card.curHp}
        position={[0, 0, 0]}
        tapped={card.tapped}
        cooldown={card.cooldown}
        isFrozen={card.isFrozen}
        isShocked={card.isShocked}
        isEffectImmune={card.isEffectImmune}
        hasCharge={card.hasCharge}
        hasDefender={card.hasDefender}
        hasInfiltrate={card.hasInfiltrate}
        showStats={true}
        isAbilityActivatable={isAbilityActivatable}
        onAbilityActivate={onAbilityActivate}
      >
        {/* Gatable indicator glow - purple to distinguish from playable green */}
        {canGate && (
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
            <planeGeometry args={[CARD_WIDTH + 0.1, CARD_HEIGHT + 0.1]} />
            <meshBasicMaterial
              color="#8844aa"
              transparent
              opacity={0.2}
              side={THREE.DoubleSide}
            />
          </mesh>
        )}
      </Card3D>
    </group>
  );
}
