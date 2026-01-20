"use client";

import { useRef, useCallback } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import { Text } from "@react-three/drei";
import * as THREE from "three";
import { Card3D, CARD_WIDTH } from "@/components/game/cards/Card3D";
import { useDragStore } from "@/stores/dragStore";
import { useGameState } from "@/contexts/GameStateContext";
import { getValidSlotsForHandCard, canPlayCard } from "@/lib/game/actionValidation";
import type { SnapshotActionMask } from "@tcg/backend-core/types/ws";
import type { ResolvedHandCard } from "@/types/game";

// Z threshold for determining if cursor is in hand zone
// MY_HAND_Z = 7.4, so we use a threshold below that
const HAND_ZONE_Z_THRESHOLD = 6.5;

interface DraggableHandCardProps {
  card: ResolvedHandCard;
  handIndex: number;
  position: [number, number, number];
  rotation: [number, number, number];
  actionMask: SnapshotActionMask | null;
}

/**
 * A hand card that can be dragged to play.
 * Handles pickup, drag detection, and initiates the drag state.
 */
export function DraggableHandCard({
  card,
  handIndex,
  position,
  rotation,
  actionMask,
}: DraggableHandCardProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const { camera } = useThree();

  const dragPhase = useDragStore((state) => state.dragPhase);
  const draggedCardIndex = useDragStore((state) => state.draggedCardIndex);
  const startPickup = useDragStore((state) => state.startPickup);
  const startDragging = useDragStore((state) => state.startDragging);
  const updateTargetPosition = useDragStore((state) => state.updateTargetPosition);
  const startReturning = useDragStore((state) => state.startReturning);

  const { gameState } = useGameState();

  // Check if we're in an ability phase - disable dragging during abilities
  const isInAbilityPhase =
    gameState?.abilitySubphase !== undefined &&
    gameState.abilitySubphase !== "NONE";

  // Check if this card can be played (also disabled during ability phases)
  const isPlayable = !isInAbilityPhase && canPlayCard(actionMask, handIndex);

  // Check if this specific card is being dragged
  const isBeingDragged = draggedCardIndex === handIndex && dragPhase !== "idle";

  // Project pointer to XZ plane at given Y height
  const projectToXZPlane = useCallback(
    (clientX: number, clientY: number, targetY: number): THREE.Vector3 => {
      // Get normalized device coordinates
      const rect = document.querySelector("canvas")?.getBoundingClientRect();
      if (!rect) return new THREE.Vector3(0, targetY, 0);

      const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;

      // Create ray from camera through pointer
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);

      // Calculate intersection with horizontal plane at targetY
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -targetY);
      const intersection = new THREE.Vector3();
      raycaster.ray.intersectPlane(plane, intersection);

      return intersection;
    },
    [camera]
  );

  const handlePointerDown = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (!isPlayable || dragPhase !== "idle") return;

      e.stopPropagation();

      // Calculate world position of this card for return animation
      if (groupRef.current) {
        const worldPos = new THREE.Vector3();
        groupRef.current.getWorldPosition(worldPos);
        const handPosition: [number, number, number] = [
          worldPos.x,
          worldPos.y,
          worldPos.z,
        ];

        // Get valid slots for this card
        const { gardenSlots, alleySlots } = getValidSlotsForHandCard(
          actionMask,
          handIndex
        );

        // Start pickup phase
        startPickup(
          handIndex,
          card.cardCode,
          handPosition,
          gardenSlots,
          alleySlots
        );

        // Set initial target position based on pointer
        if (e.clientX !== undefined) {
          const intersection = projectToXZPlane(
            e.clientX,
            e.clientY,
            3 // Pickup Y height
          );
          updateTargetPosition([intersection.x, 3, intersection.z]);
        }
      }
    },
    [
      isPlayable,
      dragPhase,
      actionMask,
      handIndex,
      card.cardCode,
      startPickup,
      updateTargetPosition,
      projectToXZPlane,
    ]
  );

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (dragPhase !== "pickup" || draggedCardIndex !== handIndex) return;

      e.stopPropagation();

      if (e.clientX === undefined) return;

      // Project to XZ plane - Y depends on phase
      const targetY = dragPhase === "pickup" ? 3 : 0.5;
      const intersection = projectToXZPlane(
        e.clientX,
        e.clientY,
        targetY
      );

      updateTargetPosition([intersection.x, targetY, intersection.z]);

      // Check if cursor has left hand zone - transition to dragging
      if (intersection.z < HAND_ZONE_Z_THRESHOLD) {
        startDragging();
      }
    },
    [
      dragPhase,
      draggedCardIndex,
      handIndex,
      projectToXZPlane,
      updateTargetPosition,
      startDragging,
    ]
  );

  const handlePointerUp = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (dragPhase !== "pickup" || draggedCardIndex !== handIndex) return;

      e.stopPropagation();

      // Released while still in pickup phase - animate back to hand
      startReturning();
    },
    [dragPhase, draggedCardIndex, handIndex, startReturning]
  );

  // Don't render if this card is being dragged (DraggedCard renders it instead)
  if (isBeingDragged) {
    return null;
  }

  return (
    <group
      ref={groupRef}
      position={position}
      rotation={rotation}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      <Card3D
        cardCode={card.cardCode}
        imageUrl={card.imageUrl}
        name={card.name}
        position={[0, 0, 0]}
        showStats={false}
      >
        {/* IKZ cost badge */}
        <group position={[CARD_WIDTH * 0.35, 0.1, -0.8]}>
          <mesh rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.15, 16]} />
            <meshBasicMaterial color={isPlayable ? "#4a4a8e" : "#2a2a4e"} />
          </mesh>
          <Text
            position={[0, 0.02, 0]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.15}
            color={isPlayable ? "#88ff88" : "#666666"}
            anchorX="center"
            anchorY="middle"
            fontWeight="bold"
          >
            {card.ikzCost}
          </Text>
        </group>

        {/* Playable indicator glow */}
        {isPlayable && (
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
            <planeGeometry args={[CARD_WIDTH + 0.1, 2.1]} />
            <meshBasicMaterial
              color="#44aa44"
              transparent
              opacity={0.15}
              side={THREE.DoubleSide}
            />
          </mesh>
        )}
      </Card3D>
    </group>
  );
}
