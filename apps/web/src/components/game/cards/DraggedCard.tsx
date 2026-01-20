"use client";

import { useRef, useCallback, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useDragStore, type HoveredZone } from "@/stores/dragStore";
import { useAssets } from "@/contexts/AssetContext";
import { CARD_WIDTH, CARD_HEIGHT, CARD_DEPTH } from "@/components/game/cards/Card3D";

// Z thresholds for zone detection (based on Board.tsx constants)
const HAND_ZONE_Z = 6.5; // MY_HAND_Z = 7.4, threshold below
const ALLEY_ZONE_Z = 4.5; // MY_ALLEY_Z = 3.6, threshold above
const GARDEN_ZONE_Z = 2.5; // MY_GARDEN_Z = 1.5, threshold above

// Spring constants
const PICKUP_SPRING = 12; // Faster follow in pickup
const DRAG_SPRING = 8; // Smooth lag when dragging
const RETURN_SPRING = 10; // Speed for return animation
const RETURN_DISTANCE_THRESHOLD = 0.1; // How close to original position to complete return

/**
 * Determine which zone the cursor is in based on Z position.
 */
function getZoneFromZ(z: number): HoveredZone {
  if (z > HAND_ZONE_Z) return "hand";
  if (z > ALLEY_ZONE_Z) return "alley";
  if (z > GARDEN_ZONE_Z) return "garden";
  return "garden"; // Below garden is still garden zone
}

/**
 * The dragged card that follows the cursor with spring physics.
 * Renders only when a drag is in progress.
 */
export function DraggedCard() {
  const groupRef = useRef<THREE.Group>(null!);
  const currentPos = useRef(new THREE.Vector3());
  const currentScale = useRef(1);

  const { camera, gl } = useThree();
  const { getCardTexture } = useAssets();

  // Drag store state
  const dragPhase = useDragStore((state) => state.dragPhase);
  const draggedCardCode = useDragStore((state) => state.draggedCardCode);
  const targetPosition = useDragStore((state) => state.targetPosition);
  const originalHandPosition = useDragStore((state) => state.originalHandPosition);
  const hoveredZone = useDragStore((state) => state.hoveredZone);
  const hoveredSlotIndex = useDragStore((state) => state.hoveredSlotIndex);
  const validGardenSlots = useDragStore((state) => state.validGardenSlots);
  const validAlleySlots = useDragStore((state) => state.validAlleySlots);

  // Actions
  const updateTargetPosition = useDragStore((state) => state.updateTargetPosition);
  const updateCurrentPosition = useDragStore((state) => state.updateCurrentPosition);
  const setHoveredSlot = useDragStore((state) => state.setHoveredSlot);
  const cancelDrag = useDragStore((state) => state.cancelDrag);
  const startReturning = useDragStore((state) => state.startReturning);
  const reset = useDragStore((state) => state.reset);

  // Check if current hover is over a valid drop target
  const isOverValidTarget =
    (hoveredZone === "garden" &&
      hoveredSlotIndex !== null &&
      validGardenSlots.has(hoveredSlotIndex)) ||
    (hoveredZone === "alley" &&
      hoveredSlotIndex !== null &&
      validAlleySlots.has(hoveredSlotIndex));

  // Get texture for the dragged card
  const texture = draggedCardCode ? getCardTexture(draggedCardCode) : null;

  // Project pointer to XZ plane at given Y height
  const projectToXZPlane = useCallback(
    (clientX: number, clientY: number, targetY: number): THREE.Vector3 => {
      const rect = gl.domElement.getBoundingClientRect();
      const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);

      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -targetY);
      const intersection = new THREE.Vector3();
      raycaster.ray.intersectPlane(plane, intersection);

      return intersection;
    },
    [camera, gl]
  );

  // Handle global pointer move during drag
  const handlePointerMove = useCallback(
    (e: PointerEvent) => {
      if (dragPhase !== "pickup" && dragPhase !== "dragging") return;

      // Determine target Y based on phase
      const targetY = dragPhase === "pickup" ? 3 : 0.5;
      const intersection = projectToXZPlane(e.clientX, e.clientY, targetY);
      updateTargetPosition([intersection.x, targetY, intersection.z]);

      // Update zone detection during dragging phase
      if (dragPhase === "dragging") {
        const zone = getZoneFromZ(intersection.z);
        if (zone !== hoveredZone) {
          setHoveredSlot(zone, null);
        }
      }
    },
    [
      dragPhase,
      projectToXZPlane,
      updateTargetPosition,
      setHoveredSlot,
      hoveredZone,
    ]
  );

  // Handle global pointer up during drag
  const handlePointerUp = useCallback(() => {
    if (dragPhase === "pickup") {
      // Released while in pickup - animate back to hand
      startReturning();
    } else if (dragPhase === "dragging") {
      // Released while dragging - check if over valid target
      // Note: actual drop handling is done in Board.tsx via EmptyCardSlot.onDrop
      // If we reach here without a drop, start returning animation
      if (hoveredZone !== "garden" && hoveredZone !== "alley") {
        startReturning();
      } else if (!isOverValidTarget) {
        startReturning();
      }
      // If over valid target, the EmptyCardSlot.onDrop will handle it
    }
  }, [dragPhase, hoveredZone, isOverValidTarget, startReturning]);

  // Set up global event listeners
  useEffect(() => {
    if (dragPhase === "idle" || dragPhase === "returning") return;

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [dragPhase, handlePointerMove, handlePointerUp]);

  // Initialize position when drag starts
  useEffect(() => {
    if (dragPhase === "pickup" || dragPhase === "dragging") {
      currentPos.current.set(
        targetPosition[0],
        targetPosition[1],
        targetPosition[2]
      );
    }
  }, [dragPhase === "idle"]);

  // Animation frame
  useFrame((_, delta) => {
    if (!groupRef.current || dragPhase === "idle") return;

    const target = new THREE.Vector3(
      targetPosition[0],
      targetPosition[1],
      targetPosition[2]
    );
    const original = new THREE.Vector3(
      originalHandPosition[0],
      originalHandPosition[1],
      originalHandPosition[2]
    );

    // Determine target scale based on phase
    let targetScale = 1;
    if (dragPhase === "pickup") {
      targetScale = 1.5;
    } else if (dragPhase === "dragging") {
      targetScale = isOverValidTarget ? 1.1 : 1;
    } else if (dragPhase === "returning") {
      targetScale = 1;
    }

    // Spring animation
    if (dragPhase === "returning") {
      // Animate back to original hand position
      const springFactor = Math.min(delta * RETURN_SPRING, 1);
      currentPos.current.lerp(original, springFactor);
      currentScale.current += (targetScale - currentScale.current) * springFactor;

      // Check if return animation is complete
      if (currentPos.current.distanceTo(original) < RETURN_DISTANCE_THRESHOLD) {
        reset();
        return;
      }
    } else if (dragPhase === "pickup") {
      // Fast follow in pickup
      const springFactor = Math.min(delta * PICKUP_SPRING, 1);
      currentPos.current.lerp(target, springFactor);
      currentScale.current += (targetScale - currentScale.current) * springFactor;
    } else if (dragPhase === "dragging") {
      // Smooth lag when dragging
      const springFactor = Math.min(delta * DRAG_SPRING, 1);
      currentPos.current.lerp(target, springFactor);
      currentScale.current += (targetScale - currentScale.current) * springFactor;
    }

    // Update position and scale
    groupRef.current.position.copy(currentPos.current);
    groupRef.current.scale.setScalar(currentScale.current);

    // Update store with current position (for potential use by other components)
    updateCurrentPosition([
      currentPos.current.x,
      currentPos.current.y,
      currentPos.current.z,
    ]);
  });

  // Don't render if not dragging
  if (dragPhase === "idle" || !draggedCardCode) {
    return null;
  }

  // Determine card color based on state
  const getCardColor = () => {
    if (isOverValidTarget) return "#88ff88"; // Green tint over valid target
    return "#ffffff";
  };

  return (
    <group ref={groupRef}>
      {/* Card mesh */}
      <mesh castShadow>
        <boxGeometry args={[CARD_WIDTH, CARD_DEPTH, CARD_HEIGHT]} />
        {texture ? (
          <>
            {/* Side faces (edges) */}
            <meshStandardMaterial attach="material-0" color="#2a2a4e" />
            <meshStandardMaterial attach="material-1" color="#2a2a4e" />
            {/* Top face (card art) */}
            <meshStandardMaterial
              attach="material-2"
              map={texture}
              color={getCardColor()}
            />
            {/* Bottom face (card back) */}
            <meshStandardMaterial attach="material-3" color="#1a1a2e" />
            {/* Front/back edges */}
            <meshStandardMaterial attach="material-4" color="#2a2a4e" />
            <meshStandardMaterial attach="material-5" color="#2a2a4e" />
          </>
        ) : (
          <meshStandardMaterial color={getCardColor()} />
        )}
      </mesh>

      {/* Valid drop indicator glow */}
      {isOverValidTarget && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, CARD_DEPTH + 0.02, 0]}>
          <planeGeometry args={[CARD_WIDTH + 0.2, CARD_HEIGHT + 0.2]} />
          <meshBasicMaterial
            color="#44ff44"
            transparent
            opacity={0.3}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
    </group>
  );
}
