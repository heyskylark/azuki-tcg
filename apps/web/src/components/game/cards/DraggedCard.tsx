"use client";

import { useRef, useCallback, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useDragStore, type HoveredZone, type DragSourceType } from "@/stores/dragStore";
import { useAssets } from "@/contexts/AssetContext";
import { CARD_WIDTH, CARD_HEIGHT, CARD_DEPTH } from "@/components/game/cards/Card3D";

// Z thresholds for zone detection (based on Board.tsx constants)
const HAND_ZONE_Z = 5.5; // Keep pickup zoom until roughly halfway through the IKZ row
const ALLEY_ZONE_Z = 4.5; // MY_ALLEY_Z = 3.6, threshold above
const GARDEN_ZONE_Z = 2.5; // MY_GARDEN_Z = 1.5, threshold above

// Z threshold for alley-to-garden drag transition (from DraggableAlleyCard)
const ALLEY_TO_GARDEN_Z_THRESHOLD = 2.5;

// Slot calculation constants (from Board.tsx)
const SLOT_SPACING = 1.8;
const NUM_SLOTS = 5;

// Spell drop zone constants (center of board)
const SPELL_DROP_CENTER: [number, number] = [0, 0]; // [x, z]
const SPELL_DROP_WIDTH = CARD_WIDTH;
const SPELL_DROP_HEIGHT = CARD_HEIGHT;

// Spring constants
const PICKUP_SPRING = 12; // Faster follow in pickup
const DRAG_SPRING = 8; // Smooth lag when dragging
const RETURN_SPRING = 10; // Speed for return animation
const RETURN_DISTANCE_THRESHOLD = 0.1; // How close to original position to complete return
const DRAG_ZOOM_LIFT_Y = 0.2; // Extra lift when re-zooming near the hand buttons

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
 * Calculate slot index from X position.
 * Returns null if outside valid slot range.
 */
function getSlotIndexFromX(x: number): number | null {
  // Slots are centered at x = (index - 2) * SLOT_SPACING
  // So index = round(x / SLOT_SPACING) + 2
  const rawIndex = Math.round(x / SLOT_SPACING) + Math.floor(NUM_SLOTS / 2);
  if (rawIndex < 0 || rawIndex >= NUM_SLOTS) return null;
  return rawIndex;
}

function isInSpellDropZone(x: number, z: number): boolean {
  return (
    Math.abs(x - SPELL_DROP_CENTER[0]) <= SPELL_DROP_WIDTH / 2 &&
    Math.abs(z - SPELL_DROP_CENTER[1]) <= SPELL_DROP_HEIGHT / 2
  );
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
  const dragSourceType = useDragStore((state) => state.dragSourceType);
  const draggedCardCode = useDragStore((state) => state.draggedCardCode);
  const targetPosition = useDragStore((state) => state.targetPosition);
  const originalHandPosition = useDragStore((state) => state.originalHandPosition);
  const originalAlleyPosition = useDragStore((state) => state.originalAlleyPosition);
  const hoveredZone = useDragStore((state) => state.hoveredZone);
  const hoveredSlotIndex = useDragStore((state) => state.hoveredSlotIndex);
  const validGardenSlots = useDragStore((state) => state.validGardenSlots);
  const validAlleySlots = useDragStore((state) => state.validAlleySlots);

  // Actions
  const updateTargetPosition = useDragStore((state) => state.updateTargetPosition);
  const updateCurrentPosition = useDragStore((state) => state.updateCurrentPosition);
  const setHoveredSlot = useDragStore((state) => state.setHoveredSlot);
  const startDragging = useDragStore((state) => state.startDragging);
  const startReturning = useDragStore((state) => state.startReturning);
  const reset = useDragStore((state) => state.reset);
  const onDropCallback = useDragStore((state) => state.onDropCallback);

  // Check if current hover is over a valid drop target
  // For alley drags (gate), only garden is valid; for hand drags, both garden and alley are valid
  const isOverValidTarget =
    dragSourceType === "spell"
      ? hoveredZone === "spell"
      : dragSourceType === "alley"
        ? // Alley drag: only garden is valid target
          hoveredZone === "garden" &&
          hoveredSlotIndex !== null &&
          validGardenSlots.has(hoveredSlotIndex)
        : // Hand drag: both garden and alley are valid
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

      // For hand/weapon/spell drags during pickup: check if cursor left hand zone
      // (DraggableHandCard removes itself from DOM during pickup, so its handler doesn't fire)
      if (
        dragPhase === "pickup" &&
        (dragSourceType === "hand" || dragSourceType === "weapon" || dragSourceType === "spell")
      ) {
        if (intersection.z < HAND_ZONE_Z) {
          if (dragSourceType === "spell") {
            const isOverSpellZone = isInSpellDropZone(intersection.x, intersection.z);
            setHoveredSlot(isOverSpellZone ? "spell" : null, null);
          } else {
            const zone = getZoneFromZ(intersection.z);
            const slotIndex = (zone === "garden" || zone === "alley")
              ? getSlotIndexFromX(intersection.x)
              : null;

            // Update hovered slot BEFORE transitioning to dragging
            if (slotIndex !== null) {
              setHoveredSlot(zone, slotIndex);
            }
          }

          startDragging();
        }
      }

      // For alley drags during pickup: check if cursor moved into garden zone
      // This handles the transition from pickup to dragging for gate actions
      // (DraggableAlleyCard removes itself from DOM during pickup, so its handler doesn't fire)
      if (dragPhase === "pickup" && dragSourceType === "alley") {
        if (intersection.z < ALLEY_TO_GARDEN_Z_THRESHOLD) {
          const slotIndex = getSlotIndexFromX(intersection.x);
          console.log("[DraggedCard] Alley pickup entering garden zone:", {
            slotIndex,
            intersectionZ: intersection.z,
          });

          // Update hovered slot to garden zone BEFORE transitioning to dragging
          if (slotIndex !== null) {
            setHoveredSlot("garden", slotIndex);
          }

          startDragging();
        }
      }

      // Update zone and slot detection during dragging phase
      if (dragPhase === "dragging") {
        if (dragSourceType === "spell") {
          const isOverSpellZone = isInSpellDropZone(intersection.x, intersection.z);
          const nextZone = isOverSpellZone ? "spell" : null;
          if (nextZone !== hoveredZone) {
            setHoveredSlot(nextZone, null);
          }
        } else {
          const zone = getZoneFromZ(intersection.z);
          const slotIndex = (zone === "garden" || zone === "alley")
            ? getSlotIndexFromX(intersection.x)
            : null;

          // Always update if zone or slot changed
          if (zone !== hoveredZone || slotIndex !== hoveredSlotIndex) {
            setHoveredSlot(zone, slotIndex);
          }
        }
      }
    },
    [
      dragPhase,
      dragSourceType,
      projectToXZPlane,
      updateTargetPosition,
      setHoveredSlot,
      startDragging,
      hoveredZone,
      hoveredSlotIndex,
    ]
  );

  // Handle global pointer up during drag
  const handlePointerUp = useCallback(() => {
    // Read current state directly from store to avoid stale closure values
    const state = useDragStore.getState();
    const currentDragPhase = state.dragPhase;
    const currentDragSourceType = state.dragSourceType;
    const currentHoveredZone = state.hoveredZone;
    const currentHoveredSlotIndex = state.hoveredSlotIndex;
    const currentOnDropCallback = state.onDropCallback;
    const currentValidGardenSlots = state.validGardenSlots;
    const currentValidAlleySlots = state.validAlleySlots;

    // Debug logging for gate action drops
    console.log("[DraggedCard] handlePointerUp:", {
      dragPhase: currentDragPhase,
      dragSourceType: currentDragSourceType,
      hoveredZone: currentHoveredZone,
      hoveredSlotIndex: currentHoveredSlotIndex,
      validGardenSlots: [...currentValidGardenSlots],
      hasCallback: !!currentOnDropCallback,
      sourceAlleyIndex: state.sourceAlleyIndex,
    });

    // Check if over valid target with current values
    // For alley drags, only garden is valid; for hand drags, both garden and alley are valid
    const currentIsOverValidTarget =
      currentDragSourceType === "spell"
        ? currentHoveredZone === "spell"
        : currentDragSourceType === "alley"
          ? currentHoveredZone === "garden" &&
            currentHoveredSlotIndex !== null &&
            currentValidGardenSlots.has(currentHoveredSlotIndex)
          : (currentHoveredZone === "garden" &&
              currentHoveredSlotIndex !== null &&
              currentValidGardenSlots.has(currentHoveredSlotIndex)) ||
            (currentHoveredZone === "alley" &&
              currentHoveredSlotIndex !== null &&
              currentValidAlleySlots.has(currentHoveredSlotIndex));

    console.log("[DraggedCard] isOverValidTarget:", currentIsOverValidTarget);

    if (currentDragPhase === "pickup") {
      // Released while in pickup - animate back to original position
      console.log("[DraggedCard] Released in pickup phase, returning");
      startReturning();
    } else if (currentDragPhase === "dragging") {
      // Released while dragging - check if over valid target
      if (currentIsOverValidTarget && currentHoveredZone !== null) {
        // Over valid target - call the drop callback directly
        if (currentOnDropCallback) {
          console.log("[DraggedCard] Calling drop callback for", currentHoveredZone, currentHoveredSlotIndex);
          if (currentHoveredZone === "spell") {
            currentOnDropCallback("spell", null);
          } else if (
            currentHoveredSlotIndex !== null &&
            (currentHoveredZone === "garden" || currentHoveredZone === "alley")
          ) {
            currentOnDropCallback(currentHoveredZone, currentHoveredSlotIndex);
          } else {
            startReturning();
          }
        } else {
          console.log("[DraggedCard] No drop callback registered!");
          startReturning();
        }
      } else {
        // Not over valid target - animate back to original position
        console.log("[DraggedCard] Not over valid target, returning");
        startReturning();
      }
    }
  }, [startReturning]);

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

    const isHandDrag =
      dragSourceType === "hand" ||
      dragSourceType === "weapon" ||
      dragSourceType === "spell";
    const isInHandZone = targetPosition[2] > HAND_ZONE_Z;
    const zoomLift =
      dragPhase === "dragging" && isHandDrag && isInHandZone
        ? DRAG_ZOOM_LIFT_Y
        : 0;
    const target = new THREE.Vector3(
      targetPosition[0],
      targetPosition[1] + zoomLift,
      targetPosition[2]
    );
    // Use the correct original position based on drag source type
    const originalPos = dragSourceType === "alley" ? originalAlleyPosition : originalHandPosition;
    const original = new THREE.Vector3(
      originalPos[0],
      originalPos[1],
      originalPos[2]
    );

    // Determine target scale based on phase
    let targetScale = 1;
    if (dragPhase === "pickup") {
      targetScale = 2.25;
    } else if (dragPhase === "dragging") {
      targetScale = isHandDrag && isInHandZone ? 2.25 : isOverValidTarget ? 1.15 : 1;
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
      {/* Card mesh - raycast disabled to allow pointer events to pass through to slots */}
      <mesh castShadow raycast={() => {}}>
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

      {/* Valid drop indicator glow - raycast disabled */}
      {isOverValidTarget && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, CARD_DEPTH + 0.02, 0]} raycast={() => {}}>
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
