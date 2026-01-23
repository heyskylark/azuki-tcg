import { create } from "zustand";

export type DragPhase = "idle" | "pickup" | "dragging" | "returning";
export type HoveredZone = "garden" | "alley" | "hand" | "leader" | null;
export type DragSourceType = "hand" | "alley" | "weapon" | null;

interface DragState {
  // Drag phase state
  dragPhase: DragPhase;

  // Drag source type - distinguishes hand card drags from alley card drags
  dragSourceType: DragSourceType;

  // Dragged card info (for hand cards, this is handIndex; for alley cards, use sourceAlleyIndex)
  draggedCardIndex: number | null;
  draggedCardCode: string | null;

  // Alley-specific drag info
  sourceAlleyIndex: number | null;

  // Position tracking
  targetPosition: [number, number, number]; // Where cursor is pointing
  currentPosition: [number, number, number]; // Actual card position (lags with spring)
  originalHandPosition: [number, number, number]; // For return animation (hand drags)
  originalAlleyPosition: [number, number, number]; // For return animation (alley drags)

  // Hover state
  hoveredZone: HoveredZone;
  hoveredSlotIndex: number | null;

  // Valid drop targets (computed from action mask)
  validGardenSlots: Set<number>;
  validAlleySlots: Set<number>;
  validWeaponAttachTargets: Set<number>; // 0-4 = garden slots, 5 = leader

  // Drop callback - called by DraggedCard when dropped on valid target
  onDropCallback: ((zone: "garden" | "alley" | "leader", slotIndex: number) => void) | null;

  // Actions
  startPickup: (
    handIndex: number,
    cardCode: string,
    handPosition: [number, number, number],
    validGardenSlots: Set<number>,
    validAlleySlots: Set<number>
  ) => void;
  startWeaponPickup: (
    handIndex: number,
    cardCode: string,
    handPosition: [number, number, number],
    validWeaponAttachTargets: Set<number>
  ) => void;
  startAlleyPickup: (
    alleyIndex: number,
    cardCode: string,
    alleyPosition: [number, number, number],
    validGardenSlots: Set<number>
  ) => void;
  startDragging: () => void;
  updateTargetPosition: (position: [number, number, number]) => void;
  updateCurrentPosition: (position: [number, number, number]) => void;
  setHoveredSlot: (zone: HoveredZone, slotIndex: number | null) => void;
  setOnDropCallback: (callback: ((zone: "garden" | "alley" | "leader", slotIndex: number) => void) | null) => void;
  drop: () => void;
  startReturning: () => void;
  cancelDrag: () => void;
  reset: () => void;
}

const initialState = {
  dragPhase: "idle" as DragPhase,
  dragSourceType: null as DragSourceType,
  draggedCardIndex: null,
  draggedCardCode: null,
  sourceAlleyIndex: null,
  targetPosition: [0, 0, 0] as [number, number, number],
  currentPosition: [0, 0, 0] as [number, number, number],
  originalHandPosition: [0, 0, 0] as [number, number, number],
  originalAlleyPosition: [0, 0, 0] as [number, number, number],
  hoveredZone: null,
  hoveredSlotIndex: null,
  validGardenSlots: new Set<number>(),
  validAlleySlots: new Set<number>(),
  validWeaponAttachTargets: new Set<number>(),
  onDropCallback: null as ((zone: "garden" | "alley" | "leader", slotIndex: number) => void) | null,
};

export const useDragStore = create<DragState>((set) => ({
  ...initialState,

  startPickup: (handIndex, cardCode, handPosition, validGardenSlots, validAlleySlots) =>
    set({
      dragPhase: "pickup",
      dragSourceType: "hand",
      draggedCardIndex: handIndex,
      draggedCardCode: cardCode,
      sourceAlleyIndex: null,
      originalHandPosition: handPosition,
      currentPosition: [...handPosition],
      targetPosition: [...handPosition],
      validGardenSlots,
      validAlleySlots,
      validWeaponAttachTargets: new Set<number>(),
      hoveredZone: "hand",
      hoveredSlotIndex: null,
    }),

  startWeaponPickup: (handIndex, cardCode, handPosition, validWeaponAttachTargets) =>
    set({
      dragPhase: "pickup",
      dragSourceType: "weapon",
      draggedCardIndex: handIndex,
      draggedCardCode: cardCode,
      sourceAlleyIndex: null,
      originalHandPosition: handPosition,
      currentPosition: [...handPosition],
      targetPosition: [...handPosition],
      validGardenSlots: new Set<number>(),
      validAlleySlots: new Set<number>(),
      validWeaponAttachTargets,
      hoveredZone: "hand",
      hoveredSlotIndex: null,
    }),

  startAlleyPickup: (alleyIndex, cardCode, alleyPosition, validGardenSlots) =>
    set({
      dragPhase: "pickup",
      dragSourceType: "alley",
      draggedCardIndex: null, // Not used for alley drags
      draggedCardCode: cardCode,
      sourceAlleyIndex: alleyIndex,
      originalAlleyPosition: alleyPosition,
      currentPosition: [...alleyPosition],
      targetPosition: [...alleyPosition],
      validGardenSlots,
      validAlleySlots: new Set<number>(), // Alley-to-alley not valid for gate
      validWeaponAttachTargets: new Set<number>(),
      hoveredZone: "alley",
      hoveredSlotIndex: null,
    }),

  startDragging: () =>
    set({
      dragPhase: "dragging",
    }),

  updateTargetPosition: (position) =>
    set({
      targetPosition: position,
    }),

  updateCurrentPosition: (position) =>
    set({
      currentPosition: position,
    }),

  setHoveredSlot: (zone, slotIndex) =>
    set({
      hoveredZone: zone,
      hoveredSlotIndex: slotIndex,
    }),

  setOnDropCallback: (callback) =>
    set({
      onDropCallback: callback,
    }),

  drop: () =>
    set({
      dragPhase: "idle",
      dragSourceType: null,
      draggedCardIndex: null,
      draggedCardCode: null,
      sourceAlleyIndex: null,
      validGardenSlots: new Set(),
      validAlleySlots: new Set(),
      validWeaponAttachTargets: new Set(),
      hoveredZone: null,
      hoveredSlotIndex: null,
      onDropCallback: null,
    }),

  startReturning: () =>
    set({
      dragPhase: "returning",
    }),

  cancelDrag: () =>
    set({
      ...initialState,
      dragSourceType: null,
      sourceAlleyIndex: null,
      validGardenSlots: new Set(),
      validAlleySlots: new Set(),
      validWeaponAttachTargets: new Set(),
    }),

  reset: () =>
    set({
      ...initialState,
      dragSourceType: null,
      sourceAlleyIndex: null,
      validGardenSlots: new Set(),
      validAlleySlots: new Set(),
      validWeaponAttachTargets: new Set(),
      onDropCallback: null,
    }),
}));
