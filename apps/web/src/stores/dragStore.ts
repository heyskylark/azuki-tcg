import { create } from "zustand";

export type DragPhase = "idle" | "pickup" | "dragging" | "returning";
export type HoveredZone = "garden" | "alley" | "hand" | null;

interface DragState {
  // Drag phase state
  dragPhase: DragPhase;

  // Dragged card info
  draggedCardIndex: number | null;
  draggedCardCode: string | null;

  // Position tracking
  targetPosition: [number, number, number]; // Where cursor is pointing
  currentPosition: [number, number, number]; // Actual card position (lags with spring)
  originalHandPosition: [number, number, number]; // For return animation

  // Hover state
  hoveredZone: HoveredZone;
  hoveredSlotIndex: number | null;

  // Valid drop targets (computed from action mask)
  validGardenSlots: Set<number>;
  validAlleySlots: Set<number>;

  // Actions
  startPickup: (
    handIndex: number,
    cardCode: string,
    handPosition: [number, number, number],
    validGardenSlots: Set<number>,
    validAlleySlots: Set<number>
  ) => void;
  startDragging: () => void;
  updateTargetPosition: (position: [number, number, number]) => void;
  updateCurrentPosition: (position: [number, number, number]) => void;
  setHoveredSlot: (zone: HoveredZone, slotIndex: number | null) => void;
  drop: () => void;
  startReturning: () => void;
  cancelDrag: () => void;
  reset: () => void;
}

const initialState = {
  dragPhase: "idle" as DragPhase,
  draggedCardIndex: null,
  draggedCardCode: null,
  targetPosition: [0, 0, 0] as [number, number, number],
  currentPosition: [0, 0, 0] as [number, number, number],
  originalHandPosition: [0, 0, 0] as [number, number, number],
  hoveredZone: null,
  hoveredSlotIndex: null,
  validGardenSlots: new Set<number>(),
  validAlleySlots: new Set<number>(),
};

export const useDragStore = create<DragState>((set) => ({
  ...initialState,

  startPickup: (handIndex, cardCode, handPosition, validGardenSlots, validAlleySlots) =>
    set({
      dragPhase: "pickup",
      draggedCardIndex: handIndex,
      draggedCardCode: cardCode,
      originalHandPosition: handPosition,
      currentPosition: [...handPosition],
      targetPosition: [...handPosition],
      validGardenSlots,
      validAlleySlots,
      hoveredZone: "hand",
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

  drop: () =>
    set({
      dragPhase: "idle",
      draggedCardIndex: null,
      draggedCardCode: null,
      validGardenSlots: new Set(),
      validAlleySlots: new Set(),
      hoveredZone: null,
      hoveredSlotIndex: null,
    }),

  startReturning: () =>
    set({
      dragPhase: "returning",
    }),

  cancelDrag: () =>
    set({
      ...initialState,
      validGardenSlots: new Set(),
      validAlleySlots: new Set(),
    }),

  reset: () =>
    set({
      ...initialState,
      validGardenSlots: new Set(),
      validAlleySlots: new Set(),
    }),
}));
