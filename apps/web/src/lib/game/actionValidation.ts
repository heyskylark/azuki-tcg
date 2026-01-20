import type { SnapshotActionMask } from "@tcg/backend-core/types/ws";

// Action type constants from action space head 0
export const ACTION_NOOP = 0;
export const ACTION_PLAY_ENTITY_TO_GARDEN = 1;
export const ACTION_PLAY_ENTITY_TO_ALLEY = 2;
export const ACTION_PLAY_SPELL = 3;
export const ACTION_PLAY_WEAPON = 4;
// ... additional play action types

// Ability action types
export const ACTION_SELECT_COST_TARGET = 13;
export const ACTION_SELECT_EFFECT_TARGET = 14;
export const ACTION_CONFIRM_ABILITY = 16;
export const ACTION_SELECT_FROM_SELECTION = 18;
export const ACTION_BOTTOM_DECK_CARD = 19;
export const ACTION_BOTTOM_DECK_ALL = 20;
export const ACTION_SELECT_TO_ALLEY = 21;
export const ACTION_SELECT_TO_EQUIP = 22;

/**
 * Get valid garden and alley slots for playing a specific hand card.
 * Returns sets of slot indices that are legal drop targets.
 */
export function getValidSlotsForHandCard(
  actionMask: SnapshotActionMask | null,
  handIndex: number
): { gardenSlots: Set<number>; alleySlots: Set<number> } {
  const gardenSlots = new Set<number>();
  const alleySlots = new Set<number>();

  if (!actionMask) {
    return { gardenSlots, alleySlots };
  }

  const { legalPrimary, legalSub1, legalSub2 } = actionMask;

  // Iterate through all legal actions to find those matching this hand index
  for (let i = 0; i < legalPrimary.length; i++) {
    const actionType = legalPrimary[i];
    const sub1 = legalSub1[i]; // Hand index
    const sub2 = legalSub2[i]; // Target slot

    // Check if this action is for the specified hand index
    if (sub1 !== handIndex) {
      continue;
    }

    // Check action type and add to appropriate slot set
    if (actionType === ACTION_PLAY_ENTITY_TO_GARDEN) {
      gardenSlots.add(sub2);
    } else if (actionType === ACTION_PLAY_ENTITY_TO_ALLEY) {
      alleySlots.add(sub2);
    }
  }

  return { gardenSlots, alleySlots };
}

/**
 * Check if a hand card has any valid play actions (garden or alley).
 */
export function canPlayCard(
  actionMask: SnapshotActionMask | null,
  handIndex: number
): boolean {
  const { gardenSlots, alleySlots } = getValidSlotsForHandCard(actionMask, handIndex);
  return gardenSlots.size > 0 || alleySlots.size > 0;
}

/**
 * Find the valid action tuple for a specific drop target.
 * Returns [actionType, handIndex, targetSlot, ikzTokenFlag] or null if invalid.
 */
export function findValidAction(
  actionMask: SnapshotActionMask | null,
  handIndex: number,
  targetZone: "garden" | "alley",
  targetSlot: number
): [number, number, number, number] | null {
  if (!actionMask) {
    return null;
  }

  const targetActionType =
    targetZone === "garden" ? ACTION_PLAY_ENTITY_TO_GARDEN : ACTION_PLAY_ENTITY_TO_ALLEY;

  const { legalPrimary, legalSub1, legalSub2, legalSub3 } = actionMask;

  // Search for matching action in the action mask
  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === targetActionType &&
      legalSub1[i] === handIndex &&
      legalSub2[i] === targetSlot
    ) {
      // Return the action tuple with ikzTokenFlag from legalSub3
      return [targetActionType, handIndex, targetSlot, legalSub3[i]];
    }
  }

  return null;
}

/**
 * Check if a specific slot is a valid drop target for a hand card.
 */
export function isValidDropTarget(
  actionMask: SnapshotActionMask | null,
  handIndex: number,
  targetZone: "garden" | "alley",
  targetSlot: number
): boolean {
  return findValidAction(actionMask, handIndex, targetZone, targetSlot) !== null;
}

// ============================================
// Ability Action Helpers
// ============================================

/**
 * Check if the CONFIRM_ABILITY action is available in the action mask.
 */
export function hasConfirmAbilityAction(
  actionMask: SnapshotActionMask | null
): boolean {
  if (!actionMask) return false;
  return actionMask.legalPrimary.some((action) => action === ACTION_CONFIRM_ABILITY);
}

/**
 * Check if the NOOP action is available (for declining/skipping abilities).
 */
export function hasNoopAction(actionMask: SnapshotActionMask | null): boolean {
  if (!actionMask) return false;
  return actionMask.legalPrimary.some((action) => action === ACTION_NOOP);
}

/**
 * Get valid cost target indices from the action mask.
 * Returns an array of hand indices that can be selected as cost targets.
 */
export function getValidCostTargets(
  actionMask: SnapshotActionMask | null
): number[] {
  if (!actionMask) return [];

  const targets: number[] = [];
  for (let i = 0; i < actionMask.legalPrimary.length; i++) {
    if (actionMask.legalPrimary[i] === ACTION_SELECT_COST_TARGET) {
      targets.push(actionMask.legalSub1[i]);
    }
  }
  return targets;
}

/**
 * Get valid effect target indices from the action mask.
 * Returns an array of target indices (0-4 = my garden, 5-9 = opponent garden, etc.).
 */
export function getValidEffectTargets(
  actionMask: SnapshotActionMask | null
): number[] {
  if (!actionMask) return [];

  const targets: number[] = [];
  for (let i = 0; i < actionMask.legalPrimary.length; i++) {
    if (actionMask.legalPrimary[i] === ACTION_SELECT_EFFECT_TARGET) {
      targets.push(actionMask.legalSub1[i]);
    }
  }
  return targets;
}

/**
 * Get valid selection indices from the action mask for SELECTION_PICK phase.
 * Returns an array of selection zone indices that can be picked.
 */
export function getValidSelectionTargets(
  actionMask: SnapshotActionMask | null
): number[] {
  if (!actionMask) return [];

  const targets: number[] = [];
  for (let i = 0; i < actionMask.legalPrimary.length; i++) {
    const actionType = actionMask.legalPrimary[i];
    if (
      actionType === ACTION_SELECT_FROM_SELECTION ||
      actionType === ACTION_SELECT_TO_ALLEY ||
      actionType === ACTION_SELECT_TO_EQUIP
    ) {
      targets.push(actionMask.legalSub1[i]);
    }
  }
  return [...new Set(targets)]; // Remove duplicates
}

/**
 * Get valid bottom deck card indices from the action mask.
 */
export function getValidBottomDeckTargets(
  actionMask: SnapshotActionMask | null
): number[] {
  if (!actionMask) return [];

  const targets: number[] = [];
  for (let i = 0; i < actionMask.legalPrimary.length; i++) {
    if (actionMask.legalPrimary[i] === ACTION_BOTTOM_DECK_CARD) {
      targets.push(actionMask.legalSub1[i]);
    }
  }
  return targets;
}

/**
 * Check if "bottom deck all" action is available.
 */
export function hasBottomDeckAllAction(
  actionMask: SnapshotActionMask | null
): boolean {
  if (!actionMask) return false;
  return actionMask.legalPrimary.some((action) => action === ACTION_BOTTOM_DECK_ALL);
}

// ============================================
// Action Builders
// ============================================

/**
 * Build a CONFIRM_ABILITY action tuple.
 */
export function buildConfirmAbilityAction(): [number, number, number, number] {
  return [ACTION_CONFIRM_ABILITY, 0, 0, 0];
}

/**
 * Build a NOOP action tuple (for declining/skipping).
 */
export function buildNoopAction(): [number, number, number, number] {
  return [ACTION_NOOP, 0, 0, 0];
}

/**
 * Build a SELECT_COST_TARGET action tuple.
 */
export function buildCostTargetAction(
  handIndex: number
): [number, number, number, number] {
  return [ACTION_SELECT_COST_TARGET, handIndex, 0, 0];
}

/**
 * Build a SELECT_EFFECT_TARGET action tuple.
 */
export function buildEffectTargetAction(
  targetIndex: number
): [number, number, number, number] {
  return [ACTION_SELECT_EFFECT_TARGET, targetIndex, 0, 0];
}

/**
 * Build a SELECT_FROM_SELECTION action tuple.
 */
export function buildSelectionPickAction(
  selectionIndex: number
): [number, number, number, number] {
  return [ACTION_SELECT_FROM_SELECTION, selectionIndex, 0, 0];
}

/**
 * Build a SELECT_TO_ALLEY action tuple.
 */
export function buildSelectToAlleyAction(
  selectionIndex: number
): [number, number, number, number] {
  return [ACTION_SELECT_TO_ALLEY, selectionIndex, 0, 0];
}

/**
 * Build a SELECT_TO_EQUIP action tuple.
 */
export function buildSelectToEquipAction(
  selectionIndex: number
): [number, number, number, number] {
  return [ACTION_SELECT_TO_EQUIP, selectionIndex, 0, 0];
}

/**
 * Build a BOTTOM_DECK_CARD action tuple.
 */
export function buildBottomDeckCardAction(
  selectionIndex: number
): [number, number, number, number] {
  return [ACTION_BOTTOM_DECK_CARD, selectionIndex, 0, 0];
}

/**
 * Build a BOTTOM_DECK_ALL action tuple.
 */
export function buildBottomDeckAllAction(): [number, number, number, number] {
  return [ACTION_BOTTOM_DECK_ALL, 0, 0, 0];
}
