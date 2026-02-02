import type { SnapshotActionMask } from "@tcg/backend-core/types/ws";

// Action type constants from action space head 0
export const ACTION_NOOP = 0;
export const ACTION_PLAY_ENTITY_TO_GARDEN = 1;
export const ACTION_PLAY_ENTITY_TO_ALLEY = 2;
export const ACTION_PLAY_SPELL_FROM_HAND = 8;
// Legacy alias for older action space naming
export const ACTION_PLAY_SPELL = ACTION_PLAY_SPELL_FROM_HAND;
export const ACTION_PLAY_WEAPON = 4;
export const ACTION_ATTACK = 6;
// ... additional play action types
export const ACTION_ATTACH_WEAPON_FROM_HAND = 7;
export const ACTION_DECLARE_DEFENDER = 9;
export const ACTION_GATE_PORTAL = 10;
export const ACTION_ACTIVATE_GARDEN_OR_LEADER_ABILITY = 11;
export const ACTION_ACTIVATE_ALLEY_ABILITY = 12;

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

// ============================================
// Spell Action Helpers
// ============================================

/**
 * Check if a hand card has a valid spell play action.
 */
export function canPlaySpell(
  actionMask: SnapshotActionMask | null,
  handIndex: number
): boolean {
  if (!actionMask) return false;
  const { legalPrimary, legalSub1 } = actionMask;
  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_PLAY_SPELL_FROM_HAND &&
      legalSub1[i] === handIndex
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Find the valid spell play action tuple for a specific hand card.
 * Returns [ACTION_PLAY_SPELL_FROM_HAND, handIndex, 0, ikzTokenFlag] or null if invalid.
 * Prefers non-token actions (ikzTokenFlag = 0) when available.
 */
export function findValidSpellAction(
  actionMask: SnapshotActionMask | null,
  handIndex: number
): [number, number, number, number] | null {
  if (!actionMask) return null;

  const { legalPrimary, legalSub1, legalSub3 } = actionMask;
  let fallback: [number, number, number, number] | null = null;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_PLAY_SPELL_FROM_HAND &&
      legalSub1[i] === handIndex
    ) {
      const action: [number, number, number, number] = [
        ACTION_PLAY_SPELL_FROM_HAND,
        handIndex,
        0,
        legalSub3[i],
      ];
      if (legalSub3[i] === 0) {
        return action;
      }
      fallback = action;
    }
  }

  return fallback;
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
// Weapon Attachment Action Helpers
// ============================================

/**
 * Get valid entity slots (0-4 = garden slots, 5 = leader) for attaching a weapon from hand.
 * Returns a Set of entity slot indices.
 */
export function getValidWeaponAttachTargets(
  actionMask: SnapshotActionMask | null,
  handIndex: number
): Set<number> {
  const targets = new Set<number>();

  if (!actionMask) return targets;

  const { legalPrimary, legalSub1, legalSub2 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_ATTACH_WEAPON_FROM_HAND &&
      legalSub1[i] === handIndex
    ) {
      targets.add(legalSub2[i]);
    }
  }

  return targets;
}

/**
 * Check if a hand card has any valid weapon attachment actions.
 */
export function canAttachWeapon(
  actionMask: SnapshotActionMask | null,
  handIndex: number
): boolean {
  return getValidWeaponAttachTargets(actionMask, handIndex).size > 0;
}

/**
 * Check if any weapon attachment actions are available in the action mask.
 */
export function hasWeaponAttachActions(
  actionMask: SnapshotActionMask | null
): boolean {
  if (!actionMask) return false;
  return actionMask.legalPrimary.some(
    (action) => action === ACTION_ATTACH_WEAPON_FROM_HAND
  );
}

/**
 * Find the valid weapon attachment action tuple for a specific drop.
 * Returns [ACTION_ATTACH_WEAPON_FROM_HAND, handIndex, entitySlot, ikzTokenFlag] or null.
 * entitySlot: 0-4 = garden slots, 5 = leader
 */
export function findValidWeaponAttachAction(
  actionMask: SnapshotActionMask | null,
  handIndex: number,
  entitySlot: number
): [number, number, number, number] | null {
  if (!actionMask) return null;

  const { legalPrimary, legalSub1, legalSub2, legalSub3 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_ATTACH_WEAPON_FROM_HAND &&
      legalSub1[i] === handIndex &&
      legalSub2[i] === entitySlot
    ) {
      return [ACTION_ATTACH_WEAPON_FROM_HAND, handIndex, entitySlot, legalSub3[i]];
    }
  }

  return null;
}

/**
 * Build a weapon attachment action tuple.
 * @param handIndex - The hand index of the weapon
 * @param entitySlot - The target entity slot (0-4 = garden slots, 5 = leader)
 * @param ikzTokenFlag - 0 for normal IKZ cost, 1 to use extra IKZ token
 */
export function buildWeaponAttachAction(
  handIndex: number,
  entitySlot: number,
  ikzTokenFlag: number = 0
): [number, number, number, number] {
  return [ACTION_ATTACH_WEAPON_FROM_HAND, handIndex, entitySlot, ikzTokenFlag];
}

// ============================================
// Attack Action Helpers
// ============================================

/**
 * Get valid attacker indices (0-4 = garden slots, 5 = leader).
 */
export function getValidAttackers(
  actionMask: SnapshotActionMask | null
): Set<number> {
  const attackers = new Set<number>();

  if (!actionMask) return attackers;

  const { legalPrimary, legalSub1 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (legalPrimary[i] === ACTION_ATTACK) {
      attackers.add(legalSub1[i]);
    }
  }

  return attackers;
}

/**
 * Get valid target indices (0-4 = enemy garden, 5 = enemy leader) for a specific attacker.
 */
export function getValidAttackTargetsForAttacker(
  actionMask: SnapshotActionMask | null,
  attackerIndex: number
): Set<number> {
  const targets = new Set<number>();

  if (!actionMask) return targets;

  const { legalPrimary, legalSub1, legalSub2 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_ATTACK &&
      legalSub1[i] === attackerIndex
    ) {
      targets.add(legalSub2[i]);
    }
  }

  return targets;
}

/**
 * Find the valid attack action tuple for a specific attacker and target.
 * Returns [ACTION_ATTACK, attackerIndex, targetIndex, 0] or null if invalid.
 */
export function findValidAttackAction(
  actionMask: SnapshotActionMask | null,
  attackerIndex: number,
  targetIndex: number
): [number, number, number, number] | null {
  if (!actionMask) return null;

  const { legalPrimary, legalSub1, legalSub2, legalSub3 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_ATTACK &&
      legalSub1[i] === attackerIndex &&
      legalSub2[i] === targetIndex
    ) {
      return [ACTION_ATTACK, attackerIndex, targetIndex, legalSub3[i]];
    }
  }

  return null;
}

/**
 * Build an ATTACK action tuple.
 */
export function buildAttackAction(
  attackerIndex: number,
  targetIndex: number
): [number, number, number, number] {
  return [ACTION_ATTACK, attackerIndex, targetIndex, 0];
}

// ============================================
// Defender Action Helpers
// ============================================

/**
 * Get valid defender indices (0-4 = garden slots).
 */
export function getValidDefenderSlots(
  actionMask: SnapshotActionMask | null
): Set<number> {
  const defenders = new Set<number>();

  if (!actionMask) return defenders;

  const { legalPrimary, legalSub1 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (legalPrimary[i] === ACTION_DECLARE_DEFENDER) {
      defenders.add(legalSub1[i]);
    }
  }

  return defenders;
}

/**
 * Find the valid DECLARE_DEFENDER action tuple for a specific garden slot.
 * Returns [ACTION_DECLARE_DEFENDER, gardenIndex, 0, 0] or null if invalid.
 */
export function findValidDeclareDefenderAction(
  actionMask: SnapshotActionMask | null,
  gardenIndex: number
): [number, number, number, number] | null {
  if (!actionMask) return null;

  const { legalPrimary, legalSub1, legalSub2, legalSub3 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_DECLARE_DEFENDER &&
      legalSub1[i] === gardenIndex
    ) {
      return [ACTION_DECLARE_DEFENDER, gardenIndex, legalSub2[i], legalSub3[i]];
    }
  }

  return null;
}

/**
 * Build a DECLARE_DEFENDER action tuple.
 */
export function buildDeclareDefenderAction(
  gardenIndex: number
): [number, number, number, number] {
  return [ACTION_DECLARE_DEFENDER, gardenIndex, 0, 0];
}

// ============================================
// Ability Action Helpers
// ============================================

/**
 * Get valid garden or leader slots for activating abilities.
 * Returns a Set of slot indices (0-4 = garden, 5 = leader).
 */
export function getActivatableGardenOrLeaderSlots(
  actionMask: SnapshotActionMask | null
): Set<number> {
  const slots = new Set<number>();

  if (!actionMask) return slots;

  const { legalPrimary, legalSub1 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (legalPrimary[i] === ACTION_ACTIVATE_GARDEN_OR_LEADER_ABILITY) {
      slots.add(legalSub1[i]);
    }
  }

  return slots;
}

/**
 * Get valid alley slots for activating abilities.
 * Returns a Set of alley slot indices.
 */
export function getActivatableAlleySlots(
  actionMask: SnapshotActionMask | null
): Set<number> {
  const slots = new Set<number>();

  if (!actionMask) return slots;

  const { legalPrimary, legalSub2 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (legalPrimary[i] === ACTION_ACTIVATE_ALLEY_ABILITY) {
      slots.add(legalSub2[i]);
    }
  }

  return slots;
}

/**
 * Find the valid garden/leader ability activation action tuple for a slot.
 * Returns [ACTION_ACTIVATE_GARDEN_OR_LEADER_ABILITY, slotIndex, sub2, sub3] or null if invalid.
 */
export function findValidGardenOrLeaderAbilityAction(
  actionMask: SnapshotActionMask | null,
  slotIndex: number
): [number, number, number, number] | null {
  if (!actionMask) return null;

  const { legalPrimary, legalSub1, legalSub2, legalSub3 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_ACTIVATE_GARDEN_OR_LEADER_ABILITY &&
      legalSub1[i] === slotIndex
    ) {
      return [
        ACTION_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
        legalSub1[i],
        legalSub2[i],
        legalSub3[i],
      ];
    }
  }

  return null;
}

/**
 * Find the valid alley ability activation action tuple for a slot.
 * Returns [ACTION_ACTIVATE_ALLEY_ABILITY, abilityIndex, alleyIndex, sub3] or null if invalid.
 */
export function findValidAlleyAbilityAction(
  actionMask: SnapshotActionMask | null,
  alleyIndex: number
): [number, number, number, number] | null {
  if (!actionMask) return null;

  const { legalPrimary, legalSub1, legalSub2, legalSub3 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_ACTIVATE_ALLEY_ABILITY &&
      legalSub2[i] === alleyIndex
    ) {
      return [
        ACTION_ACTIVATE_ALLEY_ABILITY,
        legalSub1[i],
        legalSub2[i],
        legalSub3[i],
      ];
    }
  }

  return null;
}

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
 * Detailed info about available selection actions for a specific selection index.
 */
export interface SelectionActionInfo {
  selectionIndex: number;
  canAddToHand: boolean;
  canSelectToAlley: boolean;
  alleySlots: number[]; // Valid alley slots if canSelectToAlley
  canSelectToEquip: boolean;
  equipTargetSlots: number[]; // Valid entity slots (0-4=garden, 5=leader) if canSelectToEquip
}

/**
 * Get detailed action info for all valid selection targets.
 * Returns structured info about what actions are available for each selection index.
 */
export function getSelectionActionInfo(
  actionMask: SnapshotActionMask | null
): SelectionActionInfo[] {
  if (!actionMask) return [];

  // Group actions by selection index
  const infoMap = new Map<number, SelectionActionInfo>();

  for (let i = 0; i < actionMask.legalPrimary.length; i++) {
    const actionType = actionMask.legalPrimary[i];
    const selectionIndex = actionMask.legalSub1[i];
    const sub2 = actionMask.legalSub2[i];

    // Skip non-selection actions
    if (
      actionType !== ACTION_SELECT_FROM_SELECTION &&
      actionType !== ACTION_SELECT_TO_ALLEY &&
      actionType !== ACTION_SELECT_TO_EQUIP
    ) {
      continue;
    }

    // Get or create info for this selection index
    let info = infoMap.get(selectionIndex);
    if (!info) {
      info = {
        selectionIndex,
        canAddToHand: false,
        canSelectToAlley: false,
        alleySlots: [],
        canSelectToEquip: false,
        equipTargetSlots: [],
      };
      infoMap.set(selectionIndex, info);
    }

    // Populate based on action type
    if (actionType === ACTION_SELECT_FROM_SELECTION) {
      info.canAddToHand = true;
    } else if (actionType === ACTION_SELECT_TO_ALLEY) {
      info.canSelectToAlley = true;
      if (!info.alleySlots.includes(sub2)) {
        info.alleySlots.push(sub2);
      }
    } else if (actionType === ACTION_SELECT_TO_EQUIP) {
      info.canSelectToEquip = true;
      if (!info.equipTargetSlots.includes(sub2)) {
        info.equipTargetSlots.push(sub2);
      }
    }
  }

  return Array.from(infoMap.values());
}

/**
 * Get action info for a specific selection index.
 */
export function getSelectionActionInfoByIndex(
  actionMask: SnapshotActionMask | null,
  selectionIndex: number
): SelectionActionInfo | null {
  const allInfo = getSelectionActionInfo(actionMask);
  return allInfo.find((info) => info.selectionIndex === selectionIndex) ?? null;
}

/**
 * Check if a specific equip action is valid (weapon at selectionIndex to entity at entitySlot).
 */
export function isValidEquipAction(
  actionMask: SnapshotActionMask | null,
  selectionIndex: number,
  entitySlot: number
): boolean {
  if (!actionMask) return false;

  for (let i = 0; i < actionMask.legalPrimary.length; i++) {
    if (
      actionMask.legalPrimary[i] === ACTION_SELECT_TO_EQUIP &&
      actionMask.legalSub1[i] === selectionIndex &&
      actionMask.legalSub2[i] === entitySlot
    ) {
      return true;
    }
  }
  return false;
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
 * @param selectionIndex - The index in the selection zone
 * @param alleySlot - The target alley slot (0-4)
 */
export function buildSelectToAlleyAction(
  selectionIndex: number,
  alleySlot: number
): [number, number, number, number] {
  return [ACTION_SELECT_TO_ALLEY, selectionIndex, alleySlot, 0];
}

/**
 * Build a SELECT_TO_EQUIP action tuple.
 * @param selectionIndex - The index in the selection zone
 * @param entitySlot - The target entity slot (0-4 = garden slots, 5 = leader)
 */
export function buildSelectToEquipAction(
  selectionIndex: number,
  entitySlot: number
): [number, number, number, number] {
  return [ACTION_SELECT_TO_EQUIP, selectionIndex, entitySlot, 0];
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

// ============================================
// Gate Portal Action Helpers
// ============================================

/**
 * Check if any gate portal actions are available in the action mask.
 */
export function hasGatePortalActions(
  actionMask: SnapshotActionMask | null
): boolean {
  if (!actionMask) return false;
  return actionMask.legalPrimary.some(
    (action) => action === ACTION_GATE_PORTAL
  );
}

/**
 * Get valid alley indices that can be gated (source cards that can move to garden).
 * Returns a Set of alley slot indices.
 */
export function getValidGateSourceAlleySlots(
  actionMask: SnapshotActionMask | null
): Set<number> {
  const alleySlots = new Set<number>();

  if (!actionMask) return alleySlots;

  const { legalPrimary, legalSub1 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (legalPrimary[i] === ACTION_GATE_PORTAL) {
      // legalSub1 is the alley index (source)
      alleySlots.add(legalSub1[i]);
    }
  }

  return alleySlots;
}

/**
 * Get valid garden indices for a specific alley card to gate into.
 * Returns a Set of garden slot indices.
 */
export function getValidGateTargetGardenSlots(
  actionMask: SnapshotActionMask | null,
  alleyIndex: number
): Set<number> {
  const gardenSlots = new Set<number>();

  if (!actionMask) return gardenSlots;

  const { legalPrimary, legalSub1, legalSub2 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_GATE_PORTAL &&
      legalSub1[i] === alleyIndex
    ) {
      // legalSub2 is the garden index (target)
      gardenSlots.add(legalSub2[i]);
    }
  }

  return gardenSlots;
}

/**
 * Find the valid gate action tuple for a specific alley to garden move.
 * Returns [ACTION_GATE_PORTAL, alleyIndex, gardenIndex, 0] or null if invalid.
 */
export function findValidGateAction(
  actionMask: SnapshotActionMask | null,
  alleyIndex: number,
  gardenIndex: number
): [number, number, number, number] | null {
  if (!actionMask) return null;

  const { legalPrimary, legalSub1, legalSub2, legalSub3 } = actionMask;

  for (let i = 0; i < legalPrimary.length; i++) {
    if (
      legalPrimary[i] === ACTION_GATE_PORTAL &&
      legalSub1[i] === alleyIndex &&
      legalSub2[i] === gardenIndex
    ) {
      return [ACTION_GATE_PORTAL, alleyIndex, gardenIndex, legalSub3[i]];
    }
  }

  return null;
}

/**
 * Build a GATE_PORTAL action tuple.
 */
export function buildGatePortalAction(
  alleyIndex: number,
  gardenIndex: number
): [number, number, number, number] {
  return [ACTION_GATE_PORTAL, alleyIndex, gardenIndex, 0];
}
