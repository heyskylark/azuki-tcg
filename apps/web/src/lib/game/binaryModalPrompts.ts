/**
 * Prompts for engine "binary modal" abilities: cards whose printed text offers
 * two modes, resolved during the CONFIRMATION ability phase where
 * ACT_CONFIRM_ABILITY picks the first printed mode and ACT_NOOP picks the second.
 *
 * Mode text is phrased from the perspective of the player making the choice.
 */
export interface BinaryModalPrompt {
  /** OPPONENT: the card's controller's opponent picks (e.g. Fatedealer "Your opponent chooses 1"). */
  chooser: "CONTROLLER" | "OPPONENT";
  /** First printed mode, sent as ACT_CONFIRM_ABILITY. */
  confirmMode: string;
  /** Second printed mode, sent as ACT_NOOP. */
  declineMode: string;
}

const BINARY_MODAL_PROMPTS: Record<string, BinaryModalPrompt> = {
  "AZK01-013": {
    chooser: "OPPONENT",
    confirmMode: "Sacrifice an entity in your Garden.",
    declineMode: "Discard 2.",
  },
  "AZK01-076": {
    chooser: "OPPONENT",
    confirmMode:
      "Their Hōren of Two Paths gains Charge (can attack the same turn it enters the Garden) until the end of the turn.",
    declineMode: "Your opponent heals 2 to their leader.",
  },
  "AZK01-079": {
    chooser: "OPPONENT",
    confirmMode: "Your opponent draws 2.",
    declineMode: "Your opponent deals 3 damage to your leader.",
  },
  "AZK01-099": {
    chooser: "CONTROLLER",
    confirmMode:
      'An entity with a cost of 5 or less in your opponent\'s Garden becomes "Shocked" (does not untap during its next untap phase).',
    declineMode:
      "This entity gains Charge (can attack the same turn it enters the Garden) until the end of the turn.",
  },
};

export function getBinaryModalPrompt(cardCode: string | undefined): BinaryModalPrompt | null {
  if (cardCode === undefined) return null;
  return BINARY_MODAL_PROMPTS[cardCode] ?? null;
}
