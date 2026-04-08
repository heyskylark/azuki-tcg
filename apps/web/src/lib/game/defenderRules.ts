import type { ResolvedCard, ResolvedPlayerBoard } from "@/types/game";

const YOJIN_CARD_CODE = "AZK01-052";

function countGardenEntities(board: ResolvedPlayerBoard): number {
  return board.garden.reduce((count, card) => count + (card == null ? 0 : 1), 0);
}

function getConditionalDefenderState(
  card: ResolvedCard,
  ownerBoard: ResolvedPlayerBoard,
  opposingBoard: ResolvedPlayerBoard
): boolean | null {
  switch (card.cardCode) {
    case YOJIN_CARD_CODE:
      return countGardenEntities(ownerBoard) < countGardenEntities(opposingBoard);
    default:
      return null;
  }
}

export function countsAsGardenDefender(
  card: ResolvedCard,
  ownerBoard: ResolvedPlayerBoard,
  opposingBoard: ResolvedPlayerBoard
): boolean {
  const conditionalDefenderState = getConditionalDefenderState(
    card,
    ownerBoard,
    opposingBoard
  );

  return conditionalDefenderState ?? card.hasDefender;
}

export function countGardenDefenders(
  ownerBoard: ResolvedPlayerBoard,
  opposingBoard: ResolvedPlayerBoard
): number {
  return ownerBoard.garden.reduce((count, card) => {
    if (card == null) {
      return count;
    }

    return count + (countsAsGardenDefender(card, ownerBoard, opposingBoard) ? 1 : 0);
  }, 0);
}
