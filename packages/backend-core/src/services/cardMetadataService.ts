import { inArray } from "drizzle-orm";
import db, { type IDatabase, type ITransaction } from "@core/database";
import { Cards } from "@core/drizzle/schemas/cards";
import type { SnapshotCardMetadata } from "@core/types/ws";

type Database = IDatabase | ITransaction;

export async function getCardMetadataByCardCodes(
  cardCodes: string[],
  database: Database = db
): Promise<Map<string, SnapshotCardMetadata>> {
  const uniqueCardCodes = Array.from(new Set(cardCodes));
  if (uniqueCardCodes.length === 0) {
    return new Map();
  }

  const rows = await database
    .select({
      cardCode: Cards.cardCode,
      name: Cards.name,
      imageKey: Cards.imageKey,
      cardType: Cards.cardType,
      attack: Cards.attack,
      health: Cards.health,
      ikzCost: Cards.ikzCost,
    })
    .from(Cards)
    .where(inArray(Cards.cardCode, uniqueCardCodes));

  const metadataByCode = new Map<string, SnapshotCardMetadata>();
  for (const row of rows) {
    metadataByCode.set(row.cardCode, {
      cardCode: row.cardCode,
      name: row.name,
      imageKey: row.imageKey,
      cardType: row.cardType,
      attack: row.attack,
      health: row.health,
      ikzCost: row.ikzCost,
    });
  }

  return metadataByCode;
}
