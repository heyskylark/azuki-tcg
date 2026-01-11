import { pgTable, integer, uuid } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
} from "@/drizzle/helpers";
import { Decks } from "@/drizzle/schemas/decks";
import { Cards } from "@/drizzle/schemas/cards";

export const DeckCardJunctions = pgTable("deck_card_junctions", {
  id: uuidv7PrimaryKeyField(),
  deckId: uuid("deck_id")
    .notNull()
    .references(() => Decks.id, { onDelete: "cascade" }),
  cardId: uuid("card_id")
    .notNull()
    .references(() => Cards.id),
  quantity: integer("quantity").notNull().default(1),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});
