import { pgTable, text, integer } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
} from "@/drizzle/helpers";
import { decks } from "@/drizzle/schemas/decks";
import { cards } from "@/drizzle/schemas/cards";

export const deckCardJunctions = pgTable("deck_card_junctions", {
  id: uuidv7PrimaryKeyField(),
  deckId: text("deck_id")
    .notNull()
    .references(() => decks.id, { onDelete: "cascade" }),
  cardId: text("card_id")
    .notNull()
    .references(() => cards.id),
  quantity: integer("quantity").notNull().default(1),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});
