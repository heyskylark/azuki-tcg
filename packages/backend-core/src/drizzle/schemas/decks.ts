import { pgTable, text, boolean, pgEnum, check } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@/drizzle/helpers";
import { DeckStatus } from "@/types";
import { users } from "@/drizzle/schemas/users";
import { cards } from "@/drizzle/schemas/cards";

export const deckStatusEnum = pgEnum("deck_status", enumToPgEnum(DeckStatus));

export const decks = pgTable(
  "decks",
  {
    id: uuidv7PrimaryKeyField(),
    name: text("name").notNull(),
    userId: text("user_id")
      .notNull()
      .references(() => users.id),
    status: deckStatusEnum("status").notNull().default(DeckStatus.IN_PROGRESS),
    isSystemDeck: boolean("is_system_deck").notNull().default(false),
    createdAt: createdAtTimestampField(),
    updatedAt: updatedAtTimestampField(),
  },
  (table) => [
    check(
      "system_deck_not_deleted",
      sql`NOT (${table.isSystemDeck} = true AND ${table.status} = 'DELETED')`
    ),
  ]
);
