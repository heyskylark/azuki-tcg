import { pgTable, text, boolean, pgEnum, check, uuid, index } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@core/drizzle/helpers";
import { DeckStatus } from "@core/types";
import { Users } from "@core/drizzle/schemas/users";

export const deckStatusEnum = pgEnum("deck_status", enumToPgEnum(DeckStatus));

export const Decks = pgTable(
  "decks",
  {
    id: uuidv7PrimaryKeyField(),
    name: text("name").notNull(),
    userId: uuid("user_id")
      .notNull()
      .references(() => Users.id, { onDelete: "cascade" }),
    status: deckStatusEnum("status").notNull().default(DeckStatus.IN_PROGRESS),
    isSystemDeck: boolean("is_system_deck").notNull().default(false),
    createdAt: createdAtTimestampField(),
    updatedAt: updatedAtTimestampField(),
  },
  (table) => [
    index("idx_decks_user_id").on(table.userId),
    check(
      "system_deck_not_deleted",
      sql`NOT (${table.isSystemDeck} = true AND ${table.status} = 'DELETED')`
    ),
  ]
);
