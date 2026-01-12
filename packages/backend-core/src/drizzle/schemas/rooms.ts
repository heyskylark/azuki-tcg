import { pgTable, text, boolean, integer, timestamp, pgEnum, uuid, index } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@/drizzle/helpers";
import { RoomStatus, RoomType } from "@/types";
import { Users } from "@/drizzle/schemas/users";
import { Decks } from "@/drizzle/schemas/decks";

export const roomStatusEnum = pgEnum("room_status", enumToPgEnum(RoomStatus));
export const roomTypeEnum = pgEnum("room_type", enumToPgEnum(RoomType));

export const Rooms = pgTable(
  "rooms",
  {
    id: uuidv7PrimaryKeyField(),
    status: roomStatusEnum("status").notNull().default(RoomStatus.WAITING_FOR_PLAYERS),
    type: roomTypeEnum("type").notNull().default(RoomType.PRIVATE),
    passwordHash: text("password_hash"),
    worldId: text("world_id"),
    rngSeed: integer("rng_seed"),
    player0Id: uuid("player0_id").references(() => Users.id, { onDelete: "cascade" }),
    player0DeckId: uuid("player0_deck_id").references(() => Decks.id, { onDelete: "cascade" }),
    player0Ready: boolean("player0_ready").notNull().default(false),
    player1Id: uuid("player1_id").references(() => Users.id, { onDelete: "cascade" }),
    player1DeckId: uuid("player1_deck_id").references(() => Decks.id, { onDelete: "cascade" }),
    player1Ready: boolean("player1_ready").notNull().default(false),
    deckSelectionDeadline: timestamp("deck_selection_deadline", {
      withTimezone: true,
    }),
    createdAt: createdAtTimestampField(),
    updatedAt: updatedAtTimestampField(),
  },
  (table) => [
    index("rooms_player0_status_idx").on(table.player0Id, table.status),
    index("rooms_player1_status_idx").on(table.player1Id, table.status),
  ]
);
