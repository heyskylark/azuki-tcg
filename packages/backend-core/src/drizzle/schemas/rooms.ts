import {
  pgTable,
  text,
  boolean,
  integer,
  timestamp,
  pgEnum,
} from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@/drizzle/helpers";
import { RoomStatus, RoomType } from "@/types";
import { users } from "@/drizzle/schemas/users";
import { decks } from "@/drizzle/schemas/decks";

export const roomStatusEnum = pgEnum("room_status", enumToPgEnum(RoomStatus));
export const roomTypeEnum = pgEnum("room_type", enumToPgEnum(RoomType));

export const rooms = pgTable("rooms", {
  id: uuidv7PrimaryKeyField(),
  status: roomStatusEnum("status")
    .notNull()
    .default(RoomStatus.WAITING_FOR_PLAYERS),
  type: roomTypeEnum("type").notNull().default(RoomType.PRIVATE),
  passwordHash: text("password_hash"),
  worldId: text("world_id"),
  rngSeed: integer("rng_seed"),
  player0Id: text("player0_id").references(() => users.id),
  player0DeckId: text("player0_deck_id").references(() => decks.id),
  player0Ready: boolean("player0_ready").notNull().default(false),
  player1Id: text("player1_id").references(() => users.id),
  player1DeckId: text("player1_deck_id").references(() => decks.id),
  player1Ready: boolean("player1_ready").notNull().default(false),
  deckSelectionDeadline: timestamp("deck_selection_deadline", {
    withTimezone: true,
  }),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});
