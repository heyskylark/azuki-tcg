import { pgTable, text, integer, pgEnum } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@/drizzle/helpers";
import { WinType } from "@/types";
import { rooms } from "@/drizzle/schemas/rooms";
import { users } from "@/drizzle/schemas/users";

export const winTypeEnum = pgEnum("win_type", enumToPgEnum(WinType));

export const matchResults = pgTable("match_results", {
  id: uuidv7PrimaryKeyField(),
  roomId: text("room_id")
    .notNull()
    .references(() => rooms.id),
  player0Id: text("player0_id")
    .notNull()
    .references(() => users.id),
  player1Id: text("player1_id")
    .notNull()
    .references(() => users.id),
  winnerId: text("winner_id").references(() => users.id),
  winType: winTypeEnum("win_type").notNull(),
  totalTurns: integer("total_turns").notNull(),
  durationSeconds: integer("duration_seconds").notNull(),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});
