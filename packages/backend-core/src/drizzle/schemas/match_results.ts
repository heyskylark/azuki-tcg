import { pgTable, text, integer, pgEnum, uuid } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@core/drizzle/helpers";
import { WinType } from "@core/types";
import { Rooms } from "@core/drizzle/schemas/rooms";
import { Users } from "@core/drizzle/schemas/users";

export const winTypeEnum = pgEnum("win_type", enumToPgEnum(WinType));

export const MatchResults = pgTable("match_results", {
  id: uuidv7PrimaryKeyField(),
  roomId: uuid("room_id")
    .notNull()
    .references(() => Rooms.id),
  player0Id: uuid("player0_id")
    .notNull()
    .references(() => Users.id, { onDelete: "cascade" }),
  player1Id: uuid("player1_id")
    .notNull()
    .references(() => Users.id, { onDelete: "cascade" }),
  winnerId: uuid("winner_id").references(() => Users.id),
  winType: winTypeEnum("win_type").notNull(),
  totalTurns: integer("total_turns").notNull(),
  durationSeconds: integer("duration_seconds").notNull(),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});
