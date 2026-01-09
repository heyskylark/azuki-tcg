import { pgTable, text, integer, jsonb } from "drizzle-orm/pg-core";
import { uuidv7PrimaryKeyField, createdAtTimestampField } from "@/drizzle/helpers";
import { rooms } from "@/drizzle/schemas/rooms";

export const gameLogs = pgTable("game_logs", {
  id: uuidv7PrimaryKeyField(),
  roomId: text("room_id")
    .notNull()
    .references(() => rooms.id),
  batchNumber: integer("batch_number").notNull(),
  sequenceNumber: integer("sequence_number").notNull(),
  logType: text("log_type").notNull(),
  player: integer("player"),
  logData: jsonb("log_data").notNull(),
  createdAt: createdAtTimestampField(),
});
