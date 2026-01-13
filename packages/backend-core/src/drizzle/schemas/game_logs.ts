import { pgTable, text, integer, jsonb, uuid } from "drizzle-orm/pg-core";
import { uuidv7PrimaryKeyField, createdAtTimestampField } from "@core/drizzle/helpers";
import { Rooms } from "@core/drizzle/schemas/rooms";

export const GameLogs = pgTable("game_logs", {
  id: uuidv7PrimaryKeyField(),
  roomId: uuid("room_id")
    .notNull()
    .references(() => Rooms.id, { onDelete: "cascade" }),
  batchNumber: integer("batch_number").notNull(),
  sequenceNumber: integer("sequence_number").notNull(),
  logType: text("log_type").notNull(),
  player: integer("player"),
  logData: jsonb("log_data").notNull(),
  createdAt: createdAtTimestampField(),
});
