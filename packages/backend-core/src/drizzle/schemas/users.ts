import { pgTable, text, pgEnum } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@/drizzle/helpers";
import { UserStatus, UserType } from "@/types";

export const userStatusEnum = pgEnum("user_status", enumToPgEnum(UserStatus));
export const userTypeEnum = pgEnum("user_type", enumToPgEnum(UserType));

export const users = pgTable("users", {
  id: uuidv7PrimaryKeyField(),
  username: text("username").notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  type: userTypeEnum("type").notNull().default(UserType.HUMAN),
  status: userStatusEnum("status").notNull().default(UserStatus.ACTIVE),
  modelKey: text("model_key"),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});
