import { pgTable, text, pgEnum, uuid } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@/drizzle/helpers";
import { UserStatus, UserType } from "@/types";
import { citext } from "@/drizzle/customTypes";

export const userStatusEnum = pgEnum("user_status", enumToPgEnum(UserStatus));
export const userTypeEnum = pgEnum("user_type", enumToPgEnum(UserType));

export const Users = pgTable("users", {
  id: uuidv7PrimaryKeyField(),
  username: citext("username").notNull().unique(),
  displayName: text("display_name").notNull(),
  passwordHash: text("password_hash").notNull(),
  type: userTypeEnum("type").notNull().default(UserType.HUMAN),
  status: userStatusEnum("status").notNull().default(UserStatus.ACTIVE),
  modelKey: text("model_key"),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});

export const Emails = pgTable("emails", {
  id: uuidv7PrimaryKeyField(),
  email: citext("email").notNull().unique(),
  userId: uuid("user_id")
    .notNull()
    .references(() => Users.id, {
      onDelete: "cascade",
    }),
  createdAt: createdAtTimestampField(),
  updatedAt: updatedAtTimestampField(),
});
