import { pgTable, uuid, pgEnum, index } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  timestampField,
  enumToPgEnum,
} from "@core/drizzle/helpers";
import { TokenType } from "@core/types/auth";
import { Users } from "@core/drizzle/schemas/users";

export const tokenTypeEnum = pgEnum("token_type", enumToPgEnum(TokenType));

export const JwtTokens = pgTable(
  "jwt_tokens",
  {
    id: uuidv7PrimaryKeyField(),
    jti: uuid("jti").notNull().unique(),
    userId: uuid("user_id")
      .notNull()
      .references(() => Users.id, { onDelete: "cascade" }),
    tokenType: tokenTypeEnum("token_type").notNull(),
    expiresAt: timestampField("expires_at").notNull(),
    revokedAt: timestampField("revoked_at"),
    createdAt: createdAtTimestampField(),
  },
  (table) => [
    index("jwt_tokens_user_id_idx").on(table.userId),
    index("jwt_tokens_expires_at_idx").on(table.expiresAt),
  ]
);
