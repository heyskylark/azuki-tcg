import { pgTable, text, pgEnum, index } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@core/drizzle/helpers";
import { AiModelStatus } from "@core/types";

export const aiModelStatusEnum = pgEnum("ai_model_status", enumToPgEnum(AiModelStatus));

export const AiModels = pgTable(
  "ai_models",
  {
    id: uuidv7PrimaryKeyField(),
    displayName: text("display_name").notNull(),
    modelKey: text("model_key").notNull().unique(),
    status: aiModelStatusEnum("status").notNull().default(AiModelStatus.ENABLED),
    createdAt: createdAtTimestampField(),
    updatedAt: updatedAtTimestampField(),
  },
  (table) => [
    index("ai_models_status_idx").on(table.status),
    index("ai_models_display_name_idx").on(table.displayName),
  ]
);
