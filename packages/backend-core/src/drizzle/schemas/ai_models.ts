import { pgTable, text, pgEnum, index, boolean, jsonb } from "drizzle-orm/pg-core";
import {
  uuidv7PrimaryKeyField,
  createdAtTimestampField,
  updatedAtTimestampField,
  enumToPgEnum,
} from "@core/drizzle/helpers";
import { AiModelStatus } from "@core/types";
import type { HumanEvaluationPlan } from "@core/types/humanEvaluations";

export const aiModelStatusEnum = pgEnum("ai_model_status", enumToPgEnum(AiModelStatus));

export const AiModels = pgTable(
  "ai_models",
  {
    id: uuidv7PrimaryKeyField(),
    displayName: text("display_name").notNull(),
    modelKey: text("model_key").notNull().unique(),
    status: aiModelStatusEnum("status").notNull().default(AiModelStatus.ENABLED),
    checkpointSha256: text("checkpoint_sha256").notNull().default(""),
    humanEvaluationEnabled: boolean("human_evaluation_enabled").notNull().default(false),
    humanEvaluationPlan: jsonb("human_evaluation_plan").$type<HumanEvaluationPlan>(),
    createdAt: createdAtTimestampField(),
    updatedAt: updatedAtTimestampField(),
  },
  (table) => [
    index("ai_models_status_idx").on(table.status),
    index("ai_models_human_evaluation_idx").on(table.status, table.humanEvaluationEnabled),
    index("ai_models_display_name_idx").on(table.displayName),
  ]
);
