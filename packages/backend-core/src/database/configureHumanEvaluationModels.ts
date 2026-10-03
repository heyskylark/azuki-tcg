import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { eq } from "drizzle-orm";
import { uuidv7 } from "uuidv7";
import { z } from "zod";
import db, { pool } from "@core/database";
import { AiModels } from "@core/drizzle/schemas/ai_models";
import { AiModelStatus } from "@core/types";
const repositoryRoot = fileURLToPath(new URL("../../../../", import.meta.url));

const modelSchema = z
  .object({
    displayName: z.string().trim().min(1).max(200),
    modelKey: z.string().trim().min(1).max(2000),
    checkpointSha256: z.string().regex(/^[a-f0-9]{64}$/),
  })
  .strict();

const rosterSchema = z
  .object({
    models: z.array(modelSchema).min(1),
  })
  .strict()
  .superRefine(({ models }, context) => {
    const keys = new Set<string>();
    for (const [index, model] of models.entries()) {
      if (keys.has(model.modelKey)) {
        context.addIssue({
          code: "custom",
          path: ["models", index, "modelKey"],
          message: "modelKey values must be unique",
        });
      }
      keys.add(model.modelKey);
    }
  });

async function configureHumanEvaluationModels(path: string): Promise<void> {
  const payload: unknown = JSON.parse(await readFile(path, "utf8"));
  const { models } = rosterSchema.parse(payload);

  await db.transaction(async (transaction) => {
    await transaction
      .update(AiModels)
      .set({ humanEvaluationEnabled: false })
      .where(eq(AiModels.humanEvaluationEnabled, true));

    for (const model of models) {
      await transaction
        .insert(AiModels)
        .values({
          id: uuidv7(),
          displayName: model.displayName,
          modelKey: model.modelKey,
          checkpointSha256: model.checkpointSha256,
          status: AiModelStatus.ENABLED,
          humanEvaluationEnabled: true,
        })
        .onConflictDoUpdate({
          target: AiModels.modelKey,
          set: {
            displayName: model.displayName,
            checkpointSha256: model.checkpointSha256,
            status: AiModelStatus.ENABLED,
            humanEvaluationEnabled: true,
            updatedAt: new Date(),
          },
        });
    }
  });

  console.log(`Configured ${models.length} human-evaluation model(s).`);
}

const rosterPath = process.argv[2];
if (!rosterPath) {
  console.error("Usage: bun core human-eval:models -- <roster.json>");
  process.exitCode = 1;
} else {
  try {
    await configureHumanEvaluationModels(resolve(repositoryRoot, rosterPath));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  } finally {
    await pool.end();
  }
}
