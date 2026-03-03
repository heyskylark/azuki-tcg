import { asc, eq } from "drizzle-orm";
import db, { type IDatabase, type ITransaction } from "@core/database";
import { AiModels } from "@core/drizzle/schemas/ai_models";
import { AiModelNotFoundError, AiModelUnavailableError } from "@core/errors";
import { AiModelStatus } from "@core/types";

type Database = IDatabase | ITransaction;

export interface SelectableAiModel {
  id: string;
  displayName: string;
}

export async function listSelectableAiModels(
  database: Database = db
): Promise<SelectableAiModel[]> {
  return await database
    .select({
      id: AiModels.id,
      displayName: AiModels.displayName,
    })
    .from(AiModels)
    .where(eq(AiModels.status, AiModelStatus.ENABLED))
    .orderBy(asc(AiModels.displayName));
}

export async function getSelectableAiModelById(
  aiModelId: string,
  database: Database = db
): Promise<{ id: string; displayName: string; modelKey: string }> {
  const selectedModel = await database
    .select({
      id: AiModels.id,
      displayName: AiModels.displayName,
      modelKey: AiModels.modelKey,
      status: AiModels.status,
    })
    .from(AiModels)
    .where(eq(AiModels.id, aiModelId))
    .limit(1)
    .then((results) => results[0]);

  if (!selectedModel) {
    throw new AiModelNotFoundError();
  }

  if (selectedModel.status !== AiModelStatus.ENABLED) {
    throw new AiModelUnavailableError(
      `Selected AI model "${selectedModel.displayName}" is not available`
    );
  }

  return {
    id: selectedModel.id,
    displayName: selectedModel.displayName,
    modelKey: selectedModel.modelKey,
  };
}
