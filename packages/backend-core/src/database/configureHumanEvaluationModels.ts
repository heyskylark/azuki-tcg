import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { eq, inArray } from "drizzle-orm";
import { uuidv7 } from "uuidv7";
import { z } from "zod";
import db, { pool } from "@core/database";
import { AiModels } from "@core/drizzle/schemas/ai_models";
import { Cards } from "@core/drizzle/schemas/cards";
import { AiModelStatus } from "@core/types";
import { CardType } from "@core/types/cards";
import type { HumanEvaluationPlan } from "@core/types/humanEvaluations";
const repositoryRoot = fileURLToPath(new URL("../../../../", import.meta.url));

const cardCodeSchema = z.string().trim().min(1).max(32);

const evaluationPlanSchema = z
  .object({
    premadeFraction: z.number().min(0).max(1),
    premadeDeckPool: z.string().trim().min(1),
    premadeDeckSlugs: z.array(z.string().trim().min(1)),
    draftGates: z.array(
      z
        .object({
          gateCardCode: cardCodeSchema,
          leaderCardCodes: z.array(cardCodeSchema).min(1),
        })
        .strict()
    ),
  })
  .strict()
  .superRefine((plan, context) => {
    if (plan.premadeFraction > 0 && plan.premadeDeckSlugs.length === 0) {
      context.addIssue({
        code: "custom",
        path: ["premadeDeckSlugs"],
        message: "premadeFraction > 0 requires at least one premade deck",
      });
    }
    if (plan.premadeFraction < 1 && plan.draftGates.length === 0) {
      context.addIssue({
        code: "custom",
        path: ["draftGates"],
        message: "premadeFraction < 1 requires at least one draft gate",
      });
    }
    if (new Set(plan.premadeDeckSlugs).size !== plan.premadeDeckSlugs.length) {
      context.addIssue({
        code: "custom",
        path: ["premadeDeckSlugs"],
        message: "premade deck slugs must be unique",
      });
    }
  });

const modelSchema = z
  .object({
    displayName: z.string().trim().min(1).max(200),
    modelKey: z.string().trim().min(1).max(2000),
    checkpointSha256: z.string().regex(/^[a-f0-9]{64}$/),
    evaluationPlan: evaluationPlanSchema.optional(),
  })
  .strict();

const deckPoolSchema = z.object({
  decks: z.array(
    z.object({
      deck_slug: z.string().min(1),
      deck_name: z.string().min(1),
      gate_card_id: cardCodeSchema,
      leader_card_id: cardCodeSchema,
    })
  ),
});

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

async function resolveEvaluationPlan(
  plan: z.infer<typeof evaluationPlanSchema>
): Promise<HumanEvaluationPlan> {
  const deckPool = deckPoolSchema.parse(
    JSON.parse(await readFile(resolve(repositoryRoot, plan.premadeDeckPool), "utf8"))
  );
  const premadeDecks = plan.premadeDeckSlugs.map((slug) => {
    const deck = deckPool.decks.find((candidate) => candidate.deck_slug === slug);
    if (!deck) throw new Error(`Premade deck ${slug} is not in ${plan.premadeDeckPool}`);
    return {
      slug,
      name: deck.deck_name,
      gateCardCode: deck.gate_card_id,
      leaderCardCode: deck.leader_card_id,
    };
  });
  const contexts = [
    ...premadeDecks,
    ...plan.draftGates.flatMap(({ gateCardCode, leaderCardCodes }) =>
      leaderCardCodes.map((leaderCardCode) => ({ gateCardCode, leaderCardCode }))
    ),
  ];
  const codes = [
    ...new Set(
      contexts.flatMap(({ gateCardCode, leaderCardCode }) => [gateCardCode, leaderCardCode])
    ),
  ];
  const cards = await db
    .select({ cardCode: Cards.cardCode, cardType: Cards.cardType, element: Cards.element })
    .from(Cards)
    .where(inArray(Cards.cardCode, codes));
  for (const { gateCardCode, leaderCardCode } of contexts) {
    const gate = cards.find((card) => card.cardCode === gateCardCode);
    const leader = cards.find((card) => card.cardCode === leaderCardCode);
    if (
      gate?.cardType !== CardType.GATE ||
      leader?.cardType !== CardType.LEADER ||
      gate.element !== leader.element
    ) {
      throw new Error(`Invalid evaluation gate/leader pair ${gateCardCode}/${leaderCardCode}`);
    }
  }
  return { premadeFraction: plan.premadeFraction, premadeDecks, draftGates: plan.draftGates };
}

async function configureHumanEvaluationModels(path: string): Promise<void> {
  const payload: unknown = JSON.parse(await readFile(path, "utf8"));
  const { models } = rosterSchema.parse(payload);
  const plans = await Promise.all(
    models.map((model) =>
      model.evaluationPlan ? resolveEvaluationPlan(model.evaluationPlan) : null
    )
  );

  await db.transaction(async (transaction) => {
    await transaction
      .update(AiModels)
      .set({ humanEvaluationEnabled: false })
      .where(eq(AiModels.humanEvaluationEnabled, true));

    for (const [index, model] of models.entries()) {
      const humanEvaluationPlan = plans[index] ?? null;
      await transaction
        .insert(AiModels)
        .values({
          id: uuidv7(),
          displayName: model.displayName,
          modelKey: model.modelKey,
          checkpointSha256: model.checkpointSha256,
          status: AiModelStatus.ENABLED,
          humanEvaluationEnabled: true,
          humanEvaluationPlan,
        })
        .onConflictDoUpdate({
          target: AiModels.modelKey,
          set: {
            displayName: model.displayName,
            checkpointSha256: model.checkpointSha256,
            status: AiModelStatus.ENABLED,
            humanEvaluationEnabled: true,
            humanEvaluationPlan,
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
