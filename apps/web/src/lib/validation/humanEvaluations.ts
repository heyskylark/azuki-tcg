import { z } from "zod";

const ratingSchema = z.union([
  z.literal(1),
  z.literal(2),
  z.literal(3),
  z.literal(4),
  z.literal(5),
  z.literal(6),
  z.literal(7),
]);

export const createHumanEvaluationSessionSchema = z
  .object({
    humanDeckId: z.string().uuid(),
    gamesPerModel: z.union([z.literal(8), z.literal(16)]),
  })
  .strict();

export const emptyHumanEvaluationBodySchema = z.object({}).strict();

export const humanEvaluationAnnotationSchema = z
  .object({
    ratings: z
      .object({
        opponentStrength: ratingSchema,
        decisionQuality: ratingSchema,
        deckCoherence: ratingSchema,
        humanLikeness: ratingSchema,
        matchEnjoyment: ratingSchema,
      })
      .strict(),
    modelGuess: z.string().trim().min(1).max(200).nullable(),
    guessConfidence: ratingSchema.nullable(),
    observations: z
      .array(
        z
          .object({
            kind: z.enum(["ISSUE", "OPPORTUNITY"]),
            tag: z.enum([
              "MISPLAY",
              "SEQUENCING",
              "RESOURCE_WASTE",
              "TARGETING",
              "ATTACK_CHOICE",
              "DEFENSE_CHOICE",
              "TEMPO",
              "DECK_SYNERGY",
              "REPETITIVE",
              "TIMEOUT_OR_STALL",
              "OTHER",
            ]),
            severity: z.enum(["MINOR", "MODERATE", "MAJOR", "CRITICAL"]),
            actionNumber: z.number().int().positive().nullable(),
            detail: z.string().trim().min(1).max(2000).nullable(),
          })
          .strict()
      )
      .max(100),
    notes: z.string().trim().min(1).max(10000).nullable(),
  })
  .strict();

export type CreateHumanEvaluationSessionInput = z.infer<typeof createHumanEvaluationSessionSchema>;
export type HumanEvaluationAnnotationRequest = z.infer<typeof humanEvaluationAnnotationSchema>;
