import { z } from "zod";

const deckNameSchema = z.string().trim().min(1, "Deck name is required");

const deckCardCountsSchema = z.record(
  z.string().uuid("Invalid card selection"),
  z.number().int().positive("Card quantities must be positive integers")
);

export const createOrUpdateDeckSchema = z
  .object({
    name: deckNameSchema,
    cardCounts: deckCardCountsSchema,
  })
  .strict();

export const copyDeckSchema = z
  .object({
    name: deckNameSchema,
  })
  .strict();

export type CreateOrUpdateDeckInput = z.infer<typeof createOrUpdateDeckSchema>;
export type CopyDeckInput = z.infer<typeof copyDeckSchema>;
