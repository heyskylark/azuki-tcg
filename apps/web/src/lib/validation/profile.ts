import { z } from "zod";

export const DISPLAY_NAME_MIN_LENGTH = 3;
export const DISPLAY_NAME_MAX_LENGTH = 30;

export const updateProfileSchema = z
  .object({
    displayName: z
      .string()
      .min(
        DISPLAY_NAME_MIN_LENGTH,
        `Display name must be at least ${DISPLAY_NAME_MIN_LENGTH} characters`
      )
      .max(
        DISPLAY_NAME_MAX_LENGTH,
        `Display name must be at most ${DISPLAY_NAME_MAX_LENGTH} characters`
      )
      .regex(
        /^[a-zA-Z0-9 ]+$/,
        "Display name can only contain letters, numbers, and spaces"
      )
      .transform((s) => s.trim()),
  })
  .strict();

export type UpdateProfileInput = z.infer<typeof updateProfileSchema>;
