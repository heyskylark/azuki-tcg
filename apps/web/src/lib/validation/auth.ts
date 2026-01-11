import { z } from "zod";
import {
  PASSWORD_MIN_LENGTH,
  PASSWORD_MAX_LENGTH,
  USERNAME_MIN_LENGTH,
  USERNAME_MAX_LENGTH,
  USERNAME_REGEX,
} from "@tcg/backend-core/constants/auth";

const emailSchema = z
  .string()
  .email("Invalid email format")
  .transform((e) => e.toLowerCase().trim());

const passwordSchema = z
  .string()
  .min(
    PASSWORD_MIN_LENGTH,
    `Password must be at least ${PASSWORD_MIN_LENGTH} characters`
  )
  .max(
    PASSWORD_MAX_LENGTH,
    `Password must be at most ${PASSWORD_MAX_LENGTH} characters`
  );

export const signUpSchema = z
  .object({
    username: z
      .string()
      .min(
        USERNAME_MIN_LENGTH,
        `Username must be at least ${USERNAME_MIN_LENGTH} characters`
      )
      .max(
        USERNAME_MAX_LENGTH,
        `Username must be at most ${USERNAME_MAX_LENGTH} characters`
      )
      .regex(
        USERNAME_REGEX,
        "Username can only contain letters, numbers, underscores, and hyphens"
      ),
    email: emailSchema,
    password: passwordSchema,
  })
  .strict();

export const loginSchema = z
  .object({
    email: emailSchema,
    password: passwordSchema,
  })
  .strict();

export type SignUpInput = z.infer<typeof signUpSchema>;
export type LoginInput = z.infer<typeof loginSchema>;
