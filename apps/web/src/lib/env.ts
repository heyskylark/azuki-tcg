import { z } from "zod";

const envSchema = z.object({
  NODE_ENV: z
    .enum(["development", "production", "test"])
    .default("development"),
  JWT_SECRET: z
    .string()
    .min(32, "JWT_SECRET must be at least 32 characters"),
  JWT_ISSUER: z.string().min(1, "JWT_ISSUER is required"),
  PASSWORD_SALT_ROUNDS: z.coerce.number().min(10).max(15).default(12),
  DATABASE_URL: z.string().url("DATABASE_URL must be a valid URL"),
});

export const env = envSchema.parse(process.env);

export const isProduction = env.NODE_ENV === "production";
export const isDevelopment = env.NODE_ENV === "development";
