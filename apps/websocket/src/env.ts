import { z } from "zod";

const envSchema = z.object({
  NODE_ENV: z
    .enum(["development", "production", "test"])
    .default("development"),
  PORT: z.coerce.number().default(3001),
  LOG_LEVEL: z
    .enum(["error", "warn", "info", "http", "verbose", "debug", "silly"])
    .default("info"),
  DATABASE_URL: z.string().optional(),
  JWT_SECRET: z.string(),
  JWT_ISSUER: z.string().default("azuki-tcg"),
  ENGINE_DEBUG: z
    .enum(["true", "false", "1", "0"])
    .optional()
    .transform((val) => val === "true" || val === "1"),
  INFERENCE_URL: z.string().url().default("http://localhost:8002"),
  INFERENCE_TIMEOUT_MS: z.coerce.number().int().positive().default(5000),
});

export const env = envSchema.parse(process.env);
