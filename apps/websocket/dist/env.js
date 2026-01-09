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
});
export const env = envSchema.parse(process.env);
//# sourceMappingURL=env.js.map