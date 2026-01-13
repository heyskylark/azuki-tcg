import * as dotenv from "dotenv";

dotenv.config({ path: [".env.local", ".env"] });

import * as schemas from "@core/drizzle/schemas";
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool, PoolConfig } from "pg";

const isLocalDb =
  process.env["DATABASE_URL"]?.includes("localhost") ||
  process.env["DATABASE_URL"]?.includes("@db:");

const ssl = isLocalDb ? false : { rejectUnauthorized: false };

const poolConfig: Partial<PoolConfig> = {
  connectionString: process.env["DATABASE_URL"],
  ssl,
  min: 1,
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
};

export const pool = new Pool(poolConfig);

pool.on("error", (error) => {
  console.error("Unexpected error in database pool", error);
});

const db = drizzle(pool, { schema: { ...schemas } });

export type IDatabase = typeof db;
export type ITransaction = Parameters<
  Parameters<IDatabase["transaction"]>[0]
>[0];

export default db;
