import * as dotenv from "dotenv";

dotenv.config({ path: [".env.local", ".env"] });

import { sql } from "drizzle-orm";
import { drizzle } from "drizzle-orm/node-postgres";
import { migrate } from "drizzle-orm/node-postgres/migrator";
import { Client } from "pg";

const isLocalDb =
  process.env["DATABASE_URL"]?.includes("localhost") ||
  process.env["DATABASE_URL"]?.includes("@db:");

const ssl = isLocalDb ? false : { rejectUnauthorized: false };

const client = new Client({
  connectionString: process.env["DATABASE_URL"],
  ssl,
});

const db = drizzle(client);

const runMigrations = async () => {
  console.log("Running migrations on", process.env["DATABASE_URL"]);

  await client.connect();
  await client.query("SET statement_timeout = '600s'");

  // Enable useful PostgreSQL extensions
  await db.execute(sql`CREATE EXTENSION IF NOT EXISTS citext;`);
  await db.execute(sql`CREATE EXTENSION IF NOT EXISTS pg_trgm;`);

  await migrate(db, { migrationsFolder: "drizzle" });

  console.log("Migrations complete!");
  await client.end();
};

runMigrations().catch((err) => {
  console.error("Migration failed:", err);
  process.exit(1);
});
