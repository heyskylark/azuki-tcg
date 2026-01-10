import { sql } from "drizzle-orm";
import type { PgTimestampConfig } from "drizzle-orm/pg-core";
import { integer, timestamp, text, uuid } from "drizzle-orm/pg-core";
import { uuidv7 } from "uuidv7";

export const uuidv7PrimaryKeyField = (fieldName = "id") =>
  uuid(fieldName)
    .primaryKey()
    .$defaultFn(() => uuidv7());

export const timestampField = (fieldName: string, config?: PgTimestampConfig) =>
  timestamp(fieldName, {
    mode: "date",
    withTimezone: true,
    precision: 3,
    ...config,
  });

export const createdAtTimestampField = (fieldName = "created_at", config?: PgTimestampConfig) =>
  timestampField(fieldName, config).notNull().defaultNow();

export const updatedAtTimestampField = (fieldName = "updated_at", config?: PgTimestampConfig) =>
  timestampField(fieldName, config)
    .notNull()
    .defaultNow()
    .$onUpdate(() => new Date());

export const incrementOnUpdateVersionField = () =>
  integer("version")
    .notNull()
    .default(1)
    .$onUpdateFn(() => sql`version + 1`);

export function enumToPgEnum<T extends Record<string, string>>(
  myEnum: T
): [T[keyof T], ...T[keyof T][]] {
  return Object.values(myEnum) as [T[keyof T], ...T[keyof T][]];
}
