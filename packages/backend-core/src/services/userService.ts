import { eq } from "drizzle-orm";
import db, { type IDatabase, type ITransaction } from "@/database";
import { Users, Emails } from "@/drizzle/schemas";
import {
  EmailAlreadyExistsError,
  UsernameAlreadyExistsError,
} from "@/errors";
import { UserStatus, UserType } from "@/types";
import type {
  CreateUserParams,
  UserWithPassword,
  UserWithEmail,
} from "@/types/auth";

type Database = IDatabase | ITransaction;

export async function emailExists(
  email: string,
  database: Database = db
): Promise<boolean> {
  const result = await database
    .select({ id: Emails.id })
    .from(Emails)
    .where(eq(Emails.email, email))
    .limit(1);
  return result.length > 0;
}

export async function usernameExists(
  username: string,
  database: Database = db
): Promise<boolean> {
  const result = await database
    .select({ id: Users.id })
    .from(Users)
    .where(eq(Users.username, username))
    .limit(1);
  return result.length > 0;
}

export async function createUser(
  params: CreateUserParams,
  database: Database = db
): Promise<{ id: string; username: string; email: string }> {
  if (await emailExists(params.email, database)) {
    throw new EmailAlreadyExistsError();
  }

  if (await usernameExists(params.username, database)) {
    throw new UsernameAlreadyExistsError();
  }

  return await db.transaction(async (tx) => {
    const result = await tx
      .insert(Users)
      .values({
        username: params.username,
        passwordHash: params.passwordHash,
        type: UserType.HUMAN,
        status: UserStatus.ACTIVE,
      })
      .returning({
        id: Users.id,
        username: Users.username,
      });

    const user = result[0]!;

    await tx.insert(Emails).values({
      email: params.email,
      userId: user.id,
    });

    return {
      id: user.id,
      username: user.username,
      email: params.email,
    };
  });
}

export async function findUserByEmail(
  email: string,
  database: Database = db
): Promise<UserWithPassword | null> {
  const result = await database
    .select({
      id: Users.id,
      username: Users.username,
      email: Emails.email,
      passwordHash: Users.passwordHash,
      status: Users.status,
      type: Users.type,
    })
    .from(Emails)
    .innerJoin(Users, eq(Emails.userId, Users.id))
    .where(eq(Emails.email, email))
    .limit(1);

  return result[0] ?? null;
}

export async function findUserById(
  userId: string,
  database: Database = db
): Promise<{ id: string; username: string; status: UserStatus; type: UserType } | null> {
  const result = await database
    .select({
      id: Users.id,
      username: Users.username,
      status: Users.status,
      type: Users.type,
    })
    .from(Users)
    .where(eq(Users.id, userId))
    .limit(1);

  return result[0] ?? null;
}

export async function getUserWithEmail(
  userId: string,
  database: Database = db
): Promise<UserWithEmail | null> {
  const result = await database
    .select({
      id: Users.id,
      username: Users.username,
      email: Emails.email,
      status: Users.status,
      type: Users.type,
    })
    .from(Users)
    .innerJoin(Emails, eq(Users.id, Emails.userId))
    .where(eq(Users.id, userId))
    .limit(1);

  return result[0] ?? null;
}
