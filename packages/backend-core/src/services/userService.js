import { eq } from "drizzle-orm";
import db from "@/database";
import { Users, Emails } from "@/drizzle/schemas";
import { EmailAlreadyExistsError, UsernameAlreadyExistsError, UpdateFailedError, } from "@/errors";
import { UserStatus, UserType } from "@/types";
import { addStarterDecks } from "@/services/DeckService";
export async function emailExists(email, database = db) {
    const result = await database
        .select({ id: Emails.id })
        .from(Emails)
        .where(eq(Emails.email, email))
        .limit(1);
    return result.length > 0;
}
export async function usernameExists(username, database = db) {
    const result = await database
        .select({ id: Users.id })
        .from(Users)
        .where(eq(Users.username, username))
        .limit(1);
    return result.length > 0;
}
export async function createUser(params, database = db) {
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
            displayName: params.displayName,
            passwordHash: params.passwordHash,
            type: UserType.HUMAN,
            status: UserStatus.ACTIVE,
        })
            .returning({
            id: Users.id,
            username: Users.username,
            displayName: Users.displayName,
        });
        const user = result[0];
        if (!user) {
            throw new Error("Failed to create user");
        }
        await tx.insert(Emails).values({
            email: params.email,
            userId: user.id,
        });
        // Add starter decks for the new user
        await addStarterDecks(user.id, tx);
        return {
            id: user.id,
            username: user.username,
            displayName: user.displayName,
            email: params.email,
        };
    });
}
export async function findUserByEmail(email, database = db) {
    const result = await database
        .select({
        id: Users.id,
        username: Users.username,
        displayName: Users.displayName,
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
export async function findUserById(userId, database = db) {
    const result = await database
        .select({
        id: Users.id,
        username: Users.username,
        displayName: Users.displayName,
        status: Users.status,
        type: Users.type,
    })
        .from(Users)
        .where(eq(Users.id, userId))
        .limit(1);
    return result[0] ?? null;
}
export async function getUserWithEmail(userId, database = db) {
    const result = await database
        .select({
        id: Users.id,
        username: Users.username,
        displayName: Users.displayName,
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
export async function updateUserDisplayName(userId, displayName, database = db) {
    const result = await database
        .update(Users)
        .set({ displayName })
        .where(eq(Users.id, userId))
        .returning({
        id: Users.id,
        displayName: Users.displayName,
    });
    const updated = result[0];
    if (!updated) {
        throw new UpdateFailedError("Failed to update display name");
    }
    return updated;
}
//# sourceMappingURL=userService.js.map