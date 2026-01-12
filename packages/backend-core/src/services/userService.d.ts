import { type IDatabase, type ITransaction } from "@/database";
import { UserStatus, UserType } from "@/types";
import type { CreateUserParams, UserWithPassword, UserWithEmail } from "@/types/auth";
type Database = IDatabase | ITransaction;
export declare function emailExists(email: string, database?: Database): Promise<boolean>;
export declare function usernameExists(username: string, database?: Database): Promise<boolean>;
export declare function createUser(params: CreateUserParams, database?: Database): Promise<{
    id: string;
    username: string;
    displayName: string;
    email: string;
}>;
export declare function findUserByEmail(email: string, database?: Database): Promise<UserWithPassword | null>;
export declare function findUserById(userId: string, database?: Database): Promise<{
    id: string;
    username: string;
    displayName: string;
    status: UserStatus;
    type: UserType;
} | null>;
export declare function getUserWithEmail(userId: string, database?: Database): Promise<UserWithEmail | null>;
export declare function updateUserDisplayName(userId: string, displayName: string, database?: Database): Promise<{
    id: string;
    displayName: string;
}>;
export {};
//# sourceMappingURL=userService.d.ts.map