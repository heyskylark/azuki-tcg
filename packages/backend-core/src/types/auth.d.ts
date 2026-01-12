import { UserStatus, UserType } from "@/types";
export declare enum TokenType {
    ACCESS = "ACCESS",
    REFRESH = "REFRESH",
    JOIN = "JOIN"
}
export interface JWTPayload {
    iss: string;
    sub: string;
    aud: string;
    exp: number;
    iat: number;
    jti: string;
}
export interface IdentityTokenPayload {
    userId: string;
    username: string;
    displayName: string;
    email: string;
    status: UserStatus;
}
export interface TokenPair {
    accessToken: string;
    refreshToken: string;
    identityToken: string;
}
export interface AuthenticatedUser {
    id: string;
    username: string;
    displayName: string;
    email: string;
    status: UserStatus;
    type: UserType;
}
export interface AuthConfig {
    jwtSecret: string;
    jwtIssuer: string;
    saltRounds: number;
}
export interface CreateUserParams {
    username: string;
    displayName: string;
    email: string;
    passwordHash: string;
}
export interface TokenUser {
    id: string;
    username: string;
    displayName: string;
    email: string;
    status: UserStatus;
}
export interface UserWithPassword {
    id: string;
    username: string;
    displayName: string;
    email: string;
    passwordHash: string;
    status: UserStatus;
    type: UserType;
}
export interface UserWithEmail {
    id: string;
    username: string;
    displayName: string;
    email: string;
    status: UserStatus;
    type: UserType;
}
export interface JoinTokenPayload {
    roomId: string;
    playerSlot: 0 | 1;
}
//# sourceMappingURL=auth.d.ts.map