import { type IDatabase, type ITransaction } from "@/database";
import { type JWTPayload, type TokenPair, type AuthConfig, type TokenUser, type JoinTokenPayload } from "@/types/auth";
type Database = IDatabase | ITransaction;
export declare function hashPassword(password: string, saltRounds?: number): Promise<string>;
export declare function verifyPassword(password: string, hash: string): Promise<boolean>;
export declare function createTokens(user: TokenUser, config: AuthConfig, database?: Database): Promise<TokenPair>;
export declare function verifyAccessToken(token: string, config: AuthConfig, database?: Database): Promise<JWTPayload>;
export declare function verifyRefreshToken(token: string, config: AuthConfig, database?: Database): Promise<JWTPayload>;
export declare function verifyJoinToken(token: string, config: AuthConfig, database?: Database): Promise<JWTPayload & JoinTokenPayload>;
export declare function revokeToken(jti: string, database?: Database): Promise<void>;
export declare function revokeAllUserTokens(userId: string, database?: Database): Promise<void>;
export declare function refreshTokens(refreshToken: string, user: TokenUser, config: AuthConfig, database?: Database): Promise<TokenPair>;
export {};
//# sourceMappingURL=authService.d.ts.map