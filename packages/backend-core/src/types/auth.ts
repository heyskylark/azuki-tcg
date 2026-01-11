import { UserStatus, UserType } from "@/types";

export enum TokenType {
  ACCESS = "ACCESS",
  REFRESH = "REFRESH",
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
  email: string;
  passwordHash: string;
}

export interface TokenUser {
  id: string;
  username: string;
  email: string;
  status: UserStatus;
}

export interface UserWithPassword {
  id: string;
  username: string;
  email: string;
  passwordHash: string;
  status: UserStatus;
  type: UserType;
}

export interface UserWithEmail {
  id: string;
  username: string;
  email: string;
  status: UserStatus;
  type: UserType;
}
