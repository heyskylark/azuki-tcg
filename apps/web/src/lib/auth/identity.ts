import { IDENTITY_TOKEN_COOKIE_NAME } from "@tcg/backend-core/constants/auth";
import type { UserStatus } from "@tcg/backend-core/types";

export interface IdentityUser {
  userId: string;
  username: string;
  displayName: string;
  email: string;
  status: UserStatus;
}

export function getIdentityTokenFromCookie(): string | null {
  if (typeof document === "undefined") return null;
  const cookies = document.cookie.split(";");
  for (const cookie of cookies) {
    const [name, value] = cookie.trim().split("=");
    if (name === IDENTITY_TOKEN_COOKIE_NAME) {
      return value ?? null;
    }
  }
  return null;
}

export function parseIdentityToken(token: string): IdentityUser | null {
  try {
    const [, payload] = token.split(".");
    if (!payload) return null;
    const decoded = atob(payload);
    return JSON.parse(decoded) as IdentityUser;
  } catch {
    return null;
  }
}

export function getIdentityUser(): IdentityUser | null {
  const token = getIdentityTokenFromCookie();
  if (!token) return null;
  return parseIdentityToken(token);
}
