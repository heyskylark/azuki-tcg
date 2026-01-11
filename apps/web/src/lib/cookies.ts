import { cookies } from "next/headers";
import type { ResponseCookie } from "next/dist/compiled/@edge-runtime/cookies";
import {
  ACCESS_TOKEN_COOKIE_NAME,
  REFRESH_TOKEN_COOKIE_NAME,
  IDENTITY_TOKEN_COOKIE_NAME,
  ACCESS_TOKEN_EXPIRY_SECONDS,
  REFRESH_TOKEN_EXPIRY_SECONDS,
  IDENTITY_TOKEN_EXPIRY_SECONDS,
} from "@tcg/backend-core/constants/auth";
import type { TokenPair } from "@tcg/backend-core/types/auth";
import { isProduction } from "@/lib/env";

const baseCookieOptions: Partial<ResponseCookie> = {
  sameSite: "strict",
  secure: isProduction,
  path: "/",
};

export async function setAuthCookies(tokens: TokenPair): Promise<void> {
  const cookieStore = await cookies();

  cookieStore.set(ACCESS_TOKEN_COOKIE_NAME, tokens.accessToken, {
    ...baseCookieOptions,
    httpOnly: true,
    maxAge: ACCESS_TOKEN_EXPIRY_SECONDS,
  });

  cookieStore.set(REFRESH_TOKEN_COOKIE_NAME, tokens.refreshToken, {
    ...baseCookieOptions,
    httpOnly: true,
    maxAge: REFRESH_TOKEN_EXPIRY_SECONDS,
  });

  cookieStore.set(IDENTITY_TOKEN_COOKIE_NAME, tokens.identityToken, {
    ...baseCookieOptions,
    httpOnly: false,
    maxAge: IDENTITY_TOKEN_EXPIRY_SECONDS,
  });
}

export async function clearAuthCookies(): Promise<void> {
  const cookieStore = await cookies();

  cookieStore.delete(ACCESS_TOKEN_COOKIE_NAME);
  cookieStore.delete(REFRESH_TOKEN_COOKIE_NAME);
  cookieStore.delete(IDENTITY_TOKEN_COOKIE_NAME);
}

export async function getAccessToken(): Promise<string | undefined> {
  const cookieStore = await cookies();
  return cookieStore.get(ACCESS_TOKEN_COOKIE_NAME)?.value;
}

export async function getRefreshToken(): Promise<string | undefined> {
  const cookieStore = await cookies();
  return cookieStore.get(REFRESH_TOKEN_COOKIE_NAME)?.value;
}
