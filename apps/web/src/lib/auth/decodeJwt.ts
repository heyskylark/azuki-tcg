import { z } from "zod";

/**
 * JWT decoding utility for middleware.
 * Decodes JWT tokens without verification (for reading expiry claims).
 */

const jwtClaimsSchema = z.object({
  exp: z.number().optional(),
  iat: z.number().optional(),
  sub: z.string().optional(),
  jti: z.string().optional(),
});

export type JwtClaims = z.infer<typeof jwtClaimsSchema>;

/**
 * Decode a JWT without verification.
 * Used in middleware to check token expiry without requiring the secret.
 * Returns null if decoding fails or validation fails.
 */
export function decodeJwt(token: string): JwtClaims | null {
  try {
    const [, payload] = token.split(".");
    if (!payload) return null;

    // Handle base64url encoding (replace - with +, _ with /)
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const decoded = atob(base64);
    const parsed: unknown = JSON.parse(decoded);

    const result = jwtClaimsSchema.safeParse(parsed);
    return result.success ? result.data : null;
  } catch {
    return null;
  }
}

/**
 * Check if a JWT token is expired.
 * Returns true if expired or if token cannot be decoded.
 */
export function isTokenExpired(token: string): boolean {
  const claims = decodeJwt(token);
  if (!claims?.exp) return true;

  // exp is in seconds, Date.now() is in milliseconds
  const nowSeconds = Math.floor(Date.now() / 1000);
  return claims.exp <= nowSeconds;
}
