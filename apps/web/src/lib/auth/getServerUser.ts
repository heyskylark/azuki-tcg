import { verifyAccessToken } from "@tcg/backend-core/services/authService";
import { getUserWithEmail } from "@tcg/backend-core/services/userService";
import { UserStatus } from "@tcg/backend-core/types";
import type { AuthenticatedUser, AuthConfig } from "@tcg/backend-core/types/auth";
import { getAccessToken } from "@/lib/cookies";
import { env } from "@/lib/env";

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

/**
 * Get the authenticated user from server-side cookies.
 * Returns null if not authenticated or user is in invalid state.
 * For use in Server Components.
 */
export async function getServerUser(): Promise<AuthenticatedUser | null> {
  const accessToken = await getAccessToken();

  if (!accessToken) {
    return null;
  }

  try {
    const payload = await verifyAccessToken(accessToken, authConfig);
    const userWithEmail = await getUserWithEmail(payload.sub);

    if (!userWithEmail) {
      return null;
    }

    if (
      userWithEmail.status === UserStatus.BANNED ||
      userWithEmail.status === UserStatus.DELETED
    ) {
      return null;
    }

    return {
      id: userWithEmail.id,
      username: userWithEmail.username,
      displayName: userWithEmail.displayName,
      email: userWithEmail.email,
      status: userWithEmail.status,
      type: userWithEmail.type,
    };
  } catch {
    return null;
  }
}
