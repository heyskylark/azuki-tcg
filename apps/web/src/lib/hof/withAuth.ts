import { NextRequest, NextResponse } from "next/server";
import { verifyAccessToken } from "@tcg/backend-core/services/authService";
import { getUserWithEmail } from "@tcg/backend-core/services/userService";
import {
  UnauthorizedError,
  UserBannedError,
  UserDeletedError,
} from "@tcg/backend-core/errors";
import { UserStatus } from "@tcg/backend-core/types";
import type { AuthenticatedUser, AuthConfig } from "@tcg/backend-core/types/auth";
import { getAccessToken } from "@/lib/cookies";
import { env } from "@/lib/env";

export interface AuthenticatedRequest extends NextRequest {
  user: AuthenticatedUser;
}

type SimpleAuthenticatedHandler = (
  request: AuthenticatedRequest
) => Promise<NextResponse> | NextResponse;

type DynamicAuthenticatedHandler<T> = (
  request: AuthenticatedRequest,
  context: T
) => Promise<NextResponse> | NextResponse;

type SimpleRouteHandler = (
  request: NextRequest
) => Promise<NextResponse>;

type DynamicRouteHandler<T> = (
  request: NextRequest,
  context: T
) => Promise<NextResponse>;

const authConfig: AuthConfig = {
  jwtSecret: env.JWT_SECRET,
  jwtIssuer: env.JWT_ISSUER,
  saltRounds: env.PASSWORD_SALT_ROUNDS,
};

export function withAuth(handler: SimpleAuthenticatedHandler): SimpleRouteHandler;
export function withAuth<T>(handler: DynamicAuthenticatedHandler<T>): DynamicRouteHandler<T>;
export function withAuth<T = unknown>(
  handler: SimpleAuthenticatedHandler | DynamicAuthenticatedHandler<T>
): SimpleRouteHandler | DynamicRouteHandler<T> {
  return async (request: NextRequest, context?: T): Promise<NextResponse> => {
    const accessToken = await getAccessToken();

    if (!accessToken) {
      throw new UnauthorizedError("No access token provided");
    }

    const payload = await verifyAccessToken(accessToken, authConfig);

    const userWithEmail = await getUserWithEmail(payload.sub);

    if (!userWithEmail) {
      throw new UnauthorizedError("User not found");
    }

    if (userWithEmail.status === UserStatus.BANNED) {
      throw new UserBannedError();
    }

    if (userWithEmail.status === UserStatus.DELETED) {
      throw new UserDeletedError();
    }

    const authenticatedRequest = Object.assign(request, {
      user: {
        id: userWithEmail.id,
        username: userWithEmail.username,
        displayName: userWithEmail.displayName,
        email: userWithEmail.email,
        status: userWithEmail.status,
        type: userWithEmail.type,
      },
    });

    if (context !== undefined) {
      return (handler as DynamicAuthenticatedHandler<T>)(authenticatedRequest, context);
    }
    return (handler as SimpleAuthenticatedHandler)(authenticatedRequest);
  };
}
