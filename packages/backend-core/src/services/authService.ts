import * as bcrypt from "bcrypt";
import { SignJWT, jwtVerify, type JWTPayload as JoseJWTPayload } from "jose";
import { eq, and, isNull, gt } from "drizzle-orm";
import { uuidv7 } from "uuidv7";
import { z } from "zod";
import db, { type IDatabase, type ITransaction } from "@/database";
import { JwtTokens } from "@/drizzle/schemas/jwt_tokens";
import {
  TokenExpiredError,
  TokenRevokedError,
  InvalidTokenError,
} from "@/errors";
import {
  TokenType,
  type JWTPayload,
  type IdentityTokenPayload,
  type TokenPair,
  type AuthConfig,
  type TokenUser,
  type JoinTokenPayload,
} from "@/types/auth";
import {
  ACCESS_TOKEN_EXPIRY_SECONDS,
  REFRESH_TOKEN_EXPIRY_SECONDS,
  IDENTITY_TOKEN_EXPIRY_SECONDS,
} from "@/constants/auth";

const joinTokenPayloadSchema = z.object({
  roomId: z.string(),
  playerSlot: z.union([z.literal(0), z.literal(1)]),
});

type Database = IDatabase | ITransaction;

export async function hashPassword(password: string, saltRounds = 12): Promise<string> {
  return bcrypt.hash(password, saltRounds);
}

export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

export async function createTokens(
  user: TokenUser,
  config: AuthConfig,
  database: Database = db
): Promise<TokenPair> {
  const now = Math.floor(Date.now() / 1000);
  const accessJti = uuidv7();
  const refreshJti = uuidv7();
  const secretKey = new TextEncoder().encode(config.jwtSecret);

  const accessToken = await new SignJWT({})
    .setProtectedHeader({ alg: "HS256" })
    .setIssuer(config.jwtIssuer)
    .setSubject(user.id)
    .setAudience(config.jwtIssuer)
    .setExpirationTime(now + ACCESS_TOKEN_EXPIRY_SECONDS)
    .setIssuedAt(now)
    .setJti(accessJti)
    .sign(secretKey);

  const refreshToken = await new SignJWT({})
    .setProtectedHeader({ alg: "HS256" })
    .setIssuer(config.jwtIssuer)
    .setSubject(user.id)
    .setAudience(config.jwtIssuer)
    .setExpirationTime(now + REFRESH_TOKEN_EXPIRY_SECONDS)
    .setIssuedAt(now)
    .setJti(refreshJti)
    .sign(secretKey);

  const identityPayload: IdentityTokenPayload = {
    userId: user.id,
    username: user.username,
    displayName: user.displayName,
    email: user.email,
    status: user.status,
  };
  const identityToken = await new SignJWT(
    identityPayload as unknown as JoseJWTPayload
  )
    .setProtectedHeader({ alg: "HS256" })
    .setExpirationTime(now + IDENTITY_TOKEN_EXPIRY_SECONDS)
    .setIssuedAt(now)
    .sign(secretKey);

  await database.insert(JwtTokens).values([
    {
      jti: accessJti,
      userId: user.id,
      tokenType: TokenType.ACCESS,
      expiresAt: new Date((now + ACCESS_TOKEN_EXPIRY_SECONDS) * 1000),
    },
    {
      jti: refreshJti,
      userId: user.id,
      tokenType: TokenType.REFRESH,
      expiresAt: new Date((now + REFRESH_TOKEN_EXPIRY_SECONDS) * 1000),
    },
  ]);

  return { accessToken, refreshToken, identityToken };
}

async function verifyToken(
  token: string,
  expectedType: TokenType,
  config: AuthConfig,
  database: Database = db
): Promise<JWTPayload> {
  const secretKey = new TextEncoder().encode(config.jwtSecret);

  try {
    const { payload } = await jwtVerify(token, secretKey, {
      issuer: config.jwtIssuer,
      audience: config.jwtIssuer,
    });

    const jti = payload.jti;
    if (!jti) {
      throw new InvalidTokenError();
    }

    const tokenRecord = await database
      .select({
        id: JwtTokens.id,
        revokedAt: JwtTokens.revokedAt,
        tokenType: JwtTokens.tokenType,
      })
      .from(JwtTokens)
      .where(
        and(
          eq(JwtTokens.jti, jti),
          eq(JwtTokens.tokenType, expectedType),
          gt(JwtTokens.expiresAt, new Date())
        )
      )
      .limit(1);

    if (tokenRecord.length === 0) {
      throw new InvalidTokenError();
    }

    const record = tokenRecord[0]!;
    if (record.revokedAt !== null) {
      throw new TokenRevokedError();
    }

    return {
      iss: payload.iss!,
      sub: payload.sub!,
      aud: payload.aud as string,
      exp: payload.exp!,
      iat: payload.iat!,
      jti: payload.jti!,
    };
  } catch (error) {
    if (
      error instanceof TokenRevokedError ||
      error instanceof InvalidTokenError
    ) {
      throw error;
    }
    if (error instanceof Error && error.name === "JWTExpired") {
      throw new TokenExpiredError();
    }
    throw new InvalidTokenError();
  }
}

export async function verifyAccessToken(
  token: string,
  config: AuthConfig,
  database: Database = db
): Promise<JWTPayload> {
  return verifyToken(token, TokenType.ACCESS, config, database);
}

export async function verifyRefreshToken(
  token: string,
  config: AuthConfig,
  database: Database = db
): Promise<JWTPayload> {
  return verifyToken(token, TokenType.REFRESH, config, database);
}

export async function verifyJoinToken(
  token: string,
  config: AuthConfig,
  database: Database = db
): Promise<JWTPayload & JoinTokenPayload> {
  const secretKey = new TextEncoder().encode(config.jwtSecret);

  try {
    const { payload } = await jwtVerify(token, secretKey, {
      issuer: config.jwtIssuer,
      audience: config.jwtIssuer,
    });

    const jti = payload.jti;
    if (!jti) {
      throw new InvalidTokenError();
    }

    const tokenRecord = await database
      .select({
        id: JwtTokens.id,
        revokedAt: JwtTokens.revokedAt,
        tokenType: JwtTokens.tokenType,
      })
      .from(JwtTokens)
      .where(
        and(
          eq(JwtTokens.jti, jti),
          eq(JwtTokens.tokenType, TokenType.JOIN),
          gt(JwtTokens.expiresAt, new Date())
        )
      )
      .limit(1);

    if (tokenRecord.length === 0) {
      throw new InvalidTokenError();
    }

    const record = tokenRecord[0];
    if (!record) {
      throw new InvalidTokenError();
    }
    if (record.revokedAt !== null) {
      throw new TokenRevokedError();
    }

    const joinPayloadResult = joinTokenPayloadSchema.safeParse(payload);
    if (!joinPayloadResult.success) {
      throw new InvalidTokenError();
    }

    const { roomId, playerSlot } = joinPayloadResult.data;

    if (!payload.iss || !payload.sub || !payload.aud || !payload.exp || !payload.iat || !payload.jti) {
      throw new InvalidTokenError();
    }

    const aud = Array.isArray(payload.aud) ? payload.aud[0] : payload.aud;
    if (!aud) {
      throw new InvalidTokenError();
    }

    return {
      iss: payload.iss,
      sub: payload.sub,
      aud,
      exp: payload.exp,
      iat: payload.iat,
      jti: payload.jti,
      roomId,
      playerSlot,
    };
  } catch (error) {
    if (
      error instanceof TokenRevokedError ||
      error instanceof InvalidTokenError
    ) {
      throw error;
    }
    if (error instanceof Error && error.name === "JWTExpired") {
      throw new TokenExpiredError();
    }
    throw new InvalidTokenError();
  }
}

export async function revokeToken(
  jti: string,
  database: Database = db
): Promise<void> {
  await database
    .update(JwtTokens)
    .set({ revokedAt: new Date() })
    .where(eq(JwtTokens.jti, jti));
}

export async function revokeAllUserTokens(
  userId: string,
  database: Database = db
): Promise<void> {
  await database
    .update(JwtTokens)
    .set({ revokedAt: new Date() })
    .where(
      and(
        eq(JwtTokens.userId, userId),
        isNull(JwtTokens.revokedAt),
        gt(JwtTokens.expiresAt, new Date())
      )
    );
}

export async function refreshTokens(
  refreshToken: string,
  user: TokenUser,
  config: AuthConfig,
  database: Database = db
): Promise<TokenPair> {
  const payload = await verifyRefreshToken(refreshToken, config, database);
  await revokeToken(payload.jti, database);
  return createTokens(user, config, database);
}
