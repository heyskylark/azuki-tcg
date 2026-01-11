// Token expiry times (in seconds)
export const ACCESS_TOKEN_EXPIRY_SECONDS = 60 * 60; // 1 hour
export const REFRESH_TOKEN_EXPIRY_SECONDS = 60 * 60 * 24 * 7; // 7 days
export const IDENTITY_TOKEN_EXPIRY_SECONDS = 60 * 60 * 24 * 7; // 7 days (matches refresh)

// Cookie names
export const ACCESS_TOKEN_COOKIE_NAME = "access_token";
export const REFRESH_TOKEN_COOKIE_NAME = "refresh_token";
export const IDENTITY_TOKEN_COOKIE_NAME = "identity_token";

// Password validation
export const PASSWORD_MIN_LENGTH = 8;
export const PASSWORD_MAX_LENGTH = 128;

// Username validation
export const USERNAME_MIN_LENGTH = 3;
export const USERNAME_MAX_LENGTH = 32;
export const USERNAME_REGEX = /^[a-zA-Z0-9_-]+$/;

