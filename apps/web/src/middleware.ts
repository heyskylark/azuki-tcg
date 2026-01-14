import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import {
  ACCESS_TOKEN_COOKIE_NAME,
  REFRESH_TOKEN_COOKIE_NAME,
} from "@tcg/backend-core/constants/auth";
import { isTokenExpired } from "@/lib/auth/decodeJwt";

const protectedRoutes = ["/dashboard", "/decks", "/rooms", "/profile"];
const authRoutes = ["/login", "/signup"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  const accessToken = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME);
  const refreshToken = request.cookies.get(REFRESH_TOKEN_COOKIE_NAME);

  const hasAccess = !!accessToken?.value;
  const hasRefresh = !!refreshToken?.value;

  // Determine if user has a valid (non-expired) access token
  const hasValidAccess = hasAccess && !isTokenExpired(accessToken.value);

  // If authenticated user tries to access auth pages, redirect to dashboard
  if (hasValidAccess && authRoutes.includes(pathname)) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  // Check if this is a protected route
  const isProtectedRoute = protectedRoutes.some(
    (route) => pathname === route || pathname.startsWith(`${route}/`)
  );

  if (isProtectedRoute) {
    if (hasValidAccess) {
      // Valid access token, allow through
      return NextResponse.next();
    }

    if (hasRefresh) {
      // No valid access token but has refresh token -> redirect to refresh
      const refreshUrl = new URL("/refresh", request.url);
      refreshUrl.searchParams.set("returnTo", pathname);
      return NextResponse.redirect(refreshUrl);
    }

    // No valid access token and no refresh token -> redirect to login
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("redirect", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - api routes
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico
     * - public files
     * - /refresh route (needs to be accessible for token refresh)
     */
    "/((?!api|_next/static|_next/image|favicon.ico|refresh).*)",
  ],
};
