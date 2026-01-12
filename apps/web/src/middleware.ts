import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { IDENTITY_TOKEN_COOKIE_NAME } from "@tcg/backend-core/constants/auth";

const protectedRoutes = ["/dashboard", "/decks", "/rooms", "/profile"];
const authRoutes = ["/login", "/signup"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  const identityToken = request.cookies.get(IDENTITY_TOKEN_COOKIE_NAME);
  const isAuthenticated = !!identityToken;

  // If authenticated user tries to access auth pages, redirect to dashboard
  if (isAuthenticated && authRoutes.includes(pathname)) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  // If unauthenticated user tries to access protected routes, redirect to login
  const isProtectedRoute = protectedRoutes.some(
    (route) => pathname === route || pathname.startsWith(`${route}/`)
  );

  if (!isAuthenticated && isProtectedRoute) {
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
     */
    "/((?!api|_next/static|_next/image|favicon.ico).*)",
  ],
};
