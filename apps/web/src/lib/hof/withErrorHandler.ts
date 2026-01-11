import { NextRequest, NextResponse } from "next/server";
import { ZodError } from "zod";
import { ApiError } from "@tcg/backend-core/errors";

type RouteHandler<T = unknown> = (
  request: NextRequest,
  context?: T
) => Promise<NextResponse> | NextResponse;

export function withErrorHandler<T = unknown>(
  handler: RouteHandler<T>
): RouteHandler<T> {
  return async (request: NextRequest, context?: T): Promise<NextResponse> => {
    try {
      return await handler(request, context);
    } catch (error) {
      if (error instanceof ApiError) {
        return NextResponse.json(
          {
            error: error.name,
            message: error.message,
          },
          { status: error.status }
        );
      }

      if (error instanceof ZodError) {
        return NextResponse.json(
          {
            error: "ValidationError",
            message: error.issues[0]?.message ?? "Validation failed",
          },
          { status: 400 }
        );
      }

      console.error("Unhandled error in API route:", error);

      return NextResponse.json(
        {
          error: "InternalServerError",
          message: "An unexpected error occurred",
        },
        { status: 500 }
      );
    }
  };
}
