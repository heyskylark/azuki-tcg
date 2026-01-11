import { NextRequest, NextResponse } from "next/server";
import { ZodError } from "zod";
import { ApiError } from "@tcg/backend-core/errors";

type SimpleRouteHandler = (
  request: NextRequest
) => Promise<NextResponse> | NextResponse;

type DynamicRouteHandler<T> = (
  request: NextRequest,
  context: T
) => Promise<NextResponse> | NextResponse;

export function withErrorHandler(handler: SimpleRouteHandler): SimpleRouteHandler;
export function withErrorHandler<T>(handler: DynamicRouteHandler<T>): DynamicRouteHandler<T>;
export function withErrorHandler<T = unknown>(
  handler: SimpleRouteHandler | DynamicRouteHandler<T>
): SimpleRouteHandler | DynamicRouteHandler<T> {
  return async (request: NextRequest, context?: T): Promise<NextResponse> => {
    try {
      if (context !== undefined) {
        return await (handler as DynamicRouteHandler<T>)(request, context);
      }
      return await (handler as SimpleRouteHandler)(request);
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
