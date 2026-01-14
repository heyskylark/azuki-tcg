import { z } from "zod";

/**
 * Validates that a redirect path is safe (internal path, not external URL).
 * Prevents open redirect attacks.
 */

// Schema for validating internal paths
const internalPathSchema = z.string().min(1).refine(
  (path) => {
    // Must start with /
    if (!path.startsWith("/")) return false;

    // Must not be protocol-relative (//example.com)
    if (path.startsWith("//")) return false;

    // Must not contain suspicious patterns that could lead to external redirects
    const suspicious = ["://", "\\", "@"];
    if (suspicious.some((s) => path.includes(s))) return false;

    return true;
  },
  { message: "Invalid redirect path" }
);

/**
 * Validates that a redirect path is safe.
 * Returns the validated path or null if invalid.
 */
export function validateRedirectPath(
  path: string | null | undefined
): string | null {
  if (!path) return null;

  const result = internalPathSchema.safeParse(path);
  return result.success ? result.data : null;
}

/**
 * Get a safe redirect path, falling back to default if validation fails.
 */
export function getSafeRedirectPath(
  path: string | null | undefined,
  defaultPath = "/dashboard"
): string {
  const validated = validateRedirectPath(path);
  return validated ?? defaultPath;
}
