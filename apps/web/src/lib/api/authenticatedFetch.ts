/**
 * A fetch wrapper that automatically handles 401 responses by refreshing
 * the access token and retrying the request.
 */

let isRefreshing = false;
let refreshPromise: Promise<boolean> | null = null;

async function refreshToken(): Promise<boolean> {
  const response = await fetch("/api/users/auth/refresh", {
    method: "POST",
  });
  return response.ok;
}

async function waitForRefresh(): Promise<boolean> {
  if (refreshPromise) {
    return refreshPromise;
  }

  isRefreshing = true;
  refreshPromise = refreshToken();

  try {
    const result = await refreshPromise;
    return result;
  } finally {
    isRefreshing = false;
    refreshPromise = null;
  }
}

export async function authenticatedFetch(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  // If already refreshing, wait for it to complete before making request
  if (isRefreshing) {
    await waitForRefresh();
  }

  const response = await fetch(input, init);

  // If we get a 401, try to refresh the token and retry
  if (response.status === 401) {
    const refreshed = await waitForRefresh();

    if (refreshed) {
      // Retry the original request
      return fetch(input, init);
    }

    // Refresh failed, redirect to login with current path
    if (typeof window !== "undefined") {
      const currentPath = window.location.pathname;
      window.location.href = `/login?redirect=${encodeURIComponent(currentPath)}`;
    }
  }

  return response;
}
