"use client";

import { useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { getSafeRedirectPath } from "@/lib/auth/validateRedirectPath";

function RefreshContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const returnTo = searchParams.get("returnTo");
    const safeReturnTo = getSafeRedirectPath(returnTo, "/dashboard");

    async function performRefresh() {
      try {
        const response = await fetch("/api/users/auth/refresh", {
          method: "POST",
        });

        if (response.ok) {
          // Refresh successful, redirect to returnTo
          router.replace(safeReturnTo);
        } else {
          // Refresh failed, redirect to login with the original destination
          const loginUrl = `/login?redirect=${encodeURIComponent(safeReturnTo)}`;
          router.replace(loginUrl);
        }
      } catch {
        // Network error, redirect to login
        const safeReturn = getSafeRedirectPath(returnTo, "/dashboard");
        const loginUrl = `/login?redirect=${encodeURIComponent(safeReturn)}`;
        router.replace(loginUrl);
      }
    }

    performRefresh();
  }, [router, searchParams]);

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Refreshing Session</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        <p className="text-muted-foreground">Please wait...</p>
      </CardContent>
    </Card>
  );
}

export default function RefreshPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Suspense
        fallback={
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>Loading...</CardTitle>
            </CardHeader>
            <CardContent className="flex justify-center py-8">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
            </CardContent>
          </Card>
        }
      >
        <RefreshContent />
      </Suspense>
    </div>
  );
}
