"use client";

import { useEffect, type ReactNode } from "react";

/**
 * Global client-side 401 interceptor.
 * Monkey-patches `window.fetch` so that any response with status 401
 * from internal `/api/` routes automatically redirects to /auth/login.
 * Mount once in the dashboard layout.
 */
export default function AuthGuard({ children }: { children: ReactNode }) {
  useEffect(() => {
    const originalFetch = window.fetch;

    window.fetch = async function patchedFetch(input: RequestInfo | URL, init?: RequestInit) {
      const response = await originalFetch(input, init);

      if (response.status === 401) {
        const url = typeof input === "string" ? input : input instanceof URL ? input.pathname : input.url;

        // Only intercept our own API proxy calls, not external fetches
        if (url.startsWith("/api/")) {
          // Clear the cookie client-side
          document.cookie = "token=; path=/; max-age=0";
          window.location.href = "/auth/login";
        }
      }

      return response;
    };

    return () => {
      window.fetch = originalFetch;
    };
  }, []);

  return <>{children}</>;
}
