import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL || "http://localhost:8001";

// Cache token validation results in-memory to avoid hitting the backend on every request.
// Map<token, { valid: boolean; checkedAt: number }>
const tokenCache = new Map<string, { valid: boolean; checkedAt: number }>();
const CACHE_TTL_MS = 60_000; // 1 minute

async function isTokenValid(token: string): Promise<boolean> {
  const cached = tokenCache.get(token);
  if (cached && Date.now() - cached.checkedAt < CACHE_TTL_MS) {
    return cached.valid;
  }

  try {
    const res = await fetch(`${BACKEND_BASE_URL}/api/v1/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const valid = res.ok;
    tokenCache.set(token, { valid, checkedAt: Date.now() });
    // Evict stale entries periodically
    if (tokenCache.size > 500) {
      const now = Date.now();
      for (const [k, v] of tokenCache) {
        if (now - v.checkedAt > CACHE_TTL_MS) tokenCache.delete(k);
      }
    }
    return valid;
  } catch {
    // If backend is unreachable, allow through to avoid lockout
    return true;
  }
}

function clearTokenAndRedirectToLogin(request: NextRequest): NextResponse {
  const url = request.nextUrl.clone();
  url.pathname = "/auth/login";
  const response = NextResponse.redirect(url);
  // Expire the cookie
  response.cookies.set("token", "", { path: "/", maxAge: 0 });
  return response;
}

export async function middleware(request: NextRequest) {
  const token = request.cookies.get("token")?.value;
  const { pathname } = request.nextUrl;

  // Already on login page with a valid token → go to dashboard
  if (pathname.startsWith("/auth/login") && token) {
    const url = request.nextUrl.clone();
    url.pathname = "/dashboard";
    return NextResponse.redirect(url);
  }

  // Dashboard routes: require a valid token
  if (pathname.startsWith("/dashboard")) {
    if (!token) {
      return clearTokenAndRedirectToLogin(request);
    }

    const valid = await isTokenValid(token);
    if (!valid) {
      return clearTokenAndRedirectToLogin(request);
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*", "/auth/login"],
};
