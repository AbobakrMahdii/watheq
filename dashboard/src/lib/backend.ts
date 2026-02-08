import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export function getBackendBaseUrl() {
  return process.env.BACKEND_BASE_URL || "http://localhost:8001";
}

export async function getBearerTokenFromCookies() {
  return (await cookies()).get("token")?.value || null;
}

/**
 * If the upstream response is 401, clear the token cookie so the middleware
 * redirects to login on the next navigation.
 */
export function clearTokenOnUnauthorized(
  upstreamStatus: number,
  response: NextResponse
): NextResponse {
  if (upstreamStatus === 401) {
    response.cookies.set("token", "", { path: "/", maxAge: 0 });
  }
  return response;
}
