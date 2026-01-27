import { cookies } from "next/headers";

export function getBackendBaseUrl() {
  return process.env.BACKEND_BASE_URL || "http://localhost:8001";
}

export async function getBearerTokenFromCookies() {
  return (await cookies()).get("token")?.value || null;
}
