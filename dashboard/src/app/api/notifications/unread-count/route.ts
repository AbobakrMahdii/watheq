import { NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

/** GET /api/notifications/unread-count */
export async function GET() {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/notifications/unread-count`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    return NextResponse.json({ message: data?.detail || "Request failed" }, { status: upstream.status });
  }
  return NextResponse.json(data);
}
