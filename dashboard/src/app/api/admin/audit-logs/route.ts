import { NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

const backendBaseUrl = getBackendBaseUrl();

export async function GET(request: Request) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });

  try {
    const url = new URL(request.url);
    const upstream = await fetch(`${backendBaseUrl}/api/admin/audit-logs${url.search}`, {
      headers: { Authorization: `Bearer ${token}` },
    });

    const data = await upstream.json().catch(() => ({}));
    if (!upstream.ok) {
      const message = data?.detail?.message || data?.detail || data?.message || "Failed to fetch audit logs";
      return NextResponse.json({ message }, { status: upstream.status });
    }

    return NextResponse.json(data);
  } catch (error: any) {
    const message = error?.message || "Cannot connect to backend";
    return NextResponse.json({ message }, { status: 503 });
  }
}
