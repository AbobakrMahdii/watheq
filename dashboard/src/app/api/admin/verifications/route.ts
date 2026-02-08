import { NextRequest, NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function GET(req: NextRequest) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  // Forward all query params
  const url = new URL(req.url);
  const upstream = await fetch(`${getBackendBaseUrl()}/api/admin/verifications?${url.searchParams.toString()}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    return NextResponse.json({ message: data?.detail || "Request failed" }, { status: upstream.status });
  }
  return NextResponse.json(data);
}
