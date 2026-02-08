import { NextRequest, NextResponse } from "next/server";

import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function GET(req: NextRequest) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const url = new URL(req.url);
  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/admin/analytics?${url.searchParams.toString()}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Request failed";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}
