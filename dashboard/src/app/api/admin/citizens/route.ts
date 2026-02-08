import { NextResponse } from "next/server";

import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function GET(req: Request) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const limit = searchParams.get("limit") || "50";
  const offset = searchParams.get("offset") || "0";

  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/admin/citizens?limit=${limit}&offset=${offset}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Request failed";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}
