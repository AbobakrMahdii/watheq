import { NextRequest, NextResponse } from "next/server";

import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function GET(request: NextRequest) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const url = new URL(request.url);
  const page = url.searchParams.get("page") || "1";
  const pageSize = url.searchParams.get("page_size") || url.searchParams.get("pageSize") || "20";

  const upstream = await fetch(
    `${getBackendBaseUrl()}/api/v1/verifications/my?page=${page}&page_size=${pageSize}`,
    { headers: { Authorization: `Bearer ${token}` } },
  );

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.message || "Failed to load verifications";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}
