import { NextRequest, NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

type RouteCtx = { params: Promise<{ id: string }> };

export async function GET(_req: NextRequest, ctx: RouteCtx) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const { id } = await ctx.params;

  const upstream = await fetch(`${getBackendBaseUrl()}/api/admin/verifications/${id}/steps`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    return NextResponse.json({ message: data?.detail || "Not found" }, { status: upstream.status });
  }
  return NextResponse.json(data);
}
