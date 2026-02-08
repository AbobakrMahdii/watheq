import { NextRequest, NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

type RouteCtx = { params: Promise<{ id: string }> };

/** PATCH /api/notifications/[id]/read */
export async function PATCH(_req: NextRequest, ctx: RouteCtx) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const { id } = await ctx.params;

  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/notifications/${id}/read`, {
    method: "PATCH",
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    return NextResponse.json({ message: data?.detail || "Request failed" }, { status: upstream.status });
  }
  return NextResponse.json(data);
}
