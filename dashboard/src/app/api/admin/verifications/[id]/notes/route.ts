import { NextRequest, NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

type RouteCtx = { params: Promise<{ id: string }> };

/** GET notes for a verification */
export async function GET(_req: NextRequest, ctx: RouteCtx) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const { id } = await ctx.params;
  const upstream = await fetch(`${getBackendBaseUrl()}/api/admin/verifications/${id}/notes`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    return NextResponse.json({ message: data?.detail || "Error" }, { status: upstream.status });
  }
  return NextResponse.json(data);
}

/** POST add a note */
export async function POST(req: NextRequest, ctx: RouteCtx) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const { id } = await ctx.params;
  const body = await req.json();
  const upstream = await fetch(`${getBackendBaseUrl()}/api/admin/verifications/${id}/notes`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    return NextResponse.json({ message: data?.detail || "Error" }, { status: upstream.status });
  }
  return NextResponse.json(data);
}
