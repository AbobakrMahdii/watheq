import { NextResponse } from "next/server";

import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function GET(_: Request, context: { params: Promise<{ nationalId: string }> }) {
  const { nationalId } = await context.params;
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/admin/citizens/${nationalId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Request failed";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}

export async function PUT(req: Request, context: { params: Promise<{ nationalId: string }> }) {
  const { nationalId } = await context.params;
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const body = await req.json().catch(() => ({}));

  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/admin/citizens/${nationalId}`, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Request failed";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}

export async function DELETE(_: Request, context: { params: Promise<{ nationalId: string }> }) {
  const { nationalId } = await context.params;
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/admin/citizens/${nationalId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Request failed";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}
