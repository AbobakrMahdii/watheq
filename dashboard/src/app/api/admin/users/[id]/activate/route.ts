import { NextResponse } from "next/server";

import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function PUT(_: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/admin/users/${id}/activate`, {
    method: "PUT",
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Request failed";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}
