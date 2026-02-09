import { NextResponse } from "next/server";

import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function GET(_: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });
  try {
    const res = await fetch(`${getBackendBaseUrl()}/api/v1/blockchain/documents/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return NextResponse.json({ message: data?.detail || "Failed" }, { status: res.status });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ message: "Cannot connect to backend" }, { status: 503 });
  }
}
