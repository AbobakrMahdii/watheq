import { NextResponse } from "next/server";

import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function POST(req: Request) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  try {
    const res = await fetch(`${getBackendBaseUrl()}/api/v1/blockchain/publish`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return NextResponse.json({ message: data?.detail || "Failed" }, { status: res.status });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ message: "Cannot connect to backend" }, { status: 503 });
  }
}
