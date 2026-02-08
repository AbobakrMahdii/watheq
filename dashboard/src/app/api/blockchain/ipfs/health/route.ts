import { NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export async function GET() {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });
  try {
    const res = await fetch(`${getBackendBaseUrl()}/api/v1/blockchain/ipfs/health`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return NextResponse.json({ message: data?.detail || "Failed" }, { status: res.status });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ message: "Cannot connect to backend" }, { status: 503 });
  }
}
