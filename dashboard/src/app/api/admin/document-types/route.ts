import { NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

const backendBaseUrl = getBackendBaseUrl();

export async function GET() {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });

  const upstream = await fetch(`${backendBaseUrl}/api/admin/document-types`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Failed to fetch document types";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}

export async function POST(request: Request) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });

  const body = await request.json();

  const upstream = await fetch(`${backendBaseUrl}/api/admin/document-types`, {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Failed to create document type";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data, { status: upstream.status });
}
