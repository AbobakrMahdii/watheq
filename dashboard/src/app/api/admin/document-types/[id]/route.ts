import { NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

const backendBaseUrl = getBackendBaseUrl();

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });

  const { id } = await params;

  const upstream = await fetch(`${backendBaseUrl}/api/admin/document-types/${id}`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Failed to fetch document type";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data);
}

export async function PUT(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });

  const { id } = await params;
  const body = await request.json();

  const upstream = await fetch(`${backendBaseUrl}/api/admin/document-types/${id}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });

  const data = await upstream.json().catch(() => ({}));
  if (!upstream.ok) {
    const message = data?.detail?.message || data?.detail || data?.message || "Failed to update document type";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return NextResponse.json(data, { status: upstream.status });
}

export async function DELETE(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });

  const { id } = await params;

  const upstream = await fetch(`${backendBaseUrl}/api/admin/document-types/${id}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!upstream.ok) {
    const data = await upstream.json().catch(() => ({}));
    const message = data?.detail?.message || data?.detail || data?.message || "Failed to delete document type";
    return NextResponse.json({ message }, { status: upstream.status });
  }

  return new NextResponse(null, { status: 204 });
}
