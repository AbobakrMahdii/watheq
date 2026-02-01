import { NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

const backendBaseUrl = getBackendBaseUrl();

export async function GET(request: Request) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });

  try {
    const url = new URL(request.url);
    const format = url.searchParams.get("format") || "pdf";
    const upstream = await fetch(`${backendBaseUrl}/api/admin/audit-logs/export?format=${encodeURIComponent(format)}`, {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (!upstream.ok) {
      const data = await upstream.json().catch(() => ({}));
      const message = data?.detail?.message || data?.detail || data?.message || "Failed to export audit logs";
      return NextResponse.json({ message }, { status: upstream.status });
    }

    const headers = new Headers();
    const contentType = upstream.headers.get("content-type") || "application/octet-stream";
    headers.set("content-type", contentType);

    const contentDisposition = upstream.headers.get("content-disposition");
    if (contentDisposition) headers.set("content-disposition", contentDisposition);

    return new Response(upstream.body, { status: upstream.status, headers });
  } catch (error: any) {
    const message = error?.message || "Cannot connect to backend";
    return NextResponse.json({ message }, { status: 503 });
  }
}
