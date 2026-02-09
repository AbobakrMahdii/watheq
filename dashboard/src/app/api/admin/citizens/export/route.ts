import { NextRequest, NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const fetchCache = "force-no-store";

/** GET export citizens file */
export async function GET(req: NextRequest) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Missing token" }, { status: 401 });

  const upstream = await fetch(`${getBackendBaseUrl()}/api/v1/admin/citizens/export${req.nextUrl.search}`, {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "text/csv,application/pdf",
    },
    cache: "no-store",
  });

  if (!upstream.ok) {
    const data = await upstream.json().catch(() => ({}));
    return NextResponse.json({ message: data?.detail || "Error" }, { status: upstream.status });
  }

  const headers = new Headers();
  const contentType = upstream.headers.get("content-type");
  const contentDisposition = upstream.headers.get("content-disposition");
  if (contentType) headers.set("content-type", contentType);
  if (contentDisposition) headers.set("content-disposition", contentDisposition);
  headers.set("cache-control", "no-store");

  return new Response(upstream.body, { status: 200, headers });
}
