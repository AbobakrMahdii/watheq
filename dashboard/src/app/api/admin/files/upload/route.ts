import { NextResponse } from "next/server";
import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

const backendBaseUrl = getBackendBaseUrl();

export async function POST(request: Request) {
  const token = await getBearerTokenFromCookies();
  if (!token) return NextResponse.json({ message: "Unauthorized" }, { status: 401 });

  try {
    const formData = await request.formData();
    const upstream = await fetch(`${backendBaseUrl}/api/v1/files/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });

    const data = await upstream.json().catch(() => ({}));
    if (!upstream.ok) {
      const message = data?.detail?.message || data?.detail || data?.message || "Failed to upload file";
      return NextResponse.json({ message }, { status: upstream.status });
    }

    return NextResponse.json(data);
  } catch (error: any) {
    const message = error?.message || "Cannot connect to backend";
    return NextResponse.json({ message }, { status: 503 });
  }
}
