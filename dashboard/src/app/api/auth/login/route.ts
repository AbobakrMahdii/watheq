import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const identifier = (body?.email || body?.username || "").toString().trim();
    const password = (body?.password || "").toString();

    if (!identifier || !password) {
      return NextResponse.json({ message: "Email/Username and password are required" }, { status: 400 });
    }

    const baseUrl = process.env.BACKEND_BASE_URL || "http://localhost:8000";
    const upstream = await fetch(`${baseUrl}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: identifier, password }),
    });

    const data = await upstream.json().catch(() => ({}));

    if (!upstream.ok) {
      const message = data?.detail?.message || data?.detail || data?.message || "Login failed";
      return NextResponse.json({ message }, { status: upstream.status });
    }

    // Normalize response for the dashboard
    const token = data?.access_token;
    if (!token) {
      return NextResponse.json({ message: "Missing access_token from backend" }, { status: 502 });
    }

    if (data?.role === "user") {
      return NextResponse.json({ message: "Users must login from the mobile app" }, { status: 403 });
    }

    const response = NextResponse.json({
      token,
      token_type: data?.token_type || "bearer",
      role: data?.role,
    });

    response.cookies.set("token", token, {
      httpOnly: true,
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24 * 7,
    });

    return response;
  } catch (err) {
    return new NextResponse(JSON.stringify({ message: "Bad request" }), { status: 400 });
  }
}
