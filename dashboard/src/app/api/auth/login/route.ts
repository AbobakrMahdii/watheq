import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => null);
    
    if (!body) {
      return NextResponse.json({ message: "Invalid JSON in request body" }, { status: 400 });
    }

    const identifier = (body?.email || body?.username || "").toString().trim();
    const password = (body?.password || "").toString();

    if (!identifier || !password) {
      return NextResponse.json({ message: "Email/Username and password are required" }, { status: 400 });
    }

    const baseUrl = process.env.BACKEND_BASE_URL || "http://localhost:8001";
    // Send email or username based on what user entered
    const loginPayload = identifier.includes("@") 
      ? { email: identifier, password }
      : { username: identifier, password };
    
    try {
      const upstream = await fetch(`${baseUrl}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(loginPayload),
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
        secure: process.env.NODE_ENV === "production",
        maxAge: 60 * 60 * 24 * 7,
      });

      return response;
    } catch (fetchError: any) {
      console.error("Backend fetch error:", fetchError);
      return NextResponse.json({ 
        message: `Cannot connect to backend: ${fetchError?.message || "Unknown error"}` 
      }, { status: 503 });
    }
  } catch (err: any) {
    console.error("Login route error:", err);
    return NextResponse.json({ 
      message: `Bad request: ${err?.message || "Unknown error"}` 
    }, { status: 400 });
  }
}
