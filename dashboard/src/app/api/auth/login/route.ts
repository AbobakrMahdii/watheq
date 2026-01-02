import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { username, password } = body ?? {};

    // Very simple credential check for scaffold/demo purposes
    if (username === "admin" && password === "admin") {
      const token = `token-${Date.now()}`;
      return NextResponse.json({ token });
    }

    return new NextResponse(JSON.stringify({ message: "Invalid credentials" }), { status: 401 });
  } catch (err) {
    return new NextResponse(JSON.stringify({ message: "Bad request" }), { status: 400 });
  }
}
