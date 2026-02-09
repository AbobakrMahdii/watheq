import { getBackendBaseUrl, getBearerTokenFromCookies } from "@/lib/backend";

/**
 * SSE proxy — pipes the backend SSE stream through to the browser.
 * The token is read from the httpOnly cookie and forwarded as a query
 * param (EventSource can't send custom headers).
 */
export async function GET() {
  const token = await getBearerTokenFromCookies();
  if (!token) {
    return new Response(JSON.stringify({ message: "Missing token" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const backendUrl = `${getBackendBaseUrl()}/api/v1/notifications/stream?token=${encodeURIComponent(token)}`;

  try {
    const upstream = await fetch(backendUrl, {
      headers: { Accept: "text/event-stream" },
      cache: "no-store",
    });

    if (!upstream.ok || !upstream.body) {
      return new Response(JSON.stringify({ message: "SSE connection failed" }), {
        status: upstream.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Pipe the backend ReadableStream through to the client
    return new Response(upstream.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return new Response(JSON.stringify({ message: "Failed to connect to backend SSE" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }
}
