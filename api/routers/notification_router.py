"""Notification router — SSE streaming + REST management for admin dashboard.

Endpoints:
  GET /api/v1/notifications/stream?token=<jwt>  — SSE (real-time push)
  GET /api/v1/notifications                     — list notifications
  GET /api/v1/notifications/unread-count        — badge count
  PATCH /api/v1/notifications/{id}/read         — mark one as read
  PATCH /api/v1/notifications/read-all          — mark all as read
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from api.security import get_current_admin, decode_token
from api.services.notification_service import (
    notification_bus,
    get_notifications,
    get_unread_count,
    mark_as_read,
    mark_all_read,
)

logger = logging.getLogger("watheq.notifications.router")

router = APIRouter(prefix="/api/v1/notifications", tags=["Notifications"])

# ─── SSE endpoint ────────────────────────────────────────────────────


@router.get("/stream")
async def notification_stream(token: str = Query(...)):
    """Server-Sent Events stream for admin notifications.

    Auth is via query param `token` because the browser EventSource API
    cannot send custom headers.
    """
    # Validate token and ensure admin role
    payload = decode_token(token)
    role = payload.get("role", "")
    if role not in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="Admins only")

    admin_id = int(payload.get("sub", 0))
    if not admin_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    queue = notification_bus.subscribe(admin_id)

    async def event_generator():
        try:
            while True:
                try:
                    # Wait for an event, but send keepalive ping every 15s
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive comment to prevent connection timeout
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            notification_bus.unsubscribe(admin_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if present
        },
    )


# ─── REST endpoints ──────────────────────────────────────────────────


@router.get("")
async def list_notifications(
    admin: dict = Depends(get_current_admin),
    unread_only: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
):
    """List notifications (newest first). Optionally filter unread only."""
    offset = (page - 1) * page_size
    items = await get_notifications(unread_only=unread_only, limit=page_size, offset=offset)
    unread = await get_unread_count()
    return {
        "items": items,
        "unread_count": unread,
        "page": page,
        "page_size": page_size,
    }


@router.get("/unread-count")
async def unread_count(admin: dict = Depends(get_current_admin)):
    count = await get_unread_count()
    return {"unread_count": count}


@router.patch("/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    admin: dict = Depends(get_current_admin),
):
    await mark_as_read(notification_id)
    return {"ok": True}


@router.patch("/read-all")
async def mark_all_notifications_read(
    admin: dict = Depends(get_current_admin),
):
    await mark_all_read()
    return {"ok": True}
