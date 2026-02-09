"""In-memory notification event bus for SSE push to admin dashboards.

Uses asyncio.Queue per connected admin — no external broker needed.
When a verification fails, the orchestrator calls `notification_bus.broadcast(...)`
and all connected admin SSE streams receive the event instantly.
Notifications are also persisted to MySQL so offline admins see them on next login.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from api.database import database

logger = logging.getLogger("watheq.notifications")


class NotificationBus:
    """Singleton in-process event bus for admin SSE connections."""

    def __init__(self) -> None:
        # admin_id -> list of asyncio.Queue (one admin can have multiple tabs)
        self._subscribers: Dict[int, list[asyncio.Queue]] = {}

    def subscribe(self, admin_id: int) -> asyncio.Queue:
        """Register a new SSE listener for *admin_id*. Returns the queue to read from."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(admin_id, []).append(queue)
        logger.info("Admin %s subscribed (connections: %s)", admin_id, len(self._subscribers[admin_id]))
        return queue

    def unsubscribe(self, admin_id: int, queue: asyncio.Queue) -> None:
        """Remove a specific SSE listener."""
        queues = self._subscribers.get(admin_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._subscribers.pop(admin_id, None)
        logger.info("Admin %s unsubscribed", admin_id)

    async def broadcast(self, event: Dict[str, Any]) -> None:
        """Push *event* to ALL connected admin streams."""
        for admin_id, queues in self._subscribers.items():
            for q in queues:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("Queue full for admin %s — dropping event", admin_id)


# Module-level singleton
notification_bus = NotificationBus()


# ─── Persistence helpers ────────────────────────────────────────────

async def persist_notification(
    verification_id: int,
    message: str,
    document_type_name: Optional[str] = None,
    user_name: Optional[str] = None,
    failure_stage: Optional[str] = None,
    failure_reason_code: Optional[str] = None,
) -> int:
    """Insert a notification row and return its id."""
    row_id = await database.execute(
        """
        INSERT INTO notifications
            (verification_id, message, document_type_name, user_name,
             failure_stage, failure_reason_code, is_read, created_at)
        VALUES
            (:vid, :msg, :dtn, :un, :fs, :frc, FALSE, :ts)
        """,
        values={
            "vid": verification_id,
            "msg": message,
            "dtn": document_type_name,
            "un": user_name,
            "fs": failure_stage,
            "frc": failure_reason_code,
            "ts": datetime.now(timezone.utc),
        },
    )
    return int(row_id)


async def get_notifications(
    *,
    unread_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[Dict[str, Any]]:
    """Fetch notifications (newest first)."""
    where = "WHERE is_read = FALSE" if unread_only else ""
    rows = await database.fetch_all(
        f"""
        SELECT id, verification_id, message, document_type_name, user_name,
               failure_stage, failure_reason_code, is_read, created_at
        FROM notifications
        {where}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
        """,
        values={"limit": limit, "offset": offset},
    )
    return [dict(r) for r in rows]


async def get_unread_count() -> int:
    row = await database.fetch_one(
        "SELECT COUNT(*) AS cnt FROM notifications WHERE is_read = FALSE"
    )
    return int(row["cnt"]) if row else 0


async def mark_as_read(notification_id: int) -> None:
    await database.execute(
        "UPDATE notifications SET is_read = TRUE WHERE id = :nid",
        values={"nid": notification_id},
    )


async def mark_all_read() -> None:
    await database.execute("UPDATE notifications SET is_read = TRUE WHERE is_read = FALSE")
