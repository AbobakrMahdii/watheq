"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

import { toast } from "sonner";

// ─── Types ──────────────────────────────────────────────────────────

export type Notification = {
  id: number;
  verification_id: number;
  message: string;
  document_type_name?: string;
  user_name?: string;
  failure_stage?: string;
  failure_reason_code?: string;
  is_read: boolean;
  created_at: string;
};

export type VerificationStats = {
  SUCCESS: number;
  FAILED: number;
  RUNNING: number;
  PENDING: number;
  total: number;
};

type NotificationContextValue = {
  notifications: Notification[];
  unreadCount: number;
  verificationStats: VerificationStats;
  markAsRead: (id: number) => Promise<void>;
  markAllRead: () => Promise<void>;
  refresh: () => Promise<void>;
};

const EMPTY_STATS: VerificationStats = { SUCCESS: 0, FAILED: 0, RUNNING: 0, PENDING: 0, total: 0 };

const NotificationContext = createContext<NotificationContextValue>({
  notifications: [],
  unreadCount: 0,
  verificationStats: EMPTY_STATS,
  markAsRead: async () => {},
  markAllRead: async () => {},
  refresh: async () => {},
});

export const useNotifications = () => useContext(NotificationContext);

/** Clear cookie + hard-redirect to login (idempotent). */
function redirectToLogin() {
  document.cookie = "token=; path=/; max-age=0";
  window.location.href = "/auth/login";
}

// ─── Provider ───────────────────────────────────────────────────────

export default function NotificationProvider({ children }: { children: ReactNode }) {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [verificationStats, setVerificationStats] = useState<VerificationStats>(EMPTY_STATS);
  const esRef = useRef<EventSource | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay = useRef(1000); // Start at 1 s, exponential backoff

  // ── Fetch persisted notifications on mount ─────────────────────
  const fetchNotifications = useCallback(async () => {
    try {
      const res = await fetch("/api/notifications?unread_only=true&page_size=50");
      if (res.status === 401) {
        redirectToLogin();
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      setNotifications(Array.isArray(data.items) ? data.items : []);
      setUnreadCount(data.unread_count ?? 0);
    } catch {
      // Silently ignore — will retry on next poll/reconnect
    }
  }, []);

  // ── Fetch admin-wide verification stats ────────────────────────
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/verifications/stats");
      if (res.status === 401) {
        redirectToLogin();
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      setVerificationStats({
        SUCCESS: data.SUCCESS ?? 0,
        FAILED: data.FAILED ?? 0,
        RUNNING: data.RUNNING ?? 0,
        PENDING: data.PENDING ?? 0,
        total: data.total ?? 0,
      });
    } catch {
      // Silently ignore
    }
  }, []);

  // ── SSE connection ─────────────────────────────────────────────
  const connectSSE = useCallback(() => {
    // Close prev connection if any
    esRef.current?.close();

    const es = new EventSource("/api/notifications/stream");
    esRef.current = es;

    es.onopen = () => {
      reconnectDelay.current = 1000; // Reset backoff on success
    };

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // ── Handle verification stats updates ──
        if (data.type === "VERIFICATION_RUNNING") {
          setVerificationStats((prev) => ({
            ...prev,
            PENDING: Math.max(0, prev.PENDING - 1),
            RUNNING: prev.RUNNING + 1,
          }));
        } else if (data.type === "VERIFICATION_SUCCESS") {
          setVerificationStats((prev) => ({
            ...prev,
            RUNNING: Math.max(0, prev.RUNNING - 1),
            SUCCESS: prev.SUCCESS + 1,
          }));
        } else if (data.type === "VERIFICATION_FAILED") {
          // Update stats counters
          setVerificationStats((prev) => ({
            ...prev,
            RUNNING: Math.max(0, prev.RUNNING - 1),
            FAILED: prev.FAILED + 1,
          }));

          // Add notification to list
          const newNotification: Notification = {
            id: data.id,
            verification_id: data.verification_id,
            message: data.message,
            document_type_name: data.document_type_name,
            user_name: data.user_name,
            failure_stage: data.failure_stage,
            failure_reason_code: data.failure_reason_code,
            is_read: false,
            created_at: data.created_at,
          };

          setNotifications((prev) => [newNotification, ...prev]);
          setUnreadCount((prev) => prev + 1);

          // Show sonner toast with action
          toast.error("فشل في التحقق", {
            description: data.message,
            duration: 8000,
            action: {
              label: "عرض التفاصيل",
              onClick: () => {
                window.location.href = `/dashboard/verifications/${data.verification_id}`;
              },
            },
          });
        }
      } catch {
        // Ignore malformed events
      }
    };

    es.onerror = () => {
      // EventSource doesn't go through window.fetch, so AuthGuard
      // can't intercept its 401s. Detect server-side rejection
      // (readyState === CLOSED before we close) vs transient errors.
      const wasRejected = es.readyState === EventSource.CLOSED;
      es.close();

      if (wasRejected) {
        // Server refused the connection (likely 401 / expired token).
        // Probe auth via a normal fetch — if 401, redirect to login.
        fetch("/api/notifications?page_size=1")
          .then((r) => {
            if (r.status === 401) redirectToLogin();
          })
          .catch(() => {});
        return; // Stop reconnecting
      }

      // Transient network error — exponential backoff reconnect (max 30s)
      const delay = Math.min(reconnectDelay.current, 30000);
      reconnectTimer.current = setTimeout(() => {
        reconnectDelay.current = delay * 2;
        connectSSE();
      }, delay);
    };
  }, []);

  useEffect(() => {
    fetchNotifications();
    fetchStats();
    connectSSE();

    return () => {
      esRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [fetchNotifications, fetchStats, connectSSE]);

  // ── Mark as read ───────────────────────────────────────────────
  const markAsRead = useCallback(async (id: number) => {
    try {
      await fetch(`/api/notifications/${id}/read`, { method: "PATCH" });
      setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, is_read: true } : n)));
      setUnreadCount((prev) => Math.max(0, prev - 1));
    } catch {
      // Ignore
    }
  }, []);

  const markAllRead = useCallback(async () => {
    try {
      await fetch("/api/notifications/read-all", { method: "PATCH" });
      setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
      setUnreadCount(0);
    } catch {
      // Ignore
    }
  }, []);

  const refresh = useCallback(async () => {
    await Promise.all([fetchNotifications(), fetchStats()]);
  }, [fetchNotifications, fetchStats]);

  return (
    <NotificationContext.Provider
      value={{ notifications, unreadCount, verificationStats, markAsRead, markAllRead, refresh }}
    >
      {children}
    </NotificationContext.Provider>
  );
}
