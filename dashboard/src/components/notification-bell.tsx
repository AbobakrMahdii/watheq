"use client";

import { useRouter } from "next/navigation";
import { Bell } from "lucide-react";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useNotifications, type Notification } from "@/components/notification-provider";

// ─── Relative time helper ───────────────────────────────────────────

function timeAgo(dateStr: string): string {
  try {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "الآن";
    if (mins < 60) return `منذ ${mins} دقيقة`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `منذ ${hours} ساعة`;
    const days = Math.floor(hours / 24);
    return `منذ ${days} يوم`;
  } catch {
    return "";
  }
}

// ─── Single notification row ────────────────────────────────────────

function NotificationRow({
  notification,
  onClick,
}: {
  notification: Notification;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-right px-3 py-2.5 rounded-lg transition-colors hover:bg-slate-100 ${
        notification.is_read ? "opacity-60" : "bg-rose-50/60"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-slate-800 leading-snug truncate">
            فشل التحقق #{notification.verification_id}
          </p>
          <p className="mt-0.5 text-xs text-slate-600 line-clamp-2">
            {notification.message}
          </p>
          {notification.failure_stage && (
            <p className="mt-0.5 text-[11px] text-rose-500">
              المرحلة: {notification.failure_stage}
            </p>
          )}
        </div>
        <span className="shrink-0 text-[10px] text-slate-400 mt-0.5">
          {timeAgo(notification.created_at)}
        </span>
      </div>
    </button>
  );
}

// ─── Main component ─────────────────────────────────────────────────

export default function NotificationBell() {
  const router = useRouter();
  const { notifications, unreadCount, markAsRead, markAllRead } =
    useNotifications();

  const handleClick = async (n: Notification) => {
    if (!n.is_read) await markAsRead(n.id);
    router.push(`/dashboard/verifications/${n.verification_id}`);
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          className="relative rounded-full p-2 text-slate-600 hover:bg-slate-100 hover:text-slate-900 transition-colors"
          aria-label="الإشعارات"
        >
          <Bell className="h-5 w-5" />
          {unreadCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-rose-500 px-1 text-[10px] font-bold text-white">
              {unreadCount > 99 ? "99+" : unreadCount}
            </span>
          )}
        </button>
      </PopoverTrigger>

      <PopoverContent
        align="end"
        sideOffset={8}
        className="w-[360px] p-0 shadow-xl border border-slate-200 rounded-xl overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-4 py-3">
          <h3 className="text-sm font-semibold text-slate-800">الإشعارات</h3>
          {unreadCount > 0 && (
            <button
              onClick={() => markAllRead()}
              className="text-xs text-blue-600 hover:text-blue-800 transition-colors"
            >
              تعيين الكل كمقروء
            </button>
          )}
        </div>

        {/* Body */}
        <div className="max-h-[380px] overflow-y-auto">
          {notifications.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-slate-400">
              <Bell className="h-8 w-8 mb-2 opacity-40" />
              <p className="text-sm">لا توجد إشعارات</p>
            </div>
          ) : (
            <div className="flex flex-col gap-0.5 p-1.5">
              {notifications.map((n) => (
                <NotificationRow
                  key={n.id}
                  notification={n}
                  onClick={() => handleClick(n)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        {notifications.length > 0 && (
          <div className="border-t border-slate-200 bg-slate-50 px-4 py-2 text-center">
            <button
              onClick={() => router.push("/dashboard/verifications?status=FAILED")}
              className="text-xs text-blue-600 hover:text-blue-800 transition-colors"
            >
              عرض جميع التحققات الفاشلة
            </button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
