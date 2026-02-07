"use client";

import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";

type Me = {
  id?: number | string;
  name?: string;
  username?: string;
  email?: string;
  role?: string;
  avatar_url?: string;
};

type Verification = {
  id: number;
  document_type_id?: number;
  status: "SUCCESS" | "FAILED" | "RUNNING" | "PENDING" | string;
  created_at?: string;
  end_time?: string;
};

type StatusCounts = {
  SUCCESS: number;
  FAILED: number;
  RUNNING: number;
  PENDING: number;
  total: number;
};

const statusStyles: Record<string, string> = {
  SUCCESS: "bg-emerald-100 text-emerald-700",
  FAILED: "bg-rose-100 text-rose-700",
  RUNNING: "bg-amber-100 text-amber-800",
  PENDING: "bg-slate-100 text-slate-700",
};

function initials(text?: string) {
  if (!text) return "U";
  const letters = text
    .split(" ")
    .filter(Boolean)
    .map((word) => word[0]?.toUpperCase())
    .filter(Boolean);
  return letters.slice(0, 2).join("") || "U";
}

function formatDate(value?: string) {
  if (!value) return "—";
  try {
    const date = new Date(value);
    return new Intl.DateTimeFormat("ar-EG", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  } catch {
    return value;
  }
}

function StatPill({ label, value, tone }: { label: string; value: number; tone?: "neutral" | "success" | "warning" | "danger" }) {
  const toneClasses: Record<string, string> = {
    neutral: "bg-slate-100 text-slate-700",
    success: "bg-emerald-100 text-emerald-700",
    warning: "bg-amber-100 text-amber-800",
    danger: "bg-rose-100 text-rose-700",
  };
  return (
    <div className="rounded-lg border border-slate-200 bg-white/80 px-3 py-2 shadow-[0_1px_0_rgba(0,0,0,0.04)]">
      <div className="text-[11px] font-medium text-slate-500">{label}</div>
      <div className={`mt-1 inline-flex items-center gap-2 rounded-full px-2 py-1 text-sm font-semibold ${toneClasses[tone || "neutral"]}`}>
        {value}
      </div>
    </div>
  );
}

export default function ProfileNav() {
  const [me, setMe] = useState<Me | null>(null);
  const [verifications, setVerifications] = useState<Verification[]>([]);
  const [counts, setCounts] = useState<StatusCounts>({
    SUCCESS: 0,
    FAILED: 0,
    RUNNING: 0,
    PENDING: 0,
    total: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const meRes = await fetch("/api/auth/me");
        const meData = await meRes.json().catch(() => ({}));
        if (meRes.ok) setMe(meData);

        const verRes = await fetch("/api/verifications/my?page=1&page_size=10");
        const verData = await verRes.json().catch(() => ({}));
        if (verRes.ok) {
          setVerifications(Array.isArray(verData.items) ? verData.items : []);
          setCounts({
            SUCCESS: verData?.status_counts?.SUCCESS || 0,
            FAILED: verData?.status_counts?.FAILED || 0,
            RUNNING: verData?.status_counts?.RUNNING || 0,
            PENDING: verData?.status_counts?.PENDING || 0,
            total: verData?.total || (verData.items?.length ?? 0),
          });
        } else {
          throw new Error(verData?.message || "تعذر تحميل عمليات التحقق");
        }
      } catch (error: any) {
        toast.error(error?.message || "تعذر تحميل بيانات الملف الشخصي");
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  const latestVerifications = useMemo(() => verifications.slice(0, 4), [verifications]);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/70 px-4 py-4 shadow-sm backdrop-blur">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-3">
          <Avatar className="h-12 w-12 ring-2 ring-slate-900/5">
            {me?.avatar_url ? <AvatarImage src={me.avatar_url} alt={me?.name || me?.email || "user"} /> : null}
            <AvatarFallback>{initials(me?.name || me?.username || me?.email)}</AvatarFallback>
          </Avatar>
          <div>
            <div className="text-xs text-slate-500">حسابي</div>
            <div className="text-base font-semibold text-slate-900">
              {me?.name || me?.username || me?.email || "مستخدم"}
            </div>
            <div className="text-xs text-slate-500">{me?.email || "—"}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="rounded-full px-3 py-1 text-[11px] uppercase tracking-wide">
            {me?.role || "ROLE"}
          </Badge>
          <span className="text-xs text-slate-500">ID: {me?.id ?? "—"}</span>
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatPill label="إجمالي التحققات" value={counts.total} tone="neutral" />
        <StatPill label="الناجحة" value={counts.SUCCESS} tone="success" />
        <StatPill label="قيد التنفيذ" value={counts.RUNNING + counts.PENDING} tone="warning" />
        <StatPill label="الفاشلة" value={counts.FAILED} tone="danger" />
      </div>

      <Separator className="my-4" />

      <div>
        <div className="mb-2 text-sm font-semibold text-slate-700">آخر عمليات التحقق الخاصة بك</div>
        {loading ? (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {Array.from({ length: 4 }).map((_, idx) => (
              <Skeleton key={idx} className="h-16 rounded-lg" />
            ))}
          </div>
        ) : latestVerifications.length === 0 ? (
          <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 px-3 py-4 text-sm text-slate-500">
            لا توجد عمليات تحقق مسجلة لهذا المستخدم بعد.
          </div>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {latestVerifications.map((v) => (
              <div key={v.id} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3">
                <div className="flex items-center justify-between">
                  <div className={`rounded-full px-2 py-1 text-[11px] font-semibold ${statusStyles[v.status] || statusStyles.PENDING}`}>
                    {v.status}
                  </div>
                  <span className="text-[11px] text-slate-500">#{v.id}</span>
                </div>
                <div className="mt-2 text-xs text-slate-600">
                  نوع الوثيقة: <span className="font-semibold text-slate-800">{v.document_type_id ?? "—"}</span>
                </div>
                <div className="text-[11px] text-slate-500">{formatDate(v.created_at || v.end_time)}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
