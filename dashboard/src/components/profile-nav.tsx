"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { toast } from "sonner";

import { useNotifications } from "@/components/notification-provider";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

type Me = {
  id?: number | string;
  name?: string;
  username?: string;
  email?: string;
  role?: string;
  avatar_url?: string;
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

function StatPill({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone?: "neutral" | "success" | "warning" | "danger";
}) {
  const toneClasses: Record<string, string> = {
    neutral: "bg-slate-100 text-slate-700",
    success: "bg-emerald-100 text-emerald-700",
    warning: "bg-amber-100 text-amber-800",
    danger: "bg-rose-100 text-rose-700",
  };
  return (
    <div className="rounded-lg border border-slate-200 bg-white/80 px-3 py-2 shadow-[0_1px_0_rgba(0,0,0,0.04)]">
      <div className="text-[11px] font-medium text-slate-500">{label}</div>
      <div
        className={`mt-1 inline-flex items-center gap-2 rounded-full px-2 py-1 text-sm font-semibold ${toneClasses[tone || "neutral"]}`}
      >
        {value}
      </div>
    </div>
  );
}

export default function ProfileNav() {
  const [me, setMe] = useState<Me | null>(null);
  const { verificationStats } = useNotifications();

  useEffect(() => {
    async function loadProfile() {
      try {
        const meRes = await fetch("/api/auth/me");
        const meData = await meRes.json().catch(() => ({}));
        if (meRes.ok) setMe(meData);
      } catch (error: any) {
        toast.error(error?.message || "تعذر تحميل بيانات الملف الشخصي");
      }
    }
    loadProfile();
  }, []);

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
          <Badge variant="secondary" className="rounded-full px-3 py-1 text-[11px] tracking-wide uppercase">
            {me?.role || "ROLE"}
          </Badge>
          <span className="text-xs text-slate-500">ID: {me?.id ?? "—"}</span>
          <Link
            href="/dashboard/profile"
            className="ml-2 rounded-md bg-slate-900 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-slate-800"
          >
            الملف الشخصي
          </Link>
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatPill label="إجمالي التحققات" value={verificationStats.total} tone="neutral" />
        <StatPill label="الناجحة" value={verificationStats.SUCCESS} tone="success" />
        <StatPill label="قيد التنفيذ" value={verificationStats.RUNNING + verificationStats.PENDING} tone="warning" />
        <StatPill label="الفاشلة" value={verificationStats.FAILED} tone="danger" />
      </div>

      <Separator className="my-4" />
    </div>
  );
}
