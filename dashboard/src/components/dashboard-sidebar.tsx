"use client";

import { useEffect, useMemo, useState } from "react";

import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

import { toast } from "sonner";

type Me = {
  role?: string;
  email?: string;
};

export default function DashboardSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [me, setMe] = useState<Me | null>(null);

  useEffect(() => {
    fetch("/api/auth/me")
      .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (!ok) return;
        setMe(data);
      })
      .catch(() => null);
  }, []);

  const links = useMemo(() => {
    const baseLinks = [
      { href: "/dashboard", label: "نظرة عامة" },
      { href: "/dashboard/profile", label: "الملف الشخصي" },
      { href: "/dashboard/verifications", label: "التحققات" },
      { href: "/dashboard/users", label: "المستخدمون" },
      { href: "/dashboard/document-types", label: "أنواع الوثائق" },
      { href: "/dashboard/citizens", label: "سجلات المواطنين" },
      { href: "/dashboard/reports", label: "التقارير" },
      { href: "/dashboard/audit-logs", label: "سجل العمليات" },
      { href: "/dashboard/blockchain", label: "البلوكتشين" },
    ];
    if (me?.role === "super_admin" || me?.role === "admin_manager") {
      baseLinks.splice(3, 0, { href: "/dashboard/admins", label: "Admins" });
    }
    return baseLinks;
  }, [me?.role]);

  function logout() {
    fetch("/api/auth/logout", { method: "POST" })
      .catch(() => null)
      .finally(() => {
        toast.success("Logged out");
        router.push("/auth/login");
      });
  }

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col overflow-y-auto border-r bg-white">
      {/* Logo & Branding */}
      <div className="flex flex-col items-center gap-2 border-b px-4 py-5">
        <Image src="/logo.jpeg" alt="Watheq Logo" width={64} height={64} className="rounded-xl shadow-md" priority />
        <span className="text-lg font-bold tracking-wide text-slate-800">
          وثّق <span className="text-xs font-normal text-slate-400">Watheq</span>
        </span>
      </div>

      {/* User info */}
      <div className="border-b px-4 py-3">
        <div className="text-[11px] tracking-wider text-slate-400 uppercase">Signed in as</div>
        <div className="truncate text-sm font-medium text-slate-700">{me?.email || "—"}</div>
        <div className="text-xs text-slate-500">{me?.role || "—"}</div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-3">
        {links.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`block rounded-lg px-3 py-2 text-sm transition-colors ${
                active ? "bg-slate-900 font-medium text-white" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>

      {/* Logout */}
      <div className="border-t px-3 py-3">
        <button
          onClick={logout}
          className="w-full rounded-lg border px-3 py-2 text-sm text-slate-700 transition-colors hover:bg-red-50 hover:text-red-600"
        >
          Logout
        </button>
      </div>
    </aside>
  );
}
