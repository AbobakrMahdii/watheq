"use client";

import { useEffect, useMemo, useState } from "react";
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
      { href: "/dashboard", label: "Overview" },
      { href: "/dashboard/users", label: "Users" },
      { href: "/dashboard/document-types", label: "Document Types" },
      { href: "/dashboard/audit-logs", label: "Audit Logs" },
    ];
    if (me?.role === "super_admin") {
      baseLinks.splice(2, 0, { href: "/dashboard/admins", label: "Admins" });
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
    <aside className="w-64 bg-white border-r min-h-screen p-4">
      <div className="mb-6">
        <div className="text-sm text-slate-500">Signed in as</div>
        <div className="font-medium truncate">{me?.email || "—"}</div>
        <div className="text-xs text-slate-500">{me?.role || "—"}</div>
      </div>

      <nav className="space-y-1">
        {links.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`block rounded px-3 py-2 text-sm ${
                active ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>

      <button
        onClick={logout}
        className="mt-6 w-full rounded border px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
      >
        Logout
      </button>
    </aside>
  );
}
