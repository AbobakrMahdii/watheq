"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

type Analytics = {
  total_users: number;
  total_admins: number;
  total_verifications: number;
  total_authentications: number;
  total_document_types: number;
  total_audit_logs: number;
};

export default function DashboardPage() {
  const router = useRouter();
  const [stats, setStats] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(false);

  const cards = useMemo(() => {
    if (!stats) return [];
    return [
      { label: "Total Users", value: stats.total_users },
      { label: "Total Admins", value: stats.total_admins },
      { label: "Total Authentications", value: stats.total_authentications },
      { label: "Total Verifications", value: stats.total_verifications },
      { label: "Document Types", value: stats.total_document_types },
      { label: "Audit Logs", value: stats.total_audit_logs },
    ];
  }, [stats]);

  async function loadAnalytics() {
    setLoading(true);
    try {
      const res = await fetch("/api/admin/analytics");
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Failed to load analytics");
      setStats(data);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load analytics");
      router.push("/auth/login");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAnalytics();
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold">System Analytics</h1>
          <p className="text-sm text-slate-500">Overview of key system metrics.</p>
        </div>
        <button
          onClick={loadAnalytics}
          className="text-sm bg-slate-900 text-white px-3 py-2 rounded hover:bg-slate-800 disabled:opacity-60"
          disabled={loading}
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {cards.length === 0 ? (
        <div className="rounded border bg-white p-6 text-sm text-slate-500">
          {loading ? "Loading analytics..." : "No data available."}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {cards.map((card) => (
            <div key={card.label} className="rounded border bg-white p-4 shadow-sm">
              <div className="text-xs text-slate-500">{card.label}</div>
              <div className="mt-2 text-2xl font-semibold">{card.value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
