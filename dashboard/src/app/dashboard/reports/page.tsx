"use client";

import { useEffect, useState } from "react";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/* ---------- Types ---------- */
type Analytics = {
  total_verifications: number;
  status_breakdown?: Record<string, number>;
  by_document_type?: { type: string; count: number }[];
  failure_reasons?: { reason: string; count: number }[];
  avg_processing_time_sec?: number;
};

const STATUS_COLORS: Record<string, string> = {
  SUCCESS: "#22c55e", // green
  FAILED: "#ef4444", // red
  PENDING: "#3b82f6", // blue
  RUNNING: "#f59e0b", // amber
};
const STATUS_AR: Record<string, string> = {
  SUCCESS: "ناجح",
  FAILED: "فشل",
  RUNNING: "قيد التنفيذ",
  PENDING: "في الانتظار",
};

/* ---------- Component ---------- */
export default function ReportsPage() {
  const [data, setData] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState<null | "csv" | "pdf">(null);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  async function load() {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (dateFrom) params.set("date_from", dateFrom);
      if (dateTo) params.set("date_to", dateTo);

      const res = await fetch(`/api/admin/analytics?${params}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json?.message || "فشل التحميل");
      setData(json);
    } catch (e: any) {
      toast.error(e?.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  function getExportFilename(headerValue: string | null) {
    if (!headerValue) return null;
    const match = headerValue.match(/filename="?([^";]+)"?/i);
    return match?.[1] || null;
  }

  async function downloadExport(format: "csv" | "pdf") {
    setExporting(format);
    try {
      const params = new URLSearchParams();
      if (dateFrom) params.set("date_from", dateFrom);
      if (dateTo) params.set("date_to", dateTo);
      params.set("format", format);

      const res = await fetch(`/api/admin/analytics/export?${params}`, { cache: "no-store" });
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        throw new Error(json?.message || "Failed to export analytics");
      }

      const blob = await res.blob();
      const filename =
        getExportFilename(res.headers.get("content-disposition")) ||
        `analytics_export_${new Date().toISOString().replace(/[:.]/g, "-")}.${format}`;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      toast.success("تم تنزيل التصدير");
    } catch (e: any) {
      toast.error(e?.message || "تعذر تنزيل التصدير");
    } finally {
      setExporting(null);
    }
  }

  const statusPie = data?.status_breakdown
    ? Object.entries(data.status_breakdown).map(([k, v]) => ({
        name: STATUS_AR[k] || k,
        value: v,
        color: STATUS_COLORS[k] || "#94a3b8",
      }))
    : [];

  const successRate =
    data?.status_breakdown && data.total_verifications
      ? (((data.status_breakdown["SUCCESS"] || 0) / data.total_verifications) * 100).toFixed(1)
      : "—";

  return (
    <div className="max-w-5xl space-y-6" dir="rtl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">التقارير والإحصائيات</h1>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => downloadExport("csv")} disabled={exporting === "csv"}>
            {exporting === "csv" ? "Exporting..." : "Export CSV"}
          </Button>
          <Button size="sm" variant="outline" onClick={() => downloadExport("pdf")} disabled={exporting === "pdf"}>
            {exporting === "pdf" ? "Exporting..." : "Export PDF"}
          </Button>
        </div>
      </div>

      {/* Date Filter */}
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="text-xs text-slate-500">من تاريخ</label>
          <input
            type="date"
            className="block rounded border px-3 py-1.5 text-sm"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
          />
        </div>
        <div>
          <label className="text-xs text-slate-500">إلى تاريخ</label>
          <input
            type="date"
            className="block rounded border px-3 py-1.5 text-sm"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
          />
        </div>
        <Button size="sm" onClick={load} disabled={loading}>
          {loading ? "جاري..." : "تطبيق"}
        </Button>
      </div>

      {/* Summary */}
      {loading ? (
        <div className="grid gap-4 sm:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : data ? (
        <div className="grid gap-4 sm:grid-cols-3">
          <Card>
            <CardContent className="pt-4">
              <div className="text-xs text-slate-500">إجمالي التحققات</div>
              <div className="mt-1 text-2xl font-semibold">{data.total_verifications}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-xs text-slate-500">نسبة النجاح</div>
              <div className="mt-1 text-2xl font-semibold">{successRate}%</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="text-xs text-slate-500">متوسط وقت التحقق</div>
              <div className="mt-1 text-2xl font-semibold">{data.avg_processing_time_sec ?? "—"}ث</div>
            </CardContent>
          </Card>
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Status breakdown pie */}
        {statusPie.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>توزيع الحالات</CardTitle>
            </CardHeader>
            <CardContent className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusPie}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={90}
                    dataKey="value"
                    label={({ name, value }) => `${name} (${value})`}
                  >
                    {statusPie.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* By document type */}
        {data?.by_document_type && data.by_document_type.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>تحليل حسب نوع الوثيقة</CardTitle>
            </CardHeader>
            <CardContent className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.by_document_type}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Failure reasons */}
      {data?.failure_reasons && data.failure_reasons.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>أنماط أسباب الفشل</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.failure_reasons} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="reason" type="category" width={160} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
