"use client";

import { useEffect, useMemo, useState } from "react";

import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

/* ---------- Types ---------- */
type Analytics = {
  total_users: number;
  total_admins: number;
  total_verifications: number;
  total_authentications: number;
  total_document_types: number;
  total_audit_logs: number;
  status_breakdown?: Record<string, number>;
  by_document_type?: { type: string; count: number }[];
  time_series?: { date: string; SUCCESS: number; FAILED: number; RUNNING: number; PENDING: number }[];
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
export default function DashboardPage() {
  const [stats, setStats] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(false);

  const summaryCards = useMemo(() => {
    if (!stats) return [];
    return [
      { label: "المستخدمون", value: stats.total_users },
      { label: "المشرفون", value: stats.total_admins },
      { label: "التحققات", value: stats.total_verifications },
      { label: "أنواع الوثائق", value: stats.total_document_types },
      { label: "سجل العمليات", value: stats.total_audit_logs },
      {
        label: "متوسط وقت التحقق",
        value: stats.avg_processing_time_sec ? `${stats.avg_processing_time_sec}ث` : "—",
      },
    ];
  }, [stats]);

  const pieData = useMemo(() => {
    if (!stats?.status_breakdown) return [];
    return Object.entries(stats.status_breakdown).map(([key, val]) => ({
      name: STATUS_AR[key] || key,
      value: val,
      color: STATUS_COLORS[key] || "#94a3b8",
    }));
  }, [stats]);

  async function loadAnalytics() {
    setLoading(true);
    try {
      const res = await fetch("/api/admin/analytics");
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "فشل تحميل البيانات");
      setStats(data);
    } catch (e: any) {
      toast.error(e?.message || "فشل تحميل الإحصائيات");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAnalytics();
  }, []);

  return (
    <div className="space-y-6" dir="rtl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">نظرة عامة</h1>
          <p className="text-sm text-slate-500">إحصائيات النظام الرئيسية</p>
        </div>
        <Button variant="outline" size="sm" onClick={loadAnalytics} disabled={loading}>
          {loading ? "جاري التحميل..." : "تحديث"}
        </Button>
      </div>

      {/* Summary Cards */}
      {!stats ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded border" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {summaryCards.map((c) => (
            <Card key={c.label}>
              <CardContent className="pt-4">
                <div className="text-xs text-slate-500">{c.label}</div>
                <div className="mt-1 text-2xl font-semibold">{c.value}</div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Charts Row */}
      {stats && (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Status Pie */}
          {pieData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>توزيع الحالات</CardTitle>
              </CardHeader>
              <CardContent className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={90}
                      dataKey="value"
                      label={({ name, value }) => `${name} (${value})`}
                    >
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* By Document Type Bar */}
          {stats.by_document_type && stats.by_document_type.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>حسب نوع الوثيقة</CardTitle>
              </CardHeader>
              <CardContent className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats.by_document_type}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Time Series Line Chart */}
      {stats?.time_series && stats.time_series.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>التحققات اليومية (آخر 30 يوم)</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stats.time_series}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="SUCCESS" stroke="#22c55e" name="ناجح" strokeWidth={2} />
                <Line type="monotone" dataKey="FAILED" stroke="#ef4444" name="فشل" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Failure Reasons */}
      {stats?.failure_reasons && stats.failure_reasons.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>أسباب الفشل الشائعة</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.failure_reasons} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="reason" type="category" width={150} tick={{ fontSize: 11 }} />
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
