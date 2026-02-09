"use client";

import { useCallback, useEffect, useState } from "react";

import Link from "next/link";

import { toast } from "sonner";

import { SortableHeader } from "@/components/sortable-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useSortState } from "@/hooks/use-sort-state";

/* ---------- Types ---------- */
type Verification = {
  id: number;
  user_id: number;
  user_name?: string;
  user_email?: string;
  document_type_id: number;
  document_type_name?: string;
  status: string;
  current_stage?: string;
  created_at?: string;
};

type ListResponse = {
  total: number;
  page: number;
  page_size: number;
  items: Verification[];
};

const STATUS_OPTIONS = ["", "PENDING", "RUNNING", "SUCCESS", "FAILED"];
const STATUS_LABELS: Record<string, string> = {
  PENDING: "في الانتظار",
  RUNNING: "قيد التنفيذ",
  SUCCESS: "ناجح",
  FAILED: "فشل",
};
const STATUS_VARIANT: Record<
  string,
  "default" | "destructive" | "outline" | "secondary" | "success" | "warning" | "info"
> = {
  PENDING: "info",
  RUNNING: "warning",
  SUCCESS: "success",
  FAILED: "destructive",
};

const STAGE_LABELS: Record<string, string> = {
  DOCUMENT_IMAGE_QUALITY: "جودة الصورة",
  DOCUMENT_CROPPING: "قص الوثيقة",
  DOCUMENT_FACE_EXTRACTION: "استخراج الوجه",
  FACE_MATCHING: "مطابقة الوجه",
  OCR: "قراءة النصوص",
  AI_VERIFICATION: "تحقق الذكاء الاصطناعي",
  DATA_VERIFICATION: "مطابقة البيانات",
  BLOCKCHAIN: "تسجيل البلوكتشين",
};

/* ---------- Component ---------- */
export default function VerificationsPage() {
  const [data, setData] = useState<ListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [reportingId, setReportingId] = useState<number | null>(null);
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [userName, setUserName] = useState("");
  const [operationType, setOperationType] = useState("");
  const [status, setStatus] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const pageSize = 20;
  const { sortBy, sortOrder, toggleSort } = useSortState("created_at", "desc");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(page),
        page_size: String(pageSize),
      });
      if (status) params.set("status", status);
      if (search) params.set("search", search);
      if (userName) params.set("user_name", userName);
      if (operationType) params.set("operation_type", operationType);
      if (dateFrom) params.set("date_from", dateFrom);
      if (dateTo) params.set("date_to", dateTo);
      params.set("sort_by", sortBy);
      params.set("sort_order", sortOrder);

      const res = await fetch(`/api/admin/verifications?${params}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json?.message || "فشل تحميل البيانات");
      setData(json);
    } catch (e: any) {
      toast.error(e?.message || "تعذر تنزيل التقرير");
    } finally {
      setLoading(false);
    }
  }, [page, status, search, userName, operationType, dateFrom, dateTo, sortBy, sortOrder]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const t = setTimeout(() => {
      setSearch(searchInput.trim());
      setPage(1);
    }, 400);
    return () => clearTimeout(t);
  }, [searchInput]);

  function getExportFilename(headerValue: string | null) {
    if (!headerValue) return null;
    const match = headerValue.match(/filename="?([^";]+)"?/i);
    return match?.[1] || null;
  }

  async function exportData() {
    setExporting(true);
    try {
      const params = new URLSearchParams();
      if (status) params.set("status", status);
      if (search) params.set("search", search);
      if (userName) params.set("user_name", userName);
      if (operationType) params.set("operation_type", operationType);
      if (dateFrom) params.set("date_from", dateFrom);
      if (dateTo) params.set("date_to", dateTo);
      params.set("format", "csv");

      const res = await fetch(`/api/admin/verifications/export?${params}`);
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        throw new Error(json?.message || "Export failed");
      }

      const blob = await res.blob();
      const filename =
        getExportFilename(res.headers.get("content-disposition")) ||
        `verifications_export_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

      toast.success("تم تنزيل التقرير");
    } catch (e: any) {
      toast.error(e?.message || "Export failed");
    } finally {
      setExporting(false);
    }
  }

  function resetFilters() {
    setSearchInput("");
    setUserName("");
    setOperationType("");
    setStatus("");
    setDateFrom("");
    setDateTo("");
    setPage(1);
  }

  async function downloadReport(id: number) {
    setReportingId(id);
    try {
      const res = await fetch(`/api/admin/verifications/${id}/report?format=pdf`);
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        throw new Error(json?.message || "Failed to download report");
      }
      const blob = await res.blob();
      const filename =
        getExportFilename(res.headers.get("content-disposition")) ||
        `verification_${id}_${new Date().toISOString().replace(/[:.]/g, "-")}.pdf`;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      toast.success("تم تنزيل التقرير");
    } catch (e: any) {
      toast.error(e?.message || "تعذر تنزيل التقرير");
    } finally {
      setReportingId(null);
    }
  }

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;
  const operationOptions = Object.keys(STAGE_LABELS);

  return (
    <div className="space-y-6" dir="rtl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">التحققات</h1>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={exportData} disabled={exporting}>
            {exporting ? "جارٍ التصدير..." : "تصدير CSV"}
          </Button>
          <Button size="sm" variant="outline" onClick={load} disabled={loading}>
            {loading ? "جاري التحميل..." : "تحديث"}
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <Input
          placeholder="بحث بالاسم أو البريد أو الوحدة..."
          className="w-64"
          value={searchInput}
          onChange={(e) => {
            setSearchInput(e.target.value);
          }}
        />
        <Input
          placeholder="اسم المستخدم"
          className="w-48"
          value={userName}
          onChange={(e) => {
            setUserName(e.target.value);
            setPage(1);
          }}
        />
        <Select
          value={operationType || "ALL"}
          onValueChange={(v) => {
            setOperationType(v === "ALL" ? "" : v);
            setPage(1);
          }}
        >
          <SelectTrigger className="w-52">
            <SelectValue placeholder="نوع العملية" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ALL">الكل</SelectItem>
            {operationOptions.map((s) => (
              <SelectItem key={s} value={s}>
                {STAGE_LABELS[s] || s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select
          value={status || "ALL"}
          onValueChange={(v) => {
            setStatus(v === "ALL" ? "" : v);
            setPage(1);
          }}
        >
          <SelectTrigger className="w-40">
            <SelectValue placeholder="الحالة" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ALL">الكل</SelectItem>
            {STATUS_OPTIONS.filter(Boolean).map((s) => (
              <SelectItem key={s} value={s}>
                {STATUS_LABELS[s] || s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Input
          type="datetime-local"
          className="w-56"
          value={dateFrom}
          onChange={(e) => {
            setDateFrom(e.target.value);
            setPage(1);
          }}
        />
        <Input
          type="datetime-local"
          className="w-56"
          value={dateTo}
          onChange={(e) => {
            setDateTo(e.target.value);
            setPage(1);
          }}
        />
        <Button variant="outline" onClick={resetFilters}>
          إعادة ضبط الفلاتر
        </Button>
      </div>

      {/* Table */}
      <div className="rounded border bg-white">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-right">
                <SortableHeader
                  label="#"
                  field="id"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setPage(1);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">
                <SortableHeader
                  label="المستخدم"
                  field="user_name"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setPage(1);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">نوع الوثيقة</TableHead>
              <TableHead className="text-right">
                <SortableHeader
                  label="الحالة"
                  field="status"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setPage(1);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">المرحلة</TableHead>
              <TableHead className="text-right">
                <SortableHeader
                  label="التاريخ"
                  field="created_at"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setPage(1);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">إجراءات</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading
              ? Array.from({ length: 5 }).map((_, i) => (
                  <TableRow key={i}>
                    {Array.from({ length: 7 }).map((_, j) => (
                      <TableCell key={j}>
                        <Skeleton className="h-4 w-full" />
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              : data?.items.map((v) => (
                  <TableRow key={v.id}>
                    <TableCell>{v.id}</TableCell>
                    <TableCell>
                      <div className="text-sm">{v.user_name || `#${v.user_id}`}</div>
                      {v.user_email && <div className="text-xs text-slate-500">{v.user_email}</div>}
                    </TableCell>
                    <TableCell>{v.document_type_name || `#${v.document_type_id}`}</TableCell>
                    <TableCell>
                      <Badge variant={STATUS_VARIANT[v.status] || "outline"}>
                        {STATUS_LABELS[v.status] || v.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs">
                      {v.current_stage ? STAGE_LABELS[v.current_stage] || v.current_stage.replace(/_/g, " ") : "—"}
                    </TableCell>
                    <TableCell className="text-xs">
                      {v.created_at ? new Date(v.created_at).toLocaleDateString("ar-YE") : "—"}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Link href={`/dashboard/verifications/${v.id}`}>
                          <Button size="sm" variant="ghost">
                            عرض
                          </Button>
                        </Link>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => downloadReport(v.id)}
                          disabled={reportingId === v.id}
                        >
                          {reportingId === v.id ? "جاري التنزيل..." : "تنزيل التقرير"}
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <Button size="sm" variant="outline" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
            السابق
          </Button>
          <span className="text-sm">
            {page} / {totalPages}
          </span>
          <Button size="sm" variant="outline" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
            التالي
          </Button>
        </div>
      )}
    </div>
  );
}
