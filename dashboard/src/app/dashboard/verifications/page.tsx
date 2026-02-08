"use client";

import { useCallback, useEffect, useState } from "react";

import Link from "next/link";

import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

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
const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  PENDING: "outline",
  RUNNING: "secondary",
  SUCCESS: "default",
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
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const pageSize = 20;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(page),
        page_size: String(pageSize),
      });
      if (status) params.set("status", status);
      if (search) params.set("search", search);

      const res = await fetch(`/api/admin/verifications?${params}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json?.message || "فشل تحميل البيانات");
      setData(json);
    } catch (e: any) {
      toast.error(e?.message || "فشل تحميل التحققات");
    } finally {
      setLoading(false);
    }
  }, [page, status, search]);

  useEffect(() => {
    load();
  }, [load]);

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0;

  return (
    <div className="space-y-6" dir="rtl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">التحققات</h1>
        <Button size="sm" variant="outline" onClick={load} disabled={loading}>
          {loading ? "جاري التحميل..." : "تحديث"}
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <Input
          placeholder="بحث بالاسم أو البريد..."
          className="w-64"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
        />
        <Select
          value={status}
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
      </div>

      {/* Table */}
      <div className="rounded border bg-white">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-right">#</TableHead>
              <TableHead className="text-right">المستخدم</TableHead>
              <TableHead className="text-right">نوع الوثيقة</TableHead>
              <TableHead className="text-right">الحالة</TableHead>
              <TableHead className="text-right">المرحلة</TableHead>
              <TableHead className="text-right">التاريخ</TableHead>
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
                      <Link href={`/dashboard/verifications/${v.id}`}>
                        <Button size="sm" variant="ghost">
                          عرض
                        </Button>
                      </Link>
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
