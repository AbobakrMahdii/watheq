"use client";

import { useCallback, useEffect, useState } from "react";

import { toast } from "sonner";

import { SortableHeader } from "@/components/sortable-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useSortState } from "@/hooks/use-sort-state";

/* ---------- Types ---------- */
type Citizen = {
  id: number;
  national_id: string;
  full_name_ar?: string;
  full_name_en?: string;
  date_of_birth?: string;
  address?: string;
  issue_date?: string;
  expiry_date?: string;
  gender?: string;
  nationality?: string;
  document_type?: string;
  created_at?: string;
  updated_at?: string;
};

const FIELD_LABELS: Record<string, string> = {
  full_name_ar: "الاسم بالعربي",
  full_name_en: "الاسم بالإنجليزي",
  date_of_birth: "تاريخ الميلاد",
  address: "العنوان",
  issue_date: "تاريخ الإصدار",
  expiry_date: "تاريخ الانتهاء",
  gender: "الجنس",
  nationality: "الجنسية",
  document_type: "نوع الوثيقة",
};

const EDITABLE_FIELDS = Object.keys(FIELD_LABELS);

type DocType = { id: number; name: string };

/* ---------- Component ---------- */
export default function CitizensPage() {
  const [citizens, setCitizens] = useState<Citizen[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [exporting, setExporting] = useState<null | "csv" | "pdf">(null);
  const pageSize = 50;
  const { sortBy, sortOrder, toggleSort } = useSortState("id", "desc");

  /* Edit dialog */
  const [editOpen, setEditOpen] = useState(false);
  const [editCitizen, setEditCitizen] = useState<Citizen | null>(null);
  const [editFields, setEditFields] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [docTypes, setDocTypes] = useState<DocType[]>([]);

  /* Load document types for dropdown */
  useEffect(() => {
    fetch("/api/admin/document-types")
      .then((r) => r.json())
      .then((data) => {
        const list = Array.isArray(data) ? data : data?.document_types || data?.items || [];
        setDocTypes(list);
      })
      .catch(() => null);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(
        `/api/admin/citizens?limit=${pageSize}&offset=${offset}&sort_by=${sortBy}&sort_order=${sortOrder}`,
      );
      const json = await res.json();
      if (!res.ok) throw new Error(json?.message || "فشل تحميل البيانات");
      setCitizens(Array.isArray(json.citizens) ? json.citizens : []);
    } catch (e: any) {
      toast.error(e?.message || "فشل تحميل سجلات المواطنين");
    } finally {
      setLoading(false);
    }
  }, [offset, sortBy, sortOrder]);

  useEffect(() => {
    load();
  }, [load]);

  function getExportFilename(headerValue: string | null) {
    if (!headerValue) return null;
    const match = headerValue.match(/filename="?([^";]+)"?/i);
    return match?.[1] || null;
  }

  async function downloadExport(format: "csv" | "pdf") {
    setExporting(format);
    try {
      const params = new URLSearchParams();
      params.set("format", format);
      const res = await fetch(`/api/admin/citizens/export?${params}`, { cache: "no-store" });
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        throw new Error(json?.message || "Failed to export citizens");
      }
      const blob = await res.blob();
      const filename =
        getExportFilename(res.headers.get("content-disposition")) ||
        `citizens_export_${new Date().toISOString().replace(/[:.]/g, "-")}.${format}`;
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

  /* Filter locally by search */
  const filtered = search.trim()
    ? citizens.filter(
        (c) =>
          c.national_id?.includes(search) ||
          c.full_name_ar?.includes(search) ||
          c.full_name_en?.toLowerCase().includes(search.toLowerCase()),
      )
    : citizens;

  /* Open edit dialog */
  function openEdit(c: Citizen) {
    setEditCitizen(c);
    const fields: Record<string, string> = {};
    for (const key of EDITABLE_FIELDS) {
      fields[key] = (c as any)[key] || "";
    }
    setEditFields(fields);
    setEditOpen(true);
  }

  /* Save edit */
  async function submitEdit() {
    if (!editCitizen) return;
    const body: Record<string, string> = {};
    for (const key of EDITABLE_FIELDS) {
      const newVal = editFields[key]?.trim() || "";
      const oldVal = ((editCitizen as any)[key] || "").toString().trim();
      if (newVal !== oldVal) body[key] = newVal;
    }
    if (Object.keys(body).length === 0) return toast.info("لا يوجد تغييرات");

    setSaving(true);
    try {
      const res = await fetch(`/api/admin/citizens/${editCitizen.national_id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "فشل التحديث");
      toast.success("تم تحديث سجل المواطن بنجاح");
      setEditOpen(false);
      await load();
    } catch (e: any) {
      toast.error(e?.message || "فشل التحديث");
    } finally {
      setSaving(false);
    }
  }

  /* Delete */
  async function deleteCitizen(nationalId: string) {
    if (!confirm(`هل أنت متأكد من حذف سجل المواطن ${nationalId}؟`)) return;
    try {
      const res = await fetch(`/api/admin/citizens/${nationalId}`, { method: "DELETE" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "فشل الحذف");
      toast.success("تم حذف سجل المواطن");
      await load();
    } catch (e: any) {
      toast.error(e?.message || "فشل الحذف");
    }
  }

  return (
    <div className="space-y-6" dir="rtl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">سجلات المواطنين</h1>
          <p className="text-sm text-slate-500">البيانات المستخرجة من OCR أثناء التحقق</p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => downloadExport("csv")} disabled={exporting === "csv"}>
            {exporting === "csv" ? "Exporting..." : "Export CSV"}
          </Button>
          <Button size="sm" variant="outline" onClick={() => downloadExport("pdf")} disabled={exporting === "pdf"}>
            {exporting === "pdf" ? "Exporting..." : "Export PDF"}
          </Button>
          <Button size="sm" variant="outline" onClick={load} disabled={loading}>
            {loading ? "جاري التحميل..." : "تحديث"}
          </Button>
        </div>
      </div>

      {/* Search */}
      <Input
        placeholder="بحث بالرقم الوطني أو الاسم..."
        className="w-80"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      {/* Table */}
      <div className="rounded border bg-white">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-right">
                <SortableHeader
                  label="الرقم الوطني"
                  field="national_id"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setOffset(0);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">
                <SortableHeader
                  label="الاسم بالعربي"
                  field="full_name_ar"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setOffset(0);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">
                <SortableHeader
                  label="الاسم بالإنجليزي"
                  field="full_name_en"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setOffset(0);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">
                <SortableHeader
                  label="تاريخ الميلاد"
                  field="date_of_birth"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setOffset(0);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">
                <SortableHeader
                  label="الجنس"
                  field="gender"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setOffset(0);
                  }}
                />
              </TableHead>
              <TableHead className="text-right">نوع الوثيقة</TableHead>
              <TableHead className="text-right">
                <SortableHeader
                  label="تاريخ الإنشاء"
                  field="created_at"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={(f) => {
                    toggleSort(f);
                    setOffset(0);
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
                    {Array.from({ length: 8 }).map((_, j) => (
                      <TableCell key={j}>
                        <Skeleton className="h-4 w-full" />
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              : filtered.map((c) => (
                  <TableRow key={c.national_id}>
                    <TableCell className="font-mono text-sm">{c.national_id}</TableCell>
                    <TableCell>{c.full_name_ar || "—"}</TableCell>
                    <TableCell>{c.full_name_en || "—"}</TableCell>
                    <TableCell className="text-sm">{c.date_of_birth || "—"}</TableCell>
                    <TableCell className="text-sm">{c.gender || "—"}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{c.document_type || "—"}</Badge>
                    </TableCell>
                    <TableCell className="text-xs">{c.created_at || "—"}</TableCell>
                    <TableCell>
                      <div className="flex gap-2">
                        <Button size="sm" variant="ghost" onClick={() => openEdit(c)}>
                          تعديل
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="text-red-600"
                          onClick={() => deleteCitizen(c.national_id)}
                        >
                          حذف
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}

            {!loading && filtered.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} className="py-8 text-center text-slate-500">
                  لا توجد سجلات
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-center gap-3">
        <Button
          size="sm"
          variant="outline"
          disabled={offset === 0}
          onClick={() => setOffset((o) => Math.max(0, o - pageSize))}
        >
          السابق
        </Button>
        <span className="text-sm text-slate-600">يبدأ من {offset + 1}</span>
        <Button
          size="sm"
          variant="outline"
          disabled={citizens.length < pageSize}
          onClick={() => setOffset((o) => o + pageSize)}
        >
          التالي
        </Button>
      </div>

      {/* Edit Dialog */}
      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent className="max-w-lg" dir="rtl">
          <DialogHeader>
            <DialogTitle>تعديل سجل المواطن — {editCitizen?.national_id}</DialogTitle>
          </DialogHeader>
          <div className="max-h-[60vh] space-y-3 overflow-y-auto pt-2">
            {EDITABLE_FIELDS.map((key) => (
              <div key={key}>
                <label className="mb-1 block text-sm font-medium text-slate-700">{FIELD_LABELS[key]}</label>
                {key === "document_type" ? (
                  <Select
                    value={editFields[key] || ""}
                    onValueChange={(v) => setEditFields((prev) => ({ ...prev, [key]: v }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="اختر نوع الوثيقة" />
                    </SelectTrigger>
                    <SelectContent>
                      {docTypes.map((dt) => (
                        <SelectItem key={dt.id} value={dt.name}>
                          {dt.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : key === "gender" ? (
                  <Select
                    value={editFields[key] || ""}
                    onValueChange={(v) => setEditFields((prev) => ({ ...prev, [key]: v }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="اختر الجنس" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ذكر">ذكر</SelectItem>
                      <SelectItem value="أنثى">أنثى</SelectItem>
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    value={editFields[key] || ""}
                    onChange={(e) => setEditFields((prev) => ({ ...prev, [key]: e.target.value }))}
                  />
                )}
              </div>
            ))}
          </div>
          <div className="flex gap-2 pt-3">
            <Button onClick={submitEdit} disabled={saving}>
              {saving ? "جاري الحفظ..." : "حفظ التعديلات"}
            </Button>
            <Button variant="outline" onClick={() => setEditOpen(false)}>
              إلغاء
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
