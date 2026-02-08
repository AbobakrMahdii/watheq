"use client";

import { useCallback, useEffect, useState } from "react";

import Link from "next/link";
import { useParams } from "next/navigation";

import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/* ---------- Types ---------- */
type Step = {
  id: number;
  stage: string;
  status: string;
  error_message?: string;
  result_data?: any;
};

type Note = {
  id: number;
  admin_id: number;
  text: string;
  created_at: string;
};

type Verification = {
  id: number;
  user_id: number;
  user_name?: string;
  user_email?: string;
  document_type_id: number;
  document_type_name?: string;
  status: string;
  current_stage?: string;
  error_message?: string;
  result_data?: Record<string, any>;
  created_at?: string;
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

const STATUS_LABELS: Record<string, string> = {
  PENDING: "في الانتظار",
  RUNNING: "قيد التنفيذ",
  SUCCESS: "ناجح",
  FAILED: "فشل",
};

/* ---------- Component ---------- */
export default function VerificationDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [verification, setVerification] = useState<Verification | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [loading, setLoading] = useState(true);
  const [noteText, setNoteText] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [vRes, sRes, nRes] = await Promise.all([
        fetch(`/api/admin/verifications/${id}`),
        fetch(`/api/admin/verifications/${id}/steps`),
        fetch(`/api/admin/verifications/${id}/notes`),
      ]);
      const vData = await vRes.json();
      const sData = await sRes.json();
      const nData = await nRes.json();

      if (!vRes.ok) throw new Error(vData?.message || "فشل تحميل التحقق");
      setVerification(vData);
      setSteps(Array.isArray(sData) ? sData : []);
      setNotes(Array.isArray(nData) ? nData : []);
    } catch (e: any) {
      toast.error(e?.message || "خطأ");
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  async function addNote() {
    if (!noteText.trim()) return;
    setSubmitting(true);
    try {
      const res = await fetch(`/api/admin/verifications/${id}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: noteText }),
      });
      if (!res.ok) throw new Error("فشل إضافة الملاحظة");
      setNoteText("");
      toast.success("تمت إضافة الملاحظة");
      load();
    } catch (e: any) {
      toast.error(e?.message);
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) {
    return (
      <div className="space-y-4" dir="rtl">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  if (!verification) {
    return <div dir="rtl">التحقق غير موجود</div>;
  }

  const rd = verification.result_data || {};
  const face = rd.FACE_MATCHING || {};
  const ai = rd.AI_VERIFICATION || {};
  const ocr = rd.OCR || {};
  const dataV = rd.DATA_VERIFICATION || {};
  const bc = rd.BLOCKCHAIN || {};
  const summary = rd.SUMMARY || {};

  return (
    <div className="max-w-4xl space-y-6" dir="rtl">
      {/* Breadcrumb */}
      <div className="text-sm text-slate-500">
        <Link href="/dashboard/verifications" className="hover:underline">
          التحققات
        </Link>
        {" / "}
        <span>#{verification.id}</span>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">تحقق #{verification.id}</h1>
        <Badge
          variant={
            verification.status === "SUCCESS" ? "default" : verification.status === "FAILED" ? "destructive" : "outline"
          }
        >
          {STATUS_LABELS[verification.status] || verification.status}
        </Badge>
      </div>

      {/* Info */}
      <Card>
        <CardHeader>
          <CardTitle>معلومات أساسية</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2 text-sm sm:grid-cols-2">
          <div>
            <span className="text-slate-500">المستخدم: </span>
            {verification.user_name || `#${verification.user_id}`}
            {verification.user_email && (
              <span className="mr-2 text-xs text-slate-400">({verification.user_email})</span>
            )}
          </div>
          <div>
            <span className="text-slate-500">نوع الوثيقة: </span>
            {verification.document_type_name || `#${verification.document_type_id}`}
          </div>
          <div>
            <span className="text-slate-500">التاريخ: </span>
            {verification.created_at ? new Date(verification.created_at).toLocaleString("ar-YE") : "—"}
          </div>
          <div>
            <span className="text-slate-500">المرحلة الحالية: </span>
            {STAGE_LABELS[verification.current_stage || ""] || verification.current_stage || "—"}
          </div>
          {verification.error_message && (
            <div className="text-red-600 sm:col-span-2">
              <span className="text-slate-500">الخطأ: </span>
              {verification.error_message}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pipeline Steps */}
      <Card>
        <CardHeader>
          <CardTitle>مراحل التحقق</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {steps.map((s) => (
              <div key={s.id} className="flex items-center justify-between rounded border p-2 text-sm">
                <span>{STAGE_LABELS[s.stage] || s.stage}</span>
                <Badge variant={s.status === "SUCCESS" ? "default" : s.status === "FAILED" ? "destructive" : "outline"}>
                  {STATUS_LABELS[s.status] || s.status}
                </Badge>
              </div>
            ))}
            {steps.length === 0 && <p className="text-sm text-slate-500">لا توجد مراحل بعد</p>}
          </div>
        </CardContent>
      </Card>

      {/* Face Matching */}
      {Object.keys(face).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>مطابقة الوجه</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            <div>
              النتيجة:{" "}
              <Badge variant={face.match ? "default" : "destructive"}>{face.match ? "مطابق" : "غير مطابق"}</Badge>
            </div>
            {face.distance != null && <div>المسافة: {Number(face.distance).toFixed(4)}</div>}
            {face.similarity_percent != null && <div>نسبة التشابه: {Number(face.similarity_percent).toFixed(1)}%</div>}
          </CardContent>
        </Card>
      )}

      {/* AI Verification */}
      {Object.keys(ai).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>تحقق الذكاء الاصطناعي</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div>
              القرار:{" "}
              <Badge variant={ai.final_decision === "AUTHENTIC" ? "default" : "destructive"}>
                {ai.final_decision || "—"}
              </Badge>
            </div>
            {ai.authenticity_percent != null && <div>نسبة الأصالة: {Number(ai.authenticity_percent).toFixed(1)}%</div>}
            {ai.elements && typeof ai.elements === "object" && (
              <div className="mt-2 space-y-1">
                <div className="font-medium">تفاصيل العناصر:</div>
                {Object.entries(ai.elements).map(([key, val]: [string, any]) => (
                  <div key={key} className="flex justify-between border-b py-1">
                    <span>{key}</span>
                    <span className="font-mono">
                      {val?.confidence != null ? `${(Number(val.confidence) * 100).toFixed(0)}%` : val?.status || "—"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Data Verification */}
      {Object.keys(dataV).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>مطابقة البيانات</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            {dataV.fraud_suspected && (
              <div className="mb-2 rounded border border-red-300 bg-red-50 p-2 font-semibold text-red-700">
                🚨 محاولة احتيال — بيانات الوثيقة لا تطابق السجل المحفوظ
              </div>
            )}
            {dataV.new_record_created && (
              <div className="mb-2 rounded border border-blue-300 bg-blue-50 p-2 text-blue-700">
                ℹ️ سجل مواطن جديد — تم حفظ البيانات المستخرجة لأول مرة
              </div>
            )}
            <div>
              {dataV.citizen_found
                ? "✅ تم العثور على سجل المواطن"
                : dataV.new_record_created
                  ? "🆕 لم يوجد سجل سابق — تم إنشاء سجل جديد"
                  : "❌ لم يتم العثور على سجل"}
            </div>
            {dataV.message && <div className="text-slate-600">{dataV.message}</div>}
            {dataV.match_details &&
              Object.entries(dataV.match_details).map(([key, val]: [string, any]) => (
                <div key={key} className="flex items-center gap-2">
                  <span>{val?.match ? "✅" : "❌"}</span>
                  <span className="font-medium">{key}</span>
                  {val?.ocr && <span className="text-xs text-slate-400">(OCR: {val.ocr})</span>}
                  {val?.db && <span className="text-xs text-slate-400">(سجل: {val.db})</span>}
                </div>
              ))}
          </CardContent>
        </Card>
      )}

      {/* OCR */}
      {Object.keys(ocr).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>بيانات OCR</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="max-h-60 overflow-auto rounded bg-slate-50 p-3 text-xs whitespace-pre-wrap">
              {ocr.text || JSON.stringify(ocr, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      {/* Blockchain */}
      {Object.keys(bc).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>البلوكتشين</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            {bc.doc_id && <div>DocId: {bc.doc_id}</div>}
            {bc.cid && <div>CID: {bc.cid}</div>}
            {bc.sha256 && <div className="break-all">SHA256: {bc.sha256}</div>}
          </CardContent>
        </Card>
      )}

      {/* Summary */}
      {Object.keys(summary).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>الملخص</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-1 text-sm sm:grid-cols-2">
            <div>مطابقة الوجه: {summary.face_match ? "✅" : "❌"}</div>
            <div>قرار الذكاء الاصطناعي: {summary.ai_final_decision || "—"}</div>
            <div>مطابقة البيانات: {summary.data_match ? "✅" : "❌"}</div>
            <div>OCR: {summary.ocr_done ? "✅" : "❌"}</div>
            <div>CID: {summary.blockchain_cid || "—"}</div>
          </CardContent>
        </Card>
      )}

      {/* Admin Notes */}
      <Card>
        <CardHeader>
          <CardTitle>ملاحظات المشرف</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {notes.length > 0 ? (
            <div className="space-y-2">
              {notes.map((n) => (
                <div key={n.id} className="rounded border p-2 text-sm">
                  <div>{n.text}</div>
                  <div className="mt-1 text-xs text-slate-500">
                    مشرف #{n.admin_id} • {new Date(n.created_at).toLocaleString("ar-YE")}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">لا توجد ملاحظات</p>
          )}
          <div className="flex gap-2">
            <Textarea
              placeholder="أضف ملاحظة..."
              value={noteText}
              onChange={(e) => setNoteText(e.target.value)}
              className="flex-1"
              rows={2}
            />
            <Button onClick={addNote} disabled={submitting || !noteText.trim()}>
              {submitting ? "..." : "إضافة"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
