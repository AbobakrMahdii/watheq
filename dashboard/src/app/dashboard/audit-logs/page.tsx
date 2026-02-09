"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";

const PAGE_SIZE = 20;

type AuditLog = {
  id: number;
  operation_id: string;
  operation_type: string;
  status: string;
  failure_reason?: string | null;
  user_id?: number | null;
  user_name?: string | null;
  user_email?: string | null;
  user_role?: string | null;
  ip_address?: string | null;
  user_agent?: string | null;
  service?: string | null;
  module?: string | null;
  path?: string | null;
  method?: string | null;
  file_name?: string | null;
  file_ext?: string | null;
  file_size?: number | null;
  file_cid?: string | null;
  file_url?: string | null;
  created_at: string;
};

type AuditLogResponse = {
  total: number;
  page: number;
  page_size: number;
  items: AuditLog[];
};

export default function AuditLogsPage() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AuditLogResponse | null>(null);

  const [page, setPage] = useState(1);
  const [query, setQuery] = useState("");
  const [operationType, setOperationType] = useState("");
  const [status, setStatus] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [userName, setUserName] = useState("");
  const [exporting, setExporting] = useState<null | "pdf" | "xlsx">(null);

  const totalPages = useMemo(() => {
    if (!data) return 1;
    return Math.max(1, Math.ceil(data.total / data.page_size));
  }, [data]);

  async function loadAuditLogs() {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.set("page", String(page));
      params.set("page_size", String(PAGE_SIZE));
      if (query.trim()) params.set("query", query.trim());
      if (operationType) params.set("operation_type", operationType);
      if (status) params.set("status", status);
      if (dateFrom) params.set("date_from", dateFrom);
      if (dateTo) params.set("date_to", dateTo);
      if (userName.trim()) params.set("user_name", userName.trim());

      const res = await fetch(`/api/admin/audit-logs?${params.toString()}`);
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload?.message || "Failed to load audit logs");
      setData(payload as AuditLogResponse);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load audit logs");
    } finally {
      setLoading(false);
    }
  }

  function getExportFilename(headerValue: string | null) {
    if (!headerValue) return null;
    const match = headerValue.match(/filename="?([^";]+)"?/i);
    return match?.[1] || null;
  }

  async function downloadExport(format: "pdf" | "xlsx") {
    setExporting(format);
    try {
      const params = new URLSearchParams();
      if (userName) params.set("user_name", userName);
      if (operationType) params.set("operation_type", operationType);
      if (status) params.set("status", status);
      if (dateFrom) params.set("date_from", dateFrom);
      if (dateTo) params.set("date_to", dateTo);
      if (query) params.set("query", query);
      params.set("format", format);

      const res = await fetch(`/api/admin/audit-logs/export?${params}`);
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        throw new Error(json?.message || "Failed to export audit logs");
      }
      const blob = await res.blob();
      const filename =
        getExportFilename(res.headers.get("content-disposition")) ||
        `audit_logs_${new Date().toISOString().replace(/[:.]/g, "-")}.${format}`;
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
      toast.error(e?.message || "فشل تنزيل التقرير");
    } finally {
      setExporting(null);
    }
  }

  useEffect(() => {
    loadAuditLogs();
  }, [page]);

  function applyFilters(e: React.FormEvent) {
    e.preventDefault();
    setPage(1);
    loadAuditLogs();
  }


  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <header className="bg-white shadow p-4 mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Audit Logs</h1>
          <p className="text-sm text-slate-500">Read-only activity tracking for all system operations.</p>
        </div>
        <Link href="/dashboard" className="text-sm text-slate-600 hover:text-slate-900">
          Back to Dashboard
        </Link>
      </header>

      <main className="flex-1 p-6 space-y-6">
        <form onSubmit={applyFilters} className="bg-white p-4 rounded shadow grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700">Search</label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="mt-1 block w-full border border-slate-300 rounded-md shadow-sm p-2"
              placeholder="name, email, module..."
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">User name</label>
            <input
              type="text"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              className="mt-1 block w-full border border-slate-300 rounded-md shadow-sm p-2"
              placeholder="Exact or partial name"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Operation type</label>
            <select
              value={operationType}
              onChange={(e) => setOperationType(e.target.value)}
              className="mt-1 block w-full border border-slate-300 rounded-md shadow-sm p-2"
            >
              <option value="">All</option>
              <option value="Login">Login</option>
              <option value="Logout">Logout</option>
              <option value="Create">Create</option>
              <option value="Update">Update</option>
              <option value="Delete">Delete</option>
              <option value="Upload">Upload</option>
              <option value="Verify">Verify</option>
              <option value="OCR">OCR</option>
              <option value="Read">Read</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Status</label>
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              className="mt-1 block w-full border border-slate-300 rounded-md shadow-sm p-2"
            >
              <option value="">All</option>
              <option value="success">Success</option>
              <option value="failed">Failed</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Date from</label>
            <input
              type="datetime-local"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="mt-1 block w-full border border-slate-300 rounded-md shadow-sm p-2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Date to</label>
            <input
              type="datetime-local"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="mt-1 block w-full border border-slate-300 rounded-md shadow-sm p-2"
            />
          </div>
          <div className="md:col-span-3 flex justify-end gap-2">
            <button
              type="button"
              onClick={() => downloadExport("pdf")}
              disabled={exporting === "pdf"}
              className="bg-slate-600 text-white px-4 py-2 rounded-md hover:bg-slate-700 disabled:opacity-60"
            >
              {exporting === "pdf" ? "جارٍ التصدير..." : "تصدير PDF"}
            </button>
            <button
              type="button"
              onClick={() => downloadExport("xlsx")}
              disabled={exporting === "xlsx"}
              className="bg-slate-700 text-white px-4 py-2 rounded-md hover:bg-slate-800 disabled:opacity-60"
            >
              {exporting === "xlsx" ? "جارٍ التصدير..." : "تصدير Excel"}
            </button>
            <button
              type="submit"
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
              disabled={loading}
            >
              Apply Filters
            </button>
          </div>
        </form>

        <div className="bg-white p-4 rounded shadow">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-semibold">Logs</h2>
            <span className="text-sm text-slate-500">{data?.total ?? 0} entries</span>
          </div>

          {!data || data.items.length === 0 ? (
            <p className="text-slate-500">No audit logs found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">Time</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">User</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">Role</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">Operation</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">Status</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">Module</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">Path</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">File</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase">Failure Reason</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-slate-200">
                  {data.items.map((log) => (
                    <tr key={log.id}>
                      <td className="px-4 py-2 text-sm text-slate-600">
                        {new Date(log.created_at).toLocaleString()}
                      </td>
                      <td className="px-4 py-2 text-sm text-slate-700">
                        {log.user_name || log.user_email || "-"}
                      </td>
                      <td className="px-4 py-2 text-sm text-slate-600">{log.user_role || "-"}</td>
                      <td className="px-4 py-2 text-sm text-slate-700">{log.operation_type}</td>
                      <td className="px-4 py-2 text-sm">
                        <span
                          className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            log.status === "success" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
                          }`}
                        >
                          {log.status}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-sm text-slate-600">{log.module || "-"}</td>
                      <td className="px-4 py-2 text-xs text-slate-500">{log.method} {log.path}</td>
                      <td className="px-4 py-2 text-xs text-slate-500">
                        {log.file_name ? (
                          <div className="flex items-center gap-2">
                            <span title={log.file_name}>
                              {log.file_name}
                              {log.file_size ? ` (${(log.file_size / 1024).toFixed(1)} KB)` : ""}
                            </span>
                            {log.file_cid ? (
                              <a
                                href={`/api/admin/ipfs/file/${encodeURIComponent(log.file_cid)}`}
                                download={log.file_name || undefined}
                                className="text-blue-600 hover:text-blue-800"
                              >
                                Download
                              </a>
                            ) : null}
                          </div>
                        ) : (
                          "-"
                        )}
                      </td>
                      <td className="px-4 py-2 text-xs text-slate-500 max-w-sm truncate" title={log.failure_reason || ""}>
                        {log.failure_reason || "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="flex items-center justify-between mt-4">
            <button
              className="px-3 py-1 border rounded-md text-sm disabled:opacity-50"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1 || loading}
            >
              Previous
            </button>
            <span className="text-sm text-slate-600">
              Page {page} of {totalPages}
            </span>
            <button
              className="px-3 py-1 border rounded-md text-sm disabled:opacity-50"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages || loading}
            >
              Next
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
