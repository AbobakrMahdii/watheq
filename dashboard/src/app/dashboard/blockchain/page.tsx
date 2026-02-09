"use client";

import { useEffect, useState, useCallback } from "react";

import { toast } from "sonner";

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

/* ─────────────────────────────── helpers ─────────────────────────────── */

function truncate(s: string | undefined | null, n = 16) {
  if (!s) return "—";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function ts(epoch: number | undefined) {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleString("en-GB");
}

function JsonBlock({ data }: { data: unknown }) {
  return (
    <pre className="max-h-72 overflow-auto rounded border bg-slate-50 p-3 text-xs break-all whitespace-pre-wrap">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function CopyBtn({ text }: { text: string }) {
  return (
    <button
      className="ml-2 text-xs text-blue-600 hover:underline"
      onClick={() => {
        navigator.clipboard.writeText(text);
        toast.success("Copied!");
      }}
    >
      Copy
    </button>
  );
}

/* ─────────────────────────────── types ─────────────────────────────── */

type ChainInfo = Record<string, unknown>;
type DocItem = {
  key?: string;
  txid?: string;
  confirmations?: number;
  blocktime?: number;
  data_json?: Record<string, unknown>;
};

/* ─────────────────────────────── main ─────────────────────────────── */

export default function BlockchainPage() {
  // ─── Status cards state
  const [chainInfo, setChainInfo] = useState<ChainInfo | null>(null);
  const [ipfsHealthy, setIpfsHealthy] = useState<boolean | null>(null);
  const [repoStat, setRepoStat] = useState<Record<string, unknown> | null>(null);
  const [docsCount, setDocsCount] = useState<number | null>(null);
  const [blockCount, setBlockCount] = useState<number | null>(null);

  // ─── Dialog controls
  const [dialog, setDialog] = useState<string | null>(null);
  const [dialogLoading, setDialogLoading] = useState(false);
  const [dialogResult, setDialogResult] = useState<any>(null);

  // ─── Form inputs
  const [inputVal, setInputVal] = useState("");
  const [inputVal2, setInputVal2] = useState("");
  const [inputJson, setInputJson] = useState("{}");
  const [inputFile, setInputFile] = useState<File | null>(null);
  const [searchFilter, setSearchFilter] = useState("");

  const openDialog = (name: string) => {
    setDialog(name);
    setDialogResult(null);
    setInputVal("");
    setInputVal2("");
    setInputJson("{}");
    setInputFile(null);
    setSearchFilter("");
    setDialogLoading(false);
  };

  // ─── Load status cards
  const loadStatus = useCallback(async () => {
    try {
      const [infoRes, ipfsRes, repoRes, docsRes, bcRes] = await Promise.allSettled([
        fetch("/api/blockchain/info").then((r) => r.json()),
        fetch("/api/blockchain/ipfs/health").then((r) => r.json()),
        fetch("/api/blockchain/ipfs/repo-stat").then((r) => r.json()),
        fetch("/api/blockchain/documents").then((r) => r.json()),
        fetch("/api/blockchain/block-count").then((r) => r.json()),
      ]);
      if (infoRes.status === "fulfilled") setChainInfo(infoRes.value);
      if (ipfsRes.status === "fulfilled") setIpfsHealthy(ipfsRes.value?.healthy ?? false);
      if (repoRes.status === "fulfilled") setRepoStat(repoRes.value);
      if (docsRes.status === "fulfilled") setDocsCount(docsRes.value?.items?.length ?? 0);
      if (bcRes.status === "fulfilled") setBlockCount(bcRes.value?.block_count ?? 0);
    } catch {
      /* silent */
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  // ─── Generic fetch helpers
  async function fetchGet(url: string) {
    setDialogLoading(true);
    try {
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.message || "Request failed");
      setDialogResult(data);
    } catch (e: any) {
      toast.error(e?.message || "Error");
    } finally {
      setDialogLoading(false);
    }
  }

  async function fetchPost(url: string, body: unknown) {
    setDialogLoading(true);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.message || "Request failed");
      setDialogResult(data);
      toast.success("Success!");
      loadStatus();
    } catch (e: any) {
      toast.error(e?.message || "Error");
    } finally {
      setDialogLoading(false);
    }
  }

  async function fetchFormData(url: string) {
    if (!inputFile) return toast.error("Select a file first");
    setDialogLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", inputFile);
      const res = await fetch(url, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.message || "Request failed");
      setDialogResult(data);
      toast.success("Success!");
      loadStatus();
    } catch (e: any) {
      toast.error(e?.message || "Error");
    } finally {
      setDialogLoading(false);
    }
  }

  async function fetchDelete(url: string) {
    setDialogLoading(true);
    try {
      const res = await fetch(url, { method: "DELETE" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.message || "Request failed");
      setDialogResult(data);
      toast.success("Unpinned!");
      loadStatus();
    } catch (e: any) {
      toast.error(e?.message || "Error");
    } finally {
      setDialogLoading(false);
    }
  }

  /* ═══════════════════════════ ACTION CARDS CONFIG ═══════════════════════════ */

  const actions: { id: string; label: string; color: string; type: string }[] = [
    // CREATE
    {
      id: "upload",
      label: "رفع وثيقة إلى البلوكتشين",
      color: "bg-green-50 border-green-200 text-green-800",
      type: "CREATE",
    },
    {
      id: "publish",
      label: "نشر بيانات يدوي على Stream",
      color: "bg-green-50 border-green-200 text-green-800",
      type: "CREATE",
    },
    {
      id: "ipfs-pin",
      label: "رفع ملف إلى IPFS فقط",
      color: "bg-green-50 border-green-200 text-green-800",
      type: "CREATE",
    },
    { id: "hash", label: "حساب بصمة SHA-256", color: "bg-green-50 border-green-200 text-green-800", type: "CREATE" },
    // READ — MultiChain
    { id: "all-docs", label: "عرض جميع الوثائق", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    { id: "doc-by-id", label: "جلب وثيقة بمعرّف", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    {
      id: "verify-hash",
      label: "التحقق من بصمة Hash",
      color: "bg-blue-50 border-blue-200 text-blue-800",
      type: "READ",
    },
    { id: "block-explorer", label: "استعراض بلوك", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    { id: "tx-details", label: "تفاصيل معاملة", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    { id: "streams", label: "قائمة Streams", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    { id: "stream-keys", label: "مفاتيح Stream", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    { id: "stream-items", label: "عناصر Stream", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    { id: "stream-publishers", label: "ناشرو Stream", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    { id: "peers", label: "الأقران المتصلون", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    { id: "permissions", label: "الصلاحيات", color: "bg-blue-50 border-blue-200 text-blue-800", type: "READ" },
    // READ — IPFS
    { id: "ipfs-get", label: "جلب ملف بـ CID", color: "bg-purple-50 border-purple-200 text-purple-800", type: "READ" },
    {
      id: "ipfs-pins",
      label: "ملفات IPFS المثبتة",
      color: "bg-purple-50 border-purple-200 text-purple-800",
      type: "READ",
    },
    // DELETE
    {
      id: "ipfs-unpin",
      label: "إلغاء تثبيت CID من IPFS",
      color: "bg-red-50 border-red-200 text-red-800",
      type: "DELETE",
    },
  ];

  /* ═══════════════════════════ RENDER ═══════════════════════════ */

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">عمليات البلوكتشين</h1>
          <p className="text-sm text-slate-500">Blockchain &amp; IPFS Operations — كل العمليات المتاحة</p>
        </div>
        <button onClick={loadStatus} className="rounded bg-slate-900 px-3 py-2 text-sm text-white hover:bg-slate-800">
          Refresh
        </button>
      </div>

      {/* ─── STATUS CARDS ─── */}
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded border bg-white p-4 shadow">
          <p className="mb-1 text-xs text-slate-500">MultiChain</p>
          <p className="text-lg font-bold">
            {chainInfo ? (
              <span className="text-green-700">متصل ✓</span>
            ) : (
              <span className="text-red-600">غير متصل ✗</span>
            )}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {chainInfo ? `${(chainInfo as any).chainname || "—"} • v${(chainInfo as any).version || "?"}` : "—"}
          </p>
        </div>

        <div className="rounded border bg-white p-4 shadow">
          <p className="mb-1 text-xs text-slate-500">IPFS</p>
          <p className="text-lg font-bold">
            {ipfsHealthy === true ? (
              <span className="text-green-700">متصل ✓</span>
            ) : ipfsHealthy === false ? (
              <span className="text-red-600">غير متصل ✗</span>
            ) : (
              <span className="text-slate-400">جاري الفحص...</span>
            )}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {repoStat ? `${((repoStat as any).RepoSize / 1024 / 1024).toFixed(1)} MB repo` : "—"}
          </p>
        </div>

        <div className="rounded border bg-white p-4 shadow">
          <p className="mb-1 text-xs text-slate-500">الوثائق على السلسلة</p>
          <p className="text-lg font-bold">{docsCount ?? "—"}</p>
          <p className="mt-1 text-xs text-slate-500">Documents on-chain</p>
        </div>

        <div className="rounded border bg-white p-4 shadow">
          <p className="mb-1 text-xs text-slate-500">ارتفاع البلوك</p>
          <p className="text-lg font-bold">{blockCount ?? "—"}</p>
          <p className="mt-1 text-xs text-slate-500">Block height</p>
        </div>
      </div>

      {/* ─── ACTION BUTTONS GRID ─── */}
      <h2 className="mb-3 text-sm font-semibold text-slate-600">العمليات المتاحة</h2>
      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {actions.map((a) => (
          <button
            key={a.id}
            onClick={() => {
              openDialog(a.id);
              // Auto-fetch for list operations
              if (a.id === "all-docs") fetchGet("/api/blockchain/documents");
              if (a.id === "streams") fetchGet("/api/blockchain/streams");
              if (a.id === "peers") fetchGet("/api/blockchain/peers");
              if (a.id === "permissions") fetchGet("/api/blockchain/permissions");
              if (a.id === "ipfs-pins") fetchGet("/api/blockchain/ipfs/pins");
            }}
            className={`rounded border p-4 text-right transition hover:shadow-md ${a.color}`}
          >
            <span className="text-xs font-medium opacity-60">{a.type}</span>
            <p className="mt-1 font-semibold">{a.label}</p>
          </button>
        ))}
      </div>

      {/* ═══════════════════════ DIALOGS ═══════════════════════ */}

      {/* ──── UPLOAD DOCUMENT TO BLOCKCHAIN ──── */}
      <Dialog open={dialog === "upload"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>رفع وثيقة إلى البلوكتشين</DialogTitle>
          </DialogHeader>
          <p className="mb-3 text-sm text-slate-500">
            يرفع الملف → يحسب SHA-256 → يثبّته على IPFS → ينشر على MultiChain
          </p>
          <input type="file" onChange={(e) => setInputFile(e.target.files?.[0] || null)} className="mb-3" />
          <button
            disabled={dialogLoading}
            onClick={() => fetchFormData("/api/blockchain/upload")}
            className="rounded bg-green-600 px-4 py-2 text-sm text-white hover:bg-green-700 disabled:opacity-60"
          >
            {dialogLoading ? "جاري الرفع..." : "رفع"}
          </button>
          {dialogResult && (
            <div className="mt-4">
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── MANUAL PUBLISH ──── */}
      <Dialog open={dialog === "publish"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>نشر بيانات يدوي على Stream</DialogTitle>
          </DialogHeader>
          <div className="space-y-2">
            <input
              placeholder="Stream name (default: documents)"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="w-full rounded border px-3 py-2 text-sm"
            />
            <input
              placeholder="Key"
              value={inputVal2}
              onChange={(e) => setInputVal2(e.target.value)}
              className="w-full rounded border px-3 py-2 text-sm"
            />
            <textarea
              placeholder='{"your": "json data"}'
              value={inputJson}
              onChange={(e) => setInputJson(e.target.value)}
              rows={4}
              className="w-full rounded border px-3 py-2 font-mono text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal2}
              onClick={() => {
                let parsed: unknown;
                try {
                  parsed = JSON.parse(inputJson);
                } catch {
                  return toast.error("Invalid JSON");
                }
                fetchPost("/api/blockchain/publish", { stream: inputVal || "documents", key: inputVal2, data: parsed });
              }}
              className="rounded bg-green-600 px-4 py-2 text-sm text-white hover:bg-green-700 disabled:opacity-60"
            >
              {dialogLoading ? "جاري النشر..." : "نشر"}
            </button>
          </div>
          {dialogResult && (
            <div className="mt-4">
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── IPFS PIN FILE ──── */}
      <Dialog open={dialog === "ipfs-pin"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>رفع ملف إلى IPFS فقط</DialogTitle>
          </DialogHeader>
          <p className="mb-3 text-sm text-slate-500">يرفع الملف ويثبّته على IPFS ويرجع CID (بدون بلوكتشين)</p>
          <input type="file" onChange={(e) => setInputFile(e.target.files?.[0] || null)} className="mb-3" />
          <button
            disabled={dialogLoading}
            onClick={() => fetchFormData("/api/blockchain/ipfs/pin")}
            className="rounded bg-green-600 px-4 py-2 text-sm text-white hover:bg-green-700 disabled:opacity-60"
          >
            {dialogLoading ? "جاري الرفع..." : "رفع إلى IPFS"}
          </button>
          {dialogResult && (
            <div className="mt-4">
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── COMPUTE SHA-256 ──── */}
      <Dialog open={dialog === "hash"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>حساب بصمة SHA-256</DialogTitle>
          </DialogHeader>
          <input type="file" onChange={(e) => setInputFile(e.target.files?.[0] || null)} className="mb-3" />
          <button
            disabled={dialogLoading}
            onClick={() => fetchFormData("/api/blockchain/hash")}
            className="rounded bg-green-600 px-4 py-2 text-sm text-white hover:bg-green-700 disabled:opacity-60"
          >
            {dialogLoading ? "جاري الحساب..." : "حساب SHA-256"}
          </button>
          {dialogResult && (
            <div className="mt-4">
              <p className="text-sm font-medium">Hash:</p>
              <div className="flex items-center gap-1">
                <code className="rounded bg-slate-100 px-2 py-1 text-xs break-all">{(dialogResult as any)?.hash}</code>
                {(dialogResult as any)?.hash && <CopyBtn text={(dialogResult as any).hash} />}
              </div>
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── ALL DOCUMENTS (with filter) ──── */}
      <Dialog open={dialog === "all-docs"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-h-[80vh] max-w-3xl overflow-auto">
          <DialogHeader>
            <DialogTitle>جميع الوثائق على البلوكتشين</DialogTitle>
          </DialogHeader>
          <input
            placeholder="بحث / فلتر..."
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            className="mb-3 w-full rounded border px-3 py-2 text-sm"
          />
          {dialogLoading ? (
            <p>جاري التحميل...</p>
          ) : (
            <table className="w-full text-left text-sm">
              <thead>
                <tr>
                  <th className="py-1">Key</th>
                  <th className="py-1">TXID</th>
                  <th className="py-1">Confirmations</th>
                  <th className="py-1">Time</th>
                </tr>
              </thead>
              <tbody>
                {((dialogResult as any)?.items || [])
                  .filter((d: DocItem) => {
                    if (!searchFilter) return true;
                    const s = searchFilter.toLowerCase();
                    return (
                      d.key?.toLowerCase().includes(s) ||
                      d.txid?.toLowerCase().includes(s) ||
                      JSON.stringify(d.data_json || {})
                        .toLowerCase()
                        .includes(s)
                    );
                  })
                  .map((d: DocItem, i: number) => (
                    <tr key={i} className="border-t">
                      <td className="py-1">{truncate(d.key, 24)}</td>
                      <td className="py-1 font-mono text-xs">
                        {truncate(d.txid, 16)}
                        {d.txid && <CopyBtn text={d.txid} />}
                      </td>
                      <td className="py-1">{d.confirmations}</td>
                      <td className="py-1">{ts(d.blocktime)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── DOC BY ID ──── */}
      <Dialog open={dialog === "doc-by-id"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>جلب وثيقة بمعرّف</DialogTitle>
          </DialogHeader>
          <div className="flex gap-2">
            <input
              placeholder="Document ID"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal}
              onClick={() => fetchGet(`/api/blockchain/documents/${encodeURIComponent(inputVal)}`)}
              className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              جلب
            </button>
          </div>
          {dialogResult && (
            <div className="mt-4">
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── VERIFY BY HASH ──── */}
      <Dialog open={dialog === "verify-hash"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>التحقق من بصمة Hash</DialogTitle>
          </DialogHeader>
          <div className="flex gap-2">
            <input
              placeholder="SHA-256 Hash"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 font-mono text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal}
              onClick={() => fetchGet(`/api/blockchain/verify/${encodeURIComponent(inputVal)}`)}
              className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              تحقق
            </button>
          </div>
          {dialogResult && (
            <div className="mt-4">
              {(dialogResult as any)?.verified ? (
                <div className="mb-2 rounded border border-green-200 bg-green-50 p-3">
                  <p className="font-semibold text-green-700">✓ الوثيقة مسجّلة على البلوكتشين</p>
                </div>
              ) : (
                <div className="mb-2 rounded border border-red-200 bg-red-50 p-3">
                  <p className="font-semibold text-red-700">✗ لم يتم العثور على هذه البصمة</p>
                </div>
              )}
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── BLOCK EXPLORER ──── */}
      <Dialog open={dialog === "block-explorer"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>استعراض بلوك</DialogTitle>
          </DialogHeader>
          <div className="flex gap-2">
            <input
              placeholder="Block height (number)"
              type="number"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal}
              onClick={() => fetchGet(`/api/blockchain/blocks/${inputVal}`)}
              className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              استعراض
            </button>
          </div>
          {dialogResult && (
            <div className="mt-4">
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── TX DETAILS ──── */}
      <Dialog open={dialog === "tx-details"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>تفاصيل معاملة</DialogTitle>
          </DialogHeader>
          <div className="flex gap-2">
            <input
              placeholder="Transaction ID (txid)"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 font-mono text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal}
              onClick={() => fetchGet(`/api/blockchain/tx/${encodeURIComponent(inputVal)}`)}
              className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              جلب
            </button>
          </div>
          {dialogResult && (
            <div className="mt-4">
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── LIST STREAMS ──── */}
      <Dialog open={dialog === "streams"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-h-[80vh] max-w-2xl overflow-auto">
          <DialogHeader>
            <DialogTitle>قائمة Streams</DialogTitle>
          </DialogHeader>
          {dialogLoading ? (
            <p>جاري التحميل...</p>
          ) : (
            <table className="w-full text-left text-sm">
              <thead>
                <tr>
                  <th className="py-1">Name</th>
                  <th className="py-1">Items</th>
                  <th className="py-1">Confirmed</th>
                  <th className="py-1">Keys</th>
                </tr>
              </thead>
              <tbody>
                {((dialogResult as any)?.streams || []).map((s: any, i: number) => (
                  <tr key={i} className="border-t">
                    <td className="py-1 font-semibold">{s.name}</td>
                    <td className="py-1">{s.items}</td>
                    <td className="py-1">{s.confirmed}</td>
                    <td className="py-1">{s.keys}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── STREAM KEYS ──── */}
      <Dialog open={dialog === "stream-keys"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-h-[80vh] max-w-2xl overflow-auto">
          <DialogHeader>
            <DialogTitle>مفاتيح Stream</DialogTitle>
          </DialogHeader>
          <div className="mb-3 flex gap-2">
            <input
              placeholder="Stream name (e.g. documents)"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal}
              onClick={() => fetchGet(`/api/blockchain/streams/${encodeURIComponent(inputVal)}/keys`)}
              className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              جلب المفاتيح
            </button>
          </div>
          {dialogResult && <JsonBlock data={dialogResult} />}
        </DialogContent>
      </Dialog>

      {/* ──── STREAM ITEMS ──── */}
      <Dialog open={dialog === "stream-items"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-h-[80vh] max-w-3xl overflow-auto">
          <DialogHeader>
            <DialogTitle>عناصر Stream</DialogTitle>
          </DialogHeader>
          <div className="mb-3 flex gap-2">
            <input
              placeholder="Stream name"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal}
              onClick={() => fetchGet(`/api/blockchain/streams/${encodeURIComponent(inputVal)}/items`)}
              className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              جلب العناصر
            </button>
          </div>
          {dialogResult && <JsonBlock data={dialogResult} />}
        </DialogContent>
      </Dialog>

      {/* ──── STREAM PUBLISHERS ──── */}
      <Dialog open={dialog === "stream-publishers"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-h-[80vh] max-w-2xl overflow-auto">
          <DialogHeader>
            <DialogTitle>ناشرو Stream</DialogTitle>
          </DialogHeader>
          <div className="mb-3 flex gap-2">
            <input
              placeholder="Stream name"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal}
              onClick={() => fetchGet(`/api/blockchain/streams/${encodeURIComponent(inputVal)}/publishers`)}
              className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              جلب الناشرين
            </button>
          </div>
          {dialogResult && <JsonBlock data={dialogResult} />}
        </DialogContent>
      </Dialog>

      {/* ──── CONNECTED PEERS ──── */}
      <Dialog open={dialog === "peers"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-h-[80vh] max-w-2xl overflow-auto">
          <DialogHeader>
            <DialogTitle>الأقران المتصلون</DialogTitle>
          </DialogHeader>
          {dialogLoading ? <p>جاري التحميل...</p> : <JsonBlock data={dialogResult} />}
        </DialogContent>
      </Dialog>

      {/* ──── PERMISSIONS ──── */}
      <Dialog open={dialog === "permissions"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-h-[80vh] max-w-2xl overflow-auto">
          <DialogHeader>
            <DialogTitle>الصلاحيات</DialogTitle>
          </DialogHeader>
          {dialogLoading ? <p>جاري التحميل...</p> : <JsonBlock data={dialogResult} />}
        </DialogContent>
      </Dialog>

      {/* ──── GET FILE BY CID (IPFS) ──── */}
      <Dialog open={dialog === "ipfs-get"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>جلب ملف بـ CID</DialogTitle>
          </DialogHeader>
          <p className="mb-2 text-sm text-slate-500">أدخل CID لتحميل / معاينة الملف من IPFS</p>
          <div className="flex gap-2">
            <input
              placeholder="IPFS CID"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 font-mono text-sm"
            />
            <a
              href={inputVal ? `http://localhost:18080/ipfs/${inputVal}` : "#"}
              target="_blank"
              rel="noopener noreferrer"
              className={`rounded px-4 py-2 text-sm text-white ${inputVal ? "bg-purple-600 hover:bg-purple-700" : "pointer-events-none bg-slate-300"}`}
            >
              فتح
            </a>
          </div>
          {inputVal && (
            <p className="mt-2 text-xs break-all text-slate-500">
              Gateway:{" "}
              <a
                href={`http://localhost:18080/ipfs/${inputVal}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 underline"
              >
                http://localhost:18080/ipfs/{inputVal}
              </a>
            </p>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── IPFS PINS LIST ──── */}
      <Dialog open={dialog === "ipfs-pins"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-h-[80vh] max-w-2xl overflow-auto">
          <DialogHeader>
            <DialogTitle>ملفات IPFS المثبتة</DialogTitle>
          </DialogHeader>
          {dialogLoading ? (
            <p>جاري التحميل...</p>
          ) : (
            <>
              <p className="mb-2 text-sm text-slate-500">Count: {(dialogResult as any)?.count ?? "—"}</p>
              <table className="w-full text-left text-sm">
                <thead>
                  <tr>
                    <th className="py-1">CID</th>
                    <th className="py-1">Type</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries((dialogResult as any)?.pins || {}).map(([cid, info]: [string, any]) => (
                    <tr key={cid} className="border-t">
                      <td className="py-1 font-mono text-xs">
                        {truncate(cid, 32)}
                        <CopyBtn text={cid} />
                      </td>
                      <td className="py-1">{info?.Type || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* ──── IPFS UNPIN ──── */}
      <Dialog open={dialog === "ipfs-unpin"} onOpenChange={() => setDialog(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>إلغاء تثبيت CID من IPFS</DialogTitle>
          </DialogHeader>
          <p className="mb-2 text-sm text-red-500">⚠ هذه العملية لا يمكن التراجع عنها!</p>
          <div className="flex gap-2">
            <input
              placeholder="CID to unpin"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="flex-1 rounded border px-3 py-2 font-mono text-sm"
            />
            <button
              disabled={dialogLoading || !inputVal}
              onClick={() => {
                if (!confirm("Are you sure you want to unpin this CID?")) return;
                fetchDelete(`/api/blockchain/ipfs/pin/${encodeURIComponent(inputVal)}`);
              }}
              className="rounded bg-red-600 px-4 py-2 text-sm text-white hover:bg-red-700 disabled:opacity-60"
            >
              {dialogLoading ? "جاري..." : "إلغاء التثبيت"}
            </button>
          </div>
          {dialogResult && (
            <div className="mt-4">
              <JsonBlock data={dialogResult} />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
