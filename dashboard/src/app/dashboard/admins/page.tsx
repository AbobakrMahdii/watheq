"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { SortableHeader } from "@/components/sortable-header";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useSortState } from "@/hooks/use-sort-state";
import { sortLocally } from "@/lib/sort-utils";

type User = {
  _id: string;
  name: string;
  username?: string | null;
  email: string;
  role: string;
  is_first_super_admin?: boolean;
};

export default function AdminsPage() {
  const [me, setMe] = useState<{ id?: string | number; role?: string; email?: string } | null>(null);
  const [admins, setAdmins] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [bulkUploading, setBulkUploading] = useState(false);
  const [templateDownloading, setTemplateDownloading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [formName, setFormName] = useState("");
  const [formUsername, setFormUsername] = useState("");
  const [formEmail, setFormEmail] = useState("");
  const [formPassword, setFormPassword] = useState("");

  // Edit dialog state
  const [editOpen, setEditOpen] = useState(false);
  const [editingAdmin, setEditingAdmin] = useState<User | null>(null);
  const [editName, setEditName] = useState("");
  const [editUsername, setEditUsername] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [editPassword, setEditPassword] = useState("");

  const { sortBy, sortOrder, toggleSort } = useSortState("name", "asc");

  function resetForm() {
    setFormName("");
    setFormUsername("");
    setFormEmail("");
    setFormPassword("");
  }

  function getErrorMessage(data: any, fallback: string) {
    return data?.message || data?.detail?.message || data?.detail || fallback;
  }

  async function loadAdmins() {
    setLoading(true);
    try {
      const meRes = await fetch("/api/auth/me");
      const meData = await meRes.json().catch(() => ({}));
      if (!meRes.ok) throw new Error(meData?.message || "Unauthorized");
      setMe(meData);

      if (meData?.role !== "super_admin" && meData?.role !== "admin_manager") {
        toast.error("Super admin or Admin Manager only");
        return;
      }

      const adminsRes = await fetch("/api/admin/admins");
      const adminsData = await adminsRes.json().catch(() => []);
      if (!adminsRes.ok) throw new Error(getErrorMessage(adminsData, "Failed to load admins"));
      setAdmins(Array.isArray(adminsData) ? adminsData : []);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load admins");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAdmins();
  }, []);

  async function createAdmin() {
    if (me?.role !== "super_admin" && me?.role !== "admin_manager")
      return toast.error("Super admin or Admin Manager only");
    if (!formName.trim() || !formEmail.trim() || !formPassword) return toast.error("All fields are required");
    try {
      const res = await fetch("/api/admin/admins/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formName.trim(),
          username: formUsername.trim() || null,
          email: formEmail.trim(),
          password: formPassword,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(getErrorMessage(data, "Failed to create admin"));
      toast.success("Admin created");
      resetForm();
      await loadAdmins();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  async function downloadTemplate() {
    setTemplateDownloading(true);
    try {
      const res = await fetch("/api/admin/admins/template", { cache: "no-store" });
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        throw new Error(getErrorMessage(json, "Failed to download template"));
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "admin_data_template.xlsx";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      toast.success("Template downloaded");
    } catch (e: any) {
      toast.error(e?.message || "Failed to download template");
    } finally {
      setTemplateDownloading(false);
    }
  }

  async function uploadAdmins(file: File) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".xlsx")) {
      toast.error("Please upload an .xlsx file");
      return;
    }
    setBulkUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/admin/admins/import", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(getErrorMessage(data, "Bulk import failed"));
      const created = data?.created ?? 0;
      const skipped = data?.skipped ?? 0;
      const errorCount = data?.error_count ?? 0;
      toast.success(`Imported: ${created}, Skipped: ${skipped}, Errors: ${errorCount}`);
      await loadAdmins();
    } catch (e: any) {
      toast.error(e?.message || "Bulk import failed");
    } finally {
      setBulkUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function demoteToUser(id: string, role: string) {
    if (me?.role !== "super_admin" && me?.role !== "admin_manager")
      return toast.error("Super admin or Admin Manager only");
    try {
      const res = await fetch(`/api/admin/users/${id}/remove-admin`, { method: "PUT" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(getErrorMessage(data, "Failed to remove admin"));
      toast.success("Admin removed");
      await loadAdmins();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  function openEdit(u: User) {
    setEditingAdmin(u);
    setEditName(u.name || "");
    setEditUsername(u.username || "");
    setEditEmail(u.email || "");
    setEditPassword("");
    setEditOpen(true);
  }

  async function submitEdit() {
    if (!editingAdmin) return;
    const body: Record<string, string> = {};
    if (editName !== (editingAdmin.name || "")) body.name = editName;
    if (editUsername !== (editingAdmin.username || "")) body.username = editUsername;
    if (editEmail !== (editingAdmin.email || "")) body.email = editEmail;
    if (editPassword) body.password = editPassword;

    if (Object.keys(body).length === 0) {
      toast.info("No changes to save");
      return;
    }

    try {
      const res = await fetch(`/api/admin/users/${editingAdmin._id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(getErrorMessage(data, "Failed to update admin"));
      toast.success("Admin updated");
      setEditOpen(false);
      await loadAdmins();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  // Determine if the current user is the first (primary) super admin
  const meId = me?.id != null ? String(me.id) : "";
  const isMeFirstSuperAdmin = admins.some((a) => a._id === meId && a.is_first_super_admin);

  if (me && me.role !== "super_admin" && me.role !== "admin_manager") {
    return (
      <div className="rounded border bg-white p-6 text-sm text-slate-600">
        This page is available for Super Admins or Admin Managers only.
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Admins</h1>
          <p className="text-sm text-slate-500">
            Logged in as {me?.email || "—"} ({me?.role || "—"})
          </p>
        </div>
        <button
          onClick={loadAdmins}
          className="rounded bg-slate-900 px-3 py-2 text-sm text-white hover:bg-slate-800 disabled:opacity-60"
          disabled={loading}
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      <div className="mb-4 rounded bg-white p-4 shadow">
        <h2 className="mb-3 font-semibold">Create Admin</h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <input
            placeholder="Name"
            value={formName}
            onChange={(e) => setFormName(e.target.value)}
            className="rounded border px-3 py-2"
          />
          <input
            placeholder="Username (optional)"
            value={formUsername}
            onChange={(e) => setFormUsername(e.target.value)}
            className="rounded border px-3 py-2"
          />
          <input
            placeholder="Email"
            value={formEmail}
            onChange={(e) => setFormEmail(e.target.value)}
            className="rounded border px-3 py-2"
          />
          <input
            placeholder="Password"
            type="password"
            value={formPassword}
            onChange={(e) => setFormPassword(e.target.value)}
            className="rounded border px-3 py-2"
          />
        </div>
        <div className="mt-3 flex gap-2">
          <button
            onClick={createAdmin}
            disabled={loading}
            className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
          >
            Add Admin
          </button>
          <button onClick={resetForm} className="rounded border px-3 py-2 text-sm hover:bg-slate-50">
            Clear
          </button>
        </div>
      </div>

      {me?.role === "super_admin" && (
        <div className="mb-4 rounded bg-white p-4 shadow">
          <h2 className="mb-2 font-semibold">Bulk Import Admins (Excel)</h2>
          <p className="mb-3 text-sm text-slate-500">
            Upload an .xlsx file with columns: name, username (optional), email, password, role (admin or super_admin).
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={downloadTemplate}
              disabled={templateDownloading}
              className="rounded border px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-60"
            >
              {templateDownloading ? "Downloading..." : "Download Template"}
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={bulkUploading}
              className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              {bulkUploading ? "Uploading..." : "Upload Excel"}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".xlsx"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) uploadAdmins(file);
              }}
            />
          </div>
        </div>
      )}

      <div className="rounded bg-white p-4 shadow">
        <table className="w-full text-left">
          <thead>
            <tr>
              <th className="py-1">ID</th>
              <th className="py-1">
                <SortableHeader
                  label="Name"
                  field="name"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={toggleSort}
                />
              </th>
              <th className="py-1">
                <SortableHeader
                  label="Username"
                  field="username"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={toggleSort}
                />
              </th>
              <th className="py-1">
                <SortableHeader
                  label="Email"
                  field="email"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={toggleSort}
                />
              </th>
              <th className="py-1">
                <SortableHeader
                  label="Role"
                  field="role"
                  currentSortBy={sortBy}
                  currentSortOrder={sortOrder}
                  onSort={toggleSort}
                />
              </th>
              <th className="py-1">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortLocally(admins, sortBy, sortOrder).map((u) => {
              const isSelf = u._id === meId;
              const isTargetSuperAdmin = u.role === "super_admin";
              const isTargetFirstSA = u.is_first_super_admin;

              // Edit: first SA can edit anyone (except self row shows no action needed),
              // other super admins can edit normal admins only
              const canEdit = isMeFirstSuperAdmin
                ? !isSelf // first SA can edit all others
                : !isTargetSuperAdmin; // non-first SA can only edit normal admins

              // Remove: first SA can remove anyone except themselves and the first SA entry,
              // other super admins can only remove normal admins
              let canRemove = false;
              if (isSelf) {
                canRemove = false; // never remove yourself
              } else if (isTargetSuperAdmin) {
                // only the first SA can remove other super admins (not the first SA itself)
                canRemove = isMeFirstSuperAdmin && !isTargetFirstSA;
              } else {
                canRemove = true; // all super admins can remove normal admins
              }

              return (
                <tr key={u._id} className="border-t">
                  <td className="py-2">{u._id}</td>
                  <td className="py-2">{u.name}</td>
                  <td className="py-2">{u.username || "—"}</td>
                  <td className="py-2">{u.email}</td>
                  <td className="py-2">
                    {u.role}
                    {u.is_first_super_admin && <span className="ml-1 text-xs text-amber-600">(primary)</span>}
                  </td>
                  <td className="space-x-2 py-2">
                    {canEdit && (
                      <button
                        className="text-sm text-blue-700 hover:underline disabled:opacity-60"
                        disabled={loading}
                        onClick={() => openEdit(u)}
                      >
                        Edit
                      </button>
                    )}
                    {canRemove && (
                      <button
                        className="text-sm text-red-700 hover:underline disabled:opacity-60"
                        disabled={loading}
                        onClick={() => demoteToUser(u._id, u.role)}
                      >
                        Remove admin
                      </button>
                    )}
                    {isSelf && <span className="text-xs text-slate-400 italic">You</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Edit Admin Dialog */}
      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Admin</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 pt-2">
            <div>
              <label className="text-sm text-slate-600">Name</label>
              <input
                className="w-full rounded border px-3 py-2"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Name"
              />
            </div>
            <div>
              <label className="text-sm text-slate-600">Username</label>
              <input
                className="w-full rounded border px-3 py-2"
                value={editUsername}
                onChange={(e) => setEditUsername(e.target.value)}
                placeholder="Username"
              />
            </div>
            <div>
              <label className="text-sm text-slate-600">Email</label>
              <input
                className="w-full rounded border px-3 py-2"
                value={editEmail}
                onChange={(e) => setEditEmail(e.target.value)}
                placeholder="Email"
              />
            </div>
            <div>
              <label className="text-sm text-slate-600">New Password (leave empty to keep)</label>
              <input
                className="w-full rounded border px-3 py-2"
                type="password"
                value={editPassword}
                onChange={(e) => setEditPassword(e.target.value)}
                placeholder="••••••••"
              />
            </div>
            <div className="flex gap-2 pt-2">
              <button
                onClick={submitEdit}
                className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700"
              >
                Save Changes
              </button>
              <button onClick={() => setEditOpen(false)} className="rounded border px-4 py-2 text-sm hover:bg-slate-50">
                Cancel
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
