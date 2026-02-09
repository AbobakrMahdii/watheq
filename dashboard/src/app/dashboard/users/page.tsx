"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { SortableHeader } from "@/components/sortable-header";
import { useSortState } from "@/hooks/use-sort-state";
import { sortLocally } from "@/lib/sort-utils";

type User = {
  _id: string;
  name: string;
  username?: string | null;
  email: string;
  role: string;
  is_active?: boolean;
  deleted_at?: string | null;
};

export default function UsersPage() {
  const [me, setMe] = useState<{ role?: string; email?: string } | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [bulkUploading, setBulkUploading] = useState(false);
  const [templateDownloading, setTemplateDownloading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const canManageRoles = me?.role === "super_admin";
  const canCreateUsers = me?.role === "admin" || me?.role === "super_admin";
  const canModerateUsers = me?.role === "admin" || me?.role === "super_admin";

  const [formName, setFormName] = useState("");
  const [formUsername, setFormUsername] = useState("");
  const [formEmail, setFormEmail] = useState("");
  const [formPassword, setFormPassword] = useState("");

  // Edit dialog state
  const [editOpen, setEditOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
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

  function openEdit(u: User) {
    setEditingUser(u);
    setEditName(u.name || "");
    setEditUsername(u.username || "");
    setEditEmail(u.email || "");
    setEditPassword("");
    setEditOpen(true);
  }

  async function submitEdit() {
    if (!editingUser) return;
    const body: Record<string, string> = {};
    if (editName.trim() && editName !== editingUser.name) body.name = editName.trim();
    if (editUsername.trim() !== (editingUser.username || "")) body.username = editUsername.trim();
    if (editEmail.trim() && editEmail !== editingUser.email) body.email = editEmail.trim();
    if (editPassword) body.password = editPassword;
    if (Object.keys(body).length === 0) return toast.error("No changes to save");

    try {
      const res = await fetch(`/api/admin/users/${editingUser._id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("User updated successfully");
      setEditOpen(false);
      setEditingUser(null);
      await loadUsers();
    } catch (e: any) {
      toast.error(e?.message || "Failed to update user");
    }
  }

  async function loadUsers() {
    setLoading(true);
    try {
      const meRes = await fetch("/api/auth/me");
      const meData = await meRes.json().catch(() => ({}));
      if (!meRes.ok) throw new Error(meData?.message || "Unauthorized");
      setMe(meData);

      const usersRes = await fetch("/api/admin/users");
      const usersData = await usersRes.json().catch(() => []);
      if (!usersRes.ok) throw new Error(usersData?.message || "Failed to load users");
      setUsers(Array.isArray(usersData) ? usersData : []);
    } catch (e: any) {
      toast.error(e?.message || "Failed to load users");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadUsers();
  }, []);

  async function createUser() {
    if (!canCreateUsers) return toast.error("Admins only");
    if (!formName.trim() || !formEmail.trim() || !formPassword) return toast.error("All fields are required");
    try {
      const res = await fetch("/api/admin/users/create", {
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
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("User created");
      resetForm();
      await loadUsers();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  async function downloadTemplate() {
    setTemplateDownloading(true);
    try {
      const res = await fetch("/api/admin/users/template", { cache: "no-store" });
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        throw new Error(json?.message || "Failed to download template");
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "user_data_template.xlsx";
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

  async function uploadUsers(file: File) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".xlsx")) {
      toast.error("Please upload an .xlsx file");
      return;
    }
    setBulkUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/admin/users/import", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Bulk import failed");

      const created = data?.created ?? 0;
      const skipped = data?.skipped ?? 0;
      const errorCount = data?.error_count ?? 0;
      toast.success(`Imported: ${created}, Skipped: ${skipped}, Errors: ${errorCount}`);
      await loadUsers();
    } catch (e: any) {
      toast.error(e?.message || "Bulk import failed");
    } finally {
      setBulkUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function promoteToAdmin(id: string) {
    if (!canManageRoles) return toast.error("Super admin only");
    try {
      const res = await fetch(`/api/admin/users/${id}/make-admin`, { method: "PUT" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("User promoted to admin");
      await loadUsers();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  async function toggleSuspend(id: string, currentlyActive?: boolean) {
    if (!canModerateUsers) return toast.error("Admins only");
    try {
      const res = await fetch(`/api/admin/users/${id}/${currentlyActive ? "suspend" : "activate"}`, {
        method: "PUT",
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success(currentlyActive ? "User suspended" : "User activated");
      await loadUsers();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  async function softDelete(id: string) {
    if (!canModerateUsers) return toast.error("Admins only");
    if (!confirm("Soft delete this user? They will be deactivated but kept in the database.")) return;
    try {
      const res = await fetch(`/api/admin/users/${id}`, { method: "DELETE" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("User soft-deleted");
      await loadUsers();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Users</h1>
          <p className="text-sm text-slate-500">
            Logged in as {me?.email || "-"} ({me?.role || "-"})
          </p>
        </div>
        <button
          onClick={loadUsers}
          className="rounded bg-slate-900 px-3 py-2 text-sm text-white hover:bg-slate-800 disabled:opacity-60"
          disabled={loading}
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {canCreateUsers && (
        <div className="mb-4 rounded bg-white p-4 shadow">
          <h2 className="mb-3 font-semibold">Create User</h2>
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
              onClick={createUser}
              disabled={loading}
              className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
            >
              Add User
            </button>
            <button onClick={resetForm} className="rounded border px-3 py-2 text-sm hover:bg-slate-50">
              Clear
            </button>
          </div>
        </div>
      )}

      {canCreateUsers && (
        <div className="mb-4 rounded bg-white p-4 shadow">
          <h2 className="mb-2 font-semibold">Bulk Import Users (Excel)</h2>
          <p className="mb-3 text-sm text-slate-500">
            Upload an .xlsx file with columns: name, username (optional), email, password.
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
                if (file) uploadUsers(file);
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
              <th className="py-1">Status</th>
              <th className="py-1">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortLocally(users, sortBy, sortOrder).map((u) => (
              <tr key={u._id} className="border-t">
                <td className="py-2">{u._id}</td>
                <td className="py-2">{u.name}</td>
                <td className="py-2">{u.username || "-"}</td>
                <td className="py-2">{u.email}</td>
                <td className="py-2">{u.role}</td>
                <td className="py-2">
                  <span
                    className={`rounded px-2 py-1 text-xs ${
                      u.deleted_at
                        ? "bg-slate-200 text-slate-600"
                        : u.is_active === false
                          ? "bg-amber-100 text-amber-800"
                          : "bg-green-100 text-green-800"
                    }`}
                  >
                    {u.deleted_at ? "Deleted" : u.is_active === false ? "Suspended" : "Active"}
                  </span>
                </td>
                <td className="space-x-3 py-2">
                  <button
                    className="text-sm text-green-700 hover:underline disabled:opacity-60"
                    disabled={!canModerateUsers || loading || !!u.deleted_at}
                    onClick={() => openEdit(u)}
                  >
                    Edit
                  </button>
                  <button
                    className="text-sm text-blue-700 hover:underline disabled:opacity-60"
                    disabled={!canManageRoles || loading || !!u.deleted_at}
                    onClick={() => promoteToAdmin(u._id)}
                  >
                    Make admin
                  </button>
                  <button
                    className="text-sm text-amber-700 hover:underline disabled:opacity-60"
                    disabled={!canModerateUsers || loading || !!u.deleted_at}
                    onClick={() => toggleSuspend(u._id, u.is_active !== false)}
                  >
                    {u.is_active === false ? "Activate" : "Suspend"}
                  </button>
                  <button
                    className="text-sm text-red-700 hover:underline disabled:opacity-60"
                    disabled={!canModerateUsers || loading || !!u.deleted_at}
                    onClick={() => softDelete(u._id)}
                  >
                    Soft delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Edit User Dialog */}
      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>تعديل بيانات المستخدم</DialogTitle>
          </DialogHeader>
          <div className="space-y-3 pt-2">
            <div>
              <label className="text-sm text-slate-600">Name</label>
              <input
                className="w-full rounded border px-3 py-2"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-sm text-slate-600">Username</label>
              <input
                className="w-full rounded border px-3 py-2"
                value={editUsername}
                onChange={(e) => setEditUsername(e.target.value)}
              />
            </div>
            <div>
              <label className="text-sm text-slate-600">Email</label>
              <input
                className="w-full rounded border px-3 py-2"
                value={editEmail}
                onChange={(e) => setEditEmail(e.target.value)}
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
