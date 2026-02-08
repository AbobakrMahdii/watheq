"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";

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

  const canManageRoles = me?.role === "super_admin";
  const canCreateUsers = me?.role === "admin" || me?.role === "super_admin";
  const canModerateUsers = me?.role === "admin" || me?.role === "super_admin";

  const [formName, setFormName] = useState("");
  const [formUsername, setFormUsername] = useState("");
  const [formEmail, setFormEmail] = useState("");
  const [formPassword, setFormPassword] = useState("");

  function resetForm() {
    setFormName("");
    setFormUsername("");
    setFormEmail("");
    setFormPassword("");
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
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold">Users</h1>
          <p className="text-sm text-slate-500">
            Logged in as {me?.email || "-"} ({me?.role || "-"})
          </p>
        </div>
        <button
          onClick={loadUsers}
          className="text-sm bg-slate-900 text-white px-3 py-2 rounded hover:bg-slate-800 disabled:opacity-60"
          disabled={loading}
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {canCreateUsers && (
        <div className="bg-white p-4 rounded shadow mb-4">
          <h2 className="font-semibold mb-3">Create User</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <input
              placeholder="Name"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
              className="border px-3 py-2 rounded"
            />
            <input
              placeholder="Username (optional)"
              value={formUsername}
              onChange={(e) => setFormUsername(e.target.value)}
              className="border px-3 py-2 rounded"
            />
            <input
              placeholder="Email"
              value={formEmail}
              onChange={(e) => setFormEmail(e.target.value)}
              className="border px-3 py-2 rounded"
            />
            <input
              placeholder="Password"
              type="password"
              value={formPassword}
              onChange={(e) => setFormPassword(e.target.value)}
              className="border px-3 py-2 rounded"
            />
          </div>
          <div className="mt-3 flex gap-2">
            <button
              onClick={createUser}
              disabled={loading}
              className="text-sm bg-blue-600 text-white px-3 py-2 rounded hover:bg-blue-700 disabled:opacity-60"
            >
              Add User
            </button>
            <button onClick={resetForm} className="text-sm px-3 py-2 rounded border hover:bg-slate-50">
              Clear
            </button>
          </div>
        </div>
      )}

      <div className="bg-white p-4 rounded shadow">
        <table className="w-full text-left">
          <thead>
            <tr>
              <th className="py-1">ID</th>
              <th className="py-1">Name</th>
              <th className="py-1">Username</th>
              <th className="py-1">Email</th>
              <th className="py-1">Role</th>
              <th className="py-1">Status</th>
              <th className="py-1">Actions</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u._id} className="border-t">
                <td className="py-2">{u._id}</td>
                <td className="py-2">{u.name}</td>
                <td className="py-2">{u.username || "-"}</td>
                <td className="py-2">{u.email}</td>
                <td className="py-2">{u.role}</td>
                <td className="py-2">
                  <span
                    className={`px-2 py-1 rounded text-xs ${
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
                <td className="py-2 space-x-3">
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
    </div>
  );
}
