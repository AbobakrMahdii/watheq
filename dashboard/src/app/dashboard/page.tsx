"use client";

import React, { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

type User = {
  _id: string;
  name: string;
  username?: string | null;
  email: string;
  role: string;
};

export default function DashboardPage() {
  const router = useRouter();
  const [view, setView] = useState<"admins" | "users">("users");
  const [me, setMe] = useState<{ role?: string; email?: string } | null>(null);
  const [admins, setAdmins] = useState<User[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);

  const canManageRoles = me?.role === "super_admin";
  const canSeeAdminsTab = me?.role === "super_admin";
  const canCreateUsers = me?.role === "admin" || me?.role === "super_admin";

  const rows = useMemo(() => (view === "admins" ? admins : users), [admins, users, view]);

  async function loadAll() {
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

      if (meData?.role === "super_admin") {
        const adminsRes = await fetch("/api/admin/admins");
        const adminsData = await adminsRes.json().catch(() => []);
        if (!adminsRes.ok) throw new Error(adminsData?.message || "Failed to load admins");
        setAdmins(Array.isArray(adminsData) ? adminsData : []);
      } else {
        setAdmins([]);
        setView("users");
      }
    } catch (e: any) {
      toast.error(e?.message || "Failed to load data");
      router.push("/auth/login");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function promoteToAdmin(id: string) {
    if (!canManageRoles) return toast.error("Super admin only");
    try {
      const res = await fetch(`/api/admin/users/${id}/make-admin`, { method: "PUT" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("User promoted to admin");
      await loadAll();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  async function demoteToUser(id: string) {
    if (!canManageRoles) return toast.error("Super admin only");
    try {
      const res = await fetch(`/api/admin/users/${id}/remove-admin`, { method: "PUT" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("Admin removed");
      await loadAll();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  function logout() {
    fetch("/api/auth/logout", { method: "POST" })
      .catch(() => null)
      .finally(() => {
        toast.success("Logged out");
        router.push("/auth/login");
      });
  }

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
      await loadAll();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  async function createAdmin() {
    if (!canManageRoles) return toast.error("Super admin only");
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
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("Admin created");
      resetForm();
      await loadAll();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 bg-white border-r p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold">Menu</h3>
          <button onClick={logout} className="text-sm text-slate-600 hover:text-slate-900">
            Logout
          </button>
        </div>
        {canSeeAdminsTab && (
          <button
            className={`block w-full text-left py-2 px-3 rounded ${view === "admins" ? "bg-slate-100" : ""}`}
            onClick={() => setView("admins")}
          >
            Admins
          </button>
        )}
        <button
          className={`block w-full text-left py-2 px-3 rounded mt-2 ${view === "users" ? "bg-slate-100" : ""}`}
          onClick={() => setView("users")}
        >
          Users
        </button>
      </aside>

      <main className="flex-1 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-xl font-semibold">{view === "admins" ? "Admins" : "Users"}</h1>
            <p className="text-sm text-slate-500">
              Logged in as {me?.email || "—"} ({me?.role || "—"})
            </p>
          </div>
          <button
            onClick={loadAll}
            className="text-sm bg-slate-900 text-white px-3 py-2 rounded hover:bg-slate-800 disabled:opacity-60"
            disabled={loading}
          >
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>

        {((view === "users" && canCreateUsers) || (view === "admins" && canManageRoles)) && (
          <div className="bg-white p-4 rounded shadow mb-4">
            <h2 className="font-semibold mb-3">{view === "admins" ? "Create Admin" : "Create User"}</h2>
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
                onClick={view === "admins" ? createAdmin : createUser}
                disabled={loading}
                className="text-sm bg-blue-600 text-white px-3 py-2 rounded hover:bg-blue-700 disabled:opacity-60"
              >
                {view === "admins" ? "Add Admin" : "Add User"}
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
                <th className="py-1">Actions</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((u) => (
                <tr key={u._id} className="border-t">
                  <td className="py-2">{u._id}</td>
                  <td className="py-2">{u.name}</td>
                  <td className="py-2">{u.username || "—"}</td>
                  <td className="py-2">{u.email}</td>
                  <td className="py-2">{u.role}</td>
                  <td className="py-2">
                    {view === "users" ? (
                      <button
                        className="text-sm text-blue-700 hover:underline disabled:opacity-60"
                        disabled={!canManageRoles || loading}
                        onClick={() => promoteToAdmin(u._id)}
                      >
                        Make admin
                      </button>
                    ) : (
                      <button
                        className="text-sm text-red-700 hover:underline disabled:opacity-60"
                        disabled={!canManageRoles || loading || u.role === "super_admin"}
                        onClick={() => demoteToUser(u._id)}
                      >
                        Remove admin
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
