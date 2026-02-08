"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";

type User = {
  _id: string;
  name: string;
  username?: string | null;
  email: string;
  role: string;
};

export default function AdminsPage() {
  const [me, setMe] = useState<{ role?: string; email?: string } | null>(null);
  const [admins, setAdmins] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);

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

  async function loadAdmins() {
    setLoading(true);
    try {
      const meRes = await fetch("/api/auth/me");
      const meData = await meRes.json().catch(() => ({}));
      if (!meRes.ok) throw new Error(meData?.message || "Unauthorized");
      setMe(meData);

      if (meData?.role !== "super_admin") {
        toast.error("Super admin only");
        return;
      }

      const adminsRes = await fetch("/api/admin/admins");
      const adminsData = await adminsRes.json().catch(() => []);
      if (!adminsRes.ok) throw new Error(adminsData?.message || "Failed to load admins");
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
    if (me?.role !== "super_admin") return toast.error("Super admin only");
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
      await loadAdmins();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  async function demoteToUser(id: string, role: string) {
    if (me?.role !== "super_admin") return toast.error("Super admin only");
    if (role === "super_admin") return toast.error("Cannot demote super admin");
    try {
      const res = await fetch(`/api/admin/users/${id}/remove-admin`, { method: "PUT" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("Admin removed");
      await loadAdmins();
    } catch (e: any) {
      toast.error(e?.message || "Request failed");
    }
  }

  if (me && me.role !== "super_admin") {
    return (
      <div className="rounded border bg-white p-6 text-sm text-slate-600">
        This page is available for super admins only.
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold">Admins</h1>
          <p className="text-sm text-slate-500">
            Logged in as {me?.email || "—"} ({me?.role || "—"})
          </p>
        </div>
        <button
          onClick={loadAdmins}
          className="text-sm bg-slate-900 text-white px-3 py-2 rounded hover:bg-slate-800 disabled:opacity-60"
          disabled={loading}
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      <div className="bg-white p-4 rounded shadow mb-4">
        <h2 className="font-semibold mb-3">Create Admin</h2>
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
            onClick={createAdmin}
            disabled={loading}
            className="text-sm bg-blue-600 text-white px-3 py-2 rounded hover:bg-blue-700 disabled:opacity-60"
          >
            Add Admin
          </button>
          <button onClick={resetForm} className="text-sm px-3 py-2 rounded border hover:bg-slate-50">
            Clear
          </button>
        </div>
      </div>

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
            {admins.map((u) => (
              <tr key={u._id} className="border-t">
                <td className="py-2">{u._id}</td>
                <td className="py-2">{u.name}</td>
                <td className="py-2">{u.username || "—"}</td>
                <td className="py-2">{u.email}</td>
                <td className="py-2">{u.role}</td>
                <td className="py-2">
                  <button
                    className="text-sm text-red-700 hover:underline disabled:opacity-60"
                    disabled={loading || u.role === "super_admin"}
                    onClick={() => demoteToUser(u._id, u.role)}
                  >
                    Remove admin
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
