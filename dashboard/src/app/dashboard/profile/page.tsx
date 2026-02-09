"use client";

import { useEffect, useState } from "react";

import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";

type Me = {
  id?: number | string;
  name?: string;
  username?: string;
  email?: string;
  role?: string;
};

export default function ProfilePage() {
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);

  // Change password state
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [changingPwd, setChangingPwd] = useState(false);
  const [pwdError, setPwdError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/auth/me");
        const data = await res.json().catch(() => ({}));
        if (res.ok) setMe(data);
      } catch {
        toast.error("Failed to load profile");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  async function handleChangePassword() {
    setPwdError("");

    if (!currentPassword || !newPassword || !confirmPassword) {
      setPwdError("All fields are required");
      return;
    }

    if (newPassword.length < 6) {
      setPwdError("New password must be at least 6 characters");
      return;
    }

    if (newPassword !== confirmPassword) {
      setPwdError("New passwords do not match");
      return;
    }

    setChangingPwd(true);
    try {
      const res = await fetch("/api/auth/change-password", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.message || "Request failed");
      toast.success("Password changed successfully");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (e: any) {
      setPwdError(e?.message || "Failed to change password");
    } finally {
      setChangingPwd(false);
    }
  }

  if (loading) {
    return <div className="p-6 text-sm text-slate-500">Loading...</div>;
  }

  return (
    <div className="max-w-2xl">
      <h1 className="mb-4 text-xl font-semibold">My Profile</h1>

      {/* Profile Details */}
      <div className="mb-6 rounded bg-white p-6 shadow">
        <div className="mb-4 flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-slate-100 text-lg font-bold text-slate-700">
            {(me?.name || me?.email || "U").slice(0, 2).toUpperCase()}
          </div>
          <div>
            <div className="text-lg font-semibold">{me?.name || "—"}</div>
            <div className="text-sm text-slate-500">{me?.email || "—"}</div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-500">Username</span>
            <div className="font-medium">{me?.username || "—"}</div>
          </div>
          <div>
            <span className="text-slate-500">Role</span>
            <div>
              <Badge variant="secondary" className="mt-1">
                {me?.role || "—"}
              </Badge>
            </div>
          </div>
          <div>
            <span className="text-slate-500">User ID</span>
            <div className="font-medium">{me?.id ?? "—"}</div>
          </div>
        </div>
      </div>

      {/* Change Password */}
      <div className="rounded bg-white p-6 shadow">
        <h2 className="mb-4 font-semibold">Change Password</h2>

        {pwdError && (
          <div className="mb-3 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">{pwdError}</div>
        )}

        <div className="space-y-3">
          <div>
            <label className="text-sm text-slate-600">Current Password</label>
            <input
              type="password"
              className="w-full rounded border px-3 py-2"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>
          <div>
            <label className="text-sm text-slate-600">New Password</label>
            <input
              type="password"
              className="w-full rounded border px-3 py-2"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>
          <div>
            <label className="text-sm text-slate-600">Confirm New Password</label>
            <input
              type="password"
              className="w-full rounded border px-3 py-2"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>
          <button
            onClick={handleChangePassword}
            disabled={changingPwd}
            className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-60"
          >
            {changingPwd ? "Changing..." : "Change Password"}
          </button>
        </div>
      </div>
    </div>
  );
}
