"use client";

import React, { useState } from "react";
import { useDataStore } from "@/stores/data-store";
import { toast } from "sonner";

export default function DashboardPage() {
  const [view, setView] = useState<"admins" | "users">("admins");
  const admins = useDataStore((s) => s.admins);
  const users = useDataStore((s) => s.users);
  const addAdmin = useDataStore((s) => s.addAdmin);
  const addUser = useDataStore((s) => s.addUser);

  const [name, setName] = useState("");
  const [role, setRole] = useState("sub_admin");

  function handleAdd() {
    if (!name.trim()) return toast.error("Name required");

    if (view === "admins") {
      addAdmin({ name: name.trim(), role });
      toast.success("Admin added");
    } else {
      addUser({ name: name.trim(), role: "user" });
      toast.success("User added");
    }

    setName("");
  }

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 bg-white border-r p-4">
        <h3 className="font-semibold mb-4">Menu</h3>
        <button
          className={`block w-full text-left py-2 px-3 rounded ${view === "admins" ? "bg-slate-100" : ""}`}
          onClick={() => setView("admins")}
        >
          Admins
        </button>
        <button
          className={`block w-full text-left py-2 px-3 rounded mt-2 ${view === "users" ? "bg-slate-100" : ""}`}
          onClick={() => setView("users")}
        >
          Users
        </button>
      </aside>

      <main className="flex-1 p-6">
        <h1 className="text-xl font-semibold mb-4">{view === "admins" ? "Admins" : "Users"}</h1>

        <div className="mb-6">
          <input
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="border px-2 py-1 mr-2 rounded"
          />
          {view === "admins" && (
            <select value={role} onChange={(e) => setRole(e.target.value)} className="border px-2 py-1 mr-2 rounded">
              <option value="admin">admin</option>
              <option value="sub_admin">sub_admin</option>
            </select>
          )}
          <button onClick={handleAdd} className="bg-green-600 text-white px-3 py-1 rounded">
            Add {view === "admins" ? "Admin" : "User"}
          </button>
        </div>

        <div className="bg-white p-4 rounded shadow">
          <table className="w-full text-left">
            <thead>
              <tr>
                <th className="py-1">ID</th>
                <th className="py-1">Name</th>
                <th className="py-1">Role</th>
              </tr>
            </thead>
            <tbody>
              {(view === "admins" ? admins : users).map((p) => (
                <tr key={p.id} className="border-t">
                  <td className="py-2">{p.id}</td>
                  <td className="py-2">{p.name}</td>
                  <td className="py-2">{p.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
