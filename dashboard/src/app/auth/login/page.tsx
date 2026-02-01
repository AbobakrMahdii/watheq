"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

export default function LoginPage() {
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Send email or username based on what user entered
      const loginPayload = identifier.includes("@") 
        ? { email: identifier, password }
        : { username: identifier, password };

      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(loginPayload),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const errorMessage = data?.message || "Invalid credentials";
        throw new Error(errorMessage);
      }

      toast.success("Login successful");
      router.push("/dashboard");
    } catch (err: any) {
      toast.error(err?.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <form onSubmit={handleSubmit} className="w-full max-w-sm bg-white p-6 rounded shadow">
        <h2 className="text-lg font-semibold mb-4">Admin Login</h2>
        <label className="block mb-2">
          <span className="text-sm">Email أو Username</span>
          <input
            className="mt-1 block w-full border px-2 py-1 rounded"
            value={identifier}
            onChange={(e) => setIdentifier(e.target.value)}
            required
          />
        </label>
        <label className="block mb-4">
          <span className="text-sm">Password</span>
          <input
            type="password"
            className="mt-1 block w-full border px-2 py-1 rounded"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
          disabled={loading}
        >
          {loading ? "Logging in..." : "Login"}
        </button>
      </form>
    </div>
  );
}
