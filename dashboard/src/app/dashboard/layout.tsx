import type { ReactNode } from "react";

import DashboardSidebar from "@/components/dashboard-sidebar";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex">
        <DashboardSidebar />
        <main className="flex-1 p-6">{children}</main>
      </div>
    </div>
  );
}
