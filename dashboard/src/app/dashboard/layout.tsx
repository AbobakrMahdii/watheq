import type { ReactNode } from "react";

import DashboardSidebar from "@/components/dashboard-sidebar";
import ProfileNav from "@/components/profile-nav";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex">
        <DashboardSidebar />
        <main className="flex-1 p-6 space-y-6">
          <ProfileNav />
          {children}
        </main>
      </div>
    </div>
  );
}
