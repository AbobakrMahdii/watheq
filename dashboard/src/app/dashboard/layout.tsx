import type { ReactNode } from "react";

import DashboardSidebar from "@/components/dashboard-sidebar";
import ProfileNav from "@/components/profile-nav";
import AuthGuard from "@/components/auth-guard";
import NotificationProvider from "@/components/notification-provider";
import NotificationBell from "@/components/notification-bell";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <AuthGuard>
      <NotificationProvider>
        <div className="h-screen overflow-hidden bg-slate-50 text-slate-900">
          <div className="flex h-full">
            <DashboardSidebar />
            <main className="flex-1 space-y-6 overflow-y-auto p-6">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <ProfileNav />
                </div>
                <div className="mr-4 self-start pt-4">
                  <NotificationBell />
                </div>
              </div>
              {children}
            </main>
          </div>
        </div>
      </NotificationProvider>
    </AuthGuard>
  );
}
