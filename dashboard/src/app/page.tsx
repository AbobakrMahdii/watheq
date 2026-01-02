import { cookies } from "next/headers";
import { redirect } from "next/navigation";

export default async function Home() {
    const token = (await cookies()).get("token")?.value;
  if(token){
    redirect('/dashboard');
    
  }
    return (
    <main className="min-h-screen flex flex-col items-center justify-center">
      <h1 className="text-2xl font-semibold">Minimal Dashboard Home</h1>
      <a href="/auth/login">Login</a>
    </main>
  );
}
