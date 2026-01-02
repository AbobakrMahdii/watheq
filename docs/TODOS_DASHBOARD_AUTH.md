# TODOs: Dashboard Authentication and Minimal Admin/User UI

This file describes the tasks and acceptance criteria for implementing a simple authentication flow and a minimal protected dashboard UI (Admins + Users) in the `dashboard/` project.

Goals
- Add a login page that issues a token and stores it in cookies.
- Protect all `/dashboard` routes so unauthenticated users are redirected to `/auth/login`.
- Provide a main dashboard page with a sidebar containing two sections: Admins and Users.
- Provide add forms to create new admins (with role) and users. New items must update the client-side list immediately and show a success toast. Changes persist only in-memory until refresh.
- Keep only reusable code in the template; the dashboard should remain minimal but usable.

Tasks (ordered — finish one before starting the next)
1. Create middleware to protect `/dashboard` routes. (Server-side redirect to `/auth/login` if cookie missing.)
   - Acceptance: Request to `/dashboard` without cookie `token` redirects to `/auth/login`.

2. Create login UI page at `/auth/login`.
   - Form: `username`, `password` fields and `Login` button.
   - On success: set cookie `token`, navigate to `/dashboard`.
   - Acceptance: After login, cookie `token` exists and user lands on `/dashboard`.

3. Create minimal API route `/api/auth/login` that returns a token for valid credentials.
   - For this scaffold: accept `admin` / `admin` as valid admin credentials; otherwise return 401.
   - Acceptance: POST returns JSON `{ token }` when valid.

4. Add global `Toaster` in layout to allow success toast notifications.
   - Acceptance: `toast('...')` works on client pages.

5. Create client-side data store (`src/stores/data-store.ts`) that holds `admins` and `users` lists and provides `addAdmin`, `addUser` actions.
   - Initial sample data included.
   - Acceptance: Adding an item updates the store and components reflect change.

6. Implement `/dashboard` page UI
   - Left sidebar with two tabs (“Admins”, “Users”).
   - Main area shows list for selected tab and a small form to add a new item.
   - On add: call store action, show success toast, and update UI immediately without server fetch.
   - Acceptance: Add works, toast shown, list updated until refresh.

7. Type-check and quick dev run verification.
   - Acceptance: `npx tsc --noEmit` produces no new errors in modified files (or reasonable minor ones).

Notes & Caveats
- This implementation is intentionally simple and uses an in-memory client store. It is not a secure production auth — tokens are not signed nor validated server-side.
- For production: implement a proper auth backend, secure cookies (`HttpOnly`, `Secure`) and server-side session validation in middleware.

If you approve, I will apply these changes now and mark each TODO as completed as I finish it.
