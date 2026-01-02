**Overview**
- **Purpose:** Instructions to reproduce the exact UI of the current dashboard template for a new project.
- **Source template:** [dashboard/README.md](dashboard/README.md)

**Prerequisites**
- **Node:** Install Node.js (recommended LTS).  
- **Package manager:** npm (works with npm install).  
- **Git:** to clone template or copy files.

**Core files to copy from this template**
- **Project config:** [dashboard/package.json](dashboard/package.json), [dashboard/next.config.mjs](dashboard/next.config.mjs), [dashboard/tsconfig.json](dashboard/tsconfig.json), [dashboard/postcss.config.mjs](dashboard/postcss.config.mjs)
- **App entry & layout:** [dashboard/src/app/layout.tsx](dashboard/src/app/layout.tsx), [dashboard/src/app/globals.css](dashboard/src/app/globals.css), [dashboard/src/app/not-found.tsx](dashboard/src/app/not-found.tsx)
- **UI components:** entire directory [dashboard/src/components/ui](dashboard/src/components/ui)
- **Shared UI & utilities:** [dashboard/src/components](dashboard/src/components), [dashboard/src/lib](dashboard/src/lib), [dashboard/src/hooks](dashboard/src/hooks)
- **Navigation & stores:** [dashboard/src/navigation](dashboard/src/navigation), [dashboard/src/stores](dashboard/src/stores)
- **Config & presets:** [dashboard/src/config](dashboard/src/config) (APP_CONFIG, theme presets)
- **Static assets:** [dashboard/public](dashboard/public) and [dashboard/media](dashboard/media)

**High-level steps (quick)**
1. Create a new repo/folder for your project.
2. Copy the files and folders listed above into the new project (preserve paths under `src/`).
3. Update `package.json` name/metadata. Keep the scripts and dependencies required for the UI.
4. Run `npm install` to install dependencies.
5. Verify `src/app/layout.tsx` and `src/app/globals.css` are present and reference the same CSS variables/themes.
6. Start dev server with `npm run dev` and verify the UI.

**Step-by-step (detailed)**
- Step 1 — Initialize new project folder
  - Create and enter a folder: `mkdir my-dashboard && cd my-dashboard`
  - Initialize git: `git init`

- Step 2 — Copy essential configuration
  - Copy `package.json`, `tsconfig.json`, `next.config.mjs`, and `postcss.config.mjs` from the template. Keep scripts: `dev`, `build`, `start`, `format`.
  - If you prefer a fresh `package.json`, copy only `dependencies` and `devDependencies` from the template's `package.json`.

- Step 3 — Copy `src/` structure (UI backbone)
  - Copy `src/app/layout.tsx` and `src/app/globals.css`.
  - Copy `src/components/ui/*` (all UI building blocks). These are shadcn-wrapped components used across the app.
  - Copy `src/navigation`, `src/stores`, `src/config`, `src/lib`, and `src/hooks` — these coordinate menus, preferences, and app state used by the layout.
  - Copy `public/` and `media/` assets used by the template for logos and screenshots.

- Step 4 — Theme & Tailwind setup
  - Ensure `tailwindcss` and `postcss` are installed (they are in `package.json` dependencies).  
  - Keep `globals.css` content including `@tailwind base; @tailwind components; @tailwind utilities;` and any CSS variables/themes.
  - If the template uses custom theme presets (see `src/config` and `src/scripts/generate-theme-presets.ts`), copy them and run any generate commands if present.

- Step 5 — Routes and pages
  - Keep the App Router structure under `src/app`. Copy routes present in the template (`(main)`, `(external)`, and page files) or scaffold your own pages but keep the `layout.tsx` to preserve global UI (sidebar, header, theme toggles).

- Step 6 — Environment & config updates
  - Open `src/config/app-config` or equivalent and update app title, URLs, and API endpoints.
  - If the template reads preferences from server actions (`src/server`), either implement minimal server-actions or stub `getPreference` to default values during initial setup.

- Step 7 — Install and run
  - Install deps: `npm install`
  - Run dev server: `npm run dev`
  - Visit `http://localhost:3000` — the UI should match the template if `layout.tsx` and `components/ui` were copied correctly.

**Common gotchas & tips**
- **Fonts & Inter:** The template uses Google fonts (`Inter`) via `next/font/google` in `layout.tsx`. Keep that import or replace with your preferred font.
- **Theme data attributes:** `layout.tsx` sets `data-theme-preset` and `className` for dark mode — keep these attributes for theme CSS to apply.
- **Server-actions:** If the template calls server helpers (e.g., `getPreference`), stub them for frontend-only work or implement equivalent logic.
- **Shadcn UI:** Many components are shadcn wrappers. Ensure `tailwindcss` and required Radix/sonner deps are installed.

**Commands (copyable)**
```bash
# from your new project root
npm install
npm run dev
# optional build
npm run build
npm start
```

**Checklist before committing the new project**
- [ ] `package.json` updated (name, repo, metadata)
- [ ] `src/app/layout.tsx` present and retained
- [ ] `src/components/ui` copied entirely
- [ ] `globals.css` and Tailwind config present
- [ ] `public/` assets copied
- [ ] App title and APP_CONFIG updated
- [ ] Dev server runs and UI renders at `/`

**Next steps I can help with**
- Scaffold a new repo automatically using this template and update APP_CONFIG.  
- Create a minimal starter repo that contains only the UI shell (layout + UI components) so you can plug backend logic later.

---
Generated from the template at `dashboard/` in this workspace. If you want, I can now scaffold a fresh project folder with the minimal UI shell copied and ready to `npm install` — shall I proceed?

**Agent-Focused Instructions (exact, runnable)**

- **Goal:** reproduce the same UI as this template. Follow these steps exactly to guarantee the UI matches the template from a component, CSS, and asset perspective.
- **Guarantee:** If you copy the files and directories listed in "Core files to copy from this template" verbatim, install identical `dependencies` and `devDependencies` from `package.json`, and preserve the `public/` and `media/` assets, the UI will be functionally and visually equivalent to the template. Minor differences can arise from environment-specific font loading, OS rendering differences, or different Node/Tailwind versions — see caveats below.

Agent task list (runnable):

1. Create a target folder for the new project and initialize git (optional):

```powershell
mkdir C:\path\to\my-dashboard
cd C:\path\to\my-dashboard
git init
```

2. Copy exact files and directories from the template (preserve structure). From the workspace root run (PowerShell command example):

```powershell
$src = "C:\Users\sadeq\Desktop\watheq\dashboard"
$dst = "C:\path\to\my-dashboard"
robocopy $src $dst package.json next.config.mjs tsconfig.json postcss.config.mjs /COPYALL
robocopy "$src\src" "$dst\src" app components config hooks lib navigation stores styles types /E /COPYALL
robocopy "$src\public" "$dst\public" /E /COPYALL
robocopy "$src\media" "$dst\media" /E /COPYALL
```

Unix/bash alternative (works in WSL or mac/linux):

```bash
TEMPLATE=~/workspace/watheq/dashboard
TARGET=~/workspace/my-dashboard
mkdir -p "$TARGET"
cp -a "$TEMPLATE/package.json" "$TARGET/"
cp -a "$TEMPLATE/next.config.mjs" "$TARGET/"
cp -a "$TEMPLATE/tsconfig.json" "$TARGET/"
cp -a "$TEMPLATE/postcss.config.mjs" "$TARGET/"
cp -a "$TEMPLATE/src" "$TARGET/src"
cp -a "$TEMPLATE/public" "$TARGET/public"
cp -a "$TEMPLATE/media" "$TARGET/media"
```

3. (Optional but recommended) If you want a minimal UI-only scaffold, remove route folders that depend on backend logic, but keep `src/app/layout.tsx`, `src/components/ui`, `src/app/globals.css`, `src/navigation`, and `src/stores`.

4. Install dependencies and generate presets (if applicable):

```bash
cd C:\path\to\my-dashboard
npm install
# If the template provides a presets generator
npm run generate:presets || true
```

5. Sanity-run the dev server:

```bash
npm run dev
# Visit http://localhost:3000
```

6. If the app references server actions (e.g., `getPreference`) and you don't have a backend yet, stub them. Example stub file `src/server/stub-server-actions.ts`:

```ts
export async function getPreference(key:string, values:unknown, def:any){
  return def;
}
```

Then adjust imports in `src/app/layout.tsx` to import from the stub until the real API is available.

**Guarantees & Caveats (concise)**
- Guarantee scope: visual/UI parity is guaranteed if the following are identical: component source (`src/components/ui`), global CSS (`src/app/globals.css`), theme presets, and static assets. Also ensure identical package versions from `package.json` are installed.
- Non-guaranteed differences: minor font rendering differences across OS/browsers, differing Node/Tailwind versions, or un-copied dynamic assets. If you require pixel-perfect screenshots for tests, pin Node and Tailwind versions and copy the exact font files (or ensure same Google font loading).
- Runtime dependencies: some features (theme presets, preferences) may call server-side utilities. For an agent reproduction that focuses on UI only, stub server calls as shown above.

**Agent acceptance criteria**
- `npm run dev` serves the app and main layout matches the template homepage (sidebar, header, theme toggles visible).
- No missing asset 404s in the browser console for `public/` or `media/` paths.
- `src/components/ui` contains the component set used across pages (e.g., `button.tsx`, `sidebar.tsx`, `table.tsx`).

If you want, I can now scaffold the minimal UI-only project automatically in `c:\Users\sadeq\Desktop\watheq\scaffolded-dashboard` and run `npm install` to verify — shall I proceed?