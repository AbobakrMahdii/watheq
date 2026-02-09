import type { SortOrder } from "@/hooks/use-sort-state";

/**
 * Generic client-side sorter for arrays of objects.
 * Handles string, number, boolean, and ISO-date comparisons.
 */
export function sortLocally<T extends Record<string, unknown>>(data: T[], sortBy: string, sortOrder: SortOrder): T[] {
  if (!sortBy || data.length === 0) return data;

  return [...data].sort((a, b) => {
    const va = a[sortBy];
    const vb = b[sortBy];

    // Nulls / undefined always sort last
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;

    let cmp = 0;

    if (typeof va === "number" && typeof vb === "number") {
      cmp = va - vb;
    } else if (typeof va === "boolean" && typeof vb === "boolean") {
      cmp = Number(va) - Number(vb);
    } else {
      const sa = String(va);
      const sb = String(vb);
      // Try date comparison (ISO strings)
      const da = Date.parse(sa);
      const db = Date.parse(sb);
      if (!isNaN(da) && !isNaN(db)) {
        cmp = da - db;
      } else {
        cmp = sa.localeCompare(sb, undefined, { sensitivity: "base" });
      }
    }

    return sortOrder === "asc" ? cmp : -cmp;
  });
}
