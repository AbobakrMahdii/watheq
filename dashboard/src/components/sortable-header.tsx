"use client";

import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";

import type { SortOrder } from "@/hooks/use-sort-state";

interface SortableHeaderProps {
  label: string;
  field: string;
  currentSortBy: string;
  currentSortOrder: SortOrder;
  onSort: (field: string) => void;
  className?: string;
}

/**
 * A clickable table header cell that shows sort direction indicators.
 * Works with both `<th>` (raw HTML tables) and shadcn `<TableHead>`.
 */
export function SortableHeader({
  label,
  field,
  currentSortBy,
  currentSortOrder,
  onSort,
  className = "",
}: SortableHeaderProps) {
  const isActive = currentSortBy === field;

  return (
    <button
      type="button"
      onClick={() => onSort(field)}
      className={`inline-flex items-center gap-1 font-medium transition-colors hover:text-slate-900 ${className}`}
    >
      {label}
      {isActive ? (
        currentSortOrder === "asc" ? (
          <ArrowUp className="h-3.5 w-3.5" />
        ) : (
          <ArrowDown className="h-3.5 w-3.5" />
        )
      ) : (
        <ArrowUpDown className="h-3.5 w-3.5 opacity-40" />
      )}
    </button>
  );
}
