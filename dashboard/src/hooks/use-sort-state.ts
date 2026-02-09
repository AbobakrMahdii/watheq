"use client";

import { useCallback, useState } from "react";

export type SortOrder = "asc" | "desc";

export interface SortState {
  sortBy: string;
  sortOrder: SortOrder;
}

/**
 * Lightweight hook for managing sort state.
 * Clicking the same column toggles asc↔desc; clicking a new column resets to defaultOrder.
 */
export function useSortState(defaultSortBy: string, defaultSortOrder: SortOrder = "desc") {
  const [sortBy, setSortBy] = useState(defaultSortBy);
  const [sortOrder, setSortOrder] = useState<SortOrder>(defaultSortOrder);

  const toggleSort = useCallback(
    (field: string) => {
      if (field === sortBy) {
        setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortBy(field);
        setSortOrder(defaultSortOrder);
      }
    },
    [sortBy, defaultSortOrder],
  );

  return { sortBy, sortOrder, toggleSort };
}
