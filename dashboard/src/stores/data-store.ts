"use client";

import create from "zustand";

export type Person = { id: number; name: string; role: string };

type State = {
  admins: Person[];
  users: Person[];
  addAdmin: (p: Omit<Person, "id">) => void;
  addUser: (p: Omit<Person, "id">) => void;
};

let nextId = 100;

export const useDataStore = create<State>((set) => ({
  admins: [
    { id: 1, name: "Primary Admin", role: "admin" },
    { id: 2, name: "Sub Admin", role: "sub_admin" },
  ],
  users: [{ id: 1, name: "Sample User", role: "user" }],
  addAdmin: (p) =>
    set((s) => ({ admins: [...s.admins, { id: ++nextId, name: p.name, role: p.role }] })),
  addUser: (p) => set((s) => ({ users: [...s.users, { id: ++nextId, name: p.name, role: "user" }] })),
}));
