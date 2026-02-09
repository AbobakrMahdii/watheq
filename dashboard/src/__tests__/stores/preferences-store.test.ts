/**
 * Unit tests for src/stores/preferences/preferences-store.ts
 */
import { createPreferencesStore } from "@/stores/preferences/preferences-store";

describe("PreferencesStore", () => {
  it("creates a store with default values", () => {
    const store = createPreferencesStore();
    const state = store.getState();

    expect(state.themeMode).toBe("light");
    expect(state.themePreset).toBe("default");
  });

  it("accepts initial themeMode", () => {
    const store = createPreferencesStore({ themeMode: "dark" });
    expect(store.getState().themeMode).toBe("dark");
  });

  it("accepts initial themePreset", () => {
    const store = createPreferencesStore({ themePreset: "blue" as any });
    expect(store.getState().themePreset).toBe("blue");
  });

  it("setThemeMode updates the mode", () => {
    const store = createPreferencesStore();
    expect(store.getState().themeMode).toBe("light");

    store.getState().setThemeMode("dark");
    expect(store.getState().themeMode).toBe("dark");
  });

  it("setThemePreset updates the preset", () => {
    const store = createPreferencesStore();
    store.getState().setThemePreset("emerald" as any);
    expect(store.getState().themePreset).toBe("emerald");
  });

  it("store supports subscribe", () => {
    const store = createPreferencesStore();
    const listener = jest.fn();

    store.subscribe(listener);
    store.getState().setThemeMode("dark");

    expect(listener).toHaveBeenCalled();
  });
});
