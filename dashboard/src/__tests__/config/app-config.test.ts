/**
 * Unit tests for src/config/app-config.ts
 */

describe("APP_CONFIG", () => {
  // Dynamic import so the test captures the current module state
  let APP_CONFIG: any;

  beforeAll(async () => {
    const mod = await import("@/config/app-config");
    APP_CONFIG = mod.APP_CONFIG;
  });

  it("has a name", () => {
    expect(APP_CONFIG.name).toBe("Studio Admin");
  });

  it("has a version matching package.json", () => {
    expect(APP_CONFIG.version).toBe("2.0.0");
  });

  it("has copyright with current year", () => {
    const year = new Date().getFullYear();
    expect(APP_CONFIG.copyright).toContain(String(year));
  });

  it("has meta title and description", () => {
    expect(APP_CONFIG.meta.title).toBeDefined();
    expect(APP_CONFIG.meta.description).toBeDefined();
    expect(APP_CONFIG.meta.title.length).toBeGreaterThan(0);
    expect(APP_CONFIG.meta.description.length).toBeGreaterThan(0);
  });
});
