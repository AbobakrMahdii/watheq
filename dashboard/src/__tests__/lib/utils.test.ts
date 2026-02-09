/**
 * Unit tests for src/lib/utils.ts
 */
import { cn, getInitials, formatCurrency } from "@/lib/utils";

describe("cn (classname merge)", () => {
  it("merges multiple class strings", () => {
    expect(cn("px-2", "py-1")).toBe("px-2 py-1");
  });

  it("deduplicates conflicting tailwind classes", () => {
    const result = cn("px-2", "px-4");
    expect(result).toBe("px-4");
  });

  it("handles conditional classes", () => {
    const isActive = true;
    expect(cn("btn", isActive && "btn-active")).toContain("btn-active");
  });

  it("filters out falsy values", () => {
    expect(cn("base", false, null, undefined, "extra")).toBe("base extra");
  });

  it("returns empty string for no input", () => {
    expect(cn()).toBe("");
  });
});

describe("getInitials", () => {
  it("returns initials for a full name", () => {
    expect(getInitials("John Doe")).toBe("JD");
  });

  it("returns single initial for one word", () => {
    expect(getInitials("Admin")).toBe("A");
  });

  it("returns ? for empty string", () => {
    expect(getInitials("")).toBe("?");
  });

  it("returns ? for whitespace-only string", () => {
    expect(getInitials("   ")).toBe("?");
  });

  it("handles three-word names", () => {
    expect(getInitials("Abobakr Ali Mahdi")).toBe("AAM");
  });

  it("uppercases lowercase initials", () => {
    expect(getInitials("john doe")).toBe("JD");
  });

  it("returns ? for non-string input", () => {
    // @ts-expect-error testing invalid input
    expect(getInitials(null)).toBe("?");
  });
});

describe("formatCurrency", () => {
  it("formats USD by default", () => {
    const result = formatCurrency(1234.56);
    expect(result).toContain("1,234.56");
  });

  it("formats with no decimals option", () => {
    const result = formatCurrency(1234.56, { noDecimals: true });
    expect(result).toContain("1,235");
    expect(result).not.toContain(".");
  });

  it("formats different currencies", () => {
    const result = formatCurrency(100, { currency: "EUR", locale: "de-DE" });
    expect(result).toBeDefined();
  });

  it("handles zero", () => {
    const result = formatCurrency(0);
    expect(result).toContain("0");
  });

  it("handles negative amounts", () => {
    const result = formatCurrency(-50);
    expect(result).toContain("50");
  });
});
