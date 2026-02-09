/**
 * Unit tests for the NotFound page component.
 */
import React from "react";

import { render, screen } from "@testing-library/react";

import NotFound from "@/app/not-found";

// Mock the Button component
jest.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}));

// Mock next/link
jest.mock("next/link", () => {
  return ({ children, href, replace, prefetch, scroll, ...rest }: any) => (
    <a href={href} {...rest}>
      {children}
    </a>
  );
});

describe("NotFound Page", () => {
  it("renders 404 heading", () => {
    render(<NotFound />);
    expect(screen.getByText("Page not found.")).toBeInTheDocument();
  });

  it("renders description text", () => {
    render(<NotFound />);
    expect(screen.getByText("The page you are looking for could not be found.")).toBeInTheDocument();
  });

  it("renders a link back to dashboard", () => {
    render(<NotFound />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/dashboard");
  });

  it("has a Go back home button", () => {
    render(<NotFound />);
    expect(screen.getByText("Go back home")).toBeInTheDocument();
  });
});
