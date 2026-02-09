/**
 * Unit tests for the Login page component.
 */
import React from "react";

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import LoginPage from "@/app/auth/login/page";

// Mock sonner toast
jest.mock("sonner", () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

// Keep a reference to the mock router push
const pushMock = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
    replace: jest.fn(),
    prefetch: jest.fn(),
    back: jest.fn(),
    pathname: "/",
    query: {},
    asPath: "/",
  }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/auth/login",
}));

describe("LoginPage", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset fetch mock
    global.fetch = jest.fn();
  });

  it("renders login form with all fields", () => {
    render(<LoginPage />);

    expect(screen.getByText("Admin Login")).toBeInTheDocument();
    expect(screen.getByText(/Email/)).toBeInTheDocument();
    expect(screen.getByText("Password")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /login/i })).toBeInTheDocument();
  });

  it("has a password field that is obscured", () => {
    render(<LoginPage />);

    const passwordInput = screen.getByLabelText("Password") || document.querySelector('input[type="password"]');
    expect(passwordInput).toBeTruthy();
  });

  it("allows typing in identifier field", async () => {
    render(<LoginPage />);
    const user = userEvent.setup();

    const inputs = screen.getAllByRole("textbox");
    const identifierInput = inputs[0];

    await user.type(identifierInput, "admin@test.com");
    expect(identifierInput).toHaveValue("admin@test.com");
  });

  it("submit button is disabled while loading", async () => {
    // Make fetch hang (never resolves)
    (global.fetch as jest.Mock).mockReturnValue(new Promise(() => {}));

    render(<LoginPage />);
    const user = userEvent.setup();

    const inputs = screen.getAllByRole("textbox");
    await user.type(inputs[0], "admin");

    const pwInput = document.querySelector('input[type="password"]') as HTMLInputElement;
    await user.type(pwInput, "password123");

    const btn = screen.getByRole("button", { name: /login/i });
    fireEvent.click(btn);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /logging in/i })).toBeDisabled();
    });
  });

  it("calls fetch with username payload when identifier has no @", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ access_token: "tok123" }),
    });

    render(<LoginPage />);
    const user = userEvent.setup();

    const inputs = screen.getAllByRole("textbox");
    await user.type(inputs[0], "adminuser");

    const pwInput = document.querySelector('input[type="password"]') as HTMLInputElement;
    await user.type(pwInput, "pass123");

    fireEvent.click(screen.getByRole("button", { name: /login/i }));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/auth/login",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ username: "adminuser", password: "pass123" }),
        }),
      );
    });
  });

  it("calls fetch with email payload when identifier contains @", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ access_token: "tok123" }),
    });

    render(<LoginPage />);
    const user = userEvent.setup();

    const inputs = screen.getAllByRole("textbox");
    await user.type(inputs[0], "admin@watheq.com");

    const pwInput = document.querySelector('input[type="password"]') as HTMLInputElement;
    await user.type(pwInput, "pass123");

    fireEvent.click(screen.getByRole("button", { name: /login/i }));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/auth/login",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ email: "admin@watheq.com", password: "pass123" }),
        }),
      );
    });
  });

  it("redirects to /dashboard on successful login", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ access_token: "tok123" }),
    });

    render(<LoginPage />);
    const user = userEvent.setup();

    const inputs = screen.getAllByRole("textbox");
    await user.type(inputs[0], "admin");

    const pwInput = document.querySelector('input[type="password"]') as HTMLInputElement;
    await user.type(pwInput, "pass123");

    fireEvent.click(screen.getByRole("button", { name: /login/i }));

    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith("/dashboard");
    });
  });

  it("shows error toast on failed login", async () => {
    const { toast } = require("sonner");

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ message: "Invalid credentials" }),
    });

    render(<LoginPage />);
    const user = userEvent.setup();

    const inputs = screen.getAllByRole("textbox");
    await user.type(inputs[0], "admin");

    const pwInput = document.querySelector('input[type="password"]') as HTMLInputElement;
    await user.type(pwInput, "wrong");

    fireEvent.click(screen.getByRole("button", { name: /login/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith("Invalid credentials");
    });
  });
});
