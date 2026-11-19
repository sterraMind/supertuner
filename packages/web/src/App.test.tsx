import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { App } from "./App";
import { ToastProvider } from "./components/Toast";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ToastProvider>
        <App />
      </ToastProvider>
    </MemoryRouter>,
  );
}

/**
 * Smoke tests for the router shell. No API is running here, so panels land in
 * their error/empty states — which is exactly the state this verifies.
 */
describe("App shell", () => {
  it("renders the dashboard at /", () => {
    renderAt("/");
    expect(screen.getByRole("heading", { level: 1, name: /train on the card/i })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /SuperTuner home/i })).toBeInTheDocument();
  });

  it("renders the recipes page without a server", () => {
    renderAt("/recipes");
    expect(screen.getByRole("heading", { level: 1, name: /known to fit/i })).toBeInTheDocument();
    expect(screen.getAllByText(/Open in planner/i).length).toBeGreaterThan(0);
  });

  it("renders the planner form with defaults", () => {
    renderAt("/planner");
    expect(screen.getByRole("heading", { level: 1, name: /Compose a RunSpec/i })).toBeInTheDocument();
    expect(screen.getByLabelText("Run name")).toBeInTheDocument();
    expect(screen.getByLabelText("Max steps")).toBeInTheDocument();
  });

  it("shows a not-found state for unknown routes", () => {
    renderAt("/does-not-exist");
    expect(screen.getByText(/This route does not exist/i)).toBeInTheDocument();
  });
});