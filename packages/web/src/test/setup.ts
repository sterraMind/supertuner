import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import { resetApiClientForTests } from "../lib/api";
import { resetThemeControllerForTests } from "../theme/theme";

afterEach(() => {
  cleanup();
  resetApiClientForTests();
  resetThemeControllerForTests();
  try {
    window.localStorage.clear();
  } catch {
    /* storage unavailable in this environment */
  }
});