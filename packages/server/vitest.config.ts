import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * Tests run against the core package source so a stale `packages/core/dist`
 * cannot make the server suite pass against old contracts. The published
 * runtime still imports `@supertuner/core` normally.
 */
export default defineConfig({
  resolve: {
    alias: {
      "@supertuner/core": fileURLToPath(new URL("../core/src/index.ts", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    setupFiles: ["./test/setup.ts"],
    testTimeout: 30_000,
    hookTimeout: 30_000,
    pool: "forks",
  },
});
