import { fileURLToPath } from "node:url";
import { join } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The harness environment exports NODE_ENV=production, which makes React load
// its production build (no React.act) inside the test worker. Pin the test run
// to a development React so Testing Library can wrap updates in act().
if (process.env.VITEST) process.env.NODE_ENV = "test";

/**
 * The browser bundle imports the wire contracts straight from `packages/core/src`
 * so the UI never types against a stale `dist/`. The aliases below mirror the
 * `paths` entries in tsconfig.json.
 *
 * `node:*` is stubbed for the browser (dev and build) because the core barrel
 * re-exports `config.ts`, which is Node-only. Vitest runs in Node, so the real
 * built-ins stay in place there.
 */
const webRoot = fileURLToPath(new URL(".", import.meta.url));
const coreSrc = fileURLToPath(new URL("../core/src", import.meta.url));

const coreAlias = [
  { find: /^@supertuner\/core\/contracts$/, replacement: join(coreSrc, "contracts.ts") },
  { find: /^@supertuner\/core\/events$/, replacement: join(coreSrc, "events.ts") },
  { find: /^@supertuner\/core\/plugins$/, replacement: join(coreSrc, "plugins.ts") },
  { find: /^@supertuner\/core\/recipes$/, replacement: join(coreSrc, "recipes.ts") },
  { find: /^@supertuner\/core$/, replacement: join(coreSrc, "index.ts") },
];

const nodeStub = {
  find: /^node:(fs|os|path|url)$/,
  replacement: join(webRoot, "src/lib/node-browser-stub.ts"),
};

export default defineConfig(({ mode }) => {
  const isVitest = mode === "test" || process.env.VITEST === "true";
  return {
    plugins: [react()],
    resolve: { alias: isVitest ? coreAlias : [...coreAlias, nodeStub] },
    server: {
      port: 5173,
      strictPort: false,
      proxy: {
        "/api": {
          target: "http://127.0.0.1:8787",
          changeOrigin: false,
        },
      },
    },
    build: {
      outDir: "dist",
      target: "es2022",
      sourcemap: false,
    },
    test: {
      environment: "jsdom",
      env: { NODE_ENV: "test" },
      setupFiles: ["./src/test/setup.ts"],
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
      restoreMocks: true,
    },
  };
});