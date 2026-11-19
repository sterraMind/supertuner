import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { expandUserPath, isPathInside, loadServerConfig, parseArgv } from "../src/config.js";

describe("parseArgv", () => {
  it("splits a quoted Windows path without losing backslashes", () => {
    expect(parseArgv('"C:\\Program Files\\uv\\uv.exe" run --project engine supertuner-engine')).toEqual([
      "C:\\Program Files\\uv\\uv.exe",
      "run",
      "--project",
      "engine",
      "supertuner-engine",
    ]);
  });

  it("supports single quotes and empty arguments", () => {
    expect(parseArgv("node 'fake engine.mjs' ''")).toEqual(["node", "fake engine.mjs", ""]);
  });

  it("does not interpret shell operators or variables", () => {
    expect(parseArgv("echo hi | grep $HOME && x")).toEqual(["echo", "hi", "|", "grep", "$HOME", "&&", "x"]);
  });

  it("throws on an unterminated quote", () => {
    expect(() => parseArgv('node "unterminated')).toThrow(/unterminated/);
  });

  it("throws on an empty command", () => {
    expect(() => parseArgv("   ")).toThrow(/empty/);
  });
});

describe("path helpers", () => {
  it("expands ~", () => {
    const expanded = expandUserPath("~/supertuner");
    expect(expanded.startsWith("~")).toBe(false);
    expect(expanded).toContain("supertuner");
  });

  it("detects containment and escapes", () => {
    const root = join(tmpdir(), "supertuner-root");
    expect(isPathInside(root, join(root, "runs", "a"))).toBe(true);
    expect(isPathInside(root, root)).toBe(true);
    expect(isPathInside(root, join(root, "..", "elsewhere"))).toBe(false);
    expect(isPathInside(join(root, "a"), join(root, "ab"))).toBe(false);
  });

  it("compares case-insensitively on Windows", () => {
    if (process.platform !== "win32") return;
    expect(isPathInside("C:\\Data", "c:\\data\\runs\\x")).toBe(true);
  });
});

describe("loadServerConfig", () => {
  it("uses the documented defaults", () => {
    const home = mkdtempSync(join(tmpdir(), "supertuner-config-"));
    const config = loadServerConfig({ env: { SUPERTUNER_HOME: home } });
    expect(config.home).toBe(home);
    expect(config.host).toBe("127.0.0.1");
    expect(config.port).toBe(8787);
    expect(config.maxConcurrentRuns).toBe(1);
    expect(config.allowAnyPath).toBe(false);
    expect(config.dbPath).toBe(join(home, "supertuner.db"));
    expect(config.runsDir).toBe(join(home, "runs"));
    expect(config.engineCmd[0]).toBe("uv");
    expect(config.allowedRoots).toContain(home);
  });

  it("honours SUPERTUNER_ENGINE_CMD, SUPERTUNER_PORT and ALLOW_ANY_PATH", () => {
    const home = mkdtempSync(join(tmpdir(), "supertuner-config-"));
    const config = loadServerConfig({
      env: {
        SUPERTUNER_HOME: home,
        SUPERTUNER_ENGINE_CMD: '"C:\\Program Files\\nodejs\\node.exe" "fake engine.mjs"',
        SUPERTUNER_PORT: "9999",
        SUPERTUNER_ALLOW_ANY_PATH: "1",
        SUPERTUNER_MAX_CONCURRENT_RUNS: "3",
      },
    });
    expect(config.engineCmd).toEqual(["C:\\Program Files\\nodejs\\node.exe", "fake engine.mjs"]);
    expect(config.port).toBe(9999);
    expect(config.allowAnyPath).toBe(true);
    expect(config.maxConcurrentRuns).toBe(3);
  });

  it("rejects invalid integers and booleans", () => {
    const env = { SUPERTUNER_HOME: mkdtempSync(join(tmpdir(), "supertuner-config-")) };
    expect(() => loadServerConfig({ env: { ...env, SUPERTUNER_PORT: "not-a-port" } })).toThrow(/SUPERTUNER_PORT/);
    expect(() => loadServerConfig({ env: { ...env, SUPERTUNER_ALLOW_ANY_PATH: "maybe" } })).toThrow(
      /SUPERTUNER_ALLOW_ANY_PATH/,
    );
    expect(() => loadServerConfig({ env: { ...env, SUPERTUNER_MAX_CONCURRENT_RUNS: "0" } })).toThrow(
      /SUPERTUNER_MAX_CONCURRENT_RUNS/,
    );
  });

  it("allows programmatic overrides for tests", () => {
    const home = mkdtempSync(join(tmpdir(), "supertuner-config-"));
    const config = loadServerConfig({ env: { SUPERTUNER_HOME: home }, overrides: { port: 0, engineCmd: ["node", "x"] } });
    expect(config.port).toBe(0);
    expect(config.engineCmd).toEqual(["node", "x"]);
  });
});