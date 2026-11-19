import { existsSync } from "node:fs";
import { join } from "node:path";
import { RECIPES, type RunPlan, type SystemResponse } from "@supertuner/core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeServer, makeSpec, type TestServer } from "./helpers.js";

let server: TestServer;

beforeAll(async () => {
  server = await makeServer();
});

afterAll(async () => {
  if (server) await server.built.close();
});

describe("health and probes", () => {
  it("GET /api/health does not require the engine", async () => {
    const response = await server.built.app.inject({ method: "GET", url: "/api/health" });
    expect(response.statusCode).toBe(200);
    const body = response.json() as { ok: boolean; version: string };
    expect(body.ok).toBe(true);
    expect(body.version).toMatch(/^\d+\.\d+\.\d+/);
  });

  it("GET /api/system merges the engine env report and paths", async () => {
    const response = await server.built.app.inject({ method: "GET", url: "/api/system" });
    expect(response.statusCode).toBe(200);
    const body = response.json() as SystemResponse;
    expect(body.env.python).toBe("3.12.0 (fake)");
    expect(body.engine.version).toBe("0.0.0-fake");
    expect(body.home).toBe(server.home);
    expect(body.runs_dir).toBe(server.built.config.runsDir);
    expect(body.datasets_dir).toBe(server.built.config.datasetsDir);
  });

  it("GET /api/plugins returns the engine plugin manifests", async () => {
    const response = await server.built.app.inject({ method: "GET", url: "/api/plugins" });
    expect(response.statusCode).toBe(200);
    const body = response.json() as { plugins: Array<{ id: string }>; errors: unknown[] };
    expect(body.plugins.map((plugin) => plugin.id)).toEqual(["fake.plugin"]);
    expect(body.errors).toEqual([]);
  });

  it("GET /api/families returns llm and video families", async () => {
    const response = await server.built.app.inject({ method: "GET", url: "/api/families" });
    expect(response.statusCode).toBe(200);
    const body = response.json() as { llm: Array<{ key: string }>; video: Array<{ key: string }> };
    expect(body.llm[0]?.key).toBe("fake-llm");
    expect(body.video[0]?.key).toBe("fake-video");
  });

  it("GET /api/recipes serves the curated list from core verbatim", async () => {
    const response = await server.built.app.inject({ method: "GET", url: "/api/recipes" });
    expect(response.statusCode).toBe(200);
    const body = response.json() as { recipes: Array<{ id: string }> };
    expect(body.recipes.map((recipe) => recipe.id)).toEqual(RECIPES.map((recipe) => recipe.id));
  });

  it("unknown routes return the {error} shape", async () => {
    const response = await server.built.app.inject({ method: "GET", url: "/api/nope" });
    expect(response.statusCode).toBe(404);
    expect((response.json() as { error: string }).error).toContain("not found");
  });
});

describe("engine unavailable", () => {
  let broken: TestServer;

  beforeAll(async () => {
    broken = await makeServer({ overrides: { engineCmd: ["__supertuner_engine_that_does_not_exist__"] } });
  });

  afterAll(async () => {
    if (broken) await broken.built.close();
  });

  it("health still answers, probes report 503", async () => {
    const health = await broken.built.app.inject({ method: "GET", url: "/api/health" });
    expect(health.statusCode).toBe(200);

    const system = await broken.built.app.inject({ method: "GET", url: "/api/system" });
    expect(system.statusCode).toBe(503);
    expect((system.json() as { error: string }).error).toContain("cannot launch engine");
  });

  it("POST /api/plan reports 503 instead of crashing", async () => {
    const response = await broken.built.app.inject({
      method: "POST",
      url: "/api/plan",
      payload: { spec: makeSpec("broken-engine") },
    });
    expect(response.statusCode).toBe(503);
    expect((response.json() as { error: string }).error).toContain("cannot launch engine");
  });
});

describe("state layout", () => {
  it("a fresh home has an empty runs directory", async () => {
    expect(existsSync(server.built.config.runsDir)).toBe(true);
    expect(existsSync(join(server.home, "supertuner.db"))).toBe(true);
  });
});