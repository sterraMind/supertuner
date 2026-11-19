import type { RunPlan } from "@supertuner/core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeServer, makeSpec, type TestServer } from "./helpers.js";

let server: TestServer;

beforeAll(async () => {
  server = await makeServer();
});

afterAll(async () => {
  if (server) await server.built.close();
});

function plan(body: { spec: unknown }): Promise<{ statusCode: number; body: string }> {
  return server.built.app.inject({ method: "POST", url: "/api/plan", payload: body }).then((response) => ({
    statusCode: response.statusCode,
    body: response.body,
  }));
}

describe("POST /api/plan", () => {
  it("returns the engine plan for a feasible spec", async () => {
    const response = await server.built.app.inject({
      method: "POST",
      url: "/api/plan",
      payload: { spec: makeSpec("plan-ok") },
    });
    expect(response.statusCode).toBe(200);
    const body = response.json() as RunPlan;
    expect(body.feasible).toBe(true);
    expect(body.execution.backend).toBe("fake-qwen3-0.6b");
    expect(body.memory.tokens_per_step).toBe(2048);
    expect(body.issues).toEqual([]);
  });

  it("returns the infeasible plan with error issues", async () => {
    const response = await plan({ spec: makeSpec("infeasible-plan") });
    expect(response.statusCode).toBe(200);
    const body = JSON.parse(response.body) as RunPlan;
    expect(body.feasible).toBe(false);
    expect(body.issues[0]?.severity).toBe("error");
    expect(body.issues[0]?.code).toBe("vram_exceeded");
  });

  it("maps a refused spec to 422 with issues", async () => {
    const response = await plan({ spec: makeSpec("refuse-plan") });
    expect(response.statusCode).toBe(422);
    const body = JSON.parse(response.body) as { error: string; issues: Array<{ code: string }> };
    expect(body.error).toMatch(/refused/);
    expect(body.issues[0]?.code).toBe("unknown_family");
  });

  it("maps non-JSON engine output to 502", async () => {
    const response = await plan({ spec: makeSpec("badjson-plan") });
    expect(response.statusCode).toBe(502);
    expect((JSON.parse(response.body) as { error: string }).error).toMatch(/invalid JSON/);
  });

  it("maps an engine crash (exit 3) to 502", async () => {
    const response = await plan({ spec: makeSpec("crash-plan") });
    expect(response.statusCode).toBe(502);
    expect((JSON.parse(response.body) as { error: string }).error).toMatch(/crashed/);
  });

  it("rejects a malformed body with 400 and field issues", async () => {
    const response = await plan({ spec: { config: { name: "" } } });
    expect(response.statusCode).toBe(400);
    const body = JSON.parse(response.body) as { error: string; issues: Array<{ field: string | null }> };
    expect(body.error).toBe("invalid request body");
    expect(body.issues.length).toBeGreaterThan(0);
  });

  it("rejects relative dataset paths without invoking the engine", async () => {
    const spec = makeSpec("relative-path", {}, { path: "data/train.jsonl" });
    const response = await plan({ spec });
    expect(response.statusCode).toBe(422);
    const body = JSON.parse(response.body) as { issues: Array<{ code: string; field: string | null }> };
    expect(body.issues[0]?.code).toBe("path.relative");
    expect(body.issues[0]?.field).toBe("dataset.path");
  });
});