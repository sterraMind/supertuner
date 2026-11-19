import { existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { RunPlan, RunRecord } from "@supertuner/core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeServer, makeSpec, waitFor, type TestServer } from "./helpers.js";

let server: TestServer;

beforeAll(async () => {
  server = await makeServer();
});

afterAll(async () => {
  if (server) await server.built.close();
});

async function createRun(spec: unknown): Promise<{ statusCode: number; record: RunRecord }> {
  const response = await server.built.app.inject({ method: "POST", url: "/api/runs", payload: { spec } });
  return { statusCode: response.statusCode, record: response.json() as RunRecord };
}

describe("POST /api/runs", () => {
  it("persists, queues and completes a happy-path run", async () => {
    const created = await createRun(makeSpec("happy-run"));
    expect(created.statusCode).toBe(201);
    expect(["queued", "running"]).toContain(created.record.status);
    expect(created.record.plan?.feasible).toBe(true);

    const row = await waitFor(
      () => server.built.store.getRun(created.record.id),
      (value) => value?.status === "completed",
    );
    expect(row).toBeTruthy();
    expect(row?.exit_code).toBe(0);
    expect(row?.steps).toBe(3);
    expect(row?.last_step).toBe(3);
    expect(row?.pid).toBeNull();
    expect(row?.finished_at).not.toBeNull();
    expect(JSON.parse(row?.artifacts_json ?? "[]")).toContain("adapter.safetensors");

    const detail = await server.built.app.inject({ method: "GET", url: `/api/runs/${created.record.id}` });
    expect(detail.statusCode).toBe(200);
    const record = detail.json() as RunRecord;
    expect(record.status).toBe("completed");
    expect(record.exit_code).toBe(0);
    expect(record.artifacts).toContain("adapter.safetensors");
    expect(record.headline_mib).toBeGreaterThan(0);

    const events = await server.built.app.inject({ method: "GET", url: `/api/runs/${created.record.id}/events` });
    expect(events.statusCode).toBe(200);
    const list = events.json() as Array<{ type: string; status?: string; step?: number }>;
    expect(list[0]?.type).toBe("run_start");
    expect(list.some((event) => event.type === "step")).toBe(true);
    expect(list[list.length - 1]?.type).toBe("run_end");
    expect(list[list.length - 1]?.status).toBe("completed");

    const artifacts = await server.built.app.inject({ method: "GET", url: `/api/runs/${created.record.id}/artifacts` });
    expect(artifacts.statusCode).toBe(200);
    const files = (artifacts.json() as { artifacts: Array<{ path: string; bytes: number }> }).artifacts;
    expect(files.map((file) => file.path)).toContain("adapter.safetensors");
    expect(files[0]?.bytes).toBeGreaterThan(0);

    const runDir = join(server.built.config.runsDir, created.record.id);
    expect(existsSync(join(runDir, "spec.json"))).toBe(true);
    expect(existsSync(join(runDir, "plan.json"))).toBe(true);
    expect(existsSync(join(runDir, "events.jsonl"))).toBe(true);
    expect(existsSync(join(runDir, "logs", "engine.stderr.log"))).toBe(true);
    const specOnDisk = JSON.parse(readFileSync(join(runDir, "spec.json"), "utf8")) as { config: { name: string } };
    expect(specOnDisk.config.name).toBe("happy-run");

    const listResponse = await server.built.app.inject({ method: "GET", url: "/api/runs?limit=5" });
    expect(listResponse.statusCode).toBe(200);
    expect((listResponse.json() as { runs: RunRecord[] }).runs.map((run) => run.id)).toContain(created.record.id);
  });

  it("returns 422 with the compiled plan when infeasible, without creating a run", async () => {
    const before = server.built.store.listRuns(500).length;
    const response = await server.built.app.inject({
      method: "POST",
      url: "/api/runs",
      payload: { spec: makeSpec("infeasible-run") },
    });
    expect(response.statusCode).toBe(422);
    const body = response.json() as { error: string; issues: Array<{ code: string; severity: string }>; plan: RunPlan };
    expect(body.error).toMatch(/infeasible/);
    expect(body.plan.feasible).toBe(false);
    expect(body.issues.some((issue) => issue.severity === "error")).toBe(true);
    expect(server.built.store.listRuns(500).length).toBe(before);
  });

  it("records the engine's failure reason (exit 3 with run_end)", async () => {
    const created = await createRun(makeSpec("fail-run"));
    expect(created.statusCode).toBe(201);
    const row = await waitFor(
      () => server.built.store.getRun(created.record.id),
      (value) => value?.status === "failed",
    );
    expect(row?.exit_code).toBe(3);
    expect(row?.error).toContain("fake dataset vanished");
    const events = server.built.store.getRun(created.record.id);
    expect(events?.finished_at).not.toBeNull();
  });

  it("falls back to the stderr tail when the engine crashes without run_end", async () => {
    const created = await createRun(makeSpec("crash-run"));
    expect(created.statusCode).toBe(201);
    const row = await waitFor(
      () => server.built.store.getRun(created.record.id),
      (value) => value?.status === "failed",
    );
    expect(row?.exit_code).toBe(3);
    expect(row?.error).toMatch(/fake training crash/);
  });

  it("treats exit 2 as a refused run and keeps the engine's reason", async () => {
    const created = await createRun(makeSpec("refuse-run"));
    expect(created.statusCode).toBe(201);
    const row = await waitFor(
      () => server.built.store.getRun(created.record.id),
      (value) => value?.status === "failed",
    );
    expect(row?.exit_code).toBe(2);
    expect(row?.error).toContain("engine refused the run");
    expect(row?.error).toContain("plan is infeasible on this budget");
  });

  it("persists unknown event types and quarantines malformed stdout lines", async () => {
    const created = await createRun(makeSpec("noisy-run"));
    expect(created.statusCode).toBe(201);
    await waitFor(
      () => server.built.store.getRun(created.record.id),
      (value) => value?.status === "completed",
    );

    const events = await server.built.app.inject({ method: "GET", url: `/api/runs/${created.record.id}/events` });
    const list = events.json() as Array<{ type: string }>;
    expect(list.some((event) => event.type === "phase_start")).toBe(true);
    expect(list.some((event) => event.type === "phase_end")).toBe(true);

    const stdoutLog = readFileSync(join(server.built.config.runsDir, created.record.id, "logs", "engine.stdout.log"), "utf8");
    expect(stdoutLog).toContain("[protocol]");
    expect(stdoutLog).toContain("this line is not an event");
  });

  it("rejects a malformed spec body with 400", async () => {
    const response = await server.built.app.inject({ method: "POST", url: "/api/runs", payload: {} });
    expect(response.statusCode).toBe(400);
    expect((response.json() as { error: string }).error).toBe("invalid request body");
  });

  it("rejects an unknown run id with 404 on detail/events/cancel/artifacts", async () => {
    for (const url of ["/api/runs/missing/events", "/api/runs/missing/artifacts"]) {
      const response = await server.built.app.inject({ method: "GET", url });
      expect(response.statusCode).toBe(404);
    }
    const cancel = await server.built.app.inject({ method: "POST", url: "/api/runs/missing/cancel" });
    expect(cancel.statusCode).toBe(404);
  });
});

describe("spec path enforcement", () => {
  it("rejects a relative output dir", async () => {
    const response = await server.built.app.inject({
      method: "POST",
      url: "/api/runs",
      payload: { spec: makeSpec("relative-output", {}, null, { dir: "checkpoints/out" }) },
    });
    expect(response.statusCode).toBe(422);
    expect((response.json() as { issues: Array<{ code: string }> }).issues[0]?.code).toBe("path.relative");
  });

  it("rejects paths outside the allowed roots by default", async () => {
    const outside = join(tmpdir(), "supertuner-outside", "data.jsonl");
    const response = await server.built.app.inject({
      method: "POST",
      url: "/api/runs",
      payload: { spec: makeSpec("outside-path", {}, { path: outside }) },
    });
    expect(response.statusCode).toBe(422);
    const body = response.json() as { issues: Array<{ code: string; hint: string }> };
    expect(body.issues[0]?.code).toBe("path.outside_roots");
    expect(body.issues[0]?.hint).toContain("SUPERTUNER_ALLOW_ANY_PATH");
  });

  it("accepts outside paths when SUPERTUNER_ALLOW_ANY_PATH is on", async () => {
    const permissive = await makeServer({ overrides: { allowAnyPath: true } });
    try {
      const outside = join(tmpdir(), "supertuner-outside", "data.jsonl");
      const response = await permissive.built.app.inject({
        method: "POST",
        url: "/api/runs",
        payload: { spec: makeSpec("outside-allowed", {}, { path: outside }) },
      });
      expect(response.statusCode).toBe(201);
      const record = response.json() as RunRecord;
      await waitFor(
        () => permissive.built.store.getRun(record.id),
        (value) => value?.status === "completed",
      );
    } finally {
      await permissive.built.close();
    }
  });
});