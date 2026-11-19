import { join } from "node:path";
import { readFile } from "node:fs/promises";
import type { RunRecord } from "@supertuner/core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeServer, makeSpec, waitFor, type TestServer } from "./helpers.js";

let server: TestServer;

beforeAll(async () => {
  server = await makeServer({ overrides: { maxConcurrentRuns: 1, cancelGraceMs: 2_000 } });
});

afterAll(async () => {
  if (server) await server.built.close();
});

async function createRun(name: string): Promise<RunRecord> {
  const response = await server.built.app.inject({ method: "POST", url: "/api/runs", payload: { spec: makeSpec(name) } });
  expect(response.statusCode).toBe(201);
  return response.json() as RunRecord;
}

async function cancel(id: string): Promise<{ statusCode: number; body: RunRecord | { error: string } }> {
  const response = await server.built.app.inject({ method: "POST", url: `/api/runs/${id}/cancel` });
  return { statusCode: response.statusCode, body: response.json() as RunRecord | { error: string } };
}

describe("POST /api/runs/:id/cancel", () => {
  it("stops a running run and records the engine's cancelled run_end", async () => {
    const record = await createRun("slow-cancel");
    await waitFor(
      () => server.built.store.getRun(record.id),
      (row) => row?.status === "running",
    );

    const response = await cancel(record.id);
    expect(response.statusCode).toBe(200);

    const row = await waitFor(
      () => server.built.store.getRun(record.id),
      (value) => value?.status === "cancelled",
      15_000,
    );
    expect(row?.error).toBeNull();
    expect(row?.exit_code).toBe(130);
    expect(row?.pid).toBeNull();

    const raw = await readFile(join(server.built.config.runsDir, record.id, "events.jsonl"), "utf8");
    const events = raw
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line) as { type: string; status?: string });
    const end = events.filter((event) => event.type === "run_end").at(-1);
    expect(end?.status).toBe("cancelled");
  });

  it("drops a queued run without starting it", async () => {
    const first = await createRun("slow-queue-a");
    await waitFor(
      () => server.built.store.getRun(first.id),
      (row) => row?.status === "running",
    );

    const second = await createRun("slow-queue-b");
    expect(server.built.store.getRun(second.id)?.status).toBe("queued");

    const response = await cancel(second.id);
    expect(response.statusCode).toBe(200);
    const row = server.built.store.getRun(second.id);
    expect(row?.status).toBe("cancelled");
    expect(row?.error).toBe("cancelled before the run started");
    expect(row?.started_at).toBeNull();

    await cancel(first.id);
    await waitFor(
      () => server.built.store.getRun(first.id),
      (value) => value?.status === "cancelled",
    );
  });

  it("rejects cancelling an already finished run with 409", async () => {
    const record = await createRun("cancel-finished");
    await waitFor(
      () => server.built.store.getRun(record.id),
      (value) => value?.status === "completed",
    );
    const response = await cancel(record.id);
    expect(response.statusCode).toBe(409);
    expect((response.body as { error: string }).error).toContain("not queued or running");
  });

  it("returns 404 for an unknown run", async () => {
    const response = await cancel("does-not-exist");
    expect(response.statusCode).toBe(404);
  });
});