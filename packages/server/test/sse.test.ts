import type { RunRecord } from "@supertuner/core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { awaitStatus, makeServer, makeSpec, readSse, requireUrl, type TestServer } from "./helpers.js";

let server: TestServer;

beforeAll(async () => {
  server = await makeServer({ listen: true });
});

afterAll(async () => {
  if (server) await server.built.close();
});

function typeOf(event: unknown): string | undefined {
  if (typeof event === "object" && event !== null && "type" in event) {
    const value = (event as { type?: unknown }).type;
    return typeof value === "string" ? value : undefined;
  }
  return undefined;
}

async function createRun(name: string): Promise<RunRecord> {
  const response = await fetch(`${requireUrl(server)}/api/runs`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ spec: makeSpec(name) }),
  });
  expect(response.status).toBe(201);
  return (await response.json()) as RunRecord;
}

describe("GET /api/runs/:id/events/stream", () => {
  it("replays finished runs and closes the stream", async () => {
    const record = await createRun("sse-finished");
    await awaitStatus(server, record.id, ["completed"]);

    const events = await readSse(
      `${requireUrl(server)}/api/runs/${record.id}/events/stream`,
      (seen) => seen.some((event) => typeOf(event) === "stream_end"),
    );
    const types = events.map(typeOf);
    expect(types[0]).toBe("run_start");
    expect(types).toContain("step");
    expect(types).toContain("run_end");
    expect(types[types.length - 1]).toBe("stream_end");
    const end = events.find((event) => typeOf(event) === "run_end") as { status: string };
    expect(end.status).toBe("completed");
  });

  it("delivers live events for a running run", async () => {
    const record = await createRun("slow-live");
    const events = await readSse(
      `${requireUrl(server)}/api/runs/${record.id}/events/stream`,
      (seen) => seen.some((event) => typeOf(event) === "run_end" || typeOf(event) === "stream_end"),
    );
    const types = events.map(typeOf);
    expect(types).toContain("run_start");
    expect(types.filter((type) => type === "step").length).toBeGreaterThanOrEqual(2);
    const end = events.find((event) => typeOf(event) === "run_end") as { status: string } | undefined;
    expect(end?.status).toBe("completed");
    // The stream stays live until the engine finishes, so the terminal event
    // for a completed run is the run_end emitted by the supervisor's fan-out.
    expect(types[types.length - 1]).toBe("run_end");
  });

  it("returns 404 for an unknown run", async () => {
    const response = await fetch(`${requireUrl(server)}/api/runs/missing/events/stream`);
    expect(response.status).toBe(404);
    expect((await response.json()) as { error: string }).toMatchObject({ error: expect.stringContaining("run not found") });
  });
});