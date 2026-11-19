import { join } from "node:path";
import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { makeServer, makeSpec } from "./helpers.js";

describe("startup recovery", () => {
  it("marks abandoned queued/running runs failed and leaves finished runs alone", async () => {
    const first = await makeServer();
    try {
      first.built.store.insertRun({
        id: "interrupted-running",
        name: "interrupted",
        status: "queued",
        spec: makeSpec("recover"),
      });
      first.built.store.updateRun("interrupted-running", {
        status: "running",
        pid: 4242,
        started_at: new Date().toISOString(),
      });
      first.built.store.insertRun({
        id: "interrupted-queued",
        name: "interrupted-queued",
        status: "queued",
        spec: makeSpec("recover-queued"),
      });
      first.built.store.insertRun({
        id: "already-completed",
        name: "done",
        status: "queued",
        spec: makeSpec("recover-done"),
      });
      first.built.store.setStatus("already-completed", "completed", {
        exitCode: 0,
        finishedAt: new Date().toISOString(),
      });
    } finally {
      await first.built.close();
    }

    expect(existsSync(join(first.home, "supertuner.db"))).toBe(true);

    const second = await makeServer({ home: first.home });
    try {
      const running = second.built.store.getRun("interrupted-running");
      expect(running?.status).toBe("failed");
      expect(running?.error).toBe("server restarted");
      expect(running?.finished_at).not.toBeNull();
      expect(running?.pid).toBeNull();

      expect(second.built.store.getRun("interrupted-queued")?.status).toBe("failed");
      expect(second.built.store.getRun("already-completed")?.status).toBe("completed");
      expect(second.built.store.getRun("already-completed")?.error).toBeNull();
    } finally {
      await second.built.close();
    }

    // A second restart must not rewrite the failure or lose the row.
    const third = await makeServer({ home: first.home });
    try {
      const row = third.built.store.getRun("interrupted-running");
      expect(row?.status).toBe("failed");
      expect(row?.error).toBe("server restarted");
    } finally {
      await third.built.close();
    }
  });
});