/** Shared test scaffolding: temp homes, fake engine config, polling helpers. */

import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { RunSpecSchema, type DatasetSpec, type RunConfig, type RunSpec } from "@supertuner/core";
import { loadServerConfig, type ServerConfig } from "../src/config.js";
import { buildServer, type BuiltServer } from "../src/index.js";
import type { LogSink } from "../src/supervisor.js";

const fakeEnginePath = fileURLToPath(new URL("./fixtures/fake-engine.mjs", import.meta.url));

export interface TestServer {
  built: BuiltServer;
  home: string;
  /** Set only when `listen: true`. */
  baseUrl: string | null;
}

export interface MakeServerOptions {
  /** Reuse an existing home (startup recovery tests). */
  home?: string;
  overrides?: Partial<ServerConfig>;
  env?: NodeJS.ProcessEnv;
  listen?: boolean;
  logger?: boolean | LogSink;
}

export async function makeServer(options: MakeServerOptions = {}): Promise<TestServer> {
  const home = options.home ?? (await mkdtemp(join(tmpdir(), "supertuner-test-")));
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    ...options.env,
    SUPERTUNER_HOME: home,
  };
  const config = loadServerConfig({
    env,
    overrides: {
      engineCmd: [process.execPath, fakeEnginePath],
      port: 0,
      compileTimeoutMs: 20_000,
      probeTimeoutMs: 10_000,
      cancelGraceMs: 1_500,
      ...options.overrides,
    },
  });

  const built = await buildServer({ config, logger: options.logger });
  let baseUrl: string | null = null;
  if (options.listen) {
    await built.app.listen({ host: "127.0.0.1", port: 0 });
    const address = built.app.server.address();
    if (address === null || typeof address === "string") throw new Error("server did not bind a TCP port");
    baseUrl = `http://127.0.0.1:${address.port}`;
  }
  return { built, home, baseUrl };
}

export function requireUrl(server: TestServer): string {
  if (!server.baseUrl) throw new Error("test server was created without listen: true");
  return server.baseUrl;
}

export function makeSpec(
  name: string,
  config: Partial<RunConfig> = {},
  dataset: Partial<DatasetSpec> | null = null,
  output: { dir?: string | null } = {},
): RunSpec {
  return RunSpecSchema.parse({
    config: { name, family: "qwen3-0.6b", stage: "sft", ...config },
    dataset: dataset ? { kind: "text", format: "jsonl", ...dataset } : null,
    train: {},
    output,
    backend: null,
  });
}

export async function waitFor<T>(
  probe: () => T | Promise<T>,
  predicate: (value: T) => boolean,
  timeoutMs = 20_000,
  intervalMs = 40,
): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  let last: T | undefined;
  while (Date.now() < deadline) {
    last = await probe();
    if (predicate(last)) return last;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`condition not met within ${timeoutMs} ms; last value: ${JSON.stringify(last)}`);
}

/** Run a fake engine spec to a terminal status and return the record. */
export async function awaitStatus(server: TestServer, runId: string, statuses: string[]): Promise<Record<string, unknown>> {
  const record = await waitFor(
    () => {
      const row = server.built.store.getRun(runId);
      if (!row) throw new Error(`run ${runId} disappeared`);
      return row;
    },
    (row) => statuses.includes(row.status),
  );
  return record as unknown as Record<string, unknown>;
}

/** Read an SSE body until `predicate` is true; returns the parsed `data:` payloads. */
export async function readSse(
  url: string,
  predicate: (events: unknown[]) => boolean,
  timeoutMs = 15_000,
): Promise<unknown[]> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  timeout.unref?.();
  const events: unknown[] = [];
  let buffer = "";

  try {
    const response = await fetch(url, { signal: controller.signal, headers: { accept: "text/event-stream" } });
    if (!response.ok || !response.body) throw new Error(`SSE request failed: ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (!predicate(events)) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          events.push(JSON.parse(line.slice("data: ".length)) as unknown);
        }
        boundary = buffer.indexOf("\n\n");
      }
    }
    await reader.cancel().catch(() => undefined);
    return events;
  } finally {
    clearTimeout(timeout);
    controller.abort();
  }
}