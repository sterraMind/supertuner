/**
 * HTTP surface. All routes from ARCHITECTURE.md §9 live here.
 *
 * Two rules shape this file:
 *  - the server re-compiles every run spec itself (a client plan is never
 *    trusted), and
 *  - spec paths are resolved and checked against configured roots before any
 *    engine process sees them.
 */

import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile, statfs } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import fastifyStatic from "@fastify/static";
import {
  RECIPES,
  RunSpecSchema,
  SystemResponseSchema,
  calibrationKey,
  type Issue,
  type RunSpec,
} from "@supertuner/core";
import type { FastifyError, FastifyInstance, FastifyReply } from "fastify";
import { z } from "zod";
import { listArtifacts } from "./artifacts.js";
import { expandUserPath,isPathInside, type ServerConfig } from "./config.js";
import { toRunRecord, type RunStore } from "./db.js";
import { EngineError, type EngineClient } from "./engine.js";
import type { Supervisor } from "./supervisor.js";

const SpecBodySchema = z.object({ spec: RunSpecSchema }).strict();

/** Free/total space of the volume holding SUPERTUNER_HOME, in MiB. */
async function diskInfo(path: string): Promise<{ path: string; total_mib: number; free_mib: number }> {
  try {
    const stats = await statfs(path, { bigint: true });
    const toMib = (blocks: bigint) => Math.round((Number(blocks) * Number(stats.bsize)) / 1_048_576);
    return { path, total_mib: toMib(stats.blocks), free_mib: toMib(stats.bavail) };
  } catch {
    return { path, total_mib: 0, free_mib: 0 };
  }
}
const ListQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(200).optional(),
});
const IdParamsSchema = z.object({ id: z.string().min(1) });

export interface RouteDeps {
  config: ServerConfig;
  store: RunStore;
  engine: EngineClient;
  supervisor: Supervisor;
  serverVersion: string;
}

interface PathCheck {
  spec: RunSpec;
  issues: Issue[];
}

/**
 * Expand `~`, reject relative paths, and confine dataset/output/checkpoint
 * paths to the configured roots unless `SUPERTUNER_ALLOW_ANY_PATH=1`.
 */
export function normalizeSpecPaths(spec: RunSpec, config: ServerConfig): PathCheck {
  const next: RunSpec = {
    ...spec,
    dataset: spec.dataset ? { ...spec.dataset } : null,
    train: { ...spec.train },
    output: { ...spec.output },
  };
  const issues: Issue[] = [];

  const check = (raw: string | null, field: string, assign: (resolved: string) => void): void => {
    if (raw === null) return;
    const expanded = expandUserPath(raw);
    if (!isAbsolute(expanded)) {
      issues.push({
        severity: "error",
        code: "path.relative",
        message: `${field} must be an absolute path (after ~ expansion), got ${JSON.stringify(raw)}`,
        hint: config.allowAnyPath
          ? "use an absolute path such as D:/datasets/demo"
          : `use an absolute path under one of: ${config.allowedRoots.join(", ")}`,
        field,
      });
      return;
    }
    const resolved = resolve(expanded);
    if (!config.allowAnyPath && !config.allowedRoots.some((root) => isPathInside(root, resolved))) {
      issues.push({
        severity: "error",
        code: "path.outside_roots",
        message: `${field} points outside the configured roots: ${resolved}`,
        hint: "set SUPERTUNER_ALLOW_ANY_PATH=1 to permit arbitrary locations, or add a root to SUPERTUNER_ALLOWED_ROOTS",
        field,
      });
      return;
    }
    assign(resolved);
  };

  if (next.dataset) {
    check(next.dataset.path, "dataset.path", (resolved) => {
      if (next.dataset) next.dataset.path = resolved;
    });
  }
  check(next.output.dir, "output.dir", (resolved) => {
    next.output.dir = resolved;
  });
  check(next.train.resume_from, "train.resume_from", (resolved) => {
    next.train.resume_from = resolved;
  });

  return { spec: next, issues };
}

/**
 * 跑后自校准（v0.2 T1）：规划前按校准键查 calibrations 表，命中则把聚合偏差注入
 * spec.calibration，让 engine 只读标注；不改动任何预算字段。返回新 spec。
 */
export function withCalibration(store: RunStore, spec: RunSpec): RunSpec {
  const key = calibrationKey({
    family: spec.config.family,
    stage: spec.config.stage,
    adapter: spec.config.adapter,
    quant_base: spec.config.quant_base,
    blocks_to_swap: spec.config.blocks_to_swap,
  });
  const row = store.getCalibration(key);
  const calibration = row ? { count: row.count, avg_delta_pct: row.avg_delta_pct } : null;
  return { ...spec, calibration };
}

function zodIssues(error: z.ZodError): Issue[] {
  return error.issues.map((issue) => ({
    severity: "error",
    code: "invalid_body",
    message: `${issue.path.join(".") || "<root>"}: ${issue.message}`,
    hint: "",
    field: issue.path.length > 0 ? issue.path.join(".") : null,
  }));
}

function sendEngineError(reply: FastifyReply, cause: unknown): FastifyReply {
  if (!(cause instanceof EngineError)) {
    throw cause;
  }
  const issues = cause.detail.issues ?? [];
  switch (cause.kind) {
    case "refused":
      return reply.code(422).send({ error: cause.message, issues });
    case "timeout":
      return reply.code(504).send({ error: cause.message });
    case "unavailable":
      return reply.code(503).send({ error: cause.message });
    case "unsupported":
      return reply.code(503).send({ error: cause.message });
    default:
      return reply.code(502).send({ error: cause.message, issues: issues.length > 0 ? issues : undefined });
  }
}

export async function registerRoutes(app: FastifyInstance, deps: RouteDeps): Promise<void> {
  const { config, store, engine, supervisor, serverVersion } = deps;

  // Dev-only CORS shim for the Vite server; the API itself is loopback-bound.
  app.addHook("onRequest", async (request, reply) => {
    const origin = request.headers.origin;
    if (origin && config.devOrigins.includes(origin)) {
      reply.header("access-control-allow-origin", origin);
      reply.header("vary", "Origin");
      reply.header("access-control-allow-methods", "GET,POST,OPTIONS");
      reply.header("access-control-allow-headers", "content-type");
      if (request.method === "OPTIONS") {
        reply.code(204).send();
      }
    }
  });

  app.setErrorHandler((error: FastifyError, request, reply) => {
    request.log.error({ err: error }, "request failed");
    const status = typeof error.statusCode === "number" && error.statusCode >= 400 ? error.statusCode : 500;
    reply.code(status).send({ error: error.message });
  });

  app.get("/api/health", async () => ({ ok: true, version: serverVersion }));

  app.get("/api/system", async (_request, reply) => {
    try {
      const [env, engineVersion] = await Promise.all([engine.env(), engine.version()]);
      const python = /^python/i.test(join(config.engineCmd[0] ?? "")) ? config.engineCmd[0] : null;
      const body = SystemResponseSchema.parse({
        env,
        engine: { version: engineVersion, python, path: config.engineCmd.join(" ") },
        server: { version: serverVersion },
        home: config.home,
        runs_dir: config.runsDir,
        models_dir: config.modelsDir,
        datasets_dir: config.datasetsDir,
        disk: await diskInfo(config.home),
      });
      return reply.send(body);
    } catch (cause) {
      return sendEngineError(reply, cause);
    }
  });

  app.get("/api/families", async (_request, reply) => {
    try {
      return reply.send(await engine.families());
    } catch (cause) {
      return sendEngineError(reply, cause);
    }
  });

  app.get("/api/plugins", async (_request, reply) => {
    try {
      return reply.send(await engine.plugins());
    } catch (cause) {
      return sendEngineError(reply, cause);
    }
  });

  app.get("/api/recipes", async (_request, reply) => reply.send({ recipes: RECIPES }));

  app.post("/api/plan", async (request, reply) => {
    const parsed = SpecBodySchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.code(400).send({ error: "invalid request body", issues: zodIssues(parsed.error) });
    }
    const { spec, issues } = normalizeSpecPaths(parsed.data.spec, config);
    if (issues.length > 0) {
      return reply.code(422).send({ error: "spec paths are not allowed", issues });
    }
    try {
      // 跑后自校准（v0.2 T1）：规划前注入历史偏差标注，engine 只读它写备注。
      const { plan } = await engine.compile(withCalibration(store, spec));
      return reply.send(plan);
    } catch (cause) {
      return sendEngineError(reply, cause);
    }
  });

  app.post("/api/runs", async (request, reply) => {
    const parsed = SpecBodySchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.code(400).send({ error: "invalid request body", issues: zodIssues(parsed.error) });
    }
    const { spec, issues: pathIssues } = normalizeSpecPaths(parsed.data.spec, config);
    if (pathIssues.length > 0) {
      return reply.code(422).send({ error: "spec paths are not allowed", issues: pathIssues });
    }

    // 跑后自校准（v0.2 T1）：规划前按校准键查库并注入历史偏差标注；
    // 编译、入库、入队统一用这份 spec，保证落盘的 spec/plan 与实际规划一致。
    const runSpec = withCalibration(store, spec);

    if (runSpec.config.family === "minimax-h3" && !runSpec.licence_ack) {
      return reply.code(422).send({
        error: "MiniMax H3 requires an explicit licence acknowledgement",
        issues: [
          {
            severity: "error",
            code: "E_LICENCE_ACK_REQUIRED",
            message:
              "MiniMax H3 使用 MiniMax H3 Community License，不是 Apache-2.0：地域排除 EU/UK/韩/美，年收入超过 2000 万美元需书面授权，且禁止用其输出训练其它模型。",
            hint: "在请求里带上 licence_ack={family:'minimax-h3', license_id:'MiniMax H3 Community License', version:'1', accepted_at:<ISO 时间>}",
            field: "licence_ack",
          },
        ],
      });
    }

    let plan;
    try {
      ({ plan } = await engine.compile(runSpec));
    } catch (cause) {
      return sendEngineError(reply, cause);
    }

    const blocking = [...plan.issues, ...plan.execution.issues].filter((issue) => issue.severity === "error");
    if (!plan.feasible || blocking.length > 0) {
      return reply.code(422).send({
        error: "run is infeasible on the requested hardware budget",
        issues: blocking.length > 0 ? blocking : plan.issues,
        plan,
      });
    }

    const id = randomUUID();
    try {
      store.insertRun({ id, name: runSpec.config.name, spec: runSpec, plan, backend: plan.execution.backend });
      supervisor.enqueue({ id, spec: runSpec, plan });
    } catch (cause) {
      // A run that never reached the queue must not sit in `queued` forever.
      try {
        store.setStatus(id, "failed", { error: `cannot enqueue run: ${(cause as Error).message}` });
      } catch {
        // the insert itself may have failed; nothing to update
      }
      return reply.code(503).send({ error: `cannot start run: ${(cause as Error).message}` });
    }

    const row = store.getRun(id);
    if (!row) return reply.code(500).send({ error: "run was inserted but cannot be read back" });
    return reply.code(201).send(toRunRecord(row));
  });

  app.get("/api/runs", async (request, reply) => {
    const parsed = ListQuerySchema.safeParse(request.query);
    if (!parsed.success) {
      return reply.code(400).send({ error: "invalid query", issues: zodIssues(parsed.error) });
    }
    const runs = store.listRuns(parsed.data.limit ?? 50).map(toRunRecord);
    return reply.send({ runs });
  });

  app.get("/api/runs/:id", async (request, reply) => {
    const params = IdParamsSchema.safeParse(request.params);
    if (!params.success) return reply.code(400).send({ error: "invalid run id" });
    const row = store.getRun(params.data.id);
    if (!row) return reply.code(404).send({ error: `run not found: ${params.data.id}` });
    return reply.send(toRunRecord(row));
  });

  app.get("/api/runs/:id/events", async (request, reply) => {
    const params = IdParamsSchema.safeParse(request.params);
    if (!params.success) return reply.code(400).send({ error: "invalid run id" });
    const row = store.getRun(params.data.id);
    if (!row) return reply.code(404).send({ error: `run not found: ${params.data.id}` });
    return reply.send(await readRunEvents(join(config.runsDir, row.id, "events.jsonl")));
  });

  app.get("/api/runs/:id/events/stream", async (request, reply) => {
    const params = IdParamsSchema.safeParse(request.params);
    if (!params.success) {
      return reply.code(400).send({ error: "invalid run id" });
    }
    const row = store.getRun(params.data.id);
    if (!row) {
      return reply.code(404).send({ error: `run not found: ${params.data.id}` });
    }

    const runId = row.id;
    reply.hijack();
    const raw = reply.raw;
    raw.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });

    let closed = false;
    let live = false;
    let heartbeat: NodeJS.Timeout | null = null;
    let unsubscribe: () => void = () => {};
    const buffered: unknown[] = [];
    const replayed = new Set<string>();

    const write = (payload: unknown): void => {
      if (closed) return;
      raw.write(`data: ${JSON.stringify(payload)}\n\n`);
    };

    const finish = (): void => {
      if (closed) return;
      closed = true;
      if (heartbeat) clearInterval(heartbeat);
      heartbeat = null;
      unsubscribe();
      raw.end();
    };

    unsubscribe = supervisor.subscribe(runId, {
      onEvent: (event) => {
        if (!live) {
          buffered.push(event);
          return;
        }
        write(event);
      },
      onEnd: () => {
        write({ type: "stream_end", run_id: runId });
        finish();
      },
    });

    raw.on("close", finish);
    raw.on("error", finish);

    raw.write(": supertuner run stream\n\n");
    for (const event of await readRunEvents(join(config.runsDir, runId, "events.jsonl"))) {
      const key = JSON.stringify(event);
      if (replayed.has(key)) continue;
      replayed.add(key);
      write(event);
    }
    live = true;
    for (const event of buffered.splice(0)) {
      const key = JSON.stringify(event);
      if (replayed.has(key)) continue;
      replayed.add(key);
      write(event);
    }

    // A client can attach after the run finished: replay, then close instead
    // of holding a stream that will never produce another event.
    const current = store.getRun(runId);
    if (!current || isTerminal(current.status)) {
      write({ type: "stream_end", run_id: runId });
      finish();
      return;
    }

    heartbeat = setInterval(() => {
      if (closed) return;
      raw.write(": ping\n\n");
    }, 15_000);
    heartbeat.unref?.();
  });

  app.post("/api/runs/:id/cancel", async (request, reply) => {
    const params = IdParamsSchema.safeParse(request.params);
    if (!params.success) return reply.code(400).send({ error: "invalid run id" });
    const outcome = supervisor.cancel(params.data.id);
    if (outcome === "not-found") return reply.code(404).send({ error: `run not found: ${params.data.id}` });
    if (outcome === "not-active") {
      const row = store.getRun(params.data.id);
      return reply.code(409).send({ error: `run is not queued or running (status ${row?.status ?? "unknown"})` });
    }
    const row = store.getRun(params.data.id);
    if (!row) return reply.code(500).send({ error: "run disappeared during cancellation" });
    return reply.send(toRunRecord(row));
  });

  app.get("/api/runs/:id/artifacts", async (request, reply) => {
    const params = IdParamsSchema.safeParse(request.params);
    if (!params.success) return reply.code(400).send({ error: "invalid run id" });
    const row = store.getRun(params.data.id);
    if (!row) return reply.code(404).send({ error: `run not found: ${params.data.id}` });
    return reply.send({ artifacts: listArtifacts(join(config.runsDir, row.id, "output")) });
  });

  app.post("/api/engine/reload", async (_request, reply) => {
    try {
      const plugins = await engine.reload();
      return reply.send({ ok: true, ...plugins });
    } catch (cause) {
      return sendEngineError(reply, cause);
    }
  });

  if (config.webDist && existsSync(join(config.webDist, "index.html"))) {
    await app.register(fastifyStatic, {
      root: config.webDist,
      index: ["index.html"],
      decorateReply: true,
    });
  }

  app.setNotFoundHandler((request, reply) => {
    if (config.webDist && request.method === "GET" && !request.url.startsWith("/api/")) {
      return reply.type("text/html; charset=utf-8").sendFile("index.html");
    }
    return reply.code(404).send({ error: `not found: ${request.method} ${request.url}` });
  });
}

function isTerminal(status: string): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

/** Parse a JSONL event file, skipping lines that are not JSON objects. */
async function readRunEvents(path: string): Promise<unknown[]> {
  let text: string;
  try {
    text = await readFile(path, "utf8");
  } catch {
    return [];
  }
  const events: unknown[] = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (typeof parsed === "object" && parsed !== null) events.push(parsed);
    } catch {
      // A partially written final line while a run is live is expected.
    }
  }
  return events;
}