/**
 * Run supervisor.
 *
 * Owns the in-memory queue, one engine child process per run, JSONL event
 * persistence, database counters, SSE fan-out and cancellation. It is the only
 * writer of `<run>/events.jsonl` and the only place that turns a process exit
 * into a terminal run status.
 */

import type { ChildProcess } from "node:child_process";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  parseEventLine,
  type RunEnd,
  type RunEvent,
  type RunPlan,
  type RunSpec,
  type RunStatus,
} from "@supertuner/core";
import { listArtifactPaths } from "./artifacts.js";
import type { ServerConfig } from "./config.js";
import type { RunStore } from "./db.js";
import type { EngineClient } from "./engine.js";
import { terminateChildTree, waitForExit } from "./process.js";

export interface LogSink {
  debug(message: string): void;
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
}

export interface SubscriberHandlers {
  /** One parsed event object, exactly as written to `events.jsonl`. */
  onEvent: (event: unknown) => void;
  /** The run reached a terminal state or the server is shutting down. */
  onEnd: () => void;
}

export type CancelOutcome = "cancelled" | "not-active" | "not-found";

export interface EnqueuedRun {
  id: string;
  spec: RunSpec;
  plan: RunPlan | null;
}

interface ActiveRun {
  child: ChildProcess;
  runDir: string;
  cancelled: boolean;
  stderrTail: string;
  finalized: boolean;
  terminating: Promise<void> | null;
  endEvent: RunEnd | null;
}

const MAX_PENDING_LINE_BYTES = 1024 * 1024;
const STDERR_TAIL_LIMIT = 8_000;

export interface SupervisorOptions {
  config: ServerConfig;
  engine: EngineClient;
  store: RunStore;
  logger: LogSink;
}

export class Supervisor {
  private readonly config: ServerConfig;
  private readonly engine: EngineClient;
  private readonly store: RunStore;
  private readonly log: LogSink;
  private readonly queue: EnqueuedRun[] = [];
  private readonly active = new Map<string, ActiveRun>();
  private readonly subscribers = new Map<string, Map<symbol, SubscriberHandlers>>();
  private shuttingDown = false;

  constructor(options: SupervisorOptions) {
    this.config = options.config;
    this.engine = options.engine;
    this.store = options.store;
    this.log = options.logger;
  }

  get runningCount(): number {
    return this.active.size;
  }

  get queuedCount(): number {
    return this.queue.length;
  }

  /** Add a persisted run to the queue and start it as soon as a slot frees up. */
  enqueue(run: EnqueuedRun): void {
    if (this.shuttingDown) throw new Error("server is shutting down");
    this.queue.push(run);
    this.pump();
  }

  /**
   * Cancel a queued or running run. Queued runs are dropped immediately;
   * running ones get a `<run>/cancel` file plus SIGTERM, then a hard process
   * tree kill after the configured grace period.
   */
  cancel(runId: string): CancelOutcome {
    const queuedIndex = this.queue.findIndex((run) => run.id === runId);
    if (queuedIndex >= 0) {
      this.queue.splice(queuedIndex, 1);
      this.store.setStatus(runId, "cancelled", {
        error: "cancelled before the run started",
        finishedAt: new Date().toISOString(),
      });
      this.endSubscribers(runId);
      return "cancelled";
    }

    const active = this.active.get(runId);
    if (!active) return this.store.getRun(runId) ? "not-active" : "not-found";
    if (!active.cancelled) {
      active.cancelled = true;
      this.requestStop(active);
      this.log.info(`run ${runId}: cancellation requested`);
    }
    return "cancelled";
  }

  subscribe(runId: string, handlers: SubscriberHandlers): () => void {
    const token = Symbol(runId);
    let bucket = this.subscribers.get(runId);
    if (!bucket) {
      bucket = new Map();
      this.subscribers.set(runId, bucket);
    }
    bucket.set(token, handlers);
    return () => {
      const current = this.subscribers.get(runId);
      if (!current) return;
      current.delete(token);
      if (current.size === 0) this.subscribers.delete(runId);
    };
  }

  /** Stop the queue, cancel every live child, and wait for them to exit. */
  async shutdown(): Promise<void> {
    this.shuttingDown = true;
    const finishedAt = new Date().toISOString();
    for (const run of this.queue.splice(0)) {
      try {
        this.store.setStatus(run.id, "cancelled", { error: "server shut down before the run started", finishedAt });
      } catch (cause) {
        this.log.error(`run ${run.id}: cannot record shutdown: ${errorMessage(cause)}`);
      }
      this.endSubscribers(run.id);
    }

    const waits: Promise<void>[] = [];
    for (const active of this.active.values()) {
      if (!active.cancelled) {
        active.cancelled = true;
        this.requestStop(active);
      }
      waits.push(this.forceTerminate(active));
    }
    await Promise.allSettled(waits);

    for (const [runId, active] of this.active) {
      if (!active.finalized) this.finalize(runId, active, { status: "cancelled", error: "server shut down" });
    }
    for (const runId of [...this.subscribers.keys()]) this.endSubscribers(runId);
  }

  private requestStop(active: ActiveRun): void {
    const cancelPath = join(active.runDir, "cancel");
    try {
      writeFileSync(cancelPath, `${new Date().toISOString()} cancel requested\n`, "utf8");
    } catch (cause) {
      this.log.warn(`cannot write cancel file ${cancelPath}: ${errorMessage(cause)}`);
    }
    // On POSIX the engine also handles SIGTERM; Windows has no SIGTERM, so the
    // cancel file above is its only graceful signal.
    if (process.platform !== "win32" && active.child.exitCode === null && active.child.signalCode === null) {
      try {
        active.child.kill("SIGTERM");
      } catch {
        // already gone
      }
    }
    if (!active.terminating) {
      active.terminating = this.forceTerminate(active);
    }
  }

  private async forceTerminate(active: ActiveRun): Promise<void> {
    const graceMs = this.config.cancelGraceMs;
    const exited = await waitForExit(active.child, graceMs);
    if (exited) return;
    this.log.warn(`engine ignored cancellation after ${graceMs} ms; killing the process tree`);
    await terminateChildTree(active.child, { force: true, graceMs: 3_000 });
  }

  private pump(): void {
    if (this.shuttingDown) return;
    while (this.active.size < this.config.maxConcurrentRuns && this.queue.length > 0) {
      const next = this.queue.shift()!;
      this.start(next);
    }
  }

  private start(run: EnqueuedRun): void {
    const runDir = join(this.config.runsDir, run.id);
    const logsDir = join(runDir, "logs");
    const outputDir = join(runDir, "output");
    const specPath = join(runDir, "spec.json");

    try {
      mkdirSync(logsDir, { recursive: true });
      mkdirSync(outputDir, { recursive: true });
      // Keep the documented run layout complete even for a silent engine.
      writeFileSync(join(logsDir, "engine.stderr.log"), "", { flag: "a" });
      writeFileSync(specPath, JSON.stringify(run.spec, null, 2), "utf8");
      if (run.plan) writeFileSync(join(runDir, "plan.json"), JSON.stringify(run.plan, null, 2), "utf8");
    } catch (cause) {
      this.finalizeById(run.id, { status: "failed", error: `cannot prepare run directory: ${errorMessage(cause)}` });
      return;
    }

    let child: ChildProcess;
    try {
      child = this.engine.spawnRun(this.engine.runArgs(specPath, runDir), {
        cwd: runDir,
        // The spawned engine may not be able to use the system temp dir (it is
        // sandboxed per-run). Point its temp at the run dir, which the engine
        // already writes to (spec.json / events.jsonl) — torch's temp files land
        // there, outside output/, so they are not reported as artifacts.
        env: {
          SUPERTUNER_RUN_DIR: runDir,
          TMPDIR: runDir,
          TEMP: runDir,
          TMP: runDir,
        },
      });
    } catch (cause) {
      this.finalizeById(run.id, { status: "failed", error: `cannot launch engine: ${errorMessage(cause)}` });
      return;
    }

    const active: ActiveRun = {
      child,
      runDir,
      cancelled: false,
      stderrTail: "",
      finalized: false,
      terminating: null,
      endEvent: null,
    };
    this.active.set(run.id, active);

    try {
      this.store.setStatus(run.id, "running", {
        startedAt: new Date().toISOString(),
        pid: child.pid ?? null,
        error: null,
      });
    } catch (cause) {
      this.log.error(`run ${run.id}: cannot mark running: ${errorMessage(cause)}`);
    }

    let pending = "";
    const onStdout = (chunk: Buffer | string): void => {
      pending += chunk.toString();
      while (true) {
        const newline = pending.indexOf("\n");
        if (newline === -1) break;
        const line = pending.slice(0, newline);
        pending = pending.slice(newline + 1);
        this.handleLine(run.id, active, line);
      }
      if (pending.length > MAX_PENDING_LINE_BYTES) {
        this.appendLog(active, `[protocol] dropped ${pending.length} bytes without a newline`);
        pending = "";
      }
    };

    const onStderr = (chunk: Buffer | string): void => {
      const text = chunk.toString();
      active.stderrTail = (active.stderrTail + text).slice(-STDERR_TAIL_LIMIT);
      try {
        appendFileSync(join(active.runDir, "logs", "engine.stderr.log"), text, "utf8");
      } catch {
        // Log file failure must not take down the supervisor.
      }
    };

    child.stdout?.setEncoding("utf8");
    child.stderr?.setEncoding("utf8");
    child.stdout?.on("data", onStdout);
    child.stderr?.on("data", onStderr);

    child.once("error", (error: Error) => {
      this.log.error(`run ${run.id}: engine failed to start: ${error.message}`);
      this.finalize(run.id, active, { status: "failed", error: `engine failed to start: ${error.message}` });
    });

    child.once("close", (code, signal) => {
      child.stdout?.removeAllListeners("data");
      child.stderr?.removeAllListeners("data");
      if (pending.trim()) this.handleLine(run.id, active, pending);
      pending = "";
      const outcome = this.exitOutcome(active, code, signal);
      this.finalize(run.id, active, outcome);
    });

    this.log.info(`run ${run.id}: started (pid ${child.pid ?? "?"})`);
  }

  private exitOutcome(
    active: ActiveRun,
    code: number | null,
    signal: NodeJS.Signals | null,
  ): { status: RunStatus; error: string | null; exitCode: number | null } {
    if (active.cancelled) return { status: "cancelled", error: null, exitCode: code ?? 130 };
    const tailLine = lastMeaningfulLine(active.stderrTail);
    if (code === 0) return { status: "completed", error: null, exitCode: 0 };
    if (code === 2) {
      const reason = active.endEvent?.status === "failed" ? active.endEvent.reason : null;
      return {
        status: "failed",
        error: `engine refused the run (exit code 2)${reason ? `: ${reason}` : tailLine ? `: ${tailLine}` : ""}`,
        exitCode: 2,
      };
    }
    if (code === 3) {
      const reason = active.endEvent?.status === "failed" ? active.endEvent.reason : null;
      return {
        status: "failed",
        error: `engine runtime failure (exit code 3)${reason ? `: ${reason}` : tailLine ? `: ${tailLine}` : ""}`,
        exitCode: 3,
      };
    }
    if (code === 130) return { status: "cancelled", error: "engine reported cancellation", exitCode: 130 };
    const reason = active.endEvent?.status === "failed" ? active.endEvent.reason : null;
    return {
      status: "failed",
      error: `engine exited with code ${code === null ? "null" : code}${signal ? ` (signal ${signal})` : ""}${
        reason ? `: ${reason}` : tailLine ? `: ${tailLine}` : ""
      }`,
      exitCode: code,
    };
  }

  private handleLine(runId: string, active: ActiveRun, line: string): void {
    if (!line.trim()) return;
    const parsed = parseEventLine(line);
    if (parsed.ok) {
      this.persist(runId, active, line, parsed.event);
      this.emit(runId, parsed.event);
      this.applyToStore(runId, active, parsed.event);
      return;
    }

    // Forward compatibility: the engine may emit event types added in core
    // after this server build. Preserve them verbatim instead of dropping them.
    const raw = tryParseObject(line);
    if (raw && typeof raw.type === "string") {
      this.persist(runId, active, line, raw);
      this.emit(runId, raw);
      return;
    }

    this.appendLog(active, `[protocol] ${parsed.reason}: ${truncate(line, 500)}`);
  }

  private applyToStore(runId: string, active: ActiveRun, event: RunEvent): void {
    try {
      switch (event.type) {
        case "step":
          this.store.recordStep(runId, event.step, event.loss ?? null);
          break;
        case "checkpoint": {
          const artifact = relativeArtifact(active.runDir, event.path);
          if (artifact) this.store.appendArtifacts(runId, [artifact]);
          break;
        }
        case "error":
          this.store.updateRun(runId, { error: event.message });
          break;
        case "memory_report": {
          // 跑后自校准（v0.2 T1）：把本次「计划 vs 实测」按校准键聚合进 calibrations。
          // 五元组缺一即无法定键，跳过（不占计数）；delta_pct 为 null 时仍占计数。
          const { family, stage, adapter, quant_base, blocks_to_swap, delta_pct } = event;
          if (family !== null && stage !== null && adapter !== null && quant_base !== null && blocks_to_swap !== null) {
            this.store.upsertCalibration({ family, stage, adapter, quant_base, blocks_to_swap, delta_pct });
          }
          break;
        }
        case "run_end":
          active.endEvent = event;
          break;
        default:
          break;
      }
    } catch (cause) {
      this.log.error(`run ${runId}: cannot persist ${event.type} event: ${errorMessage(cause)}`);
    }
  }

  private persist(runId: string, active: ActiveRun, line: string, _event: unknown): void {
    try {
      appendFileSync(join(active.runDir, "events.jsonl"), `${line.trim()}\n`, "utf8");
    } catch (cause) {
      this.log.error(`run ${runId}: cannot append events.jsonl: ${errorMessage(cause)}`);
    }
  }

  private appendLog(active: ActiveRun, message: string): void {
    try {
      appendFileSync(join(active.runDir, "logs", "engine.stdout.log"), `${new Date().toISOString()} ${message}\n`, "utf8");
    } catch {
      // ignore
    }
  }

  private finalizeById(
    runId: string,
    outcome: { status: RunStatus; error: string | null; exitCode?: number | null },
  ): void {
    this.store.setStatus(runId, outcome.status, {
      exitCode: outcome.exitCode ?? null,
      error: outcome.error,
      finishedAt: new Date().toISOString(),
    });
    this.endSubscribers(runId);
    this.pump();
  }

  private finalize(
    runId: string,
    active: ActiveRun,
    outcome: { status: RunStatus; error: string | null; exitCode?: number | null },
  ): void {
    if (active.finalized) return;
    active.finalized = true;
    this.active.delete(runId);

    let artifacts: string[] = [];
    try {
      artifacts = listArtifactPaths(join(active.runDir, "output"));
    } catch (cause) {
      this.log.warn(`run ${runId}: cannot scan artifacts: ${errorMessage(cause)}`);
    }

    const exitCode = outcome.exitCode ?? null;
    try {
      if (artifacts.length > 0) this.store.appendArtifacts(runId, artifacts);
      this.store.setStatus(runId, outcome.status, {
        exitCode,
        error: outcome.error,
        finishedAt: new Date().toISOString(),
        pid: null,
      });
    } catch (cause) {
      this.log.error(`run ${runId}: cannot record final status: ${errorMessage(cause)}`);
    }

    this.log.info(
      `run ${runId}: ${outcome.status}${exitCode !== null ? ` (exit ${exitCode})` : ""}${
        outcome.error ? ` — ${outcome.error}` : ""
      }`,
    );
    this.endSubscribers(runId);
    this.pump();
  }

  private emit(runId: string, event: unknown): void {
    const bucket = this.subscribers.get(runId);
    if (!bucket) return;
    for (const handlers of bucket.values()) {
      try {
        handlers.onEvent(event);
      } catch (cause) {
        this.log.warn(`run ${runId}: subscriber failed: ${errorMessage(cause)}`);
      }
    }
  }

  private endSubscribers(runId: string): void {
    const bucket = this.subscribers.get(runId);
    if (!bucket) return;
    this.subscribers.delete(runId);
    for (const handlers of bucket.values()) {
      try {
        handlers.onEnd();
      } catch (cause) {
        this.log.warn(`run ${runId}: subscriber end handler failed: ${errorMessage(cause)}`);
      }
    }
  }
}

function relativeArtifact(runDir: string, absolute: string): string | null {
  const outputDir = join(runDir, "output");
  const normalized = absolute.replace(/\\/g, "/");
  const prefix = outputDir.replace(/\\/g, "/");
  if (!normalized.toLowerCase().startsWith(`${prefix.toLowerCase()}/`)) return null;
  return normalized.slice(prefix.length + 1);
}

function tryParseObject(line: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(line);
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function lastMeaningfulLine(text: string): string {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  return lines.length > 0 ? truncate(lines[lines.length - 1] ?? "", 300) : "";
}

function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}…`;
}

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}