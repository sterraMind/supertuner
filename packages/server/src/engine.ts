/**
 * Client for the Python engine CLI.
 *
 * The server never computes memory numbers or backend argv: it writes a spec,
 * runs `supertuner-engine <command> --json`, and validates the response against
 * the shared zod contracts. Spawning always uses argv arrays with
 * `shell: false`.
 *
 * Engine contract (confirmed with the engine implementation):
 *   compile --spec <file> --json   -> CompileResponse; exit 2 when the plan is
 *                                     infeasible (still returns ok:true), 3 on
 *                                     an internal crash
 *   run     --spec <file> --run-dir <dir> --events-stdout
 *                                 -> JSONL RunEvent lines on stdout
 *   plugins --json                -> PluginsResponse
 *   env     --json                -> EnvReport
 *   families --json               -> FamiliesResponse
 *   version                       -> {version}
 *   reload (POST /api/engine/reload) re-runs `plugins` with a fresh cache.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  CompileResponseSchema,
  EnvReportSchema,
  FamiliesResponseSchema,
  PluginsResponseSchema,
  type CompileResponse,
  type EnvReport,
  type FamiliesResponse,
  type Issue,
  type PluginsResponse,
  type RunPlan,
  type RunSpec,
} from "@supertuner/core";
import type { ServerConfig } from "./config.js";
import { terminateChildTree, waitForExit } from "./process.js";

export type EngineFailureKind =
  /** The program could not be launched at all (ENOENT, EACCES). */
  | "unavailable"
  /** The command exceeded its deadline and was killed. */
  | "timeout"
  /** The engine rejected the spec (CompileResponse `ok:false`). */
  | "refused"
  /** Non-zero exit without a usable payload, or exit 3. */
  | "crash"
  /** Exit 0 but stdout was not the documented JSON shape. */
  | "protocol"
  /** The subcommand does not exist in this engine build. */
  | "unsupported";

export class EngineError extends Error {
  constructor(
    message: string,
    readonly kind: EngineFailureKind,
    readonly detail: {
      issues?: Issue[];
      exitCode?: number | null;
      stderr?: string;
      stdout?: string;
    } = {},
  ) {
    super(message);
    this.name = "EngineError";
  }
}

export interface ExecResult {
  code: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  durationMs: number;
}

const MAX_STDOUT_BYTES = 8 * 1024 * 1024;
const MAX_STDERR_BYTES = 256 * 1024;

interface CacheEntry<T> {
  at: number;
  ttlMs: number;
  value: T;
}

export interface EngineClientOptions {
  config: ServerConfig;
  /** Extra environment for every child (tests set it; production leaves it empty). */
  env?: NodeJS.ProcessEnv;
}

/**
 * Runs the engine and validates its JSON. Probe results are cached briefly so
 * `/api/system` and `/api/plugins` do not fork a Python process on every poll;
 * `POST /api/engine/reload` clears the cache.
 */
export class EngineClient {
  private readonly config: ServerConfig;
  private readonly extraEnv: NodeJS.ProcessEnv;
  private readonly cache = new Map<string, CacheEntry<unknown>>();
  private readonly defaultTtlMs: number;

  constructor(options: EngineClientOptions) {
    this.config = options.config;
    this.extraEnv = options.env ?? {};
    this.defaultTtlMs = 30_000;
  }

  /** argv prefix for a child process, including the engine program and its flags. */
  argvFor(args: readonly string[]): string[] {
    return [...this.config.engineCmd, ...args];
  }

  /** Environment every engine process inherits (paths come from config only). */
  childEnv(extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
    // torch's import chain calls tempfile.gettempdir(). The inherited TEMP/TMP
    // can point at a location the engine cannot write (or a drive it cannot
    // map), so pin every temp var to a home-relative dir we own and created.
    const engineTmp = join(this.config.home, "tmp");
    mkdirSync(engineTmp, { recursive: true });
    return {
      ...process.env,
      ...this.extraEnv,
      SUPERTUNER_HOME: this.config.home,
      SUPERTUNER_REPO_ROOT: this.config.repoRoot,
      SUPERTUNER_RUNS_DIR: this.config.runsDir,
      SUPERTUNER_ARTIFACTS_DIR: this.config.artifactsDir,
      SUPERTUNER_CACHE_DIR: this.config.cacheDir,
      SUPERTUNER_MODELS_DIR: this.config.modelsDir,
      SUPERTUNER_DATASETS_DIR: this.config.datasetsDir,
      TMPDIR: engineTmp,
      TEMP: engineTmp,
      TMP: engineTmp,
      PYTHONUNBUFFERED: "1",
      PYTHONIOENCODING: "utf-8",
      ...extra,
    };
  }

  runArgs(specPath: string, runDir: string): string[] {
    return ["run", "--spec", specPath, "--run-dir", runDir, "--events-stdout"];
  }

  /**
   * Compile a spec server-side. Returns the plan even when it is infeasible
   * (the engine signals that with exit code 2 but a complete `ok:true` plan);
   * throws `EngineError` when the engine cannot produce a plan at all.
   */
  async compile(spec: RunSpec): Promise<{ plan: RunPlan; exitCode: number | null }> {
    const tmpDir = join(this.config.home, "tmp");
    mkdirSync(tmpDir, { recursive: true });
    const specPath = join(tmpDir, `spec-${randomUUID()}.json`);
    writeFileSync(specPath, JSON.stringify(spec, null, 2), "utf8");

    let result: ExecResult;
    try {
      result = await this.exec(["compile", "--spec", specPath, "--json"], this.config.compileTimeoutMs, this.config.repoRoot);
    } finally {
      rmSync(specPath, { force: true });
    }

    if (result.code === 3) {
      throw new EngineError("engine crashed while compiling the spec", "crash", {
        exitCode: result.code,
        stderr: tail(result.stderr),
        stdout: tail(result.stdout),
      });
    }

    const payload = this.parseJson(result, "compile");
    const parsed = CompileResponseSchema.safeParse(payload);
    if (!parsed.success) {
      throw new EngineError(
        `engine compile output does not match CompileResponse: ${formatIssues(parsed.error.issues)}`,
        "protocol",
        { exitCode: result.code, stdout: tail(result.stdout), stderr: tail(result.stderr) },
      );
    }

    const response: CompileResponse = parsed.data;
    if (!response.ok) {
      throw new EngineError(response.error, "refused", {
        issues: response.issues,
        exitCode: result.code,
        stderr: tail(result.stderr),
      });
    }

    return { plan: response.plan, exitCode: result.code };
  }

  plugins(options: { refresh?: boolean } = {}): Promise<PluginsResponse> {
    return this.cached("plugins", options.refresh ?? false, this.defaultTtlMs, async () => {
      const result = await this.exec(["plugins", "--json"], this.config.probeTimeoutMs, this.config.repoRoot);
      const payload = this.parseJson(result, "plugins", result.code);
      const parsed = PluginsResponseSchema.safeParse(payload);
      if (!parsed.success) {
        throw new EngineError(`engine plugins output does not match PluginsResponse: ${formatIssues(parsed.error.issues)}`, "protocol", {
          exitCode: result.code,
          stdout: tail(result.stdout),
          stderr: tail(result.stderr),
        });
      }
      return parsed.data;
    });
  }

  env(options: { refresh?: boolean } = {}): Promise<EnvReport> {
    return this.cached("env", options.refresh ?? false, this.defaultTtlMs, async () => {
      const result = await this.exec(["env", "--json"], this.config.probeTimeoutMs, this.config.repoRoot);
      const payload = this.parseJson(result, "env", result.code);
      const parsed = EnvReportSchema.safeParse(payload);
      if (!parsed.success) {
        throw new EngineError(`engine env output does not match EnvReport: ${formatIssues(parsed.error.issues)}`, "protocol", {
          exitCode: result.code,
          stdout: tail(result.stdout),
          stderr: tail(result.stderr),
        });
      }
      return parsed.data;
    });
  }

  families(options: { refresh?: boolean } = {}): Promise<FamiliesResponse> {
    return this.cached("families", options.refresh ?? false, this.defaultTtlMs, async () => {
      const result = await this.exec(["families", "--json"], this.config.probeTimeoutMs, this.config.repoRoot);
      const payload = this.parseJson(result, "families", result.code);
      const parsed = FamiliesResponseSchema.safeParse(payload);
      if (!parsed.success) {
        throw new EngineError(
          `engine families output does not match FamiliesResponse: ${formatIssues(parsed.error.issues)}`,
          "protocol",
          { exitCode: result.code, stdout: tail(result.stdout), stderr: tail(result.stderr) },
        );
      }
      return parsed.data;
    });
  }

  async version(): Promise<string> {
    const entry = this.cache.get("version");
    if (entry) return entry.value as string;
    try {
      const result = await this.exec(["version"], this.config.probeTimeoutMs, this.config.repoRoot);
      if (result.code !== 0) throw new Error(`exit ${result.code}`);
      const raw: unknown = JSON.parse(result.stdout);
      const version = typeof raw === "object" && raw !== null && "version" in raw ? String(raw.version) : "unknown";
      this.setCache("version", version, Number.POSITIVE_INFINITY);
      return version;
    } catch (cause) {
      // A missing `version` command must not take down /api/system.
      if (cause instanceof EngineError) throw cause;
      this.setCache("version", "unknown", 5_000);
      return "unknown";
    }
  }

  /** Drop cached probes and re-probe plugins; used by POST /api/engine/reload. */
  async reload(): Promise<PluginsResponse> {
    this.cache.clear();
    return await this.plugins({ refresh: true });
  }

  /** Spawn a run child process. The supervisor owns its lifecycle. */
  spawnRun(args: readonly string[], options: { cwd: string; env?: NodeJS.ProcessEnv }): ChildProcess {
    const argv = this.argvFor(args);
    const program = argv[0];
    if (!program) throw new EngineError("engine command is empty", "unavailable");
    return spawn(program, argv.slice(1), {
      cwd: options.cwd,
      env: this.childEnv(options.env),
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
  }

  private setCache(key: string, value: unknown, ttlMs: number): void {
    this.cache.set(key, { at: Date.now(), ttlMs, value });
  }

  private async cached<T>(key: string, refresh: boolean, ttlMs: number, load: () => Promise<T>): Promise<T> {
    const entry = this.cache.get(key) as CacheEntry<T> | undefined;
    if (!refresh && entry && Date.now() - entry.at < entry.ttlMs) return entry.value;
    const value = await load();
    this.setCache(key, value, ttlMs);
    return value;
  }

  private parseJson(result: ExecResult, command: string, exitCode?: number | null): unknown {
    const code = exitCode ?? result.code;
    const trimmed = result.stdout.trim();
    if (trimmed === "") {
      // Typer prints usage to stderr and exits 2 for an unknown subcommand.
      const unsupported = code === 2 && /no such (command|option)|unexpected argument|usage:/i.test(result.stderr);
      throw new EngineError(
        unsupported
          ? `engine build does not implement '${command}' (exit ${code}): ${tail(result.stderr)}`
          : `engine '${command}' produced no output (exit ${code ?? "signal"})${result.stderr ? `: ${tail(result.stderr)}` : ""}`,
        unsupported ? "unsupported" : "crash",
        { exitCode: code, stderr: tail(result.stderr) },
      );
    }
    try {
      return JSON.parse(trimmed) as unknown;
    } catch (cause) {
      throw new EngineError(
        `engine '${command}' wrote invalid JSON: ${(cause as Error).message}`,
        "protocol",
        { exitCode: code, stdout: tail(result.stdout), stderr: tail(result.stderr) },
      );
    }
  }

  private exec(args: readonly string[], timeoutMs: number, cwd: string): Promise<ExecResult> {
    const argv = this.argvFor(args);
    const program = argv[0];
    if (!program) {
      return Promise.reject(new EngineError("engine command is empty", "unavailable"));
    }
    const startedAt = Date.now();

    return new Promise<ExecResult>((resolvePromise, rejectPromise) => {
      let child: ChildProcess;
      try {
        child = spawn(program, argv.slice(1), {
          cwd,
          env: this.childEnv(),
          shell: false,
          windowsHide: true,
          stdio: ["ignore", "pipe", "pipe"],
        });
      } catch (cause) {
        rejectPromise(
          new EngineError(`cannot launch engine '${program}': ${(cause as Error).message}`, "unavailable", {
            exitCode: null,
          }),
        );
        return;
      }

      const stdout: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      let stdoutBytes = 0;
      let stderrBytes = 0;
      let timedOut = false;
      let settled = false;

      const timer = setTimeout(() => {
        timedOut = true;
        // SIGTERM first (Python flushes), then a hard kill if the child ignores it.
        void terminateChildTree(child, { signal: "SIGTERM", graceMs: 2_000, force: true }).then(() => {
          if (settled) return;
          settled = true;
          rejectPromise(
            new EngineError(`engine '${args[0] ?? "?"}' timed out after ${timeoutMs} ms`, "timeout", {
              exitCode: null,
              stdout: tail(Buffer.concat(stdout).toString("utf8")),
              stderr: tail(Buffer.concat(stderrChunks).toString("utf8")),
            }),
          );
        });
      }, timeoutMs);
      timer.unref();

      const cleanup = (): void => {
        clearTimeout(timer);
        child.stdout?.removeAllListeners();
        child.stderr?.removeAllListeners();
        child.removeAllListeners("error");
      };

      child.stdout?.on("data", (chunk: Buffer) => {
        stdoutBytes += chunk.length;
        if (stdoutBytes <= MAX_STDOUT_BYTES) stdout.push(chunk);
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        stderrBytes += chunk.length;
        if (stderrBytes <= MAX_STDERR_BYTES) stderrChunks.push(chunk);
        else stderrChunks.push(chunk.subarray(Math.max(0, chunk.length - 4096)));
      });

      child.on("error", (error: Error) => {
        if (settled) return;
        settled = true;
        cleanup();
        rejectPromise(
          new EngineError(`cannot launch engine '${program}': ${error.message}`, "unavailable", {
            exitCode: null,
            stderr: tail(Buffer.concat(stderrChunks).toString("utf8")),
          }),
        );
      });

      child.on("close", (code, signal) => {
        if (settled) return;
        settled = true;
        cleanup();
        if (timedOut) return; // the timeout path is already rejecting
        resolvePromise({
          code,
          signal,
          stdout: Buffer.concat(stdout).toString("utf8"),
          stderr: Buffer.concat(stderrChunks).toString("utf8"),
          durationMs: Date.now() - startedAt,
        });
      });
    });
  }
}

function tail(text: string, limit = 4_000): string {
  const trimmed = text.trim();
  return trimmed.length <= limit ? trimmed : `…${trimmed.slice(-limit)}`;
}

function formatIssues(issues: readonly { path: (string | number)[]; message: string }[]): string {
  return issues.map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`).join("; ");
}

/** Re-exported for tests that need to wait on a killed engine process. */
export { waitForExit };