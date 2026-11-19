/**
 * Server configuration.
 *
 * Every deployment-specific value comes from the environment, mirroring the
 * `SUPERTUNER_*` contract in `packages/core/src/config.ts`. The defaults here
 * are the v0.1 server defaults: a loopback-only API on 8787, one concurrent
 * run, and state under `SUPERTUNER_HOME`.
 */

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, isAbsolute, join, relative, resolve } from "node:path";
import { loadSettings, type Settings } from "@supertuner/core";

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface ServerConfig {
  /** `SUPERTUNER_HOME`: runs, logs, database and the engine's caches. */
  home: string;
  runsDir: string;
  artifactsDir: string;
  cacheDir: string;
  modelsDir: string;
  datasetsDir: string;
  dbPath: string;
  repoRoot: string;
  /** Engine argv prefix (program plus its own flags), never a shell string. */
  engineCmd: string[];
  host: string;
  port: number;
  /** Built web UI; `null` when the directory does not exist yet. */
  webDist: string | null;
  maxConcurrentRuns: number;
  allowAnyPath: boolean;
  /** Roots a run spec path must live under unless `allowAnyPath`. */
  allowedRoots: string[];
  compileTimeoutMs: number;
  probeTimeoutMs: number;
  /** Grace period between SIGTERM and a hard kill on cancellation. */
  cancelGraceMs: number;
  logLevel: LogLevel;
  /** Browser origins allowed by the dev CORS shim. */
  devOrigins: string[];
}

const DEFAULT_PORT = 8787;
const DEFAULT_HOST = "127.0.0.1";

function expandHome(p: string): string {
  if (p === "~") return homedir();
  if (p.startsWith("~/") || p.startsWith("~\\")) return join(homedir(), p.slice(2));
  return p;
}

function envPath(env: NodeJS.ProcessEnv, key: string, fallback: string, base: string): string {
  const raw = env[key];
  if (raw && raw.trim()) {
    const expanded = expandHome(raw.trim());
    return isAbsolute(expanded) ? resolve(expanded) : resolve(base, expanded);
  }
  return resolve(fallback);
}

function envInt(env: NodeJS.ProcessEnv, key: string, fallback: number, min: number, max: number): number {
  const raw = env[key];
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number(raw.trim());
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${key} must be an integer between ${min} and ${max}, got ${JSON.stringify(raw)}`);
  }
  return value;
}

function envBool(env: NodeJS.ProcessEnv, key: string, fallback = false): boolean {
  const raw = env[key]?.trim().toLowerCase();
  if (raw === undefined || raw === "") return fallback;
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off"].includes(raw)) return false;
  throw new Error(`${key} must be one of 1/true/yes/on or 0/false/no/off, got ${JSON.stringify(env[key])}`);
}

/**
 * Split a command line into argv without a shell: whitespace separates
 * arguments, single and double quotes group them, and `\"` escapes a quote
 * inside a double-quoted argument. There is deliberately no variable
 * expansion, globbing, or pipe handling — `SUPERTUNER_ENGINE_CMD` names a
 * program and its flags, not a shell pipeline.
 */
export function parseArgv(input: string): string[] {
  const out: string[] = [];
  let current = "";
  let quote: '"' | "'" | null = null;
  let started = false;

  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i]!;
    if (quote === '"') {
      if (ch === "\\" && i + 1 < input.length) {
        // Windows paths keep their backslashes; only \" and \\ are escapes.
        const next = input[i + 1]!;
        if (next === '"' || next === "\\") {
          current += next;
          i += 1;
        } else {
          current += ch;
        }
      } else if (ch === '"') {
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }
    if (quote === "'") {
      if (ch === "'") quote = null;
      else current += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      started = true;
      continue;
    }
    if (/\s/.test(ch)) {
      if (started) {
        out.push(current);
        current = "";
        started = false;
      }
      continue;
    }
    current += ch;
    started = true;
  }

  if (quote !== null) throw new Error(`unterminated ${quote} quote in SUPERTUNER_ENGINE_CMD`);
  if (started) out.push(current);
  if (out.length === 0) throw new Error("SUPERTUNER_ENGINE_CMD is empty");
  return out;
}

/** Expand a leading `~` without resolving relative paths. */
export function expandUserPath(raw: string): string {
  return expandHome(raw.trim());
}

/** Expand `~`, resolve against the repo root, and require an absolute result. */
export function resolveUserPath(raw: string, base: string): string {
  const expanded = expandHome(raw.trim());
  return isAbsolute(expanded) ? resolve(expanded) : resolve(base, expanded);
}

/** True when `child` is `root` or lives under it (case-insensitive on Windows). */
export function isPathInside(root: string, child: string): boolean {
  const rel = relative(resolve(root), resolve(child));
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
}

export interface LoadConfigOptions {
  env?: NodeJS.ProcessEnv;
  /** Programmatic overrides, mainly for tests. They win over the environment. */
  overrides?: Partial<ServerConfig>;
}

export function loadServerConfig(options: LoadConfigOptions = {}): ServerConfig {
  const env = options.env ?? process.env;
  // Validate numeric overrides before core settings coerces them, so a bad
  // value names the variable instead of surfacing as a zod path error.
  envInt(env, "SUPERTUNER_PORT", DEFAULT_PORT, 1, 65535);
  const coreSettings: Settings = loadSettings(env);
  const repoRoot = resolve(coreSettings.repoRoot);
  const home = resolve(coreSettings.home);

  const engineCmd = env.SUPERTUNER_ENGINE_CMD?.trim()
    ? parseArgv(env.SUPERTUNER_ENGINE_CMD)
    : ["uv", "run", "--project", join(repoRoot, "engine"), "supertuner-engine"];

  const webDistPath = envPath(env, "SUPERTUNER_WEB_DIST", join(repoRoot, "packages", "web", "dist"), repoRoot);
  const configuredRoots = (env.SUPERTUNER_ALLOWED_ROOTS ?? "")
    .split(delimiter)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => resolveUserPath(p, repoRoot));

  const config: ServerConfig = {
    home,
    runsDir: resolve(coreSettings.runsDir),
    artifactsDir: resolve(coreSettings.artifactsDir),
    cacheDir: resolve(coreSettings.cacheDir),
    modelsDir: envPath(env, "SUPERTUNER_MODELS_DIR", join(home, "models"), home),
    datasetsDir: envPath(env, "SUPERTUNER_DATASETS_DIR", join(home, "datasets"), home),
    dbPath: envPath(env, "SUPERTUNER_DB_PATH", join(home, "supertuner.db"), home),
    repoRoot,
    engineCmd,
    host: env.SUPERTUNER_HOST?.trim() || DEFAULT_HOST,
    port: envInt(env, "SUPERTUNER_PORT", DEFAULT_PORT, 1, 65535),
    webDist: existsSync(join(webDistPath, "index.html")) ? webDistPath : null,
    maxConcurrentRuns: envInt(env, "SUPERTUNER_MAX_CONCURRENT_RUNS", 1, 1, 64),
    allowAnyPath: envBool(env, "SUPERTUNER_ALLOW_ANY_PATH"),
    allowedRoots: [],
    compileTimeoutMs: envInt(env, "SUPERTUNER_COMPILE_TIMEOUT_MS", 120_000, 1_000, 3_600_000),
    probeTimeoutMs: envInt(env, "SUPERTUNER_PROBE_TIMEOUT_MS", 60_000, 1_000, 3_600_000),
    cancelGraceMs: envInt(env, "SUPERTUNER_CANCEL_GRACE_MS", 10_000, 100, 600_000),
    logLevel: (env.SUPERTUNER_LOG_LEVEL as LogLevel | undefined) ?? "info",
    devOrigins: ["http://localhost:5173", "http://127.0.0.1:5173"],
  };

  config.allowedRoots = dedupePaths([
    ...configuredRoots,
    config.home,
    config.datasetsDir,
    config.modelsDir,
    config.runsDir,
    config.artifactsDir,
  ]);

  return { ...config, ...options.overrides };
}

function dedupePaths(paths: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const path of paths) {
    const key = process.platform === "win32" ? resolve(path).toLowerCase() : resolve(path);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(resolve(path));
  }
  return out;
}

/** Recursively verify every requirement that HTTP requests rely on. */
export function assertConfigUsable(config: ServerConfig): void {
  if (config.engineCmd.length === 0) throw new Error("engine command is empty");
  if (config.allowedRoots.length === 0) throw new Error("no allowed path roots configured");
}