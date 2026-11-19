/**
 * Runtime configuration.
 *
 * Every path and process invocation is derived from the environment: nothing
 * here hardcodes an absolute location. Defaults live under SUPERTUNER_HOME so a
 * fresh clone works, while CI and packagers can point each directory elsewhere.
 */

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

const SettingsSchema = z.object({
  home: z.string().min(1),
  runsDir: z.string().min(1),
  artifactsDir: z.string().min(1),
  cacheDir: z.string().min(1),
  dbPath: z.string().min(1),
  repoRoot: z.string().min(1),
  enginePython: z.string().min(1),
  engineModule: z.string().min(1),
  host: z.string().min(1),
  port: z.number().int().min(1).max(65535),
  logLevel: z.enum(["debug", "info", "warn", "error"]),
  gpuIndex: z.number().int().min(0),
  repoTimeoutMs: z.number().int().positive(),
});

export type Settings = z.infer<typeof SettingsSchema>;

function expand(p: string): string {
  if (p === "~") return homedir();
  if (p.startsWith("~/") || p.startsWith("~\\")) return join(homedir(), p.slice(2));
  return p;
}

/** Absolute-or-relative to the repo root; env values win, defaults are internal. */
function pathFromEnv(env: NodeJS.ProcessEnv, key: string, fallback: string, base: string): string {
  const raw = env[key];
  if (raw && raw.trim()) {
    const expanded = expand(raw.trim());
    return isAbsolute(expanded) ? expanded : resolve(base, expanded);
  }
  return isAbsolute(fallback) ? fallback : resolve(base, fallback);
}

/**
 * Find the workspace root by walking up from this module looking for the pnpm
 * workspace file. Keeps bundlers, tsx, and dist runs all working without a
 * hardcoded checkout path.
 */
export function findRepoRoot(startDir: string = dirname(fileURLToPath(import.meta.url))): string {
  let dir = resolve(startDir);
  for (let i = 0; i < 12; i += 1) {
    if (existsSync(join(dir, "pnpm-workspace.yaml")) || existsSync(join(dir, "engine", "pyproject.toml"))) {
      return dir;
    }
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

export function loadSettings(env: NodeJS.ProcessEnv = process.env): Settings {
  const repoRoot = env.SUPERTUNER_REPO_ROOT
    ? resolve(expand(env.SUPERTUNER_REPO_ROOT))
    : findRepoRoot();

  const home = pathFromEnv(env, "SUPERTUNER_HOME", join(homedir(), ".supertuner"), repoRoot);
  const defaultPython =
    process.platform === "win32"
      ? join(repoRoot, "engine", ".venv", "Scripts", "python.exe")
      : join(repoRoot, "engine", ".venv", "bin", "python");

  const candidate = SettingsSchema.parse({
    home,
    runsDir: pathFromEnv(env, "SUPERTUNER_RUNS_DIR", join(home, "runs"), home),
    artifactsDir: pathFromEnv(env, "SUPERTUNER_ARTIFACTS_DIR", join(home, "artifacts"), home),
    cacheDir: pathFromEnv(env, "SUPERTUNER_CACHE_DIR", join(home, "cache"), home),
    dbPath: pathFromEnv(env, "SUPERTUNER_DB_PATH", join(home, "supertuner.sqlite3"), home),
    repoRoot,
    enginePython: pathFromEnv(env, "SUPERTUNER_ENGINE_PYTHON", defaultPython, repoRoot),
    engineModule: env.SUPERTUNER_ENGINE_MODULE?.trim() || "supertuner_engine.cli",
    host: env.SUPERTUNER_HOST?.trim() || "127.0.0.1",
    port: Number(env.SUPERTUNER_PORT ?? 4319),
    logLevel: (env.SUPERTUNER_LOG_LEVEL as Settings["logLevel"]) ?? "info",
    gpuIndex: Number(env.SUPERTUNER_GPU_INDEX ?? 0),
    repoTimeoutMs: Number(env.SUPERTUNER_REPO_TIMEOUT_MS ?? 30_000),
  });

  return candidate;
}

/** Dir for one run's checkpoints, logs, and manifest. */
export function runDir(settings: Settings, runId: string): string {
  return join(settings.runsDir, runId);
}

/** Where a run's JSONL event log is kept after the process exits. */
export function runEventsPath(settings: Settings, runId: string): string {
  return join(runDir(settings, runId), "events.jsonl");
}