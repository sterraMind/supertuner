/**
 * SQLite persistence for runs.
 *
 * `node:sqlite` is a Node built-in since 22.5 (and unflagged on 26). The import
 * goes through `createRequire` so a runtime without it fails with a single
 * actionable message instead of a module-resolution stack trace at startup.
 *
 * Schema changes are applied with `PRAGMA user_version`; append a statement
 * list to `MIGRATIONS` instead of editing an existing one.
 */

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { DatabaseSync, StatementSync } from "node:sqlite";
import {
  RUN_STATUSES,
  RunPlanSchema,
  RunRecordSchema,
  RunSpecSchema,
  calibrationKey,
  type RunPlan,
  type RunRecord,
  type RunSpec,
  type RunStatus,
} from "@supertuner/core";

const requireBuiltin = createRequire(import.meta.url);

function loadSqlite(): typeof import("node:sqlite") {
  try {
    return requireBuiltin("node:sqlite") as typeof import("node:sqlite");
  } catch (cause) {
    throw new Error(
      "SUPERTUNER DB UNAVAILABLE: this Node runtime has no 'node:sqlite'. " +
        "SuperTuner needs Node >= 22.5 (Node 22.5-23.x may require --experimental-sqlite; " +
        "Node 26 ships it unflagged). " +
        `Original error: ${(cause as Error).message}`,
    );
  }
}

/** One row of the `runs` table, exactly as stored. */
export interface RunRow {
  id: string;
  name: string;
  status: RunStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  updated_at: string;
  spec_json: string;
  plan_json: string | null;
  exit_code: number | null;
  error: string | null;
  steps: number;
  last_step: number | null;
  loss: number | null;
  artifacts_json: string;
  pid: number | null;
  backend: string | null;
}

/** Fields `updateRun` may change; JSON columns accept the parsed value. */
export interface RunPatch {
  status?: RunStatus;
  started_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  error?: string | null;
  steps?: number;
  last_step?: number | null;
  loss?: number | null;
  artifacts?: string[];
  pid?: number | null;
  backend?: string | null;
  plan?: RunPlan | null;
}

export interface NewRun {
  id: string;
  name: string;
  status?: RunStatus;
  spec: RunSpec;
  plan?: RunPlan | null;
  backend?: string | null;
  createdAt?: string;
}

/** calibrations 表的一行，按校准键（五元组拼接字符串）聚合。 */
export interface CalibrationRow {
  key: string;
  family: string;
  stage: string;
  adapter: string;
  quant_base: string;
  blocks_to_swap: number;
  count: number;
  sum_delta_pct: number;
  avg_delta_pct: number | null;
  updated_at: string;
}

/** upsertCalibration 的输入：一次运行回报的校准键五元组 + 实测偏差。 */
export interface CalibrationInput {
  family: string;
  stage: string;
  adapter: string;
  quant_base: string;
  blocks_to_swap: number;
  /** 实测相对计划的百分比偏差；null 表示本次未测得，不计入求和。 */
  delta_pct: number | null;
}

export class DatabaseError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "DatabaseError";
    if (cause !== undefined) this.cause = cause;
  }
}

const MIGRATIONS: readonly string[][] = [
  [
    `create table if not exists runs (
       id text primary key,
       name text not null,
       status text not null check (status in ('queued','running','completed','failed','cancelled')),
       created_at text not null,
       started_at text,
       finished_at text,
       updated_at text not null,
       spec_json text not null,
       plan_json text,
       exit_code integer,
       error text,
       steps integer not null default 0,
       last_step integer,
       loss real,
       artifacts_json text not null default '[]',
       pid integer,
       backend text
     )`,
    `create index if not exists runs_created_at_idx on runs(created_at desc)`,
    `create index if not exists runs_status_idx on runs(status)`,
  ],
  // 跑后自校准（v0.2 T1）：按校准键 (family|stage|adapter|quant_base|blocks_to_swap)
  // 聚合每次运行「计划 vs 实测」的偏差，供规划时注入 engine 做只读标注。
  [
    `create table if not exists calibrations (
        key text primary key,
        family text not null,
        stage text not null,
        adapter text not null,
        quant_base text not null,
        blocks_to_swap integer not null,
        count integer not null default 0,
        sum_delta_pct real not null default 0,
        avg_delta_pct real,
        updated_at text not null
      )`,
  ],
];

const PATCH_COLUMNS: ReadonlyArray<[keyof RunPatch, string]> = [
  ["status", "status"],
  ["started_at", "started_at"],
  ["finished_at", "finished_at"],
  ["exit_code", "exit_code"],
  ["error", "error"],
  ["steps", "steps"],
  ["last_step", "last_step"],
  ["loss", "loss"],
  ["pid", "pid"],
  ["backend", "backend"],
];

function nowIso(): string {
  return new Date().toISOString();
}

function assertStatus(status: string): asserts status is RunStatus {
  if (!(RUN_STATUSES as readonly string[]).includes(status)) {
    throw new DatabaseError(`unknown run status: ${JSON.stringify(status)}`);
  }
}

/**
 * Thin typed wrapper around `DatabaseSync`. One prepared statement per
 * operation keeps the SQL in one place; every method throws `DatabaseError`
 * with the run id attached so route handlers can report which row failed.
 */
export class RunStore {
  private readonly db: DatabaseSync;
  private readonly insertStmt: StatementSync;
  private readonly getStmt: StatementSync;
  private readonly listStmt: StatementSync;
  private readonly artifactsStmt: StatementSync;
  private readonly setArtifactsStmt: StatementSync;
  private readonly stepStmt: StatementSync;
  private readonly recoverStmt: StatementSync;
  private readonly calibUpsertStmt: StatementSync;
  private readonly calibGetStmt: StatementSync;
  private readonly calibListStmt: StatementSync;
  private readonly updateStmtCache = new Map<string, StatementSync>();
  private closed = false;

  constructor(readonly path: string) {
    const sqlite = loadSqlite();
    mkdirSync(dirname(path), { recursive: true });
    this.db = new sqlite.DatabaseSync(path);
    this.db.exec("PRAGMA journal_mode = WAL");
    this.db.exec("PRAGMA synchronous = NORMAL");
    this.db.exec("PRAGMA busy_timeout = 5000");
    this.migrate();

    this.insertStmt = this.db.prepare(
      `insert into runs (id, name, status, created_at, updated_at, spec_json, plan_json, artifacts_json, backend)
       values (?, ?, ?, ?, ?, ?, ?, '[]', ?)`,
    );
    this.getStmt = this.db.prepare("select * from runs where id = ?");
    this.listStmt = this.db.prepare("select * from runs order by created_at desc, id desc limit ?");
    this.artifactsStmt = this.db.prepare("select artifacts_json from runs where id = ?");
    this.setArtifactsStmt = this.db.prepare("update runs set artifacts_json = ?, updated_at = ? where id = ?");
    this.stepStmt = this.db.prepare(
      `update runs
          set steps = steps + 1,
              last_step = ?,
              loss = coalesce(?, loss),
              updated_at = ?
        where id = ?`,
    );
    // 跑后自校准（v0.2 T1）：按校准键聚合。count 每来一次 memory_report 就 +1；
    // sum_delta_pct 累加实测偏差（未测得时传 0，不拉偏但占计数）；
    // avg_delta_pct 恒等于 sum/count，在 SQL 里同步算好。
    this.calibUpsertStmt = this.db.prepare(
      `insert into calibrations
         (key, family, stage, adapter, quant_base, blocks_to_swap, count, sum_delta_pct, avg_delta_pct, updated_at)
       values (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
       on conflict(key) do update set
         count = calibrations.count + 1,
         sum_delta_pct = calibrations.sum_delta_pct + excluded.sum_delta_pct,
         avg_delta_pct = (calibrations.sum_delta_pct + excluded.sum_delta_pct) / (calibrations.count + 1),
         updated_at = excluded.updated_at`,
    );
    this.calibGetStmt = this.db.prepare("select * from calibrations where key = ?");
    this.calibListStmt = this.db.prepare("select * from calibrations order by updated_at desc, key");
    this.recoverStmt = this.db.prepare(
      `update runs
          set status = 'failed',
              error = ?,
              finished_at = coalesce(finished_at, ?),
              updated_at = ?,
              pid = null
        where status in ('queued','running')`,
    );
  }

  private migrate(): void {
    const row = this.db.prepare("PRAGMA user_version").get() as { user_version?: number } | undefined;
    const current = Number(row?.user_version ?? 0);
    for (let version = current; version < MIGRATIONS.length; version += 1) {
      const statements = MIGRATIONS[version]!;
      this.db.exec("BEGIN");
      try {
        for (const sql of statements) this.db.exec(sql);
        this.db.exec(`PRAGMA user_version = ${version + 1}`);
        this.db.exec("COMMIT");
      } catch (cause) {
        this.db.exec("ROLLBACK");
        throw new DatabaseError(`migration ${version + 1} failed`, cause);
      }
    }
  }

  private assertOpen(): void {
    if (this.closed) throw new DatabaseError("database is closed");
  }

  insertRun(run: NewRun): void {
    this.assertOpen();
    const createdAt = run.createdAt ?? nowIso();
    const status = run.status ?? "queued";
    assertStatus(status);
    try {
      this.insertStmt.run(
        run.id,
        run.name,
        status,
        createdAt,
        createdAt,
        JSON.stringify(run.spec),
        run.plan ? JSON.stringify(run.plan) : null,
        run.backend ?? null,
      );
    } catch (cause) {
      throw new DatabaseError(`insertRun(${run.id}) failed`, cause);
    }
  }

  updateRun(id: string, patch: RunPatch): void {
    this.assertOpen();
    const sets: string[] = [];
    const values: Array<string | number | null> = [];
    for (const [key, column] of PATCH_COLUMNS) {
      const value = patch[key];
      if (value === undefined) continue;
      if (key === "status") assertStatus(String(value));
      sets.push(`${column} = ?`);
      values.push(value as string | number | null);
    }
    if (patch.artifacts !== undefined) {
      sets.push("artifacts_json = ?");
      values.push(JSON.stringify(patch.artifacts));
    }
    if (patch.plan !== undefined) {
      sets.push("plan_json = ?");
      values.push(patch.plan === null ? null : JSON.stringify(patch.plan));
    }
    if (sets.length === 0) return;
    sets.push("updated_at = ?");
    values.push(nowIso());
    values.push(id);

    const sql = `update runs set ${sets.join(", ")} where id = ?`;
    let stmt = this.updateStmtCache.get(sql);
    if (!stmt) {
      stmt = this.db.prepare(sql);
      this.updateStmtCache.set(sql, stmt);
    }
    try {
      const result = stmt.run(...values);
      if (Number(result.changes) === 0) throw new DatabaseError(`run not found: ${id}`);
    } catch (cause) {
      if (cause instanceof DatabaseError) throw cause;
      throw new DatabaseError(`updateRun(${id}) failed`, cause);
    }
  }

  getRun(id: string): RunRow | null {
    this.assertOpen();
    const row = this.getStmt.get(id) as RunRow | undefined;
    return row ?? null;
  }

  listRuns(limit = 50): RunRow[] {
    this.assertOpen();
    const safeLimit = Math.max(1, Math.min(500, Math.trunc(limit)));
    return this.listStmt.all(safeLimit) as unknown as RunRow[];
  }

  setStatus(
    id: string,
    status: RunStatus,
    extra: {
      exitCode?: number | null;
      error?: string | null;
      startedAt?: string | null;
      finishedAt?: string | null;
      pid?: number | null;
    } = {},
  ): void {
    this.assertOpen();
    assertStatus(status);
    const patch: RunPatch = { status };
    if (extra.exitCode !== undefined) patch.exit_code = extra.exitCode;
    if (extra.error !== undefined) patch.error = extra.error;
    if (extra.startedAt !== undefined) patch.started_at = extra.startedAt;
    if (extra.finishedAt !== undefined) patch.finished_at = extra.finishedAt;
    if (extra.pid !== undefined) patch.pid = extra.pid;
    this.updateRun(id, patch);
  }

  recordStep(id: string, step: number, loss: number | null): void {
    this.assertOpen();
    const result = this.stepStmt.run(step, loss, nowIso(), id);
    if (Number(result.changes) === 0) throw new DatabaseError(`run not found: ${id}`);
  }

  /** Merge artifacts into the run's list, preserving order and dropping duplicates. */
  appendArtifacts(id: string, artifacts: readonly string[]): string[] {
    this.assertOpen();
    const existing = this.getArtifacts(id);
    const merged = [...existing];
    for (const artifact of artifacts) {
      if (!merged.includes(artifact)) merged.push(artifact);
    }
    if (merged.length !== existing.length) {
      this.setArtifactsStmt.run(JSON.stringify(merged), nowIso(), id);
    }
    return merged;
  }

  private getArtifacts(id: string): string[] {
    const row = this.artifactsStmt.get(id) as { artifacts_json?: string } | undefined;
    if (!row) throw new DatabaseError(`run not found: ${id}`);
    return parseJsonArray(row.artifacts_json);
  }

  /**
   * Runs that were queued or running when the previous server process died can
   * never finish: their child process is gone. Mark them failed rather than
   * leaving the UI waiting forever.
   */
  recoverInterrupted(reason = "server restarted"): number {
    this.assertOpen();
    const now = nowIso();
    const result = this.recoverStmt.run(reason, now, now);
    return Number(result.changes);
  }

  /**
   * 跑后自校准（v0.2 T1）：把一次运行的 memory_report 按校准键聚合进 calibrations。
   * 重复运行同一校准键时 count++ 并刷新均值。delta_pct 为 null（本次未测得）时
   * 仍占一次计数，但贡献 0 到求和。
   */
  upsertCalibration(input: CalibrationInput): CalibrationRow {
    this.assertOpen();
    const key = calibrationKey({
      family: input.family,
      stage: input.stage,
      adapter: input.adapter,
      quant_base: input.quant_base,
      blocks_to_swap: input.blocks_to_swap,
    });
    const delta =
      typeof input.delta_pct === "number" && Number.isFinite(input.delta_pct) ? input.delta_pct : 0;
    try {
      this.calibUpsertStmt.run(
        key,
        input.family,
        input.stage,
        input.adapter,
        input.quant_base,
        input.blocks_to_swap,
        delta,
        delta,
        nowIso(),
      );
    } catch (cause) {
      throw new DatabaseError(`upsertCalibration(${key}) failed`, cause);
    }
    const row = this.getCalibration(key);
    if (!row) throw new DatabaseError(`upsertCalibration(${key}) row missing after write`);
    return row;
  }

  getCalibration(key: string): CalibrationRow | null {
    this.assertOpen();
    const row = this.calibGetStmt.get(key) as CalibrationRow | undefined;
    return row ?? null;
  }

  listCalibrations(): CalibrationRow[] {
    this.assertOpen();
    return this.calibListStmt.all() as unknown as CalibrationRow[];
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.db.close();
  }
}

function parseJsonArray(raw: string | undefined): string[] {
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is string => typeof item === "string");
  } catch {
    return [];
  }
}

/**
 * Convert a stored row into the wire `RunRecord`. Parsing through zod here
 * means a corrupted database row fails loudly at the boundary instead of
 * leaking a half-shaped object to the UI.
 */
export function toRunRecord(row: RunRow): RunRecord {
  let spec: RunSpec;
  let plan: RunPlan | null;
  try {
    spec = RunSpecSchema.parse(JSON.parse(row.spec_json) as unknown);
    plan = row.plan_json ? RunPlanSchema.parse(JSON.parse(row.plan_json) as unknown) : null;
  } catch (cause) {
    throw new DatabaseError(`stored spec/plan for run ${row.id} is not a valid RunSpec/RunPlan`, cause);
  }

  const candidate = {
    id: row.id,
    name: row.name,
    status: row.status,
    created_at: row.created_at,
    started_at: row.started_at,
    finished_at: row.finished_at,
    spec,
    plan,
    headline_mib: plan ? plan.memory.headline_mib : null,
    tokens_per_step: plan ? Math.round(plan.memory.tokens_per_step) : null,
    exit_code: row.exit_code,
    error: row.error,
    artifacts: parseJsonArray(row.artifacts_json),
  };
  const parsed = RunRecordSchema.safeParse(candidate);
  if (!parsed.success) {
    throw new DatabaseError(
      `stored run ${row.id} is not a valid RunRecord: ${parsed.error.issues.map((i) => `${i.path.join(".")} ${i.message}`).join("; ")}`,
    );
  }
  return parsed.data;
}