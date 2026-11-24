/**
 * Run event stream.
 *
 * The engine writes newline-delimited JSON to stdout: exactly one event object
 * per line, flushed per step so the UI can chart a run while it is still going.
 * The server parses each line, persists it, and re-emits it over SSE.
 */

import { z } from "zod";
import { MemoryPlanSchema } from "./contracts.js";

const base = { ts: z.number().describe("epoch milliseconds") };

export const RunStartEvent = z.object({
  ...base,
  type: z.literal("run_start"),
  run_id: z.string(),
  plan: MemoryPlanSchema.nullable().default(null),
  resumed_from_step: z.number().int().nullable().default(null),
});

export const StepEvent = z.object({
  ...base,
  type: z.literal("step"),
  step: z.number().int().nonnegative(),
  loss: z.number().nullable(),
  lr: z.number().nullable(),
  grad_norm: z.number().nullable().default(null),
  tokens_per_s: z.number().nullable().default(null),
  vram_peak_mib: z.number().nullable().default(null),
  elapsed_s: z.number().nullable().default(null),
});

export const EvalEvent = z.object({
  ...base,
  type: z.literal("eval"),
  step: z.number().int().nonnegative(),
  metrics: z.record(z.string(), z.number()),
});

export const CheckpointEvent = z.object({
  ...base,
  type: z.literal("checkpoint"),
  step: z.number().int().nonnegative(),
  path: z.string(),
  bytes: z.number().nullable().default(null),
});

export const LogEvent = z.object({
  ...base,
  type: z.literal("log"),
  level: z.enum(["debug", "info", "warn", "error"]),
  message: z.string(),
});

export const ErrorEvent = z.object({
  ...base,
  type: z.literal("error"),
  message: z.string(),
  detail: z.string().nullable().default(null),
});

export const RunEndEvent = z.object({
  ...base,
  type: z.literal("run_end"),
  status: z.enum(["completed", "failed", "cancelled"]),
  steps: z.number().int().nonnegative(),
  reason: z.string().nullable().default(null),
});

/**
 * End-of-training memory reconciliation: the planner's training-peak prediction
 * versus the measured peak the backend actually hit. Lets the UI say "the model
 * was 2% under plan" instead of hand-waving.
 */
export const MemoryReportEvent = z.object({
  ...base,
  type: z.literal("memory_report"),
  planned_mib: z.number().nullable().default(null),
  measured_peak_mib: z.number().nullable().default(null),
  delta_mib: z.number().nullable().default(null),
  delta_pct: z.number().nullable().default(null),
  device: z.string().nullable().default(null),
  family: z.string().nullable().default(null),
  stage: z.string().nullable().default(null),
  adapter: z.string().nullable().default(null),
  // 校准五元组（v0.2 T1）：加上 quant_base 与 blocks_to_swap 后，(family, stage,
  // adapter, quant_base, blocks_to_swap) 构成一次运行唯一的「校准键」。
  quant_base: z.string().nullable().default(null),
  blocks_to_swap: z.number().int().nullable().default(null),
});

/**
 * Phase boundaries around one execution step. A step is `prepare`/`cache`/
 * `train`/`export` in the plan; the engine brackets each one so the UI can show
 * "caching latents (12s)" instead of a silent gap between step events.
 */
export const PhaseStartEvent = z.object({
  ...base,
  type: z.literal("phase_start"),
  step_id: z.string(),
  phase: z.enum(["prepare", "cache", "train", "export"]),
  title: z.string(),
});

export const PhaseEndEvent = z.object({
  ...base,
  type: z.literal("phase_end"),
  step_id: z.string(),
  phase: z.enum(["prepare", "cache", "train", "export"]),
  status: z.enum(["completed", "failed", "cancelled"]),
  exit_code: z.number().int().nullable().default(null),
  reason: z.string().nullable().default(null),
});

/**
 * OOM 有限次自动重试（v0.2 T4 / G4）：训练步真实峰值撞穿预算（计划认为可行
 *  但实际 OOM）时，engine 即将以「更保守的配置」重跑整个计划。此事件在每个
 *  重试前发出一次，`attempt` 从 1 开始（第几次重试），`new_*` 是升级后候选的
 *  摘要（更多 block swap / 更低量化 / 更低显存预算）。字段与 engine 侧
 *  `events.py::Emitter.oom_retry` 一一对应。
 */

/**
 * v0.3 P0 训练/评估汇总报告：稳定性、过拟合、可选 prompt 评估与（视频）契约占位。
 * 由 engine 在训练结束后（或独立 evaluator）写出 eval_report.json 并随事件发送摘要。
 */
export const EvalReportEvent = z.object({
  ...base,
  type: z.literal("eval_report"),
  status: z.enum(["ok", "partial", "skipped", "unavailable"]).default("ok"),
  path: z.string().nullable().default(null),
  report: z.record(z.string(), z.unknown()).default({}),
});
export const OOMRetryEvent = z.object({
  ...base,
  type: z.literal("oom_retry"),
  attempt: z.number().int().nonnegative(),
  max_retries: z.number().int().nonnegative(),
  reason: z.string(),
  new_blocks_to_swap: z.number().int().nonnegative(),
  new_quant_base: z.string(),
  new_vram_budget_mib: z.number().int().nonnegative(),
});

export const RunEventSchema = z.discriminatedUnion("type", [
  RunStartEvent,
  StepEvent,
  EvalEvent,
  CheckpointEvent,
  LogEvent,
  ErrorEvent,
  PhaseStartEvent,
  PhaseEndEvent,
  RunEndEvent,
  MemoryReportEvent,
  EvalReportEvent,
  OOMRetryEvent,
]);

export type RunEvent = z.infer<typeof RunEventSchema>;
export type RunStart = z.infer<typeof RunStartEvent>;
export type Step = z.infer<typeof StepEvent>;
export type Eval = z.infer<typeof EvalEvent>;
export type Checkpoint = z.infer<typeof CheckpointEvent>;
export type Log = z.infer<typeof LogEvent>;
export type ErrorEvt = z.infer<typeof ErrorEvent>;
export type PhaseStart = z.infer<typeof PhaseStartEvent>;
export type PhaseEnd = z.infer<typeof PhaseEndEvent>;
export type RunEnd = z.infer<typeof RunEndEvent>;
export type MemoryReport = z.infer<typeof MemoryReportEvent>;
export type EvalReport = z.infer<typeof EvalReportEvent>;
export type OOMRetry = z.infer<typeof OOMRetryEvent>;

/** Parse one JSONL line. Returns the reason instead of throwing on bad input. */
export function parseEventLine(line: string): { ok: true; event: RunEvent } | { ok: false; reason: string } {
  const trimmed = line.trim();
  if (!trimmed) return { ok: false, reason: "empty line" };
  let raw: unknown;
  try {
    raw = JSON.parse(trimmed);
  } catch (e) {
    return { ok: false, reason: `invalid json: ${(e as Error).message}` };
  }
  const parsed = RunEventSchema.safeParse(raw);
  if (!parsed.success) {
    return { ok: false, reason: parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ") };
  }
  return { ok: true, event: parsed.data };
}