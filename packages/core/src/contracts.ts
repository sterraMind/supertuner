/**
 * Wire contracts shared with the Python engine.
 *
 * The engine owns these shapes (engine/src/supertuner_engine/schema.py). This
 * file mirrors them field-for-field in snake_case so a JSON object validates
 * identically on both sides — no silent rename layer between the two runtimes.
 */

import { z } from "zod";

export const STAGES = ["pretrain", "sft", "preference", "rl", "distill"] as const;
export const ADAPTERS = ["none", "lora", "qlora"] as const;
export const ATTENTION_IMPLS = ["flash", "sdpa", "math"] as const;
export const PRECISIONS = ["bf16_pure", "fp32_master", "fp32"] as const;
export const OPTIMIZERS = ["adamw", "adamw_8bit", "adamw_paged_8bit", "came"] as const;
export const QUANT_BASES = ["none", "nf4", "fp8", "int8", "nvfp4"] as const;
export const CACHE_PLANS = ["none", "text", "latents", "both"] as const;
export const DATASET_KINDS = ["auto", "text", "chat", "preference", "video", "image"] as const;
export const DATASET_FORMATS = ["auto", "jsonl", "parquet", "directory", "webdataset"] as const;
export const SCHEDULERS = ["constant", "cosine", "linear", "wsd"] as const;
export const ISSUE_SEVERITIES = ["error", "warning", "info"] as const;
export const STEP_KINDS = ["prepare", "cache", "train", "export"] as const;
// 预算来源（v0.2 T2 实测进预算 / G3 = AP-01）。RunConfig 用「意图」
// （auto = 用机器实测值、manual = 用手填值）；MemoryPlan 用「解析结果」
// （measured = 实际采用了实测值、manual = 实际用的是手填/默认值）。
export const BUDGET_SOURCES = ["auto", "manual"] as const;
export const RESOLVED_BUDGET_SOURCES = ["measured", "manual"] as const;

export const StageSchema = z.enum(STAGES);
export const AdapterSchema = z.enum(ADAPTERS);
export const AttnImplSchema = z.enum(ATTENTION_IMPLS);
export const PrecisionSchema = z.enum(PRECISIONS);
export const OptimizerSchema = z.enum(OPTIMIZERS);
export const QuantBaseSchema = z.enum(QUANT_BASES);
export const CachePlanSchema = z.enum(CACHE_PLANS);
export const BudgetSourceSchema = z.enum(BUDGET_SOURCES);
export const ResolvedBudgetSourceSchema = z.enum(RESOLVED_BUDGET_SOURCES);

export type Stage = z.infer<typeof StageSchema>;
export type Adapter = z.infer<typeof AdapterSchema>;
export type AttnImpl = z.infer<typeof AttnImplSchema>;
export type Precision = z.infer<typeof PrecisionSchema>;
export type Optimizer = z.infer<typeof OptimizerSchema>;
export type QuantBase = z.infer<typeof QuantBaseSchema>;
export type CachePlan = z.infer<typeof CachePlanSchema>;
export type BudgetSource = z.infer<typeof BudgetSourceSchema>;
export type ResolvedBudgetSource = z.infer<typeof ResolvedBudgetSourceSchema>;
export type DatasetKind = (typeof DATASET_KINDS)[number];
export type DatasetFormat = (typeof DATASET_FORMATS)[number];
export type Scheduler = (typeof SCHEDULERS)[number];

export const LoraTargetSchema = z.enum(["q", "k", "v", "o", "mlp", "gate", "up", "down"]);

export const RunConfigSchema = z.object({
  name: z.string().min(1).max(120),
  family: z.string().min(1),
  stage: StageSchema,
  adapter: AdapterSchema.default("lora"),
  precision: PrecisionSchema.default("bf16_pure"),
  optimizer: OptimizerSchema.default("adamw"),
  quant_base: QuantBaseSchema.default("none"),
  lora_rank: z.number().int().min(1).max(512).default(16),
  lora_alpha: z.number().int().min(1).max(1024).default(16),
  lora_targets: z.array(LoraTargetSchema).default(["q", "k", "v", "o", "mlp"]),
  width: z.number().int().min(64).max(4096).default(480),
  height: z.number().int().min(64).max(4096).default(832),
  frames: z.number().int().min(1).max(1024).default(49),
  seq_len: z.number().int().min(16).max(131072).default(2048),
  rl_group_size: z.number().int().min(1).max(32).default(4),
  rl_max_new_tokens: z.number().int().min(8).max(1024).default(64),
  rl_temperature: z.number().min(0.01).max(2).default(0.8),
  rl_beta: z.number().min(0).max(1).default(0.04),
  batch: z.number().int().min(1).max(1024).default(1),
  grad_accum: z.number().int().min(1).max(4096).default(4),
  grad_checkpointing: z.boolean().default(true),
  attn_impl: AttnImplSchema.default("sdpa"),
  cache: CachePlanSchema.default("both"),
  blocks_to_swap: z.number().int().min(0).max(512).default(0),
  /** Target transformer-block window for a video LoRA, e.g. "20-49" or
   *  "3-12,22,31-33". null keeps the backend default (no block restriction). */
  train_blocks: z.string().nullable().default(null),
  /** When true and the family is a block-swap-capable video model, the engine
   *  searches quant_base x blocks_to_swap x train_blocks and runs the cheapest
   *  feasible candidate instead of the typed values. Off by default. */
  auto_budget: z.boolean().default(false),
  /** 预算来源（v0.2 T2 实测进预算 / G3 = AP-01）："auto" = 机器实测 VRAM/RAM
   *  可用时优先用实测值，否则回落默认；"manual" = 直接使用上面显式给出的
   *  vram_budget_mib / ram_budget_mib。 */
  budget_source: BudgetSourceSchema.default("auto"),
  /** OOM 有限次自动重试（v0.2 T4 / G4）：训练步真实峰值撞穿预算（计划认为可行
   *  但实际 OOM）时，engine 用更保守的候选（更多 block swap / 更低量化 + 更低
   *  预算）重跑整个计划，最多重试 max_oom_retries 次；0 表示禁用重试。 */
  max_oom_retries: z.number().int().min(0).max(16).default(2),
  text_encoder_quant: QuantBaseSchema.default("none"),
  text_encoder_blocks_to_swap: z.number().int().min(0).max(512).default(0),
  vram_budget_mib: z.number().int().min(1024).max(262144).default(8192),
  ram_budget_mib: z.number().int().min(1024).max(1048576).default(32768),
  overlap_windows_mib: z.number().int().min(0).max(65536).default(1600),
  params_b_override: z.number().positive().nullable().default(null),
});

export type RunConfig = z.infer<typeof RunConfigSchema>;

export const MemoryLineSchema = z.object({
  label: z.string(),
  mib: z.number(),
  note: z.string().default(""),
});

/**
 * 跑后自校准（v0.2 T1）：server 按「校准键」查到该配置历史上「计划 vs 实测」的
 * 聚合偏差后，规划前注入给 engine 的标注信息。engine 只读它做备注/建议，
 * 绝不据此改动 vram_budget_mib。
 */
export const CalibrationSchema = z.object({
  /** 该校准键累计观测到的次数（完成过并回报 memory_report 的运行数）。 */
  count: z.number().int().nonnegative().default(0),
  /** 历史平均偏差百分比（实测 - 计划）/ 计划 * 100；尚无有效样本时为 null。 */
  avg_delta_pct: z.number().nullable().default(null),
});
export type Calibration = z.infer<typeof CalibrationSchema>;

/**
 * 校准五元组 -> 字符串键。(family, stage, adapter, quant_base, blocks_to_swap)
 * 五个字段全部来自 RunConfig / memory_report 事件，顺序固定，保证 server 规划
 * 时的查键与事件 upsert 的键一致。
 */
export function calibrationKey(input: {
  family: string;
  stage: string;
  adapter: string;
  quant_base: string;
  blocks_to_swap: number;
}): string {
  return [input.family, input.stage, input.adapter, input.quant_base, String(input.blocks_to_swap)].join("|");
}

/**
 * 统一自动选择的一个候选（v0.2 T3 / G2 = AP-02）。engine 的
 * `AutoCandidate` 模型的 TS 镜像；字段名两侧完全一致。
 */
export const AutoCandidateSchema = z.object({
  quant_base: z.string(),
  blocks_to_swap: z.number().int(),
  train_blocks: z.string().nullable().default(null),
  feasible: z.boolean(),
  headline_mib: z.number(),
  note: z.string().default(""),
});
export type AutoCandidate = z.infer<typeof AutoCandidateSchema>;

export const MemoryPlanSchema = z.object({
  feasible: z.boolean(),
  vram_budget_mib: z.number(),
  ram_budget_mib: z.number().default(32768),
  training_peak_mib: z.number(),
  sampling_peak_mib: z.number(),
  headline_mib: z.number(),
  headroom_mib: z.number(),
  ram_peak_mib: z.number().default(0),
  ram_headroom_mib: z.number().default(0),
  lines: z.array(MemoryLineSchema),
  risks: z.array(z.string()).default([]),
  tokens_per_step: z.number(),
  notes: z.array(z.string()).default([]),
  // 跑后自校准（v0.2 T1）：engine 只读它做标注，不改动预算。
  calibration: CalibrationSchema.nullable().default(null),
  // 实测进预算（v0.2 T2 / G3 = AP-01）：预算的解析来源与采用的实测值。
  // "measured" = 预算取自机器实测；"manual" = 手填/默认。
  budget_source: ResolvedBudgetSourceSchema.default("manual"),
  budget_measured_vram_mib: z.number().int().nullable().default(null),
  budget_measured_ram_mib: z.number().int().nullable().default(null),
  // 统一自动选择（v0.2 T3 / G2 = AP-02）：auto_budget 评估过的全部候选。
  // 可行在前（按 headline_mib 升序）、不可行在后（按缺口升序）；
  // 仅当 auto_budget 真正执行了搜索时非空。
  auto_candidates: z.array(AutoCandidateSchema).default([]),
  // 单向 block swap 搬运时间带（v0.2 T6 / AP-08）：只统计换出权重在 PCIe 上的
  // 搬运时间，不含注意力/计算。无 block swap 时保持 null/""。
  // swap_floor_s：单向（H2D only）每步搬运下限（秒）。
  swap_floor_s: z.number().nullable().default(null),
  // swap_roundtrip_floor_s：往返（H2D+D2H）下限 = 2 x 单向（秒）。
  swap_roundtrip_floor_s: z.number().nullable().default(null),
  step_time_band_s: z.string().default(""),
});

export type MemoryLine = z.infer<typeof MemoryLineSchema>;
export type MemoryPlan = z.infer<typeof MemoryPlanSchema>;

/** Engine envelope for `supertuner-engine plan`. */
export const PlanResponseSchema = z.discriminatedUnion("ok", [
  z.object({ ok: z.literal(true), plan: MemoryPlanSchema }),
  z.object({ ok: z.literal(false), error: z.string() }),
]);

export type PlanResponse = z.infer<typeof PlanResponseSchema>;

export const ModelFamilySchema = z.object({
  key: z.string(),
  kind: z.enum(["llm", "video"]),
  params_b: z.number().nullable(),
  hidden: z.number(),
  layers: z.number(),
  heads: z.number(),
  kv_heads: z.number().default(0),
  head_dim: z.number(),
  ffn: z.number(),
  vocab: z.number().default(0),
  tie_embeddings: z.boolean().default(false),
  text_encoder_params_b: z.number().default(0),
  vae_params_b: z.number().default(0),
  patch: z.tuple([z.number(), z.number(), z.number()]).default([1, 2, 2]),
  vae_temporal_stride: z.number().default(4),
  vae_spatial_stride: z.number().default(8),
  latent_channels: z.number().default(16),
  fps: z.number().default(0),
  pruned_params_b: z.number().nullable().default(null),
  adaln_params_b: z.number().default(0),
  supports_block_swap: z.boolean().default(false),
  text_encoder_layers: z.number().default(0),
  approx: z.boolean().default(false),
});

export type ModelFamily = z.infer<typeof ModelFamilySchema>;

export const FamiliesResponseSchema = z.object({
  llm: z.array(ModelFamilySchema),
  video: z.array(ModelFamilySchema),
});

export type FamiliesResponse = z.infer<typeof FamiliesResponseSchema>;

// ---------------------------------------------------------------------------
// Run specification
// ---------------------------------------------------------------------------

export const DatasetSpecSchema = z.object({
  path: z.string().min(1),
  kind: z.enum(DATASET_KINDS).default("auto"),
  format: z.enum(DATASET_FORMATS).default("auto"),
  text_column: z.string().default("text"),
  caption_column: z.string().default("caption"),
  messages_column: z.string().default("messages"),
  chosen_column: z.string().default("chosen"),
  rejected_column: z.string().default("rejected"),
  prompt_column: z.string().default("prompt"),
  answer_column: z.string().default("answer"),
  video_column: z.string().default("video"),
  audio_column: z.string().default("audio"),
  max_samples: z.number().int().positive().nullable().default(null),
  val_fraction: z.number().min(0).max(0.5).default(0),
  notes: z.string().default(""),
});

export const DistillSpecSchema = z.object({
  teacher: z.string().nullable().default(null),
  mode: z.enum(["dmd", "dmad", "dmd2"]).default("dmd"),
  student_steps: z.number().int().min(1).max(64).default(4),
  guidance: z.number().min(0).max(20).default(1),
});

export const TrainSpecSchema = z.object({
  learning_rate: z.number().positive().max(1).default(1e-4),
  scheduler: z.enum(SCHEDULERS).default("cosine"),
  warmup_steps: z.number().int().min(0).default(20),
  weight_decay: z.number().min(0).default(0.01),
  max_steps: z.number().int().positive().default(1000),
  epochs: z.number().int().positive().default(1),
  save_every: z.number().int().positive().default(200),
  eval_every: z.number().int().positive().default(100),
  seed: z.number().int().default(42),
  grad_clip: z.number().positive().default(1),
  resume_from: z.string().nullable().default(null),
  dpo_beta: z.number().positive().default(0.1),
  distill: DistillSpecSchema.nullable().default(null),
});

export const OutputSpecSchema = z.object({
  dir: z.string().nullable().default(null),
  save_format: z.enum(["safetensors", "pt"]).default("safetensors"),
  export: z.array(z.enum(["safetensors", "gguf", "onnx", "diffusers"])).default([]),
  push_to_hub: z.boolean().default(false),
  hub_repo: z.string().nullable().default(null),
});

export const LicenceAckSchema = z.object({
  family: z.string(),
  license_id: z.string(),
  version: z.string().default("1"),
  accepted_at: z.string(),
});

export const RunSpecSchema = z.object({
  schema_version: z.literal(1).default(1),
  config: RunConfigSchema,
  dataset: DatasetSpecSchema.nullable().default(null),
  train: TrainSpecSchema.default({}),
  output: OutputSpecSchema.default({}),
  backend: z.string().nullable().default(null),
  licence_ack: LicenceAckSchema.nullable().default(null),
  /** 规划前由 server 按校准键查库注入；engine 只读它做备注，不改动预算。 */
  calibration: CalibrationSchema.nullable().default(null),
});

export type DatasetSpec = z.infer<typeof DatasetSpecSchema>;
export type DistillSpec = z.infer<typeof DistillSpecSchema>;
export type TrainSpec = z.infer<typeof TrainSpecSchema>;
export type OutputSpec = z.infer<typeof OutputSpecSchema>;
export type RunSpec = z.infer<typeof RunSpecSchema>;

// ---------------------------------------------------------------------------
// Compilation / plugin execution
// ---------------------------------------------------------------------------

export const IssueSchema = z.object({
  severity: z.enum(ISSUE_SEVERITIES),
  code: z.string(),
  message: z.string(),
  hint: z.string().default(""),
  field: z.string().nullable().default(null),
});

export const PlanStepSchema = z.object({
  id: z.string(),
  kind: z.enum(STEP_KINDS),
  title: z.string(),
  description: z.string().default(""),
  command: z.array(z.string()).nullable().default(null),
  cwd: z.string().nullable().default(null),
  env: z.record(z.string(), z.string()).default({}),
  optional: z.boolean().default(false),
  estimated_mib: z.number().nullable().default(null),
});

export const ExecutionPlanSchema = z.object({
  backend: z.string(),
  backend_version: z.string().default(""),
  steps: z.array(PlanStepSchema).default([]),
  issues: z.array(IssueSchema).default([]),
  notes: z.array(z.string()).default([]),
  expected_artifacts: z.array(z.string()).default([]),
});

export const RunPlanSchema = z.object({
  feasible: z.boolean(),
  memory: MemoryPlanSchema,
  execution: ExecutionPlanSchema,
  issues: z.array(IssueSchema).default([]),
  estimated_minutes: z.number().nullable().default(null),
});

export const CompileResponseSchema = z.discriminatedUnion("ok", [
  z.object({ ok: z.literal(true), plan: RunPlanSchema }),
  z.object({ ok: z.literal(false), error: z.string(), issues: z.array(IssueSchema).default([]) }),
]);

export type Issue = z.infer<typeof IssueSchema>;
export type PlanStep = z.infer<typeof PlanStepSchema>;
export type ExecutionPlan = z.infer<typeof ExecutionPlanSchema>;
export type RunPlan = z.infer<typeof RunPlanSchema>;
export type CompileResponse = z.infer<typeof CompileResponseSchema>;

export const EnvPackageSchema = z.object({
  name: z.string(),
  version: z.string().nullable().default(null),
  ok: z.boolean(),
  hint: z.string().default(""),
});

export const GpuInfoSchema = z.object({
  index: z.number(),
  name: z.string(),
  vram_mib: z.number(),
  driver: z.string().default(""),
  vram_total_mib: z.number().default(0),
  vram_free_mib: z.number().default(0),
});

export const EnvReportSchema = z.object({
  python: z.string(),
  platform: z.string(),
  packages: z.array(EnvPackageSchema).default([]),
  gpus: z.array(GpuInfoSchema).default([]),
  ram_mib: z.number().default(0),
  ffmpeg: z.string().nullable().default(null),
  warnings: z.array(z.string()).default([]),
  // 可选加速 kernel 探测结果（v0.2 T7）：键固定为 triton / liger / sageattention。
  capabilities: z.record(z.string(), z.boolean()).default({}),
});

export type EnvReport = z.infer<typeof EnvReportSchema>;

// ---------------------------------------------------------------------------
// Server-side records (persisted in SQLite, streamed over SSE)
// ---------------------------------------------------------------------------

export const RUN_STATUSES = ["queued", "running", "completed", "failed", "cancelled"] as const;
export type RunStatus = (typeof RUN_STATUSES)[number];

export const RunRecordSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: z.enum(RUN_STATUSES),
  created_at: z.string(),
  started_at: z.string().nullable().default(null),
  finished_at: z.string().nullable().default(null),
  spec: RunSpecSchema,
  plan: RunPlanSchema.nullable().default(null),
  headline_mib: z.number().nullable().default(null),
  tokens_per_step: z.number().int().nullable().default(null),
  exit_code: z.number().int().nullable().default(null),
  error: z.string().nullable().default(null),
  artifacts: z.array(z.string()).default([]),
});

export type RunRecord = z.infer<typeof RunRecordSchema>;

export const ArtifactSchema = z.object({
  path: z.string(),
  bytes: z.number().int().nonnegative().default(0),
  modified_at: z.string().default(""),
});

export type Artifact = z.infer<typeof ArtifactSchema>;

export const SystemResponseSchema = z.object({
  env: EnvReportSchema,
  engine: z.object({
    version: z.string(),
    python: z.string().nullable().default(null),
    path: z.string().nullable().default(null),
  }),
  server: z.object({ version: z.string() }),
  home: z.string(),
  runs_dir: z.string(),
  models_dir: z.string(),
  datasets_dir: z.string(),
  disk: z
    .object({
      path: z.string(),
      total_mib: z.number(),
      free_mib: z.number(),
    })
    .default({ path: "", total_mib: 0, free_mib: 0 }),
});

export type SystemResponse = z.infer<typeof SystemResponseSchema>;

export const RecipeSchema = z.object({
  id: z.string(),
  name: z.string(),
  tier: z.string(),
  kind: z.enum(["llm", "video"]),
  family: z.string(),
  summary: z.string(),
  vram_mib: z.number(),
  ram_mib: z.number(),
  caveats: z.array(z.string()).default([]),
  /** verified = has an end-to-end run on this repo; compile_only = plan/argv only;
   *  unverified = neither (e.g. needs a dependency that is not installed). */
  status: z.enum(["verified", "compile_only", "unverified"]).default("unverified"),
  /** License of the *model weights*, not of the training tool. */
  model_license: z.string().default("Apache-2.0"),
  config: RunConfigSchema.partial().required({ name: true, family: true, stage: true }),
  train: TrainSpecSchema.partial().default({}),
  dataset_kind: z.enum(DATASET_KINDS).default("auto"),
});

export type Recipe = z.infer<typeof RecipeSchema>;