/**
 * Planner domain model: the default RunSpec, recipe prefill, and the reducer the
 * form dispatches into. Everything here is pure so it can be unit-tested without
 * React; the plan preview is just `POST /api/plan` with the current spec.
 */

import {
  RunSpecSchema,
  type DatasetSpec,
  type OutputSpec,
  type Recipe,
  type RunConfig,
  type RunSpec,
  type TrainSpec,
} from "@supertuner/core";

export function defaultConfig(): RunConfig {
  return {
    name: "",
    family: "",
    stage: "sft",
    adapter: "lora",
    precision: "bf16_pure",
    optimizer: "adamw",
    quant_base: "none",
    lora_rank: 16,
    lora_alpha: 16,
    lora_targets: ["q", "k", "v", "o", "mlp"],
    width: 480,
    height: 832,
    frames: 49,
    seq_len: 2048,
    batch: 1,
    grad_accum: 4,
    grad_checkpointing: true,
    attn_impl: "sdpa",
    cache: "both",
    blocks_to_swap: 0,
    train_blocks: null,
    auto_budget: false,
    budget_source: "auto",
    max_oom_retries: 2,
    text_encoder_quant: "none",
    text_encoder_blocks_to_swap: 0,
    vram_budget_mib: 8192,
    ram_budget_mib: 32768,
    overlap_windows_mib: 1600,
    params_b_override: null,
    rl_group_size: 4,
    rl_max_new_tokens: 64,
    rl_temperature: 0.8,
    rl_beta: 0.04,
  };
}

export function defaultTrain(): TrainSpec {
  return {
    learning_rate: 0.0001,
    scheduler: "cosine",
    warmup_steps: 20,
    weight_decay: 0.01,
    max_steps: 1000,
    epochs: 1,
    save_every: 200,
    eval_every: 100,
    seed: 42,
    grad_clip: 1,
    resume_from: null,
    dpo_beta: 0.1,
    distill: null,
  };
}

export function defaultOutput(): OutputSpec {
  return {
    dir: null,
    save_format: "safetensors",
    export: [],
    push_to_hub: false,
    hub_repo: null,
  };
}

export function defaultDataset(kind: DatasetSpec["kind"] = "auto", path = ""): DatasetSpec {
  return {
    path,
    kind,
    format: "auto",
    text_column: "text",
    caption_column: "caption",
    messages_column: "messages",
    chosen_column: "chosen",
    rejected_column: "rejected",
    prompt_column: "prompt",
    answer_column: "answer",
    video_column: "video",
    audio_column: "audio",
    max_samples: null,
    val_fraction: 0,
    notes: "",
  };
}

export function defaultRunSpec(): RunSpec {
  return {
    schema_version: 1,
    config: defaultConfig(),
    dataset: null,
    train: defaultTrain(),
    output: defaultOutput(),
    backend: null,
    licence_ack: null,
    calibration: null,
  };
}

/**
 * Recipe values win over the defaults; anything the recipe leaves out keeps its
 * planner default so the form never shows an undefined field.
 */
export function specFromRecipe(recipe: Recipe, base: RunSpec = defaultRunSpec()): RunSpec {
  const config = { ...base.config, ...recipe.config } as RunConfig;
  if (!config.name) config.name = recipe.id;
  return {
    ...base,
    config,
    train: { ...base.train, ...recipe.train },
    dataset: base.dataset,
  };
}

/** Local (pre-flight) shape check; the server still compiles the plan. */
export function validateSpec(spec: RunSpec): { ok: true; spec: RunSpec } | { ok: false; issues: string[] } {
  const parsed = RunSpecSchema.safeParse(spec);
  if (parsed.success) return { ok: true, spec: parsed.data };
  const issues = parsed.error.issues.map((issue) => {
    const path = issue.path.join(".");
    return path ? `${path}: ${issue.message}` : issue.message;
  });
  return { ok: false, issues };
}

export type PlannerAction =
  | { type: "prefill"; spec: RunSpec; recipeId: string | null }
  | { type: "patchConfig"; patch: Partial<RunConfig> }
  | { type: "patchTrain"; patch: Partial<TrainSpec> }
  | { type: "patchDataset"; patch: Partial<DatasetSpec> }
  | { type: "setDataset"; dataset: DatasetSpec | null }
  | { type: "patchOutput"; patch: Partial<OutputSpec> }
  | { type: "setBackend"; backend: string | null }
  | { type: "reset" };

export interface PlannerState {
  spec: RunSpec;
  recipeId: string | null;
  /** True once the operator has changed anything after the last prefill/reset. */
  dirty: boolean;
}

export function initialPlannerState(spec: RunSpec = defaultRunSpec(), recipeId: string | null = null): PlannerState {
  return { spec, recipeId, dirty: false };
}

export function plannerReducer(state: PlannerState, action: PlannerAction): PlannerState {
  switch (action.type) {
    case "prefill":
      return { spec: action.spec, recipeId: action.recipeId, dirty: false };

    case "reset":
      return initialPlannerState();

    case "patchConfig": {
      const config: RunConfig = { ...state.spec.config, ...action.patch };
      // Keep the run name useful: only auto-fill while the operator has not named it.
      if (action.patch.family !== undefined && state.spec.config.name.trim() === "") {
        config.name = `${action.patch.family}-${config.stage}`;
      }
      return { ...state, spec: { ...state.spec, config }, dirty: true };
    }

    case "patchTrain":
      return { ...state, spec: { ...state.spec, train: { ...state.spec.train, ...action.patch } }, dirty: true };

    case "patchDataset": {
      const dataset = { ...(state.spec.dataset ?? defaultDataset()), ...action.patch };
      return { ...state, spec: { ...state.spec, dataset }, dirty: true };
    }

    case "setDataset":
      return { ...state, spec: { ...state.spec, dataset: action.dataset }, dirty: true };

    case "patchOutput":
      return { ...state, spec: { ...state.spec, output: { ...state.spec.output, ...action.patch } }, dirty: true };

    case "setBackend":
      return { ...state, spec: { ...state.spec, backend: action.backend }, dirty: true };

    default:
      return state;
  }
}

/** Stable identity for "does this need re-planning" comparisons. */
export function specFingerprint(spec: RunSpec): string {
  return JSON.stringify(spec);
}