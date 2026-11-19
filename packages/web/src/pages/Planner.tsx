import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  ADAPTERS,
  ATTENTION_IMPLS,
  CACHE_PLANS,
  DATASET_FORMATS,
  DATASET_KINDS,
  OPTIMIZERS,
  PRECISIONS,
  QUANT_BASES,
  SCHEDULERS,
  STAGES,
  recipeById,
  type Issue,
  type PlanStep,
  type RunConfig,
} from "@supertuner/core";
import { ApiError, formatApiError, getApiClient } from "../lib/api";
import { formatDuration, formatGib, formatMib, formatNumber } from "../lib/format";
import { useDebouncedValue, useDocumentTitle, useResource } from "../lib/hooks";
import {
  defaultConfig,
  defaultDataset,
  initialPlannerState,
  plannerReducer,
  specFingerprint,
  specFromRecipe,
  validateSpec,
} from "../lib/planner-state";
import { familiesByKey } from "../lib/recipe-utils";
import { CheckGroup, NumberField, SelectField, SwitchField, TextAreaField, TextField } from "../components/Field";
import { Badge, FeasibilityBadge } from "../components/StatusBadge";
import { IssueList } from "../components/IssueList";
import { MemoryBar } from "../components/MemoryBar";
import { useToast } from "../components/Toast";
import { Icon } from "../components/icons";

const LORA_TARGETS: RunConfig["lora_targets"] = ["q", "k", "v", "o", "mlp", "gate", "up", "down"];
const EXPORT_FORMATS = ["safetensors", "gguf", "onnx", "diffusers"] as const;
/** 低显存量化档的基座权重字节数/参数（v0.2 T7）：与 engine 的
 *  _weight_bytes_per_param 保持一致（nf4 0.55 / nvfp4 0.5625 / fp8 1 / int8 1.05）。 */
const LOW_VRAM_QUANT_BYTES: Record<string, number> = { nf4: 0.55, nvfp4: 0.5625, fp8: 1, int8: 1.05 };

function humanize(value: string): string {
  const spaced = value.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function options<T extends string>(values: readonly T[]): { value: T; label: string }[] {
  return values.map((value) => ({ value, label: humanize(value) }));
}

function shellJoin(command: readonly string[]): string {
  return command
    .map((part) => (/\s/.test(part) ? `"${part.replace(/"/g, '\\"')}"` : part))
    .join(" ");
}

function kindIcon(kind: PlanStep["kind"]): "folder" | "download" | "play" | "disk" {
  switch (kind) {
    case "prepare":
      return "folder";
    case "cache":
      return "download";
    case "train":
      return "play";
    case "export":
      return "disk";
  }
}

export function Planner() {
  useDocumentTitle("Planner");
  const navigate = useNavigate();
  const toast = useToast();
  const client = getApiClient();
  const [searchParams] = useSearchParams();
  const recipeParam = searchParams.get("recipe");

  const [state, dispatch] = useReducer(plannerReducer, undefined, () => initialPlannerState());
  const [starting, setStarting] = useState(false);
  const [licenceAccepted, setLicenceAccepted] = useState(false);
  const appliedRecipe = useRef<string | null>(null);

  const families = useResource((signal) => client.families(signal), []);
  const system = useResource((signal) => client.system(signal), []);

  const familyList = useMemo(() => (families.data ? [...families.data.llm, ...families.data.video] : []), [families.data]);
  const familyMap = useMemo(() => familiesByKey(familyList), [familyList]);

  // Recipe prefill: config + train come from the curated entry, everything else
  // keeps the planner default so no field renders undefined.
  useEffect(() => {
    if (!recipeParam || appliedRecipe.current === recipeParam) return;
    const recipe = recipeById(recipeParam);
    if (!recipe) {
      toast.push("warn", `Unknown recipe "${recipeParam}"`, "Falling back to the default spec.");
      return;
    }
    appliedRecipe.current = recipeParam;
    dispatch({ type: "prefill", spec: specFromRecipe(recipe), recipeId: recipe.id });
  }, [recipeParam, toast]);

  // Pick the first known family when the spec has none (fresh planner visit).
  useEffect(() => {
    const data = families.data;
    const first = data ? data.llm[0] ?? data.video[0] : undefined;
    if (!first || state.spec.config.family) return;
    dispatch({ type: "patchConfig", patch: { family: first.key } });
  }, [families.data, state.spec.config.family]);

  // Default the VRAM budget to the card that is actually installed, once.
  const seededBudget = useRef(false);
  useEffect(() => {
    if (seededBudget.current) return;
    const gpu = system.data?.env.gpus[0];
    if (!gpu || gpu.vram_mib <= 0) return;
    seededBudget.current = true;
    if (state.spec.config.vram_budget_mib === defaultConfig().vram_budget_mib) {
      dispatch({ type: "patchConfig", patch: { vram_budget_mib: gpu.vram_mib } });
    }
  }, [system.data, state.spec.config.vram_budget_mib]);

  const { spec } = state;
  const config = spec.config;
  const validity = validateSpec(spec);
  const fingerprint = specFingerprint(spec);
  const debouncedFingerprint = useDebouncedValue(fingerprint, 320);
  const settled = debouncedFingerprint === fingerprint;

  const planResource = useResource((signal) => client.plan(spec, signal), [debouncedFingerprint], {
    enabled: validity.ok && settled,
  });

  const plan = validity.ok && settled ? planResource.data ?? null : null;
  const planIssues = useMemo<Issue[]>(() => {
    if (!plan) return [];
    const combined = [...plan.issues, ...plan.execution.issues];
    const seen = new Set<string>();
    return combined.filter((issue) => {
      const key = `${issue.code}|${issue.field ?? ""}|${issue.message}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [plan]);

  const errorIssues = planResource.error instanceof ApiError ? planResource.error.issues : [];
  const localIssues = validity.ok ? [] : validity.issues;

  const planning = validity.ok && (!settled || planResource.status === "loading");
  const needsLicence = config.family === "minimax-h3";
  const canStart =
    validity.ok &&
    settled &&
    planResource.status === "ready" &&
    plan?.feasible === true &&
    !starting &&
    (!needsLicence || licenceAccepted);

  const gpu = system.data?.env.gpus[0];
  const family = familyMap.get(config.family);
  const kind = family?.kind ?? (recipeParam?.includes("video") ? "video" : "llm");
  const isVideo = kind === "video";
  const isLlm = kind === "llm";

  const dataset = spec.dataset;

  const startRun = async (): Promise<void> => {
    if (!validity.ok) {
      toast.push("warn", "Spec is incomplete", validity.issues[0]);
      return;
    }
    setStarting(true);
    try {
      const runSpec = needsLicence
        ? {
            ...spec,
            licence_ack: {
              family: "minimax-h3",
              license_id: "MiniMax H3 Community License",
              version: "1",
              accepted_at: new Date().toISOString(),
            },
          }
        : spec;
      const run = await client.createRun(runSpec);
      toast.push("ok", `Run queued: ${run.name}`, run.id);
      navigate(`/runs/${encodeURIComponent(run.id)}`);
    } catch (error) {
      const detail = formatApiError(error);
      if (error instanceof ApiError && error.issues.length > 0) {
        toast.push("error", "Run rejected by the compiler", error.issues[0]?.message ?? detail);
      } else {
        toast.push("error", "Could not start the run", detail);
      }
    } finally {
      setStarting(false);
    }
  };

  const copySpec = async (): Promise<void> => {
    const copied = await navigator.clipboard?.writeText(JSON.stringify(spec, null, 2)).then(
      () => true,
      () => false,
    );
    toast.push(copied ? "ok" : "warn", copied ? "RunSpec copied as JSON" : "Clipboard unavailable");
  };

  return (
    <>
      <div className="page-head">
        <div className="page-head-copy">
          <span className="eyebrow">Planner</span>
          <h1>Compose a RunSpec</h1>
          <p>
            Every change re-plans against the engine's memory model. The plan you see is the plan that will be
            compiled — nothing starts until it fits.
          </p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-sm" onClick={copySpec}>
            <Icon name="copy" size={13} />
            Copy JSON
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => {
              appliedRecipe.current = null;
              dispatch({ type: "reset" });
            }}
          >
            <Icon name="refresh" size={13} />
            Reset
          </button>
        </div>
      </div>

      {state.recipeId ? (
        <div className="panel">
          <div className="panel-body row spread">
            <p className="small">
              <Icon name="recipes" size={13} /> Prefilled from recipe <span className="mono">{state.recipeId}</span>
            </p>
            <Badge tone="accent">recipe</Badge>
          </div>
        </div>
      ) : null}

      <div className="grid-dash">
        <div className="stack">
          <section className="panel" aria-labelledby="spec-heading">
            <div className="panel-head">
              <h2 id="spec-heading">Run identity</h2>
            </div>
            <div className="panel-body grid-2">
              <TextField
                label="Run name"
                value={config.name}
                onChange={(name) => dispatch({ type: "patchConfig", patch: { name } })}
                placeholder="qwen3-8b-sft"
                hint="Appears in the run list and on disk."
              />
              <SelectField
                label="Model family"
                value={config.family}
                options={familyList.map((entry) => ({
                  value: entry.key,
                  label: `${entry.key} · ${entry.kind}${entry.params_b ? ` · ${entry.params_b}B` : ""}`,
                }))}
                onChange={(familyKey) => dispatch({ type: "patchConfig", patch: { family: familyKey } })}
                hint={
                  families.status === "error"
                    ? `Family list unavailable: ${formatApiError(families.error)}`
                    : familyList.length === 0
                      ? "No model families reported by the engine."
                      : undefined
                }
                placeholder="select a family"
              />
              <SelectField
                label="Stage"
                value={config.stage}
                options={options(STAGES)}
                onChange={(stage) => dispatch({ type: "patchConfig", patch: { stage } })}
              />
              <SelectField
                label="Adapter"
                value={config.adapter}
                options={options(ADAPTERS)}
                onChange={(adapter) => dispatch({ type: "patchConfig", patch: { adapter } })}
                hint={config.adapter === "none" ? "Full fine-tune: no adapter weights are added." : undefined}
              />
              <TextField
                label="Backend override"
                value={spec.backend ?? ""}
                onChange={(backend) => dispatch({ type: "setBackend", backend: backend.trim() ? backend : null })}
                placeholder="default"
                mono
                hint="Leave blank to let the engine choose."
              />
            </div>
          </section>

          <section className="panel" aria-labelledby="memory-heading">
            <div className="panel-head">
              <h2 id="memory-heading">Memory strategy</h2>
            </div>
            <div className="panel-body grid-2">
              <SelectField
                label="Weight precision"
                value={config.precision}
                options={options(PRECISIONS)}
                onChange={(precision) => dispatch({ type: "patchConfig", patch: { precision } })}
              />
              {LOW_VRAM_QUANT_BYTES[config.quant_base] !== undefined ? (
                <p className="small muted">
                  低显存量化档 <span className="mono">{config.quant_base}</span>：基座权重约{" "}
                  {LOW_VRAM_QUANT_BYTES[config.quant_base]} B/参数（4-bit/8-bit 压缩，省显存）
                </p>
              ) : null}
              <SelectField
                label="Quantisation base"
                value={config.quant_base}
                options={options(QUANT_BASES)}
                onChange={(quant_base) => dispatch({ type: "patchConfig", patch: { quant_base } })}
                hint={config.adapter === "qlora" ? "QLoRA expects nf4 or int8." : undefined}
              />
              <SelectField
                label="Optimizer"
                value={config.optimizer}
                options={options(OPTIMIZERS)}
                onChange={(optimizer) => dispatch({ type: "patchConfig", patch: { optimizer } })}
              />
              <SelectField
                label="Attention"
                value={config.attn_impl}
                options={options(ATTENTION_IMPLS)}
                onChange={(attn_impl) => dispatch({ type: "patchConfig", patch: { attn_impl } })}
              />
              <SelectField
                label="Cache plan"
                value={config.cache}
                options={options(CACHE_PLANS)}
                onChange={(cache) => dispatch({ type: "patchConfig", patch: { cache } })}
              />
              <NumberField
                label="Blocks to swap"
                value={config.blocks_to_swap}
                min={0}
                max={512}
                onChange={(blocks_to_swap) => dispatch({ type: "patchConfig", patch: { blocks_to_swap } })}
                hint={family?.supports_block_swap === false ? "This family does not support block swap." : "0 keeps every block resident."}
              />
              <SwitchField
                label="Auto budget"
                checked={config.auto_budget}
                onChange={(auto_budget) => dispatch({ type: "patchConfig", patch: { auto_budget } })}
                disabled={family?.supports_block_swap !== true}
                hint={
                  family?.supports_block_swap === true
                    ? "Let the engine pick quant_base + block swap that fit."
                    : "Only available for block-swap video families."
                }
              />
              <NumberField
                label="VRAM budget"
                unit="MiB"
                value={config.vram_budget_mib}
                min={1024}
                max={262144}
                step={512}
                onChange={(vram_budget_mib) => dispatch({ type: "patchConfig", patch: { vram_budget_mib } })}
                hint={gpu ? `Installed card reports ${formatMib(gpu.vram_mib)}.` : "GPU not reported yet."}
              />
              <NumberField
                label="RAM budget"
                unit="MiB"
                value={config.ram_budget_mib}
                min={1024}
                max={1048576}
                step={1024}
                onChange={(ram_budget_mib) => dispatch({ type: "patchConfig", patch: { ram_budget_mib } })}
                hint="Host memory allowed for offload."
              />
              <NumberField
                label="Window overlap"
                unit="MiB"
                value={config.overlap_windows_mib}
                min={0}
                max={65536}
                step={128}
                onChange={(overlap_windows_mib) => dispatch({ type: "patchConfig", patch: { overlap_windows_mib } })}
              />
            </div>
          </section>

          <section className="panel" aria-labelledby="adapter-heading">
            <div className="panel-head">
              <h2 id="adapter-heading">Adapter &amp; text encoder</h2>
            </div>
            <div className="panel-body grid-2">
              <NumberField
                label="LoRA rank"
                value={config.lora_rank}
                min={1}
                max={512}
                onChange={(lora_rank) => dispatch({ type: "patchConfig", patch: { lora_rank } })}
                disabled={config.adapter === "none"}
              />
              <NumberField
                label="LoRA alpha"
                value={config.lora_alpha}
                min={1}
                max={1024}
                onChange={(lora_alpha) => dispatch({ type: "patchConfig", patch: { lora_alpha } })}
                disabled={config.adapter === "none"}
              />
              <CheckGroup
                label="LoRA targets"
                options={LORA_TARGETS.map((target) => ({ value: target, label: target }))}
                selected={config.lora_targets}
                onChange={(targets) =>
                  dispatch({ type: "patchConfig", patch: { lora_targets: targets as RunConfig["lora_targets"] } })
                }
              />
              <div className="stack-sm">
                <SelectField
                  label="Text encoder quant"
                  value={config.text_encoder_quant}
                  options={options(QUANT_BASES)}
                  onChange={(text_encoder_quant) => dispatch({ type: "patchConfig", patch: { text_encoder_quant } })}
                  disabled={isLlm}
                />
                <NumberField
                  label="Text encoder swap"
                  value={config.text_encoder_blocks_to_swap}
                  min={0}
                  max={512}
                  onChange={(text_encoder_blocks_to_swap) =>
                    dispatch({ type: "patchConfig", patch: { text_encoder_blocks_to_swap } })
                  }
                  disabled={isLlm}
                />
              </div>
            </div>
          </section>

          <section className="panel" aria-labelledby="shape-heading">
            <div className="panel-head">
              <h2 id="shape-heading">{isVideo ? "Resolution & frames" : "Sequence shape"}</h2>
            </div>
            <div className="panel-body grid-2">
              <NumberField
                label="Width"
                unit="px"
                value={config.width}
                min={64}
                max={4096}
                step={16}
                onChange={(width) => dispatch({ type: "patchConfig", patch: { width } })}
                disabled={!isVideo}
              />
              <NumberField
                label="Height"
                unit="px"
                value={config.height}
                min={64}
                max={4096}
                step={16}
                onChange={(height) => dispatch({ type: "patchConfig", patch: { height } })}
                disabled={!isVideo}
              />
              <NumberField
                label="Frames"
                value={config.frames}
                min={1}
                max={1024}
                onChange={(frames) => dispatch({ type: "patchConfig", patch: { frames } })}
                disabled={!isVideo}
                hint={isVideo ? undefined : "Only video families sample frames."}
              />
              <NumberField
                label="Sequence length"
                unit="tokens"
                value={config.seq_len}
                min={16}
                max={131072}
                step={128}
                onChange={(seq_len) => dispatch({ type: "patchConfig", patch: { seq_len } })}
                disabled={!isLlm}
              />
              <TextField
                label="目标 block 范围"
                value={config.train_blocks ?? ""}
                onChange={(train_blocks) =>
                  dispatch({
                    type: "patchConfig",
                    patch: { train_blocks: train_blocks.trim() ? train_blocks.trim() : null },
                  })
                }
                placeholder="后端默认"
                mono
                disabled={!isVideo}
                hint={isVideo ? "留空表示后端默认，例如 20-49 或 3-12,22,31-33" : "仅视频家族可设置。"}
              />
            </div>
          </section>

          <section className="panel" aria-labelledby="batch-heading">
            <div className="panel-head">
              <h2 id="batch-heading">Batching</h2>
            </div>
            <div className="panel-body grid-2">
              <NumberField
                label="Batch size"
                value={config.batch}
                min={1}
                max={1024}
                onChange={(batch) => dispatch({ type: "patchConfig", patch: { batch } })}
              />
              <NumberField
                label="Gradient accumulation"
                value={config.grad_accum}
                min={1}
                max={4096}
                onChange={(grad_accum) => dispatch({ type: "patchConfig", patch: { grad_accum } })}
                hint={`Effective batch ${formatNumber(config.batch * config.grad_accum)}`}
              />
              <SwitchField
                label="Gradient checkpointing"
                checked={config.grad_checkpointing}
                onChange={(grad_checkpointing) => dispatch({ type: "patchConfig", patch: { grad_checkpointing } })}
                hint="Saves activation memory at the cost of a second forward pass."
              />
              <NumberField
                label="Params override"
                unit="B"
                value={config.params_b_override ?? 0}
                min={0}
                max={2000}
                step={0.1}
                onChange={(params_b_override) =>
                  dispatch({ type: "patchConfig", patch: { params_b_override: params_b_override > 0 ? params_b_override : null } })
                }
                hint="0 uses the family registry value."
              />
            </div>
          </section>

          <section className="panel" aria-labelledby="dataset-heading">
            <div className="panel-head">
              <h2 id="dataset-heading">Dataset</h2>
            </div>
            <div className="panel-body">
              <SwitchField
                label="Attach a dataset"
                checked={dataset !== null}
                onChange={(checked) =>
                  dispatch({ type: "setDataset", dataset: checked ? defaultDataset(spec.config.stage === "sft" ? "chat" : "auto") : null })
                }
                hint="Without a dataset the engine runs its smoke profile."
              />
              {dataset ? (
                <div className="grid-2 stack">
                  <TextField
                    label="Dataset path"
                    value={dataset.path}
                    onChange={(path) => dispatch({ type: "patchDataset", patch: { path } })}
                    placeholder="C:\\data\\corpus.jsonl"
                    mono
                  />
                  <SelectField
                    label="Kind"
                    value={dataset.kind}
                    options={options(DATASET_KINDS)}
                    onChange={(kind) => dispatch({ type: "patchDataset", patch: { kind } })}
                  />
                  <SelectField
                    label="Format"
                    value={dataset.format}
                    options={options(DATASET_FORMATS)}
                    onChange={(format) => dispatch({ type: "patchDataset", patch: { format } })}
                  />
                  <NumberField
                    label="Max samples"
                    value={dataset.max_samples ?? 0}
                    min={0}
                    onChange={(max_samples) =>
                      dispatch({ type: "patchDataset", patch: { max_samples: max_samples > 0 ? max_samples : null } })
                    }
                    hint="0 loads every sample."
                  />
                  <TextField
                    label="Text column"
                    value={dataset.text_column}
                    onChange={(text_column) => dispatch({ type: "patchDataset", patch: { text_column } })}
                    mono
                    disabled={dataset.kind === "video"}
                  />
                  <TextField
                    label="Caption column"
                    value={dataset.caption_column}
                    onChange={(caption_column) => dispatch({ type: "patchDataset", patch: { caption_column } })}
                    mono
                  />
                  <TextField
                    label="Messages column"
                    value={dataset.messages_column}
                    onChange={(messages_column) => dispatch({ type: "patchDataset", patch: { messages_column } })}
                    mono
                  />
                  <TextField
                    label="Video column"
                    value={dataset.video_column}
                    onChange={(video_column) => dispatch({ type: "patchDataset", patch: { video_column } })}
                    mono
                  />
                  <TextField
                    label="Audio column"
                    value={dataset.audio_column}
                    onChange={(audio_column) => dispatch({ type: "patchDataset", patch: { audio_column } })}
                    mono
                  />
                  <NumberField
                    label="Validation fraction"
                    value={dataset.val_fraction}
                    min={0}
                    max={0.5}
                    step={0.01}
                    onChange={(val_fraction) => dispatch({ type: "patchDataset", patch: { val_fraction } })}
                  />
                  <TextAreaField
                    label="Dataset notes"
                    value={dataset.notes}
                    onChange={(notes) => dispatch({ type: "patchDataset", patch: { notes } })}
                    hint="Passed to the engine as-is."
                  />
                </div>
              ) : null}
            </div>
          </section>

          <section className="panel" aria-labelledby="train-heading">
            <div className="panel-head">
              <h2 id="train-heading">Training</h2>
            </div>
            <div className="panel-body grid-2">
              <NumberField
                label="Learning rate"
                value={spec.train.learning_rate}
                min={0.000001}
                max={1}
                step={0.00001}
                onChange={(learning_rate) => dispatch({ type: "patchTrain", patch: { learning_rate } })}
              />
              <SelectField
                label="Scheduler"
                value={spec.train.scheduler}
                options={options(SCHEDULERS)}
                onChange={(scheduler) => dispatch({ type: "patchTrain", patch: { scheduler } })}
              />
              <NumberField
                label="Warmup steps"
                value={spec.train.warmup_steps}
                min={0}
                max={100000}
                onChange={(warmup_steps) => dispatch({ type: "patchTrain", patch: { warmup_steps } })}
              />
              <NumberField
                label="Weight decay"
                value={spec.train.weight_decay}
                min={0}
                max={1}
                step={0.001}
                onChange={(weight_decay) => dispatch({ type: "patchTrain", patch: { weight_decay } })}
              />
              <NumberField
                label="Max steps"
                value={spec.train.max_steps}
                min={1}
                max={10000000}
                step={100}
                onChange={(max_steps) => dispatch({ type: "patchTrain", patch: { max_steps } })}
              />
              <NumberField
                label="Epochs"
                value={spec.train.epochs}
                min={1}
                max={1000}
                onChange={(epochs) => dispatch({ type: "patchTrain", patch: { epochs } })}
              />
              <NumberField
                label="Save every"
                unit="steps"
                value={spec.train.save_every}
                min={1}
                max={1000000}
                onChange={(save_every) => dispatch({ type: "patchTrain", patch: { save_every } })}
              />
              <NumberField
                label="Eval every"
                unit="steps"
                value={spec.train.eval_every}
                min={1}
                max={1000000}
                onChange={(eval_every) => dispatch({ type: "patchTrain", patch: { eval_every } })}
              />
              <NumberField
                label="Seed"
                value={spec.train.seed}
                min={0}
                max={4294967295}
                onChange={(seed) => dispatch({ type: "patchTrain", patch: { seed } })}
              />
              <NumberField
                label="Gradient clip"
                value={spec.train.grad_clip}
                min={0}
                max={1000}
                step={0.1}
                onChange={(grad_clip) => dispatch({ type: "patchTrain", patch: { grad_clip } })}
              />
              <TextField
                label="Resume from"
                value={spec.train.resume_from ?? ""}
                onChange={(resume_from) =>
                  dispatch({ type: "patchTrain", patch: { resume_from: resume_from.trim() ? resume_from : null } })
                }
                placeholder="checkpoint path or run id"
                mono
              />
              {config.stage === "rl" ? (
                <>
                  <NumberField
                    label="GRPO group size"
                    value={config.rl_group_size}
                    min={2}
                    max={16}
                    onChange={(rl_group_size) => dispatch({ type: "patchConfig", patch: { rl_group_size } })}
                  />
                  <NumberField
                    label="Max new tokens"
                    value={config.rl_max_new_tokens}
                    min={8}
                    max={256}
                    onChange={(rl_max_new_tokens) => dispatch({ type: "patchConfig", patch: { rl_max_new_tokens } })}
                  />
                  <NumberField
                    label="Rollout temperature"
                    value={config.rl_temperature}
                    min={0.05}
                    max={2}
                    step={0.05}
                    onChange={(rl_temperature) => dispatch({ type: "patchConfig", patch: { rl_temperature } })}
                  />
                  <NumberField
                    label="KL beta"
                    value={config.rl_beta}
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={(rl_beta) => dispatch({ type: "patchConfig", patch: { rl_beta } })}
                  />
                </>
              ) : null}
              {config.stage === "preference" ? (
                <NumberField
                  label="DPO beta"
                  value={spec.train.dpo_beta}
                  min={0.0001}
                  max={10}
                  step={0.01}
                  onChange={(dpo_beta) => dispatch({ type: "patchTrain", patch: { dpo_beta } })}
                />
              ) : null}
            </div>
          </section>

          <section className="panel" aria-labelledby="output-heading">
            <div className="panel-head">
              <h2 id="output-heading">Output</h2>
            </div>
            <div className="panel-body grid-2">
              <TextField
                label="Output directory"
                value={spec.output.dir ?? ""}
                onChange={(dir) => dispatch({ type: "patchOutput", patch: { dir: dir.trim() ? dir : null } })}
                placeholder="default run directory"
                mono
              />
              <SelectField
                label="Save format"
                value={spec.output.save_format}
                options={[
                  { value: "safetensors", label: "safetensors" },
                  { value: "pt", label: "PyTorch (.pt)" },
                ]}
                onChange={(save_format) => dispatch({ type: "patchOutput", patch: { save_format } })}
              />
              <CheckGroup
                label="Export after training"
                options={EXPORT_FORMATS.map((value) => ({ value, label: value }))}
                selected={spec.output.export}
                onChange={(exported) =>
                  dispatch({ type: "patchOutput", patch: { export: exported as (typeof EXPORT_FORMATS)[number][] } })
                }
              />
              <div className="stack-sm">
                <SwitchField
                  label="Push to Hugging Face"
                  checked={spec.output.push_to_hub}
                  onChange={(push_to_hub) => dispatch({ type: "patchOutput", patch: { push_to_hub } })}
                />
                <TextField
                  label="Hub repository"
                  value={spec.output.hub_repo ?? ""}
                  onChange={(hub_repo) => dispatch({ type: "patchOutput", patch: { hub_repo: hub_repo.trim() ? hub_repo : null } })}
                  placeholder="user/model-name"
                  mono
                  disabled={!spec.output.push_to_hub}
                />
              </div>
            </div>
          </section>
        </div>

        <aside className="stack planner-aside" aria-label="Plan preview">
          <section className="panel">
            <div className="panel-head">
              <h2>Plan preview</h2>
              <span className="small faint" aria-live="polite">
                {!validity.ok
                  ? "spec incomplete"
                  : planning
                    ? "planning…"
                    : planResource.status === "error"
                      ? "plan failed"
                      : plan
                        ? "up to date"
                        : "waiting"}
              </span>
            </div>

            <div className="panel-body">
              {!validity.ok ? (
                <div className="stack-sm">
                  <p className="small muted">The form needs attention before the server can plan it:</p>
                  <ul className="issues">
                    {localIssues.map((issue) => (
                      <li key={issue} className="issue issue-error">
                        <span className="issue-title">{issue}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : planning && !plan ? (
                <p className="small muted">Compiling memory model…</p>
              ) : planResource.status === "error" ? (
                <div className="stack-sm">
                  <p className="small" role="alert">
                    {formatApiError(planResource.error)}
                  </p>
                  {errorIssues.length > 0 ? <IssueList issues={errorIssues} /> : null}
                </div>
              ) : plan ? (
                <div className="stack">
                  <div className="row spread">
                    <FeasibilityBadge
                      fits={plan.feasible}
                      detail={`peak ${formatMib(plan.memory.training_peak_mib)} of ${formatMib(plan.memory.vram_budget_mib)}`}
                    />
                    <span className="small faint">
                      {plan.estimated_minutes === null ? "duration unknown" : `≈ ${formatDuration(plan.estimated_minutes * 60)}`}
                    </span>
                  </div>

                  <div className="stat-row">
                    <div className="stat">
                      <span className="stat-label">training peak</span>
                      <span className="stat-value sm">{formatMib(plan.memory.training_peak_mib)}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">sampling peak</span>
                      <span className="stat-value sm">{formatMib(plan.memory.sampling_peak_mib)}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">headroom</span>
                      <span className="stat-value sm">{formatMib(plan.memory.headroom_mib)}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">tokens / step</span>
                      <span className="stat-value sm">{formatNumber(plan.memory.tokens_per_step)}</span>
                    </div>
                  </div>

                  {plan.memory.budget_source === "measured" ? (
                    <p className="small">
                      <Badge tone="info" title="预算取自本机实测的 VRAM / RAM">
                        预算来源 实测
                        {plan.memory.budget_measured_vram_mib
                          ? `（${formatMib(plan.memory.budget_measured_vram_mib)}）`
                          : ""}
                      </Badge>
                    </p>
                  ) : (
                    <p className="small">
                      <Badge tone="neutral" title="预算来自手填的 vram / ram budget">
                        预算来源 手填
                      </Badge>
                    </p>
                  )}

                  {system.data ? (
                    <p className="small">
                      {(["triton", "liger", "sageattention"] as const).map((name) => (
                        <Badge
                          key={name}
                          tone={system.data?.env.capabilities?.[name] === true ? "ok" : "neutral"}
                          title={`加速 kernel ${name}：${system.data?.env.capabilities?.[name] ? "已检测到" : "未检测到"}`}
                        >
                          {name}
                        </Badge>
                      ))}
                    </p>
                  ) : null}

                  {plan.memory.calibration ? (
                    <p className="small">
                      已校准 {plan.memory.calibration.count} 次
                      {plan.memory.calibration.avg_delta_pct !== null
                        ? `（平均偏差 ${plan.memory.calibration.avg_delta_pct.toFixed(1)}%）`
                        : ""}
                    </p>
                  ) : null}

                  {plan.memory.auto_candidates.length > 0 ? (
                    <div className="stack-sm">
                      <span className="eyebrow">可训练选项 / what you can train</span>
                      <div className="table-wrap">
                        <table className="data">
                          <thead>
                          <tr>
                            <th scope="col">量化</th>
                            <th scope="col">换块</th>
                            <th scope="col">block 范围</th>
                            <th scope="col" className="num">峰值</th>
                            <th scope="col">状态</th>
                          </tr>
                          </thead>
                          <tbody>
                            {plan.memory.auto_candidates.map((candidate, index) => {
                              const chosen =
                                index ===
                                plan.memory.auto_candidates.findIndex((c) => c.feasible);
                              return (
                                <tr key={`${candidate.quant_base}|${candidate.blocks_to_swap}|${candidate.train_blocks ?? "all"}`} className={chosen ? "row-chosen" : undefined}>
                                  <td className="mono">{candidate.quant_base}</td>
                                  <td className="mono">{candidate.blocks_to_swap}</td>
                                  <td className="mono">{candidate.train_blocks ?? "全部"}</td>
                                  <td className="num mono">{formatMib(candidate.headline_mib)}</td>
                                  <td>
                                    {candidate.feasible ? (
                                      <Badge tone="ok">可行</Badge>
                                    ) : (
                                      <span className="small muted">{candidate.note || "缺显存"}</span>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : null}

                  <MemoryBar memory={plan.memory} />

                  {plan.memory.swap_floor_s !== null && plan.memory.swap_roundtrip_floor_s !== null ? (
                    <p className="small">
                      block swap 搬运时间带：单向 ≈ {plan.memory.swap_floor_s.toFixed(1)}s/step，往返 ≈{" "}
                      {plan.memory.swap_roundtrip_floor_s.toFixed(1)}s/step
                      <span className="faint">（swap 搬运下限，不含注意力/计算）</span>
                    </p>
                  ) : null}

                  {plan.memory.risks.length > 0 ? (
                    <ul className="stack-sm small muted" aria-label="Memory risks">
                      {plan.memory.risks.map((risk) => (
                        <li key={risk} className="row">
                          <Icon name="warning" size={14} />
                          <span>{risk}</span>
                        </li>
                      ))}
                    </ul>
                  ) : null}

                  <IssueList issues={planIssues} />

                  {plan.execution.notes.length > 0 ? (
                    <ul className="stack-sm small faint" aria-label="Compiler notes">
                      {plan.execution.notes.map((note) => (
                        <li key={note}>{note}</li>
                      ))}
                    </ul>
                  ) : null}

                  {plan.execution.expected_artifacts.length > 0 ? (
                    <div className="stack-sm">
                      <span className="eyebrow">Expected artifacts</span>
                      <ul className="artifacts">
                        {plan.execution.expected_artifacts.map((artifact) => (
                          <li key={artifact} className="artifact mono">
                            {artifact}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
              ) : (
                <p className="small muted">Pick a family to begin planning.</p>
              )}
            </div>

            {needsLicence ? (
              <div className="licence-gate" role="note">
                <strong>MiniMax H3 Community License（不是 Apache-2.0）</strong>
                <p className="small">
                  地域排除 EU、UK、韩国、美国；年收入超过 2000 万美元需要书面授权；禁止用 H3 或其输出训练其它模型（含蒸馏）。
                </p>
                <label className="checkline small">
                  <input
                    type="checkbox"
                    checked={licenceAccepted}
                    onChange={(event) => setLicenceAccepted(event.target.checked)}
                  />{" "}
                  我确认所在地区与用途符合上述条款
                </label>
              </div>
            ) : null}
            <div className="panel-foot">
              <span className="small faint">
                {plan ? `${plan.execution.steps.length} compile steps` : `${formatGib(config.vram_budget_mib, 0)} budget`}
              </span>
              <button type="button" className="btn btn-primary btn-sm" disabled={!canStart} onClick={() => void startRun()}>
                <Icon name={starting ? "refresh" : "play"} size={13} />
                {starting ? "Starting…" : "Start run"}
              </button>
            </div>
          </section>

          <section className="panel" aria-labelledby="steps-heading">
            <div className="panel-head">
              <h2 id="steps-heading">Compile steps</h2>
            </div>
            <div className="panel-body">
              {plan && plan.execution.steps.length > 0 ? (
                <ol className="steps">
                  {plan.execution.steps.map((step, index) => (
                    <li key={step.id} className={step.kind === "train" ? "step step-train" : "step"}>
                      <span className="step-marker" aria-hidden="true">
                        <Icon name={kindIcon(step.kind)} size={14} />
                      </span>
                      <div className="step-body">
                        <div className="row spread">
                          <span className="step-title">
                            {index + 1}. {step.title}
                          </span>
                          <Badge tone={step.optional ? "neutral" : "info"}>{step.kind}</Badge>
                        </div>
                        {step.description ? <p className="step-desc">{step.description}</p> : null}
                        {step.command ? (
                          <div className="step-command">
                            <code>{shellJoin(step.command)}</code>
                          </div>
                        ) : null}
                        <span className="small faint">
                          {step.optional ? "optional" : "required"}
                          {step.estimated_mib ? ` · ≈ ${formatMib(step.estimated_mib)}` : ""}
                        </span>
                      </div>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="small faint">Compile steps appear once the plan is feasible.</p>
              )}
            </div>
          </section>
        </aside>
      </div>
    </>
  );
}