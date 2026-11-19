# SuperTuner architecture

Status: v0.1 implementation contract. This document is normative for the first
build: if code and this document disagree, either fix the code or change this
document in the same change.

## 1. What we are building

SuperTuner is a **single-GPU training control plane**. It does not re-implement
DiT or LLM training kernels. It owns four things that existing trainers do not:

1. **A memory planner that is honest before the run starts.** Inputs: model,
   stage, adapter, precision, quantisation, resolution, frames, block swap,
   cache plan, VRAM and RAM budgets. Output: a line-item budget, a feasibility
   verdict, and concrete next actions when the run does not fit.
2. **A compiler from a portable RunSpec to a concrete backend plan.** The plan
   is inspectable JSON: ordered steps, argv arrays, env, expected artifacts.
3. **A supervisor.** Process isolation, JSONL events, run directories,
   cancellation, resume, artifact discovery, SSE to the UI.
4. **Recipes and guardrails.** Curated, tested combinations per VRAM tier.
   Unsupported combinations are refused with a reason, not allowed to fail
   three hours into a run.

Non-goals for v0.1: multi-node training, cloud provisioning, a model hub,
training from scratch beyond small LLMs, DMD/DMAD execution, arbitrary
user-supplied plugin code from the browser.

## 2. Component map

```
┌──────────────────────────── Browser (React + Vite) ─────────────────────────┐
│ Recipes · Planner preview · Run monitor · Plugins · System · Settings       │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ /api (JSON, SSE)
┌───────────────────────────────▼─────────────────────────────────────────────┐
│ Fastify server (TypeScript, Node 22+)                                       │
│  routes · run supervisor · SQLite (node:sqlite) · event fan-out · static UI  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ spawn argv, JSONL stdout
┌───────────────────────────────▼─────────────────────────────────────────────┐
│ Python engine (uv project `engine/`)                                        │
│  schema/planner · plugin registry · compiler · runtime · env probe          │
│  built-in backends: native LLM (transformers+peft), musubi adapter (video)  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ subprocess per step
┌───────────────────────────────▼─────────────────────────────────────────────┐
│ Training backends                                                           │
│  native LLM worker (torch) · musubi-tuner modules (Wan/H3/…) · exporters     │
└─────────────────────────────────────────────────────────────────────────────┘
```

There is exactly one source of truth per concern:

| Concern | Owner |
| --- | --- |
| Memory arithmetic | `engine/src/supertuner_engine/plan.py` |
| Wire shapes | `engine/.../schema.py` (Python) mirrored by `packages/core/src/*.ts` |
| Backend selection / argv | engine plugins |
| Run state / scheduling | server (`packages/server/src/runs.ts`) |
| Rendering / interaction | web (`packages/web/src`) |
| Paths | `SUPERTUNER_*` env vars only (see `packages/core/src/config.ts`) |

## 3. Data flow

1. UI builds a `RunSpec` (nested: `config`, `dataset`, `train`, `output`,
   optional `backend`).
2. `POST /api/plan` → server writes the spec to a temp file and runs
   `supertuner-engine compile --spec <file> --json`. The engine returns
   `RunPlan { feasible, memory, execution, issues }`. The UI renders the
   memory lines and the ordered step list.
3. `POST /api/runs` re-compiles server-side (never trust a client plan),
   persists the spec/plan in SQLite, and enqueues it. The supervisor starts
   `supertuner-engine run --spec <run>/spec.json --run-dir <run>` with
   `--events-stdout`.
4. The engine emits one JSON object per line on stdout (`RunEvent`). The
   server validates each line, appends it to `events.jsonl`, updates the run
   row, and fans it out to SSE subscribers.
5. Checkpoints/adapters appear under `<run>/output`; the server scans them on
   completion and records artifacts.

The engine never talks to the database and never binds a port. The server never
does memory math and never builds backend argv.

## 4. RunSpec

Flat `RunConfig` remains the planner input and the public shape consumed by
existing tests. A run adds `dataset`, `train`, `output` and `backend` around it.
All paths are absolute after the server resolves `~` and rejects relative paths.

```jsonc
{
  "schema_version": 1,
  "config": {
    "name": "h3-lora-8gb",
    "family": "minimax-h3",
    "stage": "sft",
    "adapter": "lora",
    "quant_base": "int8",
    "width": 320, "height": 576, "frames": 49,
    "blocks_to_swap": 48,
    "cache": "both",
    "vram_budget_mib": 8192,
    "ram_budget_mib": 32768
  },
  "dataset": { "path": "D:/datasets/demo", "kind": "video", "format": "directory" },
  "train": { "learning_rate": 0.0001, "max_steps": 800 },
  "output": { "export": ["safetensors"] },
  "backend": null
}
```

`backend=null` means auto-select: the compiler asks each enabled backend whether
it supports the capability required by `(family.kind, stage, adapter)`, then
picks the highest-priority backend with no error-level issue.

## 5. Plugin model

Plugins are Python objects registered through entry points in the group
`supertuner.plugins` plus built-ins compiled into the engine. A plugin is one of:

- `backend`: compiles and executes a RunSpec (native LLM, musubi adapter).
- `model`: advertises a family and its geometry (H3, Wan, Qwen…). The planner's
  registry is seeded from these manifests; family geometry lives in
  `schema.py` for v0.1 and may move to plugins in v0.2.
- `dataset`: validates/prepares a dataset kind.
- `strategy`: stage-specific logic (SFT, DPO, DMD) that a backend can call.
- `optimizer`, `exporter`, `ui`: optional extensions.

A backend implements:

```python
class Backend(Protocol):
    manifest: PluginManifest
    def probe(self, ctx: Context) -> list[Issue]: ...
    def supports(self, spec: RunSpec) -> bool: ...
    def compile(self, spec: RunSpec, ctx: Context) -> ExecutionPlan: ...
    def run(self, spec: RunSpec, plan: ExecutionPlan, ctx: Context, emit: Emit) -> None: ...
```

Rules:

- **No import-time heavy dependencies.** `probe()` and `compile()` must work
  without torch so the UI can plan on a machine that has not installed the
  training stack. Import torch/diffusers/musubi inside `run()`.
- **argv arrays, never shell strings.** The runtime uses `subprocess` with
  `shell=False`; the server uses `spawn` with `shell=False`.
- **No path guessing.** `Context` carries `repo_root`, `run_dir`, `models_dir`,
  `datasets_dir`, `cache_dir`, all derived from `SUPERTUNER_*` env vars.
- **Manifests are inert.** The UI only reads `capabilities`, `families`,
  `requires`, `experimental`. It never executes plugin code.
- **Refuse loudly.** A backend returns error `Issue`s for unsupported
  stage/adapter/quant combinations; it must not silently downgrade.

Entry-point failures are collected, not fatal: `supertuner-engine plugins`
returns `{plugins, errors}` and the UI shows a degraded state.

## 6. Execution plan

`ExecutionPlan.steps` is ordered. Each step is one of:

- `prepare`: dataset validation/metadata (fast, no GPU)
- `cache`: VAE latents, text embeddings, tokenised data (GPU/CPU, may be long)
- `train`: the actual optimisation
- `export`: convert/save artifacts

A step carries either a `command` (external backend, e.g. musubi) or is executed
by the backend in-process (`command=null`, native LLM). The runtime emits
`phase_start` / `phase_end` around each step and forwards backend events.

## 7. Events and persistence

One JSON object per line, discriminated by `type` (see
`packages/core/src/events.ts`). Required event types for v0.1:

`run_start`, `phase_start`, `step`, `eval`, `checkpoint`, `log`, `error`,
`phase_end`, `run_end`.

`step` carries `step`, `loss`, `lr`, `grad_norm`, `tokens_per_s`, `vram_mib`,
`elapsed_s`. The UI chart consumes only these fields.

Run directory layout (created by the server, passed to the engine):

```
<SUPERTUNER_HOME>/runs/<run_id>/
  spec.json
  plan.json
  events.jsonl
  logs/engine.stderr.log
  output/            # checkpoints, adapter, export
```

SQLite lives at `<SUPERTUNER_HOME>/supertuner.db`. Tables: `runs`, `events` is
not stored separately (events.jsonl is the record); `runs` holds status,
timestamps, spec/plan JSON, exit code, error, artifact list.

## 8. Memory planner model (v0.1)

Line items (VRAM): resident base weights, LoRA weights, optimizer state,
gradients, fp32 master (optional), activations (+chunked CE logits), optionally
text encoder/VAE. RAM: swapped DiT blocks, swapped text-encoder blocks.

Key facts encoded (2026-11-11):

- MiniMax-H3 is a 33B dense omni transformer; ~13B sits in AdaLN branches.
  LoRA training can use the pruned 20B transformer (40GB bf16, 21GB int8).
  Full fine-tuning must load 33B (66GB bf16) and is out of scope for one
  consumer card.
- H3 visual VAE: spatial 16x, temporal 4x, 24 channels, patch 1x2x2 → 32x
  effective spatial. 24fps.
- musubi block swap keeps ≥2 blocks resident and supports 48/50 on H3.
  It requires gradient checkpointing and LoRA.
- For H3 LoRA at 320x576x49f (2.3k tokens), pruned int8 + 48/50 swap +
  text/latent cache, the plan peaks at ~3.6GB VRAM and ~19GB RAM: the 8GB
  proof point. At 480p (~20k tokens) the activation term alone is ~7GB, so
  8GB is not enough; 24GB is.
- Full-parameter video training and DMD/DMAD on one consumer card are refused
  in v0.1 with an explanation, not silently attempted.

The planner is intentionally conservative (±15% for approximate families).
Every risk string tells the user a concrete lever (8-bit optimizer, cache,
quantisation, swap, lower resolution).

## 9. Server API (v0.1)

- `GET /api/health` → `{ok, version}`
- `GET /api/system` → env probe (GPU, RAM, torch/peft/musubi/ffmpeg) + paths
- `GET /api/families` → `{llm:[...], video:[...]}`
- `GET /api/plugins` → `{plugins, errors}`
- `GET /api/recipes` → curated recipes
- `POST /api/plan` `{spec}` → `RunPlan` (compiled, never executed)
- `POST /api/runs` `{spec}` → 201 `RunRecord`; 422 with plan when infeasible
- `GET /api/runs?limit=` → `{runs}`
- `GET /api/runs/:id` → `RunRecord`
- `GET /api/runs/:id/events` → JSONL replay (JSON array)
- `GET /api/runs/:id/events/stream` → SSE (`text/event-stream`)
- `POST /api/runs/:id/cancel` → cancels queued or running run
- `GET /api/runs/:id/artifacts` → list of files under `output/`
- `POST /api/engine/reload` → re-probe plugins (dev)

Error shape: `{error: string, issues?: Issue[]}` with 4xx/5xx.

## 10. Testing strategy

- **Engine unit**: planner arithmetic (golden scenarios), schema round-trips,
  capability selection, compile argv for every recipe, event serialisation,
  dry-run runtime.
- **Contract**: Python-generated JSON validates against the zod schemas in
  `packages/core` (golden fixtures under `engine/tests/fixtures/`).
- **Server**: Fastify `inject()` against a temp `SUPERTUNER_HOME` and a fake
  engine executable; supervisor lifecycle, SSE, cancel, path rejection.
- **Web**: theme store, API client validation, planner form reducer, SVG chart
  path, recipe selection.
- **Hardware smoke**: `engine/tests/test_smoke_gpu.py` skipped unless
  `SUPERTUNER_RUN_GPU_TESTS=1`; tiny LLM SFT on the local card.
- **Regression**: root `pnpm test` + `uv run --project engine pytest`; a
  dry-run of every built-in recipe must compile without errors.

## 11. Security and safety

- The server binds `127.0.0.1` by default. Do not expose it.
- RunSpec paths are resolved and must be absolute; the server rejects paths
  outside configured roots unless `SUPERTUNER_ALLOW_ANY_PATH=1`.
- Backend argv is assembled as arrays. No `shell=True`, no `eval`.
- Plugin manifests are data. Installing a plugin is a deliberate Python
  environment change, not a browser action.
- Model licences: H3 is **not** Apache-2.0 (MiniMax H3 Community License).
  Applicable territory excludes the EU, UK, Korea and the US; commercial use
  above USD 20M revenue needs written authorisation, and the licence forbids
  using H3 or its outputs to improve any other AI model (so distillation from
  H3 is out of scope). The UI must show the licence, its territory limits and
  a link before a H3 run can be started; the server records the acknowledgement
  in the run spec.
- The planner must never claim a run fits when RAM for block swap does not.

## 12. Decisions (ADR summary)

| # | Decision | Rationale | Rejected alternative |
| --- | --- | --- | --- |
| 1 | Wrap musubi-tuner for video, native transformers+peft for LLM | musubi already supports Wan/H3 block swap and caching; rewriting DiT training would take months and lose upstream fixes. Native LLM SFT is small and gives event-level control. | Rewrite everything in one framework; wrap diffusers only (no H3 block swap). |
| 2 | Python engine as a subprocess, not embedded in Node | torch lives in Python; process isolation makes crashes/cancellation/OOM survivable; the CLI is independently testable. | Node native bindings; HTTP microservice (more ports/auth). |
| 3 | Fastify + SQLite (`node:sqlite`) | zero native build, single process, SSE trivial, Node 22+ built-in DB. | Electron + Python sidecar (packaging cost); Postgres (ops cost). |
| 4 | Nested RunSpec around flat RunConfig | keeps planner contract/tests stable, separates data/training/output concerns. | Break RunConfig into nested config everywhere. |
| 5 | Plan before run, server re-compiles | prevents client-side forged plans and stale UI; one source of truth. | trust client plan. |
| 6 | No DMD/DMAD execution in v0.1 | requires a teacher/student loop and much more VRAM; advertise capability as `experimental` and refuse. | fake it with a LoRA labelled DMD. |

## 12.1 Capability matrix (v0.1, what the code actually does)

| Stage / adapter | LLM | Video |
| --- | --- | --- |
| pretrain | executable (native backend, full or LoRA) | refused: no video pretrain recipe |
| sft | executable (native backend; verified on 8GB, see §15) | video LoRA is the SFT analogue: compile preview only (argv + plan); execution needs musubi-tuner installed |
| lora / qlora | executable (LoRA needs no bitsandbytes; QLoRA needs it) | compile + supervise only; execution requires musubi-tuner installed and the weights present |
| preference (DPO) | executable with LoRA/QLoRA (`_train_dpo`); full fine-tuning is refused; unit-tested, full GPU run pending | refused |
| rl (GRPO) | executable with LoRA/QLoRA (`_train_grpo`): group rollouts, exact-match reward from the dataset answer, KL to the adapter-disabled reference; full fine-tuning is refused; unit-tested and run end to end on Qwen2.5-0.5B-Instruct (copy-number task, 6 steps) | refused |
| distill (DMD/DMAD) | refused at compile with an error issue (`DistillSpec` is already in the schema) | refused |

"Refused" means the compiler returns `feasible=false` plus an error-level
`Issue` naming the missing capability; it never fabricates a successful run.
The refused stages are the P1 backlog, not hidden behaviour.

## 13. Known risks

- **H3 licence and 210GB checkpoint** make the 8GB H3 story a compile preview,
  not a turnkey product. Mitigation: recipes carry a `status` column
  (verified / compile_preview / unverified), and the UI requires a licence
  acknowledgement before an H3 run is accepted.
- **Memory estimates carry a fixed ~0.9 GB CUDA-context term** (override with
  `SUPERTUNER_CUDA_CONTEXT_MIB`). The planner lists it as its own line so a small
  model is not reported 50% too low.
- **Plan feasibility covers every step**, not just the training step: H3's
  cache-text step loads the 32B text encoder (~48 GB bf16, ~25 GB int8), so an
  8 GB H3 plan reports `E_STEP_OOM` instead of claiming the run fits.
- **Wan2.2 TI2V 5B is not supported by v0.1**: upstream musubi-tuner has no
  `ti2v-5B` task; the family was removed from the adapter and its recipe removed.
- **musubi CLI/log format is not a stable API.** Mitigation: adapter pins a
  tested version range, compiles argv from one module, and parses only
  documented progress lines; a dry-run prints the exact command.
- **Block swap throughput on laptop PCIe** can make 8GB H3 impractically slow.
  Mitigation: planner reports the theoretical lower bound and a warning; UI
  shows tokens/s after the first steps.
- **bitsandbytes on Windows** is fragile. Mitigation: QLoRA probe reports
  unavailable instead of crashing; LoRA remains the default.
- **Node `node:sqlite` is experimental** and may need a flag on older Node.
  Mitigation: require Node >=22.5, document the flag; isolate behind `db.ts`.
- **Approximate H3 geometry** (hidden/ffn) is not public in config.json.
  Mitigation: `approx=true`, ±15% note, and a `params_b_override` escape hatch.

## 14. Delivery order

1. Contracts + planner (done in this change).
2. Engine plugin registry + compiler + dry-run runtime + env probe.
3. Native LLM SFT backend (LoRA/QLoRA; SFT, DPO and GRPO implemented; full-parameter preference/RL refused).
4. musubi adapter (compile + argv + cache/train steps; execution gated on
   installed musubi + 24GB+).
5. Fastify API + supervisor + SSE.
6. React UI: system, recipes, planner, run monitor, plugins, settings, theme.
7. Tests: planner goldens, compile goldens, server lifecycle, web unit.
8. Hardware smoke + docs.

## 15. Verified hardware smoke (2026-11-11)

Machine: RTX 3070 Ti Laptop 8GB (driver 621.14), i7-12700H, 31.7GiB RAM,
Windows 11, Python 3.12.8, torch 2.15.0+cu132.

- `POST /api/plan` for `qwen3-0.6b` / LoRA / `seq_len=512` / `batch=2`:
  feasible, training peak 1751.5 MiB, headroom 4840.5 MiB, 2048 tokens/step.
  The real run peaked at 1787 MiB — the planner is within 2% of measured on
  this recipe (a LoRA forward/backward with chunked cross-entropy).
- Real SFT through the API: wikitext-2 (10.8MB, 36,718 lines), 20 steps,
  completed in 29.9s, 442.9 tokens/s, 1787 MiB peak, adapter + checkpoint
  artifacts listed by `/api/runs/:id/artifacts`.
- Same recipe from the CLI: 30 steps in 49.5s, loss 2.75–3.98, 2.15–2.73 GiB
  peak, `adapter_model.safetensors` 40,422,168 bytes.
- **Windows allocator caveat:** the default caching allocator dies at ~1.7 GiB
  on this machine even though `nvidia-smi` reports several GiB free (WDDM
  per-process budget). The engine now sets
  `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` on Windows before importing
  torch; the same run then peaks at ~2.7 GiB. Keep this in mind when comparing
  planner numbers with reports from Linux machines.
- Not verified: any video training (musubi-tuner is not installed and the H3
  weights are ~210GB), QLoRA (bitsandbytes missing), DPO/GRPO/DMD/DMAD.

## 16. v0.2 features (2026-11-18)

Closed out by `docs/development/04-v0.2-execution-plan.md` (T1–T7). All additive;
the v0.1 planner/compile/execute contract is unchanged.

- **Post-run self-calibration (T1)**: `memory_report` is aggregated server-side
  by `(family,stage,adapter,quant_base,blocks_to_swap)`; the planner annotates
  "calibrated N× (avg delta x%)" — read-only, never mutates the budget.
- **Measured budget (T2)**: `env.py` measured `vram_mib`/`ram_mib` feed the default
  `vram_budget_mib`/`ram_budget_mib`; plan marks budget source = measured/manual.
- **Unified auto-select (T3)**: `auto_budget` searches `quant_base ×
  blocks_to_swap × train_blocks` (quant ladder `NONE→FP8→INT8→NF4→NVFP4`) and
  returns the cheapest feasible plan plus the full candidate menu.
- **Bounded OOM retry (T4)**: a CUDA-OOM step re-runs the plan up to
  `max_oom_retries` times with a strictly-more-conservative config
  (`_escalate_for_oom`); each retry emits `oom_retry` + a fresh `run_start`.
- **Swap time band (T6)**: `MemoryPlan.swap_floor_s` (one-way H2D floor) and
  `step_time_band_s` ("excludes attention/compute"), one-way vs round-trip.
- **NVFP4 tier + capability probe (T7)**: `QuantBase.NVFP4` (0.5625 B/param) is a
  selectable low-VRAM tier (rejected for LLM families, valid H3 edge recipe);
  `env.py` probes `triton`/`liger_kernel`/`sageattention` into `EnvReport.capabilities`.
- **External-declaration alignment (T5)**: `musubi_tasks.json` firmware pins
  `_TASK_BY_FAMILY` to upstream; recipe/README/plan status render from one
  source (`verified`/`compile_only`/`unverified`).