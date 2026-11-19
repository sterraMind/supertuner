# SuperTuner

Single-GPU training studio for LLM and video models on consumer hardware.

SuperTuner is a control plane, not a training kernel library. It answers the
question every consumer-card user hits first — **"will this run fit, and what do
I change if it does not?"** — before the run starts, then compiles the answer
into an executable plan and supervises the run.

- Python engine (`engine/`): schema, memory planner, plugin registry, compiler,
  runtime, backends (native LLM SFT/LoRA, musubi-tuner adapter for video).
- TypeScript server (`packages/server/`): Fastify API, SQLite run registry,
  process supervisor, SSE event stream.
- React UI (`packages/web/`): system panel, recipes, planner with live memory
  preview, run monitor with live loss chart, plugin list, settings.
- Shared contracts (`packages/core/`): zod schemas, event union, plugin
  manifest, curated recipes.

## Status

v0.1 is honest about three separate scopes:

- **LLM SFT/LoRA/QLoRA** (transformers + peft): the only path with an end-to-end
  run on this repository's hardware. Qwen3-0.6B LoRA is verified; QLoRA is
  implemented but needs bitsandbytes, which is not installed on the reference
  Windows machine.
- **Preference optimisation (DPO)**: implemented natively (LoRA policy plus the
  frozen base model as reference), covered by unit tests, and compiled into a
  runnable plan. A full GPU run was not completed on the reference laptop because
  its Windows commit limit was exhausted by other applications, not because of a
  training-code failure.
- **Video LoRA (Wan / MiniMax-H3)**: **compile preview only**. SuperTuner
  generates the musubi-tuner argv, the step-by-step plan and the memory accounting.
  Execution requires you to install musubi-tuner and the weights; v0.1 does not
  promise a successful video run, and the review in `docs/reviews/01-architecture-review.md`
  explains why.
- RL, DMD and DMAD are declared as capabilities and refused with a reason.

### Verified on consumer hardware

On a laptop RTX 3070 Ti (8GB) the LLM path was exercised end to end:

- Qwen3-0.6B LoRA SFT on wikitext-2 through `POST /api/runs`, 20 steps,
  completed in 29.9s at 442.9 tokens/s, 1787 MiB peak, adapters and checkpoints
  listed under `/api/runs/:id/artifacts`.
- The same configuration via the CLI: 30 steps in 49.5s, loss 2.75–3.98,
  `adapter_model.safetensors` 40,422,168 bytes.

Windows needs the async CUDA allocator: the engine sets
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` before importing torch, because
the default caching allocator dies at ~1.7 GiB on WDDM even when `nvidia-smi`
reports free memory. Video training, QLoRA and a full DPO run are **not** verified
on this machine (musubi-tuner and bitsandbytes are not installed).

## Truth about the hardware

Every row below is a planner output that includes the fixed ~0.9 GB CUDA-context
term; the status column says how far the path has actually been exercised.

| Card | Configuration | Status |
| --- | --- | --- |
| 8GB | Qwen3-0.6B bf16 LoRA SFT | verified |
| 8GB | Qwen3-1.7B QLoRA (needs bitsandbytes) | unverified |
| 8GB | Wan2.1 1.3B 480p LoRA | unverified; training peak ~7.1 GB exceeds the 8 GB budget after desktop reserve |
| 8GB | MiniMax-H3 320p LoRA with 48/50 blocks swapped | compile preview; caching the 32B text encoder needs ~25 GB and does not fit |
| 12GB | Qwen3-4B QLoRA | unverified |
| 16GB | Qwen3-8B QLoRA | unverified |
| 16GB | Wan2.2 TI2V 5B | **not supported in v0.1** — musubi-tuner has no `ti2v-5B` task; use DiffSynth-Studio or diffusion-pipe |
| 24GB | MiniMax-H3 480p LoRA (40/50 swapped) | compile preview |
| 32GB | MiniMax-H3 720p LoRA (32/50 swapped) | compile preview |
| 48GB | MiniMax-H3 720p LoRA, rank 64 (16/50 swapped) | compile preview |

Full fine-tuning of MiniMax-H3 needs all 33B weights (~66GB bf16) and is refused
on a single consumer card. H3 uses the MiniMax H3 Community License (not
Apache-2.0): territory excludes the EU, UK, Korea and the US, revenue above
USD 20M needs written permission, and outputs may not be used to train other
models.

## Requirements

- Node.js >= 22.5 (the server uses the built-in `node:sqlite`), pnpm 10
- Python 3.12 and `uv`
- NVIDIA GPU with a driver matching the installed CUDA build
- ffmpeg on PATH for video datasets

## Setup

```bash
pnpm install
uv sync --project engine --extra dev --extra torch
```

`uv sync` installs the CPU build of torch from PyPI. For a CUDA build pick the
index that matches your driver (check `nvidia-smi` for the CUDA version):

```bash
# CUDA 13.2 (torch 2.15.0 is the last release with cu132 Windows wheels)
uv pip install --python engine/.venv/Scripts/python.exe "torch==2.15.0+cu132" \
  --index-url https://download.pytorch.org/whl/cu132
```

Verify what the engine sees:

```bash
uv run --project engine supertuner-engine env --json
```

## Run

```bash
pnpm dev          # Fastify API on http://127.0.0.1:8787
pnpm dev:web      # Vite UI on http://127.0.0.1:5173 (proxies /api)
```

Plan a run without starting it:

```bash
echo '{"name":"demo","family":"qwen3-1.7b","stage":"sft","adapter":"qlora","quant_base":"nf4","seq_len":1024}' \
  | uv run --project engine supertuner-engine plan --json
```

## Tests

```bash
pnpm test                 # core + server + web (vitest)
uv run --project engine pytest   # engine unit tests
```

GPU tests are opt-in:

```bash
SUPERTUNER_RUN_GPU_TESTS=1 uv run --project engine pytest -m gpu
```

## Configuration

All paths and endpoints come from environment variables (see `.env.example`).
No absolute path is hardcoded in code; the server refuses relative dataset
paths unless `SUPERTUNER_ALLOW_ANY_PATH=1`.

## Documentation

- `docs/ARCHITECTURE.md` — component map, data flow, plugin model, memory model,
  API surface, ADRs, risks.
- `docs/research/01-market.md` — competitor matrix, user pain points, market gaps.
- `docs/research/02-feasibility.md` — single-card feasibility per model and VRAM tier.
- `docs/research/03-stack.md` — library shortlist, plugin conventions, engineering norms.
- `docs/TESTING.md` — test environments, tiers, what runs where.

## Licences

- SuperTuner code: Apache-2.0.
- MiniMax-H3 weights: **MiniMax H3 Community License** (not Apache-2.0). Read it
  before redistributing model outputs.
- musubi-tuner declares Apache-2.0 in its README but ships no LICENSE file.
  SuperTuner calls it as an external process and does not vendor its code;
  confirm the licence before shipping a bundled installer.
- SimpleTuner (AGPL) and diffusion-pipe (GPL) are used as design references
  only; neither is vendored.
