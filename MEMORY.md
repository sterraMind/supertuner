# SuperTuner 项目记忆

更新时间：2026-11-18

## 目标

单卡消费级显卡（本机 RTX 3070 Ti Laptop 8GB）上可实际执行 LLM / 视频模型的训练控制平面。当前版本 0.1.0-preview。

## 已完成并验证

- LLM SFT/LoRA/QLoRA：原生 transformers + peft，单卡进程内训练。Qwen3-0.6B + wikitext-2，20 步，442.9 tok/s，峰值 1787 MiB，adapter 40,422,168 B，端到端跑通。
- DPO 偏好训练：LoRA policy + 关闭 adapter 的基座作参考模型。SmolLM2-360M 与 Qwen3-0.6B 各 10 步跑通（Qwen3 loss 1.385→1.342，峰值 3.6 GiB）；全参偏好训练在编译期拒绝（`E_PREFERENCE_FULL_UNSUPPORTED`）。
- RL（GRPO）：exact-match reward + group advantage + KL 到无 adapter 参考模型。Qwen2.5-0.5B-Instruct + 数字复制任务 6 步跑通，reward 1.000，loss 有真实梯度（grad_norm 24.5）；全参 RL 编译期拒绝（`E_RL_FULL_UNSUPPORTED`）。
- 内存规划器：张量账本 + 0.9 GiB CUDA 上下文常数（`SUPERTUNER_CUDA_CONTEXT_MIB` 可覆盖）+ block swap + 文本编码器/VAE 分步估算。编译期逐步骤检查，超预算报 `E_STEP_OOM`。
- 自动预算（`auto_budget`）：engine 搜索 quant_base×blocks_to_swap 选显存最低可行配置；server 经 `resolve_config` 在 spawn 前套用（`I_AUTO_BUDGET_APPLIED`）；Web Planner 有开关。
- 计划 vs 实测显存报告（`memory_report`）：训练结束写 `memory_report.json` 并发 `memory_report` 事件（planned vs measured，含 delta%）。Qwen3-0.6B LoRA SFT 经 server API 端到端：计划 2651.5 MiB vs 实测 2814 MiB（+6.1%）；RunDetail 有对照卡片。
- 视频：musubi-tuner 适配器只做编译预览（prepare → cache-text → cache-latents → train 的 argv），未安装 musubi 或未缓存文本编码器时明确报不可行。MiniMax-H3 8GB 边缘配置只过训练步，cache-text 需要 25 GB，因此整体不可行，界面如实显示。支持可选 `train_blocks`（目标 block 范围，当前只记录不映射上游参数，编译期 warning）与预览降档步骤（768×640×22 → 768×640×8 → 512×512×8，VAE 解码单独估算，超预算只 warning 不阻塞）。
- H3 许可门禁：POST /api/runs 必须带 `licence_ack`，否则 422；Planner 勾选确认后才能启动。
- Web：7 个页面，状态徽标、许可展示、规划预览、运行监控 SSE、loss 图、产物列表、system/light/dark 主题。
- 测试：engine pytest 183 passed；server 52 passed；web 68 passed；typecheck/build 通过（web JS 441.38 kB / gzip 129.68 kB）。

## v0.2 进行中（执行闸 docs/development/04-v0.2-execution-plan.md）

- **T1 跑后自校准（G1）已完成并回归**：server 新增 SQLite `calibrations` 表，按 `(family,stage,adapter,quant_base,blocks_to_swap)` 五元组聚合 `memory_report`；server 在 `memory_report` 事件处 upsert，规划前 `withCalibration` 注入 `spec.calibration`；engine `plan()` 只读标注「已校准 N 次（平均偏差 x%）」，N<3 只标样本数，绝不动 `vram_budget_mib`。engine 183 / server 52 / web 68 / core 5 全绿（engine 3 个失败为既有 HF 文件锁环境问题）。
- **T2 实测进预算已完成**：`RunConfig.budget_source`（auto/manual）+ `MemoryPlan.budget_source`/`budget_measured_*`；`plan()` 经 `_resolve_budget` 用 `env.py` 实测 `vram_mib`/`ram_mib` 作为默认预算并标注「预算来源=实测/手填」。engine 191 / server 52 / web 68 全绿。
- **T3 统一自动选择已完成**：`_auto_budget_resolve` 量化阶梯补全 `{NONE,FP8,INT8,NF4}`，video 家族加 `train_blocks` 预设 `[None,"20-49"]`；新增 `AutoCandidate` + `MemoryPlan.auto_candidates`（可行在前按峰值升序），UI 渲染「可训练选项」表；`_pick_best_candidate` 同分时训练 block 更少者优先。独立评审补正：H3 守卫 `H3_EDGE_QUANTS=(FP8,INT8,NF4)` 接受所有量化基座的小卡边缘配方（原只认 INT8，与自动搜索会选 NF4 矛盾）。engine 198 / server 52 / web 68 全绿（engine 3 个失败为既有 HF 文件锁环境问题）。
- **T4 OOM 有限次兜底（G4）已完成并独立验收**：`runtime.execute` 加 OOM 有界重试循环——步骤因 CUDA OOM 失败时，`_escalate_for_oom` 用 `auto_budget` 生成「严格更保守」配置（量化阶梯下探 + swap 上探 + `vram_budget_mib` 每轮降 1024MiB），最多 `RunConfig.max_oom_retries`（默认 2，0 关闭）次，每次发 `oom_retry` 事件 + 新 `run_start`；OOM 判定覆盖后端异常（`_is_oom_error`）与 musubi 子进程 stderr 签名（`OOM_STDERR_MARKERS`）。非 OOM 失败/取消/不可行计划不重试。回归：engine 108 / server 52 / web 68 / core 20 全绿，typecheck/build 通过。
- **T5 对外声明对齐（AP-07）已完成**：新增 `engine/tests/fixtures/musubi_tasks.json`（musubi 上游 `--task` 固件）+ `test_musubi_firmware.py`（5 项：`_TASK_BY_FAMILY`/`_NETWORK_MODULE`/`MUSUBI_FAMILIES` 逐项命中固件，5B 在 unsupported 且不在家族表）；`packages/core/test/status.test.ts` 解析真实 README 真相表，断言每配方 status ∈ {verified, compile_only, unverified} 且表行与配方 1:1。修正 README Wan2.1 1.3B 8GB 行状态 compile preview→unverified（与 recipes.ts 对齐）。
- **T6 单向换块区间（AP-08）+ T7 NVFP4+probe 已完成**：`MemoryPlan` 增 `swap_floor_s`（单向 H2D 下限）/`swap_roundtrip_floor_s`（2×）/`step_time_band_s`（「不含注意力/计算」）；`QuantBase.NVFP4`（0.5625 B/param）加入自动搜索阶梯；`env.py` 探测 triton/liger_kernel/sageattention 记入 `EnvReport.capabilities`。独立评审补 NVFP4 一致性：`compiler.py` 与 `native_llm.py` 对 LLM 家族拒绝 NVFP4（同 INT8/FP8，E_QUANT_UNSUPPORTED），`H3_EDGE_QUANTS` 纳入 NVFP4（plan/compile 一致）。
- **v0.2 全量回归（收口）**：engine 230 / core 25 / server 52 / web 68 全绿；`pnpm typecheck` 与 `pnpm build` 通过（web JS 445.32 kB / gzip 130.98 kB）。T1–T7 全部完成，v0.2 收口。

## 关键环境事实

- GPU：RTX 3070 Ti Laptop 8GB，驱动 621.14；CPU i7-12700H；RAM 31.7 GiB；C 盘约 143 GB。
- Windows 下 CUDA 分配受系统提交量限制：默认 caching allocator 在约 1.7 GiB 出现假性 OOM，引擎自动追加 `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`。
- torch 2.15.0+cu132（cu132 索引），transformers 5.22.2，peft 0.21.2，accelerate 1.15.0，datasets 5.2.0。
- 数据与权重默认放 `$SUPERTUNER_HOME`（默认 `C:\Users\jaggar\.supertuner`）下的 datasets/models/runs。

## 已知限制

- 视频训练未在本机实际执行（缺 musubi-tuner、权重与足够显存）；任何“8GB 跑 H3 视频 LoRA”都不是可执行承诺。
- Wan2.1 1.3B 480p 在 8GB 上训练步估算 7.1 GiB，超过 8GB 卡扣除桌面保留后的 6.6 GiB，建议 12GB 或先在大卡缓存文本编码器。
- H3 权重约 210 GB 下载、精简后仍需约 55 GB；H3 社区许可地域排除 EU/UK/韩国/美国。
- Windows 的 bitsandbytes 未安装，QLoRA 配方会明确拒绝而不是静默降级。

## 仓库结构

- `engine/`：Python 引擎（schema、plan、compiler、runtime、plugins、backends/native_llm.py、backends/musubi.py、CLI）。
- `packages/core/`：TypeScript 契约、配方、事件解析。
- `packages/server/`：Fastify API、SQLite 运行记录、SSE、进程监管。
- `packages/web/`：React + Vite 界面。
- `docs/ARCHITECTURE.md`：架构与 ADR；`docs/research/`：市场、可行性、技术栈调研；`docs/reviews/01-architecture-review.md`：独立评审与整改清单。

## 消费卡框架调研（2026-11-17）

- 新增 `docs/research/04-consumer-frameworks/` 六份文档：索引、NInfer/NInfer-all/Strata、Fizgig、LoHan/Ratel、RoundPipe/LLMQ、同类框架与市场空缺。
- Ratel 与 LoHan 属于同一研究线，Ratel 是 LoHan 的 ICDE 2025 会议版本和工件名称。
- Fizgig 是当前最直接竞品，已支持 MiniMax H3 LoRA 训练、自动精度与 block swap、预览分析和 ComfyUI 导出；SuperTuner 视频路径仍是编译预览。
- NInfer、NInfer-all、Strata 解决消费卡推理，不解决训练；RoundPipe 主要面向多张消费卡；LLMQ 依赖 C++/CUDA、CUDA 12、NCCL、cuDNN。
- SuperTuner 当前可占有的产品空位是显存与主存预算、逐步计划、许可门禁、运行记录、SSE 监控、LLM 与视频统一 RunSpec；视频训练真实验证仍是缺口。
- 调研落地：`docs/development/02-adoption-plan.md` 把可借鉴点拆成 P0/P1 任务；其中 auto_budget、train_blocks、预览降档、memory_report 已实现并回归。musubi 没有 `--train_blocks` 直接 flag：`train_blocks`（如 `20-49`）现映射为 `--network_args` 的 `include_patterns=[r'.*blocks\.(?:20|21|…|49)\..*']`（完整匹配模块名 `transformer_blocks.20.attn.to_q` 等）并保留默认 `exclude_patterns`（img_mod/txt_mod/modulation），编译期发 `I_TRAIN_BLOCKS_MAPPED`。
- 环境注意：engine 由 server 以 `uv run` 派生的 python 受 DSH sandbox 写根限制（workspace 可写，`~/.supertuner` 与 `~/.cache/huggingface` 默认不可写），server 启动时把 `SUPERTUNER_RUNS_DIR`/`HF_HOME`/`HF_DATASETS_CACHE` 指向 workspace 可端到端跑通。
