# SuperTuner 项目记忆

更新时间：2026-11-24

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

## v0.3 方向调整（2026-11-20，用户"高人指点"）
- 定位从"再做训练器"改为"训练实验与验证层"：训练框架（musubi/Fizgig）是工具，SuperTuner 回答"什么数据→什么效果、什么训法→什么问题"，支撑更强编辑/更长视频/更强提示词遵循/更多场景。DMD/DMAD 与自研算子暂缓。
- 落盘 `docs/development/05-experiment-harness.md`（评估指标 + 单变量矩阵 + 8GB/免费云分档 + v0.3 优先级）与 `docs/research/08-trainer-code-analysis.md`（197 行，深读 musubi@f8a1b03 与 Fizgig@b5199ee，25 条结论表 + 16 项可实验变量，均有 file:line 依据）。
- **关键发现（影响后续所有视频实验前提）**：H3 检查点是 CFG 蒸馏过的，普通 flow-matching LoRA 会 wash out（视频随步数变差、纯图像约 50 步崩坏）。H3 LoRA 必须三选一：de-distill adapter / guidance loss / teacher matching；身份 LoRA 走 teacher 匹配（teacher 看首尾帧/参考片段/主体参考图，学生 caption 只留触发词）。Wan2.2 无此问题，可先做 Wan/LLM 线建立评估闭环。
- **8GB H3 修正**：Fizgig "8GB 起"针对其它家族（Krea2/Flux/Anima/SDXL），H3 本身最低 16GB（12GB 需 4-bit）。SuperTuner 现有 H3 8GB 配方标 unverified 是对的。
- 数据契约先于超参：H3 帧数 17n+5、24fps、16 整除；无 caption dropout（正则靠 teacher 体系 + network_dropout + DC 衰减）；checkpoint 按 teacher 残差 plateau 选，不按 loss 最小。
- 低精度是隐式训练变量：NF4 基座 ~9% 误差消耗 LoRA 容量；int8 反向余弦 0.2-0.6 不可用（W8A16 是修正）；TREAD/只训 20-49 块/预览降档各有对应效果丢失通路。

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

## v0.3 P0 实验与评估层（2026-11-20，LLM 线实现）

- 用户方向（m05252/m05284）：训练框架是工具，SuperTuner 作为“训练实验与验证层”回答“数据→效果、训法→问题”；先做 LLM 线最小可自动评估闭环，视频评估先做接口契约（本机 8GB、无 musubi/视频权重，CLIP/LPIPS 需模型）。
- **评估契约（已落地）**：ngine/src/supertuner_engine/schema.py 新增 EvalSpec（enabled/prompts_path/holdout/source_dir/metrics，默认启用）并挂到 RunSpec.eval；ngine/src/supertuner_engine/events.py 增加 val_report 事件类型与 Emit/Emitter/NullEmitter.eval_report；packages/core/src/contracts.ts 增加 EvalSpecSchema/RunSpec.eval，packages/core/src/events.ts 增加 EvalReportEvent 并入 RunEventSchema；packages/web/src/lib/planner-state.ts 默认 spec 显式包含 val；packages/web/src/pages/RunDetail.tsx 解析并展示最新 val_report（状态、训练/梯度/评估曲线摘要、过拟合信号、产物路径、notes）。
- **曲线摘要与报告产物（已落地）**：ngine/src/supertuner_engine/backends/eval_report.py 提供 summarise_stability/uild_eval_report/write_eval_report（纯 Python，无 torch），输出训练 loss/梯度稳定性、eval 曲线、train-eval gap 过拟合信号，以及 LLM/视频指标契约字段。
ative_llm 在 SFT/DPO/GRPO 路径记录 step 与 eval 曲线，训练结束写 output/eval_report.json、发射 val_report 事件，并将 val_report.json 加入 compile expected artifacts（ngine/tests/test_native_compile.py 已同步）。
- **冻结 prompt 集与 LLM prompt 评估（已落地，启发式）**：docs/development/eval/prompts/llm_smoke_v1.json（6 条，固定 seed/max_new_tokens，math/instruction/completion/zh 烟雾样本，非 benchmark）；ngine/src/supertuner_engine/backends/prompt_eval.py 负责加载 JSON/JSONL/裸 list prompt 集与轻量打分（non_empty、exact/contains、token_overlap_f1 等）；ngine/src/supertuner_engine/backends/llm_prompt_eval.py 在训练结束且配置 spec.eval.prompts_path 时用已加载模型做固定种子 greedy 生成并聚合结果；val_report 记录 prompt_eval 摘要与状态（ok/partial/unavailable，失败降级不阻断训练/产物），并在契约中补充 prompt_eval_non_empty_rate/exact_match_rate/token_overlap_f1。
- **单变量矩阵展开与 CLI（已落地）**：ngine/src/supertuner_engine/experiment.py（xpand_matrix/write_expanded_matrix，单变量 sweep.path×values，生成命名/每变体 output 子目录/可选 eval_prompts 覆盖，逐个 RunSpec 校验，输出 specs 与 matrix.json manifest）；ngine/src/supertuner_engine/cli.py 新增 matrix --matrix --out-dir [--dry-run] [--run]，默认 dry-run 仅展开写 manifest，--run --no-dry-run 时顺序执行展开后的 specs。
- **server/web 接入现状**：val_report.json 落 run output 目录，可由现有 artifacts 扫描与 /api/runs/:id/artifacts 暴露；val_report 事件进入事件流/前端 RunDetail 展示。未新增 eval_report 专用 DB 表/查询或实验对比页（P1）。
- **测试与回归（2026-11-20）**：新增/更新 ngine/tests/test_prompt_eval.py、ngine/tests/test_experiment.py、ngine/tests/test_eval_report.py（含 prompt_eval 集成/降级）、packages/core/test/eval.test.ts、ngine/tests/test_events.py、ngine/tests/test_native_compile.py。全量回归通过：engine 252 passed；core 27 passed；server 52 passed；web 68 passed；pnpm -r typecheck 通过；pnpm build 通过（web 447.46 kB / gzip 131.47 kB）。
- **限制**：prompt 评估为启发式（exact/contains/token-overlap、non-empty，greedy+固定 seed），非外部 judge/benchmark；未实现独立 evaluator 后端、CLIP/LPIPS/时序一致性/视频生成打分与 web 实验对比页；本机无 musubi/视频权重，视频线仍为契约/接口层；矩阵执行为顺序单机（--run 可能很慢/占 GPU），未做云排期。

## dev4090 VBench 评估环境（2026-11-24 实测通过）

- 目标：在 dev4090（RTX 4090 24GB，Ubuntu 24.04 WSL）用 VBench 0.1.5 跑视频评测，支撑 B 路线（窄域视频质量评估）。用户批评过「先探查和网络搜索再动手」，本段所有解法都是先读源码/搜上游 issue 再落地。
- 环境：Python 3.11.9 + torch 2.15.0+cu132 + vbench 0.1.5，venv 在 `~/vbench-eval/.venv`；detectron2 未装（语义 9 维不可用，custom_input 只支持 6 技术维）。
- **6 技术维全部跑通**：3 个 768×512×125 帧测试视频，结果落 `~/vbench-eval/smoke/out/smoke_final_eval_results.json`。聚合分 subject_consistency 0.996 / background_consistency 0.997 / motion_smoothness 0.967 / dynamic_degree 0.0（静态视频预期）/ aesthetic_quality 0.227 / imaging_quality 0.302。之前「6 维 blocked」判断在走代理后已被推翻，`docs/research/09-vbench-feasibility.md` 已追加实测更正。
- 五个踩坑与解法（详见 `docs/development/eval/dev4090/README.md`）：① Python 3.12 无 cu121 轮子 + tokenizers 要 Rust → 3.11；② pypi.org 超时 → 清华镜像；③ VBench 写死 `unzip -d` 而 WSL 无 unzip 且 sudo 要密码 → `~/.local/bin/unzip` 包装器调 Windows 侧 7z（本机 `C:\Program Files\7-Zip\7z.exe`，wslpath 转路径），已固化脚本；④ `clip` 要 pkg_resources → 装 setuptools<81；⑤ aesthetic 权重旧 URL 只给 4KB 错误页（上游 issue #187）→ 改 raw.githubusercontent.com；MUSIQ 权重中断留坏文件 → 重下完整 108,610,983 字节。
- 复用脚本固化在 `docs/development/eval/dev4090/`：`install-unzip.sh`、`smoke-vbench.sh`、`README.md`。运行要点：ssh 保持前台（`ServerAliveInterval=30`），脚本放 `~/vbench-eval/` 持久路径（/tmp 跨会话丢、后台进程随 ssh 断被杀）；`VBench.evaluate()` 返回 None，结果写文件，接入 evaluator 时读 json 而非返回值。
