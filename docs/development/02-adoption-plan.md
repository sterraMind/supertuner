# SuperTuner 调研经验落地开发计划

> 文档版本：v1.0 | 日期：2026-11-17 | 作者：adoption-plan | 状态：完成

本文件把 `docs/research/04-consumer-frameworks/`（00-overview、01-inference-ninfer-strata、02-fizgig、03-lohan-ratel、04-roundpipe-llmq、05-similar-and-market-gap）与 `docs/reviews/01-architecture-review.md` 的结论转成原子任务。本文件只做规划，不改代码。

## 进度（2026-11-17 更新）

- 已完成并回归：AP-02（`auto_budget` 自动选精度+换块，经 server `resolve_config` 落到 argv）、AP-03（`train_blocks` 映射为 musubi `--network_args` include/exclude_patterns）、AP-04（预览步骤自动降档）、AP-05（计划 vs 实测 `memory_report.json` + 事件 + RunDetail 卡片，已端到端验证）、AP-06（工具许可/模型许可分列 + H3 许可门禁）。
- 未做（P0 剩余）：AP-01（机器实测 RAM/空闲显存自动进预算，当前仍用 spec 手填 `vram_budget_mib`/`ram_budget_mib`）、AP-07（`musubi_tasks` 固件与对外声明对齐）。
- P1（AP-08~AP-11）未开始。

## 1. 目标与经验分类

目标：把 NInfer、Strata、Fizgig、LoHan/Ratel、RoundPipe、LLMQ 的成功经验按「可立即做 / 后续做 / 不做」分类，且每个任务都能由一个代理独立完成并验收。分类依据 `docs/ARCHITECTURE.md` 的 v0.1 契约：SuperTuner 是单卡训练控制面，不重写训练内核，显存算术与后端 argv 各归各的源。

- 可立即做（P0，进下一次对外发布前）：
  - Fizgig：按空闲显存同时选精度与 swap 数量、目标 block 范围、预览自动降档、OOM 后有限次自动加 swap（写事件）、把工具许可与模型许可分列。
  - NInfer/NInfer-all/Strata：设备 profile 与首次校准（env 已有探针，可加分组）、权重预构建→缓存步骤账本、专家 offload 与 block swap 共用「预算 / 搬运时间 / 命中率」三类指标。
  - LoHan/Ratel：把 CPU↔GPU 与 NVMe↔CPU 搬运量、重算时间做成规划字段；主存预算与机器实测绑定。
  - RoundPipe：单卡 block swap 的预取顺序，吸收优先级传输调度思想。
  - LLMQ：显存分项日志（memory_report）、启动期预算检查、logits 与 attention 反向分块的规划表达。
- 后续做（P1/P2，需实测或新后端）：LLMQ e4m3 低精度执行（先做编译估算）、RoundPipe 多卡流水线运行模型、投机解码与前缀缓存用于采样预览（不用于训练正确性）、TREAD token routing 作为策略插件评估、校准回路 calibration.jsonl、LoHan active gradient offloading 策略插件放实验分支。
- 不做（v0.1 明确排除）：把 NInfer/Strata 的 C++/CUDA 内核移植进来、视频全参训练与 DMD/DMAD 执行（编译期拒绝）、内嵌 AGPL/GPL 项目代码、12 块 NVMe RAID 加 root huge pages 的 LoHan/Ratel 工件照搬、把 8GB 未验证视频训练写成已实现。

## 2. 原子任务表

列含义：输入是什么、输出是什么、涉及哪些文件、前置依赖、怎么验收、主要风险。每个任务一个代理独立完成。

| ID | 标题 | 输入 | 输出 | 涉及文件范围 | 前置依赖 | 验收标准 | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP-01 | 机器实测 RAM 与 GPU 空闲显存写入预算 | `env.py` 探针（`EnvReport.ram_mib`、`GpuInfo.vram_free_mib`）与 spec 的 `vram_budget_mib`/`ram_budget_mib` | spec 新增 `machine:{ram_mib,vram_free_mib}`；planner 缺省取实测值并输出「预算来源（实测/手填）」note | `engine/src/supertuner_engine/{schema,env,plan,config}.py`、`packages/server/src/routes.ts`、`packages/core/src/contracts.ts`、`packages/web/src/lib/planner-state.ts`、`engine/tests/{test_env,test_plan}.py` | 无 | monkeypatch `ram_mib=16384` 时 `h3-lora-320p-8gb` 判 infeasible 且 issue 含「需要 19.2GB 主存，检测到约 11GB 可用」；预算来源 note 存在；`pnpm test` 与 `pnpm engine:test` 全绿 | 实测值受 Windows/WDDM 影响，须写明 free 口径，不得用 total |
| AP-02 | 按空闲显存自动选择精度与 block swap 数量 | family、目标分辨率/帧数、`machine` 实测值、现有逐项 planner | 搜索函数输出多个可行 (precision, quant_base, blocks_to_swap) 组合，各带误差与搬运代价，并给「需手动确认」标记 | `engine/src/supertuner_engine/{plan,schema}.py`、`engine/tests/test_plan.py`、`packages/web/src/pages/Planner.tsx`、`packages/core/src/contracts.ts` | AP-01 | 给定 `vram_free_mib` 约 30720/22528/15360/12288 分别命中 int8 无 swap、int8 约 14 swap、int8 约 36 swap、4-bit；每个组合经 planner 复核 feasible；结果表格写进单测 | Fizgig 表格来自其文档而非本机实测，必须标「参考值，需实测校准」，不得当承诺 |
| AP-03 | 视频目标 block 范围字段 | `RunConfig` 新字段 `train_blocks:[start,end]|null`，默认对齐 Fizgig 20–49 与 6–49 | schema 字段 + 编译传给 musubi backend + 计划页展示，null 时行为不变 | `engine/src/supertuner_engine/{schema.py,backends/musubi.py}`、`packages/core/src/contracts.ts`、`packages/web/src/pages/Planner.tsx`、`engine/tests/test_compile.py` | 无 | 单测 `train_blocks=[20,49]` 与 `[6,49]` 生成不同 argv、`null` 保持原 argv；契约测试 TS 与 Python schema 接受该字段且 dump 一致；compile 对视频家族在 PlanStep.command 含对应参数并标「未对上游 CLI 校验」 | musubi CLI 参数名未核实（评审附录 B），须写 compile-only，不得声称已验证可跑 |
| AP-04 | 预览步骤自动降档 | video family 的 VAE/文本编码器峰值、`machine` 实测值、预览目标尺寸/帧数 | 独立 preview/sample PlanStep 估算 + 失败回退顺序（先缩片段→降分辨率→512×512）+ 自动选合适预览方案 | `engine/src/supertuner_engine/{schema,plan,backends/musubi}.py`、`packages/core/src/contracts.ts`、`packages/web/src/pages/{Planner,RunDetail}.tsx`、`engine/tests/test_plan.py` | AP-01 | 8GB 预算下 H3 预览估算含 VAE decoder 峰值并返回「不可预览」或降档序列；单测覆盖回退顺序；compile 后 preview step 不再默认 feasible | VAE decoder 峰值未实测，数字须标「估算」 |
| AP-05 | 计划与实测显存对照报告 | `run_start.plan`、`step.vram_peak_mib`、进程采样、README 1787MiB 那次实测 fixture | planner 加 `context_mib` 行（默认 900，可 env 覆盖）+ run_end 写 `memory_report.json`{planned,max_allocated,max_reserved,max_process,delta_pct} + RunDetail 偏差行 | `engine/src/supertuner_engine/{plan,events,runtime}.py`、`packages/core/src/events.ts`、`packages/web/src/pages/RunDetail.tsx`、`engine/tests/{test_plan,test_events}.py` | AP-01 | 用 1787MiB 那次 spec 作 fixture 断言 planned+context_mib≥实测≥planned−10%；偏差>20% 出警报；RunDetail 无实测显示「—」而非回退估算 | process 采样需 nvidia-smi 轮询，注意 Windows 权限与采样间隔 |
| AP-06 | 扩展模型许可登记 | 现有 H3 门禁、各模型许可文本（H3 Community License、Krea 2、Qwen Image 等）与来源链接 | manifest 拆 `license`（工具）/`model_license`（模型）；`RunSpec.licence_ack`；新模型许可登记清单；配方卡展示模型许可 + 不可折叠限制块 | `engine/src/supertuner_engine/{schema,plugins,backends/musubi}.py`、`packages/core/src/{contracts,recipes,plugins}.ts`、`packages/server/src/routes.ts`、`packages/web/src/pages/Recipes.tsx`、`engine/tests/test_plugins.py` | 无 | `licenceForFamily` 对 minimax-h3 返回模型许可而非工具 Apache-2.0；无 ack 的 H3 spec 由 server 返回 422；H3 卡片含「MiniMax H3 Community License」且权重许可处不含「Apache-2.0」；每条新登记有来源链接与版本 | 部分模型许可文本未取得，登记须标「待法务确认」，不得默认放行 |
| AP-07 | 视频能力与对外声明对齐 | musubi `docs/wan.md` 与 `docs/minimax_h3.md` 的 task/flag 清单、README 真相表、`recipes.ts` 状态 | 测试固件 `engine/tests/fixtures/musubi_tasks.json`；`_TASK_BY_FAMILY` 逐项命中；真相表状态列由配方数据生成；`PlanStep.estimated_mib` 覆盖所有步骤、`RunPlan.feasible` 为所有步骤 AND | `engine/tests/fixtures/musubi_tasks.json`、`engine/src/supertuner_engine/{backends/musubi,plan}.py`、`packages/core/src/recipes.ts`、`README.md`、`docs/ARCHITECTURE.md`、`engine/tests/test_compile.py` | 无（对外口径总闸，合并前完成） | 5B 若不在上游 task 列表则移除或标 compile_only；`h3-lora-320p-8gb` 在 8GB 预算 compile 结果 infeasible 且 issue 含缓存文本编码器所需容量；真相表状态标签 ∈ {verified, compile_only, unverified} | musubi 未安装，固件依赖文档，版本迭代快须锁版本范围 |

P1（v0.2 前，影响质量与可扩展性，每个仍可独立拆给一个代理）：

| ID | 标题 | 输入 | 输出 | 涉及文件范围 | 前置依赖 | 验收标准 | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP-08 | 单向 block swap 搬运时间区间 | `plan.py` 现 `PCIE_GB_PER_S=8.0` 单向值、Fizgig 单向比往返快约 6.4 倍、实测锚点（16GB+swap12 约 10 s/it） | 区分单向/往返，输出 `swap_floor_s`（理论下界）与 `step_time_band_s`（×3 趟÷实测带宽），UI 标「未含注意力计算」 | `engine/src/supertuner_engine/plan.py`、`packages/web/src/pages/Planner.tsx`、`engine/tests/test_plan.py` | AP-02 | 单测对 H3 320p 输出 band 落在 8–15s 区间；band 与 floor 两个字段在 contracts.ts 有定义 | 带宽随机型变化，band 须给区间而非单值 |
| AP-09 | 设备 profile 与首次校准 | NInfer-all 48 个 GPU profile 与首次校准、`env.py` 现探针 | 设备 profile 目录按显存/带宽/驱动/CUDA 能力分组，首次运行生成本机默认配置 | `engine/src/supertuner_engine/{env,config}.py`、`packages/server/src/routes.ts` | AP-01 | profile 命中后 planner 默认值随本机调整；`GET /api/system` 返回命中分组名 | profile 数值需实机校准，未校准前只作建议不作硬门槛 |
| AP-10 | 数据移动策略插件（实验分支） | LoHan active gradient offloading 与 activation swapping 思想 | strategy 插件骨架，把搬运量与重算写进 step 事件，默认路径不启用 | `engine/src/supertuner_engine/{plugins,schema}.py`、新 strategy 插件文件 | AP-08 | 插件可注册并输出策略 issue；默认 0.1 行为不变 | 属研究分支，不进默认，须标 experimental |
| AP-11 | 多卡路线文档（不做实现） | RoundPipe 论文与 8×RTX4090 性能表 | 单页设计文档记录多进程/多设备/集合通信的运行模型改动清单，明确不进 v0.1 | 新增 `docs/development/` 下多卡路线页、`docs/ARCHITECTURE.md` 非目标段 | 无 | 文档写明触发条件（完成单卡实测闭环）与不做项，代码不动 | 易被误读为承诺，须标「后续路线」 |

## 3. P0 任务测试清单

每条给出「单元测试 / 契约测试 / 编译期测试 / 端到端测试」各验证什么。编译期指 `supertuner-engine compile` 这一步，外加 TS 端 `pnpm typecheck`。

- AP-01：单测 `plan()` 在 machine.ram_mib=16384 对 H3 320p 判 infeasible 且 issue 文案含主存数字、预算来源 note 为「实测」；契约测试 `EnvReport` 的 Python dump 经 zod 解析后 ram_mib 与 machine.* 不被剥掉；编译期测试 spec 缺 machine 时 compile 回退手填预算且 `pnpm typecheck` 通过；端到端 `POST /api/runs` 在探针 ram_mib=16384 对 H3 spec 返回 422 且指向主存不足。
- AP-02：单测固定 vram_free_mib 四档输入命中 int8/swap/4-bit 且逐项 planner 复核 feasible、结果表格锁进测试；契约测试搜索结果结构在 contracts.ts 定义且 zod 校验通过；编译期把每个组合生成 spec 后 compile 无 error issue、4-bit 带「需手动确认」；端到端 Planner 页输入空闲显存可见多个可行组合，选中后 spec 与搜索结果一致。
- AP-03：单测 train_blocks 两档生成不同 argv、null 不变；契约测试 TS 与 Python schema 接受 train_blocks 且 dump 一致；编译期视频家族 PlanStep.command 含对应参数且带「未对上游 CLI 校验」；端到端 API 提交带 train_blocks 的 spec 后 plan.json 可查到该字段并原样存进 run 记录。
- AP-04：单测 8GB H3 预览估算含 VAE decoder 峰值并返回「不可预览」或降档序列、回退顺序断言；契约测试 preview PlanStep 的 kind/字段在 events.ts 与 contracts.ts 存在且 zod 不丢；编译期 compile 对 video+preview 出独立 step 或单列估算行、不再默认 feasible；端到端 Planner 页显示「预览采样峰值」一行且 8GB H3 出不可预览提示。
- AP-05：单测用 1787MiB 那次 spec 作 fixture 断言 planned+context_mib≥实测≥planned−10%；契约测试 memory_report.json 六字段在 contracts.ts 有 schema、step.vram_peak_mib 字段名与文档一致（修 N1 漂移）；编译期 compile 产生的 run_start.plan 含 context_mib 行且行数与 MemoryLine 定义一致；端到端跑一次 0.6B 短训练后 run_end 生成 memory_report.json、RunDetail 显示偏差行、无实测显示「—」。
- AP-06：单测 licenceForFamily 对 H3 返回模型许可而非工具 Apache-2.0、缺 ack 校验函数报错；契约测试 licence_ack 字段在 contracts.ts 与 schema.py 双向一致、manifest 同时有 license 与 model_license；编译期 compile 不修改 ack、缺 ack 由 server 拒而非编译器拒；端到端 `POST /api/runs` 无 ack 返回 422、Recipes 页 H3 卡片含「MiniMax H3 Community License」且权重许可处不含「Apache-2.0」、ack 时间在 run 详情可见。
- AP-07：单测 musubi_tasks.json 固件逐项命中 `_TASK_BY_FAMILY`、8GB H3 320p compile 返回 infeasible 且 issue 含缓存文本编码器所需容量；契约测试状态标签在 contracts.ts 定义为 verified/compile_only/unverified、配方与真相表行数一致测试可运行；编译期 PlanStep.estimated_mib 对 prepare/cache/train/export 均有值、RunPlan.feasible 为所有步骤 AND；端到端 README、配方卡、计划页三处状态由同一份数据渲染，5B 与 H3 显示 compile_only 而非「可训练」。

## 4. 任务依赖（文字依赖）

- AP-01 无前置，是所有预算相关任务的数据源，最先做。
- AP-02 依赖 AP-01（需要实测显存作为搜索输入）。
- AP-03 无前置，可与 AP-01 并行。
- AP-04 依赖 AP-01（需要实测显存判断预览是否可跑）。
- AP-05 依赖 AP-01（需要预算真值做对照）。
- AP-06 无前置，可与其它 P0 并行。
- AP-07 无前置，但作为对外口径总闸，应在所有 P0 合并进主分支前完成。
- P1：AP-08 依赖 AP-02（搬运区间并入搜索代价）；AP-09 依赖 AP-01；AP-10 依赖 AP-08；AP-11 无前置。
- 顺序建议：先 AP-01，再并行 AP-02/AP-03/AP-04/AP-06，最后 AP-05 与 AP-07 收尾对外口径。

## 5. 明确禁止

- 不把未验证的视频训练写成已实现：视频在 README、配方卡、计划页只能标 compile_only 或 unverified；真机训练记录产生前，不得用「已验证」「可训练」「8GB 可训 H3」等表述。
- 不内嵌 AGPL/GPL 项目代码：OneTrainer（AGPL-3.0）、SimpleTuner、diffusion-pipe 只作设计参考，不复制作业代码进仓库。
- 不复制 NInfer/Strata 的 C++/CUDA 内核进 SuperTuner：只吸收设备 profile、首次校准、专家 offload 与 block swap 共用的预算、搬运、命中率等设计思想，不搬底层实现。
- 不把推理吞吐数字当成训练可行性证据，不把用户手填的预算常数当成机器事实，不做视频全参训练与 DMD/DMAD 执行。

## 6. 最终验收清单（lead 收齐后执行）

命令（PowerShell 7，工作区根目录）：

- `pnpm build`：全部 @supertuner/* 包类型与产物通过。
- `pnpm typecheck`：TS 契约与组件类型无错误（对应各任务编译期测试）。
- `pnpm test`：server 与 web 全绿（基准 48 / 68）。
- `pnpm engine:test`（即 `uv run --project engine pytest -q`）：引擎全绿（基准 144），且新增 musubi_tasks 固件、train_blocks、预览、memory_report 用例全部通过。

看哪些文档：

- `docs/ARCHITECTURE.md` 与 `docs/reviews/01-architecture-review.md`：确认 B1–B4 与 M1/M2/M3 对应修改已在代码落地，文档与代码不再矛盾。
- `docs/research/04-consumer-frameworks/02-fizgig.md` 与 `05-similar-and-market-gap.md`：确认吸收项都有任务承接，未吸收项都落在 P1/不做。

检查哪些 UI 文案：

- Planner 页：预算来源 note（实测/手填）、自动搜索组合、train_blocks 输入、预览采样峰值行、swap 时间区间且标「未含注意力计算」。
- Recipes 页：H3 卡片含「MiniMax H3 Community License」且权重许可处不含「Apache-2.0」；不可折叠的地域与蒸馏限制块。
- RunDetail 页：估算/实测/偏差行，无实测显示「—」，偏差>20% 出警报。
- README 与真相表：状态列 ∈ {verified, compile_only, unverified}，由同一份数据渲染；5B 与 H3 显示 compile_only。
- System/设置页：GPU total/free/预留三档，以及 `machine` 实测字段来源。
