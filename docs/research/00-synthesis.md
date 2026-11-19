# 00 调研汇总与产品定位（Leader 仲裁稿）

> 输入：[01-market.md](./01-market.md)、[02-feasibility.md](./02-feasibility.md)、[03-stack.md](./03-stack.md)（均为 2026-11-11 抓取）。
> 本文只做取舍与排期，事实与出处保留在三份原始调研里。

## 1. 结论

1. **市场空缺是"显存可执行性"，不是"又一个训练脚本"。** 主流视频训练官方下限 24GB，32GB 用户仍失败，96GB 也 OOM；8–16GB 没有可完成路径。
2. **单卡现实入口是 5B 视频模型与 4–8B LLM，不是 14B，更不是 33B。** MiniMax-H3 是 33B dense 单流 Omni-Transformer（文本条件取 Qwen3-VL-32B 第 50 层），社区许可排除 EU/UK/韩/美，并禁止用 H3 或其输出改进其它模型。
3. **蒸馏（DMD/DMAD）不是单卡任务。** 官方 DMAD 训练在 8×H200 上进行；单卡只能消费已发布的 1.4GB 学生 LoRA 做 4 步推理。v0.1 只声明能力、不执行。
4. **视频侧不自研训练核。** musubi-tuner 已有 block swap / fp8 / int8 / 分块预缓存；我们做适配器与调度，不 fork、不内嵌（仓库无 LICENSE 文件，需法务确认）。
5. **性价比拐点在 16GB（5B 480p）与云端 4090（$0.89/h，一次视频 LoRA 约 $1–5）。** 本地 8GB 的价值是"可离线、可反复验证"，不是跑长任务。

## 2. 定位声明

SuperTuner = **单卡训练的可执行性控制平面**：

- 训练前给出逐项显存/主存账本与可行性判定，附可执行的下一步（降到哪个参数）。
- 把账本编译成可检查的执行计划（步骤、argv、环境、产物）。
- 统一 LLM 与视频的数据、运行、事件、导出、监控链路。
- 失败可读化：不写"out of memory"，写"把分辨率降到 320×576、把 8 块换到主存、batch 设为 1"。

## 3. 可行性真相表（2026-11 口径）

| 显存 | LLM | 视频 | 说明 |
| --- | --- | --- | --- |
| 8GB | QLoRA 4B–8B（seq 1024–2048） | Wan2.1 1.3B @480p；H3 @320p + 48/50 换块（实验） | H3 需要约 19GB 主存与 PCIe 搬运，属于证明性配置 |
| 12GB | QLoRA 7B–8B | Wan2.1 14B 低分辨率需换块；5B 需 12GB 起 | 5B 的 fp8 权重约 7.4–8.9GB |
| 16GB | QLoRA 14B 边缘 | Wan2.2 5B @480p（换块后 10 s/it 量级） | 社区公认的 5B 甜点档 |
| 24GB | 8B LoRA bf16；14B QLoRA | Wan2.2 A14B 换块 20–30 @480p；H3 @480p | 关掉换块约 3–5 s/it |
| 32GB | — | H3 @720p 换块 32/50 | 5090 档 |
| 48GB+ | — | H3 @720p rank 64；全参需 33B/66GB bf16，多卡 | 单卡全参仍不可行 |

判据来自本仓库 planner（`engine/src/supertuner_engine/plan.py`），配方数值见 `packages/core/src/recipes.ts`。

## 4. 选型

| 领域 | 选择 | 不选 | 原因 |
| --- | --- | --- | --- |
| 视频训练核 | 调用 musubi-tuner 进程 | fork musubi、SimpleTuner、diffusion-pipe | 上游已有 block swap/fp8/缓存；SimpleTuner 为 AGPL、diffusion-pipe 为 GPL 且原生 Windows 基本不可用 |
| LLM 训练核 | transformers + peft 自研 loop | torchtune（停维护）、Lightning、DeepSpeed | 事件级控制与单卡路径；DeepSpeed 在 Windows 编译门槛高，单卡收益不如 bnb 8bit |
| 后训练 | v0.1 只做 SFT/LoRA/QLoRA | TRL GRPO、veRL、OpenRLHF | GRPO 8B@20K 上下文约 54GB，单卡不成立；v0.2 再评估 |
| 服务端 | Fastify + node:sqlite | Electron、Postgres | 零原生编译、单进程、SSE 简单 |
| 前端 | React + 自研设计系统 | Tailwind/UI 套件、图表库 | 体积与风格一致 |
| 注意力 | SDPA | flash-attn / SageAttention / xformers | Windows 上三者要么只有 sdist，要么需要 CUDA Toolkit |
| 量化 | NF4（bnb）/ int8 / fp8 权重量化 | DeepSpeed offload | 单卡路径 |

## 5. v0.1 范围

**做**：LLM SFT + LoRA/QLoRA（原生实现）；视频 LoRA 编译与调度（musubi 适配器，24GB+ 常规路径、8GB H3 实验路径）；显存/主存规划器；配方；运行监督与 SSE；实时监控与图表；插件注册表与能力声明。

**不做（并给出原因）**：全参微调 H3（66GB bf16）；DMD/DMAD 执行（8×H200 量级，且 H3 许可禁止用其输出改进其它模型）；RL/GRPO（单卡不成立）；多机与云端调度（v0.2）；数据集打标流水线（v0.2）；P2P/远程 GPU 镜像（v0.3）。

## 6. 优先级

- **P0（v0.1 本轮）**：契约与 planner → 引擎插件/编译/运行时 → 原生 LLM SFT → musubi 适配器 → 服务端 API/SSE → UI（系统/配方/规划器/监控）→ 测试与回归。
- **P1**：偏好优化（DPO/ORPO/KTO，TRL 集成）、数据集工具（抽帧/打标/审核）、导出到 ComfyUI 目录结构、跨机断点续训。
- **P2**：云端回退配方（同一 RunSpec 搬到按小时 GPU，带续训）、多 GPU 编排、更多模型族（HunyuanVideo、Qwen-Image、LTX-Video）。

## 7. 风险登记

| # | 风险 | 影响 | 缓解 | 状态 |
| --- | --- | --- | --- | --- |
| R1 | H3 权重约 210GB + 地域限制许可 | 8GB H3 只是技术演示；部分地区不可用 | 配方显式标注下载量与许可；H3 路径标 experimental | 已缓解 |
| R2 | musubi CLI/日志不是稳定 API | 适配器易碎 | 只解析文档化进度行；dry-run 打印完整命令；锁版本范围 | 进行中 |
| R3 | 笔记本 PCIe 换块吞吐低 | 8GB H3 每步几十秒以上 | planner 给出理论下界与警告；UI 显示实测 tokens/s | 已缓解 |
| R4 | Windows bitsandbytes 不稳 | QLoRA 不可用 | 探测失败即报不可用，保留 LoRA bf16；不静默降级 | 进行中 |
| R5 | torch Windows 轮子按 cu 变体分仓 | 装到 CPU 版 torch | 安装脚本探测驱动与 cu 索引；`env` 命令显示 torch/CUDA 状态 | 进行中 |
| R6 | node:sqlite 仍标记实验性 | 未来 API 变动 | 隔离在 `db.ts`；固定 Node 版本要求 | 已缓解 |
| R7 | H3 几何为近似值（config.json 未公开） | 规划误差 ±15% | `approx=true` 标记；提供 params_b_override | 已缓解 |
| R8 | 云端比本地快且便宜 | 用户不用本地卡 | 定位为"离线可验证 + 必要时上云"，云端回退列入 P2 | 已决策 |

## 8. 被否决的方案

- **自研视频 DiT 训练核**：等于重写 musubi/diffusion-pipe 的 block swap、缓存、分布式，换来的是长期维护负担与更少的模型覆盖。
- **在浏览器里跑训练（WASM/WebGPU）**：显存与算子不满足视频训练，且与本地显卡市场无关。
- **把 H3 全参训练写进路线图**：单卡不可能；写进产品等于虚假宣传。
- **直接内嵌 SimpleTuner/diffusion-pipe**：AGPL/GPL 传染 + Windows 不可用。
- **让服务端直接调用 torch（Node 原生绑定）**：把 CUDA 崩溃风险放进主进程，放弃进程隔离。

## 9. 行动项与对应实现

| 调研结论 | 实现位置 |
| --- | --- |
| 显存账本先行 | `engine/src/supertuner_engine/plan.py`（VRAM+RAM 双账本、换块/量化/缓存/精度全部建模） |
| 失败可读化 | `Issue` 结构与 UI IssueList：每条风险带可执行的下一步 |
| 统一 LLM/视频链路 | `RunSpec`（config/dataset/train/output/backend）+ 统一事件协议 |
| 插件隔离 | entry points 只发现清单；执行在子进程，JSONL 事件回流 |
| 5B 是入口 | `packages/core/src/recipes.ts` 配方按 8/12/16/24/32/48GB 分档 |
| 许可与地域限制 | 配方 caveats、UI 许可提示、运行前确认 |
