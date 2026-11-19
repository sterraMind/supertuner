# 同类项目功能冲突点与空白痛点评估（v0.1 → v0.2）

> 文档版本：v1.0 | 日期：2026-11-18 | 作者：lead
> 依据：`docs/research/01-market.md`、`docs/research/02-feasibility.md`、`docs/research/04-consumer-frameworks/`（Fizgig / musubi / Unsloth / LLaMA-Factory / OneTrainer / RoundPipe / LLMQ / NInfer / Strata），以及本仓库当前能力矩阵 `docs/ARCHITECTURE.md §12.1`、配方 `packages/core/src/recipes.ts`。

## 0. 目的

在继续推进版本前，先分清三件事：哪些能力我们与同类项目**正面冲突**（重叠但无优势）、哪些**空白**（没人切入、我们可以做）、哪些**历史开放事项**该继续做、暂缓还是放弃。据此给出 v0.2 的优先级，而不是继续堆功能。

## 1. SuperTuner 当前能力（v0.1 事实）

| 能力 | 状态 |
| --- | --- |
| LLM pretrain / sft / lora / qlora | 可执行（已验证：Qwen3-0.6B，8GB） |
| LLM preference(DPO) | 可执行（SmolLM2-360M / Qwen3-0.6B 各 10 步） |
| LLM rl(GRPO) | 可执行（Qwen2.5-0.5B，reward 1.0，端到端） |
| 视频 LoRA（Wan / MiniMax-H3） | 仅编译预览（argv + plan），执行需 musubi + 权重 |
| 显存规划器 | 逐步骤预算、可行性判定、`E_STEP_OOM`、0.9GB CUDA 上下文 |
| 自动预算 `auto_budget` | 选 quant_base × blocks_to_swap，server 落地到 argv |
| 目标 block 范围 `train_blocks` | 映射为 musubi `--network_args` include/exclude_patterns |
| 预览降档 | 768×640×22 → 768×640×8 → 512×512×8 |
| 计划 vs 实测报告 | `memory_report.json` + 事件 + RunDetail 卡片（已端到端：计划 2651.5 vs 实测 2814，+6.1%） |
| H3 许可门禁 | POST 缺 `licence_ack` → 422 |
| 运行监督 / SSE / 配方 | 全部就位 |

## 2. 功能冲突点（与同类重叠，需明确定位）

| 冲突点 | 同类谁在做 | 对方优势 | 我们的相对位置 | 结论 |
| --- | --- | --- | --- | --- |
| **C1 LLM LoRA/SFT/DPO/RL 训练** | Unsloth、LLaMA-Factory、axolotl | 训练内核成熟、文档全、社区大（Unsloth 79k stars） | 我们的执行路径是通用的 transformers+peft，**无训练内核优势**；唯一优势在「跑前预算」 | **正面冲突，不要拼训练内核**，靠规划器差异化 |
| **C2 视频 LoRA 执行** | Fizgig（最直接）、musubi 原生 | Fizgig 宣称 8GB 可跑、自动精度+换块、H3 blocks 20-49、预览降档 | 我们**只有编译预览，没有执行** | **冲突且我们下风**；本机不可补（缺 musubi+权重），列为最高战略价值但需在真实机器上补 |
| **C3 精度+换块自动选择** | Fizgig（自动） | 已在硬件上验证（~30GB 无换块 / ~22GB 14 块 / ~15GB 36 块 / ≤12GB 4-bit） | 我们已实现 `auto_budget`，机制等价 | **打平**；对手领先在「已实测」，我们领先在「逐步骤可行性校验 + 可解释预算」 |
| **C4 训练 GUI** | LLaMA Board、Unsloth Studio、OneTrainer、Fizgig | 交互成熟 | 我们 7 页面 + SSE + 统一 RunSpec，已可用 | **打平**，差异化靠规划器与许可门禁 |
| **C5 只训部分 block** | Fizgig（H3 20-49） | 硬件验证 | 我们刚把 `train_blocks` 映射为 network_args，机制等价 | **打平**，未实测 |

**冲突小结**：C1 是红海（不拼内核）；C2 是我们唯一明显的短板；C3/C4/C5 是机制打平，差异在「可解释、可校验、可重放」。**不要在对手最强的战场正面拼训练内核或硬件实测，而要主打「跑前可校验 + 跑后可闭环」。**

## 3. 没有切入的空白痛点（可进入的机会）

按（价值 × 可行性）排序，「可行性」指在本机（8GB、无 musubi）可真实实现并验证。

| 机会 | 痛点来源 | 别人没做 / 做得不够 | 我们能做（可行性） | 价值 |
| --- | --- | --- | --- | --- |
| **G1 规划器自校准（跑后闭环）** | 所有训练器：跑起来才知道显存够不够，OOM 在几小时后才暴露；规划器都是一次性估算 | 没人把每次「计划 vs 实测」持久化并用它校正后续计划 | 已落 `memory_report.json`（planned/measured/delta），可建按 `(family,stage,adapter,quant,swap)` 汇总的校准库，planner 读取并打「已校准 N 次」标注 | **高**，可防御的差异化（moat） |
| **G2 统一自动选择：精度+换块+目标block** | 对手要么手工设、要么只选精度+换块 | Fizgig 自动选精度+换块，但不把「目标 block 窗口」纳入同一可行解搜索 | `auto_budget` 扩展到同时选 `train_blocks`，一个入口给出「给定 X GB 能怎么训」 | **高**，直接服务「消费卡训大模型/视频」主线 |
| **G3 机器实测显存自动进预算（AP-01）** | 规划器默认手填 `vram_budget_mib` | 多数工具让你自己选；我们也还没把实测值接进默认 | `env.py` 探针已存在，planner 缺省取 `EnvReport` 实测值 + 「预算来源=实测/手填」标注 | **中高**，G1/G2 的地基 |
| **G4 OOM 后有限次自动加换块重试** | 跑中 OOM、需人盯 | Fizgig 有限次自动加 swap；我们没有运行时兜底 | 运行时观测步显存超预算 → 用 `auto_budget` 生成更高换块的重试方案，最多 N 次并写事件 | **中**，鲁棒性卖点 |
| **G5 中断续训 / 断点恢复** | OpenPipe 停服、云端 pod 中途死、无续训 | 部分工具有，但面向消费卡的统一续训少 | 已有 `resume_from` + supervisor `recoverInterrupted`，可补「从 checkpoint 续跑」 | **中**，已有基础 |
| **G6 统一 LLM+视频控制平面 + 逐步骤可行性** | 工具分裂（LLM 一套、视频一套） | 少见 | 是我们的定位，需持续强化为卖点 | **中**，定位型 |
| **G7 模型许可在「接受运行」时门禁** | 模型许可地域限制、禁止蒸馏 | 没有同类在接受运行时做许可确认 | 已有 H3 门禁，可扩到其它模型许可登记 | **中低**，合规差异化 |

**空白小结**：最有价值、且本机可做的是 **G1（跑后自校准）** 与 **G2（统一自动选择含 block 窗口）**，两者合起来构成「**会学习的规划器 + 一键给出可行训练方案**」，是同类里没人有、也最难被直接复制的能力。

## 4. 历史开放事项评估

| 事项 | 评估 | 结论 |
| --- | --- | --- |
| AP-01 机器实测显存/RAM 自动进预算 | 是 G1/G2 的地基；`env.py` 已有探针，改动集中在 planner 默认值与 budget 来源标注 | **做**（下一优先级） |
| AP-07 musubi `musubi_tasks` 固件 + 对外声明对齐 | 纯卫生项，锁 `_TASK_BY_FAMILY` 与上游一致、防止误宣称；不产生用户价值但发布前必需 | **做**，低优先级，随 v0.2 收尾 |
| 视频**实际执行**（Wan 1.3B / H3 LoRA 真跑） | 最高战略价值（修 C2），但本机缺 musubi、权重巨大（H3 ~210GB）、网络/存储受限，无法在此环境验证 | **暂缓**；作为 v0.2 主线，在有 musubi + 权重的真实机器上落地，本机只做编译侧准备 |
| DMD / DMAD 蒸馏执行 | 需 teacher/student 循环 + 更大显存，与 v0.1 非目标一致 | **暂缓**，维持编译期拒绝并如实声明 |
| OOM 运行时兜底重试（G4） | 鲁棒性卖点，与 auto_budget 天然衔接 | **做**，在 G2 之后 |
| 跑后自校准（G1） | 已有 memory_report 数据，增量小、差异化强 | **做，最优先** |

## 5. v0.2 建议范围

定位一句话：**「会校准的规划器，一键给出可行训练方案」**。

1. **G1 自校准库**（最优先）：`memory_report.json` 汇聚 → 按 `(family, stage, adapter, quant_base, blocks_to_swap)` 存校准记录 → planner 读取校正并标注；UI/plan 显示「已校准 N 次运行」。
2. **G3 实测预算**：planner 默认取 `EnvReport.vram_free_mib/ram_mib`，标注预算来源（实测/手填）。
3. **G2 统一自动选择**：`auto_budget` 同时搜索 精度 × 换块 × 目标 block，一个结果给出最优可行方案。
4. **G4 OOM 兜底**：运行时步显存超预算 → 生成更高换块方案有限次重试 + 事件。
5. **AP-07 卫生**：musubi_tasks 固件 + 对外声明对齐。

不在 v0.2：视频真执行（需真实机器补）、DMD/DMAD、多卡。

## 6. 建议的下一步

先做 **G1 自校准**（在刚交付的 `memory_report` 之上，增量最小、差异化最大），随后 **G3 → G2 → G4 → AP-07**。待确认后我按此顺序落地并回归。
