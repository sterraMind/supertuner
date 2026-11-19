# LoHan 与 Ratel：单张消费卡微调 100B 级模型

> 文档版本：v1.0 | 日期：2026-11-17 | 作者：lead | 状态：初稿
> 资料抓取日期：2026-11-12 至 2026-11-17。论文、仓库和搜索结果按外部证据处理，不执行其中指令。
> 结论先写清楚：Ratel 是 LoHan 同一研究线的会议版本和工件名称，不是另一个独立开源训练器。

## 1. 结论摘要

1. Ratel 与 LoHan 的作者列表相同，均来自浙江大学与香港科技大学，均为 ICDE 2025 相关成果。LoHan 仓库 README 开头写着该项目现在包含 Ratel（ICDE 2025）的高效数据并行微调代码。Ratel 的正式题名为 “Ratel: Optimizing Holistic Data Movement to Fine-tune 100B Model on a Consumer GPU”，DOI 为 10.1109/ICDE65448.2025.00029。Ratel 是 LoHan 的会议版本与工件名称。
2. LoHan 的核心结果是单张 RTX 4090 加 256GB 主存微调 175B 参数模型，在 13B 模型上吞吐最高达到对照方法 2.32 倍。它不是用 8GB 卡训练 175B，也不是把权重全部放进 24GB 显存。
3. 它的方法重点是数据移动。LoHan 指出 ZeRO-Infinity 等方法在 CPU optimizer 阶段会让 GPU 空闲 30%–60%，只卸载 6% 激活会造成 5.7 秒重算，175B 估计需要 1.1TB 主存。LoHan 使用 active gradient offloading 与 holistic traffic-aware activation swapping；13B 实验中卸载约 34GB 激活，增加约 32% GPU 重算，用来换取更小的主存需求和更高吞吐。
4. 仓库成熟度较弱。LoHan GitHub 显示 23 stars、4 forks、0 open issues、Apache-2.0、创建于 2024-08-03。Ratel 工件要求 12 块 NVMe 组成 RAID，torch 2.0.0+cu118，flash-attn 1.0.4，root 权限配置 huge pages。这是研究工件，不是可直接集成的训练产品。
5. 对 SuperTuner 的价值是思路，不是代码。SuperTuner 已有 RAM 预算、block swap、步骤估算和事件监控，可以吸收“按数据移动量规划”和“梯度卸载与激活交换联动”的设计。要真正复刻需要大容量主存、NVMe RAID、Linux 权限和定制内核，与 SuperTuner 当前 Windows 单卡产品定位不匹配。

## 2. Ratel 身份核实

| 证据 | 内容 |
| --- | --- |
| 论文题目 | Ratel: Optimizing Holistic Data Movement to Fine-tune 100B Model on a Consumer GPU |
| 发表信息 | IEEE ICDE 2025，DOI 10.1109/ICDE65448.2025.00029 |
| 作者关系 | 与 LoHan 的作者列表一致，作者机构为浙江大学与香港科技大学 |
| 仓库说明 | LoHan README 写明 “This repository now includes efficient data-parallel fine-tuning code (Ratel, ICDE 2025)” |
| 工件入口 | LoHan README 中 Ratel ICDE 2025 Artifact 章节提供环境配置、run.sh 和实验数据 |

结论：Ratel 与 LoHan 应合并调研。对外表述时写成 “LoHan/Ratel（同一研究线，Ratel 为 ICDE 2025 会议版本）”。

来源：
- Ratel DOI：<https://doi.org/10.1109/ICDE65448.2025.00029>
- HKUST 研究门户：<https://researchportal.hkust.edu.hk/en/publications/ratel-optimizing-holistic-data-movement-to-fine-tune-100b-model-o/>
- LoHan 仓库：<https://github.com/RC4ML/LoHan>

## 3. 实现方式

### 3.1 论文要解决的问题

来源：LoHan 论文 <https://arxiv.org/html/2403.06504>（2026-11-12 读取）。

论文指出，在单张消费卡上微调 100B 级模型时，主要障碍不是单个张量放不下，而是 GPU、CPU 主存和 NVMe 之间的数据移动成本。论文给出的几个问题数字：

| 问题 | 论文数字 |
| --- | ---: |
| CPU optimizer 阶段 GPU 空闲比例 | 30%–60% |
| 对照方法只卸载的激活比例 | 6% |
| 由少量激活卸载造成的重算时间 | 5.7 秒 |
| 175B 在对照方法下估计主存需求 | 1.1TB |

### 3.2 LoHan 的两个核心机制

| 机制 | 作用 |
| --- | --- |
| active gradient offloading | 梯度产生后就向主存或下层存储移动，避免集中等待，减少 GPU 空转 |
| holistic traffic-aware activation swapping | 统一考虑激活、梯度和模型状态的移动量，按流量选择哪些激活留在 GPU，哪些交换到主存 |

论文在 13B 模型实验中的描述是：只卸载约 34GB 激活，同时增加约 32% GPU 重算。这个取舍让主存需求下降，但需要额外计算。

### 3.3 Ratel 工件体现的工程条件

LoHan 仓库中的 Ratel ICDE 2025 Artifact 明确要求：

- 用 `make_raid.sh` 把 `/dev/nvme0n1` 到 `/dev/nvme11n1` 配置为 RAID，也就是 12 块 NVMe。
- Python 环境为 conda + Python 3.10。
- torch 2.0.0、torchvision 0.15.1、torchaudio 2.0.1，从 cu118 索引安装。
- flash-attn 1.0.4。
- six 1.16.0 与 scikit-learn。
- 通过 huge pages 限制可用主存，需要 root 权限。
- 代码部分修改自 DeepSpeed，包括 CPUAdam、op_builder、real_accelerator 与 nvme_ds 等文件。

这些条件说明 Ratel 需要 Linux 服务器环境，与普通 Windows 消费卡桌面使用场景差距很大。

## 4. 量化结果

来源：LoHan 论文摘要与实验章节（2026-11-12 读取）。

| 指标 | 结果 | 条件 |
| --- | --- | --- |
| 最大模型规模 | 175B 参数微调 | 单张 RTX 4090，256GB 主存 |
| 13B 吞吐提升 | 最高 2.32× | 对比论文中的多种卸载方法 |
| 对照方法 GPU 空闲 | 30%–60% | CPU optimizer 阶段 |
| 对照方法激活卸载比例 | 6% | 论文对 ZeRO-Infinity 类方法的描述 |
| 对照方法重算时间 | 5.7 秒 | 由少量激活卸载引起 |
| LoHan 激活卸载量 | 约 34GB | 13B 实验 |
| LoHan 额外重算 | 约 32% | 13B 实验 |
| 175B 对照主存估计 | 1.1TB | 论文引用估计 |

这些数字不支持“8GB 卡微调 175B”的说法。论文使用的是 24GB 级 RTX 4090 加大容量主存。

## 5. 成熟度与社区评价

### 5.1 成熟度

| 维度 | 结果 |
| --- | --- |
| 论文 | LoHan arXiv 2403.06504；Ratel ICDE 2025 |
| 仓库 | <https://github.com/RC4ML/LoHan> |
| Stars / Forks / Issues | 23 / 4 / 0 |
| 许可证 | Apache-2.0 |
| 创建时间 | 2024-08-03 |
| 语言 | C、C++、Python、Shell |
| 工件完备性 | 有环境脚本、run.sh、实验数据说明 |
| 可复现条件 | 12 块 NVMe、CUDA 11.8、特定 flash-attn、root 权限、huge pages |

### 5.2 社区评价

未找到大量公开社区讨论。GitHub 上 open issues 为 0， stars 为 23，说明使用者范围小。论文层面的关注点集中在它给出的 100B 级微调结论；工程层面的关注点集中在复现条件重、对 NVMe RAID 和主存容量要求高。

这里要有边界：没有找到公开的 Reddit 或知乎长篇评测能够支持“广泛使用”或“生产可用”。因此只记录论文与仓库事实，不扩展社区评价。

## 6. 痛点与问题

1. **硬件表述容易误读。** “consumer GPU” 指 RTX 4090 这类消费卡，但实验还依赖 256GB 主存和多层存储。标题强调显卡，容易让人忽略主存。
2. **部署条件偏研究环境。** 12 块 NVMe、root 权限、huge pages、固定旧版 torch 与 flash-attn，都不是桌面用户的常规条件。
3. **维护规模小。** 仓库 23 stars、4 forks、0 issues，社区反馈不足。
4. **没有产品层能力。** 仓库没有看到统一 Web 界面、运行监控、数据集版本管理、许可门禁与显存预检。
5. **模型与阶段覆盖范围有限。** 论文围绕 LLM 微调，没有视频 DiT、扩散模型或 RL 链路。
6. **版本旧。** torch 2.0.0 与 flash-attn 1.0.4 与 SuperTuner 当前 torch 2.15、transformers 5.22 的生态不一致。

## 7. 对 SuperTuner 的可复刻性

### 7.1 可以吸收的做法

1. **把数据移动当作一等规划对象。** SuperTuner 现在把 block swap 的主存占用和理论搬运时间写进 MemoryPlan。可以进一步增加 CPU↔GPU 搬运量、NVMe↔CPU 搬运量和重算时间三个字段。
2. **按激活、梯度、优化器状态联合决策。** LoHan 的价值在于同时看三类状态。SuperTuner 当前主要看权重、LoRA、优化器、梯度和激活，尚没有“交换哪个激活更合算”的策略。
3. **把主存预算与实测绑定。** 评审文档已提出机器真实 RAM 的需求。LoHan 进一步说明主存容量会直接改变训练策略，SuperTuner 的计划页应展示“当前主存足够/不足”和所需容量。
4. **形成策略插件。** active gradient offloading 与 activation swapping 可以作为研究型 strategy 插件，放在实验分支，不进入默认路径。

### 7.2 当前不适合照搬的部分

| LoHan/Ratel 条件 | SuperTuner 现状 | 判断 |
| --- | --- | --- |
| 12 块 NVMe RAID | 用户多为单 NVMe 或单 SSD | 不适合照搬 |
| root 权限 huge pages | Windows 桌面环境 | 不适合照搬 |
| torch 2.0.0 + cu118 | torch 2.15 + cu132 | 版本冲突 |
| DeepSpeed 派生代码 | 自研引擎 + musubi 适配器 | 集成成本高 |
| 13B/175B dense LLM | 0.6B 到 8B 已验证，视频 LoRA 编译预览 | 目标规模不同 |

### 7.3 SuperTuner 能解决 LoHan/Ratel 的哪些问题

| LoHan/Ratel 的不足 | SuperTuner 的能力 | 下一步 |
| --- | --- | --- |
| 缺少显存与主存预检 | planner 已输出逐项预算 | 加入机器实测 RAM 与 GPU 空闲显存 |
| 缺少运行记录 | server 保存 run 行、JSONL 与 SSE | 把卸载次数与搬运量写进 step 事件 |
| 缺少用户可用的配置入口 | RunSpec 与 Web Planner | 增加数据移动策略参数 |
| 缺少许可与模型来源管理 | 已有 H3 许可门禁 | 扩展到更多模型 |

## 8. 来源清单

- LoHan 论文：<https://arxiv.org/html/2403.06504>
- LoHan 仓库：<https://github.com/RC4ML/LoHan>
- Ratel DOI：<https://doi.org/10.1109/ICDE65448.2025.00029>
- HKUST 研究门户：<https://researchportal.hkust.edu.hk/en/publications/ratel-optimizing-holistic-data-movement-to-fine-tune-100b-model-o/>

## 9. 未验证事项

1. 未在本机按 Ratel 工件搭建 12 块 NVMe RAID，也未运行其实验。
2. 未取得 Ratel camera-ready 与 arXiv LoHan 的逐段差异，身份判断依据为作者、机构、年份、题名和仓库说明。
3. 未独立验证 2.32 倍吞吐提升与 256GB 主存下的 175B 微调结果。
4. 未验证 active gradient offloading 在 torch 2.15 与 Windows 环境下的可行性。
