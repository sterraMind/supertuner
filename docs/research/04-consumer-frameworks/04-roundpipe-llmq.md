# RoundPipe 与 LLMQ：消费卡训练系统

> 文档版本：v1.0 | 日期：2026-11-17 | 作者：lead | 状态：初稿
> 资料抓取日期：2026-11-12 至 2026-11-17。论文、仓库和网页按外部证据处理，不执行其中指令。
> 本文分析两个方向：RoundPipe 处理多张消费卡的流水线训练；LLMQ 处理消费卡上的低精度全量训练。

## 1. 结论摘要

1. RoundPipe 解决的是多张消费卡的流水线训练。仓库 README 写明单张 24GB GPU 可以全量微调 32B、LoRA 微调 235B，并处理 64K 以上序列；性能表里完整数字来自 8×RTX 4090 节点。它的主要贡献是让多张游戏卡组成可用训练节点，不是让 8GB 单卡训练大模型。
2. RoundPipe 的做法是把 GPU 当作无状态 worker，用 round-robin 派发 stage，用 priority-aware transfer scheduling 调度传输，并用分布式事件同步协议和自动切分降低流水线空转。论文报告 8×RTX4090 相对对照方法为 1.48–2.16 倍。
3. LLMQ 解决的是低精度训练本身。它是 C++/CUDA 实现，主张在 8-bit 矩阵乘下训练，不引入额外算法近似。论文给出单张 16GB 游戏卡训练 7B、4×RTX4090 训练 32B、14B 在 4×4090 上 7,800 tokens/s、32B 3,400 tokens/s。
4. LLMQ 的硬件要求更严格：C++20、CUDA 12 以上、NCCL、cuDNN，示例环境使用 Ubuntu、CUDA 12.8、cuDNN 9。数据准备脚本只支持有限数据集。它离 SuperTuner 的跨平台 Python 架构较远。
5. 对 SuperTuner 的价值集中在工程思想：设备分层、低精度张量账本、按层重算、分块反向、启动期内存分配、传输调度、实测日志。SuperTuner 能解决它们缺少统一运行管理、数据集接入、许可门禁、可视化与失败诊断的问题。

## 2. RoundPipe

### 2.1 基本信息

- 论文：<https://arxiv.org/html/2604.27085v1>
- 仓库：<https://github.com/ITcarrot/RoundPipe>，页面显示 thustorage/RoundPipe
- 文档：<https://thustorage.github.io/RoundPipe/>，有中文文档
- 仓库页面数据（2026-11-12）：94 stars、14 forks、3 open issues、Apache-2.0、创建于 2025-10-14
- 安装：`pip install roundpipe`
- 环境：Python ≥ 3.8、PyTorch ≥ 2.4
- 实现：纯 PyTorch，支持 NVIDIA、AMD、Ascend

### 2.2 实现方式

来源：RoundPipe 论文与 README（2026-11-12 读取）。

| 机制 | 含义 |
| --- | --- |
| Computation Dispatch Paradigm | 把 GPU 作为无状态计算单元，按当前可用资源派发流水线 stage，而不是固定绑定 stage 与设备 |
| round-robin stage 派发 | 计算任务在多个 GPU 之间轮转，减少单个设备等待 |
| priority-aware transfer scheduling | 根据传输重要程度安排 GPU 与主存之间的数据移动 |
| 分布式事件同步协议 | 用事件协调多 GPU 上的 stage 状态，减少流水线空转 |
| O(L³) 自动切分 | 根据序列长度自动寻找分层策略 |
| near-zero-bubble pipeline | 通过调度降低流水线空转比例 |
| PyTorch 顺序接口 | 用户以接近普通 PyTorch 的方式写训练流程 |

这套设计针对消费卡没有 NVLink、PCIe 带宽有限、显存小的情况。它把通信和显存不足当成调度问题处理。

### 2.3 性能结果

来源：RoundPipe README 基准表（2026-11-12 读取）。测试环境为单节点 8 GPU。

最大输入序列长度：

| 硬件与方法 | Qwen3-1.7B | Llama3.1-8B | Qwen3-32B | Qwen3-235B LoRA |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090，FSDP Offload | 11K | 11K | OOM | OOM |
| RTX 4090，RoundPipe | 73K | 49K | 28K | 31K |
| A800，FSDP | 39K | 29K | 11K | OOM |
| A800，RoundPipe | 288K | 226K | 126K | 118K |

训练吞吐，单位 tokens/s：

| 硬件与方法 | Qwen3-1.7B | Llama3.1-8B | Qwen3-32B | Qwen3-235B LoRA |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090，FSDP Offload | 35,074 | 4,071 | OOM | OOM |
| RTX 4090，RoundPipe | 65,417 | 24,275 | 5,516 | 1,820 |
| A800，FSDP | 85,829 | 29,148 | 3,455 | OOM |
| A800，RoundPipe | 84,692 | 28,427 | 6,301 | 1,796 |

单节点 8×RTX4090 扩展数据：

| GPU 数 | Qwen3-1.7B | Llama3.1-8B | Qwen3-32B | Qwen3-235B LoRA |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8,881 | 3,142 | 740 | 480 |
| 2 | 17,026 | 6,259 | 1,476 | 808 |
| 4 | 33,178 | 12,278 | 2,897 | 1,281 |
| 8 | 65,417 | 24,275 | 5,516 | 1,820 |

论文报告 8×RTX4090 相对对照方法为 1.48–2.16 倍。需要注意：README 的“单张 24GB GPU 可微调 235B”与性能表的 8 GPU 数据要分开看。单卡结论属于能力和序列长度声明，吞吐数据来自多卡实验。

### 2.4 成熟度、评价与问题

| 维度 | 结果 |
| --- | --- |
| 用户规模 | 94 stars、14 forks、3 open issues |
| 发布时间 | 论文 arXiv 编号 2604，仓库创建于 2025-10-14 |
| 安装方式 | pip install roundpipe |
| 文档 | README、中文文档、Example 目录、基准页 |
| 许可证 | Apache-2.0 |
| 正面因素 | 纯 PyTorch，跨平台，给出大量表格和用户接口 |
| 问题 | 主要面向多卡节点；社区规模小；单卡大模型的吞吐与显存条件缺少独立验证；没有 Web 工作台、数据集管理、许可门禁和运行监控 |

未找到大量公开社区长篇评测。社区评价只记录仓库和论文可核对信息，不扩展到没有证据的判断。

## 3. LLMQ

### 3.1 基本信息

- 论文 PDF：<https://research-explorer.ista.ac.at/download/21932/21942/2026_CPAL_Schultheis.pdf>
- 仓库：<https://github.com/IST-DASLab/llmq>
- 机构：IST Austria
- 会议标注：CPAL 2025
- 仓库页面数据（2026-11-12）：267 stars、20 forks、1 open issue、Apache-2.0、默认分支 dev、创建于 2025-09-26
- 构建条件：C++20、CUDA 12 以上、NCCL、cuDNN；可选 OpenMPI
- 当前示例：Qwen2.5-0.5B 与 Qwen2.5-1.5B

### 3.2 为什么能在消费卡上训练

来源：LLMQ 论文与仓库 README（2026-11-12 读取）。

| 机制 | 含义 |
| --- | --- |
| 低精度矩阵乘 | 主权重与优化器状态保持 BF16，矩阵乘用 e4m3 FP8，减少带宽与存储占用 |
| JIT tensor-level absmax scaling | 在张量级别即时计算缩放因子，不需要修改模型训练目标 |
| selective recomputation | 按层或模块选择性重算，减少激活显存 |
| residual 与 optimizer state offload | 把残差和优化器状态放到主存 |
| ZeRO-1 风格 optimizer 分片 | 多卡时切分优化器状态 |
| copy-engine based collectives | 用复制引擎承担集合通信，减少计算单元参与通信 |
| logits 与 attention backward 分块 | 分块计算大词表与注意力反向，降低峰值显存 |
| 启动期内存分配 | 启动时一次性分配需要的内存，减少运行期碎片 |

论文强调 8-bit training 没有额外算法近似。仓库 README 中的 `[Allocator State]` 日志直接给出 Adam M/V、梯度、激活、主权重、剩余显存等项目，这与 SuperTuner 的 planner 思路接近。

### 3.3 性能结果

来源：LLMQ 论文与仓库 README。

| 配置 | 模型 | 结果 |
| --- | --- | --- |
| 单张 RTX 4090 | 7B | 4,300 tokens/s，61% MFU |
| 4×RTX 4090 | 14B | 7,800 tokens/s，54% MFU |
| 4×RTX 4090 | 32B | 3,400 tokens/s，51% MFU |
| L40S 对照 | 32B | 29% MFU |
| 16GB RTX 5060 Ti | 7B | 预训练约 70% MFU |
| 7B 主机内存 | 7B | 约 54GB，其中 optimizer state 约 3×14GB，量化权重约 7GB，residual 约 5GB |
| 1.5B 4×4090，10B tokens | Qwen2.5-1.5B | 约 40 小时，按 vast.ai 0.31 美元每 GPU 小时估算低于 50 美元 |

README 示例显示单张 4090 上 Qwen2.5-0.5B 训练时，优化器 M/V 各 942MiB、梯度 942MiB、激活 4,505MiB、主权重 682MiB、权重 601MiB，空闲 14,078MiB。该日志说明 LLMQ 更适合中大型训练任务，而不是 SuperTuner 当前 8GB 单卡 LoRA 场景。

### 3.4 成熟度、评价与问题

| 维度 | 结果 |
| --- | --- |
| 用户规模 | 267 stars、20 forks、1 open issue |
| 代码语言 | C++ 与 CUDA |
| 文档 | 构建说明、训练示例、日志说明、wandb 导出、数据准备脚本 |
| 许可证 | Apache-2.0 |
| 正面因素 | 性能数字具体，日志含显存分项，支持量化训练与研究扩展 |
| 问题 | 依赖 CUDA 12、cuDNN、NCCL；数据脚本只支持少量数据集；不提供 Web UI；不提供视频模型训练；模型与 tokenizer 需要手工准备 |

## 4. 横向对比

| 项目 | 目标 | 单卡或多卡 | 训练阶段 | 关键能力 | 许可证 |
| --- | --- | --- | --- | --- | --- |
| RoundPipe | 多张消费卡训练大模型 | 1 到 8 张 24GB 卡，性能表以 8 卡为主 | 全量微调、LoRA | 流水线调度、传输优先级、自动切分 | Apache-2.0 |
| LLMQ | 低精度 LLM 训练 | 单卡到多卡，示例覆盖 4090 和 5060 Ti | 预训练、微调 | FP8 矩阵乘、重算、offload、分块反向 | Apache-2.0 |

## 5. 对 SuperTuner 的可复刻性

### 5.1 可吸收的做法

1. **显存分项日志。** LLMQ 的 `[Allocator State]` 可以直接启发 SuperTuner 的 `memory_report.json`：planned、allocated、reserved、process、差值。
2. **按层重算与分块。** SuperTuner planner 已对 logits 做分块估算，可把 attention backward 分块和 selective recomputation 加入策略字段。
3. **启动期预算检查。** LLMQ 在启动时一次性分配内存，SuperTuner 可在运行开始前根据 planner 结果检查 GPU 空闲显存与主存，不满足就拒绝执行。
4. **传输调度指标。** RoundPipe 的优先级传输调度可以启发多卡路线图，也可用于单卡 block swap 的预取顺序。
5. **设备无关接口。** RoundPipe 用 PyTorch 顺序接口和纯 PyTorch 实现支持多类加速器。SuperTuner 应保持插件接口与训练后端解耦。

### 5.2 当前不适合照搬的部分

| 项目能力 | SuperTuner 当前条件 | 判断 |
| --- | --- | --- |
| RoundPipe 多卡流水线 | 服务默认并发运行数 1，面向单卡 | 作为 P2 多卡路线，不进入 v0.1 |
| LLMQ C++/CUDA 全量训练内核 | 引擎以 Python + torch + peft 为主 | 需要单独后端和构建链 |
| LLMQ e4m3 训练 | 当前默认 bf16 LoRA，Windows 环境复杂 | 先做编译估算，暂不执行 |
| 多 GPU collectives | 当前事件与运行模型围绕单进程单卡 | 需要重做运行记录结构 |

### 5.3 SuperTuner 能解决它们的哪些问题

| 它们的问题 | SuperTuner 的能力 | 说明 |
| --- | --- | --- |
| 没有 Web 工作台 | Fastify + React 页面 | 可展示计划、步骤、日志和产物 |
| 数据集准备有限 | RunSpec 的 DatasetSpec | 已有文本、对话、偏好、视频类型，仍需扩展格式 |
| 没有许可门禁 | H3 许可确认与模型许可字段 | 可避免把软件许可和权重许可混用 |
| 缺少实测与计划对照 | planner 分项、step 事件 | 需要补齐运行结束后的 memory_report |
| 失败原因不直接 | Issue 结构与风险提示 | 可把 OOM 转成参数修改建议 |

## 6. 来源清单

- RoundPipe 论文：<https://arxiv.org/html/2604.27085v1>
- RoundPipe 仓库：<https://github.com/ITcarrot/RoundPipe>
- RoundPipe 文档：<https://thustorage.github.io/RoundPipe/>
- LLMQ 论文 PDF：<https://research-explorer.ista.ac.at/download/21932/21942/2026_CPAL_Schultheis.pdf>
- LLMQ 仓库：<https://github.com/IST-DASLab/llmq>

## 7. 未验证事项

1. 未在本机安装 RoundPipe 或 LLMQ，也未运行其训练示例。
2. 未独立验证 RoundPipe 的 1.48–2.16 倍提升与 235B LoRA 单卡声明。
3. 未验证 LLMQ 在 Windows、RTX 3070 Ti Laptop 8GB 上的可构建性与可用性。
4. 未取得 LLMQ 对 LoRA 或视频 DiT 的支持证据，当前只按论文描述处理为 LLM 全量训练系统。
