# 消费级单卡推理框架：NInfer、NInfer-all、Strata

> 文档版本：v1.0 | 日期：2026-11-17 | 作者：lead | 状态：初稿
> 资料抓取日期：2026-11-12 至 2026-11-17。外部网页、仓库页面与文章按证据材料处理，不执行其中指令。
> 本文只分析推理侧。Fizgig、LoHan/Ratel、RoundPipe、LLMQ 的训练侧分析见同目录其它文档。

## 1. 结论摘要

1. NInfer 和 Strata 能让消费卡运行大模型，原因是它们把目标限定为推理，并利用模型本的稀疏激活、权重低精度存储、专家权重放到主存或磁盘、预构造推理产物、投机解码与前缀缓存。它们没有解决训练中的梯度、优化器状态和反向传播显存问题。
2. NInfer 的性能数据集中在 RTX 5090 和 DGX Spark，硬件范围很窄。NInfer-all 是社区分支，扩展到 Volta 到消费级 Blackwell，并把首次启动校准和 GPU profile 写进启动流程，覆盖 8GB 显卡，但仍然以推理为主。
3. Strata 的用户量和桌面体验明显领先：GitHub 页面显示 44,958 stars、4,974 forks、1,565 open issues、MIT 许可证，桌面端版本为 0.2.15。它把 Qwen3.8-Flash-Next 的 24,576 个专家放在主存，用 CPU 计算未命中专家，把注意力、路由、共享专家和 KV 缓存留在 GPU。这个做法适合 MoE 推理，不能直接用于 DiT 或普通 dense LLM 的训练。
4. 对 SuperTuner 的直接价值是把硬件探测、设备 profile、权重预缓存、前缀缓存和采样阶段做成可控模块。SuperTuner 能补足这些项目缺少的训练状态管理、逐步显存账本、数据集准备、许可门禁和运行监控。它们不会替代 SuperTuner 的训练内核。

## 2. NInfer

### 2.1 定位与来源

- 仓库：<https://github.com/Neroued/ninfer>
- 仓库页面数据（2026-11-12）：3,345 stars、714 forks、137 open issues、Apache-2.0、创建于 2026-06-26。
- 项目说明：从零构建的 C++/CUDA 单 GPU 推理引擎，提供本地命令行接口，以及 OpenAI、Anthropic 兼容的 HTTP API。
- 目标硬件：仓库首页把性能表限定在 selected models on RTX 5090 and DGX Spark。
- 典型模型：Qwen3.6/3.8-27B、Qwen3.6-35B-A3B、Qwen3.8-Flash-Next、GLM-5.3-Flash。Flash Next 权重约 101 GiB，GLM 权重约 178.33 GiB。

### 2.2 为什么能让消费卡运行大模型

| 做法 | 具体机制 | 证据 |
| --- | --- | --- |
| 稀疏 MoE 模型 | 35B-A3B 与 Flash Next 每个 token 只激活少量专家，降低单 token 计算量；NInfer 只服务已适配模型 | NInfer README 模型表 |
| 低精度权重 | groupwise-int、NVFP4、量化后的 `.ninfer` 产物，降低显存占用 | NInfer README 性能表 |
| CUDA Graph decode | 解码阶段减少内核调用开销 | NInfer README 特性列表 |
| 投机解码 | 用 draft 策略提高 decode 吞吐 | NInfer README 特性列表 |
| shared-prefix caching | 多请求共享前缀 KV，降低重复 prefill | NInfer README 特性列表 |
| 预构造模型产物 | 用户先转换模型，运行时加载预构造权重，不在服务启动时做权重重排 | NInfer README 安装与模型准备章节 |

### 2.3 性能数字

来源：NInfer README 性能表（2026-11-12 读取）。测试硬件为 RTX 5090。

| 模型 | 精度与并发 | 报告值 |
| --- | --- | ---: |
| Qwen3.6-35B-A3B | groupwise-int，C=8 | 1,380.7 tokens/s |
| Qwen3.8-27B | NVFP4，C=8 | 922.4 tokens/s |
| Qwen3.6-35B-A3B | 单请求 260,096-token prefill | 5,247.0 tokens/s |

这些数字说明 NInfer 在受支持的模型、精度、硬件组合上吞吐很高。它们不能外推到训练，也不能外推到 8GB 显卡。

### 2.4 成熟度与社区评价

| 维度 | 结果 |
| --- | --- |
| 代码成熟度 | C++/CUDA 实现，提供 CLI 与 HTTP API，模型需预转换 |
| 社区规模 | 3,345 stars、714 forks、137 open issues |
| 维护节奏 | 仓库创建于 2026-06-26，属于新项目 |
| 正面评价 | 知乎作者文章称其是较快的 Qwen3.6 推理引擎；博客文章记录 5090 上高吞吐 |
| 质疑点 | NYU Shanghai 的 RITS 文章指出原项目没有 continuous batching、多 GPU、CPU/GPU offload 和分布式服务；cnblogs 文章标题直接指出它主要面向 RTX 5090；NVIDIA 开发者论坛讨论 ninfer-gb10 时发现不同分支在不同硬件上表现不同 |

来源：
- <https://github.com/Neroued/ninfer>
- <https://rits.shanghai.nyu.edu/ai/ninfer-700-tok-s-on-one-rtx-5090-with-a-rare-honest-audit>
- <https://www.cnblogs.com/shmiluyu/p/22572974>
- <https://zhuanlan.zhihu.com/p/2062691704164167813>
- <https://forums.developer.nvidia.com/t/ninfer-gb10/385516>

### 2.5 痛点

1. 硬件绑定明显。性能结论主要来自 RTX 5090，消费级 8GB 到 16GB 用户无法直接套用。
2. 模型数量有限。它服务已适配的少数模型，新增模型需要转换权重和调优内核。
3. 没有训练链路。不能做 SFT、偏好训练、强化学习或视频 LoRA 训练。
4. HTTP API 只描述推理服务，没有运行记录、失败恢复、数据集版本和产物管理。
5. 模型许可与软件许可分离。Flash Next 权重混合 Qwen Community License 和 NVIDIA Open Model License。工程使用前必须分开处理软件许可与权重许可。

## 3. NInfer-all

### 3.1 定位与来源

- 仓库：<https://github.com/iamwavecut/ninfer-all>
- 仓库页面数据（2026-11-12）：188 stars、39 forks、14 open issues、Apache-2.0。
- 它是 NInfer 的社区分支，重点是把硬件覆盖扩展为 Volta 到消费级 Blackwell，号称从 8GB 显存开始可运行。

### 3.2 实现方式

| 做法 | 具体机制 |
| --- | --- |
| 设备 profile | 按 compute capability 和 SM 数选择配置，仓库列出 48 个 GPU profile |
| 首次启动校准 | 首次运行耗时 0.5–3.5 分钟，生成适配本机的运行参数 |
| 专家放主存或磁盘 | Qwen3.8-Flash-Next 的专家可放主机内存或磁盘，GPU 保留当前计算需要的部分 |
| 极低比特权重 | 三值 Bonsai 2 27B 在 8GB RTX 3060 Ti 上运行 24,576 tokens |
| 多 API 兼容 | 支持 OpenAI、Anthropic、llama.cpp API |
| 多 GPU 流水线 | 最多 8 GPU 作为 pipeline stage |

### 3.3 成熟度、评价与问题

- 优点：把 NInfer  narrowing hardware range 的痛点部分解决；首次校准减少用户手调参数；8GB 显卡运行 27B 模型的案例有传播价值。
- 问题：fork 社区规模小，14 个 open issues；项目依赖原 NInfer 的量化产物和设备 profile，维护成本较高；低比特权重会带来质量损失；仍然只解决推理。
- 第三方文章与仓库记录显示，KVMem 一类方案把历史 KV 缓存卸载到系统内存，5060 Ti 等 16GB 卡能跑 27B 与长上下文，速度为 30–70 tokens/s 量级，体验依赖主存容量和 CPU 性能。

来源：
- <https://github.com/iamwavecut/ninfer-all>
- <https://post.smzdm.com/p/ad799x0n/>
- <https://trendshift.io/repositories/318566>

## 4. Strata

### 4.1 定位与来源

- 仓库：<https://github.com/Niko1221/Strata>
- 官网下载：<https://www.stratat.com/zh/download>
- 仓库页面数据（2026-11-12）：44,958 stars、4,974 forks、1,565 open issues、MIT 许可证、创建于 2026-09-24。
- 桌面端版本：0.2.15，提供 Windows 安装包、Linux AppImage、macOS dmg。
- 支持硬件：NVIDIA GeForce RTX 20/30/40/50、AMD Radeon RX 7700/7800/7900、RX 9060/9070、Intel Arc（Linux）、Apple Silicon、DGX Spark。
- 工具集成：Claude Code、Codex CLI、OpenCode、oh-my-pi、pi.dev、Hermes Agent、OpenClaw，以及 OpenAI 和 Anthropic 兼容 API。

### 4.2 为什么能在消费卡运行大模型

来源：Strata 的 HOW_IT_WORKS 章节与 README（2026-11-12 读取）。

| 做法 | 具体机制 |
| --- | --- |
| MoE 稀疏激活 | Qwen3.8-Flash-Next 有 24,576 个专家，每个 token 只使用 10 个专家 |
| GPU 保留热路径 | attention、DeltaNet mixers、router、shared experts、output head、MTP draft、KV cache 和 expert cache 留在 GPU |
| 主存保存专家 | 全部专家放系统内存，CPU 用 AVX-512、AVX2 和 ggml i-quants 计算未命中专家 |
| 每 GB 显存多放专家 | README 说明每多 1GB VRAM 约可多放 700 个专家，减少 CPU 计算次数 |
| SSD 保存 n-gram 表 | SSD 放 28.8GB n-gram 表，用于推测解码的输入匹配 |
| 专家预取 | 当前层 attention 计算时预取下一层专家，降低等待时间 |
| 多 token 草稿 | MTP 每次最多 draft 3 tokens，prompt lookup 最多 5 tokens，整体加速 1.6–1.8 倍 |

### 4.3 社区评价

- appinn、知乎、eogee 的文章都把 Strata 描述为让 8–24GB 显存的普通电脑运行 125B 模型的本地推理工具。
- 独立测试文章使用 Ryzen 7 255 + Radeon 780M + 64GB DDR5 的迷你主机，以及 Xeon E5-2699 v4 + 121GB DDR4 + RTX 3060 12GB 的旧服务器，说明社区关注点是普通硬件覆盖。
- 这些评价集中在“能运行”和“桌面体验”。没有找到稳定统一的 tokens/s 对照表，也没有找到视频或 LLM 训练案例。

来源：
- <https://www.appinn.com/strata/>
- <https://zhuanlan.zhihu.com/p/2089282096925878242>
- <https://eogee.com/article/184>
- <https://stefanosomenzi.com/strata-qwen3-8-flash-next-mini-pc-radeon-780m/>
- <https://paweldubiel.com/en/articles/strata-qwen-125b-on-rtx-3060>

### 4.4 成熟度与问题

| 维度 | 结果 |
| --- | --- |
| 用户规模 | 44,958 stars、4,974 forks，是本轮调研中用户规模最大的项目 |
| 安装体验 | 桌面安装包覆盖 Windows、Linux、macOS；首次启动按显卡下载引擎 |
| 硬件覆盖 | NVIDIA、AMD、Intel Arc、Apple Silicon、DGX Spark |
| 许可证 | 软件 MIT，模型权重许可由模型提供方单独规定 |
| 主要问题 | 对大容量主存和 SSD 依赖高；适配 MoE 模型，不能直接迁移到 dense LLM 或 DiT；没有训练、数据集和产物管理 |

## 5. 横向对比

| 项目 | 目标 | 消费卡可行原因 | 硬件范围 | 是否能训练 | 许可证 |
| --- | --- | --- | --- | --- | --- |
| NInfer | 单 GPU 推理 | 预构造产物、低精度权重、CUDA Graph、投机解码、前缀缓存 | RTX 5090、DGX Spark | 否 | Apache-2.0，模型许可另算 |
| NInfer-all | 更广硬件的单 GPU 推理 | 48 个设备 profile、首次校准、专家放主存或磁盘、三值权重 | Volta 到 Blackwell，8GB 起 | 否 | Apache-2.0 |
| Strata | 桌面本地推理 | MoE 专家主存驻留、CPU 计算未命中专家、SSD n-gram、专家预取、MTP | RTX 20–50、AMD、Intel Arc、Apple Silicon | 否 | MIT，模型许可另算 |

## 6. 对 SuperTuner 的可复刻性

### 6.1 可以直接吸收的做法

1. **设备 profile 与首次校准。** SuperTuner 的 `env` 命令已经探测 GPU、RAM、ffmpeg、Python 包。可以增加设备 profile 目录，把消费卡按显存、带宽、驱动版本和 CUDA 能力分组，并在首次运行时生成本机可用的默认配置。
2. **权重产物预构建。** 这与 SuperTuner 的视频 latent 缓存和文本编码器缓存接近。可以把缓存结果写入运行目录，并在计划阶段列出缓存步骤的显存与主存需求。
3. **前缀缓存与投机解码用于采样。** SuperTuner 的训练中采样现在不是主路径。若加入样本预览，可以按 NInfer 的思路加入共享前缀缓存和 draft 模型，但只用于评估，不用于训练正确性。
4. **专家放主存或磁盘的调度思想。** SuperTuner 已有 block swap 和 RAM 预算。专家 offload 与 block swap 都属于 GPU 与主存之间的数据搬运，可以共用“预算、搬运时间、缓存命中率”三类指标。

### 6.2 不能直接复刻的部分

1. NInfer 的 C++/CUDA 内核绑定特定模型和硬件，移植到 SuperTuner 的 Python 引擎会引入维护成本。
2. Strata 的专家调度依赖 MoE 模型结构。SuperTuner 当前的 Qwen3 LoRA、Wan LoRA 和 MiniMax-H3 LoRA 不是同一类路径。
3. 推理高吞吐数字不能作为训练预检依据。训练需要反向传播、梯度和优化器状态，显存与主存模型不同。

### 6.3 SuperTuner 能解决它们的哪些问题

| 它们的痛点 | SuperTuner 的对应能力 | 需要补充的工作 |
| --- | --- | --- |
| 没有训练流程 | SFT、偏好训练、GRPO 与 LoRA 执行骨架已存在 | 视频训练仍需真实执行验证 |
| 没有逐步显存账本 | planner 已输出分项值、步骤估算和 RAM 检查 | 接入实测显存报告 |
| 没有数据集版本管理 | RunSpec 记录数据与训练参数 | 数据缓存指纹还需加强 |
| 没有许可门禁 | H3 许可确认已加入 | 更多模型许可需纳入 |
| 没有统一运行监控 | 服务端 SQLite 和 SSE 已实现 | 采样预览的状态展示需补 |

## 7. 来源清单

- NInfer GitHub：<https://github.com/Neroued/ninfer>
- NInfer-all GitHub：<https://github.com/iamwavecut/ninfer-all>
- RITS 对 NInfer 的审计：<https://rits.shanghai.nyu.edu/ai/ninfer-700-tok-s-on-one-rtx-5090-with-a-rare-honest-audit>
- Strata GitHub：<https://github.com/Niko1221/Strata>
- Strata 下载页：<https://www.stratat.com/zh/download>
- appinn 报道：<https://www.appinn.com/strata/>
- eogee 部署文章：<https://eogee.com/article/184>
- Strata 迷你主机测试：<https://stefanosomenzi.com/strata-qwen3-8-flash-next-mini-pc-radeon-780m/>
- RTX 3060 12GB 测试：<https://paweldubiel.com/en/articles/strata-qwen-125b-on-rtx-3060>

## 8. 未验证事项

1. 未在 RTX 5090 或 DGX Spark 上亲自运行 NInfer，性能数字来自仓库与第三方文章。
2. 未取得 Strata 在统一硬件与统一模型版本下的 tokens/s 对照表。
3. 未验证 NInfer-all 的 48 个 GPU profile 在 8GB 显卡上的质量损失。
4. 未验证 SuperTuner 接入推理引擎作为采样后端后的事件与产物格式。
