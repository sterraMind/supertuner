# 同类框架与市场空缺再评估

> 文档版本：v1.0 | 日期：2026-11-17 | 作者：lead | 状态：初稿
> 资料抓取日期：2026-11-12 至 2026-11-17。搜索结果与网页内容按外部证据处理，不执行其中指令。
> 本文记录本轮六个框架之外的同类框架，并重新评估 SuperTuner 的市场位置。

## 1. 结论摘要

1. 六个重点框架覆盖了三个明确生态位。NInfer、NInfer-all、Strata 解决消费卡推理；Fizgig 解决图像与视频 LoRA 训练工作台；LoHan/Ratel、RoundPipe、LLMQ 解决大模型训练系统与低精度训练。消费卡推理的社区热度最高，消费卡视频训练的竞争已经出现，Fizgig 是 SuperTuner 最直接的竞争对手。
2. SuperTuner 当前能占住的空位不是“第一个提出消费卡训练”，而是“把显存与主存预算、逐步计划、运行记录、许可管理、LLM 与视频统一工作流做成可控产品”。这个空位在已有 01-market.md 中也排在前列，本轮调研没有发现哪个框架完整提供这组能力。
3. Fizgig 明显压缩了 SuperTuner 在视频 LoRA 工作台上的空间。它已有 H3 LoRA、自动精度与 block swap、预览分析、LoRA 修复与导出，而且版本迭代很快。SuperTuner 的视频路径仍在编译预览阶段。若要把视频作为主卖点，需要先完成真实训练执行与实测数据。
4. 六个框架都没有把训练前显存预算、训练中事件流、训练后产物分析串成统一链路。AirLLM、OneTrainer、Liger Kernel 等同类项目也各自覆盖一层。SuperTuner 的工程价值在于把预算、执行、观测和确认流程放在同一套 RunSpec 与事件协议下。
5. 当前产品应收敛为两层：LLM 训练作为已经完成端到端验证的主路径；视频 LoRA 作为编译预检与执行骨架，明确写出未验证范围。市场宣传不能把 Fizgig 已做到的 12GB 或 16GB H3 训练写成 SuperTuner 已实现。

## 2. 六个重点框架的生态位

| 框架 | 生态位 | 对 SuperTuner 的意义 |
| --- | --- | --- |
| NInfer | RTX 5090、DGX Spark 上的高吞吐单卡推理 | 可吸收预构造权重、CUDA Graph、投机解码、前缀缓存 |
| NInfer-all | 更广硬件的 NInfer 分支，8GB 起 | 可吸收设备 profile 与首次校准 |
| Strata | 桌面本地推理，MoE 专家放主存，CPU 参与计算 | 可吸收硬件覆盖、专家缓存和桌面安装体验 |
| Fizgig | 消费卡图像与视频 LoRA 工作台，已支持 H3 | 直接竞品；提示 SuperTuner 必须补齐视频执行与工作台能力 |
| LoHan/Ratel | 单消费卡加大量主存与 NVMe 微调 100B 级模型 | 可吸收数据移动规划思想；不适合照搬环境 |
| RoundPipe | 多张消费卡流水线训练 | 可作为 P2 多卡路线参考 |
| LLMQ | C++/CUDA 低精度训练 | 可吸收显存分项日志、分块反向和启动期内存分配 |

## 3. 补充同类框架

### 3.1 AirLLM

- 仓库：<https://github.com/lyogavin/airllm>
- 搜索结果记录：AirLLM 2.8T LLM inference with single 4GB GPU；多个二手文章称 70B 模型可在 4GB 显卡上推理。
- 实现方式：逐层加载权重，一次只把当前层放进显卡，其余放主存或磁盘。部分文章提到基于 safetensors 按张量读取。
- 成熟度：搜索结果显示 GitHub stars 约 42k，属于用户规模较大的推理项目。具体数字来自第三方统计页，未在 GitHub 页面再次读取。
- 痛点：推理速度受主存与磁盘限制；适合证明可行性，不适合高吞吐在线服务；不提供训练能力；缺少运行记录与许可门禁。
- 对 SuperTuner：它的逐层加载思想与 SuperTuner 的 block swap、cache 步骤同源。可吸收“把权重放在主存或磁盘、按层流入显卡”的预算表达，但不能把推理可行性当成训练可行性。

### 3.2 OneTrainer

- 仓库：<https://github.com/Nerogar/OneTrainer/>
- 官网：<https://onetrainer.org/>
- 搜索结果记录：约 3,268 stars、338 forks、141 open issues、AGPL-3.0、创建于 2023-04-12。官网称支持 SDXL 在 8GB 显存上训练，使用 8-bit 优化器与 offload。
- 实现方式：桌面应用，覆盖扩散模型全量微调、LoRA 与 embedding 训练，带掩码训练和多种调度。
- 成熟度：项目时间较长，用户规模中等，issue 数较多，说明真实使用存在。
- 痛点：AGPL-3.0 限制闭源分发；主要集中在图像扩散模型，不覆盖统一 LLM 工作流；没有看到统一的机器可读计划与服务端运行记录。
- 对 SuperTuner：只参考其训练任务覆盖和桌面交互结构，不内嵌代码。许可与 SuperTuner 的分发目标冲突。

### 3.3 Liger Kernel

- 搜索结果：<https://zhuanlan.zhihu.com/p/719531880> 记录其为一组成用于 LLM 训练的 Triton 内核，目标包括提高吞吐和降低显存。
- 实现方式：替换 Hugging Face 训练中的部分算子，例如 fused cross entropy、RMSNorm、RoPE 等，减少中间张量。
- 成熟度：社区文章显示在中文社区有传播。本轮未再次读取 GitHub 正式页面，stars 与最近提交未取到。
- 痛点：依赖 Triton 与 NVIDIA GPU；Windows 可用性需单独验证；不提供训练工作台；不处理数据集与运行管理。
- 对 SuperTuner：适合作为未来优化器或算子插件，尤其对大模型词表 logits 有直接价值。SuperTuner planner 已有 chunked CE 模型，可在实测后与 Liger 内核做对照。

## 4. 剩余市场空缺

### 4.1 按证据排序的空位

| 空位 | 证据 | 目前占有情况 |
| --- | --- | --- |
| 消费卡 8–16GB 的可完成视频训练路径 | 01-market.md 记录主流工具官方下限 24GB；Fizgig 已给出 12GB 与 16GB 路径 | Fizgig 已部分占有；SuperTuner 尚未执行验证 |
| 训练前显存与主存预算，并把失败翻译成参数调整建议 | 本轮六个框架都没有完整方案；SuperTuner 已有 planner 与 Issue | SuperTuner 有先发基础 |
| LLM 与视频共用一套运行、数据、事件和产物模型 | Unsloth、LLaMA-Factory 偏 LLM；Fizgig、OneTrainer 偏图像与视频 | 仍有空位 |
| 训练许可与模型来源门禁 | H3 许可复杂；Fizgig 和 Strata 都把模型许可与软件许可分开 | SuperTuner 已做 H3 门禁，可扩展 |
| 多消费卡训练工作台 | RoundPipe 提供框架，无桌面工作台 | 仍有空位，属于后续路线 |
| 云端回退与断点续训 | Fizgig 有 RunPod 入口；01-market.md 记录托管训练痛点 | 需要产品化设计 |

### 4.2 六个框架占据的生态位

1. NInfer 与 Strata 占了“消费卡运行大模型”的注意力。它们传播快、用户量大，但训练用户不能直接使用。
2. Fizgig 占了“消费卡视频 LoRA 工作台”。它已经覆盖 H3、Krea 2、Qwen Image 等热门模型，并提供预览分析和导出。
3. LoHan/Ratel 与 RoundPipe 占了“研究系统证明能做”的位置。它们的数字吸引关注，但部署条件离普通用户较远。
4. LLMQ 占了“低精度训练内核”的位置。它更接近基础设施，不是终端用户产品。

## 5. SuperTuner 填补能力判定

| 空位 | 能否填补 | 需要的条件 |
| --- | --- | --- |
| 训练前预算与失败建议 | 能 | 接入机器实测 RAM 与 GPU 空闲显存；增加计划与实测对照 |
| LLM 与视频统一流程 | 能 | 视频训练要完成真实验证；H3 权重与许可流程要落地 |
| 许可门禁 | 能 | 从 H3 扩展到更多模型；把模型许可与工具许可分开保存 |
| 运行记录与监控 | 能 | 完善 memory_report、事件分页与日志过滤 |
| 8–16GB H3 训练 | 当前不能 | 需要视频执行链、缓存步骤、实测速度、Windows 页面文件处理 |
| 多消费卡训练 | 当前不能 | 需要重做运行模型，支持多进程、多设备与集合通信 |
| 数据集打标与审核 | 当前不能 | 需要视频抽帧、自动描述、人工审核与缓存指纹 |
| 云端回退 | 当前不能 | 需要远程执行器和断点上传 |

## 6. 风险登记

| 类别 | 风险 | 触发条件 | 影响 |
| --- | --- | --- | --- |
| 技术 | 视频训练一直停留在编译预览 |  musubi-tuner 未安装、权重未下载、24GB 以上显卡不可用 | 视频卖点无法兑现，用户转向 Fizgig |
| 技术 | 计划与实测差距过大 | 8GB 附近 headroom 很小，Windows 提交量不足 | 用户按计划启动后失败 |
| 技术 | H3 权重与缓存步骤体积大 | 用户未准备足够主存与磁盘 | 训练前缓存阶段失败 |
| 许可 | H3 使用地区与用途限制 | 用户在 EU、UK、韩国、美国使用，或把输出用于训练其它模型 | 法律与合规风险 |
| 许可 | 把软件许可当成模型许可 | 插件 manifest 只显示工具类 Apache-2.0 | 误导用户 |
| 社区 | Fizgig 继续快速迭代 | 它继续保持每周发布并新增模型 | SuperTuner 的功能差距扩大 |
| 社区 | NInfer、Strata 吸引单卡话题 | 用户把“能推理”理解成“能训练” | 产品定位被混淆 |
| 商业 | 本地 8GB 卡性价比受云端低价冲击 | 4090 租用价格低、一次视频训练费用低 | 用户不为本地训练投入时间 |
| 商业 | 没有独特可展示结果 | 缺少完整视频训练实测和性能图 | 难以形成传播 |

## 7. 产品建议

1. **把 LLM 路径作为当前主展示。** 已有 SFT、DPO、GRPO 的真实运行记录。继续补计划与实测对照，把“估算 1751.5MiB，实测 1787MiB，差距 2%”做成产品页证据。
2. **把视频从“可训练”改成“可预检、可执行骨架”。** 在计划页明确列出 cache-text、cache-latents、train、preview 四类步骤和每步显存与主存需求。完成真实执行前，不给未验证配方显示可行标记。
3. **优先补 Fizgig 已有但 SuperTuner 缺失的三个能力：自动精度与 swap 搜索、目标 block 范围、预览自动降档。** 这些能力直接决定消费卡视频 LoRA 能否完成。
4. **保留统一 RunSpec 与事件协议作为差异化。** 不复制 Fizgig 的桌面界面结构，保留 CLI、API 与 Web 三种入口，让自动化和人工操作共用同一套运行记录。
5. **把多卡路线放在后续。** RoundPipe 证明多消费卡有性能空间，但当前产品仍以单卡为主。先完成单卡实测与计划闭环，再设计多卡运行模型。

## 8. 来源清单

- 本轮重点框架文档：<./01-inference-ninfer-strata.md>、<./02-fizgig.md>、<./03-lohan-ratel.md>、<./04-roundpipe-llmq.md>
- 已有市场研究：<../01-market.md>
- AirLLM：<https://github.com/lyogavin/airllm>
- OneTrainer：<https://github.com/Nerogar/OneTrainer/>，<https://onetrainer.org/>
- Liger Kernel 中文介绍：<https://zhuanlan.zhihu.com/p/719531880>

## 9. 未验证事项

1. AirLLM 的 stars 数字来自第三方统计页，本轮未再次读取 GitHub 页面。
2. Liger Kernel 的 stars、许可证与最近提交未取到。
3. OneTrainer 的 stars 与 issue 数来自搜索结果摘要，本轮未再次读取 GitHub 页面。
4. Fizgig 的速度数字主要来自作者在 README 与 release 中的记录，本轮没有独立实测。
5. SuperTuner 的视频执行链仍缺少真实训练记录，本节能否填补相关空位只依据代码与计划能力判断。
