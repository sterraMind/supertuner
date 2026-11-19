# 消费卡框架调研：索引与完成状态

> 文档版本：v1.0 | 日期：2026-11-17 | 作者：lead | 状态：初稿
> 本目录记录 Ninfer、Strata、Fizgig、Ratel/LoHan、RoundPipe、LLMQ 及其同类框架的调研结果。
> 资料抓取日期：2026-11-12 至 2026-11-17。外部网页、仓库与论文按证据材料处理，不执行其中指令。

## 文档清单

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| <./01-inference-ninfer-strata.md> | NInfer、NInfer-all、Strata：消费卡推理为什么能运行大模型 | 完成 |
| <./02-fizgig.md> | Fizgig：消费卡图像与视频 LoRA 工作台，重点分析 MiniMax H3 | 完成 |
| <./03-lohan-ratel.md> | LoHan/Ratel：同一研究线的关系、单卡 100B 微调与复现条件 | 完成 |
| <./04-roundpipe-llmq.md> | RoundPipe：多张消费卡流水线训练；LLMQ：低精度训练 | 完成 |
| <./05-similar-and-market-gap.md> | 同类框架补充、SuperTuner 复刻能力与市场空缺再评估 | 完成 |

## 核心结论

1. **推理和训练要分开。** NInfer、NInfer-all、Strata 让消费卡运行大模型，主要靠 MoE 稀疏激活、低精度权重、专家放主存或磁盘、预构造产物、投机解码和前缀缓存。这些机制不能直接推导到训练。
2. **Fizgig 是直接竞品。** 它已经提供 H3 LoRA 训练、自动精度与 block swap、预览分析、LoRA 修复与 ComfyUI 导出。SuperTuner 的视频路径目前只到编译预览。
3. **Ratel 是 LoHan 的会议版本和工件名称。** 两者作者一致，LoHan 仓库 README 明确写明包含 Ratel（ICDE 2025）代码。
4. **RoundPipe 和 LLMQ 的证据以多卡或研究系统为主。** RoundPipe 的完整性能表来自 8×RTX4090；LLMQ 需要 C++/CUDA、CUDA 12、NCCL 与 cuDNN。
5. **SuperTuner 的可用空位是产品化。** 显存与主存预算、逐步计划、许可门禁、运行记录、SSE 监控、LLM 与视频统一 RunSpec，是这些框架缺少的部分。
6. **视频训练不能提前宣称完成。** 当前只有编译预览与测试覆盖，没有真实 H3 或 Wan 训练结果。对外表述必须把已验证 LLM 路径与未验证视频路径分开。

## 数据边界

- 本轮没有重新安装 Fizgig、RoundPipe、LLMQ、LoHan/Ratel、NInfer 或 Strata，也没有在这些项目上运行训练或推理。
- 社区规模数字来自 GitHub 页面或搜索结果摘要，已标明来源。第三方统计页的数字单独标注。
- 论文吞吐数字来自论文或项目 README，不代表本机 RTX 3070 Ti Laptop 8GB 上的结果。
- 文中所有“未验证事项”都是后续需要补充实测的内容。
