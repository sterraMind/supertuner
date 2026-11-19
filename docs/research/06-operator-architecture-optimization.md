# 算子优化与模型架构优化调研（面向消费卡训练 H3/Wan）

> 日期：2026-11-18 | 作者：lead
> 来源：arXiv 原文（走代理抓取）+ GitHub 原文，全部按不可信外部数据处理，仅引证。
> 关键分类：**训练兼容（有 backward）** vs **仅推理（不能用于训练）**——这是决定能否用于 SuperTuner 的分水岭。

## 0. 结论先行

1. **算子优化里，绝大多数 H3 专用 kernel 是推理侧**（ComfyUI 去噪），不能直接用于训练；真正带 forward+backward 的只有 SageAttention3 的 8-bit attention、Liger Kernel、AdaptiveLoad 的融合算子这几类。
2. **模型架构优化（剪枝/量化/block swap/梯度检查点）才是消费卡能训 H3 的真正原因**，且已在社区验证（12GB 跑 H3 LoRA）；这部分是 SuperTuner 规划器需要精确建模的对象。
3. 对 SuperTuner 而言："接入第三方 kernel"符合现有「不重写内核」定位；"自研融合算子"要改架构 ADR，成本高、风险大，除非拿到实测收益证据，否则不建议。

## 1. 算子优化（Operator / Kernel）

### 1.1 训练兼容（有反向传播，可用于训练）

| 方案 | 机制 | 收益（原文数字） | 训练兼容 | 来源 |
| --- | --- | --- | --- | --- |
| **SageAttention3 8-bit attention** | 8-bit 量化 attention，**forward + backward 都实现** | fine-tuning **无损**；pretraining 收敛变慢；FP4 版在 RTX5090 达 1038 TOPS（比该卡最快 FlashAttention 快 5×，但 FP4 是推理） | ✅ 8bit 版；⚠️ 微调可用、预训练慎用 | arXiv:2505.11594 v3 (2026-01-14) |
| **Liger Kernel**（LinkedIn，Triton） | 融合 RMSNorm/LayerNorm/RoPE/SwiGLU/GeGLU/CrossEntropy/**FusedLinearCrossEntropy**；输入分块 | LLM 训练吞吐 **+20%**、显存 **−60%**（对比 HF 实现） | ✅ 专为训练 | arXiv:2410.10989；Spheron 解析 |
| **Liger for WanVideo**（kijai PR #862） | 同上算子用于 Wan 视频 LoRA | 吞吐 +20%、显存 −60%（项目页口径；高分辨率/高帧数关键） | ✅ 训练 | ComfyUI-WanVideoWrapper PR #862 |
| **AdaptiveLoad**（JDT/Tsinghua 等） | ① 双约束动态负载均衡（考虑 self-attention 二次复杂度）② **融合 LayerNorm-Modulate CUDA kernel** ③ 激活计算图简化与显存回收 | 面向 DiT/MMDiT 视频训练，提升吞吐与显存效率（论文未在摘要给单一倍数） | ✅ 视频 DiT 训练 | arXiv:2605.17923 (2026-05-18) |
| **FlashAttention-3 / SDPA** | IO 感知 attention，FP8 块量化 | 1.5–2× 更快；FA3 是 SDPA 后端之一 | ✅ | arXiv:2407.08608 |
| **layerwise upcasting**（finetrainers windows） | 前向/反向按层把权重转 `float8_e4m3fn/e5m2` | 权重显存约减半 | ✅ 训练（LoRA） | finetrainers-for-windows docs |

### 1.2 仅推理（不能用于训练，别误当训练方案）

| 方案 |  mechanism | 收益 | 为什么不能用于训练 |
| --- | --- | --- | --- |
| **CAKE fused kernels**（FlashInfer/SGLang） | 4 段融合：RMSNorm→indexed AdaLN scale/shift→BF16 QKV GEMM→Q/K RMSNorm+3D RoPE；把 DiT 原生张量直接喂给融合段 | 多数 shape 超过 SOTA baseline | 服务 **diffusion 推理 route**（sglang `minimax_h3_diffusion`）；无 backward | flashinfer PR #5583/#5629、sglang PR #42531 |
| **H3 Fused MLP**（ByronLeeeee 套件） | NVFP4 Fused MLP；AdaLN/gated-residual 整段打包 kernel | denoise 208.8→194.1s（**−7.03%**）、peak **−712 MiB** | ComfyUI 推理；latent hash 一致（仅推理验证） | H3 Optimization Suite EVALUATION_REPORT (2026-08-12) |
| **Low-Memory Sage2 / Memory Efficient Sage Attention** | 把 H3 self-attention 换成 int8 q/k + fp32 累加的 Sage kernel，缩小 attention 工作集 | peak 4410→3844 MiB（**−566 MiB**），代价 denoise +2.13% | 推理节点（ComfyUI KJNodes）；训练需 backward | 同上 + comfy.icu node |
| **CAB / res_multistep / few-step LoRA** | 少步数（12/10 步） | denoise −39.79%/−49.79%，SSIM 0.815/0.806 | 推理省步数，不改变训练 | 同上 |
| **Long-Sequence 16gb_chunked** | FC1 临时张量分块 | FC1 上界 5488.6→224.0 MiB/chunk，16GB 卡跑完 15s 视频 | 推理 | 同上 |
| **Sol-H3 全栈**（NVIDIA） | AdaLN precompute、prompt caching、KV caching、**precision-preserving LoRA fusion**、kernel fusion | 8×GB200 上 3.95× 端到端 | 明确是**推理引擎**（Sol-Engine） | arXiv:2609.35110；NVIDIA Sol 页面 |
| **SpargeAttention / Sparge** | 稀疏 attention | 推理加速 | Wan2.2-I2V-5B 开启会出噪声，且是推理 | DCAI blog |

### 1.3 Windows 现实（此前判断需要修正）

- 之前调研写「flash-attn/SageAttention/xformers 在 Windows 默认不可用，基线 SDPA」——**过于保守**。
- 反证：H3 优化套件在 **Windows + RTX 5070 Ti 16GB + PyTorch 2.10.0+cu130 + Triton 3.6.0** 上完整跑通 fused MLP / Sage2 / chunked（EVALUATION_REPORT §2）。
- 结论：Triton 系算子在 Windows 可用；但成功率依赖 PyTorch/CUDA/Triton 版本对齐，**需要在 SuperTuner 里做 capability probe 而非硬编码禁用**。

## 2. 模型架构优化（结构层省显存/加速）

| 方案 | 机制 | 收益 | 训练/规划意义 | 来源 |
| --- | --- | --- | --- | --- |
| **AdaLN 分支剪枝（pruned DiT）** | 裁掉约 13B AdaLN 分支，33B→20B | BF16 66→**40GB**；INT8 约 **21GB** | musubi 的 `*_pruned_*` / `--prune_adaln`，消费卡通路核心 | musubi docs/minimax_h3.md |
| **int8 / NVFP4 权重** | DiT 权重 INT8、文本编码器 NVFP4+AWQ | DiT ~21GB；Qwen3-VL-32B NVFP4 ~15.7GB | 规划器已有 INT8=1.05 B/param；NVFP4 需新增量化档 | musubi docs；Fizgig H3 文档；H3 suite |
| **单向 block swap（H2D-only）** | 只把 block 从 CPU 搬到 GPU，不来回搬 | Fizgig：比往返快约 **6.4×** | 规划器当前按往返估算，需区分单向/往返（adoption AP-08） | musubi deepwiki；Fizgig 文档 |
| **梯度检查点 / selective recomputation** | 重算换激活显存 | LLMQ 用 selective recomputation 降显存 | 已强制用于 block swap；规划器已建模 | LLMQ (IST Austria, CPAL 2025) |
| **TREAD token routing** | 只训部分 token | 减激活显存 | Fizgig H3 用；SuperTuner 未接入 | Fizgig 文档 |
| **MOE 专家放 RAM/SSD**（Strata 思路） | 每 token 只用 10/24576 专家 | 消费卡跑超大 MoE | 推理侧；超大模型训练不适用 | Strata 调研 |
| **VAE 时间/空间压缩** | f16t4d24（空间16×、时间4×、24 通道） | 降低 latent token 数 | 规划器已用 stride 计算 token | 既有调研 |

## 3. Fizgig 专项（最直接的竞品，第一方实现）

Fizgig 不是「musubi 的 GUI 包装」——它对 H3 自带**纯 PyTorch 原生实现**（`MiniMaxH3DiT` 是 ComfyUI H3 DiT 的 PyTorch 移植），有自己的 VAE 与采样，算子/架构优化是一手的：

| 优化 | 机制 | 出处 |
| --- | --- | --- |
| **ConvRot INT8** | 权重存旋转 int8 基（ConvRot）降 outlier、提升 scale 效率；**forward+backward 都 int8**，稳态步 ~2× 更快（一次性 warm-up ~90s），更损一点 | `src/fizgig/minimax/convrot.py`；CLI.md |
| **Async H2D-only block swap** | 单向 CPU→GPU 流式换块；2.7× 提速（RTX 5060 Ti 16GB、blocks swap 19、无采样） | issue #73（已合并） |
| **双向 / H2D-only 流式 offload** | 冻结基座权重的流式 ring（bidirectional + H2D-only） | `src/fizgig/modules/offloading.py`、`minimax/h3_nf4_h2d_offload.py` |
| **H3 NF4 常驻基座 + 换块** | `H3NF4Rotator`：bitsandbytes `Linear4bit` ↔ `nn.Linear` 激活切换 | Full Fine-Tune 子系统 |
| **全量微调（不只 LoRA）** | 窗口化部分旋转 + 流式（窗口装不下就 streaming）；Adafactor 默认，也可 AdamW8bit/AdamW/AdEMAMix8bit/Lion8bit | `src/fizgig/families/ft.py` |
| **Torch.compile** | transformer 块 auto/on 编译（边界 checkpoint 外置换显存） | CLI.md |
| **TREAD token routing** | 只训部分 token | 前次调研 |
| **16GB = H3 最低卡** | LoRA / LoKR / Slider / 全微调（实验） | README |

**对 SuperTuner 的含义**
1. Fizgig 证明「消费卡训 H3」的执行路径（ConvRot int8、async H2D 换块、compile、offload ring、全微调窗口化）是真实可跑、且已在 16GB 卡验证。
2. 它把「执行」做成了第一方 PyTorch 实现；我们走 musubi 委托（musubi 自己也有 block swap + int8）。所以**执行层不必自研**，musubi 已覆盖等价能力。
3. 我们的差异仍在「跑前可校验 + 跑后校准 + 许可门禁」，不靠重造内核取胜。
4. 需向 musubi 对齐的指标：H2D-only 单向换块、int8 backward（~2×）、窗口化部分训练（train_blocks 已有对应）、compile。这些是规划器要建模的 cost/throughput 因子，不是我们要重写的东西。

## 4. 对 SuperTuner 的落地判断

**可接入（不改「不重写内核」定位，优先做）**
- 在 backend 的 capability probe 里探测 SageAttention/Liger/Triton 是否可用，记录到 `EnvReport`，规划器据此给出「启用后预计显存/吞吐」提示，但**训练仍走稳妥的 SDPA**，不默认替换内核。
- 把单向 block swap 的收益（6.4×）写进规划器的搬运时间估算（AP-08）。
- 规划器新增 NVFP4 量化档（当前只有 INT8/NF4/FP8）。

**需改架构、暂不做（除非有实测收益证据）**
- 自研或内嵌 CAKE/Sol 风格的融合算子（要 backward，工作量与正确性风险大）。
- 把 SageAttention3 8-bit attention 接进训练 backward——技术可行（论文已实现），但要先在 8GB 卡上实测收敛是否无损，否则不碰。

**核心认知**：SuperTuner 的竞争力在「把上述架构优化精确建模 + 跑前判定 + 跑后校准」，而不是自己去写 Triton kernel。算子优化是第三方提供、我们探测并接入；架构优化是规划器精确计算的对象。这与「训练器为主体、规划器为信任层」的定位一致。
