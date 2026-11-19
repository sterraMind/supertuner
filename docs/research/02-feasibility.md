# 02 单卡可行性评估：视频 DiT 与 LLM 的训练、微调、蒸馏

> 调研日期 2026-11-11。机器基线：本地 RTX 3070 Ti Laptop 8GB（GA104，Ampere，8GB GDDR6），另覆盖 3060 12GB / 4070 Ti Super 16GB / 4090 24GB / 5090 32GB。每条数字后的 `[S#]` 对应第 9 节来源表（URL 与取数日期）。显存单位按 GB=GiB。

## 0 结论

1. **8GB 的分界线在"文本 vs 视频"**：LLM 侧 7B/8B 的 QLoRA(4bit) 可行（LLaMA-Factory 表估 6GB），14B QLoRA 需 12GB 不可行；视频侧 Wan2.2 5B 在 480×832×49 需要约 7.4–8.9GB（fp8）到 12.5–13.9GB（bf16），8GB 只能降到 256–360p / ≤21–33 帧并大量换块，或改训 Wan2.1 1.3B。
2. **5B 是单卡视频训练的现实入口**，不是 14B：Wan2.2 TI2V-5B 是 5B dense，官方权重文件 18.63GB（fp32 存盘）、bf16 载入约 10.0GB。14B 与 A14B(27B MoE) 在 24GB 上必须 `blocks_to_swap` 20–30 且分辨率 ≤480p。
3. **MiniMax-H3 真实存在**，但它是 33B dense 音视频模型 + 32B 文本编码器 + 带地域限制的社区许可；单卡只能做**推理**（官方 DMAD 学生版 8GB 可出 5s、12GB 可出 15s），训练侧没有任何 ≤32GB 的公开成功案例。
4. **蒸馏（DMD/DMAD）不是单卡任务**：DMAD 官方在 8×H200 上每 100 iter 约 6.7 小时、800–1600 iter、约 300GB 训练数据；单卡只能消费已发布的 1.4GB 学生 LoRA 做 4 步推理。
5. **性价比拐点在 16GB**：16GB（4070 Ti Super）能跑 5B 的 480p LoRA（社区实测 10 s/it 量级），24GB 关掉换块后约 3–5 s/it；租 4090（$0.89/h）做一次视频 LoRA 约 $1–5，比本地 8GB 跑几天更划算。

## 1 显卡基线与第二约束

| 卡 | 显存 | 架构 | 原生 fp8 matmul | 备注 |
|---|---|---|---|---|
| RTX 3070 Ti Laptop | 8GB | GA104 / Ampere | 否（fp8 只省显存，不加速） | 本地实测卡；移动版持续功耗受限，长跑降频 |
| RTX 3060 | 12GB | GA106 / Ampere | 否 | 带宽 360GB/s，同设置比 4070 Ti S 慢约 3–4×（推算） |
| RTX 4070 Ti Super | 16GB | AD103 | 是 | 社区最常引用的"能训 Wan2.2"档 |
| RTX 4090 | 24GB | AD102 | 是 | Wan2.2 5B 官方 720P 推理参考卡 |
| RTX 5090 | 32GB | GB202 | 是 | 32GB 可让 A14B 双专家同时驻留 |

- 显存之外的第二约束是**主存**：musubi 建议 64GB+，32GB+swap 或许可以；Wan2.2 双模型 fp8 合计约 42GB 共享显存、需约 96GB 主存 [S1]。8GB 卡靠 block swap 时主存不足会直接卡在换页。
- Windows 上 block swap 走共享显存，与桌面/浏览器抢同一块内存 [S1]；diffusion-pipe 官方则说原生 Windows 基本不可行，建议 WSL2 [S16]。

## 2 视频模型核查

### 2.1 核查表

| 模型 | 参数 | 架构要点 | 官方仓库 | 许可证 | 单卡训练器 | 可行性（2026-11） |
|---|---|---|---|---|---|---|
| Wan2.1 T2V | 1.3B / 14B | DiT + Wan2.1 VAE（4×8×8） | Wan-Video/Wan2.1 [S31] | Apache-2.0 | musubi `--task t2v-1.3B / t2v-14B` | 1.3B 在 8GB 的低分辨率可训；14B 需 16–24GB+换块 |
| Wan2.2 TI2V-5B | 5B dense | 30 层、dim 3072、ffn 14336、24 头×128、patch(1,2,2)、48 通道 latent；Wan2.2-VAE 4×16×16（总压缩 4×32×32）[S9] | Wan-AI/Wan2.2-TI2V-5B [S7] | Apache-2.0 | DiffSynth-Studio、diffusion-pipe（仅 t2v/t2i）、Unsloth Studio | 12GB 起（480p），16GB 舒适 |
| Wan2.2 T2V/I2V-A14B | 27B MoE（每步激活 14B） | 高噪/低噪双专家 | Wan-AI/Wan2.2-T2V-A14B | Apache-2.0 | musubi `t2v-A14B` / `i2v-A14B` | 24GB + 换块 20–30，480p [S18] |
| HunyuanVideo | 13B | 双流 MMDiT + 3D VAE | Tencent/HunyuanVideo | 腾讯社区许可 | musubi | 不指定 `--fp8_base` 建议 24GB+ [S4] |
| CogVideoX | 2B / 5B | 3D 因果 VAE + DiT | zai-org/CogVideo | Apache-2.0（代码） | 官方 finetune（DDP/ZeRO） | 2B LoRA 16GB、5B LoRA 24GB（官方表）[S14] |
| LTX-Video | 2B / 13B | 高压缩 VAE + DiT，蒸馏版 | Lightricks/LTX-Video | LTX 许可（开源权重） | LTX-Video-Trainer | 官方未给显存表；2B 蒸馏版 8GB 可推理 [S15] |
| MiniMax-H3 | 33B（含约 13B AdaLN） | 单流 Omni-Transformer，音视频联合 | MiniMax-AI/MiniMax-H3 [S10] | H3 Community License（地域限制）[S11] | musubi `docs/minimax_h3.md`、diffusion-pipe、LightX2V/DMAD | 单卡不可训 |

### 2.2 MiniMax-H3 到底是什么（核实结论）

最容易被传错的一点先说：**H3 不是虚构模型**。GitHub `MiniMax-AI/MiniMax-H3` 创建于 2026-07-30，HF `MiniMaxAI/MiniMax-H3` 与 ModelScope 同步，仓库 11.7k stars（2026-11-11 抓取）[S10]。

- **参数与结构**：H3-Base 是 **33B dense 单流 Omni-Transformer**，其中约 13B 位于 AdaLN 分支（只依赖采样步，可预计算缓存，推理时不加载）；文本条件直接用 **Qwen3-VL-32B 全量权重取第 50 层隐状态**（H3-Encoder）；视频 VAE 为 f16t4d24（时空 16×、时间 4×、24 latent 通道），再 patchify 1×2×2，等效 32× 空间下采样；音频 VAE 为 32kHz 立体声、40Hz latent；MM-RoPE 三维位置编码；BF16；CFG 蒸馏（推理不需要 CFG）。
- **能力**：T2VA、FL2VA、Ref2VA（≤9 图 / ≤3 视频段 / ≤3 音频段）；输出 4–15s、24fps、默认短边 768；2K 依赖 `H3-Regenerate-2K`（**未开源**），`H3-Context-IR` 也**未开源** [S10]。
- **权重体积**：transformer 66GB BF16 → 裁剪 40GB BF16；INT8 34GB → 裁剪 21GB；文本编码器 48GB BF16 / 25GB INT8 / 15GB NVFP4+AWQ；`--blocks_to_swap` 最多 48/50 块 [S5]。
- **许可证要点**（License date 2026-08-02，逐条读过 LICENSE 全文）[S11]：
  - Applicable Territory = 全球**排除 EU、UK、韩国、美国**，这些地区需另行申请。
  - 年收入 >2000 万美元的商用产品需 MiniMax 书面授权，并在 UI 显著标注 "MiniMax H3"。
  - **禁止**用 H3 或其输出改进任何其它 AI 模型（含蒸馏、合成数据训练）；输出本身不算 Model Derivative。
  - 对外发布衍生模型需随附许可证与 NOTICE。
- **训练工具链**：musubi `docs/minimax_h3.md` 支持 T2VA/FL2VA/Ref2VA 的 LoRA（含 `--block_swap_h2d_only`，冻结底模专用，要求 gradient_checkpointing）[S5]；diffusion-pipe 自 2026-08-06 支持 H3（仅 T2I 与 T2VA），2026-08-08 加 CFG-augmented training [S16]；DMAD 提供基于 LightX2V 的蒸馏训练代码 [S12]。
- **单卡判定**：只有推理可行。DMAD 4 步学生 LoRA（rank 128，1.4GB）在 1344×768、124 帧下采样峰值 **5.9GiB**、文本编码 3.4GiB、解码 3.3GiB，官方明说 8GB 卡可出 5s、12GB 可出 15s（`--low-vram`）[S12]。

### 2.3 musubi-tuner 与 diffusion-pipe 的官方口径

- musubi README「Hardware Requirements」：**12GB+ 用于图像训练、24GB+ 用于视频训练**；12GB 需分辨率 ≤960×544 并开 `--blocks_to_swap` / `--fp8_llm`；主存建议 64GB+ [S1]。
- `docs/wan.md` 的 `--task` 只有 `t2v-1.3B / t2v-14B / i2v-14B / t2i-14B / *-FC` 与 Wan2.2 的 `t2v-A14B / i2v-A14B`，**没有 TI2V-5B**；5B 只能用 DiffSynth-Studio 或 diffusion-pipe [S2]。
- `docs/hunyuan_video.md`：不指定 `--fp8_base` 时建议 24GB+ 显存；<16GB 用 `--fp8_llm` 跑文本编码器 [S4]。
- `docs/block_swap.md`：Qwen-Image LoRA 在 RTX 3090 换 40 块，经典 swap 约 14s/sample，H2D-only 约 11s/sample（约 25% 提速）[S3]。
- diffusion-pipe README「Reducing VRAM requirements」给出四项：`AdamW8bitKahan`（lr 5e-5、betas [0.9,0.99]）、`blocks_to_swap = 32`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`activation_checkpointing = 'unsloth'`；并强调训练前把 latent 与文本嵌入缓存到磁盘，训练期不加载 VAE 与文本编码器。`examples/wan_14b_min_vram.toml` 注明 Wan 14B t2v 512×512×81 **可在 24GB 训练**（`transformer_dtype = float8` + LoRA rank 32）[S16]。
- 低显存优化器：musubi 侧常用 `adamw8bit` 或 `came_pytorch.CAME`（CAME 自带 8bit 状态，社区反馈更稳）[S17][S18]；diffusion-pipe 侧是 `AdamW8bitKahan` [S16]。

## 3 显存拆解与可行性矩阵

### 3.1 公式（可直接进 planner）

```
VRAM_peak ≈ W_base + W_lora + S_lora + A_act + A_ws + C_ctx + frag
W_base  = P × bytes(dtype)                       # bf16=2B, fp8=1B（scale 另计 ~1-2%）
W_lora  = P_lora × 2B
S_lora  = P_lora × (4B 主权重 + 4B 梯度 + 2×1B 优化器状态)   # AdamW8bit / CAME
A_act   ≈ L × N × d × 2B                         # gradient checkpointing 只存每层输入
A_ws    ≈ 2×N×d_ff×2B + 3×N×d×2B                 # 单块重算的 FFN/注意力工作区
C_ctx   ≈ 0.6–1.2GB                              # CUDA 上下文 + 分配器碎片
N = ((F-1)/4+1) × H/32 × W/32                    # Wan2.2 5B token 数（VAE 4×16×16 + patch 2×2）
```

- Wan2.2 5B 常量：`L=30`、`d=3072`、`d_ff=14336`、latent 48 通道 [S9]；权重文件 18.63GB（fp32 存盘），bf16 载入约 10.0GB [S8]。
- LoRA r32 打在 q,k,v,o,ffn.0,ffn.2 上约 **57.0M 参数 → 114MB**；梯度+优化器约 0.45GB。
- 文本编码器 UMT5-XXL bf16 约 10.58GB、Wan2.2-VAE 2.63GB [S8]；训练前用 `cache_text_encoder_outputs.py` 与 `cache_latents.py` 预缓存，训练期常驻 **0**——这是 8–16GB 能跑的前提。

### 3.2 分项（Wan2.2 TI2V-5B LoRA）

| 项 | 480×832×49（N=5070） | 1280×704×121（N=27280） | 依据 |
|---|---|---|---|
| DiT 权重 bf16 | 10.0GB | 10.0GB | 5B×2B，文件 18.63GB [S8] |
| DiT 权重 fp8 | 5.0GB | 5.0GB | 同上 |
| LoRA r32 权重 | 0.11GB | 0.11GB | 57.0M×2B |
| LoRA 梯度+优化器 | 0.45GB | 0.45GB | AdamW8bit/CAME |
| 激活（GC） | 0.93GB | 5.03GB | 30×N×3072×2B |
| 重算工作区 | 0.6–1.2GB | 2.5–4.0GB | split_attn 后 |
| 上下文+碎片 | 0.7–1.2GB | 0.7–1.2GB | 经验值 |
| **合计 bf16** | **12.5–13.9GB** | **18.8–20.8GB** | |
| **合计 fp8** | **7.4–8.9GB** | **13.7–15.7GB** | |

### 3.3 Wan2.2 5B LoRA 可行性矩阵（rank 32、batch 1、GC 开、文本/VAE 预缓存）

| 配置（F 为帧数） | 8GB 3070 Ti | 12GB 3060 | 16GB 4070 Ti S | 24GB 4090 | 32GB 5090 |
|---|---|---|---|---|---|
| 480×832×33（N=3510） | ⚠️ 需 256–360p 或 fp8+换块 20+ | ✓ fp8 无换块（bf16 需换块 6–10） | ✓ bf16 无换块 | ✓ 无换块 | ✓ 无换块 |
| 480×832×49（N=5070） | ✗（fp8 也差 0.4–0.9GB，且不含桌面占用）→ 降到 256×256×33 | ✓ fp8+换块 8–12 | ✓ bf16+换块 5–10 | ✓ 无换块 | ✓ 无换块 |
| 720p 1280×704×81（N=18480） | ✗ | ✗ | ⚠️ fp8+换块 15–20 | ✓ bf16+换块 0–6 | ✓ 无换块 |
| 720p 1280×704×121（N=27280） | ✗ | ✗ | ✗（fp8 需 13.7–15.7GB） | ⚠️ bf16 余量 3–5GB | ✓ 无换块 |

符号：✓ 可跑完；⚠️ 可跑但慢/接近上限；✗ 不成立。所需档位按 3.1 公式推导，实测锚点见 3.4。

### 3.4 速度锚点与推算

| 数据点 | 数值 | 来源 |
|---|---|---|
| 16GB，Wan2.2，`blocks_to_swap=12` | 约 10 s/it（swap=5 时 245 s/it） | [S19] |
| 16GB，360×640×81 帧，rank16，176 步 | 3.5–5 小时 → 72–102 s/it | [S21] |
| 24GB，81 帧、>256×256 | 可训（靠 blockswap） | [S23] |
| 5090 32GB，Ostris 无 block swap | 256 以上分辨率卡住 | [S24] |
| 5090，768×768 图像 LoRA | 1.7 s/it（swap 5）；1024 时 3 s/it | [S22] |
| Wan2.2 5B 官方：5s 720P 单卡生成 | <9 分钟（未优化） | [S6] |
| H3 蒸馏学生推理：一次模型评估 | H200 约 37s；4090/5090 约 1 分钟；3090 约 2 分钟 | [S12] |

3.3 的 s/it 为推算：按 `6×P×N` FLOPs（含约 1/3 重算开销）除以该卡有效 bf16 吞吐（3070 Ti Laptop 8–15、3060 6–10、4070 Ti S 55–75、4090 70–95、5090 80–110 TFLOPS）估算，再用上表实测点校准；720p 的 token 数是 480p 的 3.6–5.4×，注意力二次项使时间约 ×5–7。

### 3.5 其它模型在 8/16/24/32GB 上的结论

| 模型/配置 | 8GB | 16GB | 24GB | 32GB |
|---|---|---|---|---|
| Wan2.1-1.3B 480×832×33（VAE 4×8×8 → N=14040） | ⚠️ 约 5.8–6.5GB，换块 0–5（推算） | ✓ | ✓ | ✓ |
| Wan2.1/2.2-14B 480p | ✗ | ⚠️ 换块 26–30（实测可行）[S17] | ✓ 换块 20+ [S18] | ✓ 换块 10–20 |
| HunyuanVideo 13B 480p 视频 LoRA | ✗（图像 LoRA 可以，512×512、24 块换出、文本编码器借 12GB 卡）[S17] | ⚠️ fp8_base + 换块 30 | ✓ | ✓ |
| CogVideoX-2B LoRA 49×480×720 | ✗（官方 16GB）[S14] | ✓（官方 16GB） | ✓ | ✓ |
| CogVideoX-5B LoRA 49×480×720 | ✗ | ✗（官方 24GB） | ✓（官方 24GB） | ✓ |
| CogVideoX-5B 全参 SFT 49×480×720 | ✗ | ✗ | ✗（单卡 ZeRO-2+opt offload 需 42GB）[S14] | ✗ |
| LTX-Video 2B LoRA 480p | ⚠️ 推算可试（VAE 32× 压缩，token 少） | ✓ | ✓ | ✓ |
| MiniMax-H3 LoRA | ✗ | ✗ | ✗ | ✗（66GB 权重 + 48GB 文本编码器）[S5] |
| MiniMax-H3 DMAD 蒸馏训练 | ✗ | ✗ | ✗ | ✗（8×H200）[S12] |

## 4 LLM 侧：7B/8B/14B 在 8GB 与 24GB

| 任务 | 7B/8B @8GB | 7B/8B @24GB | 14B @8GB | 14B @24GB |
|---|---|---|---|---|
| QLoRA 4bit（SFT/DPO） | ✓ 6GB（表值 estimated） | ✓ 余量充裕，可长上下文 | ✗ 12GB | ✓ 12GB |
| QLoRA 8bit | ⚠️ 10GB 超 | ✓ | ✗ 20GB | ⚠️ 20GB 勉强 |
| LoRA 16bit | ✗ 16GB | ✓ 16GB | ✗ 32GB | ✗ 32GB |
| 全参 SFT（pure_bf16） | ✗ 60GB | ✗ 60GB | ✗ 120GB | ✗ 120GB |
| 全参 SFT（32bit 优化器） | ✗ 120GB | ✗ | ✗ 240GB | ✗ |
| GRPO（4bit 策略） | ⚠️ 仅短上下文（1–2K）+ 小 batch | ✓ 15GB 档可训到 17B | ✗ | ⚠️ 24GB 紧 |

- 数值取自 LLaMA-Factory「Hardware Requirement」表（页面自注 *estimated*）[S25]：7B 的 QLoRA 4bit=6GB、LoRA 16bit=16GB、pure_bf16 全参=60GB、32bit=120GB；14B 依次 12/32/120/240GB；官方另有 FSDP+QLoRA 在 2×24GB 训 70B 的示例。
- Unsloth RL 文档：**"15GB VRAM 可把 ≤17B 模型（Llama 3.1 8B、Phi-4 14B）做成 RL"**；QLoRA 4bit 的 GRPO 经验法则是"参数量 ≈ 显存 GB"；20K 上下文 8 代时 Llama 3.1 8B 需 54.3GB，标准实现 510.8GB，Unsloth 把 78.3GB logits 降到 9.8GB；Qwen2.5-3B 的"aha moment"原需 2×A100(160GB)，Unsloth 单卡 5GB 可复现 [S26]。
- 全参 7B 在 24GB 上唯一路径是 ZeRO-3/FSDP + 参数与优化器 CPU offload，吞吐掉一个数量级，单卡不建议。
- DPO 需要参考模型（或冻结 adapter 副本），8GB 上 7B 的 DPO 即便 QLoRA 也吃紧，建议 24GB。

## 5 蒸馏（DMD / DMAD）在单卡上的位置

- **DMD**（Distribution Matching Distillation）训练少步学生，需要维护一个跟随学生分布的辅助扩散模型（额外显存+算力）；**DMD2** 是 NeurIPS 2024 Oral 的改进版（图像）[S30]。
- **DMAD**（Distribution Matching as Adversarial Distillation，arXiv 2610.02188，ByteDance + Texas A&M，2026-10-02）把 DMD 的 score 拟合换成两路判别头（real vs generated、teacher vs generated），在判别器最优点可证明恢复 DMD 梯度 [S12]。
- 已发布成果：MiniMax-H3-33B → 4 步学生（rank-128 LoRA，1.4GB；ComfyUI 布局 2.0GB，另 8 步版），人评胜过 DMD2 79.1%、胜过 rCM 84.6%；Wan2.1-1.3B/14B → 4 步，VBench total 84.70 / 85.15 [S12][S13]。
- **训练成本**：H3 学生用 **8×H200**，每 100 iter 约 6.7h（LoRA critic）/5.5h（full critic）/6.2h（8 步），主 run 800 iter、full critic 1600 iter；数据 UltraVideo 约 300GB（gated）[S12]。Wan2.1 学生用 8 GPU×17k iter（1.3B）与 64 GPU×19.5k iter（14B），训练数据 531GB / 5270 shards [S13]。
- **单卡判定**：训练不可行（判别器要再挂一份 33B 级模型）；单卡只能消费已发布 LoRA 做 4 步推理（5.9GiB 峰值，见 2.2）。另有 Self-Forcing / Self-Forcing-Plus（4 步 T2V-14B）与 LightX2V（Wan 步蒸馏权重）可作为替代来源 [S29]。
- 许可交叉风险：H3 许可证禁止"用其输出改进其它模型"，用 H3 输出做蒸馏属于被限制行为 [S11]。

## 6 备选方案对比

| 方案 | 成本 | 时间（一次 5B 视频 LoRA，1000 步） | 隐私 | 可复现性 | 适用 |
|---|---|---|---|---|---|
| 本地 3070 Ti 8GB | 0（电费） | 只能 256–360p：30–90 s/it → 8–25 小时；480p 不成立 | 最高（数据不出机器） | 高，但移动卡降频导致速度漂移 | 原型、LLM QLoRA 7B、视频推理 |
| 本地 3060 12GB | 0 | 20–40 s/it → 6–11 小时 | 最高 | 高 | 480p 小 LoRA |
| 本地 4070 Ti S 16GB | 0 | 6–12 s/it → 2–3.5 小时 | 最高 | 高 | 480p 常规 LoRA |
| 租 RunPod 4090 24GB | **$0.89/h** [S27] | 3–5 s/it → 1–1.5 小时 ≈ **$1–2**；含镜像与数据准备约 $3–5 | 中（需上传数据集） | 高（固定镜像 + seed） | 一次性训练、对照实验 |
| 租 RunPod 5090 32GB | $1.19/h [S27] | 1.5–3 s/it → 0.5–1 小时 ≈ $1–2 | 中 | 高 | 720p、A14B |
| 租 8×H200 做 DMAD | $5.29/h/卡（Secure 价）→ **$283/100 iter**；一个 800 iter 学生约 **$2270**、墙钟 53.6 小时 [S27][S12] | 2–3 天 | 低 | 高 | 蒸馏（团队级预算） |
| Vast.ai 竞价 | 通常低于 RunPod；页面为 JS 动态、公开 API 403，本次**未取到可引用数值** | 同上 | 低–中 | 中（机器异构） | 便宜的短任务，需自行核价 |
| 云 API 微调（文本） | Together：Llama 3.1 8B SFT **$0.34/1M tokens**（全参表为 $0.38）、DPO $0.84/1M、最低 $4 [S28]；OpenAI gpt-4.1 $25/1M 训练，**已对新用户关闭** [S29b] | 30–60 分钟 | 最低（数据上云） | 低（黑盒超参，拿不到权重） | 纯文本 SFT，不需要权重 |
| LoRA 合并导出 | 0 | 分钟级（需基座全量显存，bf16 5B=10GB，8GB 需 CPU/分片合并） | 高 | 高 | 交付、部署、GGUF 量化 |

结论：8GB 本地适合"验证流程 + 小模型 LLM"，不适合视频训练；视频训练用 16GB 本地或 $0.89/h 的 4090 租卡；蒸馏与 33B 级模型只在租多卡时考虑。

## 7 风险清单与营销夸大核对

| 常见说法 | 核查结果 | 依据 |
|---|---|---|
| "8GB 能训视频 LoRA" | 能跑通的是**图像 LoRA**（512×512、20 张、24 块换出，且文本编码器要另找 12GB 卡缓存）或**推理**；视频训练官方底线 24GB | [S1][S17] |
| "MiniMax-H3 开源可商用" | 排除 EU/UK/韩国/美国；>$20M 营收需书面授权；禁止用其输出改进其它模型 | [S11] |
| "H3 支持 2K、参考图生视频" | 2K 依赖未开源的 H3-Regenerate-2K；H3-Context-IR 未开源 | [S10] |
| "消费级显卡可跑 720P" | Wan2.2 5B 官方口径是 24GB（4090），且指推理并加 `--offload_model --t5_cpu` | [S6][S7] |
| "diffusion-pipe 支持 Windows" | 官方 README：原生 Windows 基本不可行，建议 WSL2 | [S16] |
| "QLoRA 7B 只要 6GB" | 6GB 是 estimated 最小值，对应短上下文/小 batch；长上下文会超 | [S25] |
| "fp8 在 30 系上更快" | Ampere 无原生 fp8 matmul，fp8 只省显存；`--fp8_fast` 只在 40 系有效 | [S4] |
| "量化无损" | DMAD 的 int8 权重与原版 PSNR 14–26dB（均值 19.6），观感接近但不是 bit-identical | [S12] |
| "1.4GB LoRA 就能复现 33B H3" | 那是**推理用学生权重**，训练仍需 8×H200 与 300GB 数据 | [S12] |
| "12GB 也能训 81 帧 480p" | 12GB 的公开报告都是 2–4s 短片 + rank 8–16；81 帧需 16GB+ | [S20][S21] |

**8GB 卡训练视频的真实上限**：能稳定跑的是 ≤360×640、≤49 帧、rank 8–16、fp8 + 换块 15–26 块的 Wan2.2 5B，或 Wan2.1 1.3B 的 480×832×33；速度 30–90 s/it，一次 1000 步要 8–25 小时。480×832×49 及以上在 8GB 上不成立。

## 8 最小验证清单（8GB 上先跑这三个）

1. **LLM**：QLoRA 4bit 训 7B/8B，`seq_len 1024`、`batch 1`、`grad_accum 8`，用 LLaMA-Factory 或 Unsloth，记录峰值显存（预期 6–7GB）。
2. **视频**：Wan2.2-TI2V-5B + fp8 + `blocks_to_swap 22` + 480×832×21 帧起步，逐次加帧到 OOM，记录 s/it 与换块比例。
3. **推理**：Wan2.2 5B 480p 单卡推理（`--offload_model True --convert_model_dtype --t5_cpu`），确认 8GB 是否可行（官方口径 24GB，offload 后待实测）。

## 9 来源表（URL 与取数日期，均为 2026-11-11 访问）

| 编号 | 来源 |
|---|---|
| S1 | https://github.com/kohya-ss/musubi-tuner （README Hardware Requirements、Windows 共享显存说明） |
| S2 | https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md |
| S3 | https://github.com/kohya-ss/musubi-tuner/blob/main/docs/block_swap.md |
| S4 | https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video.md |
| S5 | https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3.md |
| S6 | https://github.com/Wan-Video/Wan2.2 （README，2025-07-28 发布） |
| S7 | https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B |
| S8 | https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B?blobs=true （文件字节数） |
| S9 | https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/raw/main/transformer/config.json |
| S10 | https://github.com/MiniMax-AI/MiniMax-H3 ；https://huggingface.co/MiniMaxAI/MiniMax-H3 |
| S11 | https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE （License date 2026-08-02） |
| S12 | https://github.com/Yzmblog/DMAD ；https://arxiv.org/abs/2610.02188 |
| S13 | https://github.com/Yzmblog/DMAD/blob/main/train/wan/README.md |
| S14 | https://github.com/zai-org/CogVideo/blob/main/finetune/README.md |
| S15 | https://github.com/Lightricks/LTX-Video ；https://github.com/Lightricks/LTX-Video-Trainer |
| S16 | https://github.com/tdrussell/diffusion-pipe （docs/supported_models.md、examples/wan_14b_min_vram.toml） |
| S17 | https://github.com/kohya-ss/musubi-tuner/issues/175 |
| S18 | https://github.com/kohya-ss/musubi-tuner/discussions/182 |
| S19 | https://www.reddit.com/r/StableDiffusion/comments/1mizk9d/ （2025-08-06） |
| S20 | https://www.reddit.com/r/StableDiffusion/comments/1rxvwyp/ （2026-03-19） |
| S21 | https://reddit.sentinel-team.org/posts/1tv9xqa/snapshots/2026-06-06T00%3A10%3A31.218907Z （r/StableDiffusion 快照，2026-06-06） |
| S22 | https://www.reddit.com/r/StableDiffusion/comments/1mmni0l/ （2025-08-10） |
| S23 | https://www.reddit.com/r/StableDiffusion/comments/1peyq51/ （2025-12-05） |
| S24 | https://www.reddit.com/r/StableDiffusion/comments/1nde4x1/ （2025-09-10） |
| S25 | https://github.com/hiyouga/LLaMA-Factory （README Hardware Requirement 表） |
| S26 | https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide |
| S27 | https://www.runpod.io/pricing |
| S28 | https://www.together.ai/pricing （Fine-Tuning 表） |
| S29 | https://github.com/GoatWu/Self-Forcing-Plus ；https://github.com/tianweiy/DMD2 |
| S29b | https://platform.openai.com/docs/pricing （gpt-4.1 微调 $25/1M，注明 closed to new users） |
| S30 | https://arxiv.org/abs/2405.14867 （DMD2, NeurIPS 2024 Oral） |
| S31 | https://github.com/Wan-Video/Wan2.1 |