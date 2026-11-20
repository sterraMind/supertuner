# 训练器代码深读：musubi-tuner（H3/Wan）与 Fizgig 的 LoRA 训练路径（2026-11-20）

> 任务：深读两个开源训练器的 LoRA/视频训练代码，提炼「什么样的数据会有什么样的影响」「什么样的训练方式会带来什么问题」，为 SuperTuner 的实验矩阵与评估设计提供依据。
> 日期：2026-11-20。行号均为克隆当日 commit 的行号，升级上游后需重新核对。
> 代码来源（均为本地浅克隆，路径相对本仓库）：
> - `.drafts/musubi-tuner`（kohya-ss/musubi-tuner，commit f8a1b03，2026-09-27）
> - `.drafts/Fizgig`（shootthesound/Fizgig，commit b5199ee，2026-11-20）
> 行号均指克隆内文件。查不到依据的结论一律标「未核实」。

## 0. 结论速览表（数据/方法 × 影响 × 证据位置）

| # | 数据/方法维度 | 对训练效果的影响 | 证据位置 |
| --- | --- | --- | --- |
| 1 | H3 检查点是 CFG 蒸馏过的：普通 flow-matching LoRA 会把模型拉出「被放大的引导空间」 | 视频 LoRA 随训练步数增加而 wash out、提示词遵循度下降；纯图像 LoRA 约 50 步就结构性崩坏 | musubi `docs/minimax_h3.md:111` |
| 2 | 过拟合保护靠「teacher 匹配」而非 caption dropout：teacher 看特权条件（首/尾帧、参考片段、主体参考图），学生看普通条件 | 身份（identity）类 LoRA 必须用 teacher 匹配；风格/动作 LoRA 可用 de-distill 适配器或 guidance loss | `docs/minimax_h3.md:113-151`；`minimax_h3_train_network.py:1778-1853` |
| 3 | H3 的 loss 不是普通 MSE：分解为范数项+方向项并 detach 方向，方向梯度是纯旋转 | 防止 wash-out 到条件均值（「wash-out to conditional mean」）；teacher 残差永不为零，checkpoint 应选 plateau 附近 | `minimax_h3_train_network.py:520-554, 589-616`；`docs/minimax_h3.md:471` |
| 4 | H3 数据管道无 caption dropout（grep 无 `caption_dropout`/`no_caption`） | 触发词绑定靠 `--network_dropout`（神经元 dropout）兜底；风格 LoRA 易把数据集调色板学成「风格偏移」，需 `--h3_teacher_loss_dc_weight<1` 抑制 | `minimax_h3_train_network.py:520`（DC 衰减）；`training/parser_common.py:567` |
| 5 | 数据管道硬性约束：batch=1、24fps 严格重采样、VAE 空间 16×（分辨率须被 16 整除）、帧数 17n+5、最低 5 帧 | 数据准备阶段就决定显存与效果上限；帧数/分辨率偏离网格直接报错或截断 | `minimax_h3_train_network.py:326, 101-107`；`minimax_h3/packing.py:276`；`generation_inputs.py:65`；`image_video_dataset.py:687-692,765-768` |
| 6 | H3 文本编码器 = Qwen3-VL-32B（50 层隐态，TEXT_WIDTH=5120，最多 32768 行） | 文本编码器本身占 48GB（NVFP4+AWQ 后 15GB），是显存大头之一；长 Ref2VA 需 `flash_attention_2`（>~3k 行时 OOM） | `text_encoder.py:41-57`；`docs/minimax_h3.md:499-514` |
| 7 | H3 LoRA 目标 = DiT 每块的 attn.qkv_proj/out_proj + mlp.fc1/fc2（默认全 50 块） | rank/alpha 16/16 是文档基准；`--network_dropout` 是 H3 路径上唯一的神经元级正则 | `networks/lora_minimax_h3.py:14-15`；`minimax_h3_train_network.py:1876-1900` |
| 8 | 音频：H3 单流设计，视频 LoRA 也会改音频路径权重 | 纯视频 LoRA 的音频输出「不受约束」（drift）；无声图像数据集必须 `--video_only`，且 caption 要写明「无声」 | `docs/minimax_h3.md:488-497, 277`；`minimax_h3_train_network.py:1143-1182` |
| 9 | Wan 与 H3 的关键差异：Wan2.2 A14B 是高低噪双模型（high/low noise 双 checkpoint + 时间步边界切换）；i2v 用 CLIP（仅 Wan2.1，v2.2 无 CLIP） | Wan 双模型需双份权重加载/交换（`--dit_high_noise`+`--timestep_boundary`），swap 路径复杂；Wan 无 H3 的 teacher 体系 | `wan_train_network.py:78-103, 531-582, 584-625`；`lora_wan.py:37` |
| 10 | Wan LoRA 排除 patch_embedding/time_embedding/text_embedding/time_projection/norm | Wan 的 LoRA 只打注意力+FFN，嵌入层与 norm 层不训 | `networks/lora_wan.py:37` |
| 11 | Fizgig 用原生 PyTorch 重写 H3 DiT（非 musubi 包装），为消费卡做了一整套「砍掉即能跑」的取舍 | 8GB 卡可训 H3（4-bit NF4，代价是 ~9% 基座误差）；16GB 可跑 int8 H3 LoRA | `Fizgig/src/fizgig/minimax/model.py:633`；`src/fizgig/families/minimax.py` |
| 12 | Fizgig 单向 H2D block swap：只从 CPU→GPU 预取，不回传（基座冻结，回传是浪费） | 比经典双向 block swap 快 ~6.4×（文档值）；每块 0.39GB 固定内存，2 槽环 | `Fizgig/src/fizgig/minimax/h3_h2d_offload.py`；`Fizgig/docs/MINIMAX_H3.md` |
| 13 | Fizgig 只训 blocks 20-49（默认）/ 6-49（More Blocks）；blocks 0-5 默认不训 | 省 backward 计算；0-5 块负责「解剖形变 + 把数据集颜色拉进渲染」，不训 = 不学新调色板；38-48 是语音核心块 | `Fizgig/src/fizgig/families/minimax.py`（H3_LIKENESS_MODE）；`docs/MINIMAX_H3.md` |
| 14 | TREAD 令牌路由：训练时随机让 ~50% 视频令牌跳过部分 DiT 块 | 显著降低显存（9MP 剪辑无 checkpointing 在 88GB 卡上 OOM，开 TREAD 可跑）；代价是每块只看到一半令牌 | `Fizgig/src/fizgig/minimax/model.py`（TREAD 段）；`Fizgig/src/fizgig/minimax/driver.py:73-78,1151-1153` |
| 15 | automagic3 优化器（Ostris ai-toolkit vendored）：自适应学习率 | 消费卡档默认优化器；adamw 在「相似度」测试中是「单项最大收益」 | `Fizgig/src/fizgig/training/optimizers.py:63-75`；`Fizgig/docs/MINIMAX_H3.md:56` |
| 16 | 预览自动降档：16GB 卡预览封顶 768×640×22 帧；OOM 时按 141→56→22 帧降档，再降分辨率到 512 下限 | 「训练能跑的分辨率」与「预览能看的分辨率」是两条独立预算；预览档 = 可验证上限 | `Fizgig/src/fizgig/minimax/driver.py:176-190, 911-987`；`common.py:1138-1165` |
| 17 | 量化基座精度直接影响 LoRA 学到什么：int8 基座 ~0.17% 误差（自身存储格式），NF4 4-bit ~9% 误差 | 在 4-bit 基座上训的 LoRA 会「浪费容量去校正推理时不存在的量化误差」——基座量化精度是隐式训练变量 | `Fizgig/src/fizgig/minimax/common.py:595-692`；`Fizgig/docs/MINIMAX_H3.md` |
| 18 | INT8 反向传播被实测移除（issue #221）：int8 梯度行舍入让离群值定标，LoRA 梯度与精确值余弦 0.2-0.6 | int8 前向保留（2/3 的 matmul 量），反向必须 bf16 精确反量化 | `Fizgig/src/fizgig/modules/int8_train.py:15-17` |
| 19 | W8A8（int8 激活）在相同 loss 下「毁掉相似度」→ 改 W8A16：权重 int8、激活 bf16、Triton 单轮舍入 | 量化路径的每个细节都直接影响 LoRA 相似度；W8A16 每 GEMM 1.14-1.42× 加速 | `Fizgig/src/fizgig/convrot_w8a16_triton.py` |
| 20 | ConvRot = 分组（256）Hadamard 旋转做 int8 量化前的离群值展平 | Hadamard 正交可自逆，避免 SmoothQuant 式失败；反旋转 0.9% 相对误差 | `Fizgig/src/fizgig/convrot.py:1-22`；arXiv:2512.03673 |
| 21 | 消费卡显存档位（Fizgig 文档表）：~30GB int8 无 swap；~22GB int8 流式 ~14 块；~15GB int8 ~36 块；≤12GB 4-bit | H3 的卡位下限是 16GB（README 表）/ 12GB（4-bit 路径）；「8GB 起」指整个训练器（Krea 2/Anima/SDXL 等家族），H3 本身 8GB 不可训 | `Fizgig/docs/MINIMAX_H3.md`（VRAM 表）；`README.md:104-115` |
| 22 | 数据质量信号：Fizgig 0.25MP 默认（「每步比 1MP 便宜 4 倍，更高分辨率没回本」）；caption dropout 0.05 默认；EMA 0.98 在 50 epoch A/B 中 +5 相似度 | 数据分辨率/数量存在「够用即止」拐点；caption dropout 与 EMA 是 H3 上被实测有效的正则 | `Fizgig/docs/MINIMAX_H3.md`；`Fizgig/src/fizgig/families/minimax.py:24-52` |
| 23 | H3 帧网格 17n+5（最低 5 帧）、Wan 帧数须 16 的倍数相关（vae_stride 4，latent=(f-1)/4+1） | 两个模型的帧数约束不同，数据管道必须按模型选帧数，不可混用 | `musubi .../minimax_h3/packing.py:276`；`wan_train_network.py:244` |
| 24 | H3 时间步采样被锁死：timestep_sampling=uniform、weighting none、discrete_flow_shift=1.0（R1 版本唯一接受值） | H3 训练器不接受常见 SD 风格时间步重加权——「方法→问题」的硬约束例 | `minimax_h3_train_network.py:711-715` |
| 25 | teacher 匹配/教师条件：`--h3_teacher_conditions` ∈ first,last / ref / subject_ref；teacher 条件 σ 门控（sigma_max 默认 1.0，subject_ref 建议 0.15） | 数据「身份」训练必须让 teacher 看到学生看不到的条件；σ 门控决定 teacher 何时「降级」为基线锚 | `minimax_h3_train_network.py:1778-1853`；`docs/minimax_h3.md:138-151,447-471` |

## 1. H3 LoRA 训练（musubi-tuner）

### 1.1 数据管道

- **帧数网格**：H3 帧数必须形如 `17n+5`，最低 5 帧（`minimax_h3/packing.py:276` `pixel_frames = 17 * n + 5`；`minimax_h3_train_network.py:101-107` 校验）。这是 VAE 时间压缩的硬约束，数据准备阶段就要把素材重采样到该网格，否则截断/丢弃。
- **分辨率**：VAE 空间下采样 16×（`generation_inputs.py:65` `VIDEO_VAE_SPATIAL_RATIO=16`），分辨率须为 32 的倍数（文档 `--video_size multiples of 32`）。`docs/minimax_h3.md:554-557`。
- **帧率**：H3 严格 24fps 重采样（`image_video_dataset.py:687-692` 中 `TARGET_FPS` 相关逻辑，L765-768 对 H3 强制目标帧率，忽略源帧率）。
- **batch=1 强制**（`minimax_h3_train_network.py:326`；`docs/minimax_h3.md:198`）：H3 每个数据集 batch 必须为 1，更大的有效 batch 用梯度累积。
- **参考视频约束**（Ref2VA）：每条记录最多 12 个参考（图/视频混合），参考视频 2-15 秒（`docs/minimax_h3.md:91`）。
- **音频**：H3 是单流多模态（视频+音频+文本同一 transformer），音频按 24kHz 采样；无声素材要显式写「无声」caption（`docs/minimax_h3.md:277`）；视频 LoRA 也会改音频权重（见 §1.4 失败模式）。

**数据→效果结论**：H3 的数据管道把「效果上限」压死在缓存阶段——帧数网格、分辨率、帧率都是硬约束；caption 内容（尤其 Ref2VA 的参考声明格式）决定条件分布，teacher 条件在文本编码器缓存时就已固化（换 teacher 种类要重新缓存，`docs/minimax_h3.md:356`）。

### 1.2 Caption 处理

- H3 路径**没有 caption dropout**：在 `minimax_h3_train_network.py` 与 dataset 代码中 grep `caption_dropout|no_caption` 均无命中。正则靠 `--network_dropout`（神经元 dropout，`training/parser_common.py:567`）。
- caption 三种任务形态（`docs/minimax_h3.md` 与 `text_encoder.py:114-205`）：
  - `t2va`（文生视频）：纯文本 caption；
  - `fl2va`（首尾帧生视频）：首帧/尾帧图 + caption，teacher 条件由首尾帧派生；
  - `ref2va`（参考生成）：图/视频/音频参考 + 时间戳对齐的 caption，caption 中引用 `<Picture N>`/`<Video N>` 等占位符。
- **文本编码器 = Qwen3-VL-32B**（H3 官方仓库的 `text_encoder` 子目录，非 Qwen 官方权重），`TEXT_WIDTH=5120`、`MAX_TEXT_ROWS=32768`（`text_encoder.py:41-57`）。缓存文本编码器输出是 H3 训练的必经步骤（`--text_cache`），长 Ref2VA 场景建议 `flash_attention_2`（>~3k token 行时 OOM，`docs/minimax_h3.md:499-514`）。
- teacher 匹配时 caption 的「特权信息」：teacher 看到的 caption 可包含学生看不到的条件（如主体参考图），学生 caption 只保留触发词（`docs/minimax_h3.md:130,145,315`）。

### 1.3 训练循环

- **入口/结构**：`MiniMaxH3NetworkTrainer`（`minimax_h3_train_network.py:659`），1913 行文件。R1（release 1）约束：`--timestep_sampling uniform`、`--weighting_scheme none`、`--discrete_flow_shift 1.0` 是唯一接受的组合（L711-715）；`blocks_to_swap<=48`（L723-726）；fp8 基座不支持（用 `--convrot_int8` 代替，L723-726）。
- **loss 分解**（这是 H3 与普通 flow-matching 训练的核心差异）：
  - `_dc_attenuated_prediction`（L520）：把预测分解为 DC（全局色/色调，「风格轴」）与 AC（细节）分量；对 DC 分量做衰减，防止把数据集调色板学成风格偏移（`--h3_teacher_loss_dc_weight<1`）。
  - `_decomposed_flow_loss`（L532）：`||p-t||² = (||p||-||t||)² + 2·||p||·||t||·(1-cos)`，detach `||p||` 使方向梯度成为纯旋转项——防 wash-out 到条件均值。
  - `_preservation_density_compensation`（L554）：锚点步骤的梯度密度补偿。
  - teacher 诊断量：`teacher/*_cos`、`_norm_ratio`、`_residual_dc_rms`、`_residual_ac_rms`（L589-616）。
- **teacher 匹配**（L1778-1853 CLI，L121 文档）：student 用普通条件（t2va），teacher 用同一 transformer（LoRA 关掉）在特权条件下跑一遍（首/尾帧、参考片段、主体参考图）；teacher 条件 σ 门控：`sigma_max`（默认 1.0，subject_ref 建议 0.15）控制 teacher 何时生效。这是「数据→效果」的直接体现：**身份类 LoRA 的效果取决于 teacher 条件中「学生看不到的信息量」**。
- **guidance loss**（L1752-1770）：`--h3_guidance_loss_scale` 默认 0.0（字段报告建议 3-4），`sigma_min` 门控 0.15；用 uncond 探针重建目标 `target' = uncond + scale*(target-uncond)`，对抗「去蒸馏漂移」（H3 发布权重是 CFG 蒸馏的）。
- **参考 LoRA 命令**（`docs/minimax_h3.md:380-397`）：`--network_dim 16 --network_alpha 16 --sdpa --mixed_precision bf16 --gradient_checkpointing --blocks_to_swap 48 --optimizer_type adamw8bit --learning_rate 1e-4 --max_train_epochs 16`。
- **LoRA 目标模块**（`networks/lora_minimax_h3.py:14-15`）：`attn.qkv_proj`、`attn.out_proj`、`mlp.fc1`、`mlp.fc2`（默认全 50 块 DiT）；支持从 Diffusers 格式（ai-toolkit/diffusion-pipe）导入第三方 LoRA（`convert_lora_state_dict`）。
- **checkpoint 选择**：`docs/minimax_h3.md:471`——loss 不会收敛到 0（teacher 知道文本不知道的事），几百步后残差进入 plateau，**最强 checkpoint 在 plateau 附近或刚过后**，要保存并评估中间 checkpoint。

### 1.4 失败模式（代码/文档可读到的）

| 失败模式 | 机制 | 证据 |
| --- | --- | --- |
| 过拟合/身份漂移 | H3 检查点是 CFG 蒸馏的，普通 flow LoRA 拉出引导空间；视频随步数 wash out，图像 ~50 步崩坏 | `docs/minimax_h3.md:111` |
| 调色板偏移（style bleed） | 数据集颜色被学进 LoRA 的风格轴 | `minimax_h3_train_network.py:520`（DC 衰减）；`docs/minimax_h3.md:471` |
| 文本渗透（text bleed） | teacher 条件中「学生看不到的身份」若漏进 caption/条件，学生学到的是渗透而非泛化 | `docs/minimax_h3.md:130,145,315`（subject_ref 契约：参考图用「同一主体的其它照片」，caption 只留触发词） |
| 音频漂移 | 单流设计：视频 LoRA 改音频路径权重，纯视频 LoRA 的音频输出不受约束 | `docs/minimax_h3.md:488-497` |
| OOM：文本编码器 | Qwen3-VL-32B 占 48GB（NVFP4+AWQ 后 15GB）；>~3k 行需 flash_attention_2 | `docs/minimax_h3.md:499-514` |
| OOM：VAE 解码峰值 | 采样/预览时 VAE 解码显存峰值高于训练步 | 未核实（代码中 VAE 解码峰值无显式注释，仅社区讨论见 05 号调研） |
| 时间步采样锁定 | H3 只接受 uniform/none/1.0，SD 风格重加权不可用 | `minimax_h3_train_network.py:711-715` |

## 2. Wan LoRA 训练（与 H3 的差异）

- **入口**：`wan_train_network.py`（772 行），`WanNetworkTrainer`（L40）。任务来自 `WAN_CONFIGS`（`wan/configs/__init__.py:43`）：`wan_t2v_14B`、`wan_t2v_A14B`、`wan_i2v_14B`、`wan_i2v_A14B`、`wan_t2v_1_3B`。
- **Wan2.2 A14B 双模型**：高噪/低噪两个 expert 模型，训练时 `--dit_high_noise` + `--timestep_boundary` 切换（L78-103）；`get_noisy_model_input_and_timesteps`（L531-582）按时间步边界重采样到正确模型；`swap_high_low_weights`（L584-625）在编译模型间交换权重（`_orig_mod` key 重映射）。这是 H3 完全没有的复杂度：H3 单模型 + σ 门控 teacher，Wan 是物理双权重。
- **i2v 条件**：Wan2.1 用 CLIP image encoder（`clip_fea`），v2.2 无 CLIP（L156/671）；`control_latents` 支持 Fun-Control 类控制。
- **LoRA 目标**（`lora_wan.py:37`）：排除 `patch_embedding|text_embedding|time_embedding|time_projection|norm`，即只训 `WanAttentionBlock` 内的注意力+FFN。与 H3 对比：H3 打 `attn.qkv_proj/out_proj + mlp.fc1/fc2`（等价范围，但排除逻辑写在代码里而非配置里）。
- **帧数**：latent 帧数 = `(frame_count-1)/vae_stride[0]+1`（`wan_train_network.py:244`），即 Wan VAE 时间 stride=4，帧数须 4n+1（与 H3 的 17n+5 不同，数据管道不可混用）。
- **采样/采样器**：Wan 支持 `--one_frame` 单帧模式（L234-241）、`--timestep_boundary`、`--fp8_scaled`（A14B 双模型需 fp8）等；采样步数与 H3 的「官方 50 步 = `--infer_steps 49`」口径不同（H3 的 49 是 flow shift 1.0 下的官方对应值，`docs/minimax_h3.md:557`）。
- **与 H3 的本质差异**：Wan 没有 teacher 匹配/teacher 条件体系、没有 CFG 蒸馏问题（Wan2.2 非蒸馏发布）、没有音频流；Wan 的 LoRA 训练更接近「标准 flow-matching LoRA」，其失败模式以普通过拟合/运动退化为主（社区配方外流见 05 号调研 §1.2）。

## 3. Fizgig 消费卡取舍与权衡

> 代码位于 `Fizgig/src/fizgig/`。Fizgig 的 H3 是**原生 PyTorch 重写**（`minimax/model.py:633` `MiniMaxH3DiT`），不是 musubi 包装——这点决定了它的取舍是「为 8GB 卡重写」而非「包装现成引擎」。

### 3.1 消费卡档位与量化

- **显存档位**（`Fizgig/docs/MINIMAX_H3.md` VRAM 表）：~30GB int8 无 swap；~22GB int8 流式 ~14 块；~15GB int8 ~36 块；≤12GB 4-bit（NF4）。H3 的卡位下限是 16GB（`README.md:107`）；「8GB 起」指整个训练器（Krea 2/SDXL/Anima/Flux 3 等家族，`README.md:104-115`），H3 本身 8GB 不可训。
- **基座量化误差是隐式训练变量**：int8 基座 ~0.17% 误差（自身存储格式），NF4 4-bit ~9% 误差（`common.py:595-692`）。代码注释直白：在 4-bit 基座上训的 LoRA「会花容量去校正推理时不存在的量化误差」（`minimax/driver.py` NF4 警告）。
- **int8 前向/反向拆分**（`modules/int8_train.py:15-17`）：int8 前向覆盖 2/3 的 matmul 量（grad checkpoint 重算），但**int8 反向被实测移除**（issue #221：int8 梯度行舍入让离群值定标，LoRA 梯度与精确值余弦仅 0.2-0.6）→ 反向必须 bf16 精确。
- **W8A8→W8A16**（`convrot_w8a16_triton.py`）：W8A8（int8 激活）「在相同 loss 下毁掉相似度」→ W8A16：int8 权重在 Triton kernel 里转 bf16、激活 bf16、fp32 累加、单次舍入；每 GEMM 1.14-1.42× 加速。
- **ConvRot**（`convrot.py:1-22`）：分组（256）Hadamard 旋转做 int8 量化前的离群值展平，正交自逆（避免 SmoothQuant 失败）；反旋转 0.9% 相对误差。来源 arXiv:2512.03673。

### 3.2 单向 H2D block swap

- `H3Int8H2DOffloader`（`minimax/h3_h2d_offload.py`）：**只**做 CPU→GPU 预取（2 槽环、每块 0.39GB pinned、`ring_size` 预取、backward hook 反向预取）。基座冻结 → 经典双向 swap 的 D2H 回传全是浪费。文档值 ~6.4× 快于经典 swap。
- 代价：每块 pinned 内存 0.39GB（31 块 ≈ 13GB pinned），16GB 卡上「块换」与「预览」抢同一块显存。

### 3.3 只训部分 block

- **blocks 20-49 默认**（`H3_LIKENESS_MODE`，`families/minimax.py`）：省 backward 计算；**0-5 块**负责「解剖形变 + 把数据集颜色拉进渲染」，不训 = 不学新调色板（有意为之：相似度 LoRA 不该改全局调色）；**38-48 是语音核心块**。
- **TREAD 令牌路由**（`minimax/model.py` TREAD 段；`driver.py:73-78`）：训练时随机让 ~50% 视频令牌跳过部分 DiT 块；9MP 剪辑不开 TREAD 时 88GB 卡无 checkpointing 仍 OOM；全量微调（full FT）时 TREAD 关闭（L1151-1153）。
- **代价**：每块只看到一半令牌 → 块间信息流变稀，「块级语义分工」（0-5 调色、38-48 语音）是 TREAD 下保持的效果前提。

### 3.4 优化器与正则

- **automagic3**（`training/optimizers.py:63-75`，Ostris ai-toolkit vendored）：自适应学习率，消费卡档默认；文档称 adamw 是「H3 上相似度单项最大收益」（`docs/MINIMAX_H3.md:56`）。
- **caption dropout 0.05 默认**、**EMA 0.98**（`families/minimax.py:24-52` 预设）：50 epoch A/B 中 EMA 0.98 比关 +5 相似度、方差减半（`docs/MINIMAX_H3.md`）。
- **0.25MP 默认分辨率**：「每步比 1MP 便宜 4 倍，更高分辨率没回本」（文档原话）。

### 3.5 预览自动降档（768×640×22 → 512×512）

- **预览是独立预算**：`driver.py:176-190` 规划预览时，16GB 卡（总显存<20GB 档）封顶 **768×640、22 帧**；OOM 时 `driver.py:911-987` 的 `_ladder` 按顺序降：帧数 141→56→22，再降分辨率（`common.py:1138-1165` `next_preview_res`：1024²→1024×768→768²→768×640→640²→640×512→512²）。
- **含义**：「训练用多大」与「预览看多大」是两条独立管线；预览降档是**可验证上限**，不是训练分辨率。512 是下限（再小无意义）。

## 4. 「数据→效果」与「方法→问题」结论汇总

**数据侧：**
1. 帧数/分辨率/帧率是硬网格（H3 17n+5/24fps/16 整除；Wan 4n+1/16fps），数据准备错网格 = 直接报错或截断，不是「效果差一点」。
2. caption 内容决定 teacher 条件的「特权信息」边界：身份 LoRA 的 caption 要**刻意少写**（触发词 only），teacher 看参考图；风格 LoRA 的 caption 可以全写，但 DC 分量要衰减防调色板泄漏。
3. 参考图契约（subject_ref）：1-9 张「同一主体的其它照片」（最好不含目标），caption 只留触发词——这是 H3 身份 LoRA 的数据契约。
4. 基座量化精度（int8 vs NF4）是隐式训练变量：4-bit 基座会消耗 LoRA 容量去校正量化误差。
5. 分辨率存在「够用即止」拐点（Fizgig 0.25MP 默认；musubi 文档示例 768×1344）。

**方法侧：**
6. H3 检查点已 CFG 蒸馏 → 普通 flow LoRA 必然漂移，必须三选一（adapter/guidance/teacher）；这是 H3 与 Wan LoRA 的最大方法差异。
7. H3 无 caption dropout，正则靠 teacher 体系 + `--network_dropout` + DC 衰减；Fizgig 显式加 caption dropout 0.05。
8. teacher 残差 loss 不收敛到 0，checkpoint 选择看 plateau 而非 loss 值。
9. 消费卡的每个「能跑」都是砍掉某条信息通路（TREAD 砍令牌、0-5 块不训砍调色板通路、int8 砍精度、4-bit 砍更多）——每个砍法都有对应「效果丢失」模式，可逐一评估。
10. 量化路径的细节（W8A8 vs W8A16、int8 反向）直接影响相似度，量化不只是「省显存」而是「改变训练分布」。

## 5. 可实验的变量清单（供 SuperTuner 实验矩阵）

| 变量 | 取值范围 | 预期影响 | 自动验证方式 |
| --- | --- | --- | --- |
| LoRA rank/alpha | 8/16/32 | 容量↑→过拟合风险↑；16/16 是文档基准 | 固定步数下 train/val loss 差、固定 prompt 的 CLIP 相似度方差 |
| caption dropout | 0 / 0.05 / 0.1 | 0.05 为 Fizgig 默认；过高伤遵循度，过低伤泛化 | 同 prompt 不同随机种子生成，测多样性与指令遵循 |
| EMA | 0.9 / 0.98 / off | 0.98 实测 +5 相似度、方差减半 | 多 seed 相似度均值-方差 |
| teacher 条件 σ 门控（sigma_max/min） | 0.4-1.0 / 0 / 0.15 | 身份 LoRA：teacher 条件越多越像但不泛化；σ 门控决定 teacher 生效区间 | 参考图与生成图的人脸/主体相似度（如 CLIP-I 或人脸相似度） |
| DC 衰减权重（`--h3_teacher_loss_dc_weight`） | 0.3 / 0.7 / 1.0 | <1 抑制调色板吸收；风格 LoRA 保持 1.0 | 生成帧的直方图/主色调与数据集调色板的距离（防「风格漂移」） |
| 时间步 focus（`--h3_timestep_focus_prob`） | 0 / 0.5 | 0.5 时 content-deciding 区间收敛 ~2× | 相同步数下 teacher 残差 plateau 到达步数 |
| 基座量化精度 | bf16 / int8 / NF4 | NF4 基座消耗 LoRA 容量校正量化误差 | 同 LoRA 在 bf16 基座 vs 4-bit 基座上的相似度差 |
| 量化激活精度 | W8A8 / W8A16 | W8A8 毁相似度 | 固定数据两种 kernel 的梯度余弦（复现 #221 的 0.2-0.6 现象） |
| TREAD 令牌跳过率 | 0 / 0.25 / 0.5 | 0.5 省显存但块间信息流变稀 | 固定显存下最大可训分辨率/帧数；运动连贯性评分 |
| 训练 block 范围 | 20-49 / 6-49 / 全 50 | 0-5 不训=不学新调色板；范围↑=backward 成本↑ | 调色板漂移指标 + 每步耗时 |
| 帧数（H3 网格） | 5/22/39/56 | 帧数↑=时间信息↑=成本↑ | 固定显存下帧数-动作连贯性曲线 |
| 分辨率 | 0.25MP / 0.5MP / 1MP | 0.25MP 为「够用」档 | 固定步数下细节 FID/CLIP 边际收益 |
| optimizer | automagic3 / adamw8bit | adamw 相似度最优（文档值） | 固定预算下相似度-步数曲线 |
| 时间步采样（H3） | uniform（唯一） | 无选择（R1 限制） | —（Wan 上可做：uniform vs 高噪偏置） |
| 蒸馏补偿方式 | adapter / guidance / teacher | 三者互斥（guidance 与 teacher 不可同用） | 固定步数的「去蒸馏漂移」指标（uncond 距离） |

## 6. 对 SuperTuner 实验/评估设计的建议

1. **评估要分「数据契约」层与「训练」层**：H3 的很多失败（文本渗透、音频漂移、调色板偏移）是数据契约问题，不是训练超参问题。评估管线应先做**数据契约校验**（帧数网格、分辨率整除、caption 格式、参考图契约、无声标注），再进训练。
2. **teacher 匹配是 H3 身份 LoRA 的必选路径**，SuperTuner 的 H3 支持至少要实现：privileged 条件缓存（teacher 条件在文本编码器缓存时固化）、σ 门控、DC/AC 分解 loss。这三者都在 `minimax_h3_train_network.py` 中已有参考实现。
3. **checkpoint 选择策略**：不要按 loss 最小选 checkpoint；按 teacher 残差 plateau 附近选（`docs/minimax_h3.md:471`）。评估器应支持「保存并评估中间 checkpoint」。
4. **可自动度量的指标**（对应上表变量）：
   - prompt 遵循度：CLIP 文本-视频相似度（固定 prompt 集）；
   - 编辑/身份保真：参考图-生成图相似度（人脸/主体）；
   - 长视频一致性：时序 CLIP 方差、运动连贯性（光流或视频质量模型）；
   - 过拟合：固定 prompt 多 seed 的生成多样性（标准差）+ train/val loss 差；
   - 稳定性：同 seed 多 checkpoint 的 loss 方差（EMA/optimizer 对比用）；
   - 风格漂移：生成帧直方图 vs 数据集调色板的统计距离（DC 分量）；
   - 量化误差：int8/NF4 基座 vs bf16 基座的 LoRA 相似度差。
5. **Fizgig 的「预览降档」是评估设计的好范式**：把「预览分辨率/帧数」做成显式可降档的评估档位（141→56→22 帧、768→512 分辨率），让评估本身也符合消费卡显存约束，且降档路径可复现（`driver.py:911-987`）。
6. **Wan 与 H3 要分实验线**：Wan2.2 A14B 的双模型（high/low noise）与 H3 单模型 teacher 体系不可互换；Wan 的 LoRA 更接近标准 flow-matching（无 CFG 蒸馏问题），评估指标可简化（无 teacher 残差，可用标准 FID/CLIP）。

## 7. 证据与未核实项

- 已核实（代码/文档直读）：§0 表中标注 `musubi ...`、`Fizgig ...` 的所有行号与文档行号；`docs/minimax_h3.md` 与 `docs/minimax_h3_advanced.md` 的 teacher/音频/VRAM 章节；`Fizgig/docs/MINIMAX_H3.md` VRAM 表与 TREAD/caption-dropout/EMA 值。
- **未核实**：
  - H3 采样/预览时 VAE 解码显存峰值的具体数值（代码无显式注释，仅社区讨论，见 05 号调研）。
  - Fizgig 「8GB 起」是否含 H3 的 4-bit 路径的完整可训练验证（文档称 12GB 可 4-bit H3；8GB 档位主要针对 Krea2/Flux/Anima 等其它家族，H3 的 8GB 路径未在文档中单列）。
  - musubi H3 的 `--h3_timestep_focus_prob` 默认值 0.4-0.8 区间与「2× 收敛」的精确出处（文档 `minimax_h3_advanced.md`，行号见上文 §1.3 引用）。
  - Wan2.2 A14B 双模型 swap 的编译模型 `_orig_mod` 重映射在 fp8 下的行为（`wan_train_network.py:584-625` 有代码但无注释说明 fp8 场景）。

## 8. 来源

- kohya-ss/musubi-tuner @ f8a1b03（2026-09-27）：`src/musubi_tuner/minimax_h3_train_network.py`、`src/musubi_tuner/networks/lora_minimax_h3.py`、`src/musubi_tuner/wan_train_network.py`、`src/musubi_tuner/networks/lora_wan.py`、`src/musubi_tuner/minimax_h3/packing.py`、`src/musubi_tuner/text_encoder.py`、`src/musubi_tuner/image_video_dataset.py`、`src/musubi_tuner/generation_inputs.py`、`docs/minimax_h3.md`、`docs/minimax_h3_advanced.md`、`src/musubi_tuner/training/parser_common.py`
- shootthesound/Fizgig @ b5199ee（2026-11-20）：`src/fizgig/minimax/model.py`、`src/fizgig/minimax/h3_h2d_offload.py`、`src/fizgig/minimax/driver.py`、`src/fizgig/minimax/common.py`、`src/fizgig/families/minimax.py`、`src/fizgig/training/optimizers.py`、`src/fizgig/modules/int8_train.py`、`src/fizgig/convrot.py`、`src/fizgig/convrot_w8a16_triton.py`、`docs/MINIMAX_H3.md`
