# SuperTuner 市场调研（01）：竞品矩阵与市场空缺

- 调研日期：2026-11-11（所有 star / issue 数、价格、版本信息均为该日抓取）
- 数据来源：GitHub REST API（`api.github.com/repos/*`）、各项目官方 README/文档页、Reddit / 知乎 / V2EX 公开帖、厂商定价页
- 口径说明："最低显存"指官方文档或官方示例配置给出的**单卡**下限；训练框架类项目若无官方数字，标注实测出处
- 目标人群：RTX 3070 Ti 8GB → RTX 5090 32GB 的单卡用户

## 1. 竞品矩阵

口径：star 数只作为活跃度代理，不等于活跃用户数；闭源厂商无公开用户量，标"无公开数字"。开源项目的 open issues 数一并列出，作为维护负担/需求积压的参考。

### 1.1 LLM 微调 / 后训练栈

| 产品 | 定位 | 支持模型/训练阶段 | 视频 | 最低显存（单卡） | 许可证 | 用户量（2026-11-11） | 关键特色 |
|---|---|---|---|---|---|---|---|
| Unsloth | 桌面应用 + no-code Web UI（Studio），LLM/扩散/TTS/Embedding 训练与推理一体（[Studio 文档](https://unsloth.ai/docs/new/studio/)） | SFT、DPO、GRPO、RL、视觉微调、Embedding、TTS；宣称可训练图像/视频扩散模型 | 是（README 称可训练图像/视频扩散；视频训练独立文档页 404，文档尚不完整） | 官方最低表：3B QLoRA 3.5GB / 7B 5GB / 14B 8.5GB / 32B 26GB（[requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)） | Core Apache-2.0 + Studio UI AGPL-3.0 | 79,302 stars / 7,386 forks / 484 open issues（[API](https://api.github.com/repos/unslothai/unsloth)） | 宣称"2× 快、70% 少显存"；Windows/macOS/Linux 原生安装包、Homebrew/Winget/Docker；Cloudflare 远程访问 |
| LLaMA-Factory | CLI + WebUI 的全阶段 LLM 微调工厂 | 连续预训练、SFT、RM、PPO、DPO、KTO、ORPO；2–8bit QLoRA、多模态 | 否 | 未给统一下限（QLoRA 路线，8GB 级 7B 可跑） | Apache-2.0 | 75,987 stars / 9,362 forks / 1,227 open issues | 每日支持新模型（Qwen3、GPT-OSS、GLM、InternVL）；YAML 驱动；生态最广 |
| TRL | HF 官方后训练库（代码级） | SFT、DPO、KTO、GRPO 等 trainer（README 主推这四种，PPO 为历史实现） | 否（多模态 LLM） | 由 Transformers Trainer + 量化决定，无官方下限 | Apache-2.0 | 19,713 stars / 3,131 forks / 315 open issues | 与 Transformers/PEFT 深度集成，DDP/ZeRO/FSDP；纯代码，无 GUI |
| Axolotl | YAML 驱动的 SFT/DPO/GRPO 工具链 | SFT、DPO、KTO、ORPO、PPO、GRPO、MoE LoRA、NVFP4 | 否 | 无统一下限，QLoRA 路线 8–12GB 级 | Apache-2.0 | 12,683 stars / 1,472 forks / 247 open issues | 强分布式（EP/FSDP2/Ulysses/CP）、插件式云启动、Async GRPO |
| torchtune | PyTorch 官方 recipes 库 | SFT、LoRA/QLoRA、DPO、蒸馏等 | 否 | 无 | BSD-3-Clause | 5,819 stars / 764 forks / 471 open issues | **已停止维护**（README 声明 2025 年结束开发，见 issue #2883） |
| HF AutoTrain Advanced | HF 托管 no-code 微调 | 传统 AutoModel 任务（文本/视觉） | 否 | 云端托管，本地跑需自备 | Apache-2.0 | 4,620 stars / 630 forks / 5 open issues | **已停止维护**（README 顶部 WARNING，建议改用 Axolotl/TRL/transformers.Trainer） |

### 1.2 视频 / 扩散训练栈

| 产品 | 定位 | 支持模型/训练阶段 | 视频 | 最低显存（单卡） | 许可证 | 用户量（2026-11-11） | 关键特色 |
|---|---|---|---|---|---|---|---|
| ai-toolkit (ostris) | 面向消费级硬件的一体化扩散训练套件，GUI + CLI | LoRA 微调：FLUX.1/2/3、Qwen-Image(-2.1/Edit/Turbo)、Wan 2.1/2.2、Hunyuan 等 | 是（Wan 2.1 T2V/I2V、2.2 T2V-A14B） | 官方示例即 24GB（`train_lora_flux_24gb.yaml`、`train_lora_wan22_14b_24gb.yaml`） | MIT | 12,612 stars / 1,619 forks / 84 open issues | 唯一"消费级"定位的开源视频训练 GUI；内置 hardware detector 一键装依赖 |
| SimpleTuner | 企业化训练平台（WebUI + CLI），多云 | 图像/视频 LoRA、Lycoris、全量微调；模型覆盖 Flux.1/2、Krea2、Cosmos2、SD3、MiniMax H3 | 是，但 **Wan 不支持 image-to-video 训练**（[WAN.md](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/WAN.md)） | Wan2.1 1.3B rank-16 LoRA 约 12GB（bs=4）；14B "塞进 24G，但需要反复调参" | AGPL-3.0 | 2,956 stars / 296 forks / 7 open issues | 内置 RBAC、审批流、成本阈值；多 provider 调度 |
| musubi-tuner (kohya) | CLI 视频 LoRA 训练脚本 | LoRA：Wan 2.1/2.2、HunyuanVideo、FramePack、FLUX.1 Kontext/FLUX.2、Qwen-Image、Z-Image、MiniMax-H3 | 是（当前视频 LoRA 的事实标准） | 官方：图像 12GB+，**视频 24GB+**，系统内存 64GB+ | 未声明（仓库无 LICENSE 文件） | 2,136 stars / 321 forks / 375 open issues | block swap / fp8 / 分块预缓存；纯 CLI，无 GUI |
| diffusion-pipe | DeepSpeed 流水线并行的视频/扩散训练 | HunyuanVideo、Wan、Flux 视频 LoRA/全量 | 是 | 通过 `wan_14b_min_vram.toml` 降显存，仍需 24GB 级 | GPL-3.0 | 2,036 stars / 289 forks / 275 open issues | **Windows 原生不可用**（DeepSpeed 限制，只支持 WSL2）；依赖版本大量不固定 |
| kohya_ss GUI | 图像 LoRA 桌面 GUI（已事实冻结于图像域） | SD1.5/2.x、SDXL、SD3、FLUX.1、Lumina、Anima、HunyuanImage-2.1 | **否** | 未给统一下限（SDXL 常见 12–24GB） | Apache-2.0 | 12,664 stars / 1,605 forks / 541 open issues | TOML 化配置 + Flux Kontext 打标工具；最后提交 2026-08-01 |
| finetrainers (HF) | HF 官方内存优化扩散训练库 | CogVideoX、LTX-Video 等视频 LoRA | 是 | 未给 8–16GB 路径 | Apache-2.0 | 1,364 stars / 146 forks / 70 open issues | 内存优化 + 分布式；最后提交 2026-09-23 |
| ComfyUI + 训练节点 | 节点式推理工作流，训练靠社区节点 | 社区节点（kijai/ComfyUI-WanVideoWrapper 等） | 部分（Wan LoRA 训练节点仍需外部数据目录约定） | 取决于节点 | GPL-3.0（ComfyUI） | 146,214 stars / 17,410 forks / 5,501 open issues | 生态最大，工作流可复用；训练能力非官方核心 |
| DiffSynth-Studio | 阿里 ModelScope 的扩散/视频训练库 | Wan 2.2、CogVideoX 等 | 是 | 官方明确 **不开启训练期 offload**（issue #1016 回复） | Apache-2.0 | 13,355 stars / 575 open issues（[API](https://api.github.com/repos/modelscope/DiffSynth-Studio)） | 工程化好，但显存优化取向保守 |

### 1.3 托管 / 云端替代

| 产品 | 定位 | 支持模型/训练阶段 | 视频 | 最低显存/成本 | 许可证 | 用户量 | 关键特色 |
|---|---|---|---|---|---|---|---|
| RunPod | GPU 租赁（Pod / Serverless） | 自建环境，视频训练只能自己装 musubi/diffusion-pipe | 是（自建） | 4090 社区云 $0.34/hr 起；RTX Pro 6000 社区 $1.69/hr、安全 $2.49/hr；L40S $0.79/hr；A100 PCIe $1.19/hr；H100 PCIe $1.99/hr（[pricing](https://www.runpod.io/pricing)） | 闭源 | 无公开数字 | 灵活，但训练被 pod 中断/网络卷计费反复困扰（见 §2.4） |
| Vast.ai | P2P 竞价 GPU | 同上 | 是（自建） | 价格随主机实时竞价，无固定价目表（[vast.ai/pricing](https://vast.ai/pricing)） | 闭源 | 无公开数字 | 主机质量参差，需自己筛 |
| Modal | Serverless GPU，按秒计费 | 容器化训练，常用官方示例 | 是（自建） | H100 SXM5 / A100 80GB / A100 40GB 按小时（[pricing](https://modal.com/pricing)） | 闭源 | 无公开数字 | 冷启动与常驻成本需自行权衡 |
| Lightning AI | 云端 Studio | 通用训练环境 | 是（自建） | 50 免费 GPU 小时后按需付费，$0.51–$5.45/hr 档（[pricing](https://lightning.ai/pricing)） | 闭源 | 无公开数字 | 交互式 Studio |
| Together / Fireworks | 托管微调 + 推理 API | 仅 LLM 文本模型微调（[together.ai/fine-tuning](https://www.together.ai/fine-tuning)） | **否** | 训练后按 token 计费（Fireworks 输入 $0.05/1M 起） | 闭源 | 无公开数字 | 便宜好用，但完全不覆盖视频/扩散训练 |
| Predibase / OpenPipe | 托管 LoRA 微调 | LLM LoRA、蒸馏 | 否 | — | 闭源 | 无公开数字 | **OpenPipe 已停止训练/服务**：据第三方迁移说明（2026-11-02）其 2026-07-30 停止训练与托管，[openpipe.ai/pricing](https://openpipe.ai/pricing) 仅剩迁移通告；Predibase 并入 Rubrik |
| NVIDIA NeMo | 企业级框架 + NeMo 微服务 | LLM SFT/LoRA/RL，Data Curator | 否（视频侧无训练路径） | 面向 A100/H100 集群 | 源码开放 | 无公开数字 | 企业治理、评测、数据流水线 |

### 1.4 桌面分发 / 一键包

| 产品 | 定位 | 支持模型/训练阶段 | 视频 | 最低显存 | 许可证 | 用户量（2026-11-11） | 关键特色 |
|---|---|---|---|---|---|---|---|
| Unsloth Desktop | 桌面应用（训练+推理+导出） | 见 §1.1 | 是 | 见 §1.1 | Core Apache-2.0 / Studio AGPL-3.0 | 同 §1.1 | winget/homebrew/docker 安装；Studio 仍是 beta |
| Pinokio | 通用一键脚本分发器 | 由各脚本决定，多为推理 | 训练脚本零散 | 取决于脚本 | MIT | 8,387 stars / 855 forks / 475 open issues | 生态最杂，同一任务多套脚本，版本漂移 |
| Stability Matrix | 多平台包管理器 + 推理 UI | 可一键装 OneTrainer 等训练包，但自身只做推理 | 间接 | 取决于包 | AGPL-3.0 | 9,014 stars / 618 forks / 192 open issues | Windows/macOS/Linux 客户端，包版本管理清晰 |

## 2. 真实用户痛点（有出处）

### 2.1 视频训练：显存是首要硬门槛

- Reddit（2025-08-11，18 评）："Possible to train LoRA for WAN2.2 on 24GB VRAM?"，作者有 24GB 卡仍不知如何配 — https://www.reddit.com/r/StableDiffusion/comments/1mnphgk/
- Reddit（2025-12-05）：用户有 **5090 32GB + 64GB RAM**，图像 LoRA 已能全分辨率训练，但视频 LoRA "找不到能塞进本机的设置" — https://www.reddit.com/r/StableDiffusion/comments/1peyq51/
- Reddit（2025-12-29，25 评）：有人总结可用 16GB 训练 Wan 2.2 LoRA，但属于个案技巧 — https://www.reddit.com/r/StableDiffusion/comments/1pz0w56/
- 官方门槛佐证：musubi-tuner README 写明"图像 12GB+，**视频 24GB+**，系统内存 64GB+" — https://github.com/kohya-ss/musubi-tuner
- 官方门槛佐证：ai-toolkit 官方示例配置即按 24GB 档给（`train_lora_flux_24gb.yaml`、`train_lora_wan22_14b_24gb.yaml`）— https://github.com/ostris/ai-toolkit/tree/main/config/examples
- SimpleTuner 官方 Wan 文档：14B"塞得进 24G，但得反复调设置" — https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/WAN.md
- 连 96GB 卡也会 OOM：musubi-tuner issue #723（RTX PRO 6000 96GB 训 Wan 2.2 low-noise 仍 OOM）— https://github.com/kohya-ss/musubi-tuner/issues/723
- 视频训练 OOM 高发：musubi issue #46（30 评）、#317（22 评，"图像训练完美，视频就 OOM"）— https://github.com/kohya-ss/musubi-tuner/issues/317
- 显存规划本身是用户教育成本：knightli.com（2026-07-28）专门写"显示内存规划要按峰值算，不要用模型 14GB 就认为 16GB 卡够用" — https://knightli.com/zh-tw/2026/07/28/wan-2-2-local-deployment-comfyui-vram-video-generation-troubleshooting/

### 2.2 依赖地狱 / 安装失败（Windows 消费级最痛）

- diffusion-pipe README：DeepSpeed 是硬依赖，"在原生 Windows 上几乎不可能跑起来"，只能 WSL2；requirements 大量不固定版本 — https://github.com/tdrussell/diffusion-pipe
- LLaMA-Factory FAQ issue #4614："`llamafactory-cli: command not found / 无法找到命令`"（安装/升级反复翻车）— https://github.com/hiyouga/LlamaFactory/issues/4614
- LLaMA-Factory issue #5763：装 `llamafactory[torch-npu]` 直接依赖冲突 — https://github.com/hiyouga/LlamaFactory/issues/5763
- Axolotl issue #945：按文档 `pip install` 后 `ModuleNotFoundError: No module named 'axolotl'` — https://github.com/axolotl-ai-cloud/axolotl/issues/945
- 知乎（2025-02-20）：用 kohya_ss 训 LoRA"难点就是各种报错"，CUDA/Python/PyTorch 版本与推荐不一致就装不上 — https://zhuanlan.zhihu.com/p/25107424388
- 升级即回归：LLaMA-Factory issue #8600（27 评）"升级后 OOM 且 tokenization 变慢" — https://github.com/hiyouga/LlamaFactory/issues/8600

### 2.3 训练过程缺可观测性、缺数据集工具

- DiffSynth-Studio issue #645："希望支持 tensorboard / wandb / swanlab"（已关闭但反映普遍诉求）— https://github.com/modelscope/DiffSynth-Studio/issues/645
- Reddit（2025-10-23）Wan 2.2 I2V 训练实践："加了梯度范数日志，光看 loss 不足以判断训练是否正常" — https://www.reddit.com/r/StableDiffusion/comments/1odwfms/
- ai-toolkit issue #624：训练成功后 info 面板里 "Training Dataset Tags" 不显示 — https://github.com/ostris/ai-toolkit/issues/624
- DiffusionToolkit issue #222：请求内置打标/描述生成 — https://github.com/RupertAvery/DiffusionToolkit/issues/222
- Forge issue #465：请求把 Dataset Tag Editor 集成进训练 UI — https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/465
- ComfyUI-WanVideoWrapper issue #968：想训 Wan 2.2 I2V LoRA，但 ai-toolkit 要求"不是单个 .safetensor，而是整套目录结构"才能用 — https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/968

### 2.4 托管 GPU 替代方案也不省心

- Reddit r/LocalLLaMA（2026-05-14）：准备上 RunPod 训模型，"看了一圈帖子很慌"：pod 训练中途死掉、checkpoint 无法恢复、pod 起不来还在计费 — https://www.reddit.com/r/LocalLLaMA/comments/1td938k/
- Reddit r/RunPod（2026-05-29）：没 GPU pod 可用却继续为 400GB network volume 付费；"frustrating" — https://www.reddit.com/r/RunPod/comments/1tqlxjz/
- Reddit r/StableDiffusion（2025-09-18）："PSA: 别用 RunPod 的 Network Volume" — https://www.reddit.com/r/StableDiffusion/comments/1nkcgvp/
- 托管微调供给萎缩：据第三方迁移说明（2026-11-02）OpenPipe 已于 2026-07-30 停止训练与服务（https://turion.ai/blog/openpipe-shutdown-migration/），Predibase 并入 Rubrik；torchtune 停止维护（https://github.com/meta-pytorch/torchtune/issues/2883）；AutoTrain README 顶部标注 "no longer maintained"（https://github.com/huggingface/autotrain-advanced）。

### 2.5 LLM 侧：单卡/多卡与显存反馈

- Unsloth issue #2435（102 评）：多卡训练需求长期挂着，是最热的需求帖之一 — https://github.com/unslothai/unsloth/issues/2435
- Unsloth 官方 VRAM 表（2026-11-11）：32B QLoRA 仍需 26GB、40B 需 30GB，70B 超出表内 — https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- LLaMA-Factory issue #6999（25 评）/ #8642（15 评）：8×80GB/H200 集群做 32B 全参微调仍 OOM — https://github.com/hiyouga/LlamaFactory/issues/6999
- V2EX（2025-06-12）主题「对于程序员，微调大模型这条路走的通吗？」：楼主用 4090D 24GB + llama-factory 调 Qwen 7B，"翻来覆去的调试，效果比 RAG 差很多"，并指出"B 站视频最多介绍怎么搭建环境，微调参数的设置基本没有系统性讲解"；回帖直言"4090D 微调 7B 到顶了，参数量再大的显存不够" — https://www.v2ex.com/t/1138109

## 3. 市场空缺（按证据强度排序）

1. **8–16GB 显存下的"视频训练可完成路径"**。事实：主流工具官方下限都是 24GB（musubi、ai-toolkit、SimpleTuner），32GB 用户在 Reddit 上仍失败（2025-12-05），96GB 也 OOM（musubi #723）。缺的是一个显存预算器 + 按预算自动降级（block swap / 分辨率 / 帧数 / 精度 / batch / 序列长度）+ 失败自动回滚的产品，而不是又一个 CLI。
2. **LLM 后训练与视频训练没有统一工作台**。LLM 侧已被 Unsloth / LLaMA-Factory / Axolotl / TRL 覆盖（SFT→DPO/KTO/ORPO→GRPO/PPO 全链路）；视频侧只有 LoRA，且 SimpleTuner 明确"Wan 不支持 image-to-video 训练"。Unsloth Studio 是唯一把视频训练带进 GUI 的项目，但其视频训练文档页 404（2026-11-11 抓取 https://unsloth.ai/docs/basics/diffusion-video），说明这块仍在早期、集成深度未经用户验证。缺口：同一套数据/评测/导出流水线同时管 LLM 与视频。
3. **训练可观测性与断点保护**。DiffSynth #645（要 wandb/swanlab）、Reddit 2025-10-23（要看梯度范数）、Reddit 2026-05-14（pod 挂掉没 checkpoint 可恢复）指向同一件事：没有"训练过程可视 + 自动安全检查点 + 断点续训 + 失败恢复"的现成方案。
4. **数据集流水线缺失**。打标工具散落在各自 GUI（kohya_ss 的 Kontext 打标、ai-toolkit tags 显示 bug #624、WanVideoWrapper #968 的目录结构约定、Forge #465 的集成请求）。缺：抽帧 → 自动打标 → 人工审核 → 直接进训练的视频数据集闭环。
5. **托管替代正在萎缩**，只剩"自己装环境"的路。OpenPipe 2026-07-30 停服、Predibase 并入 Rubrik、AutoTrain/torchtune 停止维护；而 RunPod/Vast 的痛点是环境自建 + pod 存活 + 存储计费（2026-05-14 / 2026-05-29 / 2025-09-18）。缺口：把本地工具直接镜像到 GPU 租赁、或提供带断点续训的托管训练层。
6. **分发层混乱**。Pinokio（8,387 stars，475 open issues）同一任务多脚本、版本漂移；Stability Matrix 只做包管理+推理（可装 OneTrainer），不提供训练界面。缺：可信、版本锁定、带健康检查的训练工作台分发。

## 4. 结论

### 4.1 Table stakes（缺了就不成立）

- 单卡可跑：至少覆盖 8GB→32GB 的显存档位，给出明确的"能不能跑/跑多快"预检，而不是让用户自己撞 OOM。
- 一键安装：Windows 优先，CUDA/PyTorch/编译依赖自动解决；不要求 WSL2 作为唯一路径（diffusion-pipe 已被证明是劝退点）。
- 覆盖 LLM 全后训练链路：SFT、LoRA/QLoRA、DPO、KTO/ORPO、GRPO/PPO，以及预训练续训。
- 视频侧至少覆盖 Wan 2.1/2.2 与一个 I2V 路径，产出 LoRA 并可导出到 ComfyUI 可直接用。
- 数据集与打标、训练日志、checkpoint 与断点续训、模型导出（safetensors/GGUF）——这四项每个竞品都缺一块，必须自建。
- 显存与速度预估、失败原因可读化（把 OOM 翻译成"降哪个参数"）。

### 4.2 差异化机会（可被 SuperTuner 占据）

1. **显存优先的产品契约**：内置"预算→参数"求解器（分辨率/帧数/精度/offload/batch 自动联动），把 12–16GB 训练 Wan LoRA 从"看运气"变成"有保证"，并给出失败时的降级路径。这是当前最大的空白且有硬证据支撑。
2. **一个工作台同时训练 LLM 与视频**：数据集、评测、导出、版本管理共用；LLM 侧带 DPO/KTO/ORPO/GRPO 全链路，视频侧补上 Unsloth 尚未文档化的部分。
3. **训练可观测 + 可恢复**：实时 loss/显存/吞吐曲线、梯度范数、显存峰值预警、自动安全 checkpoint、跨机断点续训——直接回应 §2.3/§2.4 的三大投诉。
4. **视频数据集闭环**：抽帧、自动打标、审核、版本化数据集、与训练配方联动。
5. **可信分发**：锁定依赖版本的安装包（不是脚本集合），带环境健康检查与一键降级路径；相对 Pinokio/Stability Matrix/Unsloth Desktop 都能讲出差异。
6. **可选的云端回退**：本地显存不够时，一键把同一配方搬到按小时 GPU 并带断点续训；这是 OpenPipe 关闭后留下的需求缺口，也是对 RunPod 痛点的直接回应。
