# 07 免费/近似免费 GPU 平台调研（SuperTuner 可运行性）

> 调研日期：2026-11-19。面向 SuperTuner（单卡训练控制平面）当前可免费或近似免费使用 GPU 的平台，对能否跑通本仓库 LLM LoRA SFT 验收进行评估。  
> 本地验收基线（见 `README.md`、`MEMORY.md`、`engine/pyproject.toml`）：Qwen3-0.6B + wikitext-2 LoRA SFT，20 步，442.9 tok/s，**峰值 1787 MiB**，adapter 40,422,168 B（RTX 3070 Ti Laptop 8GB，Windows）。引擎依赖 Python `>=3.10,<3.14`、`uv`、`torch>=2.4` + `transformers/peft/accelerate/datasets`，README 推荐 CUDA 13.2 轮子 `torch==2.15.0+cu132`；服务端需 Node.js >= 22.5（`node:sqlite`）、pnpm 10。  
> 说明：以下外部信息按不可信网页内容处理，仅据可核实来源记录；抓取日期统一为 **2026-11-19**（除特别标注外）；查不到或页面未明确给出的数字写「未核实」。

## 1. 结论速览

| 平台 | GPU / 显存 | 免费额度（2026-11 现状） | 能否跑 Qwen3-0.6B LoRA SFT（峰值 1787 MiB） | 能否跑 H3/视频 LoRA（4–8GB 训练 / ≥16GB 场景） | 是否要信用卡/支付方式 | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| Google Colab（免费版） | 通常 T4 16GB（可用时）；付费可解锁 L4/A100/H100 等 | 动态无固定配额；单会话上限 **≤12 小时**；空闲/超限会断连 | **可**（16GB 远超 1.75 GiB） | **受限/未核实**：16GB 可做规划与小规模尝试，但 H3 文本编码缓存等步骤显存/存储需求高，真实训练未验证 | 免费一般不需要；Pro/Pro+/PayG 需付费/地区可用 | [S1][S2][S3][S4] |
| Kaggle Notebooks | 当前免费 GPU 机型 **T4 ×2（各 16GB，独立卡）**；历史 P100 已于 2026-09-15 退役 | **约 30 GPU 小时/周**（浮动，视资源可能更高）；CPU/GPU 会话 ≤12h，TPU ≤9h；交互空闲约 60 分钟超时 | **可**（单卡 16GB 足够，需指定单卡使用） | **受限/未核实**：显存单卡可覆盖部分 4–8GB 档估算，但视频权重/缓存/依赖（musubi-tuner）与持久空间限制大，未验证 | 通常不需要信用卡；需账号/手机验证以启用 GPU 与联网 | [S5][S6][S7][S8][S9] |
| Google AI Studio（免费代码执行） | **无 GPU**（Python 沙箱，CPU） | 免费功能（按 token 计费语境）；单次代码执行最长 **30 秒**；不可安装自定义库（无 torch） | **否**（无 GPU、无 torch、30s 限制） | **否** | 需 Google 账号；功能免费但非算力平台 | [S10] |
| GCP 免费层 / 免费试用 | Always Free **无 GPU/TPU**；试用期账户限制添加 GPU | $300/90 天试用金（新用户）；试用期不可给 VM 加 GPU；Always Free 仅 CPU 类资源 | **否（免费层）**；用试用金升级付费后理论可行但非免费 | **否（免费层）** | **需要**信用卡/支付方式验证 | [S11][S12] |
| AWS Free Tier / 新账户额度 | 无 always-free GPU；g4dn(T4)/g5(A10G) 等为付费实例 | 2025-07-15 后新账户模式为 **$100 即时额度 + 最高再 $100**（探索奖励）；传统 CPU 免费额度见文档 | **否（免费 GPU）**；用额度租 GPU 为付费抵扣 | **否（免费 GPU）** | **需要**信用卡/支付方式 | [S13][S14][S15] |
| Azure 免费层 / 试用 | 无 always-free GPU；NC/ND/NV 为付费 | $200/30 天额度 + 部分 12 个月免费 CPU 服务（B1s 等） | **否（免费 GPU）** | **否（免费 GPU）** | **需要**信用卡/支付方式 | [S16] |
| Oracle Cloud Always Free / 试用 | Always Free **无 GPU**（AMD micro / Ampere A1 ARM 4 OCPU/24GB 等）；$300/30 天试用 | Always Free 长期可用但仅 CPU；试用金可用于付费资源 | **否（Always Free）** | **否** | **通常需要**支付方式验证（试用/部分场景） | [S17][S18] |
| RunPod（付费低价对照） | 按实例选：含 T4/L4/A10/A100/H100 等（显存视型号） | 无免费档；按秒/按小时计费 | **可**（选 ≥8GB NVIDIA GPU 实例） | **取决于实例显存/存储**（≥16GB/24GB 更现实；未在本平台实测） | **需要**充值/支付方式 | [S19][S20] |
| Vast.ai（付费低价对照） | 社区市场 GPU 多样（RTX 4090/A100/T4 等，显存视卡） | 无免费档；按分钟计费，价格随供需波动 | **可**（选 ≥8GB NVIDIA GPU） | **取决于所租卡与存储**；未实测 | **需要**充值/支付方式（最低充值门槛见平台） | [S21][S22] |
| Paperspace / DigitalOcean Gradient（付费对照） | 历史资料提及部分工作区有免费 M4000（旧/小显存，现状未核实）；付费 GPU 多档 | 免费额度/机型现状 **未核实**；付费按小时计费 | **未核实**（免费档若为 M4000 8GB 可能勉强但环境受限） | **未核实/不建议** | 付费通常需要支付方式；免费档条件未核实 | [S23][S4] |
| Lightning AI（免费 Studio + 额度） | 免费档可用 GPU（型号按可用性/文档，常见 T4 等） | 官方定价页标注 **“50 free GPU hours”** 起步；Studio 每 4 小时需重启；存储/额度细节有版本差异 | **大概率可**（单卡 ≥8GB 且可装依赖时） | **未核实**（受时长、重启、存储与依赖限制） | 通常免费注册可用，部分额度需手机验证；付费需卡 | [S24][S25] |
| Modal（每月免费计算额度，近似免费） | 按需分配 GPU（T4/A10/A100/H100 等计费档） | **$30/月免费计算额度**（按秒计费抵扣） | **可**（按代码/任务方式运行，需适配环境） | **未核实**（取决于选型与镜像/存储配置） | 需账号，超额需付费方式 | [S26] |
| Hugging Face Spaces / ZeroGPU | ZeroGPU 共享 GPU（适合推理/演示），非独占训练机 | 免费账户有限配额（社区信息约 5 分钟 GPU/天量级，具体以官方为准 **未核实**） | **否（不适合作为训练运行环境）** | **否** | 免费账号可用；Pro 扩额度 | [S27] |
| Google Colab Pro / Pro+（付费对照） | L4 / A100 40/80GB / H100 等（随可用性与 CU） | Pro 100 CU/月；Pro+ 600 CU/月（$49.99/月）；Pro+ 后台执行 ≤24h | **可**（资源更稳定） | **部分场景可尝试**（仍受 musubi-tuner、权重与缓存步骤限制，未实测） | **需要**订阅付费；地区限制 | [S1][S3] |

**结论速览要点**：对本仓库“Qwen3-0.6B LoRA SFT 10–20 步、峰值 1787 MiB”的验收，**Kaggle Notebooks** 与 **Google Colab 免费版** 是最现实的免费 GPU 路径；Kaggle 提供明确的周配额与 20GB 可保存工作区，Colab 更易上手但配额与持久化更不稳定。公有云免费层（GCP/AWS/Azure/Oracle）**不提供可直接用于训练的免费 GPU**；Google AI Studio 代码执行**无 GPU 且不可装 torch**。RunPod/Vast.ai 为低价付费补充，适合需要稳定显存或更长时间的场景。

## 2. 平台详细评估

### 2.1 Google Colab / Colab Enterprise（免费版）

- **来源与抓取日期（2026-11-19）**：[S1] Colab Paid Services Pricing / FAQ https://colab.research.google.com/signup ；[S2] Colab FAQ https://research.google.com/colaboratory/intl/en-GB/faq.html ；[S3] Fastero, *Google Colab GPU: Which GPU You Get…* https://fastero.com/blog/google-colab-gpu （2026-11-11 发布）；[S4] FreakyJolly, *Google Colab Free GPU Limits 2026* https://freakyjolly.com/google-colab-free-gpu-limits/ （2026-11-06 发布）；另参考 [Runpeek 2026-10-30](https://runpeek.app/guides/google-colab-limits)、[Hivenet 2026-02-11](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations)。
- **GPU 型号与显存**：免费版在可用时通常分配 **NVIDIA T4 16GB**（[S3] 指出签出状态运行时对话框提供 CPU/T4/v5e-1 TPU；[S1] 说明免费版 GPU 类型随时间变化、访问受限）。T4 16GB **≥8GB**，满足 Qwen3-0.6B LoRA SFT 峰值 **1787 MiB**。H3/视频 LoRA：本仓库基线显示 8GB 视频训练已紧张（README/MEMORY：Wan2.1 1.3B 480p 训练步估算 7.1 GiB；H3 cache-text 等步骤远超 16GB），16GB 免费 T4 可用于规划/编译预览，但**真实视频训练可行性未核实**。
- **免费额度**：无公开固定 GPU 配额（动态限额，重度使用后可能出现 “cannot currently connect to a GPU…” 并封禁一段时间，常见约一天，属非官方经验 [S4]）。单个 notebook **最长约 12 小时**（可用性与使用模式相关）[S1]；空闲会断连、VM 被回收 [S1][S2]。
- **信用卡/支付方式**：免费使用通常不要求绑定支付方式；Pro/Pro+/Pay As You Go 为付费服务且有地区限制（[S1] 列出可用国家/地区，未包含中国大陆）。
- **存储与持久化**：`/content` 为运行时临时磁盘，VM 重启后内容丢失；可挂载 Google Drive 持久化模型/数据/产物（Drive 免费 15GB，个人配额限制）。临时磁盘容量官方未在该页给出精确数值，社区资料常见约 100GB（[S4] 引述场景、KDnuggets 2026-02-20 等），**具体额度未核实**。
- **环境与 CUDA/torch**：Linux 环境，预装 CUDA 驱动/工具链版本随镜像变化（常见 CUDA 12.x，**未逐镜像核实**）。可通过 `pip`/`uv` 安装 PyTorch wheel（如 cu121/cu124/cu128/cu132，取决于宿主驱动支持的最高 CUDA 版本）；本仓库 README 指定 `torch==2.15.0+cu132`，可尝试安装但**能否成功取决于运行时驱动，未实测**。`uv` 与 Python 3.12 可安装。
- **网络**：通常可直连 Hugging Face Hub（`huggingface.co`）下载模型/数据集；受地区与临时网络波动影响，**未保证**。
- **结论**：**可**在免费 T4 上 clone 本仓库并跑通 Qwen3-0.6B LoRA SFT 10–20 步验收（峰值 1.75 GiB 远低于 16GB）。预计耗时：首次安装依赖与下载模型/数据集约 5–20 分钟（视网络），20 步训练通常 **< 1 分钟到数分钟**。主要风险是配额不确定、断连与临时存储丢失。

- **Colab Enterprise（免费版）**：Colab Enterprise 为 GCP 集成产品（[S1] 说明与 BigQuery/Vertex AI 集成、按 GCP 计费），不存在独立于 GCP 的“永久免费 GPU 笔记本”；GPU 使用受 GCP 计费与试用限制约束（见 2.4 GCP）。**无免费 GPU**。

### 2.2 Kaggle Notebooks

- **来源与抓取日期（2026-11-19）**：[S5] Kaggle Docs *Notebooks* https://www.kaggle.com/docs/notebooks ；[S6] Kaggle Docs *Efficient GPU Usage* https://www.kaggle.com/docs/efficient-gpu-usage ；[S7] fastero.com, *Kaggle Notebooks Free GPU* https://fastero.com/blog/kaggle-notebooks （2026-10-31）；[S8] Substack, *Kaggle’s Free GPUs for AI Experiments* https://nandigamharikrishna.substack.com/p/kaggles-free-gpus-for-ai-experiments （2026-11-05）；[S9] aimultiple.com, *Comparison of Top 6 Free Cloud GPU Services* https://aimultiple.com/free-cloud-gpu （页面标注 2026-11-06 上下文引用）。另参考 [Kaggle 讨论：启用互联网](https://www.kaggle.com/discussions/product-feedback/63544)。
- **GPU 型号与显存**：自 **2026-09-15** 起免费 GPU 机型为 **T4 ×2**（两张 NVIDIA T4，各 16GB 显存，**独立 GPU 非单卡 32GB**），4 CPU 核心、约 29–32GB RAM（[S8][S9]；[S6] 文档仍保留 P100 描述但标注配额机制，实际可选机型以控制台为准）。单张 T4 16GB **≥8GB**，足以运行 Qwen3-0.6B LoRA SFT（1787 MiB）；训练时应固定使用单卡（如 `CUDA_VISIBLE_DEVICES=0`）。H3/视频 LoRA：单卡显存对 4–8GB 训练步估算可能覆盖，但视频权重下载（H3 约数十至上百 GB）、缓存、musubi-tuner 依赖与 20GB 持久空间限制使真实训练**未核实/难度高**。
- **免费额度**：**约 30 GPU 小时/周**（配额每周重置，“30 hours or sometimes higher depending on demand and resources” [S6]；[S7] 同述并提及 TPU 20 小时/周）。会话时长：CPU/GPU **≤12 小时**，TPU **≤9 小时**（[S5][S7]）。交互会话空闲约 **60 分钟**超时 [S6]。
- **信用卡/支付方式**：免费使用通常**不需要信用卡**；需 Kaggle 账号并完成手机验证以启用加速器与互联网访问（常见要求，[S5] 流程相关）。
- **存储与持久化**：`/kaggle/working` 最多 **20 GB** 可在会话间保存（[S5]）；`/kaggle/input` 为只读挂载数据集；其余磁盘为会话临时空间，结束后丢失。模型与数据集可作为 Kaggle Dataset 挂载或在联网时下载到 working（需注意 20GB 上限）。
- **环境与 CUDA/torch**：Linux 镜像，预装 CUDA 与常见 ML 库（版本随镜像更新，**未核实具体 CUDA 版本**）；可 `pip install`/使用 `uv` 安装依赖与指定 PyTorch wheel（cu121/cu128/cu132 等取决于驱动）。Python 3.12、uv、Node 22+ 可自行安装。bitsandbytes 在 Linux 通常可用，但本验收路径为 LoRA（非 QLoRA）可不依赖。
- **网络**：可在 Notebook 设置中启用互联网访问（[Kaggle 反馈帖](https://www.kaggle.com/discussions/product-feedback/63544)），启用后通常可直连 Hugging Face 下载模型/数据集；**受网络环境影响**。
- **结论**：**推荐作为免费首选**。可 clone 本仓库并在单张 T4 上跑通 Qwen3-0.6B LoRA SFT 10–20 步验收。预计耗时：首次环境准备与下载 10–25 分钟，20 步训练 **数分钟内**（与本地 8GB 卡量级相当或更快，视 CPU/IO 而定）。需注意 20GB 持久空间与单卡选择。

### 2.3 Google AI Studio 免费代码执行

- **来源与抓取日期（2026-11-19）**：[S10] Google AI for Developers, *Code execution* https://aistudio.google.com/docs/code-execution （抓取内容显示 Gemini API 代码执行：Python 沙箱，30 秒上限，固定库列表，不可安装自定义库）。
- **GPU 型号与显存**：**无 GPU**，为模型侧 Python 沙箱执行环境（CPU）。
- **免费额度**：功能本身无单独 GPU 额度概念；文档说明代码执行环境**最长运行 30 秒**、不可安装自有库、文件输入受模型上下文窗口限制。
- **信用卡/支付方式**：使用 Google AI Studio/Gemini API 需 Google 账号；代码执行功能按文档计费规则与 token 相关（“no additional charge for enabling code execution…”），非独立算力订阅。
- **存储与持久化**：无持久训练工作区；输出以内联形式返回。
- **环境与 CUDA/torch**：支持库包含 numpy/pandas/scipy/scikit-learn/tensorflow/opencv/matplotlib 等，**不包含 torch**，且不可 `pip install`，无法安装 `uv`/项目依赖栈；无法运行 SuperTuner engine/server。
- **网络**：沙箱受限，非通用可联网训练环境（**未核实**外网访问细节，但与训练无关）。
- **结论**：**不能**运行本项目或完成 Qwen3-0.6B LoRA SFT 验收，仅可作为“免费代码执行”能力的反例记录。

### 2.4 GCP / AWS / Azure / Oracle Cloud 免费层 GPU

- **GCP（Google Cloud）**  
  - 来源（2026-11-19）：[S11] *Free Google Cloud features and trial offer* https://cloud.google.com/free/docs/free-cloud-features （抓取确认：$300/90 天试用；“During the Free Trial … (non-billable account) you can’t Add GPUs to your VM instances”；Always Free 含 e2-micro 等 CPU 资源）。[S12] DataStorage, *The 2026 Guide to Cloud Free Tiers and Credits* https://datastorage.com/articles/the-2026-guide-to-cloud-free-tiers-and-credits-compute-storage-and-gpu/ （2026-09-23，称 GPU/TPU 不在 Free Tier）。  
  - GPU/额度：**Always Free 无 GPU/TPU**；免费试用金 $300/90 天需支付方式验证，但试用账户阶段禁止为 VM 添加 GPU。升级为付费账户后可购买 GPU，但已超出“免费 GPU”范畴。  
  - 存储/网络/CUDA：付费 GPU VM 可自选镜像与 CUDA，理论上可安装 uv+Python3.12+torch cu132 并直连 HF，**本项针对免费层结论为不可行**。  
  - 结论：**无可用于训练的免费 GPU**。

- **AWS**  
  - 来源（2026-11-19）：[S13] AWS Free Tier https://aws.amazon.com/free/ （新账户 $100 即时额度 + 最高再 $100）；[S14] *Track your Free Tier usage for Amazon EC2* https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html ；[S15] akalcloud.ai, *AWS free tier changes* https://akalcloud.ai/blog/aws-free-tier-changes/ （2026-07-18，称 2025-07-15 起新账户以额度替代 12 个月传统模式）。补充核实：g4dn/g5 为付费 GPU 实例族（https://aws.amazon.com/ec2/instance-types/g4/、https://aws.amazon.com/ec2/instance-types/g5/）。  
  - GPU/额度：**无 always-free GPU**；历史上部分账户曾有 g4ad 等促销小时现状**未核实/不作为可依赖免费路径**。额度可抵扣付费 GPU 实例费用但需开通与付费方式。  
  - 结论：**无免费 GPU**（可用额度≠免费 GPU 资源）。

- **Azure**  
  - 来源（2026-11-19）：[S16] Microsoft Q&A *Azure Free Tier* https://learn.microsoft.com/en-us/answers/questions/1191439/azure-free-tier （提及 B1s 12 个月免费、$200 30 天额度，历史答复）；Azure 官方免费页结构随时间调整，NC/ND/NV 系列为付费 GPU VM（https://azure.microsoft.com/en-us/pricing/details/virtual-machines/windows/ 等产品页可佐证）。  
  - GPU/额度：**无 always-free GPU**；$200/30 天信用额度可用于付费资源，需支付方式。**2026 年是否新增免费 GPU 未核实**。  
  - 结论：**免费层无 GPU**。

- **Oracle Cloud**  
  - 来源（2026-11-19）：[S17] Oracle Always Free Resources https://docs.oracle.com/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm ；[S18] space-node.net, *Oracle Always Free VPS Review 2026* https://space-node.net/blog/oracle-vps-free-tier-review-2026 （2026-04-07，明确 “No GPU”）。另有 [gmicloud.ai 2025-10-21](https://www.gmicloud.ai/en/blog/where-can-i-get-free-gpu-cloud-trials-in-2026-a-complete-guide) 称存在 A10 免费访问，与 [S18]/官方资源列表不一致，**GPU 可用性未核实，按无免费 GPU 处理**。  
  - GPU/额度：Always Free 为 2×E2.1.Micro 或 Ampere A1（合计 4 OCPU/24GB RAM）等 **CPU 资源**；$300/30 天试用需验证。  
  - 结论：**Always Free 无 GPU**，无法满足本项目 NVIDIA CUDA 训练需求。

### 2.5 RunPod / Vast.ai / Paperspace（付费低价对照）

- **RunPod**  
  - 来源（2026-11-19）：[S19] RunPod Pricing https://www.runpod.io/pricing ；[S20] RunPod H100 PCIe https://www.runpod.io/gpu-models/h100-pcie （Community $1.99/hr、Secure $2.89/hr，页面抓取时信息）；northflank.com *Runpod GPU pricing*（2025-12-07）给出 Serverless H100 约 $4.18–4.55/hr、A100 $2.72/hr 等参考。  
  - GPU/显存：按实例提供 T4/L4/A10/A100/H100 等，显存 16–80GB+ 视型号。  
  - 额度/信用卡：**无免费档**；按秒/按小时计费，需充值/支付方式。  
  - 存储：提供容器/网络卷等选项，持久化需配置卷（具体容量/价格 **未逐项核实**）。  
  - 环境：可选官方/自定义镜像，通常可安装 CUDA PyTorch、uv、Python 3.12 并运行 engine；HF 可直连（视区域）。  
  - 结论：**付费可行**，适合需要稳定 16GB+ 显存或更长训练；跑 Qwen3-0.6B LoRA SFT 20 步成本极低（分钟级），但非免费。

- **Vast.ai**  
  - 来源（2026-11-19）：[S21] Vast.ai Pricing https://vast.ai/pricing ；[S22] thundercompute.com, *Vast.ai GPU Pricing 2026* https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute （示例：RTX 4090 ~$0.27–0.54/hr、A100 80GB ~$1.28/hr，随市场波动）。  
  - GPU/显存：社区供给，RTX 3090/4090、A100、T4 等多样，显存依具体卡。  
  - 额度/信用卡：**无免费档**；按分钟计费，通常需最低充值（文档示例 $5，**当前门槛未核实**）。  
  - 存储：实例磁盘多为临时，持久化需额外卷/对象存储方案（**未核实**）。  
  - 环境：镜像可自选，Linux + NVIDIA 驱动/CUDA 视宿主；可安装 torch cu 系列 wheel 与 uv，但兼容性需按实例确认。  
  - 结论：**低价付费可行**，性价比高但稳定性/网络/数据安全需自行评估。

- **Paperspace（现 Gradient / DigitalOcean）**  
  - 来源（2026-11-19）：[S23] Paperspace/Gradient 官方站点 https://www.paperspace.com/gradient （定价页结构与具体免费 GPU/小时价本次未能可靠抓取到结构化数据）；[S4] FreakyJolly 汇总表提及 “Free M4000 in some workspaces, 6-hour auto-shutdown, 5 GB storage”（历史/社区信息，**当前是否仍提供未核实**）。  
  - GPU/额度：免费档机型与可用性**未核实**；付费档价格**未核实**（需以官网实时报价为准）。  
  - 结论：作为对照项保留；若存在免费 M4000（8GB）理论上接近 1787 MiB 门槛但环境老旧、存储极小，**不建议作为首选**，建议直接以官网当前信息复核。

### 2.6 Colab Pro / Pro+ 等其它付费/增强选项

- **Colab Pro / Pro+**：见 [S1][S3][S4]。Pro 提供 100 CU/月、Pro+ 提供 600 CU/月与后台执行（≤24h），可选 L4/A100/H100 等更强 GPU，适合需要稳定显存与更长会话的用户；仍受地区、CU 消耗与配额策略影响，**非免费**。
- **Lightning AI Studios**  
  - 来源（2026-11-19）：[S24] Lightning AI Pricing https://lightning.ai/pricing （“Start on the free tier with 50* free GPU hours. … Free Studios run 24/7 but require restart every 4 hours.”）；[S25] Lightning AI Docs *AI Studio* https://lightning.ai/docs/platform/build/ai-studio （提及手机验证可获额度、购买额外 GPU 小时）。第三方文章存在不同额度表述（如 22 GPU h/月），以官方页面为准但细节仍可能变动。  
  - GPU/显存：免费可用 GPU 型号随资源而变（常见 T4 等，**未固定保证**）；≥8GB 时可满足 1787 MiB。  
  - 额度/限制：标称 50 免费 GPU 小时起步；**每 4 小时需重启 Studio** 影响长任务；持久存储与并发限制**未完全核实**。  
  - 环境：类 VS Code/Jupyter 工作区，Linux，可安装 uv/Python 依赖与 torch wheel；HF 联网通常可行。  
  - 结论：**可作为免费/近似免费备选**跑通短验收，但需适应重启与配额。

- **Modal**  
  - 来源（2026-11-19）：[S26] Modal Pricing https://modal.com/pricing （“Start with $30/month in free compute”）。  
  - 性质：按代码定义的 serverless/函数式计算平台，非传统常驻 Notebook；每月 $30 免费计算额度可抵扣 CPU/GPU 秒级用量（示例折算见 [blaxel.ai 2026-02-06](https://blaxel.ai/blog/modal-pricing-alternatives-guide)：约可换算为数小时 T4，**具体随机型与计费变动**）。  
  - 结论：**近似免费**，能否“clone 本仓库并按原启动方式运行 server+web”需改造为脚本/任务形态；跑 engine 训练任务技术上可行但需额外工程，适合作为补充选项，**按本仓库原生工作流可行性未核实**。

### 2.7 其它免费 GPU notebook 平台

- **Hugging Face Spaces + ZeroGPU**  
  - 来源（2026-11-19）：[S27] Hugging Face Docs *Spaces ZeroGPU* https://huggingface.co/docs/hub/en/spaces-zerogpu （动态 GPU 分配，面向 Spaces 应用）。社区信息（如 Reddit r/LocalLLaMA 讨论）称免费账号约 **5 分钟 GPU/天**级别额度，**官方精确免费配额未在本次抓取页明确列出，未核实**。  
  - 结论：适合推理 Demo/短时推理，不适合运行本仓库多步训练与完整服务栈；**不能**作为跑通 LoRA SFT 验收的平台。

- **Amazon SageMaker Studio Lab**：历史上提供免费 GPU，但社区与汇总资料显示已对新用户关闭或限制（[S4] 表中标注 “Closed to new users”），**2026-11 当前可用性未核实**，不纳入可推荐路径。

## 3. 推荐平台与选择理由

1. **首选：Kaggle Notebooks** — 提供免费 T4（16GB）单卡可满足 1787 MiB 峰值；有**明确的周配额（~30 GPU h/周）**、12 小时会话与 20GB `/kaggle/working` 持久输出空间；可启用互联网直连 Hugging Face；无需信用卡（需手机验证）。对“clone 仓库 → 装依赖 → 下载小模型/数据集 → 跑 10–20 步”最匹配。  
2. **备选：Google Colab 免费版** — T4 16GB 易得、上手快，适合快速验证；但 GPU 配额与空闲策略不透明、VM 临时存储易丢失，需配合 Drive 持久化并接受断连风险。

> 付费补充：若需稳定 24GB+ 显存、长时间训练或视频工作流，可考虑 RunPod/Vast.ai 等低价按量实例（见 2.5），但不属于免费方案。

## 4. 在 Kaggle Notebooks 上跑通本仓库 Qwen3-0.6B LoRA SFT 的逐步 Runbook

> 目标：在 **Kaggle Notebook（GPU T4 x2，启用 Internet）** 上完成最小验收：`engine` 可计划并执行 **10–20 步** Qwen3-0.6B LoRA SFT（wikitext-2 或等价小数据集），产出 adapter/checkpoint 并记录峰值显存接近本地基线量级。  
> 前提：已注册 Kaggle、完成手机验证；在 Notebook Settings 中选择 **Accelerator = GPU T4 x2** 并打开 **Internet = On**。

| 步骤 | 操作 | 说明/命令示例（Linux Notebook） |
| --- | --- | --- |
| 1 | 创建 Notebook 并确认 GPU | 新建 Notebook，Settings 选 GPU T4 x2、Internet On。运行 `!nvidia-smi` 确认看到 T4；记录 CUDA Version（仅供选轮子参考）。 |
| 2 | 固定单卡（避免双卡歧义） | 在后续训练命令前设置 `import os; os.environ["CUDA_VISIBLE_DEVICES"]="0"` 或 shell `export CUDA_VISIBLE_DEVICES=0`，确保训练使用单张 16GB T4。 |
| 3 | Clone 仓库 | `!git clone <本仓库 URL> /kaggle/working/supertuner && cd /kaggle/working/supertuner`（将 `<本仓库 URL>` 替换为实际地址；`/kaggle/working` 内内容可保存，≤20GB）。 |
| 4 | 安装 Python 3.12 与 uv | Kaggle 镜像 Python 版本可能非 3.12。推荐：`!pip install -U uv` 或用官方安装脚本安装 uv；再 `!uv python install 3.12`（如需）。验证 `!uv --version`、`!python3 --version` 或 `!uv run python -V`。 |
| 5 | 安装 Node.js >= 22.5 与 pnpm（可选，跑 server+web 时需要） | 仅跑 engine 验收可跳过。否则用 nvm/fnm 或官方二进制安装 Node 22+，再 `corepack enable && corepack prepare pnpm@10 --activate`（版本按仓库要求）；`node -v` 应 >= 22.5。 |
| 6 | 同步引擎依赖 | 在仓库根目录：`!cd /kaggle/working/supertuner && uv sync --project engine --extra dev --extra torch`。这会创建 `engine/.venv` 并安装 schema/CLI/torch 相关 Python 包（不含 CUDA 专用 torch 轮子时可能为 CPU 版，下一步覆盖）。 |
| 7 | 安装 CUDA 版 PyTorch（关键） | 按驱动支持选择 wheel。优先尝试 README 推荐：`!uv pip install --python engine/.venv/bin/python "torch==2.15.0+cu132" --index-url https://download.pytorch.org/whl/cu132`。若驱动/CUDA 不匹配，改用 `cu121`/`cu124`/`cu128` 等对应索引（`https://download.pytorch.org/whl/cu121` 等）并选兼容 torch 版本。验证：`!uv run --project engine python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` 应为 True 且显示 Tesla T4。 |
| 8 | （可选）安装前端/服务端依赖 | 仅 engine 路径可跳过。若需 UI：`!cd /kaggle/working/supertuner && pnpm install`。注意 `/kaggle/working` 20GB 上限。 |
| 9 | 配置环境变量与目录 | 建议使用工作区内可持久路径：`export SUPERTUNER_HOME=/kaggle/working/.supertuner`，`export HF_HOME=/kaggle/working/.cache/huggingface`，`export HF_DATASETS_CACHE=$HF_HOME/datasets`，`export SUPERTUNER_ALLOW_ANY_PATH=1`（Notebook 场景下便于指定数据/输出路径；生产使用请按安全策略收紧）。Windows 专用的 `PYTORCH_CUDA_ALLOC_CONF` 在 Linux 通常不需要，engine 在 Linux 行为按实现为准。 |
| 10 | 准备模型与数据集 | 方案 A（推荐，最小）：使用 Hugging Face 上的 `Qwen/Qwen3-0.6B` 与 `wikitext`/`wikitext-2-raw-v1`，由 datasets/transformers 首次运行自动下载到 `$HF_HOME`（需联网）。方案 B：预先将模型/数据集上传为 Kaggle Dataset 挂载到 `/kaggle/input/...` 并在配置中指向绝对路径。 |
| 11 | 规划（plan）验证 | 用 engine CLI 做一次计划检查（示例，字段以实际 schema 为准）：准备 JSON 配置（name/family/stage/adapter/quant_base/seq_len/steps 等，指向 wikitext-2 与 Qwen3-0.6B），执行 `echo '<spec-json>' | uv run --project engine supertuner-engine plan --json` 或按 CLI 支持的子命令/参数形式。确认计划可行、预计显存与 1787 MiB 同量级。 |
| 12 | 执行 10–20 步 LoRA SFT（验收） | 通过 engine 的 run/train 命令（或 server API）启动短跑：设置 `max_steps=20`（或 10）、LoRA、bf16/fp16 按环境、batch/grad_accum 取保守值。确保输出目录在 `/kaggle/working` 下。记录控制台 loss 与步速。 |
| 13 | （可选）通过 server + web 运行 | 若已装 Node/pnpm：`pnpm dev` 启动 API（:8787），`pnpm dev:web` 启动 UI（:5173）；在 Kaggle 中需通过端口转发/代理访问（**平台对外暴露端口能力未核实**），通常不如 CLI 直接。 |
| 14 | 验收检查 | 满足：① 10–20 步正常完成无 OOM；② loss 有数值输出（可下降或波动但有限）；③ 产物目录生成 adapter/检查点（如 `adapter_model.safetensors` 等）；④ 峰值显存约 **1.5–2.5 GiB** 量级（与 1787 MiB 基线一致，允许波动）；⑤ `uv run --project engine supertuner-engine env --json` 可识别 CUDA GPU。 |
| 15 | 保存产物与清理 | 将 adapter、日志、`memory_report`/plan 等复制或保留在 `/kaggle/working`；注意总量 ≤20GB；结束前停止会话释放配额。 |

**Colab 备选差异**：步骤类似，但将持久路径改为 Google Drive 挂载点（如 `/content/drive/MyDrive/...`）；`/content` 本身在 VM 回收后清空；GPU 配额更不可预测，遇限额需等待恢复。

## 5. 风险与限制

| 类别 | 风险/限制 | 影响与缓解 |
| --- | --- | --- |
| 断连与时长 | Colab 免费版最长 ~12h 且空闲即回收；Kaggle GPU/CPU 12h、交互 60 分钟空闲超时；Lightning AI 免费 Studio 每 4 小时需重启 | 长任务可能中断；验收仅需 10–20 步影响小。缓解：缩短步数、开启检查点、将输出写入持久目录（Kaggle `/kaggle/working`、Colab Drive）。 |
| 存储与持久化 | Kaggle `/kaggle/working` 仅 20GB；Colab `/content` 临时；HF/H3 权重体积大（H3 下载量级可达百 GB） | 大模型/视频缓存易超限。缓解：仅做 Qwen3-0.6B 小模型验收；清理缓存；使用外部数据集挂载。 |
| CUDA / 驱动 / torch 轮子 | 本仓库 README 推荐 `torch==2.15.0+cu132`（CUDA 13.2），但云端镜像驱动支持的 CUDA 上限各异；安装不匹配 wheel 会导致 `torch.cuda.is_available()==False` 或运行失败 | 缓解：先 `nvidia-smi` 查看驱动 CUDA 版本，选择兼容的 cu118/cu121/cu124/cu128/cu132 轮子；必要时降级/升级 torch 小版本。**各平台 cu132 可用性未逐一实测**。 |
| Python / uv / Node 版本 | engine 要求 Python 3.10–3.13；server 要求 Node >= 22.5；镜像预装版本可能不符 | 缓解：用 `uv python install 3.12`、nvm/fnm 安装 Node 22+；仅跑 engine 可不装 Node。 |
| 网络与代理 | Hugging Face 模型/数据集下载、PyPI/apt 访问受地区与平台策略影响；部分环境可能限速或受限 | 缓解：提前上传数据集为平台数据集、配置 HF 镜像/缓存（如可用）、重试下载；无法直连时需代理但多数 notebook 环境配置代理受限（**未核实各平台代理支持**）。 |
| 多卡与设备选择 | Kaggle 免费提供 T4×2（两张独立卡），未指定时可能选错设备或不支持多卡训练逻辑 | 缓解：设置 `CUDA_VISIBLE_DEVICES=0`，按单卡运行。 |
| bitsandbytes / QLoRA | Windows 本地未装 bnb；Linux 可装但 NF4/QLoRA 路径增加依赖与显存行为差异 | 本次验收目标为 LoRA（非 QLoRA），可规避；QLoRA 可行性**未在免费平台实测**。 |
| 视频 LoRA（H3/Wan） | musubi-tuner 未安装、权重与缓存体积巨大、H3 许可地域限制、8–16GB 显存对真实训练不友好 | 本仓库对视频路径定位为编译预览/未验证；免费 notebook 平台跑通 H3 视频 LoRA **不作为可交付承诺**。 |
| 路径与沙箱限制 | server 默认限制数据/输出路径在允许根目录内；Kaggle/Colab 路径与本地不同 | 缓解：设置 `SUPERTUNER_HOME`、`SUPERTUNER_ALLOW_ANY_PATH=1`（受控环境下）或正确配置 allowed roots；使用绝对路径。 |
| 信息时效性 | GPU 配额、机型、免费额度随平台策略频繁调整（Colab 动态、Kaggle 2026-09 机型已变更） | 文档结论以 2026-11-19 可查证来源为准，执行前应再次核对平台官网。

## 6. 来源清单

| 编号 | 来源 | URL | 抓取/参考日期 |
| --- | --- | --- | --- |
| S1 | Google Colab Paid Services Pricing / FAQ（官方） | https://colab.research.google.com/signup | 2026-11-19 |
| S2 | Google Colaboratory FAQ（官方） | https://research.google.com/colaboratory/intl/en-GB/faq.html | 2026-11-19 |
| S3 | Fastero — Google Colab GPU: Which GPU You Get, How to Turn It On, and What Each Costs (2026) | https://fastero.com/blog/google-colab-gpu | 2026-11-19 抓取（发布 2026-11-11） |
| S4 | FreakyJolly — Google Colab Free GPU Limits (Nov 2026) | https://freakyjolly.com/google-colab-free-gpu-limits/ | 2026-11-19 抓取（发布 2026-11-06） |
| S5 | Kaggle Docs — Notebooks（官方） | https://www.kaggle.com/docs/notebooks | 2026-11-19 |
| S6 | Kaggle Docs — Efficient GPU Usage（官方） | https://www.kaggle.com/docs/efficient-gpu-usage | 2026-11-19 |
| S7 | Fastero — Kaggle Notebooks Free GPU: 30 Hours a Week… (2026) | https://fastero.com/blog/kaggle-notebooks | 2026-11-19 抓取（发布 2026-10-31） |
| S8 | Substack — Kaggle’s Free GPUs for AI Experiments: What You Actually Get | https://nandigamharikrishna.substack.com/p/kaggles-free-gpus-for-ai-experiments | 2026-11-19 抓取（发布 2026-11-05） |
| S9 | AIMultiple — Comparison of Top 6 Free Cloud GPU Services | https://aimultiple.com/free-cloud-gpu | 2026-11-19 |
| S10 | Google AI for Developers — Code execution（官方） | https://aistudio.google.com/docs/code-execution | 2026-11-19 |
| S11 | Google Cloud — Free Trial and Free Tier features（官方） | https://cloud.google.com/free/docs/free-cloud-features | 2026-11-19 |
| S12 | DataStorage — The 2026 Guide to Cloud Free Tiers and Credits | https://datastorage.com/articles/the-2026-guide-to-cloud-free-tiers-and-credits-compute-storage-and-gpu/ | 2026-11-19 抓取（发布 2026-09-23） |
| S13 | AWS Free Tier（官方） | https://aws.amazon.com/free/ | 2026-11-19 |
| S14 | AWS EC2 — Track your Free Tier usage（官方文档） | https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html | 2026-11-19 |
| S15 | Akalcloud — AWS free tier changes: what you get (2026) | https://akalcloud.ai/blog/aws-free-tier-changes/ | 2026-11-19 抓取（发布 2026-07-18） |
| S16 | Microsoft Learn Q&A — Azure Free Tier | https://learn.microsoft.com/en-us/answers/questions/1191439/azure-free-tier | 2026-11-19 抓取（历史答复 2023-03-20，2026 现状部分未核实） |
| S17 | Oracle — Always Free Resources（官方文档） | https://docs.oracle.com/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm | 2026-11-19 |
| S18 | Space-Node — Oracle Always Free VPS Review 2026 | https://space-node.net/blog/oracle-vps-free-tier-review-2026 | 2026-11-19 抓取（发布 2026-04-07） |
| S19 | RunPod — GPU Cloud Pricing（官方） | https://www.runpod.io/pricing | 2026-11-19 |
| S20 | RunPod — H100 PCIe GPU Rental（官方） | https://www.runpod.io/gpu-models/h100-pcie | 2026-11-19 |
| S21 | Vast.ai — GPU Pricing（官方） | https://vast.ai/pricing | 2026-11-19 |
| S22 | Thunder Compute — Vast.ai GPU Pricing 2026 对比 | https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute | 2026-11-19 |
| S23 | Paperspace / DigitalOcean Gradient（官方站点） | https://www.paperspace.com/gradient | 2026-11-19（具体免费档/小时价未核实） |
| S24 | Lightning AI — Pricing（官方） | https://lightning.ai/pricing | 2026-11-19 |
| S25 | Lightning AI Docs — AI Studio（官方） | https://lightning.ai/docs/platform/build/ai-studio | 2026-11-19 |
| S26 | Modal — Pricing（官方） | https://modal.com/pricing | 2026-11-19 |
| S27 | Hugging Face Docs — Spaces ZeroGPU（官方） | https://huggingface.co/docs/hub/en/spaces-zerogpu | 2026-11-19 |
