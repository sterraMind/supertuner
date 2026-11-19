# H3/视频 LoRA 消费卡训练需求市场调研（2026-11-18）

> 回答的问题：市场买「训练器」还是买「规划器」？「消费卡训 MiniMax H3」是不是真需求？
> 所有外部来源按不可信数据处理，仅引证。

## 1. 需求侧：消费卡训视频 LoRA 是已被验证的活需求

### 1.1 MiniMax H3 LoRA（12–24GB 卡，社区已在真实训练）

| 来源 | 事实 |
| --- | --- |
| musubi-tuner GitHub Discussion #1061（2026-08） | 「MiniMax H3 Ref2VA-Omni LoRA on 20GB/48GB GPUs: Windows setup, cache verification, and block-swap workflow」——RTX 3080 20GB 1 步冒烟、48GB 完整流程；社区在系统性摸索 H3 LoRA 训练 |
| note.com かみもと（2026-08，日文） | 「[VRAM 12GB] musubi-tuner で H3 t2v キャラ LoRA を作成」——RTX 4070 12GB 实际完成 H3 角色 LoRA，用 musubi dev 分支 |
| sokugame.com（2026-08，日文） | 「H3 用 LoRA 单图快速制作（VRAM 16GB〜）」——musubi dev #1057 的 One-Frame T2 流程 |
| omnichar.org | 「H3 LoRA on 24GB GPU, stills or clips」——实测 12 步/rank16/batch1/梯度检查点的峰值；关键结论：**峰值在第一个训练步之前就已确定**（加载+缓存阶段），16GB 卡基本无望 |
| github.com/lcdrzxlcdrzx/minimax-h3-lora-runbook | 个人 runbook：20GB 档「rank 16 + 梯度检查点 + 高 block swap，先跑 1 步」——社区在自建配方文档 |
| DreamWeebs 教程（2026-09） | H3 动漫风 LoRA 教程 + 配套云镜像——教程/镜像生态开始出现 |

### 1.2 Wan 2.x LoRA（12–16GB 卡，量更大的基本盘）

| 来源 | 事实 |
| --- | --- |
| Reddit（2026-06 快照） | 「Training wan 2.2 action loras on 16 GB VRAM 64 GB RAM: managed not to OOM on rank 16 but is it realistic?」——16GB 用户在贴结果，但**不确定自己这样训是否靠谱**（需求+焦虑并存） |
| SeaArt 模型页 | 「Wan2.2 16GB Training」——4070 Ti Super 16GB 训 Wan2.2 LoRA 作为技术展示发布 |
| HuggingFace 数据集（wan22_lora_training.md） | 实操配方外流：rank 32/alpha 16、LR≈2e-4、16–24GB batch 1–2——社区在用文档互相教 |
| wan27.org 博客（2026-06） | 「Wan 2.2 LoRA Training Guide … 12GB VRAM Settings」——12GB 档参数表成体系 |
| DiffSynth-Studio issue #1016 | 「有办法降低训练 wan2.2 animate LoRA 的显存要求吗」——需求方在问「怎么降显存」，官方未给答案 |
| YouTube「Wan 2.2 Lora 训练保姆级教程！低显存也能高效训练」 | 中文需求面直接点名痛点：「**它没有 UI 界面**」 |

### 1.3 供给侧在快速拥挤，但都是「工具」而非「产品」

- musubi-tuner：事实标准引擎（H3/Wan/Qwen-Image/Krea…），但 **CLI、无官方 UI**，H3 支持仍在 dev 分支；仓库无 LICENSE 文件。
- Fizgig：最接近的 GUI（musubi 包装），宣称 8GB 起、H3 12–16GB 档、自动精度+换块、预览降档、ComfyUI 导出；迭代极快（v9.7.x 周更级）。
- ai-toolkit：PR #829 正在做「8 种 LoRA 训练架构的双 5090 模型并行」——「dual RTX 5090 is the best consumer GPU setup」，消费卡多卡路线被认真推进。
- DiffSynth-Studio / OneTrainer(AGPL) / Unsloth Studio(beta)：各有覆盖，无一家把「跑前可行性」做进主流程。

## 2. 需求方真实痛点（按出现频率）

1. **「不知道会不会爆」——OOM 试错是默认成本**：Reddit「not OOM … but is it realistic?」、runbook「先跑 1 步」、omnichar「峰值在第一个训练步之前就定了」。没人提前告诉你 12GB 够不够、差多少、怎么改才够。
2. **CLI only / 无官方 UI**：Wan 中文教程直接点名；H3 流程（缓存文本编码器→缓存 latents→block swap 参数→int8 权重选择）全是手工 CLI 组合。
3. **配方外流、互相抄**：rank/LR/batch/block swap 参数散落在 GitHub、note、YouTube、HF 数据集——没有权威来源，全是「我家机器实测」。
4. **缓存工作流复杂**：H3 要 Qwen3-VL 文本编码器输出 + 视频/音频 VAE 缓存，顺序错、文件错都会白等数小时；cache 验证（Discussion #1061 专门一节）。
5. **H3 许可没人管**：地域排除 EU/UK/韩/美、年收入 >2000 万美元需书面授权、禁止用输出改进其它模型——社区教程几乎不提，踩坑后才发现。
6. **dev 分支不稳**：H3 支持「正式并入前必须用 dev」，note 作者明说「未来更新后可能没法这样训了」。

## 3. 结论：市场买「训练器」，规划器是信任层，不是产品

- **用户为「训出来」付费**（LoRA/角色/风格是交付物），不会为「算显存」单独付费。纯规划器没有独立市场。
- **「消费卡训 H3」是真需求且已被验证**：12GB 4070 已能训 H3 角色 LoRA，20/24/48GB 社区流程成型——需求成立，且目前只能靠 CLI + 手工 runbook 满足。
- **我们的正确姿势 = 训练器为主体 + 规划器为信任层**：
  - 主卖点（headline）：「你的 12–24GB 卡能训 H3 / Wan LoRA，而且**跑之前**就告诉你够不够、怎么配」。
  - 信任层（护城河）：逐步骤显存预算 + 可行性判定 + 许可门禁 + 校准后的预估 + 可重放 run 记录。这是所有 CLI 工具都不做的事，也是痛点 1 的直接答案。
  - 竞品关系：Fizgig 抢的是「GUI 包装 musubi」；我们抢的是「**可校验**的训练工作室」，两者可共存，Fizgig 越活跃说明市场越热。
- **「市场爆」的前提**（按重要性）：
  1. 必须**真能执行**（H3 16–24GB 档、Wan 2.2 5B 12–16GB 档），编译预览撑不住 headline——这是当前最大缺口。
  2. 门槛要低：GUI + 配方 + 一键。
  3. 信任：许可门禁、跑前预算、跑后对照报告——已有，是差异化。

## 4. 对本仓库路线的修正建议

1. **优先级上移「视频真实执行」**：本机可验证的最小闭环是 Wan 1.3B LoRA（装 musubi + 下载权重，8GB 边缘可行）；H3 在本机只能验证编译侧 + 文本编码器缓存方案（text_encoder_blocks_to_swap 需对上游核实），完整 H3 执行验证需 12GB+ 卡或大缓存。
2. **G1 自校准保留**：它同时服务 LLM 与视频，是「预估可信」的证据链，且增量小。
3. **配方即 runbook**：把社区外流的参数表（§1.2 来源）做成带「本机实测」标注的配方，直接吃掉痛点 3。
4. 对外口径从「规划器」改为「**消费卡训练工作室：H3/Wan LoRA，跑前可校验**」。
