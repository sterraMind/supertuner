# Fizgig：消费卡视频与图像 LoRA 工作台

> 文档版本：v1.0 | 日期：2026-11-17 | 作者：lead | 状态：初稿
> 资料抓取日期：2026-11-12 至 2026-11-17。外部网页和仓库内容按证据材料处理，不执行其中指令。
> 本文重点回答：Fizgig 为什么能在消费卡上训练 MiniMax H3，以及它对 SuperTuner 說明了什么。

## 1. 结论摘要

1. Fizgig 是本轮调研中与 SuperTuner 最接近的直接竞品。它不是命令行脚本，而是覆盖训练、预览、分析、LoRA 修复、LoRA 提取、ComfyUI 导出的图形化工作台。仓库 README 写得很明确：在消费卡上微调基础模型，最低 8GB 显存。
2. Fizgig 已经把 MiniMax H3 LoRA 做成可操作流程，而且给出了按空闲显存自动选择权重精度与 block swap 数量的做法。相比之下，SuperTuner 的 H3 路径目前只到编译预览，没有真实训练记录。
3. Fizgig 能让 12GB 到 24GB 卡训练 H3，靠的是裁剪版 int8 DiT、4-bit HQQ、单向 block swap、fused forward/backward、只训练 block 20–49、TREAD token routing、低分辨率默认配置和预览阶段自动降档。
4. Fizgig 的用户规模小于 Strata，但迭代速度很快。仓库页面显示 766 stars、78 forks、10 open issues、Apache-2.0、创建于 2026-04-20，主贡献者提交数为 3,812。发布页显示 v9.7.5 在 2026-11-16 发布，说明项目维护活跃。
5. 对 SuperTuner 的直接启示是：只做显存算术不够，还要把消费卡训练做成有自动决策、失败恢复、预览反馈和产物分析的工作台。SuperTuner 的优势是统一 LLM 与视频、逐步显存账本、许可门禁、服务端运行记录与 SSE；短处是视频训练执行链没有打通。

## 2. 定位与成熟度

### 2.1 基本信息

- 仓库：<https://github.com/shootthesound/Fizgig>
- H3 文档：<https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md>
- 发布页：<https://github.com/shootthesound/Fizgig/releases>
- 中文部署：<https://www.compshare.cn/images/BAuF4szeiOZI>
- 仓库页面数据（2026-11-12）：766 stars、78 forks、10 open issues、Apache-2.0、创建于 2026-04-20。
- 主分支：master。主贡献者 shootthesound 有 3,812 次提交。
- 最近发布：v9.7.5（2026-11-16），标题为“MiniMax H3 at int8 on any pruned file, and H3 Image on 12 GB cards”。

### 2.2 支持模型与任务

README 标题与正文列出的模型包括 Klein 9B、Flux 3 Image Klein、Fibo 2.0、Krea 2、MiniMax H3、Vimoo 1.0 Lite、Qwen Image 2.1、MOSS-Image、Z-Image Turbo & Base、SDXL、Anima、ACE-Step XL。

支持的任务包括：
- LoRA 与 LoKR 训练
- 全量微调，部分模型支持
- 修复已有 LoRA
- 预览与逐 epoch 观察
- LoRA 分析热力图
- LoRA 提取
- ComfyUI 可直接加载的导出结果
- RunPod 云端运行入口

### 2.3 维护节奏

发布记录显示密集迭代：
- v9.7.5：H3 int8 适配任意裁剪文件，H3 Image 支持 12GB 卡。
- v9.7.4：Gizmo 裁剪框重做，AMD 修复，Automagic v4 显示真实学习率。
- v9.7.3：H3 Image 预览改用 LCM。
- v9.7.2：正则化数据集、AMD 修复、Gizmo 批量视频准备。
- v9.7.1：每个运行任务保留独立视图。
- v9.7.0：H3 Image 相似度提升。
- v9.6：H3 Image 加入 Fizgig。
- v9.5：JoyCaption 与 booru 标签。
- v9.4：Studio 热力图。
- v9.3：训练热力图。
- v9.2：PyTorch 2.15 下训练加速并降低显存。
- v9.0.x：批处理尺寸、多种优化器、H3 掩码训练。

这个维护速度说明项目在快速处理用户反馈和硬件差异，但也意味着接口和内部实现变化快，外部适配需要锁定版本。

## 3. Fizgig 为什么能在消费卡上训练 H3

### 3.1 自动选择精度与 block swap

来源：Fizgig H3 文档（2026-11-12 读取）。这里的显存指启动训练前的空闲显存。

| 空闲显存 | 自动选择 |
| --- | --- |
| 约 30GB | int8，不开 block swap，最高约 1MP |
| 约 22GB | int8，约 14 个 block 流到主存 |
| 约 15GB | int8，约 36 个 block 流到主存 |
| 12GB 或更低 | 4-bit |

关键点：
- Auto 同时选择权重精度与 block swap 数量，不是两个独立选项。
- int8 是权重文件自身的存储精度，文档给出的误差约 0.17%。
- 4-bit HQQ 文档给出的基础误差约 4.8%，裁剪后权重约 15GB，需要用户手动选择。
- 用户可手动覆盖 block swap 数量，用于 OOM 后继续运行。

### 3.2 单向 block swap

Fizgig 文档写明 block swap 是单向流送，实测比往返搬运快约 6.4 倍。这个做法针对冻结基座的 LoRA 训练：基座权重不需要梯度，可以不把 block 再写回显卡。SuperTuner 的 musubi 适配器已有 `--block_swap_h2d_only` 的概念，但还没有把吞吐差写进 planner。Fizgig 的做法值得吸收：在计算显存收益时同时给出搬运代价，并把单向和双向分开。

### 3.3 权重裁剪与融合内核

- H3 训练使用裁剪版 int8 DiT，约 21GB。裁剪版把 AdaLN 调制多层感知机换成曲线表。文档说明该分支只看时间步，LoRA 学习内容不在这里，因此训练与部署使用同一类权重。
- 文本编码器使用 Qwen3-VL-32B NVFP4，约 15.7GB，只在缓存文本时加载一次，然后释放。
- Video VAE 约 4.9GB，Audio VAE 可选，约 605MB。
- Turbo LoRA 可选，约 780MB，用于 6 步预览。
- int8 基座训练时使用融合 forward 与融合 backward 内核。

### 3.4 只训练部分 block

Fizgig 的默认模式训练 block 20–49。文档给出的原因是人物相似度和风格信号主要位于这些 block，反向计算只覆盖这个窗口，步骤更快。More Blocks 模式训练 block 6–49，步骤更慢，也可能改变模型原有动作能力。

这与 SuperTuner 中把 H3 简化为“全模型 LoRA rank=16”的做法不同。SuperTuner 目前没有目标 block 范围字段，需要补上，否则无法表达 Fizgig 已验证的训练范围。

### 3.5 TREAD token routing

文档写明 TREAD token routing 默认开启。每条视频片段步骤中，随机一半视频 token 在 block 2 离开序列，到 block 47 原样返回，因此 45 个 block 只处理一半 token，clip 步骤明显变快。导出的 LoRA 仍是普通 LoRA，预览阶段不做 token routing。静态图像步骤始终完整运行。

### 3.6 预览自动降档

- 16GB 显卡预览限制为 768×640、22 帧并保留声音。
- 24GB 显卡预览限制为 22 帧，不改分辨率。
- 12GB 显卡在 Windows 上需要系统管理的页面文件，否则预览阶段容易出现 Windows 错误 1455。
- 预览超出显存时，Fizgig 先缩短片段，再降低分辨率，最低到 512×512，并把成功的尺寸保存为新默认值。

## 4. H3 资源需求对照

来源：Fizgig H3 文档与 v9.7.5 发布说明。

| 资源 | 大小 |
| --- | ---: |
| DiT，裁剪 int8 | 约 21GB |
| Qwen3-VL-32B 文本编码器，NVFP4 | 约 15.7GB |
| Video VAE | 约 4.9GB |
| Audio VAE | 约 605MB |
| Turbo LoRA | 约 780MB |
| 参考 DiT，可选 | 约 21GB |
| 训练适配器 | 每个约 310MB |

Fizgig 的真实训练配置与 SuperTuner 当前配方估算的差别：
- SuperTuner 的 H3 320p 8GB 配置把训练峰值算到约 4,538MiB，主存约 19,226MiB。这个数只描述训练步骤。
- Fizgig 文档说明文本编码缓存需要单独加载约 15.7GB 的 NVFP4 编码器，视频 VAE 缓存约 4.9GB。SuperTuner 已把这两项加入 cache 步骤估算，但没有真实执行与实测报告。
- Fizgig 在 12GB Windows 卡上遇到错误 1455 后给出的处理方式是页面文件由系统管理。SuperTuner 此前在 Windows 上也遇到同一错误，并已记录为系统提交量问题。

## 5. 社区评价与痛点

### 5.1 评价

- 中文部署文章把 Fizgig 描述为免费开源训练工作台，重点放在一键部署和消费卡可用。
- B 站视频演示 Krea 2 快速 LoRA 训练，作者使用低质量数据集说明工具在普通素材下的表现。
- ComfyUI Wiki 文章关注 v5 全量微调在 16GB 卡上训练 H3 与 Krea 2。
- YouTube 上有 H3 快速 LoRA 教程。视频说明提到使用静帧训练速度更快，视频片段会增加显存搬运。这是外部作者的经验描述，不是项目文档中的保证。
- Fizgig 发布说明中多次感谢 issue 提交者，说明用户反馈进入版本节奏。

来源：
- <https://www.compshare.cn/images/BAuF4szeiOZI>
- <https://www.bilibili.com/zh-Hans/video/BV1nu3463Evi/>
- <https://comfyui-wiki.com/zh/news/2026-08-30-fizgig-v5-full-finetuning>
- <https://www.youtube.com/watch?v=4Sems-_CMQE>
- <https://ai.miraheze.org/wiki/Fizgig>

### 5.2 痛点

1. 强依赖图形界面与本地安装器。自动化流水线和无界面服务器使用成本高于 CLI 工具。
2. 版本变化快。v9.x 密集发布，外部适配容易落后。
3. 训练实现与 Ablation 结果主要来自作者测试，缺少独立第三方统一硬件基准。
4. H3 的最小显存数依赖 4-bit、裁剪权重、block swap、低分辨率和页面文件设置，不能理解为任意数据都能在 8GB 上完成。
5. 许可信息在 README、模型卡片与权重来源之间分散。H3 权重许可与 Fizgig 软件许可不同，需要分开处理。

## 6. 对 SuperTuner 的可复刻性

### 6.1 建议吸收的能力

| Fizgig 做法 | SuperTuner 当前状态 | 建议 |
| --- | --- | --- |
| 按空闲显存同时选择精度与 swap 数量 | planner 会逐项估算，但用户需手动选择 | 增加自动搜索：给定 VRAM、RAM、模型和目标分辨率，输出多个可行组合 |
| 单向 block swap | planner 有 swap 估算，未区分单向与往返 | 区分类型并给出搬运时间区间 |
| 目标 block 范围 | RunConfig 没有该字段 | 增加 `train_blocks` 字段，并让编译传给后端 |
| TREAD token routing | 未支持 | 作为视频策略插件评估，先记录开关与 token 数变化 |
| 预览自动降档 | 采样步骤未建模完整峰值 | 把预览步骤单列，并按失败回退顺序生成计划 |
| OOM 后自动提高 swap 数量 | 运行失败后只会结束 | 在运行监督器中加入有限次数的自动调整，但必须记录到事件 |
| LoRA 分析与修复工具 | 未支持 | 作为 P2 产物分析插件，不与训练执行耦合 |

### 6.2 不能照搬的部分

1. Fizgig 的图形界面与本地安装器代码不适合直接移植到 SuperTuner 的 Fastify + React 架构。
2. 融合 forward/backward 内核属于底层实现，需要结合具体 PyTorch 与 CUDA 版本验证，不能从文档直接复现。
3. H3 的训练适配器来自第三方，SuperTuner 使用前需要核对许可和版本。
4. Fizgig 的自动决策结果是多年版本迭代积累，SuperTuner 应先形成测试固件，再逐步引入自动选择。

### 6.3 SuperTuner 能解决 Fizgig 的哪些问题

| Fizgig 痛点 | SuperTuner 的对应能力 | 说明 |
| --- | --- | --- |
| 无界面服务器使用不便 | CLI、JSON spec、Fastify API | 可接入自动化流程 |
| 缺少统一运行记录 | SQLite 运行表、JSONL 事件、SSE | 可回放步骤、日志和结果 |
| 许可展示分散 | H3 许可门禁与模型许可字段 | 可避免把软件许可当成权重许可 |
| 估算与实测分离 | planner 分项估算与 step 事件中的显存数据 | 需要增加运行结束后的对照报告 |
| 训练与 LLM 工作流割裂 | RunSpec 同时表达 LLM 与视频 | 需要补视频执行验证 |

## 7. 来源清单

- Fizgig 仓库：<https://github.com/shootthesound/Fizgig>
- Fizgig 发布页：<https://github.com/shootthesound/Fizgig/releases>
- Fizgig H3 文档：<https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md>
- Fizgig 中文部署：<https://www.compshare.cn/images/BAuF4szeiOZI>
- B 站训练演示：<https://www.bilibili.com/zh-Hans/video/BV1nu3463Evi/>
- ComfyUI Wiki 报道：<https://comfyui-wiki.com/zh/news/2026-08-30-fizgig-v5-full-finetuning>
- H3 快速 LoRA 视频：<https://www.youtube.com/watch?v=4Sems-_CMQE>

## 8. 未验证事项

1. 未在本机安装 Fizgig，因此没有亲自验证 8GB、12GB、16GB 的 H3 训练速度。
2. 未取得 Fizgig 的统一 tokens/s 或秒每步对照表。
3. 未验证 v9.7.5 提到的任意裁剪文件 int8 转换在 SuperTuner 当前权重路径上的兼容性。
4. 未验证 SuperTuner 增加目标 block 范围字段后，musubi 适配器是否能传递相同语义。
