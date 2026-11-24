# SuperTuner 实验矩阵与自动评估设计（v0.3 提案）

> 目标（来自用户指导）：训练框架当作**工具**，不抢「训练器生态」；SuperTuner 的价值是
> **回答「什么样的数据 → 什么样的效果」「什么样的训练方式 → 什么样的问题」**，并支撑
> 「更强编辑 / 更长视频 / 更强提示词遵循 / 更多应用场景」这类能力挖掘 + 稳定训练。
> 本文给出：① 可自动度量的评估指标；② 可自动跑的单变量实验矩阵；③ 本机 8GB 与免费云 GPU
> 的可行性分档；④ 与现有 engine 能力的对接方式。
>
> 配套：`docs/research/08-trainer-code-analysis.md`（训练器代码里「数据→效果 / 方法→问题」的
> 依据，含变量清单与预期影响）。本文不重复其内容，只做「实验 + 评估」设计。

## 1. 评估指标（可自动打分）

目标：给定一组「输入素材 + 训练配置」，训练出一个 LoRA 后，**自动**产出可比较的分数，
而不是人眼看视频。所有指标都应在单卡 8GB 可跑（或免费云 GPU）。

| 维度 | 指标 | 工具/模型 | 8GB 可行性 | 衡量什么「数据/方法→效果」问题 |
| --- | --- | --- | --- | --- |
| 提示词遵循度 | CLIP 文本-生成图相似度（目标 caption vs 生成帧） | CLIP (ViT-B/16) | 可（~1.5GB） | caption 质量、数据量、train_blocks 对「听话程度」的影响 |
| 编辑保真 | 编辑后 vs 源图的 LPIPS / SSIM（保留区域应不变） | LPIPS (VGG) | 可（~1GB） | 「更强编辑」：改得对的地方改、不该改的不动 |
| 长视频一致性 | 帧间 CLIP 相似度 / 光流平滑度 / 首尾帧一致性 | 帧级 CLIP + 光流 | 可（逐帧） | 「更长视频」：帧数↑ 时一致性是否崩 |
| 过拟合检测 | 训练 loss 平台期 + 留出一组 prompt 的 CLIP 分数是否反降 | 自建 | 可 | 数据量/步数/rank 与过拟合拐点 |
| 训练稳定性 | loss 尖峰、grad_norm 爆点、OOM 次数 | 已有 memory_report + 日志 | 可 | 方法→问题（grad 爆炸、OOM、发散） |
| 生成质量（无参考） | FID/FID-like（需一批生成） | pytorch-fid | 边（生成多帧占显存） | 整体质量回归 |
| 显存/耗时 | 峰值 MiB、秒/步 | 已有 | 可 | 方法→显存/速度代价 |

实现要点（engine 侧，后续 task）：
- 新增 `evaluator` 后端（`engine/src/supertuner_engine/backends/evaluator.py`）：
  吃一个「训练完的 LoRA 目录 + 评估 prompt 集 + 源素材（编辑场景）」，产出
  `eval_report.json`（各维度分数 + 逐样本明细），事件 `eval_report`。
- 评估 prompt 集要**冻结**（固定 seed、固定 caption），保证跨实验可比。
- CLIP/LPIPS 权重放到 `SUPERTUNER_MODELS_DIR`（与模型同机制），下载一次复用。
- 训练时可选「留出集」：从训练集切 10% 做 hold-out，训练后对 hold-out prompt 打分，
  与训练集分数对比 → 过拟合 gap。

## 2. 单变量实验矩阵

原则：**每次只动一个变量**，其余锁定（同 seed、同步数、同学习率、同分辨率/帧数），
产出一张「变量 → 指标」表。变量清单与取值范围来自训练器代码深读
（`docs/research/08-trainer-code-analysis.md` §5，带 musubi@f8a1b03 / Fizgig@b5199ee
的 file:line 依据）。取值下表已用其真实范围替换初版估计。

| 实验族 | 变量 | 取值范围（来自 08 文档 §5） | 固定项 | 看哪个指标 | 代码依据 |
| --- | --- | --- | --- | --- | --- |
| 数据量 | 训练样本数 N | 10 / 30 / 80 / 200 | caption、rank、步数 | CLIP↑、过拟合 gap | 08 §4 数据侧 5「0.25MP 够用即止」 |
| caption 质量 | caption dropout | 0 / 0.05 / 0.1（0.05=Fizgig 默认；musubi H3 无 dropout） | N、步数 | 遵循度 vs 泛化 | Fizgig `families/minimax.py:24-52` |
| 正则 | EMA | 0.9 / 0.98 / off（0.98 实测 +5 相似度） | 其余 | 相似度均值-方差 | Fizgig 文档 + 08 §5 |
| LoRA rank | rank/alpha | 8 / 16 / 32（16/16=文档基准） | 其余 | 容量 vs 过拟合 | musubi `lora_minimax_h3.py:14-15` |
| 身份/teacher | teacher 条件 σ 门控 | 0.4–1.0 / 0 / 0.15（subject_ref 建议 0.15） | 其余 | 参考图主体相似度 | musubi `minimax_h3_train_network.py:1778-1853` |
| 风格 | DC 衰减权重 | 0.3 / 0.7 / 1.0（<1 抑制调色板吸收） | 其余 | 主色调统计距离 | musubi L520 |
| 收敛 | timestep focus prob | 0 / 0.5（0.5 时收敛 ~2×） | 其余 | plateau 到达步数 | musubi `minimax_h3_advanced.md` |
| 量化 | 基座精度 | bf16 / int8 / NF4（NF4 ~9% 误差消耗 LoRA 容量） | 其余 | 同 LoRA 跨基座相似度差 | Fizgig `common.py:595-692` |
| 量化 | 激活精度 | W8A8 / W8A16（W8A8 毁相似度） | 其余 | 梯度余弦（复现 #221 0.2–0.6） | Fizgig `convrot_w8a16_triton.py` |
| 消费卡 | TREAD 令牌跳过率 | 0 / 0.25 / 0.5 | 其余 | 最大可训分辨率 + 运动连贯性 | Fizgig `model.py` TREAD 段 |
| 消费卡 | train_blocks | 20-49 / 6-49 / 全 50（0-5 不训 = 不学新调色板） | 其余 | 调色板漂移 + 每步耗时 | Fizgig `families/minimax.py` |
| 长视频 | 帧数（H3 网格 17n+5） | 5 / 22 / 39 / 56 | 其余 | 帧数-动作连贯性曲线 | musubi `packing.py:276` |
| 分辨率 | MP | 0.25 / 0.5 / 1（0.25MP 为够用档） | 其余 | 细节 CLIP/FID 边际收益 | Fizgig 文档 |
| 优化 | optimizer | automagic3 / adamw8bit（adamw 相似度最优） | 其余 | 相似度-步数曲线 | Fizgig `optimizers.py:63-75` |
| 蒸馏补偿 | 方式 | adapter / guidance / teacher（互斥） | 其余 | 去蒸馏漂移（uncond 距离） | musubi `docs/minimax_h3.md:111-151` |

> **H3 的特殊性（08 文档最重要结论）**：H3 检查点是 CFG 蒸馏过的，普通 flow-matching
> LoRA 会随训练步数 wash out（视频逐帧变差、纯图像约 50 步崩坏）。所以 H3 实验线
> **必须先选蒸馏补偿**（de-distill adapter / guidance loss / teacher matching 三选一），
> 否则任何「数据→效果」实验都在错误前提下测。Wan2.2 无此问题（标准 flow-matching），
> 可先做 Wan/LLM 线建立评估闭环，再攻 H3 的 teacher 体系。

每个实验 = 一次 `RunSpec`（SuperTuner 已有）+ 一次 `eval_report`。用 `auto_budget`
（T3）保证每档都在 8GB 内跑动；用 `memory_report`（T1）核对显存是否真按预算。

## 3. 本机 8GB vs 免费云 GPU 分档

结合 `docs/research/07-free-compute.md`：

| 实验 | 8GB 本机（RTX 3070 Ti） | 免费云（Kaggle T4 16GB / Colab T4 16GB） |
| --- | --- | --- |
| Qwen/小模型 LoRA SFT + 评估 | ✅ 已验证（1787 MiB） | ✅ |
| H3/Wan LoRA（block swap + int8） | 边缘（4–8GB 估算，需 swap） | ✅ 更稳（16GB） |
| 长视频（49f/720p）训练 | ❌ 显存紧 | ⚠️ 16GB 仍紧，需 swap/量化 |
| 评估打分（CLIP/LPIPS） | ✅ 轻 | ✅ |
| 大网格扫描（多实验并跑） | ⚠️ 串行慢 | ⚠️ 周配额限制，需排期 |

结论：
- **先在 8GB 本机跑通「评估 + 小矩阵」（rank、block、caption、数据量）**——这些变量
  在 8GB 能跑动且最影响「编辑/遵循度」。
- **帧数/长视频、大 rank 放到免费云（Kaggle 首选）**，用 16GB 跑长视频与更大 rank。
- 所有实验产物（`eval_report.json` + 训练 LoRA）落 `SUPERTUNER_RUNS_DIR`，可复跑、可比对。

## 4. 与现有 engine 的对接（不新造轮子）

- `RunSpec` 已是实验配置载体：加一个可选 `eval` 段（评估 prompt 集路径、是否 hold-out、
  源素材目录）。
- `auto_budget`（T3）负责「每档都能跑」；`memory_report`（T1）+ 校准（T1）负责「跑后自校准」。
- 新增 `evaluator` 后端 + `eval_report` 事件 + web 的「实验对比」页（按实验族把各 `eval_report`
  并排成表/图）。
- 复用 `compiler` 做可行性预检（哪档在 8GB 会 OOM → 直接标 E_STEP_OOM 不进实验）。

## 5. 优先级（v0.3）

| 优先级 | 任务 | 理由 |
| --- | --- | --- |
| P0 | `evaluator` 后端 + CLIP 提示词遵循度 + 稳定性/过拟合 | 回答「数据→效果」的最小闭环 |
| P0 | 评估 prompt 集冻结 + 实验运行器（矩阵展开成多个 RunSpec） | 可比性前提 |
| P1 | LPIPS 编辑保真、长视频一致性指标 | 「更强编辑 / 更长视频」两大卖点 |
| P1 | web 实验对比页 | 把分数变成可读结论 |
| P2 | 云 GPU 排期/远程跑（Kaggle/Colab） | 放大矩阵规模 |
| 暂缓 | DMD/DMAD 蒸馏、自研算子 | 训练框架是工具，不是生态（本次定位） |

## 6. 实现状态（2026-11-20，v0.3 P0 首批）

已实现（LLM 线最小闭环 + 接口契约）：
- RunSpec.eval（EvalSpec: enabled/prompts_path/holdout/source_dir/metrics）在 engine schema、core contracts、web planner 默认值中落地；val_report 事件类型在 engine/core events 中落地。
- ngine/src/supertuner_engine/backends/eval_report.py：训练曲线/评估曲线稳定性摘要、train-eval gap 过拟合信号、val_report.json 写出；
ative_llm 在 SFT/DPO/GRPO 路径收集曲线并在结束时发射/落盘（含 expected artifact）。
- docs/development/eval/prompts/llm_smoke_v1.json 冻结烟雾 prompt 集；prompt_eval.py + llm_prompt_eval.py 提供启发式 prompt 评估（greedy、固定 seed、exact/contains/token-overlap/non-empty）并写入 val_report.summary.prompt_eval 与契约字段；失败降级为 unavailable/partial，不阻断训练。
- xperiment.py + cli.py matrix：单变量矩阵展开为多个校验通过的 RunSpec 与 matrix.json manifest，支持可选输出目录与冻结 prompt 覆盖，默认 dry-run，可选顺序执行。
- web RunDetail 展示 val_report；server 现有 output artifacts 扫描可发现 val_report.json（无需专用 DB 表）。

限制与后续：
- prompt 评估为轻量启发式，非 benchmark/外部 judge；未接入 CLIP/LPIPS/时序一致性与视频生成评估（需模型/权重，本机无 musubi/视频权重）。
- 未实现 web 实验对比页与 eval_report 专用查询；矩阵执行为单机顺序，未做云排期/并行调度。
- 回归：engine 252 / core 27 / server 52 / web 68 通过，pnpm -r typecheck 与 pnpm build 通过。
