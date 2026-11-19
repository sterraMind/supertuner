# SuperTuner 架构评审 01

- 评审对象：`docs/ARCHITECTURE.md`、`docs/research/00-synthesis.md`、`01-market.md`、`02-feasibility.md`、`03-stack.md`、`README.md`；`engine/src/supertuner_engine/{schema,plan,compiler,runtime}.py`、`backends/{native_llm,musubi}.py`、`plugins.py`、`events.py`、`config.py`、`control.py`、`cli.py`；`packages/server/src/{routes,supervisor}.ts`、`packages/core/src/{contracts,plugins,recipes}.ts`（为核对契约与行为，另读了 `packages/web/src/pages/{Planner,RunDetail}.tsx`、`packages/web/src/lib/recipe-utils.ts`、`packages/server/src/config.ts`、`engine/tests/test_plan.py`）。
- 评审方式：只读。除本文档外未修改任何文件。除读码外做了三件可复现的验证：
  1. 用 `plan()` 重算 `packages/core/src/recipes.ts` 全部 9 个配方，逐项比对文件里写死的 `vram_mib` / `ram_mib`；
  2. 跑 `uv run --project engine pytest engine/tests -q`（**144 passed**）与 `pnpm test`（**server 48 passed、web 68 passed**，全绿）；
  3. 对关键假设做敏感性重算（Wan2.2-5B 的 VAE stride、H3 的 cache 模式、8GB 卡可用显存下修）。
  脚本在 `.drafts/review-arch/check_plan.py`，附录 A 给出复现命令。
- 文中「未验证」= 我没有在本机取得证据（musubi-tuner 未安装、bitsandbytes 未安装、无 24GB 以上显卡、未跑过任何视频训练）。「实测」= 本仓库文档或代码里已有的可核对数字。

---

## 1. 评审结论

### 1.1 能否交付 v0.1

**分两条路径给结论，不要合成一句话。**

| 路径 | 结论 | 依据 |
| --- | --- | --- |
| LLM SFT/LoRA/QLoRA | **可以按 v0.1-preview 交付（内部 dogfood 级）** | 唯一有端到端实测的路径：Qwen3-0.6B + wikitext-2 LoRA SFT 在 8GB 卡上跑通，产物 `adapter_model.safetensors` 40,422,168 B（`README.md:29-35`）；拒绝路径完备（`compiler.py:112-212`、`native_llm.py:244-343`）；计划→编译→监督→SSE→产物发现闭环在测试里被覆盖（server 48 tests 含 cancel/sse/runs）。 |
| 视频 LoRA（Wan/H3） | **不可交付，甚至不可称为“可执行”** | 三个独立事实叠加：① musubi-tuner 本机未安装，适配器从未对真实 CLI 跑过（`musubi.py:277-301` 只在探测仓库脚本，`README.md:40-41` 自认“Video training … not verified”）；② 5B 的 `--task` 值在文档层面就与上游不符（见 B1）；③ H3 的 8GB 证明性配置漏算了缓存步骤本身（见 B2），按 `plan.py` 判 feasible 的配方在真实机器上会在第一步 OOM。 |

**不建议的对外说法**：把“视频 LoRA 训练”和 LLM 并列写进 v0.1 能力范围。建议改成“视频为**编译预览**（compile-only）：生成 argv 与显存/主存账本，执行需要用户自备 musubi-tuner 与权重，v0.1 不承诺可跑通”。`README.md:21-23` 现在写的是“video LoRA through musubi-tuner (compile + supervise; needs 24GB+ or a reduced 320p H3 profile)”，把“需要 24GB+ 就能跑”写成了事实，但 `ARCHITECTURE.md:351` 与 `README.md:40-41` 又都承认“video training … not verified”——这处自我矛盾必须在发版前收敛。

### 1.2 能否对外宣称“单卡训练”

- **LLM：可以，但必须带三个限定**——(a) 单卡单进程；(b) 已验证的规模是 0.6B bf16 LoRA，不是 7B/8B；(c) **QLoRA 在本机根本没装 bitsandbytes**（`native_llm.py:108-116` 只给 info，`native_llm.py:303-320` 只发 `W_BITSANDBYTES_MISSING` 警告，到 `native_llm.py:516-520`（QLoRA）/`:1130-1131`（8-bit 优化器）才抛 RuntimeError），所以 8GB 那一行里真正被验证的只有 `adapter=lora`，`adapter=qlora` 全链路未验证——而且它是“编译通过、运行时必失败”。`00-synthesis.md:27` 写的“8GB → QLoRA 4B–8B”目前没有任何本地证据。
- **视频：不可以。** 宣称“单卡训练”意味着这条链路至少跑通过一次；本仓库没有任何一次视频训练记录，`engine/tests/` 里也没有 `test_smoke_gpu.py`（`ARCHITECTURE.md:253-254`、`README.md:107-111` 都引用了它，文件不存在，`-m gpu` 在一个测试都没有的情况下也会“通过”）。唯一的端到端证据是 README 里那段文字，不可在 CI 复跑。
  - 值得强调的是：**规范文档本身的口径是诚实的**。`ARCHITECTURE.md:290-292` 的阶段表把视频路径明确写成 “compiled and supervised through musubi” 与 “compile + supervise only; execution requires musubi-tuner installed and the weights present”，`:351` 也写着 “Not verified: any video training”。越过这条线的是 `README.md:43-51` 的真相表——它把 H3 320p（8GB）、Wan2.2 5B（16GB）写成硬件能力，而不是“编译预览”。修法不是改架构文档，是给真相表补一列状态（verified / compile-only / unverified），并让它由配方数据生成。
- **H3：绝对不可以**，而且不只是“未验证”：完整权重 66GB bf16、裁剪后 40GB bf16 / 21GB int8（`02-feasibility.md:46`），整套权重约 210GB（`recipes.ts:87`）；许可排除 EU/UK/韩/美、>$20M 营收需书面授权、禁止用输出改进其它模型（含蒸馏）（`02-feasibility.md:47-51`）；8GB 路径依赖一个本机没装、仓库无 LICENSE 文件的第三方 checkout（`README.md:133-135`）。`recipes.ts:78-115` 把它叫“proof point”，这个词对外会被读成“已证明”。

### 1.3 一句话依据

值得肯定的部分先说清楚：**planner 的数字不是手写的**。我把 9 个配方逐条重算，8 个与 `recipes.ts` 的 `vram_mib`/`ram_mib` **完全一致（0.0% 偏差）**，只有 `wan22-5b-lora-16gb` 差 +3.3%（8446 → 8727，见 4.1）。也就是说“配方 = planner 输出”这条契约是成立的，问题出在 planner 的**模型本身**（缺项与几何错误），不在配方抄写。这决定了 P0 应该修 planner 而不是重排配方。

---

## 2. 架构缺陷清单

严重度口径：**blocker** = 会让对外声明变成虚假、造成法律暴露，或让主推路径在真实机器上必然失败且无法靠改文案绕开；**major** = 会让用户做出错误决策、或让某条已声明能力不可靠；**minor** = 一致性/可维护性问题，短期不影响判断。

### 2.1 Blocker

| # | 位置 | 严重度 | 为什么是问题 | 建议修法 |
| --- | --- | --- | --- | --- |
| B1 | `engine/src/supertuner_engine/backends/musubi.py:55`（`"wan2.2-ti2v-5b": "ti2v-5B"`）、`musubi.py:44-49`、`packages/core/src/recipes.ts:173-208`、`README.md:48` | blocker | `02-feasibility.md:58` 自己写明：musubi `docs/wan.md` 的 `--task` 只有 `t2v-1.3B / t2v-14B / i2v-14B / t2i-14B / *-FC` 与 `t2v-A14B / i2v-A14B`，**没有 TI2V-5B**，5B 只能用 DiffSynth-Studio 或 diffusion-pipe。适配器却给 5B 编了 `ti2v-5B`，并把它列为可训练家族。结果是：用户在 16GB 上选“5B 甜点档”配方 → 计划显示 feasible、产物路径齐全 → 第一个 `cache-text` 步骤就因未知 `--task` 退出。这是“编译成功但永远不可执行”的最坏形态，比直接拒绝更伤信任。 | 二选一，不要含糊：① 删掉 5B 的 musubi 支持（从 `MUSUBI_FAMILIES`、`_TASK_BY_FAMILY`、配方、README 真相表同时移除），并给出 `E_BACKEND_UNAVAILABLE` + “5B 需要 DiffSynth-Studio/diffusion-pipe，v0.1 不支持”；② 若坚持保留，必须写成 compile-only 并在 `ExecutionPlan.notes` 与 UI 上标“未对上游 CLI 校验”。验收标准见 P0-1。 |
| B2 | `plan.py:282-313`（`cache` 含 TEXT 时直接把文本编码器记为“不占显存”）、`musubi.py:524-555`（cache-text 步骤）、`schema.py:155`（`text_encoder_params_b=32.0`）、`recipes.ts:88` | blocker | 计划只估 `train` 步骤的峰值，而 H3 的 `cache-text` 步骤必须在**同一张卡**上把 Qwen3-VL-32B 文本编码器载入显存：bf16 约 48GB（`02-feasibility.md:46` 口径；按 32B 全量参数理论值约 64GB）、int8 约 25GB。`plan.py:285-286` 因为“输出已缓存”就把它从账本里划掉，`musubi.py:538-539` 只把 `--text_encoder_blocks_to_swap` 传给该步骤、并不回填任何显存估算。于是 `h3-lora-320p-8gb` 被算成 3638 MiB 峰值、headroom 2954 MiB（我复算的结果），在 8GB 卡上标 feasible——但这条配方的第三步必然 OOM。`recipes.ts:88` 的 caveat 已经承认“Text-encoder caching needs a separate pass with more VRAM … it will not fit beside training on this card”，等于文档自认与 planner 结论矛盾。 | 让 `plan` 覆盖**所有步骤**：给 `PlanStep.estimated_mib`（字段已存在于 `schema.py:320`，目前永远为 None）填值，`RunPlan.feasible` 改为“所有步骤都可行”；`cache` 步骤按 `text_encoder_quant` + `text_encoder_blocks_to_swap` 走与训练同一条换块公式（`_swap_resident_fraction`，`plan.py:143-154`）。验收标准见 P0-2。 |
| B3 | `plan.py:395-399`（预算取 `cfg.vram_budget_mib` / `cfg.ram_budget_mib`）、`packages/core/src/contracts.ts:69`（默认 32768）、`packages/web/src/lib/planner-state.ts:42`、对照 `schema.py:378`（`EnvReport.ram_mib` 已探到但没人用） | blocker | 主存预算是用户滑块常数，不是机器事实。H3 320p 需要 19226 MiB 常驻主存（我复算），在一台 16GB 内存的机器上照样显示 `feasible=true`——因为比较对象是硬编码的 32768。这直接违反 `ARCHITECTURE.md:273` 的原话“The planner must never claim a run fits when RAM for block swap does not”。而 §13 Known risks（`ARCHITECTURE.md:301-317`）只列了 H3 权重体积、musubi CLI 不稳、PCIe 慢、Windows bnb、node:sqlite 与 H3 几何近似，**没有把“主存预算用常数”列为风险**。显存侧略好一点（`Planner.tsx:116-119` 会用 `gpu.vram_mib` 自动填），但用的是 **total 不是 free**，见 4.3。 | `Context` 增加 `machine_ram_mib` / `gpu_free_mib`（`env.py` 已有数据），server 在 `normalizeSpecPaths` 之后把这些值写进 spec；planner 对 `ram_budget_mib` 缺省取 `machine_ram − os_reserve(≈4GB)`，并显式出一条“系统内存实测 15.2GB，可用 11.2GB”的 MemoryLine。验收标准见 P0-3。 |
| B4 | `ARCHITECTURE.md:266-272`（要求 UI 展示 H3 许可与地域限制、server 把确认记录进 run spec）对照 `packages/core/src/contracts.ts:193-200`（RunSpec 无任何许可字段）、`packages/server/src/routes.ts:221-264`（`POST /api/runs` 无许可检查）、`musubi.py:101`（`license="Apache-2.0"`）、`packages/web/src/lib/recipe-utils.ts:104-111` + `pages/Recipes.tsx:262-264`（按插件 manifest 显示许可） | blocker | 许可门禁完全没有实现：没有任何字段、没有任何 422、没有任何确认记录。更糟的是**展示的许可是错的**：`licenceForFamily` 取第一个声明该 family 的 backend 的 `license`，而 musubi 插件的 `license` 是**工具自身的** Apache-2.0，于是 H3 配方卡会显示“Apache-2.0 via musubi-tuner”——与 H3 Community License 的地域排除与禁止蒸馏条款正面冲突。这类错误在合规上比“没写许可”更危险（构成误导性陈述）。公平地说，`recipes.ts:90` 的 caveat 文本本身是对的（“MiniMax H3 Community License, not Apache-2.0 …”），错的只是**卡片 badge 上展示的那个 licence 字段**——两条互相矛盾的信息出现在同一个页面上，用户更可能记住醒目的 badge。 | ① RunSpec 增 `licence_ack: {family, license_id, version, accepted_at}`；② `POST /api/runs` 对 `family=minimax-h3` 缺 ack 一律 422，并把 ack 持久化进 run 行（`ARCHITECTURE.md:272` 要求的正是这个）；③ manifest 拆成 `license`（工具）与 `model_license`（模型），配方卡展示后者；④ 配方卡把“排除 EU/UK/韩/美”与“禁止用输出训其它模型”做成不可折叠的块，不是 caveat 列表里的一行。验收标准见 P0-4。 |

### 2.2 Major

| # | 位置 | 严重度 | 为什么是问题 | 建议修法 |
| --- | --- | --- | --- | --- |
| M1 | `plan.py:377-379`（训练峰值只加 5 项：常驻权重+LoRA+优化器+梯度+fp32 master，再加激活）；`ARCHITECTURE.md:219`（“故意保守 ±15%”） | major | 账本里**没有 CUDA 上下文与分配器保留池**这一项。`02-feasibility.md:75` 的公式明确有 `C_ctx ≈ 0.6–1.2GB`。实测对照最能说明问题：0.6B 那次 `max_memory_allocated` = 1787 MiB（`README.md:32`），planner 在 seq_len=1024 下算 1751.5 MiB，差 **+2.0%**——账本项本身很准；但同一进程的**实际占用是 ~2.7GB**（`native_llm.py:434-441` 的 docstring 与简报一致），差 **+0.9GB**。也就是说误差不是百分比而是**常数偏移**：小模型上相对误差 50%，大模型上 5%。写成“±15%”会让用户以为大配方也可能偏 15%，实际上大配方反而更准、小配方更危险。 | 新增一条 `MemoryLine(label="CUDA 上下文与分配器保留", mib=context_mib)`，`context_mib` 取 600–1200（默认 900，可被 env 覆盖），并在 `risks` 里说明“这是进程级开销，与模型大小无关”。同时把 `overlap_windows_mib` 的语义改成“桌面/浏览器占用”，与 context 分开，避免两项混在一个减项里（现在 `plan.py:395` 把 `overlap_windows_mib` 从 budget 里扣，相当于把它当成了估算的一部分）。 |
| M2 | `schema.py:134-138`（`wan2.2-ti2v-5b` 未设 `vae_spatial_stride`，继承 `schema.py:93` 的默认 8）、对照 `02-feasibility.md:76,85`（`N = …×H/32×W/32`，480×832×49 的 **N=5070**）、`recipes.ts:215`（H3 480p 文案“20,280 latent tokens per step”）、`ARCHITECTURE.md:214`（“At 480p (~20k tokens) the activation term alone is ~7GB”） | major | 5B 的 VAE 总压缩应是 4×32×32（16 空间 stride × patch 2），planner 用 8×2=16，于是把每样本 token 数算成 20280 而研究文档是 5070——**激活项被高估 4×**（我复算：改 stride=16 后 5B 激活 6496 → 1624 MiB，headline 8727 → 3855）。同一个 4× 误差还传播到了 `ARCHITECTURE.md:214`（“480p ≈ 20k tokens → 激活 ~7GB”）——按研究文档的 N=5070，同一几何下激活只有约 1.6GB。反过来，`recipes.ts:215` 给 H3 480p 写的“20,280 tokens/step”也是错的：H3 480p 每样本就是 5070、每步 40560（我复算的 `tokens_per_step=40560`）。**同一份产品里两个 token 数互相矛盾，且都不是它们各自标的那个量**。这直接决定用户对“这一步多慢”的预期。 | 按 `02-feasibility.md:76` 修正 5B 的 `vae_spatial_stride=16`（或把 patch 与 stride 合成一个 `vae_total_stride` 字段，消除这类可漏设的隐含默认值）；`ARCHITECTURE.md:214` 与 `recipes.ts:215` 的 token 数改成从 planner 取而不是手写。验收标准见 P0-6。 |
| M3 | `events.py:92-100`（`run_start.plan` 带整份 MemoryPlan）、`events.py:154`（每步 `vram_peak_mib`）、对照 `packages/web/src/pages/RunDetail.tsx:284`（实测缺失时回退 `headline_mib`）与 `:349-350`（估算值单列） | major | 数据已经全在事件流里了，但 UI **从不把估算和实测放在一起算偏差**。于是产品的核心卖点（“诚实的前置估算”）无法被自己的运行记录证伪或修正：用户看不到“估 3.6GB、实测 4.3GB”，也看不到“这次偏差 19%，下次把预算调到 X”。`RunDetail.tsx:284` 的 `vramPeak ?? headline_mib` 更是个陷阱：没实测时静默显示估算值，用户会把它当成实测。 | 见 4.6 的呈现方案；最小改动是把 `:284` 的回退去掉（无实测就显示“—”），并加一行“估算 3.6GB（±15% + 0.9GB 运行时）/ 实测 4.3GB / 偏差 +17%”，偏差 >20% 时给黄色 callout 并在 `run_end` 后写 `memory_report.json`。 |
| M4 | `native_llm.py:830`（`labels[index, : len(ids)] = tensor`）、`contracts.ts:45-72`（无掩码开关） | major | 对 `dataset.kind=chat` 的数据，prompt 与回答一起算 loss。这不是“SFT 的默认变体”，而是**训练目标错了**：指令微调的标准做法是只对 assistant 段计 loss（`-100` 掩码）。产品首页写的是“指令微调”，用户拿 chat 数据训练会得到质量明显偏低、且与他处经验不一致的 adapter，而 UI 上没有任何提示。 | 在 `DatasetSpec` 或 `TrainSpec` 增加 `mask_prompt: bool = True`（chat 默认开）；tokenize 时记录 assistant 段区间，collate 只保留这些位置为 labels。至少要在 chat 模式下给一条 `W_PROMPT_UNMASKED` 警告。 |
| M5 | `native_llm.py:785`（`train_ds = [tokenize(row) for row in raw]`）、`:793`（eval 同）、`control.py:67`（`check()` 只在 `native_llm.py:605` 被调用）、`runtime.py:292`（只在 step 之间检查取消） | major | 整个语料在训练开始前被逐条 tokenize 成 **Python int 列表**并常驻内存。Python 一个 int 约 28 B + list 指针 8 B，按 2048 token/样本算，10 万样本 ≈ 20–30 GB RSS——在 32GB 机器上这一步就会把内存打满，而且**此时按取消没有任何反应**（`_load_split` 里没有 cancel 检查），用户只能等 server 在 `cancelGraceMs`（`config.ts:203`，默认 10s）后强杀进程树。这跟“消费级 31.7GB RAM”的定位正面冲突。 | tokenize 改成流式的 `IterableDataset`（或至少存 `array('I')`/numpy 而不是 list），并把 `ctx.cancel.check()` 插进 tokenize 循环；或者干脆把「编码数据集」做成一个独立 `prepare` 步骤，让进度和取消都能被看到。 |
| M6 | `plan.py:56-58`（`PCIE_GB_PER_S = 8.0`）、`plan.py:275-280`（`seconds = swapped_bytes / (8e9)`，输出“每步理论最低搬运 2.5s”） | major | 这个数只算了一次单向搬运全部 swap 字节，而实际每步要搬 **前向 + 反向 + 重算** 三趟，且笔记本 PCIe 4.0 x8 的有效带宽是 6–7GB/s、还要加上 pinned-memory 拷贝与 CPU 侧解压。研究文档的实测锚点是：16GB + `blocks_to_swap=12` 约 **10 s/it**（`02-feasibility.md:112`），4090 换 40 块 Qwen-Image 约 **11–14 s/sample**（`:60`）。2.5s 这个数字会被用户当成“每步用时”，实际差 3–10 倍，而 `ARCHITECTURE.md:309` 与 `00-synthesis.md` R3 都把它当作“已缓解”。 | 改成区间并标注口径：`2.5s（单向理论下界）× 3 趟 × (8/实测带宽) ≈ 8–15s`；把“理论下界”从“建议”升级为 `risks` 里的一条（`severity=warning`），并在 UI 上明确写“这不是 ETA”。更好的做法是让 planner 输出 `step_time_floor_s` 与 `step_time_band_s` 两个字段。 |
| M7 | `native_llm.py:176-180`（`expected_artifacts` = `<output>/adapter_model.safetensors` / `model.safetensors`）、对照 `native_llm.py:664-665`（实际写 `<output>/adapter` 或 `<output>/model` 目录）、`Planner.tsx:835-839`（直接展示该字段） | major | 计划页会告诉用户产物在 `output/adapter_model.safetensors`，实际在 `output/adapter/adapter_model.safetensors`。产物路径是“跑完去哪里拿文件”的唯一指引，错了会让用户以为训练失败。 | 由 backend 用与 `_save_checkpoint` 同一函数计算路径（避免两处手写）；或把 `expected_artifacts` 改成“目录 + 通配”，并加一条测试断言该路径在 dry-run 后确实会被创建（可用 tmp run dir + 0 步训练）。 |
| M8 | `plugins.py:239-240`（`_looks_like_backend` 只查 `compile/supports/run`）对照 `plugins.py:52-71`（Protocol 还要求 `probe` / `run_step`）、调用点 `compiler.py:367`、`runtime.py:315` | major | 第三方 backend 只要实现三个方法就能通过选择，然后在 `compile` 阶段被 `probe` 的 `AttributeError` 降级成 warning（`compiler.py:366-375` 吞掉异常），到 `run_step` 时再炸。插件契约是产品声明的一部分（`ARCHITECTURE.md:116-155`），校验不完整意味着“插件生态”这条承诺在第一次第三方接入时就会坏。 | `_looks_like_backend` 补 `probe`、`run_step`，并让 `Backend` Protocol 的 `isinstance` 检查真正被调用；再加一个 conformance 测试集（`03-stack.md:81` 建议过），每个 backend 跑同一组契约用例。 |
| M9 | `musubi.py:374`（`nvidia_vram_mib()` 在 `compile()` 内跑 nvidia-smi）、`musubi.py:385`（用它做 24GB 门槛）、对照 `compiler.py:3-8`（“deterministic JSON”）与 `ARCHITECTURE.md:230`（`POST /api/plan` “compiled, never executed”） | major | 编译期产生了外部副作用，且结果依赖跑在哪台机器上：同一个 spec 在 8GB 机器上被拒、在 24GB 机器上通过；而且这里用的是 **显存总量**，开着浏览器的 24GB 卡一样能过门槛（`musubi.py:385`）。同时系统里有两套显存真源——`cfg.vram_budget_mib`（用户/UI 填）与 `nvidia-smi`——它们不一致时没有任何仲裁。 | 把硬件探测挪到 `probe()`/`env`（那里本来就有 `GpuInfo`），`compile` 只读 `ctx` 里已探到的硬件信息；H3 门槛改用 `cfg.vram_budget_mib − overlap_windows_mib` 判定，并在 issue 里同时打印“配置预算”和“实测可用”，让两套数字对得上。 |
| M10 | `supervisor.ts:392`（每行事件 `appendFileSync`）、`:280`（stderr 每 chunk `appendFileSync`）、`:369`（每步 `recordStep` 写 DB） | major | 这些同步 I/O 全在服务事件循环上，同一条循环还要给 SSE 客户端 `raw.write`。视频 run 的 musubi 子进程会把日志行转发成 `log` 事件（`runtime.py:218-228`），量级可达每秒数十到数百行；同步 append + 同步 SQLite 会让 SSE 卡顿、心跳（`routes.ts:373-376` 的 15s ping）被推迟，长 run 上表现为“监控页越来越卡”。 | 事件落盘改成有界内存队列 + 定时/批量 flush（`fs.createWriteStream` 追加、按 50ms 或 200 行合并）；DB 的 step 计数改成内存累加、按秒批量写。 |
| M11 | `routes.ts:288`、`:350`（SSE 回放整文件）、`:432-450`（`readFile` 全文 split）、`:316-362`（`replayed` Set 无上限） | major | `/api/runs/:id/events` 与每次 SSE 连接都把 `events.jsonl` 整个读进内存再返回。一个 2000 步的 run 加上日志转发轻松到几十 MB、十万行量级；每次刷新页面都重新读一遍。`replayed` Set 还按连接无界增长（每事件一个字符串）。这会随 run 长度线性恶化，而本产品的卖点正是长 run 监控。 | `/events` 加 `?after_seq=` 与 `limit`（事件已带 `ts`，建议再加单调 `seq`）；SSE 回放按行流式读取（`readline`）而不是全文入内存；“live 前的缓冲 + 文件回放”二选一的去重逻辑可以简化成“先记录文件字节偏移，历史只读一次到偏移处”。 |
| M12 | `contracts.ts:45-72`（无 `model_path`）、`schema.py:200`（有）、`contracts.ts:276-284`（EnvReport 无 `musubi/home/models_dir/datasets_dir/cache_dir`）、`routes.ts:170`（`SystemResponseSchema.parse(...)` 后再返回）、`contracts.ts:292`（`RUN_STATUSES` 无 `infeasible`）、`ARCHITECTURE.md:247-248`（声称有“Python 生成 JSON → zod 校验”的契约测试与 `engine/tests/fixtures/`） | major | 契约漂移有两处会造成真实功能缺失：① `model_path` 在 Python 侧是一等字段（`config.py:166-170` 会用它解析基座权重），但 TS 侧没有，zod 默认剥掉未知键 → 用户从 UI/API 传 `model_path` 会被静默丢弃，而 `routes.ts:59-110` 的“路径必须在允许根内”检查也永远看不到它（这条同时也是路径逃逸的口子：Python 侧接受任意绝对路径）；② `EnvReport` 的 `musubi` 字段被 `parse` 剥掉 → **UI 永远看不到“musubi 未安装”**，而这是视频路径最关键的一个状态。而 `ARCHITECTURE.md:247-248` 承诺的契约测试并不存在：我检查了 `engine/tests/`（8 个测试文件）与 `packages/server/test/`（8 个），**没有任何 `fixtures/` 目录，也没有把 Python dump 拿去喂 zod schema 的测试**——所以这些漂移不会被 CI 发现。 | 用 `schema.py` 的 `model_json_schema()` 生成 TS 契约（`03-stack.md:74,113` 已经这么建议），或在 CI 里加一个“Python dump ↔ zod parse 双向相等”的契约测试并补上 `engine/tests/fixtures/`；`model_path` 补进 TS 并纳入 `normalizeSpecPaths` 的根检查；`EnvReport` 补齐字段（或让 `routes.ts:170` 不用 `parse` 丢字段）。 |

| M13 | `plan.py:384-387`（采样峰值只算常驻权重，video 明确不计 VAE decoder）、`:307-310`（只在不缓存 latents 时计 VAE 权重） | major | “先出 1 帧预览再决定要不要训练”是视频用户最自然的第一个动作，但 H3 的 `vae_params_b=2.5`（`schema.py:155`）在 bf16 下约 4.8GB（我复算 `cache=text` 时 VAE 行 4768.4 MiB），而采样峰值里完全没有它。8GB 卡上这句“预览一下”会直接 OOM，而计划页显示 feasible。这个缺口只在 `notes` 里，用户看不到。 | 给 video 家族加一个 `sample_peak` 估算（VAE decoder + 少量 latent），在 `PlanStep` 里作为独立 step 或在计划页单列一行“预览采样峰值”，并对 8GB H3 明确标“不可预览”。 |
| M14 | `packages/server/src/routes.ts:186-188`（`/api/families` 直接转发 `engine.families()`）、`engine/src/supertuner_engine/cli.py:177-183`（返回全部 `VIDEO_FAMILIES`，含无 backend 的 `cogvideox-2b` / `ltx-video-2b`，`schema.py:139-148`） | major | 家族选择器会列出两个没有任何 backend 声明的家族。用户选了 `cogvideox-2b` 后，编译阶段才会因为没有匹配插件而报错——这是“能选但一定失败”，与 B1 同一类用户体验错误，只是后果轻一些（有明确 error issue）。 | `/api/families` 只返回至少有一个已注册 backend 声明的家族，或在每个家族上附 `backend_ids` 让 UI 置灰无后端的家族；验收见 P1-8。 |

### 2.3 Minor

| # | 位置 | 严重度 | 为什么是问题 | 建议修法 |
| --- | --- | --- | --- | --- |
| N1 | `ARCHITECTURE.md:178-179`（`step` 带 `vram_mib`）对照 `events.py:154`、`packages/core/src/events.ts:30`（`vram_peak_mib`） | minor | 文档自称 normative（`ARCHITECTURE.md:3-5`），字段名却不一致；实现 `parse_event_line`（`events.py:227-248`）只校验 `type`/`ts`，所以这种漂移不会被测试发现。 | 改文档或改代码，同一 change 内完成；并给事件加一个“未知字段允许、已知字段必须精确”的契约测试。 |
| N2 | `contracts.ts:102-107`（`PlanResponseSchema` 只含 `MemoryPlan`）对照 `routes.ts:213-215`（返回完整 `RunPlan`） | minor | 旧契约残留，会误导后来者以为 `/api/plan` 不返回执行计划。 | 删除或改名 `MemoryPlanResponseSchema`。 |
| N3 | `schema.py:247`（`format` Literal 无 `txt`/`text`，有 `webdataset`）对照 `native_llm.py:61`（`_DATASET_FORMATS` 有 `txt/text`、无 `webdataset`） | minor | `txt/text` 分支永远不可达（schema 不允许），`webdataset` 能通过 schema 但被 backend 拒；两边各有一段死逻辑。 | 以 backend 能力为准收敛成一份枚举；`webdataset` 要么实现要么从 schema 移除。 |
| N4 | `musubi.py:576`（`if cfg.blocks_to_swap == 0 and cfg.seq_len == 0:  # pragma: no cover - never true`） | minor | 目录里留了一段自己承认永假的分支，说明 `--vae_cache_cpu` 这个真实的省显存开关被写坏了。 | 改成有意义的条件（例如 `cfg.cache` 或显存预算不足时开），或删掉并记 issue。 |
| N5 | `native_llm.py:850-857`（`_pack`） | minor | pretrain 的 packing 直接拼接文档、不重置 position、不区分样本边界，且丢弃尾部不足一块的数据；会让长上下文预训练的 loss 与标准做法有系统性差异（`03-stack.md:80` 对数据集格式有明确规范，唯独没有规定 packing 的边界语义）。 | 要么按 padding-free + varlen batch 的标准做法实现边界，要么在 notes 里明确“packing 是简版，跨文档注意力存在”。 |
| N6 | `native_llm.py:664-665` + `:940`（`optimizer.pt`/`scheduler.pt` 与 adapter 写进同一目录） | minor | 交付物目录里混入优化器状态，体积远超 `README.md:35` 讲的 40MB，用户 `ls` 时会困惑，也容易被误上传到 HF。 | 最终目录只放 adapter + tokenizer，优化器状态放 `checkpoints/step-N/optim/`。 |
| N7 | `engine/tests/test_plan.py:114-125`（`test_pretrain_360m_fits_8gb`：`FP32_MASTER` + `ADAMW_PAGED_8BIT` 的 golden 场景）对照 `native_llm.py:334-343`（fp32_master 直接拒绝）、`:516-520` 与 `:1130-1131`（缺 bnb 时运行时抛错） | minor | 单测固化了“planner 认为可行、但没有任何 backend 能执行”的组合，等于给未来的人留下一条假路径。同类问题：`_static_issues` 对缺 bnb 的 QLoRA 只发 `W_BITSANDBYTES_MISSING`（`native_llm.py:303-320`），于是 8/12/16GB 的默认 QLoRA 配方会显示“可跑”，运行时才炸。 | 保留 planner 的纯数学测试，但加一条 `compile_spec` 级别的断言：该组合必须产出 error issue；缺 bnb 而 `adapter=qlora` 时同样升级为 error（或在配方/计划页按探测结果过滤）。 |
| N8 | `ARCHITECTURE.md:253-254`、`README.md:107-111` 引用的 `engine/tests/test_smoke_gpu.py` | minor | 文件不存在；`pytest -m gpu` 会“成功”地跑 0 个测试，给人一种“有 GPU 冒烟”的错觉。 | 要么补文件（真正的 0.6B 2-step 冒烟，含显存断言），要么从文档移除；并在 CI 里断言 `-m gpu` 至少收集到 1 个。 |
| N9 | `musubi.py:373-402`（`_h3_guard` 用 `memory.total`） | minor | 门槛用总量，不扣桌面占用；24GB 卡的可用余量常常只有 20–21GB。 | 改 `memory.free` 并与 `overlap_windows_mib` 一致。 |
| N10 | `routes.ts:158-162`（500 返回 `error.message` 原文） | minor | 会把内部异常细节（含路径）返回给浏览器；虽然是 loopback，但日志与 UI 混在一起后难以排查真实原因。 | 统一成 `{error: "internal", request_id}`，细节只进日志。 |
| N11 | `plugins.py:113-119`（同 id 同 version 静默忽略） | minor | 两个不同实现用同一个 id/version 时，第二个被无声吞掉，用户会疑惑“为什么我装的插件没生效”。 | 记录到 `errors`（`PluginError`）而不是直接 `return`。 |

---

## 3. 风险登记表

| 类别 | 风险 | 触发条件 | 影响 | 缓解措施 | 需在 UI 暴露 |
| --- | --- | --- | --- | --- | --- |
| 技术 | 视频后端 argv 与上游 CLI 不匹配（B1、`musubi.py:55`；以及 `--block_swap_h2d_only` 未使用、H3 脚本参数名未核实） | 用户第一次在装了 musubi 的机器上点“开始” | 计划显示可行、运行在第一个 cache/train 步骤退出；用户损失的是对“编译计划可信”的信任 | 把上游 `docs/wan.md`/`docs/minimax_h3.md` 的 task/flag 清单做成测试固件；`--dry-run` 输出与固件逐项比对；升级 musubi 后先跑对照 | 是：计划页每个外部步骤显示“已对 musubi vX.Y 校验”或“未校验（本机无 musubi）”，并给出 argv 原文 |
| 技术 | planner 缺 CUDA 上下文项与几何错误（M1、M2），估算与实测存在**常数**偏移 | 小模型（≤1B）或 Wan5B 配方 | 小 run 低估 ~0.9GB（相对 50%），Wan5B 高估 4× 激活；用户据此选错档位 | 加 `context_mib` 行；修 stride；把实测回填进 planner 的校准表（见 4.6） | 是：预算条旁显示“含运行时开销 0.9GB”，并显示估算区间而不是单值 |
| 技术 | block swap 吞吐被低估 3–10×（M6） | 8–16GB 卡 + 大 `blocks_to_swap` | 用户以为“一晚上能跑完”，实际要几天；H3 320p 每步可能几十秒 | 输出 `step_time_band_s`；用 H2D-only 与 pinned memory 的实测数据替换 8GB/s 假设 | 是：配方卡与计划页都写“每步 8–15s 起（PCIe x8 实测区间）”，并提示“先跑 20 步测速” |
| 技术 | 8GB 卡实际可用显存低于 `nvidia-smi` 总量（见 4.3） | Windows + 桌面/浏览器占用 + WDDM | 计划 headroom 只有 400–800 MiB 的配方（`wan21-1.3b-lora-8gb`：6165 / 预算 6592）在真实环境频繁 OOM | 预算默认取 `free − reserve`；planner 加 allocator 项；首次运行前给“关闭浏览器/其它 GPU 程序”的检查 | 是：系统页显示 total / free / 预留三档，计划页显示“本次可用 6.4GB（已扣桌面 1.6GB）” |
| 技术 | Windows 下 `PYTORCH_CUDA_ALLOC_CONF` 只对 engine 自身生效，musubi 子进程不受控（`native_llm.py:434-448` 只在 native 路径调用） | 视频路径 | musubi 侧仍会撞 WDDM 分配器问题，表现为“显存明明够却 OOM” | 在 `run_command_step` 的 env 里统一注入（`runtime.py:159-161` 已经有 env 拼装点），或写进 musubi 步骤的 `PlanStep.env` | 是：运行详情的“环境”块显示实际生效的 `PYTORCH_CUDA_ALLOC_CONF` |
| 许可 | H3 地域排除（EU/UK/韩/美）与“禁止用输出改进其它模型”没有门禁（B4） | 任何 H3 配方被选 | 用户在自己所在地区使用即违约；用 H3 输出做数据蒸馏（DMD/DMAD）额外违约；产品方无记录可自证 | RunSpec 加 ack 字段 + server 强制 422 + 记录到 run 行；禁用蒸馏（已在 `compiler.py:112-121` 拒绝，但拒绝理由应引用许可条款而不只是“单卡未验证”） | 是：配方卡与运行确认页都要有不可跳过的许可块；运行详情显示 ack 时间与许可证版本 |
| 许可 | musubi 插件把工具许可当模型许可展示（B4）；musubi 仓库本身无 LICENSE 文件（`README.md:133-135`） | 任何视频配方 | 用户以为 H3/Wan 权重许可是 Apache-2.0 | 拆分 `license`（工具）与 `model_license`（模型）；在文档里注明 musubi 的许可状态需法务确认 | 是：插件页与配方卡分别显示“工具许可”和“权重许可” |
| 合规 | 数据不出机器是卖点，但 `train.resume_from` / `model_path` 可指向任意外部路径（`routes.ts:59-110` 漏检 `model_path`，M12） | 恶意/误配置的 spec | 与“本地、私有”的定位不符；也可能把用户数据写到非预期位置 | `normalizeSpecPaths` 覆盖全部路径字段（`dataset.path`/`output.dir`/`train.resume_from`/`config.model_path`），并给出一条“本次将读写这些路径”的确认 | 是：运行确认页列出全部将被读写的绝对路径 |
| 可靠性 | 事件热路径同步 I/O（M10）+ 事件 API 全量读（M11） | 长 run（>1000 步）或带日志转发的视频 run | SSE 卡顿、监控页变慢、内存增长；长 run 中途页面无法恢复 | 批量异步落盘；`/events` 加分页与 `seq`；SSE 流式回放 | 是：监控页显示“已接收 N 事件 / 落后 M 事件”，让卡顿可见 |
| 可靠性 | 取消在数据准备阶段无效（M5，`control.py:67` 只在 `native_llm.py:605` 调用） | 大语料首次 tokenize | 用户点取消后进程继续占 GPU/CPU/RAM，直到 server 强杀；强杀后 run 标 cancelled 但可能留下半个 cache | 在 `_load_split` 与 musubi prepare 循环里插 `ctx.cancel.check()`；`forceTerminate` 后清理半成品 | 是：取消按钮旁注明“当前阶段：准备数据（可在 N 秒内停止）” |
| 可靠性 | 队列不跨重启存活：启动时 `store.recoverInterrupted("server restarted")`（`packages/server/src/index.ts:44`、`db.ts:345`）把 `queued`/`running` 的 run 一律标成 `failed`（有测试覆盖：`packages/server/test/recovery.test.ts`） | server 崩溃 / 断电 / 手动重启 | 正在跑的任务被标 failed 且不可自动续跑；`train.resume_from` 只在单个 run 内可用（`native_llm.py:536-585`），不构成跨重启恢复 | 保留现有行为（诚实优于假装），但 UI 上要把失败原因显示成“服务重启，任务被中止”，并给“可从 checkpoint 续训”的入口 | 是：运行列表对这类失败显示可点击的“从此 checkpoint 续训” |
| 可靠性 | 并发 run 争显存（`config.ts:198` 默认 `maxConcurrentRuns=1`，但可配到 64） | 用户把并发调大 | 每个 run 的计划都假设独占 GPU，两个 run 同时跑必然 OOM；planner 的诚实性被配置项抵消 | 并发 >1 时做显存预算准入（sum(headline) ≤ free），或在 UI 上把并发标成“实验性，需要显存准入（未实现）” | 是：设置页显示“当前并发 2，预计峰值 7.1GB / 可用 6.4GB”这类冲突警告 |
| 可靠性 | musubi CLI/日志格式变动（`ARCHITECTURE.md:306-307` 已识别） | musubi 升级 | 进度解析失败、argv 被拒 | 锁版本范围；`--dry-run` 对照；只解析文档化行；把“未知日志行转 log 事件”这条兜底（`runtime.py:227-228`）保留 | 是：运行详情显示“musubi 版本 / 已校验范围” |
| 产品定位 | 真相表（`README.md:45-51`）与配方 tier 宣称超过证据 | 对外发版 | 被用户或竞品指出“8GB 能训 H3”“16GB 能训 5B”，信任一次性崩塌 | 给真相表加“已验证/编译预览/未验证”三列；H3 行明确写“需外部 musubi + 许可限制 + 未在本机跑通” | 是：README、配方卡、计划页三处口径必须由同一份数据生成（见 P0-1 的验收） |
| 产品定位 | 8GB 档的 QLoRA 实际不可用（本机无 bnb） | 用户在 Windows 上选 QLoRA 配方 | 配方列表里 8GB/12GB/16GB 的 LLM 配方全部默认 QLoRA，但运行时缺 bnb 直接失败 | 配方按“本机实际可用”过滤或标注“需要 bitsandbytes（未安装）”；给出 LoRA bf16 的等价配方 | 是：系统页与配方卡都显示依赖状态，不是只在运行时抛错 |
| 产品定位 | “估算 vs 实测”没有闭环（M3） | 每次运行结束 | 卖点无法被自证；用户也不知道该信多少 | 见 4.6 | 是：运行详情与计划页都要有偏差数字 |

---

## 4. 记忆模型准确性审查

### 4.1 我复算的配方数字（与 `recipes.ts` 声明值比对）

| 配方 | 声明 VRAM / RAM (MiB) | 复算 VRAM / RAM | 偏差 | 每步 token |
| --- | --- | --- | --- | --- |
| qwen3-1.7b-qlora-8gb | 1626 / 0 | 1626 / 0 | 0.0% / — | 8192 |
| wan21-1.3b-lora-8gb | 6165 / 0 | 6165 / 0 | 0.0% / — | 81120 |
| h3-lora-320p-8gb | 3638 / 19226 | 3638 / 19226 | 0.0% / 0.0% | 18720 |
| qwen3-4b-qlora-12gb | 3304 / 0 | 3304 / 0 | 0.0% / — | 16384 |
| qwen3-8b-qlora-16gb | 5775 / 0 | 5775 / 0 | 0.0% / — | 16384 |
| wan22-5b-lora-16gb | 8446 / 3338 | **8727** / 3338 | **+3.3%** / 0.0% | 81120 |
| h3-lora-480p-24gb | 10008 / 16022 | 10008 / 16022 | 0.0% / 0.0% | 40560 |
| h3-lora-720p-32gb | 19991 / 12817 | 19991 / 12817 | 0.0% / 0.0% | 104832 |
| h3-lora-720p-48gb | 28125 / 6409 | 28125 / 6409 | 0.0% / 0.0% | 104832 |

结论：**配方与 planner 同步率高**（唯一漂移是 5B 那行 +3.3%，需要重跑一次配方导出）。这排除了“配方手抄错”这一假设，把问题定位到 planner 的建模假设上。

### 4.2 与唯一可核对的实测对照

| 量与来源 | 值 | 与 planner 的差 |
| --- | --- | --- |
| planner：qwen3-0.6b LoRA，seq_len=512 / 1024 / 2048 | 1705.5 / 1751.5 / 1843.5 MiB | — |
| `README.md:32` 记录的 run：`max_memory_allocated` 峰值 | **1787 MiB** | 相对 seq=1024 估算 **+2.0%**；相对 seq=512 **+4.8%** |
| `native_llm.py:434-441` 与简报里的进程级峰值（async allocator 下） | **~2700 MiB** | 相对估算 **+0.9GB（+51%）** |

这条对照是本文档里最关键的证据：**planner 的“张量账本”很准，缺的是账本之外的固定开销**。同一份 `ARCHITECTURE.md:219` 说“±15%”，但真实误差结构是“约 0.9GB 常数 + 百分比误差”，小配置上百分比被常数项主导。这也解释了为什么 8GB 卡上很多配方“理论上差 400MB、实际就是 OOM”：那些配方的 headroom 恰好小于这个常数项。

### 4.3 8GB 卡“实际可用”与预算口径

- UI 用 `gpu.vram_mib`（= `nvidia-smi` 报的 **total**）填 `vram_budget_mib`（`Planner.tsx:116-119`），planner 再减 `overlap_windows_mib=1600`（`plan.py:395`，`schema.py:194`）。所以 8GB 卡的有效预算恒为 6592 MiB。
- 但 `EnvReport.gpus[].vram_free_mib` 已经探到了（`schema.py:370`、`cli.py:166-167`），**planner 不用它**。Windows/WDDM 下 total 8GB 的卡在开着浏览器+桌面合成时 free 通常只有 6.2–7.0GB；减去 reserve 后与 6592 大致同量级，但**余量极小**：`wan21-1.3b-lora-8gb` 复算是 6165 MiB，headroom 仅 **427 MiB**，小于 4.2 的常数项。
- 敏感性验证（把可用降到 7168 MiB 重算）：`wan21-1.3b-lora-8gb`（6165 vs 预算 5568）与 `qwen3-8b-qlora-16gb`（5775）都从 feasible 变 infeasible。这不是“估算错了”，而是**预算口径选错了**：用 total 让配方看起来可行，用 free 会让一部分配方直接不可行——后者才是真实答案。
- 附带发现：`qwen3-8b-qlora-16gb`（headline 5775）在 8GB 档的 6592 预算下其实是 feasible（headroom 817），但配方把它归到 16GB 档；`README.md:47` 又写“8GB → QLoRA up to 4B”，`00-synthesis.md:27` 写“4B–8B”。三处口径互不一致（配方 tier / README / 市场文档），而 planner 自己的数字落在第 4 种取值上。对外前必须统一。

### 4.4 block swap 的搬运成本

- 代码口径：`plan.py:276` `seconds = swapped_bytes / (8e9)`，H3 320p 得 **2.5s**（我复算的输出里能看到这条 note）。
- 这个数少算了至少三件事：(a) 每步前向+反向+重算各搬一趟（≈3×）；(b) 笔记本 PCIe 4.0 x8 的有效带宽 6–7GB/s（≈1.2×）；(c) pinned memory 在 Windows 上还会额外吃共享显存（`03-stack.md:96`），而 `plan.py` 完全不建模 pinned 副本。
- 实测锚点：16GB + swap12 ≈ 10 s/it（`02-feasibility.md:112`）、3090 换 40 块 11–14 s/sample（`:60`）。据此，H3 320p 的“每步最低 2.5s”实际应是 **8–15s 起**，720p（104832 tokens/步）会更糟。
- 建议：`plan.py` 输出两个字段 `swap_floor_s`（单向理论值，保留现有算法）与 `step_time_band_s`（×3 趟 ÷ 实测带宽，附来源），UI 只把后者当 ETA 区间，且明确“未含注意力计算时间”。

### 4.5 文本编码器与 VAE 的缓存代价

- **缓存不是免费的**：`plan.py:285-286` 在 `cache` 含 TEXT 时把陪伴项记为 0，这是对**训练步**正确的，但对 `cache-text` 步骤完全错误。H3 的编码器是 Qwen3-VL-32B（`schema.py:155`，`02-feasibility.md:44`）：bf16 ≈ 48GB（`02-feasibility.md:46` 口径；32B 全量理论值 ~64GB）、int8 ≈ 25GB，即使 `text_encoder_blocks_to_swap=50`（`recipes.ts:107`）也只把它搬到主存，仍需 ~25GB RAM 与每步搬运——而计划里这条 RAM 完全没算（`plan.py:302` 只在**不缓存**时才把 TE 加进 `ram_peak_bytes`）。
- 于是 H3 320p 的真实资源需求是：训练峰值 3.6GB VRAM + 19.2GB RAM（换块）+ **缓存步骤的 25GB+ 一次性需求**（分两次跑或另找卡）。`recipes.ts:88` 说了这件事，planner 没说。用户如果只读计划页，会得到“8GB 能跑”的结论。
- VAE：`schema.py:155` 给 H3 的 `vae_params_b=2.5`，我复算 `cache=text` 时 VAE 权重行是 **4768.4 MiB**（bf16）。这个数决定了“不缓存 latents 就没法跑”，也说明 **latents 缓存对 8GB 是硬前提**，应该在配方卡第一行而不是 caveat 第五行。另外 `plan.py:384-387` 明确不把 VAE decoder 计入采样峰值，所以“先预览一张图再决定是否训练”这个自然操作在 H3 上会 OOM——这条只写在 notes 里，用户看不到。
- Wan 侧：`schema.py:136` 的 5B 与 Wan2.1 共用 UMT5-XXL（`text_encoder_params_b=5.6`，bf16 ~10.6GB，`02-feasibility.md:79`），缓存步骤同样要把它整只载入；`recipes.ts:183` 只写“Keep the text encoder cached or quantised”，没有给出这条步骤的显存。同样需要进计划。

### 4.6 H3 几何的自洽性检查（另一处系统性偏差）

用 `schema.py:153-159` 的常量反推参数：每层 `4h² + 3h·ffn = 4×6144² + 3×6144×24576 ≈ 0.604B`，×50 层 ≈ **30.2B**。而 `schema.py:157` 的 `pruned_params_b=20.0`、`ARCHITECTURE.md:204-205` 的“33B 总量、~13B 在 AdaLN”意味着非 AdaLN 部分是 20B。两者差 **+51%**：要么几何偏大（→ 激活项高估 ~1.5×），要么 20B 偏小（→ 权重/换块 RAM 低估 ~50%）。**这两个方向不可能同时正确**，因为权重项锚在 20B、激活项锚在 hidden/ffn。

`ARCHITECTURE.md:316-317` 把这一点标为“近似、±15%、`params_b_override` 兜底”（`:219` 也重复了同一口径），但实测的内部不一致是 ~50%，不是 15%。建议：H3 的 `approx` 提示从“±10~15%”改成“几何未公开，权重与激活的偏差方向相反，可能在 ±50% 量级”；并在 `params_b_override` 之外再加 `hidden/ffn/layers` 的 override，允许用户在拿到真实 config 后校正。这一条属于**未验证**：我没有 H3 的 config.json 可核对。

### 4.7 该如何向用户呈现“估算 vs 实测”

现有事件已经具备全部数据：`run_start.plan`（`events.py:92-100`）带整份账本，`step.vram_peak_mib`（`events.py:154`）带每步 `max_memory_allocated`。缺的只是呈现。建议按三层做，从便宜到贵：

1. **口径分离（P0，纯展示）**：把“显存”拆成 `allocated`（torch 统计）、`reserved`（torch 保留池）、`process`（`nvidia-smi --query-gpu=memory.used --id=<idx>`，每 5s 采样一次）三个数，再加一个 `planned`。`run_end` 后写 `memory_report.json`：`{planned, max_allocated, max_reserved, max_process, delta_pct}`。运行详情页显示一行：`估算 3.6GB（含运行时 0.9GB）/ 实测 4.2GB（allocated）/ 4.6GB（进程）/ 偏差 +17%`。
2. **区间而不是单值（P0）**：计划页把 `headline_mib` 显示成区间 `3.6–4.4GB`（`±15% 建模误差` + `0.9GB 运行时开销` 分开标注），并同时显示“可用 6.4GB（total 8.0 − 桌面 1.6）”。这样即便估计偏差 50%（4.2 章的情形），用户也已经在区间里被提示过。
3. **校准回路（P1）**：每次 run 结束把 `(family, geometry, adapter, quant, blocks_to_swap, planned, measured)` 写进本地 `calibration.jsonl`；planner 在同类配置上优先用最近 N 次实测的 `max_process` 作为 `headline` 的下界，并在计划页注明“基于你机器上 3 次同类运行的实测”。这是把“诚实估算”从文案变成产品行为的关键一步，也是 `00-synthesis.md:16-21` 定位里唯一还没落地的部分。
4. **明确“不是 ETA”**：`estimated_minutes` 在 `compiler.py:407` 永远是 `None`，UI 的 “expected” 一直显示 `—`（`RunDetail.tsx:353-354`）。与其留空，不如先用 `swap_floor_s` + `tokens_per_step` 给一个带区间的粗估，并标注“未含注意力计算，仅供量级参考”。

---

## 5. 优先级建议

### P0（进入下一次对外发布前必须完成）

| # | 事项 | 理由 | 验收标准 |
| --- | --- | --- | --- |
| P0-1 | 视频声明与实现对齐（B1 + 真相表口径） | 这是唯一会让“计划可行 → 必然失败”的组合；且它同时污染 README、真相表、配方三处对外文案 | ① 把 musubi `docs/wan.md` + `docs/minimax_h3.md` 的 `--task` 与关键 flag 清单写成测试固件（`engine/tests/fixtures/musubi_tasks.json`），`_TASK_BY_FAMILY` 必须逐项命中，否则测试失败；② 若 5B 不在上游 task 列表，则 `MUSUBI_FAMILIES`、`recipes.ts`、`README.md:45-51` 同时移除或标 `compile_only`；③ 新增一条测试断言“每个 `Recipe` 的 `(family, tier)` 必须在真相表里有对应行，且该行的状态标签 ∈ {verified, compile_only, unverified}”。 |
| P0-2 | 计划覆盖所有步骤（B2） | 8GB H3 的结论（3638 MiB / feasible）如果不算 cache 步骤，等于把“证明性配置”说成已验证 | `PlanStep.estimated_mib` 由 musubi backend 填值（cache-text = TE 权重 × 常驻比例，cache-latents = VAE 权重），`RunPlan.feasible` 要求全部步骤可行；新增测试：8GB 预算下 `h3-lora-320p-8gb` 的 compile 结果为 `feasible=false`，且 issue 文案含“缓存文本编码器需要 X GB，超过预算”；若用户提供已缓存的 cache 目录，则 cache 步骤标 `optional` 且计划可过。 |
| P0-3 | 预算取机器真值（B3 + 4.3） | 直接对应“planner 绝不虚报”这条产品承诺 | ① `POST /api/runs` 从 `env` 探针取 `ram_mib` 与 GPU `free`，写入 spec 的新字段（如 `machine: {ram_mib, vram_free_mib}`）；② `plan()` 在 `ram_budget_mib`/`vram_budget_mib` 缺省时用机器值，并把“预算来源（实测/手填）”作为一条 note；③ 测试：monkeypatch `ram_mib=16384` 时 `h3-lora-320p-8gb` 必须 infeasible，issue 文案含“需要 19.2GB 主存，检测到 ~11GB 可用”。 |
| P0-4 | 许可门禁（B4） | 法律暴露 + 误导性展示，且修复成本低 | ① `RunSpec` 增 `licence_ack`，`POST /api/runs` 对 H3 缺 ack 返回 422；② ack 持久化并在 `GET /api/runs/:id` 可见；③ manifest 区分工具/模型许可，`licenceForFamily` 取模型许可；④ 测试：无 ack 的 H3 spec → 422；Recipes 页 H3 卡片渲染文本含 “MiniMax H3 Community License” 且不含 “Apache-2.0”。 |
| P0-5 | 估算/实测闭环 + 上下文项（M1、M3、4.6） | 这是产品最核心的差异点，现在完全无法自证 | ① `plan.py` 增加 `context_mib` 项（默认 900，可 env 覆盖）；② 每个 run 结束写 `memory_report.json`（`planned/max_allocated/max_reserved/max_process/delta_pct`）；③ RunDetail 显示偏差行，>20% 时给 warn callout，且 `:284` 的“实测缺失回退估算”被移除；④ 回归测试：把 1787 MiB 那次运行的参数写成 fixture，断言“估算 + 0.9GB ≥ 实测 ≥ 估算 − 10%”。 |
| P0-6 | Wan5B 几何与 token 文案（M2） | 4× 的 token 误差直接决定 16GB 档的可行性判断，还会误导用户对步时的预期 | ① `schema.py` 的 5B 设 `vae_spatial_stride=16`（或改为单一 `vae_total_stride`）；② 新增测试断言 `480×832×49` 的 5B 每样本 token = 5070、H3 320p = 2340，且与 `02-feasibility.md:76` 的公式一致；③ `recipes.ts:215` 的 token 文案改为由测试从 planner 取值比对。 |

### P1（v0.2 前，影响质量与可扩展性）

1. **SFT 指令掩码（M4）**：chat 数据默认 `mask_prompt=True`；验收：单测断言 assistant 段以外 labels 为 `-100`。
2. **数据集流式化 + 可取消（M5）**：验收：10 万样本 jsonl 的 RSS 增量 < 1GB；tokenize 期间按取消 10s 内退出。
3. **事件路径异步化 + 分页（M10、M11）**：验收：5000 步 run 的 SSE 心跳间隔稳定 < 2s；`/events?after_seq=` 返回增量。
4. **插件契约校验补全（M8）+ conformance 测试集**：验收：只实现 3 个方法的假 backend 在选择阶段被拒。
5. **compile 纯化（M9）**：把硬件探测移出 `compile`；验收：同 spec 在两台不同显存的机器上 `compile` 结果的 `feasible` 只依赖 spec 字段。
6. **产物路径（M7、N6）**：验收：dry-run 产出的 `expected_artifacts` 与真实 `_save_checkpoint` 路径一致，且最终目录不含 `optimizer.pt`。
7. **采样/解码峰值（M13/4.5）**：把 VAE decoder 与文本编码器推理峰值做成独立 step 估算；验收：H3 的“预览采样”步骤不再默认 feasible。
8. **视频家族按已注册 backend 过滤（M14）**：`/api/families` 只返回至少有一个 backend 声明的家族，或给每个家族加 `backend_ids`；验收：`cogvideox-2b` 不再出现在可选列表。
9. **GPU 冒烟测试（N8）**：补 `test_smoke_gpu.py`，跑 0.6B 2 步并断言 `max_memory_allocated` 落在估算区间内；验收：CI 上 `-m gpu` 至少收集 1 个。

### P2（v0.3 及以后）

- 断点续训完整化（RNG / dataloader 游标 / 优化器 state 的跨版本兼容，`03-stack.md:78` 的做法）。
- `estimated_minutes` 的粗估与“先跑 20 步测速”流程。
- 校准回路 `calibration.jsonl`（4.7 第 3 层）做成产品功能。
- DPO/GRPO 插件化（当前 `native_llm.py:244-405` 的 `_static_issues` 对 preference/rl/distill 全报 error，方向正确，别提前做）。
- 导出（gguf / diffusers）与 ComfyUI 目录结构。
- 云端回退（同 RunSpec 搬到按小时 GPU + 续训）。

---

## 6. 被否决方案与其理由是否成立（对照 `00-synthesis.md` §8）

| 被否决方案 | 原理由 | 复核结论 |
| --- | --- | --- |
| 自研视频 DiT 训练核（`00-synthesis.md:75`） | 等于重写 block swap/缓存/分布式，维护负担大、模型覆盖少 | **理由成立，但后果没被接住。** 拒绝自研意味着产品正确性 100% 依赖一个外部、未安装、仓库无 LICENSE 文件的 checkout（`README.md:133-135`、`03-stack.md:19`）。代码里对它的适配是“只构造 argv、不校验 argv”（B1），于是“不自研”这个正确的架构决定，落地成了“无法验证的能力宣称”。正确的后续动作不是重新考虑自研，而是把视频路径显式降级为 compile-only 并把校验固件化（P0-1）。 |
| 浏览器里跑训练（WASM/WebGPU，`:76`） | 显存与算子不满足 | **成立，无异议。** 代码侧也没有任何相关痕迹。 |
| 把 H3 全参写进路线图（`:77`） | 单卡不可能，写进产品等于虚假宣传 | **成立，且在代码里被执行了**：`musubi.py:150-159` 直接拒绝视频全参，`compiler.py:156-165` 拒绝全参 + swap，`plan.py:253-256` 只对全参用完整 33B 权重。这条是“文档承诺 ↔ 代码行为”一致的最好例子，可以当作其它几条的模板。 |
| 内嵌 SimpleTuner / diffusion-pipe（`:78`） | AGPL/GPL 传染 + Windows 不可用 | **成立。** `README.md:136-137` 明确只作设计参考；`03-stack.md:20-21` 给了许可依据。唯一要注意的是 musubi 自己的许可状态（无 LICENSE 文件）没有被同等对待——见风险表的“许可”行。 |
| 服务端直接调 torch（Node 原生绑定，`:79`） | 把 CUDA 崩溃风险放进主进程，放弃进程隔离 | **成立，且实现到位**：`runtime.py:164-175`（argv + shell=False）、`supervisor.ts:230-233`（spawn 子进程 + JSONL）、`compiler.py:366-388`（插件异常不杀编译）三处构成了真实的隔离边界。这条是整个仓库里工程质量最高的部分。 |

补充两条 `00-synthesis.md` 里**没有被否决、但应该重新评估**的判断：

1. **“视频侧不自研训练核”是正确的，但“8GB 的 H3 证明点”是错的定位**（`00-synthesis.md:27` 的 8GB 行）。研究文档自己说 H3 在 8/16/24/32GB 全 ✗（`02-feasibility.md:133`），且文本编码器缓存要另找卡（`:181`）。把“技术上能编译出 argv”包装成“8GB 可训 H3”，正是 `00-synthesis.md:77` 自己批判过的“把不可能写进路线图”。建议把 8GB 的 H3 从真相表里删除，只在“实验/compile-only”区块出现。
2. **“SDPA 而非 flash-attn”成立**，代码也一致（`plan.py:193-194` 拒绝 math、`musubi.py:632` 传 `--sdpa`）；但 `attn_impl=flash` 在 planner 里与 `sdpa` 完全等价（没有任何校验或提示），用户选了 flash 也不会得到任何警告——建议对 `AttnImpl.FLASH` 在 Windows 上加一条“未安装/无 wheel，实际会回退 SDPA”的 warning。

---

## 附录 A：可复现的验证命令

```powershell
# 1) 引擎测试（144 passed）
uv run --project engine pytest engine/tests -q

# 2) 前端/服务端测试（server 48 / web 68）
pnpm test

# 3) 重算全部配方并与 recipes.ts 比对（脚本见 .drafts/review-arch/check_plan.py）
$env:PYTHONPATH='engine/src'
uv run --project engine python .drafts/review-arch/check_plan.py
```

`check_plan.py` 还包含三段敏感性重算：Wan5B 的 `vae_spatial_stride` 8→16、H3 的 `cache=text`（不缓存 latents）、8GB 卡按 7168 MiB 可用重算。本文档 4.1–4.4 的数字全部来自这个脚本的输出。

## 附录 B：未验证清单（不要当成结论使用）

- musubi-tuner 的全部 argv 细节：`--task ti2v-5B` 是否真的不存在、H3 脚本的参数名（`--video_vae`/`--audio_vae`/`--text_encoder`）、`networks.lora_minimax_h3` 的模块名、`--block_swap_h2d_only` 是否必需——本机未安装 musubi（`musubi.py:277-301` 的探测在本机会返回 None）。
- H3 的真实几何（hidden/ffn/layers）与 20B/33B/13B AdaLN 的关系（4.6 的 51% 差异方向未知）。
- H3 文本编码器（Qwen3-VL-32B）与 visual/audio VAE 的真实显存占用。
- bitsandbytes 在 Windows 上能否装、QLoRA 全链路是否可用（本机缺 bnb）。
- 任何视频训练的真实峰值与步时（没有任何一次运行记录）。
- `explorer` 相关：`packages/web` 中除 `Planner.tsx` / `RunDetail.tsx` / `recipe-utils.ts` 以外的页面未逐行评审（不在本次指定范围）。