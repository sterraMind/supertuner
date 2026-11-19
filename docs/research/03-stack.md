# 03 技术栈与工程规范调研

> 调研日期 2026-11-11。版本号取自 PyPI JSON API（`https://pypi.org/pypi/<pkg>/json`）与 GitHub Releases 元数据，许可证取自包元数据或仓库 LICENSE 文件；每条结论后的 `[S#]` 对应第 6 节来源清单。`[S1]—[S6]` 为逐包查询入口。

## 1 训练/推理库清单

| 库 | 用途 | 许可证 | 稳定版本（2026-11-11 查询） | 单卡相关能力 | 风险 / 坑 | 来源 |
|---|---|---|---|---|---|---|
| torch | 训练/推理底座 | BSD-3-Clause（含第三方条款） | 2.15.0（2026-10-28） | torch.compile、SDPA、FSDP2、CUDA 官方 wheel | PyPI 上的 win 轮子是 CPU 版；CUDA 轮子必须从 download.pytorch.org 按 cu 变体装 | [S1] |
| transformers | 模型/分词/Trainer | Apache-2.0 | 5.22.2（2026-11-06） | Trainer、padding-free collator、量化加载 | 5.x 是破坏性大版本，必须锁小版本；与 peft/trl 的组合矩阵需实测 | [S2] |
| peft | 参数高效微调 | Apache-2.0 | 0.21.2（2026-10-01） | LoRA/LoHa/LoKr/IA3，可叠 4bit base | 与 transformers 5.x 的适配滞后风险 | [S3] |
| trl | SFT/DPO/GRPO | Apache-2.0 | 1.17.1（2026-10-30） | 单卡可跑小模型 GRPO | 1.x 相对 0.x API 重写，文档碎片化 | [S4] |
| accelerate | 单卡/多卡封装 | Apache-2.0 | 1.15.0（2026-09-09） | `save_state`/`load_state` 恢复 RNG 与优化器 | 状态目录布局不保证跨版本兼容 | [S5] |
| bitsandbytes | 8bit 优化器/量化 | MIT | 0.50.2（2026-08-27） | 8bit/paged AdamW、NF4 双量化 QLoRA | 优化器状态 省 ~75% 显存；<4096 元素参数仍留 32bit | [S6] |
| torchao | 训练/推理量化 | BSD-3-Clause（LICENSE 为 Meta 版权） | 0.18.0（2026-08-03） | int4/int8/fp8 权重、QAT | PyPI 只有 `py3-none-any` 与 manylinux 轮子，Windows 上部分算子缺失 | [S7] |
| diffusers | 扩散 pipeline | Apache-2.0 | 0.42.1（2026-10-27） | tiled VAE、model/sequential CPU offload | 与自研 DiT 训练循环职责重叠，只取组件不取其 Trainer | [S8] |
| DeepSpeed | ZeRO/Offload | Apache-2.0 | 0.19.9（2026-10-29） | ZeRO-Offload 把优化器状态与计算移到 CPU | Windows 仅部分支持；编译门槛高；对单卡收益不如 bnb 8bit | [S9] |
| PyTorch FSDP | 参数分片 | BSD-3-Clause | 随 torch 2.15.0 | 单卡无分片收益，仅作多卡预留 | FSDP2 API 仍在演进，别提前抽象 | [S1] |
| musubi-tuner | 视频 DiT LoRA 训练 | Apache-2.0（README 声明，仓库无 LICENSE 文件，部分目录继承上游许可证） | v0.3.6（2026-09-27） | `--blocks_to_swap`、`--fp8_base/--fp8_scaled`、12GB 可训 | 无 PyPI 包；脚本式仓库；许可证文件缺失需法务确认 | [S10] |
| diffusion-pipe | 扩散 pipeline parallel | GPL-3.0 | 无 tag（main，2026-10-28 最后 push） | 面向多卡「单卡放不下」场景 | 强依赖 DeepSpeed，README 明说原生 Windows 困难；GPL 传染 | [S11] |
| SimpleTuner | 图像/视频全量+LoRA | AGPL-3.0-or-later | 4.11.0（2026-10-28） | 配置体系成熟、自定义 tracker 插件目录 | AGPL 传染，闭源分发受限；只建议读其结构不建议内嵌 | [S12] |
| finetrainers | 视频 DiT 训练 | Apache-2.0 | 0.2.0（2025-04-25） | DCP 状态、accelerate 配置矩阵 | 2025-04 后无新版本，依赖已旧 | [S13] |
| veRL | RL(GRPO/PPO) | Apache-2.0 | 0.9.1（2026-09-20） | 小模型单卡可跑 | 主线按多卡+Ray 设计，单卡是次要路径 | [S14] |
| OpenRLHF | Ray 编排 RLHF | Apache-2.0 | 0.11.2（2026-09-14） | 理论可单卡 | Ray 依赖重，单卡不划算 | [S15] |
| torchtune | 单卡 recipe 微调 | BSD-3-Clause | 0.6.1（2025-04-07） | recipe + config 结构干净，适合抄设计 | 版本停滞，新模型支持慢 | [S16] |
| Lightning | Trainer 抽象 | Apache-2.0 | 2.6.6（2026-09-10） | 单卡训练循环抽象完整 | 与 accelerate/HF 生态重叠，插件化时会绑死回调体系 | [S17] |

## 2 插件架构最佳实践

### 2.1 Python entry points（进程内发现的事实标准）
`importlib.metadata` 已取代 `pkg_resources`，按 group + name 枚举已安装分发的入口点 [S18][S19]：

```toml
# 插件包 pyproject.toml
[project.entry-points."supertuner.trainers"]
qwen_image = "supertuner_qwen.trainer:QwenImageTrainer"
```

```python
from importlib.metadata import entry_points
eps = entry_points(group="supertuner.trainers")   # 3.10+ 支持 group 关键字
for ep in eps:                                    # ep.value 可先做静态校验，再 ep.load()
    spec = ep.load()
```
关键约束：`ep.load()` 会立即 import 整个依赖树，一个插件的 `ImportError`/CUDA 初始化就把整个宿主进程带坏；因此「发现」与「加载」必须分成两步，发现阶段只读 `ep.name/ep.value` 与插件清单（JSON），加载阶段放到隔离进程。

### 2.2 ComfyUI custom nodes（目录扫描式）
ComfyUI 启动时扫描 `custom_nodes` 目录，把每个模块当 Python 模块 import，模块导出 `NODE_CLASS_MAPPINGS`（dict，名→类）才算节点；**import 出错时 Comfy 继续启动，只把该模块标记为加载失败** [S20]。这证明目录扫描能容忍加载期失败，但运行期（OOM、非法张量）无隔离，与 2.1 的结论一致。

### 2.3 Kohya / SimpleTuner 的配置方式
- Kohya sd-scripts：训练参数集中在 **TOML 或 JSON** 配置文件，用 `--config_file` 指向，官方推荐 TOML [S21]。
- SimpleTuner：`CONFIG_FILENAMES = ("config.json", "config.toml", "config.env")`，并有独立 `loader.py / validator.py / registry.py / json_file.py / toml_file.py / cmd_args.py` 分层（`simpletuner/helpers/configuration/`），三种格式优先级与路径回退写死在 loader 里；插件扩展点是一个目录：`simpletuner/custom-trackers/` + `--report_to=custom-tracker --custom_tracker=<name>` [S12]。
即：**文件格式可以有多种，但必须收敛到一个 loader + 一个 validator + 一份 registry**，否则配置分支会失控。

### 2.4 pytest plugin 模式
pytest 自身就是 entry point 插件系统：`pytest11` group 的入口点会在启动时被加载，插件再通过 `pytest_*` 钩子介入；项目内插件则用 `conftest.py` [S22]。可借鉴的是「钩子 + 注册表」而不是「继承基类」，并且插件必须能被 `-p no:<name>` 关掉。

### 2.5 「进程内 import」vs「独立进程 + JSON 协议」

| 维度 | 进程内 import 插件 | 独立进程 + JSON/JSONL 协议 |
|---|---|---|
| 崩溃隔离 | 无，插件 panic/OOM 拖垮宿主 | 强，子进程死宿主只收退出码 |
| 依赖冲突 | 一个 venv 里共存，torch/CUDA 版本互相打架 | 每插件可独立 venv |
| 热更新 | 需重启宿主 | 重启子进程即可 |
| 调用开销 | 函数调用级 | 序列化 + 进程启动（百毫秒级） |
| 观测 | 日志混在一起 | stdout/stderr 天然分流，可按 run 收集 |
| 复杂度 | 低 | 中：协议、心跳、超时、僵尸进程回收 |
| 适合的扩展点 | 配置 schema、CLI 适配器、指标后端、纯数据处理 | 训练循环、模型构建、CUDA 内核、长任务 |

**结论（推荐）**：双轨。①**发现层进程内**：entry points 只声明插件清单（名字、版本、支持的 engine 类型、所需显存档位、schema 版本），不 load 重依赖；TS 服务端读清单渲染 UI。②**执行层独立进程**：每个训练 run 就是 `python -m supertuner_engine run --config <json>` 子进程，插件在子进程内 import，与宿主用 **JSONL（事件）+ JSON（控制）over stdio** 通信，run_id 作为关联键。这样既保留插件生态的易安装性，又把「单卡训练崩溃」限制在子进程内——这与 ComfyUI/Kohya 把风险留在主进程的做法相反，是更符合「长时间单卡任务」的取舍。

## 3 训练运行时的工程规范

1. **配置 schema**：pydantic v2（2.14.3）+ pydantic-settings（2.16.2）[S30]做「环境变量为唯一入口」的 typed 层：`Settings` 用 `env_prefix="ST_"`、`env_nested_delimiter="__"`，启动时 `model_dump()` 出**生效配置**并算 sha256 写进 run 目录与 checkpoint。纯环境变量的问题是无嵌套、无类型、无默认值文档、50+ 超参无法版本化；pydantic 还能直接 `model_json_schema()` 给 TS 侧生成 zod schema，保证前后端一份真源。dataclass 只用在热循环内部的小结构（如 batch 描述）。
2. **事件流**：引擎 → 服务端用 **JSONL 追加写**（每行 `{ts, run_id, seq, type, payload}`，`type` ∈ progress/metric/checkpoint/log/error），理由是可 `tail -f`、可断线按字节偏移续读、崩溃后文件仍可解析；服务端 → 浏览器再转 **SSE**。要设 `PYTHONUNBUFFERED=1`，Windows 上显式 `\n` 不用 `\r\n`，并对单行长度设上限（异常 traceback 截断）。
3. **检查点元数据**：safetensors 的 `__metadata__` 只允许 **string→string** map [S23]，因此元数据里只塞字符串：`run_id / step / seed / config_sha256 / base_model_fingerprint / peft_type / rank / dataset_fingerprint`。musubi-tuner 就是这么做的：`src/musubi_tuner/minimax_h3/checkpoint.py` 里 `load_safetensors_metadata()` 合并分片元数据、`fingerprint_checkpoint()` 用「分片名 + 文件指纹」的 sha256 判断缓存是否过期。
4. **优化器状态单独存**：8bit 优化器状态不是干净的连续张量，不能进 safetensors；用 `torch.save` 或 `torch.distributed.checkpoint`（DCP）。finetrainers 的 `finetrainers/state.py` 定义 `TrainState(dataclass, torch.distributed.checkpoint.stateful.Stateful)`，只把 `step / observed_data_samples / 历史 loss / log_steps` 做状态化，这个粒度和思路值得照搬。
5. **断点续训**：用 `accelerator.load_state(dir)` / `register_save_state_pre_hook` 恢复模型+RNG+优化器+step（musubi-tuner `src/musubi_tuner/training/trainer_base.py:492` 的 `resume_from_local_or_hf_if_specified` 与 `:1898` 的 hook 注册就是范例）；裸训循环则必须自己存 `torch.get_rng_state()`、`torch.cuda.get_rng_state_all()`、dataloader 的 epoch+sample offset。HF Trainer 的 `resume_from_checkpoint` / `save_only_model` 是同语义参照 [S24]。
6. **随机种子与可复现**：一次 run 一个 `seed`，派生 `data_seed / sampler_seed / dropout_seed`；DataLoader `worker_init_fn` 里按 `base_seed + worker_id` 重建 generator。`torch.use_deterministic_algorithms(True)` 会让部分 CUDA 算子直接报错或显著变慢，建议做成 `ST_DETERMINISTIC=1` 开关而不是默认开；文档里明确「同 seed + 同驱动 + 同 cu 变体才可复现」。
7. **数据集格式**：元数据 sidecar 用 **JSONL**（一行一样本：路径、caption、分辨率、时长、可选 crop/桶信息），musubi-tuner 的 `docs/dataset_config.md` 就是 TOML 数据集定义 + caption txt 或 metadata JSONL，并明确**每个数据集的 latent cache 目录必须独立**，否则缓存串样本；百万级表格元数据用 **parquet**；分片流式/多机用 **WebDataset**。视频务必做 latent/文本 embedding 预缓存，训练循环只读缓存。
8. **测试策略**：CPU 小模型 smoke（tiny DiT/1 层 LLM、2 step、1 batch，断言 loss 有限、checkpoint 可读回）——musubi-tuner 的 `tests/` 全是这类可离线单测（如 `tests/test_krea2_gradient_checkpointing.py`）；GPU 测试用 `pytest -m gpu` + 无卡自动 skip；fixture 用真数据小样本，finetrainers 的 `assets/tests/metadata.csv` + `assets/tests/videos/hiker_tiny.mp4` 是现成范式；插件要有 conformance 测试集，每个插件跑同一组契约用例。

**可直接参考的三个真实代码结构**
- **musubi-tuner**（`https://github.com/kohya-ss/musubi-tuner` [S10]）：`src/musubi_tuner/` 按模型族分包（`wan/ hunyuan_model/ qwen_image/ frame_pack/ …`）+ 共享 `training/ dataset/ networks/ utils/ gui/`；顶层是极薄的 `*_train_network.py`、`*_cache_latents.py`、`*_generate_image.py` 脚本。优点是「新模型=新子包+三个薄脚本」，缺点是脚本与库边界靠约定，无插件注册表。
- **finetrainers**（`https://github.com/huggingface/finetrainers` [S13]）：`finetrainers/{config.py,args.py,state.py,trackers.py,optimizer.py,parallel/accelerate.py,utils/activation_checkpoint.py,models/,data/,processors/}` + `accelerate_configs/*.yaml` + `tests/`；分层最清晰，适合做「引擎内核」的骨架。
- **SimpleTuner**（`https://github.com/bghira/SimpleTuner` [S12]）：`simpletuner/helpers/configuration/*` 配置分层 + `simpletuner/custom-trackers/` 插件目录 + `simpletuner/simpletuner_sdk/` 服务端 SDK；在产品化（多用户、WebUI、tracker 扩展）上参考价值最高，但许可证是 AGPL。

## 4 单卡省显存技术清单

| 技术 | 适用阶段 | 收益 | 代价 | 备注 |
|---|---|---|---|---|
| gradient checkpointing | 训练前向/反向 | 激活显存显著下降（层数越多越明显） | 反向重算，约 +20–30% 时间；需模型支持 `gradient_checkpointing_enable` | musubi `--gradient_checkpointing` [S10] |
| 8bit / paged 优化器 | 优化器 step | 优化器状态 ~75% 显存下降 [S6] | 少量精度/速度差；paged 版防 OOM 抖动但需 CUDA 内核 | `<4096` 元素参数仍 32bit，可调 `min_8bit_size` [S6] |
| LoRA / QLoRA | 训练 | 可训练参数 <1%，4bit base（NF4 双量化） | 4bit 反量化有速度损失；合并/推理需还原精度 | 视频 DiT 张量大，LoRA 是单卡主要路线 |
| fp8 权重（fp8_base/fp8_scaled） | 前向 | 基座权重显存减半 | 需要 fp8 硬件（Ada/Hopper）；30 系无原生 fp8，退回 int8/convrot | musubi `--fp8_base`；`--convrot_int8` 面向无 fp8 的卡 [S10] |
| block swap / CPU offload 权重 | 前向 | 让 12GB 卡训大 DiT [S10] | 传输开销、速度降；需足够主内存 | musubi `--blocks_to_swap`；`--use_pinned_memory_for_block_swap` 在 **Windows 会增加共享显存占用** [S25] |
| ZeRO-Offload（优化器到 CPU） | 优化器 step | 官方称单卡 V100 可训 10B/13B 级 [S9] | 依赖 DeepSpeed、Windows 仅部分支持、CPU-GPU 通信开销 | 备选，不进主路径 |
| diffusers CPU offload | 推理/采样 | 权重按需搬 GPU [S8] | `sequential_cpu_offload` 极慢 | 只在采样/预览用 |
| tiled VAE | VAE 编解码 | 峰值显存下降，可解高分辨率 [S8] | 分块间可能有轻微色调差；`use_stream` 需先 dummy forward | 训练前 latent 预缓存时用 |
| chunked / fused linear CE | LM 训练 | 避免 `BT×V` 大 logits 物化 | 实现复杂；部分融合核平台受限 | Liger-Kernel 的 fused linear cross entropy 是现成实现 [S26] |
| FlashAttention / SageAttention / xformers | 前向+反向 | 更快且更省注意力显存 | 编译与版本匹配成本高（见第 5 节） | 基线用 torch SDPA，注意力核当作可选加速 |
| sequence packing（padding-free） | 数据整理 | 吞吐最多约 2×，无 padding 浪费 [S27] | 需 varlen attention 与正确的 position_ids/边界处理 | transformers `DataCollatorWithFlattening` + FA2 varlen [S27][S28] |

## 5 推荐技术选型与理由

| 场景 | 推荐 | 备选 | 理由 |
|---|---|---|---|
| 引擎底座 | torch 2.15.0 + accelerate 1.15.0 + 自研 trainer loop | Lightning 2.6.6 | 插件化要求控制权在自己手里；Lightning 的回调体系会把宿主绑死 |
| LLM 微调 | transformers 5.22.2 + peft 0.21.2 + bnb 0.50.2（QLoRA） | torchtune 0.6.1（仅参考其 recipe 结构） | 生态最全；torchtune 停滞 |
| 视频 DiT 训练 | 参考 musubi-tuner 的 block swap + fp8 设计，自研实现 | SimpleTuner（仅读代码，AGPL 不外链） | 要拿到 12GB 档能力又不想背 GPL/AGPL |
| RL | trl 1.17.1（GRPO，小模型单卡） | veRL 0.9.1（多卡再上） | 单卡场景 trl 依赖最轻 |
| 插件 | entry points（发现）+ 子进程 JSONL（执行） | 纯进程内 import（仅限无 CUDA 的扩展点） | 见 2.5，崩溃隔离是硬需求 |
| schema 同步 | pydantic `model_json_schema()` → TS zod | 手写双份 schema | 一份真源，避免前后端漂移 |
| 配置入口 | 环境变量 + pydantic-settings 校验 | TOML 文件（仅测试/示例） | 符合「配置全走环境变量」，但要有 typed 层与生效配置快照 |

**Windows + NVIDIA 消费卡的坑（逐条实测结论）**
- **torch 轮子与驱动强绑定**：`download.pytorch.org/whl/cu132/torch/` 才有 2.15.0 的 `win_amd64` 轮子（cp311–cp314）；**cu128 索引最高只到 2.11.0**，2.12+ 只发 cu126/cu130/cu132。CUDA 13.x 最低驱动 580.65.06 [S29]。→ 先查用户驱动，再决定 torch 版本，别在 CI 里拍死一个 cu 变体。
- **bitsandbytes**：0.50.2 提供 `bitsandbytes-0.50.2-py3-none-win_amd64.whl` [S6]，Windows 可用；但 8bit 优化器/paged 版依赖与 CUDA 运行时匹配，且 WDDM 下更容易触发共享显存回退，需在 UI 里把「QLoRA 需要 ≥8GB 且关闭其他显存占用」写清楚。
- **xformers**：0.0.35 有 `win_amd64` 轮子，但声明 `torch>=2.10` 且与具体 torch 构建绑定 [S31]，torch 小版本一升就可能 `undefined symbol` → 列为可选加速，装不上就退 SDPA。
- **FlashAttention**：PyPI 只有 sdist，README 明说「Linux；Windows 从 v2.3.2 起可能可用，但 Windows 编译仍需更多测试」[S32] → **不在 Windows 默认依赖里**，需要 MSVC + CUDA Toolkit 才能编译，用户装不上就是灾难。
- **SageAttention**：主线 2.x 需 `pip install sageattention==2.2.0 --no-build-isolation`，要 CUDA Toolkit 编译，并依赖 `triton`；Windows 需 `triton-windows`（3.9.0.post30）[S33][S34] → 只做「高级选项」，默认关。
- **DeepSpeed**：官方仅部分支持 Windows，diffusion-pipe README 直接写「原生 Windows 上很难或不可能跑起来」[S11] → 单卡省显存用 bnb 8bit + block swap，不引入 DeepSpeed。
- **进程与路径**：WDDM 显存回收慢、杀软扫描大文件、长路径 → 每个 run 独立子进程 + 独立输出目录，退出即回收。

## 6 来源（查询日期 2026-11-11）

- [S1] https://pypi.org/pypi/torch/json ；https://download.pytorch.org/whl/cu132/torch/ ；https://download.pytorch.org/whl/cu128/torch/
- [S2] https://pypi.org/pypi/transformers/json ；https://github.com/huggingface/transformers/releases/tag/v5.22.2
- [S3] https://pypi.org/pypi/peft/json ｜ [S4] https://pypi.org/pypi/trl/json ｜ [S5] https://pypi.org/pypi/accelerate/json
- [S6] https://pypi.org/pypi/bitsandbytes/json ；https://huggingface.co/docs/bitsandbytes/main/en/optimizers
- [S7] https://pypi.org/pypi/torchao/json ；https://github.com/pytorch/ao
- [S8] https://pypi.org/pypi/diffusers/json ；https://huggingface.co/docs/diffusers/main/en/optimization/memory
- [S9] https://pypi.org/pypi/deepspeed/json ；https://www.deepspeed.ai/tutorials/zero-offload/
- [S10] https://github.com/kohya-ss/musubi-tuner ；https://github.com/kohya-ss/musubi-tuner/releases/tag/v0.3.6 ；源码 `src/musubi_tuner/minimax_h3/checkpoint.py`、`src/musubi_tuner/training/trainer_base.py`、`docs/dataset_config.md`
- [S11] https://github.com/tdrussell/diffusion-pipe （README「Windows support」节；GPL-3.0）
- [S12] https://github.com/bghira/SimpleTuner ；https://github.com/bghira/SimpleTuner/releases/tag/v4.11.0 ；`simpletuner/helpers/configuration/loader.py`
- [S13] https://github.com/huggingface/finetrainers ；https://pypi.org/pypi/finetrainers/json ；`finetrainers/state.py`
- [S14] https://pypi.org/pypi/verl/json ｜ [S15] https://pypi.org/pypi/openrlhf/json ｜ [S16] https://pypi.org/pypi/torchtune/json ｜ [S17] https://pypi.org/pypi/lightning/json
- [S18] https://docs.python.org/3/library/importlib.metadata.html
- [S19] https://setuptools.pypa.io/en/latest/userguide/entry_point.html
- [S20] https://docs.comfy.org/custom-nodes/backend/lifecycle
- [S21] https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md
- [S22] https://docs.pytest.org/en/stable/how-to/writing_plugins.html
- [S23] https://github.com/huggingface/safetensors#format （`__metadata__` 仅 string→string）
- [S24] https://huggingface.co/docs/transformers/en/main_classes/trainer （`resume_from_checkpoint`/`save_only_model`）
- [S25] https://github.com/kohya-ss/musubi-tuner/blob/main/docs/block_swap.md
- [S26] https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py
- [S27] https://huggingface.co/blog/packing-with-FA2 （packing 最高约 2× 吞吐）
- [S28] https://huggingface.co/docs/transformers/padding_free
- [S29] https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html （CUDA 13.0 最低驱动 580.65.06）
- [S30] https://pypi.org/pypi/pydantic/json ；https://pypi.org/pypi/pydantic-settings/json
- [S31] https://pypi.org/pypi/xformers/json
- [S32] https://github.com/Dao-AILab/flash-attention （README「Requirements」；PyPI 仅 sdist）
- [S33] https://github.com/thu-ml/SageAttention （`pip install sageattention==2.2.0 --no-build-isolation`）
- [S34] https://pypi.org/pypi/triton-windows/json