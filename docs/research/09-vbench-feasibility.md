# VBench 环境部署可行性报告（dev4090 WSL，2026-11-23）

## 环境概况

| 组件 | 安装方式 | 状态 | 备注 |
|------|----------|------|------|
| Python 3.11 | Miniconda3 pipx | ✅ 安装成功 | Python 3.11.9 |
| torch | PyPI 默认 build (cu12x) | ✅ 安装成功 | 2.15.0+cu132，CUDA True |
| torchvision | 随 torch 安装 | ✅ | 0.29.2 |
| vbench 0.1.5 | tuna 镜像 | ✅ 安装成功 | 核心模块可用 |
| detectron2 | git clone FacebookResearch | ❌ 访问被阻断 | GitHub 443 端口不可达 |
| VBench 模型权重 | 主要来自 dl.fbaipublicfiles / GitHub releases | ⚠️ 部分不可达 | subject_consistency, dynamic_degree 等可能下载失败 |

## 受阻的维度

### 1. detectron2 依赖的语义维度（9维）
- action, object_class, scene, spatial_relationship, human_action, multiple_objects, appearance_style, color, overall_consistency
- **结论**：dev4090 无法运行这些维度，需改为离线评估或改走不同镜像/代理。

### 2. custom_input 模式的 6 个技术维度
| 维度 | 依赖模型 | 来源 | 可否访问 |
|------|----------|------|----------|
| subject_consistency | DINOv2 / OVTracker | GitHub | ❌ blocked |
| background_consistency | CLIP | HuggingFace/pytorch | ✅ |
| motion_smoothness | RAFT / UMT | GitHub | ❌ blocked |
| dynamic_degree | RAFT | GitHub | ❌ blocked |
| aesthetic_quality | aesthetic-predictor | HuggingFace | ✅ |
| imaging_quality | MUSIQ / BRISQUE | GitHub | ❌ blocked |

## 可行的替代方案

### 方案 A：本地视频评估指标（推荐）
使用 PyPI 上的纯 Python 实现：
- **CLIP 文本-图像相似度**：`timm` + `open_clip_pytorch` 实现 prompt_following 度
- **LPIPS**：`lpips` Python 实现，测量风格一致性
- **SSIM / PSNR**：`scikit-image` 测量结构相似度
- **帧间差异**：ffmpeg/`opencv` 计算运动平滑度

这些库无需 GitHub 访问，全部通过 tuna 镜像安装。

### 方案 B：云端评估
- 将生成的视频上传至 RunPod/Kaggle 等平台运行 VBench
- 费用估算：125 帧 768x512 的视频，6 维评估约 1-2 GPU 分钟 / 视频

## 推荐的 v0.3 P0 后路

**优先级**：
1. **实现本地评估器**：CLIPSscore、LPIPS、SSIM + 自制评分 rubric
2. **视频合成**：用模型生成的视频进行回归测试
3. **矩阵跑通**：多设置/rank/数据量的实验矩阵

**文件新增**：
- `engine/src/supertuner_engine/backends/video_eval.py`（本地 LPIPS/SSIM 实现）
- `engine/src/supertuner_engine/backends/clip_eval.py`（CLIP similarity 实现）
- `docs/development/eval/prompts/video_eval_v1.json`（已创建，12 条）
- 对应的事件 `video_eval_report`

## 下一步建议

若目标是 **消费卡训练 H3 LoRA**：可直接跑 v0.3 P0，评估用本地指标代替 VBench；若目标是 **填补 VBench 评测能力**：需解决 GitHub 代理问题（如 Shadowsocks、clash、v2ray）或改用云端运行。

## 实测更正（2026-11-24，代理可用后）

上面的「6 维 blocked」判断是在没有走代理时得出的，**已被实测推翻**。设置 WSL 网关代理 `http://172.28.0.1:7897`（`ip route` 的 default 网关 + 7897 端口）后，GitHub release、HuggingFace、raw.githubusercontent.com 都能访问，6 个技术维度全部跑通。

### 验证结果（dev4090，RTX 4090 24GB，VBench 0.1.5）

`vbench.evaluate(..., dimension_list=[6 技术维], mode="custom_input")` 对 3 个 768×512×125 帧测试视频完整评分，结果落 `~/vbench-eval/smoke/out/smoke_final_eval_results.json`。各维聚合分：

| 维度 | 聚合分 | 说明 |
|------|--------|------|
| subject_consistency | 0.996 | DINOv2/OVTracker，可用（静态噪点视频分数偏高属预期） |
| background_consistency | 0.997 | CLIP，可用 |
| motion_smoothness | 0.967 | RAFT/AMT，可用 |
| dynamic_degree | 0.0 | 测试视频是静态噪点，预期为 false |
| aesthetic_quality | 0.227 | LAION aesthetic predictor，可用 |
| imaging_quality | 0.302 | MUSIQ，可用 |

单视频级分数也正常写入，例如 aesthetic_quality 三条分别 0.232/0.225/0.224，imaging_quality 三条 29.7/30.6/30.2（MUSIQ 原始分）。

### 部署踩坑与解法（全部已验证）

1. **Python 版本**：torch 2.1.0+cu121 没有 cp312 轮子，vbench 锁的 transformers 4.33.2→tokenizers 0.13.3 要 Rust 编译。改用 Python 3.11 + torch 2.15.0+cu132 解决。
2. **包索引**：pypi.org 直连超时，用 `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。
3. **unzip 缺失**：vbench `utils.py:286,290` 写死 `['unzip','-d',dest,zip]`，WSL 无 unzip 且 sudo 要密码。解法：`~/.local/bin/unzip` 包装器，优先调 Windows 侧 `/mnt/c/Program Files/7-Zip/7z.exe`（`wslpath -w` 转路径，`unzip -d`→`7z x -o`），7z 不可用再退 Python zipfile。已解压 RAFT 模型验证通过。
4. **pkg_resources 缺失**：`clip` 包 `from pkg_resources import packaging`，Python 3.11 新 venv 无 setuptools。用 `uv pip install --python <venv>/bin/python "setuptools<81"` 装 80.10.2（≥81 会移除 pkg_resources）。
5. **aesthetic 权重 4KB 错误页**：vbench 旧版用 `github.com/blob/...?raw=true` 只拿到 4KB 错误页（上游 issue #187，commit ec87716 已修）。手动从 `raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth` 下载（6556 字节，torch.load 得 OrderedDict weight/bias）。
6. **MUSIQ 权重不完整**：首次下载卡在 95% 留了 103,772,258 字节的坏文件，报 `PytorchStreamReader failed finding central directory`。重新下载得完整 108,610,983 字节，torch.load 成功。
7. **后台进程被杀**：nohup/setsid 在 WSL 里随 ssh 断开被杀，且 `/tmp` 跨会话丢失。用保持 ssh 连接的前台运行（`ServerAliveInterval=30`）+ 脚本放 `~/vbench-eval/` 持久路径解决。

### 对 SuperTuner 的含义

- VBench 6 个技术维度可在 dev4090 上真实运行，足以支撑 B 路线的「技术质量 + 一致性」评估；语义 9 维仍需 detectron2（GitHub 443 直连阻断，代理下可尝试但尚未验证）。
- `evaluate()` 返回 None、结果写文件是 VBench 的正常行为，接入 evaluator 时要读结果 json 而不是拿返回值。
- 结论文件：`docs/development/eval/prompts/video_eval_v1.json`（12 条冻结视频 prompt）。