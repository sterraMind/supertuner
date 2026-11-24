# dev4090 VBench 评测运行手册

在 dev4090（RTX 4090 24GB，Ubuntu 24.04 WSL）上用 VBench 0.1.5 跑视频评测的可复用步骤。2026-11-24 实测 6 个技术维度全部跑通。

## 环境

- VBench venv：`~/vbench-eval/.venv`（Python 3.11.9、torch 2.15.0+cu132、vbench 0.1.5、detectron2 未装）
- 元数据：`~/vbench-eval/VBench_full_info.json`
- 代理：WSL 默认路由网关 + 宿主 7897 端口，即 `http://$(ip route | awk '/^default/{print $3;exit}'):7897`
- 包索引：`UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`（pypi.org 直连超时）

## 一次性修复（已在 dev4090 完成，换机器/venv 重建时需重做）

1. Python 3.11：torch 2.1.0+cu121 无 cp312 轮子，vbench 锁的 tokenizers 0.13.3 要 Rust 编译。用 3.11 + torch 2.15.0+cu132。
2. `pkg_resources`：`uv pip install --python ~/vbench-eval/.venv/bin/python "setuptools<81"`（clip 包需要，≥81 移除该模块）。
3. unzip：跑 `install-unzip.sh`（VBench 写死 `unzip -d`，WSL 无该命令且 sudo 要密码）。
4. aesthetic 权重：从 `https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth` 下载到 `~/.cache/vbench/aesthetic_model/emb_reader/`（旧 URL 只返回 4KB 错误页，见上游 issue #187 / commit ec87716）。
5. MUSIQ 权重：从 `https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth` 下到 `~/.cache/vbench/pyiqa_model/`，完整大小 108,610,983 字节（中断会留下不完整的坏文件，导致 `PytorchStreamReader failed finding central directory`）。

## 运行

SSH 必须保持连接（WSL 里 nohup/setsid 会随 ssh 断开被杀，且 /tmp 跨会话丢失）。用后台 job 维持一个前台 ssh：

```
ssh -o ServerAliveInterval=30 dev4090 "bash -lc 'bash ~/vbench-eval/smoke-vbench.sh; echo EXIT=\$?'"
```

脚本放 `~/vbench-eval/` 持久路径（不要放 /tmp）。日志：`~/vbench-smoke-final.log`。

## 结果

结果写到 `~/vbench-eval/smoke/out/<name>_eval_results.json`。`VBench.evaluate()` 返回 None，必须读文件。6 维聚合分（静态噪点测试视频，仅验证流程）：

| 维度 | 分数 |
|------|------|
| subject_consistency | 0.996 |
| background_consistency | 0.997 |
| motion_smoothness | 0.967 |
| dynamic_degree | 0.0 |
| aesthetic_quality | 0.227 |
| imaging_quality | 0.302 |

语义 9 维需 detectron2（GitHub 443 直连阻断，代理下未验证）；custom_input 模式只支持上述 6 维。
