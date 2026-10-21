# qwen35-streaming-2b

基于 **Qwen/Qwen3.5-2B 官方权重**的流式视频理解协作仓库，运行目标是 Linux GPU/超算集群。

继承 [qwen35-streaming](https://github.com/wshy531/qwen35-streaming) 主线提交 `c32a5951622dc6ce9e7e770b457af77cca7a883d`（已合并 PR #6），保留原作者提交与 0.8B 实验记录。
**2026-10-20已完成2B真实GPU接口验收及171题自然视频比较。** 原生完整与完整pair回答在该固定输入上相同；原生严格正确107/171。长短期记忆及B02短事件保护均已实际测试，尚未证明稳定优于简单最近窗口，候选不默认开启。旧0.8B历史结果仍单独保留。

最新协作交付见[2B长视频与记忆实验复核入口（2026-10-21）](experiments/36tar/long-memory-review-2b/README.md)：新增39段/98题来源验证、6段长视频中途问答、旧记忆可读性诊断，以及A06/S24与容量对照；包含1262条历史原答的独立复分入口。尚未证明森林稳定优于近期窗口，默认路径不变。新的小时级测试由用户暂缓，视频尚未下载，本阶段GPU尚未提交。

全部数据的统一入口见[数据集总目录：批次、题型与时长](experiments/36tar/dataset-catalog-20261021/catalog/数据集总目录.md)：182个既有视频、6张控制照片，519道官方原题及人工/延迟诊断条目；候选全集和171/98/34实际子集分别列明。目录及既有媒体已同步集群并逐SHA核验，Git只保存目录与题目；未下载的小时级计划仍单列为暂停。

前次交付见[2B评测与长短期记忆同步入口](experiments/36tar/memory-eval-2b/README.md)：包括源码、逐题原答、负结果、标准库复核工具，以及124视频/385候选题的数据清单。视频本体已在授权集群目录逐SHA校验，Git中不存媒体、模型或环境。

2026-10-20本次交付：仓库要求的209项CPU回归与新增7项记忆契约均通过；标准库交付工具可独立校验文件和复算分数。测试输出保存在交付包的`evidence/publication`。

2026-10-16 本机 CPU 检查：**209 项通过，0 失败/错误/跳过**。使用 2B 官方处理器及微型随机模型，未下载 2B 权重；[机器可读摘要](reports/bootstrap-2b/cpu_summary.json)与[测试输出](reports/bootstrap-2b/cpu_console.txt)。Linux CI 结果以 GitHub Actions 为准。

## 开发入口

```text
qwen35_local/                      原始模型实现，用于官方基线
scripts/prepare_model.py            固定版本下载 / 离线完整性校验
scripts/smoke_test.py               2B 文本 / 单图基线
scripts/run_cpu_checks.py           CPU 小模型回归，不加载预训练权重
scripts/run_2b_video.py             2B 预检 / FP32 文件流式运行
requirements-cluster.txt            集群 Python 依赖
docs/CLUSTER.md                     完整部署、运行与结果回收步骤
CONTRIBUTING.md                     分支、PR 和证据约定
experiments/36tar/streaming-learning/
  code/qwen35-streaming/            旧原始实现对照
  code/qwen35-streaming-s4/         活动流式实现，开发主要在这里
  实验总纲.md                      统一进度记录（包含历史）
```

没有复制整个 Transformers。模型源码固定源自 Transformers v5.3.0，许可与来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型权重、视频、环境和下载缓存不提交 Git。

## 集群快速开始

以下命令在 Bash 中运行。安装和下载在允许联网的节点进行；**推理只在已分配的 GPU 计算资源中运行**。

```bash
git clone https://github.com/wshy531/qwen35-streaming-2b.git
cd qwen35-streaming-2b
git switch -c yourname/2b-baseline
python3.11 -m venv .venv
source .venv/bin/activate
# 示例 CUDA wheel；先按集群驱动/管理员要求选择兼容构建。
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-cluster.txt

export HF_HOME="$PWD/.cache/huggingface"
export MODEL_DIR="$PWD/models/Qwen3.5-2B"
python scripts/prepare_model.py --model-dir "$MODEL_DIR"
python scripts/run_cpu_checks.py --model-dir "$MODEL_DIR"
```

模型固定在官方 revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，完整下载生成 `DOWNLOAD_MANIFEST.json`；离线节点可以直接使用校验后的完整目录。

申请 GPU 后，先做原版短输入验证，再运行相同输入的两阶段流式入口：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.smoke_test --model "$MODEL_DIR" --dtype float32
export VIDEO=/absolute/path/to/short-video.mp4
python scripts/run_2b_video.py --stage preflight --model-dir "$MODEL_DIR" --video "$VIDEO" --out outputs/2b-preflight-01
python scripts/run_2b_video.py --stage gpu --model-dir "$MODEL_DIR" --video "$VIDEO" --preflight outputs/2b-preflight-01 --out outputs/2b-gpu-01
```

每次使用新输出目录。预检与 GPU 的源码、模型文件、视频和参数必须一致；CPU 预检只证明接口与调度契约，回答没有语义意义。GPU 输出包括 `summary.json`、`events.jsonl` 和 `viewer.html`，下载 HTML 可离线查看。

流式入口先沿用 FP32，不提前启用尚未验证的 BF16 混合状态路径。默认有限运行：1 FPS、最多 8 帧、每答最多 16 tokens、上下文 288 tokens、600 秒总超时；容量耗尽会明确 reset。资源门槛是预检保护，不是实测显存需求或性能承诺。详见 [集群说明](docs/CLUSTER.md)。

## 多人协作与下一步

- 从最新 `main` 建个人功能分支，通过 PR 合并，不直接往主线堆实验。
- 优先顺序：集群环境 → 官方 2B 基线 → 原版/流式同输入对照 → 真实视频质量 → 精度与性能优化。
- 历史读取、帧库和压缩候选保留为实验，不默认开启；本轮不加入 SFT。
- 每次记录模型 revision、代码 commit、命令、输入哈希、环境和失败信息，区分 CPU 检查与真实 GPU 结果。

[协作约定](CONTRIBUTING.md) · [继承来源](docs/PROVENANCE.md) · [0.8B 历史 README](README_08B.md)
