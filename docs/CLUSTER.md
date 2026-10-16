# 集群运行

## 环境与文件

使用 Python 3.11、固定依赖以及独立虚拟环境。核心依赖沿用 0.8B 已验证版本，Linux/集群仍需首次验证；不自动安装 FlashAttention/FLA。
按 README 安装 torch/torchvision；CUDA wheel 必须适配集群驱动。不要在登录节点跑模型。调度器、队列、账户、资源数量由集群实际配置决定，本仓库不硬编码 Slurm/PBS 参数。

模型固定为 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`。
完整目录含 10 个推理资产及 `DOWNLOAD_MANIFEST.json`，权重约 4.55 GB（4.24 GiB）。下载器检查官方文件大小及可用的 LFS SHA256，再生成本地清单；离线使用时逐文件核验清单。
如果计算节点无法联网，在联网节点运行 `prepare_model.py`，将完整模型目录复制到共享存储，再执行：

```bash
python scripts/prepare_model.py --model-dir /shared/models/Qwen3.5-2B --verify-only
```

只需开发 CPU 接口时可以运行 `--processor-only`，仅下载配置与 tokenizer，并生成 `PROCESSOR_MANIFEST.json`。这种目录不能进行真实权重推理。后续在同一目录不加该参数即可补全。

## 验证流程

1. 新建个人分支，记录 `git rev-parse HEAD` 和 `git status --short`。
2. `python scripts/run_cpu_checks.py --model-dir "$MODEL_DIR"`：多个独立进程分别检查根目录原版、旧基线、活动流式和证据工具。微型随机权重不代表理解能力。Linux 会跳过 Windows 专用 owner-handle 测试，应记录跳过原因。
3. 在 GPU 分配中运行 `python -m scripts.smoke_test --model "$MODEL_DIR" --dtype float32`，也可加 `--image /path/image.jpg --prompt '描述图片'`。
4. 使用 README 中同一组视频/参数，先 preflight 再 gpu。修改源码、视频或参数后必须重做预检。不能用旧仓库的预检记录放行本仓库。
5. 检查 `status.json`、`watchdog_status.json`、`summary.json`、原始回答和 `viewer.html`。正常退出、EOS、相同 Token 都不能单独作为语义正确证明。
6. 首轮通过后用 `--fps 0.2` 重做预检与 GPU，视频需超过 5 秒；观察正常等待采样时没有误判断流。原版/流式同输入对照、自然长视频和历史问答仍需单独登记与运行。

运行默认使用单 GPU FP32，直接以 FP32 加载，保留原始 FP32 参数，不先转成 BF16 再转回。没有启用多 GPU 分片或 DDP；集群资源充足不意味着当前推理代码会自动并行。
2B 包装入口保守要求加载前至少 12 GiB 可用显存与 12 GiB 可用 RAM；这是首轮固定输入的防护预算，并非实测需求或不会 OOM 的保证。若资源不满足，先调整作业分配，不修改模型精度绕过验证。底层通用入口仍允许显式资源阈值，其值进入预检证据。

GPU 命令必须在个人分支执行，`main`/`master`/detached HEAD 会被拒绝。每次新建输出目录；不要覆盖失败记录。包装器总运行时间限制 600 秒，超时退出非零并写明状态；过长视频或性能实验需要另行登记预算。

## 结果交付

新生成内容默认存入被 Git 忽略的 `outputs/`。选取小型文本结果到 `reports/<experiment-name>/` 再提交，包含：

- 模型 revision、代码 commit、GPU/驱动、Python 和依赖版本、完整命令、随机种子。
- 输入视频来源、SHA256、采样帧时间、问题与原始输出，失败、超时与跳过记录。
- 时间、峰值显存、reset、丢帧和问答前后状态检查。
- 哪些是工程检查、哪些是人工语义判断、哪些尚未运行。

`viewer.html` 含视频帧，留作本地查看，不默认上传。模型、媒体和环境不进 Git；需要共享数据时使用团队有权访问的数据目录。

## 开发接续

活动实现位于 `experiments/36tar/streaming-learning/code/qwen35-streaming-s4`。核心是注意力 KV + 卷积 + 递归混合状态，不能只按 Transformer KV cache 理解。问答从视频主状态 fork，生成不得污染主状态。
BF16 数值问题来自历史 0.8B 实验，尚未证明在 2B 消失。先建立 FP32 同输入对照，再单独分支研究 BF16、缓存优化、历史读取或 SFT。
