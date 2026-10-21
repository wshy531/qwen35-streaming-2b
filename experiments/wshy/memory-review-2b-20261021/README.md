# 有界外部视觉记忆与长短期策略：阶段交付

2026-10-21。开发基础为 `bf14b753d17f73142ae67e0c337d7936e16b4cbe`；发布分支从 PR #2 合并后的 `df83acb7441b3d035d621616348c2568343a39df` 建立。原工作区及其未提交内容保留。

本阶段已经结束实验，交付实现、历史运行证据和离线复核入口。没有训练、调参或变更默认模型/演示；LS 和 A06 均须显式选择。归档不授予继续连接集群、下载、提交或重投作业的权限。

## 先看结论

- 工程上完成逐帧对编码一次、有界外部视觉银行、每问新建语言状态、问答隔离、reset 与连续查询接入。
- 171 题六方法对照：M0/M2/A06/LS/固定蓄水池/纯文字严格匹配分别为 106/107/109/109/109/66。LS 对 M2 修正 5、退化 3，未稳定超过蓄水池。
- 六长视频原题 28：相同顺序为 14/12/14/16/16/12；人工开发六题 LS 为 4，其余为 3。严格格式收益、纯文字先验、近期证据和未独立审核标签均单列，不能宣布长期记忆能力已稳定提升。
- 机制正确性与能力收益分开。保留 LS 为可选候选，默认策略不变；本轮不继续算法或 GPU 尝试。

详细设定、逐题得失、负结果、结构代价、资源和失败记录见 [RESULTS.md](RESULTS.md)。队友成果见 [36tar 阶段归档](../../36tar/long-memory-review-2b/README.md)，数据身份及集合关系沿用 [统一目录](../../36tar/dataset-catalog-20261021/README.md)，不重复计算题量。

## 源码入口

- [online_memory_2b](../online_memory_2b/README.md)：reset/append_pair/ask/status/close，M0/M2/蓄水池、独立 A06、LS；固定 171 题四阶段入口为 `scripts/run_2b_memory_eval.py`。
- [long_video_2b](../long_video_2b/README.md)：连续查询、真实 PTS/RGB 身份和未配对帧管理，独立 CPU prepare/preflight。
- [long_video_gpu_2b](../long_video_gpu_2b/README.md)：绑定 CPU217847 的固定 34 题六方法 GPU/score 扩展。
- [long_stream_audit](../long_stream_audit/README.md)：较早的不完整 A06 取回证据审计，保留历史用途；当前完整分数以本交付为准。

## 不需要模型的复核

在仓库根目录运行 Python 3.11+：

```bash
python -I -S -B experiments/wshy/memory-review-2b-20261021/verify_delivery.py
```

仅用标准库，读取 Git 内文本与 gzip；不联网、不读取视频、不写缓存、不导入模型、不加载权重、不执行任何提交脚本。独立复算 1914 条历史主回答，核对 192 条不计准确率的前缀检查、逐帧对记录、长视频查询身份、银行元数据、输出 SHA 和执行源码绑定。

输出中的 `pretrained_loads=0` 等表示本次离线核验；历史三轮 GPU 各实际载入一次，由各自 completed.json 记录。回答数不是独立题数，三个运行的分数不合并成新准确率。

## 证据布局与身份

- [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json)：158 个原始文本文件的原字节数/SHA，以及实际存储字节数/SHA。大文本使用无损 gzip，原 JSON/JSONL 未改写。
- `evidence/gpu-replay-217796/`、`evidence/ls-gpu-217845/`、`evidence/long-video-gpu-217852/`：输入、原答、逐帧对记录、完成回执、评分和人工待审清单。
- `evidence/cpu-release-217794/`、`evidence/ls-cpu-217843/`、`evidence/long-video-cpu-217847/`：历史 CPU 日志和回执；217790、217792 失败及本机模拟夹具失败也保留。
- `audits/`：当时独立复核与结构/得失分析 JSON。`historical_tools/` 保存原本机诊断源码，它们依赖原隔离目录布局；可移植入口为本目录 verify_delivery.py。
- [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)：当前执行源码与 LS 的 134 条、长视频的 141 条 CPU 锁，以及 GPU 扩展锁均一致。217796 是更早版本，其中四个文件后来扩展过，不能用其旧源码锁证明最终版本。
- [SLURM_RECEIPTS.json](SLURM_RECEIPTS.json)：从已取回日志摘录的完成行及原日志 SHA，均为历史回执，没有新查调度器。

不保存权重、视频、照片、帧图、NPZ、环境、依赖、凭据或可复用 GPU 授权。历史绝对路径只作运行来源，不表示其他机器已经部署；运行脚本不能直接用这些旧回执重交。

## 复现边界

FP32、192×256、原事件阈值 `0.047536224126815796`、原合并公式/位置/提示词保持；模型 revision 为 `15852e8c16360a2fea060d615a32b45270f8a8fc`。171 题为已经打开的开发数据；长视频六来源和人工题也已经打开，独立像素/语义复核仍为 pending。

本机写入边界为 `D:/AI_project/projects/streaming_video_wsy`，集群为 `/home/share/czy/wsy`。运行入口会检查独立内部 `.git`、真实解析路径和缓存边界；不是任意目录可直接启动的通用 CLI。共享环境仅可在兼容且获批时复用，所有集群操作另行授权。
