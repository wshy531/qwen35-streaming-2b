# Token压缩与读取解耦：2026-10-24固定交付

本目录用于协作审核，原模型与默认流式核心不变。当前采用决定见[阶段结果与采用决定](reports/阶段结果与采用决定.md)，先阅读[E1成本](reports/E1_auto4压缩独立成本结果.md)和[E2读取压缩](reports/E2读取时压缩结果.md)。它们说明实验目标、实际结果、逐题损失与停止条件；结果不当作未见泛化或端侧实测。

## 审核入口

- `bundle/experiments/2b-readout-compression-v1/banks.py`：三个原银行与临时半数读取；`session.py`：独立问答；`audit_and_decide.py`：预登记质量/成本门槛。
- `bundle/experiments/2b-auto4-cost-v1/`：独立三方法、两次完整重复，无共享解码/视觉。
- E1父进程EOF计数断言失败的原状态与日志保留；六个推理子进程均完成。`completed_reconciled.json`及`analyze_cost_v2.py`只恢复完整结果汇总，没有重跑推理或覆盖原记录。
- `bundle/experiments/video-input-cpu-v1/integration.py`：固定文件默认auto4，显式slice1保留。仅规范输入适配器，支持依赖路径初始化。
- `bundle/runs/`：冻结协议、输入清单、两地回执、原始回答/Token/成本与负结果。大JSON/JSONL按原字节无损gzip；manifest记录存储和原始SHA。
- `docs/`：本轮设计及阶段审视快照。它们不是另外的持续进度主记录。

## 不加载模型的核验

```bash
python verify_delivery.py
# 若已有官方2B资产，可进一步核对处理器/配置SHA：
python verify_delivery.py --model-dir /absolute/path/to/Qwen3.5-2B
```

该工具逐文件校验、检查Python语法，按冻结解析规则重算E1/E2共2046条内容/格式结果；不加载权重、使用GPU或联网。模型资产不重复上传，仅在external_assets.json记录所需元数据SHA。

## 恢复独立实验工作区

```bash
python verify_delivery.py --materialize /absolute/path/to/NEW_EMPTY_PROJECT
```

目标必须尚不存在。只恢复源码、规范输入与文本结果，不执行恢复出的Python。复现推理还需既有官方模型、按清单SHA核对的视频及匹配环境，分别置于恢复根的models、data、envs目录；可在获准项目内复用已验证资产，不必复制媒体/权重。bundle/repo是冻结版本的必要运行文件，保留它以复现历史，不覆盖当前main的活动实现。

Slurm脚本中的绝对PROJECT是原已授权集群项目。换工作区时必须先适配路径、重新冻结并做CPU预检，再获得对应GPU资源授权。GPU_authorization.json不随包分发；不会因为恢复文件而自动提交作业。

本快照从最新main的独立个人分支提交，所有新增内容位于本人目录，不改同学的记忆算法、主实现或综合汇报。
