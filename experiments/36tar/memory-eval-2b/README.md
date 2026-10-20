# 2B评测与长短期记忆协作交付

2026-10-20固定交付快照。持续研究进度的唯一主记录在作者工作区《实验总纲.md》；本包的`records`是只读交接快照。后续进展通过新提交更新，不能把历史protocol里的pending字段当作当前状态。

## 先看什么

1. [首轮2B自然视频结果](reports/2B自然视频与长短期记忆首轮结果.md)：171题/67视频，原生与完整pair回答一致，M0/M1/M2同384视觉Token比较，森林未证明稳定更优，粗时间词门控R0降低成绩。
2. [B02短事件保护首版](reports/2B短事件保护首版结果.md)：限制开放事件连续折叠已实际运行，1242回答；首版仍无稳定能力收益，不接默认。
3. [数据目录](catalog/README.md)：124视频、385候选原题；171题实际比较不是全部385都准入。视频本体已放在集群，Git保存题目、校验清单和小型证据。
4. [首次GPU接口验收](reports/2B首次集群GPU验收结果.md)与[早期25题审核](reports/早期25题全前缀审核结果.md)。

## 目录职责

```text
catalog/                    385候选题、124媒体清单、171题准入目录与数据同步回执
reports/                    已完成结果说明，含负结果和限制
evidence/                   逐题原答、节点轨迹、评分、CPU与Slurm小证据
bundle/experiments/
  2b-natural-eval-v1/        冻结的2B自然视频/森林/诊断/R0源码
  2b-short-event-v1/         B02与蓄水池基线源码、测试、分析及原作业脚本
bundle/runs/                推理输入与依赖锁；不提供可重用GPU授权回执
verify_delivery.py          标准库校验包与复算主要分数，可附加124视频SHA检查
materialize.py              在新目录恢复实验布局；不运行模型或提交作业
records/                    总纲只读快照
PACKAGE_MANIFEST.json       本包逐文件SHA/大小；不含清单自身
```

`bundle`保持源文件原始字节，供源码锁与结果复核；不是第二套默认模型。根目录原始模型、同学的活动实现与运行入口没有被这些候选覆盖。共享帧特征与完整参照缓存属于评测开销，不能把整个进程说成只占384Token。

## 不用GPU即可复核

在仓库根目录执行：

```bash
python experiments/36tar/memory-eval-2b/verify_delivery.py
# 有PyTorch时，七项契约测试不加载真实权重：
python experiments/36tar/memory-eval-2b/bundle/experiments/2b-short-event-v1/test_policies.py
```

标准库校验会重新计算原始预测的严格格式/开头选项分数。开头选项只是明确输出选项的解析，不能当作对整段解释的语义判分。原来的严格评分没有被回写。

## 恢复到新的个人实验目录

在有模型和共享视频的Linux环境执行，替换三个实际路径；目标目录必须尚不存在：

```bash
python experiments/36tar/memory-eval-2b/materialize.py \
  --destination /absolute/path/to/new-personal-run \
  --repo "$PWD" \
  --model /absolute/path/to/Qwen3.5-2B \
  --dataset /home/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/data/2b-memory-eval-20261020
```

该操作只复制约20MiB源码和输入，并链接既有模型/视频。复制后的CPU预检可用：

```bash
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python /absolute/path/to/new-personal-run/experiments/2b-short-event-v1/evaluate.py cpu \
  --run /absolute/path/to/new-personal-run/runs/2b-short-event-inputs-20261020-01
```

CPU预检通过只说明接口。历史作业脚本含本次提交者的工作目录，旧协议也记录历史预算/截止；不要直接照抄提交。实际GPU复验应在自己的分支新建输出，先核对环境、重新说明目的/目录/资源并获得该账号使用者授权。原模型revision为`15852e8c16360a2fea060d615a32b45270f8a8fc`，FP32；本次没有训练或下载新权重。

## 协作边界

现有171题均已打开，只作开发/比较；不得重新称为新留出。后续T准入方案在`evidence/short-event-v1/validation_T_admission_protocol.json`，目标40新源/120候选题尚不是已取得数据。希望搭档优先复核关键证据时间段、歧义题与代码中的预算变化，再决定下一版改造。
