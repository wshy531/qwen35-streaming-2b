# qwen35-streaming-2b

基于 **Qwen/Qwen3.5-2B** 的流式视频理解研究仓库：逐帧对接收视频，用有限的视觉记忆回答当前及历史问题，同时探索视觉Token压缩的质量与成本取舍。

**结果更新：2026-11-05，依据已合入主分支的阶段交付（截至PR #8）。** 以下是历史实验汇总，不表示本次重新运行模型。早期Qwen3.5-0.8B实现与作者提交历史继续保留，见[继承来源](docs/PROVENANCE.md)及[0.8B记录](README_08B.md)。

## 当前结论

- **工程机制已验证，长期记忆收益仍未稳定。** 已实现有界外部视觉银行、每问独立语言状态、问答隔离与reset；保存更久不保证答得更准。
- **低分不只来自记忆策略。** S-EMBER上的原模型也有较多错误；NExT-QA和StreamingBench的匹配输入对照中，视觉组明显高于纯文字。不同数据集、输入范围和协议的分数不能直接比较。
- **压缩能减少Token，但质量与实际加速要分别测量。** 中层剪枝已有约7%的完整流程省时证据，也存在答题损失；训练后的选择器尚未稳定超过简单固定选择。
- **最新开放式问答中，原模型得分高于LS、A06和纯文字。** 原模型读取完整采样前缀，记忆组受预算限制；结果不能单独归因为某种记忆算法。

## 实验推进与主要结果

### 1. 建立流式接口，再比较长期记忆

在171道已打开的自然视频题上，M0最近窗口、M2事件森林、A06、LS、固定蓄水池和纯文字的历史匹配正确数分别为106、107、109、109、109、66。六个长视频的28道原题分别为14、12、14、16、16、12。LS相对M2有修正也有退化，尚未稳定超过简单蓄水池。

这一阶段证明了逐帧对编码、查询后继续接收、银行容量和状态隔离等机制；不能据此证明长期记忆能力已经解决。历史分数保留原评分协议，与后续开放式语义分数分开。

[在线记忆实现与完整结果](experiments/wshy/memory-review-2b-20261021/README.md) · [历史可读性与A06对照](experiments/36tar/long-memory-review-2b/README.md)

### 2. S-EMBER低分：检查读取链路和输入质量

128视频478题的完整流式历史参照为135/478，纯文字106/478。这一参照不是官方原生接口。另在21视频24题的同帧审计中，官方两种读取与流式实现均为9/24；该小面板未观察到答案退化，但不能排除其他输入上的对齐问题。

102道开发题、18视频、每题五次选项轮换的证据片段实验中：

- 192×256：LS 141/510，A06 144/510，原生无记忆145/510。
- 288×384：LS 170/510，A06 163/510，原生无记忆175/510。
- 384×512：LS 178/510，A06 179/510，原生无记忆179/510。
- 纯文字为117/510。

288×384保留为效果与成本折中候选。最高档相对中档的编码成本约增加59%，增益的来源重采样区间包含零。这里使用作者标注证据片段，分辨率提高也增加视觉Token预算；510次测量不是510道独立题，不能作为长期检索效果的证明。

[读取审计与478题结果](experiments/36tar/sember-vision-stage-20261026/README.md) · [三档分辨率与成本](experiments/wshy/sember-resolution-stage-20261025/README.md)

### 3. 分开检查保存、读取和视觉编码的压缩

固定银行后仅在回答时读一半特征，首版没有组合通过预设质量与成本门槛。跨帧视觉缓存首版反而使视觉耗时90.571→96.849秒，支持长题10→8，未采用。

后续将剪枝移到视觉编码器中层。在另一套固定478题条件下，完整、固定剪枝、免训练内容剪枝分别为111、104、103题正确；独立12来源57题的完整处理耗时分别305.73、282.19、285.22秒。Token减半不等于总耗时减半。这套近期输入条件与上面的完整历史135/478不同，不混为同一基线。

[保存与读取压缩](experiments/36tar/token-readout-stage-20261024/README.md) · [中层剪枝最终结果](experiments/36tar/model-evaluation-20261102/搭档交接.md)

### 4. 训练、模型规模和历史路由：局部改善未必迁移

- 动作分类191段：原2B为80/191，普通LoRA为95/191；VideoMAE教师条件和打乱教师条件也均95/191，没有体现教师的额外收益。
- 9B在一个旧题与当前题联合正确面板上由2B的5/30提高到14/30，但102题五排列的联合读取由124/510变为120/510。增大模型不是这批问题的统一解决办法。
- 100个提问时刻、五次排列中，联合读取、时间词规则、小头分别165、176、172/500；小头未稳定胜过简单规则。
- 视觉评分器的首版训练发生退化；修订为二值选择与答案交叉熵训练后恢复，但相对固定剪枝在开发及回归集仅多0和1题，配对区间均跨零。

这些结果支持继续定位实际读入的证据与错误来源，尚不支持默认启用训练头或更复杂路由。

[训练、9B与历史路由复盘](experiments/36tar/model-evaluation-20261102/搭档交接.md)

### 5. 扩大到两个数据集，比较相同输入下的视觉前端

300题分开报告，每组均为Qwen3.5-2B：

- **NExT-QA，150题：** 完整视觉121，固定剪枝120，免训练内容剪枝118，训练后CE剪枝120，纯文字54。
- **StreamingBench，150题：** 完整视觉97，固定剪枝97，免训练内容剪枝101，训练后CE剪枝93，纯文字62（冻结选项解析计数；自然语言补读另有1题正确、1题待定）。

NExT-QA按2 FPS读完整短片；StreamingBench按1 FPS读提问前最近8个完整帧对。五组内输入范围一致，但这不是长短期记忆算法比较。免训练方法在StreamingBench多答对4题，配对区间仍跨零。23题辅助盲判中，纯文字选对10题，仅4题解释充分，说明答对与解释得到画面支持是两件事。

[300题逐题结果、辅助判读与复核](experiments/36tar/model-evaluation-20261102/README.md)

### 6. 最新：200题开放式问答，按答案含义评分

不提供选项，让模型用自然语言回答。四组共800条回答，采用统一**0分错误、1分部分正确、2分关键事实完整正确**，不统计格式指标。31条回答涉及11题仍待人工核对，因此四组共同成绩使用189题、满分378：

- **原模型：113/378，综合得分率29.89%；完全正确27/189，14.29%。**
- **LS：78/378，20.63%；完全正确16/189，8.47%。**
- **A06：71/378，18.78%；完全正确14/189，7.41%。**
- **纯文字：50/378，13.23%；完全正确11/189，5.82%。**

采用FP32、288×384、1 FPS。原模型读取完整因果采样前缀；LS/A06保留4个近期帧对加最多4帧对等价历史。原模型与记忆组不是等Token预算。

800条已由Codex对照作者参考文本复审，未逐题回看视频，不是独立人工标注。纯文字猜中按同一规则给分；2分不证明模型使用了视频。综合得分率与完全正确率是不同指标，也不能与旧选择题正确率直接比较。

[800条回答、评分理由、待审清单及复算](experiments/wshy/openqa-200-reviewed-20261104/README.md)

## 如何使用这些结果

优先保留原模型、近期窗口和纯文字作为对照，LS/A06及剪枝方法作为显式候选。下一步重点是数据问题、原生与流式输入对齐、实际历史证据保留，以及同预算下的感知—记忆取舍；局部分类提升、Token下降或单批净增几题，都不足以单独证明整体升级有效。

各阶段数据多数已用于开发或回归，不称未见留出。各目录是固定交付快照，其中“下一步”“作业运行中”等描述只代表当时状态；中层剪枝的后续结果以上述11月2日交付为准。不同数据集、题目子集、窗口、提示词和评分协议分开解释，不合并成总正确率。

[早期数据总目录](experiments/36tar/dataset-catalog-20261021/catalog/数据集总目录.md)只覆盖当时的数据，不是当前所有数据集的全集。研究由wshy与36tar协作推进，具体源码来源、贡献和原始回执保留在各阶段归档。

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
- 历史读取、帧库、压缩及训练候选按各阶段结果评估，不因归档发布自动改为默认。
- 每次记录模型 revision、代码 commit、命令、输入哈希、环境和失败信息，区分 CPU 检查与真实 GPU 结果。

[协作约定](CONTRIBUTING.md) · [继承来源](docs/PROVENANCE.md) · [0.8B 历史 README](README_08B.md)
