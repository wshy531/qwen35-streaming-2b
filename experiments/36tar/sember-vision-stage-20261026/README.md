# S-EMBER评测与视觉计算优化：阶段交付

本目录汇总PR #4之后本方完成的实验，并保存R2的源码与最后已核实启动状态。默认模型、搭档的记忆算法和旧综合汇报均未修改。先读[阶段结果摘要](阶段结果摘要.md)，再按需要阅读`reports/`中的详细结果。

## 当前结论

- S-EMBER正式128视频478题已经完整比较。完整流式历史参照135/478，纯文字106/478；有视觉贡献，但整体能力有限。该完整流式参照与官方原生读取不是同一个接口。
- 21视频24题官方读取审计中，官方同帧对两版本与流式均9/24；同输入没有观察到答案退化。用户已决定结束该排查，不把小面板结论推广到所有输入。
- 高分辨率和后置选择并未形成稳定的默认升级依据；问题条件读取首版也未通过质量门槛。失败、成本与逐题输出均保留。
- 将固定空间选择提前到视觉第12层之后，12对真实画面全部更快，视觉耗时中位降低14.46%。原15%成本门槛略未达；本轮0问答，不声称质量无损或整流程加速14%。
- 用户同意继续R2：固定478题、最近8对画面的四组保真比较，以及12源57题的独立完整成本。218670已单次提交并核实正常启动；**本次上传时SSH连接被服务端关闭，未复核最新状态，也没有获取R2最终结果。** `GPU_status_latest.json`只是此前的已核实快照，不能当作实时队列。

## 查阅顺序

1. `阶段结果摘要.md`及`reports/`：工作目的、改动、实际结果和限制。
2. `bundle/experiments/2b-midprune-quality-v1/`：本轮四组同覆盖设计、独立成本及冻结分析器；中层编码直接复用R1原SHA。
3. `bundle/experiments/2b-vision-midprune-v1/midprune.py`：视觉第12层后按完整2×2组保留54输出位置，保留原空间坐标。
4. `bundle/runs/`：协议、固定来源/问题ID、两地CPU回执、已完成原始回答与分项成本。实际文件位置以manifest为准；相同旧依赖直接引用PR #4交付，避免再复制一套。

## 无模型核验

在完整仓库中运行：

```bash
python -I -S -B experiments/36tar/sember-vision-stage-20261026/verify_delivery.py
```

只使用标准库，核SHA、运行依赖、Python语法和完整回答数量，检查R1原门槛与R2未完成的边界。解析选项与格式分开，不把格式当内容准确率。没有官方答案键时不能独立裁定事实正确；提供自己获准取得、SHA匹配的580题本地标签映射后可复算内容分数：

```bash
python -I -S -B experiments/36tar/sember-vision-stage-20261026/verify_delivery.py --labels /authorized/labels_LOCAL_ONLY.json
```

标签映射为现有准备脚本产生的`question_id/correct_letter`列表。实际题干、选项、官方答案键、视频、模型、环境、Tensor快照及可复用GPU授权不在Git中。`external_inputs.json`记录恢复时所需官方输入的SHA与大小；不得把缺少这些输入的恢复工作区称为可直接完整运行。

```bash
python -I -S -B experiments/36tar/sember-vision-stage-20261026/verify_delivery.py --materialize /absolute/path/to/NEW_EMPTY_PROJECT
```

该命令只还原代码、协议与文本证据，不下载或运行模型。模型元数据要求见`external_assets.json`。输入文件自行依官方授权获得并核SHA；Slurm中的绝对工作路径应重新适配、冻结和预检，新GPU必须按实际预算另批。恢复文件不会启动或重新授权218670。

S-EMBER来源：[facebook/S-EMBER](https://huggingface.co/datasets/facebook/S-EMBER)，固定revision见`source_record.json`。保留官方归属，不在本次交付中转发官方题干/答案。模型为Qwen/Qwen3.5-2B固定版本，官方与原项目许可头随冻结源码保留。

所有集合均已打开，用于开发/回归，不作未见泛化或端侧1倍速证明。问题—时刻次数不等于独立来源数；85/25题的时间间隔是代理分组。持续进度只在项目实验总纲维护，此目录是上传时的固定证据快照。
