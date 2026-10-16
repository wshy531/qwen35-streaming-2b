# 来源与验证边界

- 来源仓库：https://github.com/wshy531/qwen35-streaming
- 继承主线：`c32a5951622dc6ce9e7e770b457af77cca7a883d`，包含 PR #6 的 0.8B 收尾及历史读取实验。
- 完整保留既有 Git 提交历史、作者、许可和实验文件；新仓库独立协作，不改变原仓库。
- 模型实现来自 Transformers v5.3.0，细节见根目录 THIRD_PARTY_NOTICES.md 和 third_party/。
- 2B 官方模型：https://huggingface.co/Qwen/Qwen3.5-2B
- 固定权重 revision：`15852e8c16360a2fea060d615a32b45270f8a8fc`。

旧文档中的本机绝对路径、0.8B 指标、PR 状态、SNAPSHOT.json 与源码快照都是历史证据。新开发以根 README、docs/CLUSTER.md、当前代码和实验总纲最新记录为准。旧路径不作为集群路径。

2B 初始化改动包括固定模型下载与完整性核验、2B 模型身份检查、默认 FP32 基线、通用 GPU 入口直接 FP32 加载、显式资源门槛、CPU 回归入口和协作说明。没有宣称完成 2B 真实推理、BF16 等价、长视频质量或集群部署。
