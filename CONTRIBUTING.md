# 协作约定

仓库管理者在 GitHub Settings → Collaborators 邀请成员；成员有写权限后推送自己的分支，无写权限则 fork 后提 PR。此初始化不自动邀请或更改任何成员权限。

```bash
git switch main
git pull --ff-only origin main
git switch -c yourname/short-task-name
# 修改、验证、检查差异后
git add <相关文件>
git commit -m "Describe one concrete change"
git push -u origin yourname/short-task-name
```

在 GitHub 将个人分支 PR 到本仓库 main；另一个成员审核源码、测试与证据后合并。不同功能分开提交，不 force-push 共有分支。初始化 main 是首次建库操作，后续修改走 PR。

运行入口见 README。活动源码在 `experiments/36tar/streaming-learning/code/qwen35-streaming-s4`；根目录 `qwen35_local` 和旧基线作为对照，不同步修改来掩盖差异。新实验放独立目录，历史 runs/source_snapshot 只读。

关键修改与结果写入活动实验的 `实验总纲.md`。报告中区分 0.8B 历史、2B CPU 接口验证和 2B 真实推理，不把前两者当后者。CPU CI 使用小配置与官方处理器，不下载模型权重，不跑 GPU。

提交前运行 `python scripts/run_cpu_checks.py --model-dir /path/to/processor-or-model` 与 `git diff --check`；GPU 行为改动需补同输入对照证据或明确列出待验证项。测试不能取代人工看视频核对回答。

提交范围是源码、文档和必要小型文本证据。权重、媒体、缓存、虚拟环境、集群凭据和含视频帧的输出不提交。
