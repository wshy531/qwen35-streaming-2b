# S-EMBER 108-Token流式输入与开发资源预检

本目录是独立适配，正式方案在项目根目录《下一阶段实验设计_S-EMBER_288x384_20261025.md》，持续进度只看《实验总纲》1.104。没有更换模型内核、重新训练或修改旧冻结实现。

## 接口与范围

- `sember_source.py`：0.5FPS顺序PTS采样、288×384等比补边、两真实帧成对；中途孤帧等待，真实EOF补齐一次。解码器可以有内部读前，但未来帧不会进入视觉编码。每个查询的接收帧摘要包含待配帧；模型已看到的部分由`pairs / last_committed_pts`单独报告。
- `sember_nodes.py / sember_banks.py`：原网格9×12、每对108Token，固定odd/change54；M0/原M2/A06/LS，原合并数值、边界阈值和预算分区不变。源码中6×8仅供旧48路径回归，不是本轮视频设置。
- `sember_runtime.py`：新画布构造器复用原成对处理和动态位置函数，不全局放宽旧入口；读取位置跨度采用新原网格，压缩后代表位置仍是启发式适配。
- `sember_session.py`：`bounded`只持有11个有界银行；`reference`只持有完整混合状态，两者在不同进程运行。问题分支完成后释放；完整参照使用已验收续接核，不能叫未修改的原生官方实现。
- `sember_protocol.py`：官方提示与首选项抽取规则，内容正确和格式分别报告。不向worker提供正确答案或证据区间。

这轮CPU检查不产生真实2B能力结论。`tiny_*`使用相同随机微型权重，原生整段与多块续接数值比较仅验证新网格下接口衔接。

## 数据与资源

默认只读项目内`datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23`。集群开发预检可显式设置`SEMBER_MEDIA_ROOT`为用户提供的同学快照`/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23`；仅允许这两个根，不能写同学目录。全部代码/输出仍在本人项目根内。

本次GPU脚本只实现固定8题开发profile，不是478题正式能力测试。每题型一个既定较长前缀，不看标签选题；11银行共88答，独立全历史参照8答，再加12预热，最多108答/1728生成Token、2次顺序FP32载入。正式478题的时长由实测外推后另行批准，不在此脚本自动启动。

`run_profile.sh`资源提案：g078t/gx_llm、1A800、4CPU、24GiB系统内存、30分钟、禁用requeue；日志≤64MiB、不训练/下载/自动重试。没有明确本次GPU批准不得提交或执行。`evaluate_profile.py`在Slurm分配缺失时会在真实权重载入前拒绝运行。

多臂共享视觉的耗时不能冒充单方法完整成本。QA计时包含上下文、状态复制、prefill和逐Token输出；一致性SHA审计单独报告，实际全流程墙钟耗时还包含它及视频输入。完整参考状态SHA不转化为持久特征档案。

## CPU预检

CPU设置`CUDA_VISIBLE_DEVICES=-1`，不加载真实模型权重，不安装新依赖。源码导入不会提交任务。

| 脚本 | 检查目标 |
|---|---|
| `preflight_core.py` | 旧48/24逐步回归，108/54上界、质量和坐标、复制隔离 |
| `preflight_boundaries.py` | 官方解析函数、真实EOF、pending、非零/不连续PTS |
| `preflight_inputs.py --scope all` | 固定146视频580查询的实际PTS/RGB摘要，至每源最后一问停止 |
| `preflight_layouts.py` | 580题×11合成预算压力布局、完整后缀、旧48位置/嵌入回归；没有LM前向 |
| `tiny_session_check.py` | 真实开发像素、随机微型模型逐Token、QA隔离和后续输入 |
| `tiny_native_check.py` | 独立原生/续接微型进程数值输出，0真实权重 |

输出为不可覆盖的检查回执。传输校验/CPU机制/真实GPU结果分别记录，不把任一种替代另一种。开发profile准备由`prepare_profile.py`根据真实PTS计数产生，`freeze_profile.py`封存依赖与输入；封存不授予GPU权限。
