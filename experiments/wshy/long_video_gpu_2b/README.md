# 长视频六方法 GPU 对照扩展

2026-10-21 阶段完成。真实作业217852已COMPLETED、0:0，11分22秒；一次FP32载入，1360次视觉forward、204主回答、413生成token，34历史查询前缀一致、30次reset。此前本机22项契约及作业内隐藏GPU后的同22项均通过。完整结果与限制见[阶段报告](../memory-review-2b-20261021/RESULTS.md)，不把模拟计数当真实运行。

gates.py把固定扩展绑定到CPU217847的协议、预检和源码SHA，限制204答、每答8token、总1632token、1360次视觉forward、30次reset、2550秒worker、RSS20GiB和GPU allocated24GiB。独占输出目录和load_started防止静默二次载入。它是本次固定评测入口，不是任意数据通用GPU启动器。

worker.py按真实PTS读取六视频，在34查询点暂停，五视觉银行共享一次编码，六方法同读出；每问新建并释放语言状态，银行/描述子/RNG/模型位置隔离。每来源最后一次查询后停止，不补末尾单帧。scoring.py在模型进程结束后另行评分，原28、历史supported16/ambiguous11/cutoff1、人工6分报；严格格式和开头选项分开，独立人工语义保持pending。

显式入口 scripts/run_2b_long_video_gpu.py --stage gpu|score --run <匹配CPU目录>。历史资源为g078t、comment=gx_llm、1GPU、4CPU、24GiB、45分钟、1次载入、不自动重投。旧完成回执和脚本不构成新运行授权；后续任何集群操作须重新确认具体范围。
