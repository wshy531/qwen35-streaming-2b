# 2B 有界外部视觉记忆与在线问答

2026-10-21 阶段交付。[完整结果与证据](../memory-review-2b-20261021/README.md)记录实际CPU/GPU验收、负结果和版本绑定；本页介绍最终接口，不再使用准备阶段的待验证状态。

## 接口与策略

MemorySession(policy, readout) 提供 reset / append_pair / ask / status / close。完整帧对的48×2048投影只编码一次，再更新位于CPU的有界银行；问答每次新建并释放语言状态，不延续长期语言KV。每个银行最多384视觉token和8节点，事件描述子另外计入驻留字节。

可选 M0、M2、reservoir4、M2_A06、LS_shared384_v1。默认171题四方法协议仍为M0/M2/reservoir4/text_only；prepare显式传 --comparison 才启用六方法。A06只改闭合相邻合并的选择，LS为近期192＋历史共享192，均不接入默认演示。没有训练、调参或叠加A06/S24。

非法时间戳、重复或跳号帧对、宽度变化在写入前拒绝；底层异常使会话失效，须reset。导出节点独立所有权，ask检查导出输入、银行、事件描述子和RNG；isolated_readout另保护模型位置/RNG。reset不保留上一视频信息。

## 四阶段入口

统一入口 scripts/run_2b_memory_eval.py，参数及资源需针对新运行重新准备：

```bash
# 仅示意阶段；不构成集群操作或GPU授权。
python -B scripts/run_2b_memory_eval.py --stage prepare --model-dir "$MODEL" --dataset-root "$DATASET" --out "$RUN"
python -B scripts/run_2b_memory_eval.py --stage preflight --out "$RUN"
python -B scripts/run_2b_memory_eval.py --stage gpu --out "$RUN" --preflight "$RUN/cpu"
python -B scripts/run_2b_memory_eval.py --stage score --out "$RUN"
```

RUN必须为新目录。prepare绑定67媒体、171完整问题、模型/源码身份；preflight检查实际RGB/PTS、完整后缀、原209回归和独立契约，真实2B载入0；GPU阶段要求匹配且成功的CPU回执，四方法684主答、最多72前缀检查；显式六方法为1026主答、最多120检查，每答最多8token。score单独执行，严格格式、开头选项、人工语义分开。

短片EOF规则允许复制真实奇数末帧一次，随后禁止继续追加直到reset。中途长视频查询使用独立long_video_2b调度器，查询和结束均不补帧；不能把34题塞入本入口的固定171题协议。

## 隔离与测试

运行入口在第三方导入前调用isolation.configure。本机边界为D:/AI_project/projects/streaming_video_wsy，集群为/home/share/czy/wsy。代码及内部.git、输出、临时文件和HF/Torch等缓存须真实解析后留在边界内，不使用外部worktree/alternates；已有环境仅在获批范围内只读复用。

tests含28项接口/部署安全契约，tests_a06含16项独立A06契约，tests_ls含25项LS契约；上游原策略副本只作测试参照。Linux精确记录原209中一个Windows owner-handle专属项不适用，不把跳过计为通过，不接受其他跳过。

实际完成：CPU217794/GPU217796四方法重放，CPU217843/GPU217845六方法比较，CPU217847及GPU217852长视频扩展。最终134/141条源码绑定见交付；更早217796的四个文件随后扩展，旧回执不是最终代码的独立证明。所有具体成绩和限制以交付RESULTS.md为准，不将模型接口成功当作能力提升。
