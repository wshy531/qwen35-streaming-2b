"""Write a readable, reproducible index without changing frozen question lists."""
from collections import Counter
import hashlib
import json

from package_dataset import DEST, ROOT, read, save


def summary(name):
    rows = read(DEST / (name + '.json'))
    durations = [q['duration_s'] for q in rows if 'duration_s' in q]
    return dict(questions=len(rows), files=len({q['source_id'] for q in rows}),
                question_types=dict(Counter(q['type'] for q in rows)),
                duration_bins=dict(Counter(q.get('length_bin', 'not_applicable') for q in rows)),
                query_range_s=[min(q['query_time_s'] for q in rows), max(q['query_time_s'] for q in rows)],
                file_duration_range_s=[min(durations), max(durations)] if durations else None)


def main():
    registry = read(DEST / 'registry.json')
    before = {name: hashlib.sha256((DEST / item['path']).read_bytes()).hexdigest()
              for name, item in registry['suites'].items()}
    assert all(before[n] == registry['suites'][n]['sha256'] for n in before)
    stats = {name: summary(name) for name in ('development_streamingbench', 'development_short',
                                            'development_diagnosis', 'short_validation')}
    save(DEST / '分布统计.json', stats)
    rows = ['# 评测数据导航', '', '本文件说明清单用途；持续进度只记录在项目《实验总纲.md》。', '',
            '## 1. 当前使用的三张表', '',
            '| 表 | 内容与规模 | 考察什么 | 使用方式 |',
            '|---|---|---|---|',
            '| 主表 | StreamingBench 150题、144视频文件 | 连续观察中的动作、属性、事件、时空、计数、文字和推理 | 已打开开发；75题快速回归是其子集 |',
            '| 短片补表 | 开发60题、52映射来源；新验证60题、60映射来源 | 一般视觉理解、因果与时序，检查优化是否损害短片能力 | 两组各自报告，不合并成一个验证分数 |',
            '| 配对诊断表 | S-EMBER旧18题、5视频，当前12题/旧6题 | 同一提问时刻，加入历史是否帮助回忆、干扰当前感知 | 已打开归因案例，不承担未见泛化结论 |', '',
            '开发组228题=主表150+短片开发60+旧诊断18。回归组153题=主表中的75+同一短片60+同一诊断18。回归不是新增153题。新短片验证60题不参与训练、阈值选择或checkpoint选择。', '',
            '新StreamingBench主表150题和新诊断80题是此前的扩容目标，不能与已完成题数混写。首批主表仅13题/12视频通过技术核验；新增诊断文件、锚点和配对草案的完成量见diagnosis扩容清单。原始官方题锚点不自动等于“近期充分/历史必需”配对题。', '',
            '## 2. 时长的两种口径', '',
            '视频文件总时长与模型截至提问时已接收的时长分别统计。StreamingBench必须在问题时刻停止输入，后半段文件不能提前送给模型。短片使用既有EOF协议。', '',
            '| 分组 | 文件时长范围 | 提问时刻范围 | 分档 |', '|---|---|---|---|']
    for name, title in [('development_streamingbench', 'StreamingBench开发'),
                        ('development_short', '短片开发'), ('short_validation', '短片新验证'),
                        ('development_diagnosis', '旧配对诊断')]:
        s = stats[name]
        duration = ('%.2f—%.2f秒' % tuple(s['file_duration_range_s'])) if s['file_duration_range_s'] else '沿原媒体及固定轨迹；不以提问时刻冒充文件时长'
        bins = '；'.join(f'{k}: {v}' for k, v in s['duration_bins'].items())
        rows.append('| %s | %s | %.2f—%.2f秒 | %s |' % (title, duration, *s['query_range_s'], bins))
    rows += ['', '短片开发与新验证各有10—20、20—40、40—60、60—90秒四档，每档15题；两组相加才是120题、每档30题。短片新验证已核本项目已知来源ID、VidOR映射来源与媒体SHA；它取自官方train标注，不能声称底座预训练未见。', '',
             '## 3. 问题类型与例子', '',
             'StreamingBench主表题型如下。这里沿用作者标签，计数或文字题不必然依赖长期记忆。', '', '| 类型 | 题量 |', '|---|---|']
    rows += [f'| {kind} | {count} |' for kind, count in stats['development_streamingbench']['question_types'].items()]
    rows += ['', 'NExT-QA按大类报告：C=因果、T=时间关系、D=描述。开发与新验证各为C/T/D=20/20/20；子类型的原标签保留在题单中。', '',
             '典型题意包括：当前人物正在做什么；某个动作之后发生什么；为什么人物有某种反应；物体是什么颜色、在哪里；早先见到的物体与现在有何对应。配对诊断则在同一时刻分别问当前画面和早期内容，观察保护近期与取回历史能否兼顾。', '',
             '## 4. 旧名称如何归档', '',
             '| 旧名称 | 后续定位 |', '|---|---|',
             '| SB75 / SB150 | 同一StreamingBench开发体系中的75题回归与150题全量开发 |',
             '| N345 / N271 | 已打开的剪枝与历史读取专项开发；保留结果，不加入统一主表重复计分。N271不是StreamingBench |',
             '| D18 | 上述旧配对诊断，不是18个独立来源 |',
             '| S-EMBER478 | 阶段性历史细节与记忆附表；128视频478题，已用于开发，不称新测试 |',
             '| 早期171、T98、长视频34题 | 历史算法回归与失败分析；不再混入当前总分 |',
             '| 120问时/100问时的五排列面板 | 路由、选择头的已打开诊断；五排列不增加独立问题数 |',
             '| 新主表候选、诊断锚点与配对草案 | 获取/证据审核状态单列；未完成项不加到正式验证分母 |', '',
             '## 5. 公平比较与运行入口', '',
             '同一题单、同一输入采样与截点、同分辨率、同提示、同原选项顺序、同评分器，比较不剪枝与各剪枝；读取策略区分近期窗口、LS等有界银行与完整历史。完整历史是参照，不伪装成有界端侧方案。相同Token预算与相同画面覆盖分别比较。', '',
             '主成绩为原排列的内容/选项正确数。格式、未解析项、换序稳定性、解释依据及成本另列；缺失回答不得补成答错后悄悄拼分。辅助证据判读仅在方法难分或与纯文本过近时使用，付费调用另按预算执行。', '',
             '题单、绑定、协议与答案通过registry.json统一管理。具体命令见[使用说明](使用说明.md)。本批整理不新增模型成绩、训练或GPU实验。', '',
             '配对草案中的当前题会复用作者晚锚点，跨视图即使题ID不同，也不能相加当作独立题。question_identity_index.json按媒体SHA、提问时刻、规范化题面和原序选项记录完全相同输入；这个去重也不等于证明拍摄来源或语义上的独立性。', '']
    release = ROOT / 'runs/diagnosis_expansion_review_20261105_01/release_summary.json'
    if release.exists() and 'diagnosis_anchors' in registry['suites']:
        s = read(release)
        rows += ['## 6. 本轮新增诊断材料', '',
                 '已获取30个视频文件，包含60道作者原题（每视频两个不同提问时刻）。文件时长240.9—691.02秒；早锚点2—300秒，晚锚点144—552秒，锚点间隔120—490秒。题型为属性14、动作18、物体16、空间12。它们主要补读取诊断，不替代主表的十类任务。', '',
                 'diagnosis_anchors可用于指定时刻的普通原题读取检查。early/current字段仅表示较早和较晚的原题，不表示已经证明历史必要或近期充分；禁止直接按该字段生成“当前/历史能力”分数。原作者答案未因采样困难、模型预测或本次观察标签而更改。', '',
                 'diagnosis_candidate为11组22题同刻当前/历史配对草案，已保存实际早期证据与晚时刻最近8对输入。它们尚未完成近期/旧证据/纯文字对照与独立内容复核，导出接口仍拒绝把它们当正式验证集。原定30组配对加20跨时刻题、共80题的目标尚未全部达到。', '',
                 '30文件SHA与本项目已知787个媒体SHA无交集；原始拍摄场次仍没有完整映射，不能声称30个独立拍摄来源。单助手查看早晚实际采样画面并保留了重复线索、题义和采样时刻等观察标记，没有查看算法新成绩；这不等于独立复核。', '',
                 '本批实际已提交帧对末PTS距离提问时刻为%.3f—%.3f秒。采样和等待成对可能使最近输入落后于题目时间，这只是本批固定协议的统计，不能推广成任意视频的延迟上界。清单分别保留请求时刻与实际提交时刻，不靠读取未来帧补齐。' % tuple(s['committed_cutoff_lag_range_s']), '',
                 '下一次新诊断资格检查固定使用这11组草案：保留全部题和失败，分别给近期、早期证据及无视频输入，并将重复/先验线索单列；只有获得证据支持后才升级记忆依赖标签。其余19文件优先复用作者其他题目，不先追加视频下载。新增StreamingBench主表剩131文件约38.22GiB暂缓，现有150题主表照常使用。', '']
    (DEST / '数据集导航.md').write_text('\n'.join(rows), encoding='utf-8')
    assert before == {name: hashlib.sha256((DEST / item['path']).read_bytes()).hexdigest()
                      for name, item in registry['suites'].items()}
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
