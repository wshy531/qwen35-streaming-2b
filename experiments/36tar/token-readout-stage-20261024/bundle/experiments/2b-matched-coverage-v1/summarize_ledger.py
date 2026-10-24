"""Describe all P3 paired outcomes without selecting the diagnostic inputs."""
from collections import Counter
import statistics
from common import OUT,read,save


def main():
    rows=read(OUT/'P3_all_question_ledger.json')['rows']
    summaries={}
    for policy in ('M0','M2','A06','LS'):
        rs=[r for r in rows if r['policy']==policy and r['cohort']!='stream34']
        summaries[policy]=dict(queries=len(rs),outcomes=dict(Counter(r['outcome'] for r in rs)),
            earlier_provenance_begin=sum(r['change_first_s']<r['full_first_s'] for r in rs),
            unchanged_provenance_begin=sum(r['change_first_s']==r['full_first_s'] for r in rs),
            mean_input_token_delta=statistics.mean(r['input_token_delta'] for r in rs),
            mean_visual_token_delta=statistics.mean(r['visual_token_delta'] for r in rs),
            corrections=[r['id'] for r in rs if r['outcome']=='correction'],
            losses=[r['id'] for r in rs if r['outcome']=='loss'])
    save(OUT/'P3_ledger_summary.json',dict(primary269=summaries,
        all303_rows=len(rows),input_selection='all old303 unchanged; no score-based filtering',
        provenance_is_not_readable_information=True))
    lines=['# P3逐题账本复核与同覆盖诊断依据','',
        '2026-10-23。账本保留303问题时刻×4记忆政策的1212条完整对照；原始回答、正确性、节点来源范围、输入长度均保留。新实验仍全量使用原303查询，不依此筛选成功或失败题。','',
        '|方法|主269纠正／新增损失|来源起点更早的查询|平均语言输入变化（Token）|',
        '|---|---:|---:|---:|']
    for policy,s in summaries.items():
        lines.append(f"|{policy}|{s['outcomes'].get('correction',0)}／{s['outcomes'].get('loss',0)}|{s['earlier_provenance_begin']}|{s['mean_input_token_delta']:+.2f}|")
    lines += ['',
        'M0压缩版在242个查询中保留更早帧对，但主集净少1题；M2/A06/LS本来已保留早期来源节点，269题的最早来源起点全部不变。它们改变的是近期保留的帧对数、空间细节与融合组织，不能笼统说成“压缩后历史范围都翻倍”。起点相同也不表示内部证据相同或信息无损。',
        '',
        'M0/LS压缩版的平均语言输入分别增加25.16和22.10Token；M2/A06约减少26.66和26.87，但其常用视觉预算从384降到360且时间节点数改变。这是本轮实际输入账，不是独立速度结果。',
        '',
        '下一步使用五个M0窗口：最近8对完整48、相同8对odd24、相同8对change24、最近16对odd24、最近16对change24。前3组比较同画面的压缩路径；后2组与相应8对压缩组比较增加历史的作用。固定因果上一完整对作为变化参照，禁止在窗口起点重新冷启动。',
        '',
        '此诊断不引入新算法，不单独解决森林融合、位置适配或新来源泛化。若同覆盖已经退化，说明问题在该压缩读取路径中，无需先怪长期融合；但不能只凭这一点断言是静态物体Token被删，位置布局也是路径的一部分。若同覆盖可读、扩大后变差，则优先检查额外历史干扰。所有推论等待真实模型对照，CPU检查仅验证实现与预算。',
        '',
        '资料依据：只读Token压缩.md第7节的DTD因果差分与静态细节风险、StreamForest.md对背景主导和反复融合的讨论。当前是固定24配额的已有适配，不声称完整复现DTD；此次不联网增加方法。',
        '',
        '持续状态只记录在实验总纲。本文件为固定账本分析，不替代总纲。','']
    with (OUT/'P3逐题账本复核与诊断依据.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print(dict(status='summarized',all_rows=len(rows)))


if __name__=='__main__':main()
