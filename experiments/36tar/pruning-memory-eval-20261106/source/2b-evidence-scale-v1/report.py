"""Readable, dataset-separated snapshot; not the continuing progress registry."""
import argparse,collections,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_scale_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--runs',nargs='+',type=Path,required=True);a=p.parse_args()
    result=read(a.analysis/'analysis.json');rows=[]
    for run in a.runs:rows += [json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    labels=read(P/'labels_LOCAL_ONLY.json');panel=read(P/'panel.json')
    names=dict(full='完整视觉108',fixed12='固定中层54',heuristic11='免训练内容54',mask_ce='冻结CE评分器54',text_only='纯文字')
    text=['# 300题扩展测评：内容结果与辅助判读触发','','2026-11-02结果快照。持续状态只记实验总纲。本文件只统计已完成的数据集，不以尚未完成部分凑总分。','',
          '本轮扩大覆盖并接入修复后的既有CE评分器，没有重新训练模型。每题同一简短解释提示、原选项次序、128输出Token；内容对错与格式分开。旧180份B解释复用，固定端点重现后才纳入。','',
          'NExT-QA完整短片2FPS；StreamingBench因果1FPS近期8帧对。后者不能当完整视频或长期记忆成绩。两集都是已打开开发/回归来源，CE训练视频与本轮Next来源交集0。','']
    details={}
    for ds,arms in result['content'].items():
        text += [f'## {ds}','','| 方法 | 内容正确/150 | 比例 | 解析待定 | 输出达到上限 | 输入Token合计 |','|---|---:|---:|---:|---:|---:|']
        for arm,x in arms.items():
            text.append(f'| {names[arm]} | {x["correct"]}/150 | {x["correct"]/150:.1%} | {x["unparsed"]} | {x["output_cap_reached"]} | {x["input_tokens"]} |')
        dsq=[q for q in panel if q['dataset']==ds];distribution=collections.Counter(labels[q['id']] for q in dsq)
        details[ds]=dict(gold_position_counts=dict(distribution),fixed_most_common_position_correct=max(distribution.values()),
            uniform_random_expected_correct=150/len(dsq[0]['options']),prediction_positions={arm:dict(collections.Counter(str(r['original_selected_index']) for r in rows if r['dataset']==ds and r['arm']==arm)) for arm in arms})
        text+=['',f'作为题目偏置参照：均匀随机选择的期望为{details[ds]["uniform_random_expected_correct"]:.1f}/150；事后最多标签位置为{max(distribution.values())}/150。这不是实际运行的模型成绩，也不代表纯文字只能随机猜。','',
               '| 配对（前者减后者） | 前者独有正确 | 后者独有正确 | 差值百分点 | 来源聚类95%区间 | 触发辅助 |','|---|---:|---:|---:|---|---|']
        for c in [x for x in result['paired'] if x['dataset']==ds]:
            lo,hi=c['source_bootstrap_95pct_pp'];text.append(f'| {names[c["a"]]} − {names[c["b"]]} | {c["a_wins"]} | {c["b_wins"]} | {c["difference_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {"是" if c["auxiliary_trigger"] else "否"} |')
    aux=read(a.analysis/'auxiliary_manifest.json')
    text+=['','## 如何使用辅助判读','',f'本批触发数据集：{", ".join(aux["triggered_datasets"])}。对应预选{aux["tasks"]}题、{aux["cases"]}份五方法解释已经导出；题目在成绩揭示前固定，不能只挑分歧或有利题。当前文件不包含新API裁判成绩。',
           '', '触发条件：正确比例差不超过5个百分点，或来源聚类配对区间包含0。这是安排辅助诊断的规则，不是统计等价/非劣证明；区间很宽也可以表示材料不足。辅助样本自己的分母单列，不将45题分数外推为300题成绩。',
           '', '裁判只核回答所需核心事实、依据是否支持回答、关键错误和不可判；细枝末节与格式另列。官方答案对但理由编造、答案错但观察有部分真实信息，分开解释。裁判仍有噪声，不能只凭少量细微分差强行决定谁更好。',
           '', '解释正确也不证明内部确实利用了视觉。五组纯文字对照帮助识别语言先验；对于碰巧一致的正确答案，仍需明确输入干预才可更强地讨论依赖关系。此处不新增训练或调提示。','',
           '## 成本解释','','完整108与剪枝54的视觉Token相差一半，但总输入还包括问题、时间与模板。表中输入Token可以比较；复用答案的耗时不可与新生成耗时横比。StreamingBench恢复选择器的9对重放只是节约质量测试计算，不是持续编码、端侧或独立端到端测速。',
           '', '每份原始输出、截断标记、待定解析和相反得失均保留，不按正确性删题。过程中的路径错误、脚本换行拒绝、慢任务主动停止和既有V0补齐另记于总纲，不计作能力提升。']
    (a.analysis/'扩展测评结果与使用说明.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (a.analysis/'position_controls.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    print('report written')
if __name__=='__main__':main()
