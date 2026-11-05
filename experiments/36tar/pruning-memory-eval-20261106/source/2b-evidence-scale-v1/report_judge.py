"""Separate auxiliary judgments from official answer correctness."""
import argparse,json
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    d=read(a.run/'analysis.json');done=read(a.run/'completed.json');check=read(a.run/'integrity.json')
    names=dict(full='完整视觉108',fixed12='固定中层54',heuristic11='免训练内容54',mask_ce='冻结CE评分器54',text_only='纯文字')
    text=['# 扩展测评的辅助证据判读','','这是固定辅助子集的结果快照，持续进度只记实验总纲。API不看方法名称、标准答案或各方法分数；同题采用共同的采样画面和任务相关标准。','',
          '“内容正确”仍按原题答案计算；“充分／部分／不足”是裁判对文字解释与画面的判断，不能合成新的正确率，也不能替代真实输入干预证明模型内部用了视频。裁判对非关键细节或格式的意见不进入内容正确数。','']
    for ds,arms in d['tables'].items():
        text += [f'## {ds}：事前固定子集','','| 方法 | 内容正确 | 依据充分 | 部分 | 不足 | 不可核实 | 关键错误 | 待审 | 正确且充分 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm,x in arms.items():
            s=x['support'];text.append(f'| {names[arm]} | {x["official_correct"]}/{x["n"]} | {s.get("sufficient",0)} | {s.get("partial",0)} | {s.get("insufficient",0)} | {s.get("unverifiable",0)} | {x["core_errors"]} | {x["pending"]} | {x["correct_with_sufficient_support"]} |')
        text+=['','| 配对 | 前者支持更好 | 后者更好 | 同级 | 不可比 |','|---|---:|---:|---:|---:|']
        for x in d['paired']:
            if x['dataset']==ds:text.append(f'| {names[x["a"]]} / {names[x["b"]]} | {x["a_better"]} | {x["b_better"]} | {x["same"]} | {x["uncomparable"]} |')
    repeats=d['repeats'];same=d['identical_response_consistency']
    text += ['','## 一致性与解释边界','',
        f'- 盲重复{len(repeats)}份：支持等级一致{sum(x["support_same"] for x in repeats)}份，关键错误标志一致{sum(x["core_error_same"] for x in repeats)}份。',
        f'- 完全相同解释的跨方法组共{len(same)}组：支持等级一致{sum(x["support_consistent"] for x in same)}组。这项不能检测所有近义文本偏差。',
        '- 裁判认为有依据但与官方标签不符的案例保留，可能来自推断宽松、标签歧义或裁判误读，不能自动改答案或删除题。',
        '- 输出达到128 Token上限的解释原样保留；不能把缺少小标题或末段当视觉错误。',
        '- 同级不代表表现完全相同；一两份评分不同也不够宣布稳定优劣。逐题差异与裁判自己的理由应一起阅读。',
        '- 子集是按来源、类型在看成绩前固定，不能外推成全部300题或独立未见数据的结果。',
        '', '## 执行与文件','',f'完成{done["requests"]}次请求，耗时{done["seconds"]:.1f}秒；按保守高峰价复算{check["peak_price_cost_recomputed_cny"]:.6f}元，账本无未结请求。无新增GPU或训练。',
        '', 'analysis.json保存统计；paired_details.json保留原解释、裁判原文及待审项；responses.jsonl保存原回执，integrity.json保存完整性核验。']
    (a.run/'辅助判读结果.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('auxiliary report written')
if __name__=='__main__':main()
