"""Index exact old rows for the unified suites; keep incompatible protocols separate."""
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/eval_reuse_20261105_01'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def load(name):return [json.loads(s) for s in (ROOT/name).read_text('utf-8').splitlines()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    sb=read(ROOT/'datasets/streaming-evaluation-v1/development_streamingbench.json')
    short=read(ROOT/'datasets/streaming-evaluation-v1/development_short.json')
    pairs=read(ROOT/'datasets/streaming-evaluation-v1/development_diagnosis.json')
    old_panel={q['id']:q for q in read(ROOT/'runs/evidence_scale_prepare_20261102_01/panel.json')}
    keys=read(ROOT/'datasets/streaming-evaluation-v1/answer_keys_LOCAL_ONLY.json')
    sys.path.insert(0,str(ROOT/'experiments/2b-evidence-eval-v1'))
    from response import question
    from response_v4 import parse
    for q in sb+short:
        if q['id'] not in old_panel:continue
        old=old_panel[q['id']]
        assert question(q,0,'B')==question(old,0,'B')
        assert q['media_sha256']==old['media_sha256'] and q['query_time_s']==old['query_time_s']
    refs={
        'old_B_SB150':'runs/evidence-scale-stream-tail-220170/answers.jsonl',
        'old_B_short150':'runs/evidence-scale-nextqa-v3-220098/answers.jsonl',
        'latest_SB75':'runs/repair-stream-220559/answers.jsonl',
        'bank_SB75':'runs/memory-prune-panel150-220382/answers.jsonl',
        'latest_D18':'runs/repair-paired-220558/answers.jsonl'}
    raw={k:load(v) for k,v in refs.items()}
    result=[]
    for suite,qs,collection,arm_map in [
        ('SB150',sb,'old_B_SB150',{'full':'full_recent8','fixed12':'fixed_recent8','text_only':'text'}),
        ('short60',short,'old_B_short150',{'full':'full_clip','fixed12':'fixed_full_clip','text_only':'text'}),
        ('SB150',sb,'latest_SB75',{'complete_history':'full_history','full_recent8':'full_recent8','full_LS':'full_LS','guarded_LS':'guarded_LS','guarded_recent8':'guarded_recent8','initial_LS':'initial_LS'}),
        ('SB150',sb,'bank_SB75',{'fixed12/LS':'fixed_LS','mask_ce/LS':'old_CE_LS'}),
        ('D18',pairs,'latest_D18',{a:a for a in {r['arm'] for r in raw['latest_D18']}})]:
        expected={q['id']:q for q in qs}
        for original,mapped in sorted(arm_map.items()):
            selected=[r for r in raw[collection] if r['id'] in expected and r['arm']==original]
            assert len(selected)==len({r['id'] for r in selected})
            correct=unknown=0
            for r in selected:
                q=expected[r['id']];p=parse(r['text'],len(q['options']),'B')
                idx=None if p['answer_conflict'] else p['original_selected_index']
                unknown+=idx is None;correct+=idx==keys[q['id']] if idx is not None else False
            result.append(dict(suite=suite,method=mapped,available=len(selected),required=len(qs),
                missing_ids=sorted(set(expected)-{r['id'] for r in selected}),correct=correct,unresolved=unknown,
                source=refs[collection],original_arm=original,
                status='complete_existing_rows' if len(selected)==len(qs) else 'partial_existing_rows',
                score_scope='available rows only; never a full-suite score when rows are missing'))
    # Existing exact cross-implementation evidence for the unpruned recent baseline.
    first={(r['id'],r['arm']):r for r in raw['old_B_SB150']}
    cross=[]
    for r in raw['latest_SB75']:
        if r['arm']!='full_recent8':continue
        old=first[r['id'],'full']
        same={k:old[k]==r[k] for k in ('input_ids_sha256','position_ids_sha256','generated_ids')}
        cross.append(dict(id=r['id'],**same))
    assert len(cross)==75 and all(all(v for k,v in r.items() if k!='id') for r in cross)
    payload=dict(rows=result,cross_implementation_recent75=cross,
        source_sha256={name:digest(ROOT/name) for name in refs.values()},
        restrictions=['Generated B answers are not merged with option-logit classifications.',
          'Same method name with different frame coverage, memory event rule or prompt remains separate.',
          'Oracle full history and bounded LS are separate read policies.',
          'New short validation60 remains untouched; no checkpoint selection from that suite.',
          'No new model inference, performance measurement, training or API in this ledger.'])
    (OUT/'ledger.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 统一清单的旧结果复用范围','','本表是现有逐题结果的覆盖盘点，不是新增模型实验。缺题时正确数仅对应已有题，不能作全量成绩。持续进度见实验总纲。','',
        '| 清单 | 条件 | 已有/需要 | 已有题正确 | 缺题 |','|---|---|---:|---:|---:|']
    for r in result:
        lines.append(f'| {r["suite"]} | {r["method"]} | {r["available"]}/{r["required"]} | {r["correct"]} | {len(r["missing_ids"])} |')
    lines += ['', 'SB150=统一StreamingBench主表150题；short60=四时长档均衡的开发短片60题；D18=既有12当前题与6旧题。short60不是新的来源验证60题。', '',
        '旧150题近期不剪枝与最新75题近期基线的75条交集，其输入Token、位置以及生成ID全部相同。这个既有复现证据支持复用旧150题近期基线，省去整组重跑。它不替代其他读取策略和新候选的检查。','',
        '短片旧N345/N271等选项概率分类只保留历史开发价值，不填进本表的B解释生成成绩；固定事件和自适应事件、近期4/8/16帧对也分别记录。','',
        '缺项次序：先统一主表近期/完整历史两个锚点和LS比较，缺失方法只补缺失题；再补均衡短片开发60题的同协议比较。新的短片验证60题保留到方法确定后。所有实际GPU预算另列，本文件不授权提交。','']
    (OUT/'旧结果复用与补测清单.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('suite','method','available','required','correct')} for r in result],ensure_ascii=False))

if __name__=='__main__':main()
