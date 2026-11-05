"""Bind frozen question IDs to original labels and existing media hashes, after selection."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
panel=read(P/'panel.json');sb=read(R/'runs/streamingbench_download_prepare_20261030_01/eval_inputs/labels_LOCAL_ONLY.json')
media={r['name']:r for r in read(R/'runs/streamingbench_router_prepare_20261101_01/media.json')}
labels=read(P/'labels_nextqa.json');bound=[]
for q in panel:
    if q['dataset']=='StreamingBench':
        label=sb[q['id']]['answer'];assert label in 'ABCD';labels[q['id']]=ord(label)-65
        m=media[q['media_name']]
        q=dict(q,media_path='datasets/StreamingBench-eval200-v1/videos/'+q['media_name'],media_sha256=m['sha256'],media_bytes=m['bytes'])
    bound.append(q)
assert len(labels)==60
(P/'panel_media_bound.json').write_text(json.dumps(bound,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(P/'labels_all_LOCAL_ONLY.json').write_text(json.dumps(labels,indent=2)+'\n')
text=['# 新测评60题清单与分布','', '固定种子按题型与时长分层，未使用方法成绩选择。现有开发/回归来源，不是未见测试。', '',
      '| 数据集 | 问题类型 | 题数 |','|---|---|---:|']
from collections import Counter
for ds in ('NExT-QA','StreamingBench'):
    for typ,n in sorted(Counter(q['type'] for q in bound if q['dataset']==ds).items()):text.append(f'| {ds} | {typ} | {n} |')
text += ['', 'NExT-QA：30题/23源，10.744—19秒，两时长档各15题；实际2FPS完整短片。',
         'StreamingBench：30题/27源，源文件193—1122秒，提问7—810秒，三个查询前缀档各10题；推理只用近期8帧对，1FPS。',
         '', '| 数据集 | 来源 | 提问秒数 | 问题 |','|---|---|---:|---|']
for q in bound:text.append(f'| {q["dataset"]} | {q["source_id"]} | {q["query_time_s"]:.3f} | {q["question"].replace("|","/")} |')
(P/'题目清单.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(dict(questions=60,labels_bound_after_selection=True,media_hashes_bound=True,actual_stream_decode_pending=True)))
