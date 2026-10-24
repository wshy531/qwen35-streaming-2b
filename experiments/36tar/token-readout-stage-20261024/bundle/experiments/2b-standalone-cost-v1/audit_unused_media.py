"""Conservative local history audit; absence of a hit is not unseen proof."""
from collections import defaultdict,Counter
from pathlib import Path
import json
from common import ROOT,OUT,read,save,sha


def main():
    inventory_path=ROOT/'runs/2b_compressed_banks_prepare_20261023_01/media_inventory.json'
    inventory=read(inventory_path)['outside_current303_media']
    catalog=ROOT/'runs/dataset_catalog_20261021_01/catalog'
    questions=read(catalog/'questions.json');media=read(catalog/'media.json')
    by_question=defaultdict(set);by_media=defaultdict(list)
    for q in questions:
        by_question[q['id']].add(q['media_id']);by_media[q['media_id']].append(q)
    for q in read(catalog/'legacy_checks15.json'):by_question[q['id']].add(q['media_id'])
    wanted={m['id'] for m in inventory};hits=defaultdict(list)
    files=sorted(set((ROOT/'runs').rglob('*answers*.jsonl'))|set((ROOT/'runs').rglob('*predictions*.jsonl')))
    scanned=[];unknown_ids=Counter();read_rows=0;parse_errors=[]
    for path in files:
        relative=path.relative_to(ROOT).as_posix();lower=relative.lower()
        # These are recorded model-output locations, not candidate/preflight JSON.
        real_location=('runs/08b_' in lower or ('runs/2b_' in lower and ('/gpu_' in lower or '/results-' in lower)) or '_gpu_' in lower)
        if not real_location:continue
        scanned.append(dict(path=relative,bytes=path.stat().st_size,sha256=sha(path)))
        seen=set()
        with path.open(encoding='utf-8') as stream:
            for number,line in enumerate(stream,1):
                try:r=json.loads(line)
                except json.JSONDecodeError:
                    parse_errors.append(dict(path=relative,line=number));continue
                read_rows+=1
                if r.get('status','ok')!='ok' or not r.get('generated_ids'):continue
                uid=r.get('id',r.get('query_id',r.get('question_id')))
                if uid not in by_question:
                    unknown_ids[relative]+=1;continue
                for mid in by_question[uid]&wanted:
                    key=mid,uid
                    if key in seen:continue
                    seen.add(key);hits[mid].append(dict(path=relative,line=number,question_id=uid,
                        model_family='0.8B' if '08b_' in lower else '2B' if '2b_' in lower else 'legacy_GPU',
                        evidence='successful_generated_ids_in_recorded_real_model_output'))
    used_sha={g['sha256'] for g in read(OUT/'inputs/groups.json')}
    used_sources={(m['dataset'],m['source_id']) for m in media if m['sha256'] in used_sha}
    results=[]
    for m in inventory:
        qs=by_media[m['id']];evidence=hits[m['id']]
        overlap=(m['dataset'],m['source_id']) in used_sources
        legacy=m['dataset']=='legacy'
        non_natural=m['material_kind']!='natural_video' or m['asset_type']!='video'
        disposition='exclude_legacy_controls' if legacy else 'exclude_non_natural_or_nonvideo' if non_natural else 'exclude_prior_pretrained_use' if evidence else 'exclude_known_source_overlap' if overlap else 'needs_content_and_parent_review'
        results.append(dict(media=m,disposition=disposition,prior_pretrained_evidence=evidence,
            catalog_question_ids=[q['id'] for q in qs],catalog_memberships=sorted({s for q in qs for s in q.get('memberships',[])}),
            catalog_review_statuses=dict(Counter(q.get('review_status') for q in qs)),
            source_identifier=dict(dataset=m['dataset'],source_id=m['source_id']),
            parent_filming_session_independence_verified=False,
            unseen_test_eligible=False,no_hit_does_not_prove_unused=True))
    report=dict(records=results,counts=dict(Counter(r['disposition'] for r in results)),
        inventory_count=len(inventory),real_output_files_checked=len(scanned),real_output_rows_read=read_rows,
        matched_media_with_real_outputs=sum(bool(hits[mid]) for mid in wanted),input_catalog_question_SHA256=sha(catalog/'questions.json'),
        inventory_SHA256=sha(inventory_path),scanned_files=scanned,parse_errors=parse_errors,
        unmapped_successful_output_rows=dict(unknown_ids),certified_unseen_test_media=0,
        limitations=['Local recorded successful outputs only; no collaborator workspace or remote missing archives searched.',
                     'Unmapped output IDs and filming parents remain unresolved.',
                     'No video content was newly reviewed; no negative hit is a proof of non-use.'])
    save(OUT/'media_history_audit.json',report)
    lines=['# 非本轮76项媒体的本地使用史初审','',
        '此为使用史与准入阻碍审核，不是新的内容标注，也不将未命中日志的素材直接认定为未见测试。','',
        f"共检查{len(inventory)}项媒体，扫描{len(scanned)}份真实模型输出位置的日志、{read_rows}条记录。当前可认证为未见测试的媒体为0；这表示审核条件尚未满足，不表示所有素材已被模型使用。",'',
        '|分类|数量|','|---|---:|']
    lines += [f'|{k}|{v}|' for k,v in report['counts'].items()]
    lines += ['',
        '旧机制素材、合成片段与静态图片不作为新的自然视频能力测试。明确已有预训练模型回答的素材保留为回归；未发现成功输出者仍须核资格审核历史、实际采样证据和拍摄父源，才能冻结为新来源。',
        '',
        '本次只用本项目已记录的question→media映射定位成功输出，未猜测短ID或用字符串子串当父源。未知ID、缺失归档和协作方使用史均保留为限制；详细命中路径/行号、输入SHA和逐媒体分类见media_history_audit.json。','']
    with (OUT/'非本轮媒体使用史初审.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    print(json.dumps({k:report[k] for k in ('counts','real_output_files_checked','real_output_rows_read','matched_media_with_real_outputs','certified_unseen_test_media')},ensure_ascii=False))


if __name__=='__main__':main()
