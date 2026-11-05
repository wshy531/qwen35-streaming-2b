"""Structured review of six premarked author-interval-miss candidates.

Existing contact sheets only; original wording, answers and splits unchanged.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/dataset_build_20261018_01/content_release_v1'
DECISIONS=[
 dict(id='nextgqa:val:6618792571:5',answer_support='crawling_reaching_sequence_supported',anchor_support='specific_spoon_identity_unresolved',
      full_question_verified=False,role='conditional_action_diagnostic_not_primary',
      observed='33秒孩子由蹲姿转向低身，36秒手撑地向有轮玩具靠近，38秒坐姿伸手。可支持爬行接近，但图中不足以确认题干指定的勺子玩具。',
      missing='具体勺子身份与取勺前提；不把有轮玩具替代勺子。'),
 dict(id='nextgqa:val:6618792571:7',answer_support='forward_movement_supported',anchor_support='spoon_pickup_identity_unresolved',
      full_question_verified=False,role='conditional_action_diagnostic_not_primary',
      observed='39–40秒孩子手中有细长蓝色物体；41秒撑地向前，42秒靠向推行玩具，47秒已站在其后。向前移动可见，具体取勺前提仍不够清楚。',
      missing='物体确为玩具勺及拿起动作的直接确认。'),
 dict(id='nextgqa:val:8224492254:0',answer_support='yawning_sequence_supported',anchor_support='start_time_and_baby_supported',
      full_question_verified=True,role='observable_action_diagnostic_not_physical_causality',
      observed='开头0–2秒婴儿持续大张嘴、面部放松，3秒闭嘴；序列与打哈欠动作相符。此题虽用why问句，选项实际对应动作解释。',
      missing='不把打哈欠标签提升为困倦等不可见生理原因；未声称全视频排除其他动作。'),
 dict(id='nextgqa:val:2868183184:1',answer_support='cooking_guidance_context_supported',anchor_support='man_girl_and_cookware_supported',
      full_question_verified=False,role='contextual_purpose_diagnostic_only',
      observed='男子在女孩旁，6–8秒手臂朝炊具方向示意，女孩在旁观察，场景为共同烹饪。可支持指导烹饪的情境解释，不能直接观察心理目的。',
      missing='帮助的意图只作有视觉依据的解释，不作为确定物理因果事实。'),
 dict(id='nextgqa:val:3851913128:0',answer_support='mask_removal_and_display_supported_intent_ambiguous',anchor_support='red_diver_and_removal_supported',
      full_question_verified=False,role='ambiguous_intent_supplement_not_primary',
      observed='32秒红衣潜水者双手在面罩处，34–35秒摘下面罩并拿在身前，其他潜水者面向他。能核实摘下和展示，不能仅凭静态图确认教学意图。',
      missing='官方选项showhow措辞不清，教学/示范动机不能独立核实；无音频判断。'),
 dict(id='nextgqa:val:4838113006:3',answer_support='approach_to_bell_table_supported',anchor_support='lady_bending_and_bell_context_supported',
      full_question_verified=False,role='contextual_spatial_purpose_diagnostic_only',
      observed='37秒女子站在放有钟铃状物的台后，38秒弯腰，40–43秒头部接近台面高度。接近钟铃的空间动作有依据，目的仍是情境解释。',
      missing='不将空间接近直接当作唯一可见动机；不声称物理因果已验证。'),
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'runs/comparison6_review_20261019_01';out.mkdir(exist_ok=False)
    overlay=json.loads((BASE/'question_annotation_overlay.json').read_text(encoding='utf-8'))
    byid={q['id']:q for q in overlay['questions']}
    traces={t['source_path']:{f['frame_id']:f for f in t['frames']} for t in json.loads((BASE/'actual_sampling_traces.json').read_text(encoding='utf-8'))['videos']}
    results=[];images={}
    for d in DECISIONS:
        q=byid[d['id']];assert q['original']['split']=='comparison'
        sid=d['id'].split(':')[2];p=ROOT/f'runs/dataset_build_20261018_01/gqa_comparison_sheets/nextgqa-{sid}.jpg'
        images[str(p.relative_to(ROOT))]=sha(p)
        witnesses=[]
        for w in q['review'].get('witnesses',[]):
            sampled=traces[q['original']['source_path']].get(w['frame_id'])
            witnesses.append(dict(**w,exact_05fps_match=bool(sampled and abs(sampled['pts_s']-w['pts_s'])<1e-7 and sampled['rgb_sha256']==w['rgb_sha256'])))
        results.append(dict(**d,original=q['original'],original_row_sha256=q['original_row_sha256'],
            prior_role=q['provisional_role'],prior_review=q['review']['verdict'],witnesses=witnesses,
            reviewer='Codex structured second pass, not independent human review',formal_evaluation_eligible=False))
    payload=dict(status='six_premarked_candidates_reviewed_not_dataset_freeze',questions=results,inspected_existing_images=images,
        question_count=6,source_count=5,role_counts=dict(Counter(d['role'] for d in results)),
        full_question_action_support_count=sum(d['full_question_verified'] for d in results),
        physical_causal_questions_independently_verified=0,new_decoded_frames=0,comparison_model_outputs_seen=0,
        holdout_reviewed=0,independent_human_reviews=0,raw_question_GT_or_split_changes=0,
        input_overlay_sha256=sha(BASE/'question_annotation_overlay.json'),
        raw_manifest_sha256=sha(ROOT/'experiments/2b-memory-eval/data/candidate_manifest.json'),
        scope='These6 are a stratum of the original comparison set, not the complete benchmark or chosen success set.',
        stop='No additional screenshot or wording rounds for these6 in this phase; preserve raw items and limitations.')
    (out/'review.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'六题证据复核.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','question','official_answer','answer_support','anchor_support','role','observed','missing','full_question_verified']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for d in results:
            r=d['original'];w.writerow(dict(**{k:d[k] for k in fields if k not in ['question','official_answer']},
                question=r['question'],official_answer=r['options'][r['answer_index']]))
    print(json.dumps({k:v for k,v in payload.items() if k not in ['questions','inspected_existing_images','scope','stop']},ensure_ascii=False))


if __name__=='__main__':main()
