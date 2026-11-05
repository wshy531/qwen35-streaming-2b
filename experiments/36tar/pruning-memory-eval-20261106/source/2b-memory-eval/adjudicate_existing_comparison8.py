"""Record a bounded second Agent pass over the first8 existing ambiguity items.

No new frames, answers, source split changes, or independent-human claims.
Judgments were made by inspecting the referenced existing contact sheets.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/dataset_build_20261018_01'
SHEETS='runs/dataset_build_20261018_01/'

DECISIONS=[
 dict(id='nextgqa:val:6618792571:2',
      images=['gqa_comparison_sheets/nextgqa-6618792571.jpg'],
      answer_support='contextual_interpretation',anchor_support='transition_not_directly_observed',
      full_question_verified=False,role='contextual_diagnostic_only',
      observed='18.018秒可见成人手靠在玩具黄色把手上，21.021秒孩子已独自推玩具；现有接触表没有直接展示放手的瞬间。',
      decision='允许作为带前提限制的意图解释诊断；不进入严格视觉因果/放手动作主分母。',
      witness_ids=[540,630],missing='放手动作本身及其与问题前提的直接对应。'),
 dict(id='nextgqa:val:8224492254:9',
      images=['gqa_comparison_sheets/nextgqa-8224492254.jpg'],
      answer_support='fine_motor_action_ambiguous',anchor_support='mouth_and_finger_contact_visible',
      full_question_verified=False,role='exclude_primary_v1_keep_raw',
      observed='12–14秒可见婴儿嘴部接触成人手指，15秒及28–29秒手的姿势改变；静态稀疏图不足以明确判定答案所指的收拢手指动作与其先后。',
      decision='本周主评测不纳入；保留原题及GT，不继续补截图追认。',
      witness_ids=[300,325,350,375,700,725],missing='可辨认的收拢手指动作与咬手指后的动作边界。'),
 dict(id='nextgqa:val:2868183184:6',
      images=['gqa_resolution_followup/nextgqa-2868183184.jpg'],
      answer_support='tool_identity_unresolved',anchor_support='cooking_scene_supported',
      full_question_verified=False,role='exclude_primary_v1_keep_raw',
      observed='29.029、37.003、41.007、51.017秒可见女子在炊具旁使用长木质工具及另一小工具；现有原分辨率图仍不足以可靠确认拿熟食的工具是夹子。',
      decision='沿原补核上限停止；不将模糊工具识别作为记忆失败主测。',
      witness_ids=[870,1109,1229,1529],missing='足以区分夹子、勺、叉等选项的直接取食证据。'),
 dict(id='nextgqa:val:5702201498:0',
      images=['gqa_comparison_sheets/nextgqa-5702201498.jpg','gqa_resolution_followup/nextgqa-5702201498.jpg'],
      answer_support='black_pillow_directly_visible',anchor_support='reclining_closed_eye_context_not_sleep_verification',
      full_question_verified=False,role='supplementary_object_fact_with_scope_limit',
      observed='男子斜躺在沙发上，双臂抱着黑色毛绒枕头，多个帧可直接辨认物体；躺卧闭眼能支持休息语境，但不能单靠静态图证实睡眠状态。',
      decision='可作黑色枕头这一物体事实的补充诊断；不冒充睡眠识别或长期因果主测。',
      witness_ids=[0,180,210,270,300],missing='睡眠状态不作已独立核实；此题也未证明必须依赖早期画面。'),
 dict(id='nextgqa:val:5702201498:1',
      images=['gqa_resolution_followup/nextgqa-5702201498.jpg'],
      answer_support='gaze_and_intent_unresolved',anchor_support='hugging_supported_eye_transition_ambiguous',
      full_question_verified=False,role='exclude_primary_v1_keep_raw',
      observed='6.006至10.010秒保持抱枕姿势；现有图无法可靠确定睁眼、看向镜头及其意图之间的关系。',
      decision='本周主评测不纳入；不以看镜头这一意图标签作为确定视觉事实。',
      witness_ids=[180,210,270,300],missing='清晰睁眼/视线变化及回答动机的可观察依据。'),
 dict(id='nextgqa:val:2599937944:6',
      images=['gqa_comparison_sheets/nextgqa-2599937944.jpg'],
      answer_support='speech_unverified_in_silent_sparse_images',anchor_support='pointing_and_person_link_unresolved',
      full_question_verified=False,role='exclude_primary_v1_keep_raw',
      observed='溪流旁白衣女子与其他人站在一起，4–9秒画面有转身和头部朝向变化；无法从这些无声稀疏图确认指向黑衣男子后对身旁人说话。',
      decision='不进入当前纯视觉记忆主测；保留原始交互题，不自动引入音频模块补救。',
      witness_ids=[120,150,180,210,240,270],missing='指向对象、随后说话及人物身份的明确对应。'),
 dict(id='nextgqa:val:2824317018:7',
      images=['gqa_comparison_sheets/nextgqa-2824317018.jpg'],
      answer_support='smile_person_identity_not_secure',anchor_support='rejection_not_independently_verified',
      full_question_verified=False,role='exclude_primary_v1_keep_raw',
      observed='47–50秒多名女孩围在男孩旁；红衣女孩的表情较清楚，蓝衣女孩多为侧背面。无法将微笑可靠归给题干指定人物，也未核实被拒绝事件。',
      decision='本周主评测不纳入；不把旁人微笑替代指定蓝衣人物的反应。',
      witness_ids=[1439,1469,1499],missing='蓝衣人物微笑与被拒绝事件的身份和先后锚点。'),
 dict(id='nextgqa:val:4147398280:4',
      images=['gqa_comparison_sheets/nextgqa-4147398280.jpg'],
      answer_support='emotion_not_uniquely_visually_grounded',anchor_support='cube_manipulation_supported',
      full_question_verified=False,role='interpretation_diagnostic_only',
      observed='男孩持续操作魔方，末尾看向镜头并有表情变化；这些图不能唯一确定害羞，而排除其他情绪或社交解释。',
      decision='仅作明确标注主观解释性的补充题，不进入确定动作/物理因果主分母。',
      witness_ids=[1649,1679],missing='害羞标签属于解释，不能升级为直接可见心理事实。'),
]


def main():
    out=ROOT/'runs/comparison8_adjudication_20261019_01';out.mkdir(exist_ok=False)
    work=BASE/'handoff/比较疑点与完整题干_复核用.csv'
    with work.open(encoding='utf-8-sig',newline='') as f:pending=list(csv.DictReader(f))
    if [x['id'] for x in pending[:8]]!=[x['id'] for x in DECISIONS]:raise ValueError('first8 order changed')
    overlay=json.loads((BASE/'content_release_v1/question_annotation_overlay.json').read_text(encoding='utf-8'))
    byid={q['id']:q for q in overlay['questions']}
    traces={t['source_path']:t for t in json.loads((BASE/'content_release_v1/actual_sampling_traces.json').read_text(encoding='utf-8'))['videos']}
    result=[];allimages={}
    for d in DECISIONS:
        q=byid[d['id']];source=q['original']['source_path'];evidence=[]
        for relative in d['images']:
            path=ROOT/(SHEETS+relative)
            if not path.is_file():raise ValueError('missing inspected existing image')
            allimages[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        old=q['review'].get('witnesses',[])
        source_frames={f['frame_id']:f for f in traces[source]['frames']}
        for w in old:
            if w.get('frame_id') not in d['witness_ids']:continue
            f=source_frames.get(w['frame_id'])
            evidence.append(dict(frame_id=w['frame_id'],pts_s=w['pts_s'],rgb_sha256=w['rgb_sha256'],
                exact_current_05fps_match=bool(f and abs(f['pts_s']-w['pts_s'])<1e-7 and f['rgb_sha256']==w['rgb_sha256'])))
        result.append(dict(**d,original=q['original'],original_row_sha256=q['original_row_sha256'],
            prior_verdict=q['review']['verdict'],witnesses_from_prior_receipt=evidence,
            reviewer='Codex second structured pass of existing contact sheets; not independent human review',
            model_outputs_for_these_comparison_questions_seen=False,
            decision_final_for_this_week_primary_scope=True,raw_question_or_GT_changed=False))
    payload=dict(status='bounded8_existing_evidence_dispositions_not_formal_dataset_freeze',
        inspected_existing_images=allimages,questions=result,
        worklist_sha256=hashlib.sha256(work.read_bytes()).hexdigest(),new_decoded_frames=0,
        raw_manifest_sha256=hashlib.sha256((ROOT/'experiments/2b-memory-eval/data/candidate_manifest.json').read_bytes()).hexdigest(),
        questions_processed=8,questions_remaining_in_original_worklist_not_processed_this_pass=34,
        upgraded_to_full_question_verified=0,independent_human_reviews=0,
        holdout_questions_reviewed=0,models_run_for_comparison=0,
        stop='No further screenshot rounds for these8 in this week primary set; retain raw items and scope decisions.')
    (out/'adjudication.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'八题处置清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','question','official_answer','answer_support','anchor_support','role','observed','decision','missing']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for d in result:
            r=d['original'];w.writerow(dict(**{k:d[k] for k in fields if k not in ['question','official_answer']},
                question=r['question'],official_answer=r['options'][r['answer_index']]))
    print(json.dumps({k:v for k,v in payload.items() if k not in ['questions','inspected_existing_images']},ensure_ascii=False))


if __name__=='__main__':main()
