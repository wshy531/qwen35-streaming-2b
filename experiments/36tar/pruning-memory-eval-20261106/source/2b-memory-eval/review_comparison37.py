"""One bounded full-question pass on the remaining37 premarked natural items.

Uses only contact sheets actually inspected and existing frame identities.
No model outputs, original-question edits, new screenshots or formal freeze.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/dataset_build_20261018_01'
OUT=ROOT/'runs/comparison37_review_20261019_01'

# F: observable action/object; T: temporal relation; C: contextual explanation;
# S: supplementary only. These are review strata, not empirical difficulty.
DECISIONS=[
 ('8224492254:3','S','成人手指入口可见；咬、含住、舔的细分意图在静态图中不能唯一排除，保留原答案但不进清晰主测。'),
 ('4276006683:0','F','手在猫胸前和头颈处移动接触，支持抚弄；不把猫抱手等同人一直抱猫。'),
 ('4276006683:2','C','猫前爪随手的位置伸出并抱住手，支持抓手这一目的解释；不作为攻击意图测量。'),
 ('4276006683:7','T','手收拢靠近后猫嘴接触手部并抱住；按粗口部接触类别解释bite，不判断咬合力度，选项抓手也部分成立须标注。'),
 ('4859057828:5','F','11–12秒三人靠在一起并搂肩背，支持合影姿势。'),
 ('4859057828:0','C','浅黄上衣女子先张臂，后续三人并排面向拍摄者；合影目的有语境依据，不当心理事实。'),
 ('4859057828:9','S','蓝衣女子靠近、抬臂可见，但mini dance和put her hand up/stand有重叠，不作唯一清楚动作答案。'),
 ('2868183184:7','T','26秒锅盖已抬开，29–41秒女孩向锅面俯身查看；开启瞬间不在接触表，属于开盖后顺序而非精确动作起止标注。'),
 ('5702201498:3','C','男子抱枕斜躺且保持姿势，支持休息的情境解释；不推断已经睡着。'),
 ('2599937944:3','S','女子面向摆过急流的男子；具体观看事件可见，但why选项混合所看内容与享受表演等动机，不作唯一因果答案。'),
 ('2599937944:0','F','男子在河上抓住上方细绳悬空摆动，支持rope；细绳在192×256输入下可辨性需特别标记，不假定模型看清。'),
 ('2824317018:5','S','女孩弯腰蹲下可见，但题干friendly的社会评价不独立可见，只保留带前提限制的动作补充。'),
 ('2824317018:2','S','男孩坐下和女孩靠近可见；静态无声图不足以确认talk而排除play等互动目的。'),
 ('4147398280:2','C','男孩转动魔方查看不同面，支持选项的行为描述；strategise属于推测，不证明特定解法。'),
 ('4147398280:7','T','31–33秒可见整片红面，35–36秒继续转动；只支持一面完成后的再次扭动，不声称整个魔方解完。'),
 ('3066064005:5','C','黑衣女子伸臂指向桌旁，之后红衣女子移向该处；手指信号可见，指示座位的功能从情境判断。'),
 ('3066064005:7','T','2秒指示后，5秒红衣女子位置改变，13秒已移向桌子另一侧；遮挡期间不臆测走路轨迹。'),
 ('8547321641:1','S','孩子向海走、成人伸手并最终牵手可见；静态图不能可靠确认题干in front及防止过深这一唯一目的。'),
 ('8547321641:8','T','孩子先贴近镜头，之后转背向海走；2秒近脸、6–7秒面向海可支持先后。'),
 ('8547321641:3','S','笑脸和靠近镜头可见，但excited与其他心理解释不能唯一判定，不入情绪主测。'),
 ('3851913128:4','F','红衣潜水者面对同伴改变手和手指姿势，支持手势交流方式；未要求识别具体手势含义。'),
 ('3489439407:6','S','参与者看监视屏、屏幕显示两位女士；静态图不独立确认speak及其为观看目的，保留视听前提疑点。'),
 ('4984417707:0','C','11秒球接近，12秒大象抬腿，13–14秒球离开；踢球解释有动作结果支持，不把意图当直接观测。'),
 ('2503404966:3','S','男孩抬头并扶小车前方可见，但未见要跌落；prevent himself from falling并非可唯一确认原因。'),
 ('2503404966:5','T','到暗角后22–27秒小车大致停留、男孩调整头手，支持短暂停留；不说全片停止。'),
 ('6163877860:5','F','婴儿在沙发上多次双手膝盖着垫移动，爬行类别清楚；不要求所有作者区间均必要。'),
 ('6329077812:9','T','43–45秒纸巾擦嘴，55–60秒奶瓶重新入口，支持清洁后继续喂奶；无声音推断。'),
 ('8594314852:8','F','粉衣女士走到前方后站住面向孩子，空间姿态支持stood and wait in front。'),
 ('8594314852:2','C','成人停在孩子前方并回身面对，孩子随后靠近；等待解释有依据，不需臆测对话。'),
 ('8594314852:5','F','2–5秒粉衣牛仔裤女士从孩子旁走到前方；身体不全但服饰和位置连续对应。'),
 ('3112351937:4','C','水下男子口部连接呼吸器和管线，供气功能有装备语境支持；未测气体成分。'),
 ('3112351937:6','T','潜水者先搭肩，赤膊男子随后也绕臂到对方肩背，支持do the same thing；不扩写成其他互动。'),
 ('3112351937:5','C','气泡从潜水者头部呼吸器附近上升，支持呼吸解释；未证明每个气泡的物理来源。'),
 ('5429688959:0','C','歌手嘴前支架麦克风可见，支持拾音设备用途；未听声或测扩音，扬声器也属扩声链须保留选项范围限制。'),
 ('7442562474:1','F','两只小猫互相伸爪、抱抓并以嘴接触对方，支持hit and bite粗动作，不区分玩耍与争斗。'),
 ('7442562474:5','C','棕猫转身仰躺后与白猫继续抱抓，playing是互动解释，不当情绪测量。'),
 ('7442562474:7','F','末段棕猫前爪绕住白猫头颈两侧，支持hold head，不排除同时轻咬。'),
]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    overlay=read(BASE/'content_release_v1/question_annotation_overlay.json')['questions']
    byid={q['id']:q for q in overlay}
    first=read(ROOT/'runs/comparison6_review_20261019_01/review.json')
    done={q['id'] for q in first['questions']}
    expected={q['id'] for q in overlay if q['provisional_role']=='provisional_naturalistic_diagnostic'}-done
    assert len(expected)==37 and {'nextgqa:val:'+k for k,_,_ in DECISIONS}==expected
    t05={t['source_path']:{f['frame_id']:f for f in t['frames']} for t in read(BASE/'content_release_v1/actual_sampling_traces.json')['videos']}
    t1={t['source_path']:{f['frame_id']:f for f in t['frames_1fps']} for t in read(BASE/'gqa48_actual_index.json')['videos'] if t['split']=='comparison'}
    result=[];images={}
    for tail,kind,note in DECISIONS:
        q=byid['nextgqa:val:'+tail];raw=q['original'];source=raw['source_path']
        image=BASE/f'gqa_comparison_sheets/nextgqa-{tail.split(":")[0]}.jpg'
        images[str(image.relative_to(ROOT))]=sha(image)
        witnesses=[]
        for w in q['review']['witnesses']:
            item=dict(w)
            for name,trace in [('exact_05fps',t05[source]),('exact_1fps',t1[source])]:
                f=trace.get(w['frame_id']);item[name]=bool(f and abs(f['pts_s']-w['pts_s'])<1e-7 and f['rgb_sha256']==w['rgb_sha256'])
            assert item['exact_1fps'] and w['pts_s']<=raw['query_time_s']
            witnesses.append(item)
        result.append(dict(id=q['id'],original=raw,original_row_sha256=q['original_row_sha256'],stratum=kind,
            decision_note=note,previous_observation=q['review'].get('observed'),previous_limit=q['review'].get('limit'),
            witnesses=witnesses,inspected_sheet=str(image.relative_to(ROOT)),
            candidate_for_stratified_1fps_review=kind!='S',strict_physical_causality_claimed=False,
            model_resolution_readability_verified=False,formal_evaluation_eligible=False,
            reviewer='Codex second structured visual pass; not independent human review',comparison_model_outputs_seen=False))
    payload=dict(status='remaining37_existing_sheets_reviewed_not_formal_freeze',questions=result,
        inspected_existing_images=images,question_count=37,source_count=len(images),stratum_counts=dict(Counter(q['stratum'] for q in result)),
        candidate_for_stratified_1fps_review=sum(q['candidate_for_stratified_1fps_review'] for q in result),
        original_overlay_sha256=sha(BASE/'content_release_v1/question_annotation_overlay.json'),
        raw_manifest_sha256=sha(ROOT/'experiments/2b-memory-eval/data/candidate_manifest.json'),
        newly_decoded_frames=0,new_screenshots=0,pretrained_model_loads=0,holdout_images_seen=0,
        exact_05fps_witnesses=sum(w['exact_05fps'] for q in result for w in q['witnesses']),
        exact_1fps_witnesses=sum(w['exact_1fps'] for q in result for w in q['witnesses']),
        note='F/T/C/S are evidence strata, not measured easy/medium/hard; plausible context is separate from unambiguous visual fact.',
        stop='One complete pass; no further screenshots or wording search. Uncertainty remains recorded.')
    (OUT/'review.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/'三十七题复核.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','question','official_answer','stratum','decision_note','exact_05fps_witnesses','exact_1fps_witnesses','formal_evaluation_eligible']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for q in result:
            raw=q['original'];w.writerow(dict(id=q['id'],question=raw['question'],official_answer=raw['options'][raw['answer_index']],
                stratum=q['stratum'],decision_note=q['decision_note'],exact_05fps_witnesses=sum(w['exact_05fps'] for w in q['witnesses']),
                exact_1fps_witnesses=sum(w['exact_1fps'] for w in q['witnesses']),formal_evaluation_eligible=False))
    print(json.dumps({k:v for k,v in payload.items() if k not in ['questions','inspected_existing_images']},ensure_ascii=False))


if __name__=='__main__':main()
