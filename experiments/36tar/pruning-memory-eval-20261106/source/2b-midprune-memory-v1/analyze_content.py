"""Post-run transparent content mapping; original generated text and parser scores stay intact."""
import hashlib,json
from analyze import analyze,read,P,R
D=R/'runs/midprune-memory-220237'
rows=[json.loads(s) for s in (D/'answers.jsonl').read_text().splitlines()]
labels=read(P/'labels_LOCAL_ONLY.json');qs={q['id']:q for q in read(P/'questions.json')}
raw=analyze(rows,labels);assert raw['methods']==read(D/'analysis.json')['methods']
missing=[r for r in rows if not r['answer_parseable']]
assert len(missing)==1
r=missing[0];assert (r['id'],r['view'],r['policy'])==('T05:late:current','heuristic11','LS')
q=qs[r['id']];assert q['options'][1]=='A brown paper shopping bag.'
assert 'the camera wearer is actively handling a brown paper shopping bag.' in r['text']
review=dict(id=r['id'],view=r['view'],policy=r['policy'],mapped_option_index=1,
    text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
    reason='Explicit answer names the exact object in option B; absent letter is not a content error.',
    reviewer='assistant text-only post-run review, not blind',
    limitation='Does not verify visual evidence or claimed timestamps. Original parseability remains false.')
# Change only semantic choice in a working copy; do not change parseability or original answers.
r['original_selected_index']=1
result=analyze(rows,labels);result['content_mapping_review']=[review]
result['input_sha256']=hashlib.sha256((D/'answers.jsonl').read_bytes()).hexdigest()
with (D/'analysis_content.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
names={'full':'完整108','fixed12':'固定中层54','heuristic11':'免训练内容54','mask_ce':'训练CE54'}
lines=['# 中层剪枝与记忆组合预检结果','',
'作业220237正常完成，Slurm耗时20分39秒。5视频、18题、19条件共342答齐全；1716次视觉编码、1716项完整银行旧新状态对照通过，生成33508Token。问答分支隔离和预算检查通过。峰值allocated约8.38GiB包含多臂比较状态，不代表单方法或端侧成本。','',
'本批是已打开的六案例开发诊断：当前12题、旧问题6题，来自5个视频。部分旧题仍有近期线索，不能称严格长期记忆必要性测试。只用原选项顺序；不是不同排列下稳定性的验证。','',
'以下为答案选项表达的内容正确数，不是格式正确率，也不是解释依据充分率。唯一未输出选项字母的回答明确说“brown paper shopping bag”，与选项B一一对应，单独补记内容正确；原回答、解析失败和原分析均保留。该条解释引用时间是否正确尚未判读。','',
'| 视觉路径 | 记忆方式 | 当前正确/12 | 旧题正确/6 | 同刻两题均对/6 | 最大视觉Token |',
'|---|---|---:|---:|---:|---:|']
for k,v in result['methods'].items():
    view,policy=k.split('/');lines.append(f"| {names[view]} | {policy} | {v['current']['option_content_correct']} | {v['old']['option_content_correct']} | {v['late_both_correct']} | {v['maximum_bank_tokens']} |")
lines+=['','M0是近期窗口，M2是原事件森林，A06调整历史融合顺序，LS共享历史空间。主比较同864视觉Token上限；剪枝M2/A06实际最多810，空余不填假Token。M0_same8是同最近8对画面的432Token控制，剪枝M0主比较则保留16对。','',
'## 结果解释与下一步','',
'1. 组合接口成立，但并非所有组合提高质量。训练CE与M2组合，当前由9/12变8/12，旧题由2/6变5/6；这是一增一减的取舍，不能只报总分。','2. 训练CE与LS组合旧题维持5/6，当前由9/12降至6/12；与A06组合当前由9/12降至6/12，旧题由4/6升至5/6。小预检已提示当前感知保护问题，但不能据此给算法作通用排名。','3. 同最近8对画面，CE为当前9/12、旧5/6，与完整总数相同，但当前有一得一失，并非逐题等价。CE扩大到16对后变当前7/12、旧6/6：当前3道丢失、1道新增，旧题新增1道。额外历史覆盖可能带来收益和干扰，不能全归因于单帧剪枝损伤。免训练内容同8对为10/12、4/6，扩16对为7/12、4/6，也出现当前退化。','4. 当前题、旧题及同刻双正确须分报。比如CE+A06/LS的同刻双正确均0/6，不能用旧题5/6掩盖与当前题错配。','5. 下一步保留完整/固定/免训练/CE和M0/M2/A06/LS公平对照，按既定478题规模准备完整质量与成本预算，同时把当前保护列为独立指标。先定位同8对已有损失与扩大历史新增损失，再决定是否用延迟提问或近期保护目标修订训练；本次不自动加轮数、不换训练目标或默认配置。','',
'原始证据为同目录answers.jsonl、event_trace.jsonl、completed.json和analysis.json；analysis_content.json保存唯一内容映射及输入SHA。后续运行/授权与进度只记实验总纲。']
with (D/'组合预检结果与分析.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({k:[v['current']['option_content_correct'],v['old']['option_content_correct'],v['late_both_correct']] for k,v in result['methods'].items()},ensure_ascii=False))
