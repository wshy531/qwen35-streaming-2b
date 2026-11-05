"""Render comparable tables and preserve historical table provenance, CPU only."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
inventory = json.loads((R/'runs/eval_inventory_20261104_01/results.json').read_text(encoding='utf-8'))
rows = inventory['rows']
dest = R/'各数据集剪枝与记忆组合测评总表_20261104.md'
out = ['# 各数据集剪枝与记忆组合测评总表', '',
       '2026-11-04建表，2026-11-05补入作业220548的完整成绩。持续进度只在实验总纲维护。', '',
       '**范围**：当前2B的剪枝/投影压缩、记忆银行、读取路由与选择头、近期保护，以及直接相关训练和辅助判读。主表从已核验JSON汇总，历史附表摘录已归档结果；没有重新运行所有旧实验，也不把尚未完成的条件填成成绩。0.8B早期架构学习、动作专家等其他研究线不在本组合表范围。', '',
       '**读表规则**：每表有自己的样本与评分协议；相同视频的多题/多排列不是独立来源。主成绩为选项内容正确，格式不乘入正确率。解释依据充分率在附表独立列出，不能当选项正确率。', '',
       '**方法**：不剪枝=完整108视觉Token/帧对；固定/训练前/旧CE/守门修补中层剪枝通常为54。免训练内容启发式不等于未训练评分器。投影后change/odd24或54也不等于中层剪枝。M0只留近期，M2是原森林，A06是深度优先版本，LS为共享历史区。LS全读不是完整采样历史。', '',
       '用户两条主基线为不剪枝＋完整因果采样历史、不剪枝＋近期8帧对。缺测项保留“—”，不能用相似命名的旧结果补格。', '',
       '## 表1：NExT-QA短片345题，40视频，10.01—19秒', '',
       '每题两排列，共690判断。近期现有不剪枝是4帧对；不是主基线8帧对。完整读取无时间淘汰；LS固定每3帧对事件。', '',
       '| 方法 | 完整短片/690 | LS联合/690 | 近期/690 |', '|---|---:|---:|---:|']

def value(panel, method, reading, protocol=None):
    found=[r for r in rows if (r['panel'],r['method'],r['reading'])==(panel,method,reading)
           and (protocol is None or r['protocol']==protocol)]
    assert len(found)<=1, (panel,method,reading)
    return str(found[0]['correct']) if found else '—'

for method,name in [('full108','不剪枝'),('initial','训练前评分器'),('fixed12','固定剪枝'),
                    ('old_CE','旧CE'),('guarded','修补版'),('sequential','失败先短后长'),('interleaved','失败交错训练')]:
    a=value('N345',method,'full_short')+('*' if method=='fixed12' else '')
    b=value('N345',method,'LS_all' if method=='full108' else 'LS')
    c=value('N345',method,'recent')
    out.append(f'| {name} | {a} | {b} | {c} |')
out+=['', '纯文字272/690；不剪枝固定时间词选路484/690。*固定完整短片489为历史生成解析，其他本轮主体为选项概率分类，不用微小差距作严格排名。', '',
      '## 表2：NExT-QA配套开发271题，30视频，20.29—90秒', '',
      '**这是训练开发诊断，不是StreamingBench长视频测试。** 两排列/542，固定每3对事件。', '',
      '| 方法 | 近期8对/542 | LS联合/542 |','|---|---:|---:|']
for method,name in [('initial','训练前评分器'),('old_CE','旧CE'),('sequential','失败先短后长'),('interleaved','失败交错训练'),('guarded','修补版')]:
    out.append(f"| {name} | {value('N271',method,'recent8')} | {value('N271',method,'LS')} |")
out+=['| 不剪枝108 | 323（4对，不是8对） | 376 |',
      '','纯文字218/542；完整采样历史未测。不剪枝108参照来自已完成220511。其近期432预算只含4对，不能直接与54组剪枝近期8对归因比较。','']
for panel,title in [('SB75','StreamingBench75题，72来源，前缀4—1005秒'),('N75','NExT-QA组合75题，54来源')]:
    out += [f'## 表{3 if panel=="SB75" else 4}：{title}', '',
            '同作业220382，原排列，B解释提示，按选项内容评分，每格/75。原自适应事件规则。M0同864预算：不剪枝8对，剪枝16对；LS近期最多4/8对。', '',
            '| 视觉路径 | M0近期 | M2 | A06 | LS |', '|---|---:|---:|---:|---:|']
    for method,name in [('full','不剪枝'),('fixed12','固定中层'),('heuristic11','免训练内容'),('mask_ce','旧CE')]:
        out.append('| '+name+' | '+' | '.join(value(panel,method,v) for v in ('M0','M2','A06','LS'))+' |')
    out+=['| 训练前评分器 | — | — | — | — |','| 最新修补版 | — | — | — | — |','',
          '完整历史未测；StreamingBench75纯文字29/75。NExT75纯文字未在本汇总提取，不搬用150题分数。','']
out+=['## 表5：StreamingBench150与148共同子集，旧短回答协议','','两列分母不同；不能和表3的解释提示直接排名。148列缺两题的原因保存在原分析中。','','| 读取 | 原150题 | 共同148题 |','|---|---:|---:|']
for method,name in [('recent','不剪枝近期4对'),('recent864','不剪枝近期8对'),('joint','LS联合'),('isolated','独立状态保护'),('lexical','固定时间词'),('API33_LS','API33头＋LS'),('A06_joint','A06联合')]:
    out.append(f"| {name} | {value('SB150','full108',method)} | {value('SB148','full108',method)} |")
out+=['','150题固定词108见旧lexical_replay；本机器索引未纳入该文件，不以148题107替换。','',
      '## 表6：S-EMBER478题，128来源，旧投影压缩组合','','每组原排列478题；列2/3是同478题中的子集，按作者证据末端到提问时刻的间隔筛分，不是独立新增题，也不是已证明旧证据必需。','','| 方法 | 总体/478 | 间隔≥120秒/85 | 间隔≥300秒/25 |','|---|---:|---:|---:|']
names={'M0_full108_8':'不剪枝近期8对','M0_odd54_16':'投影固定54＋近期16对','M0_change54_16':'投影变化54＋近期16对',
       'M2_full108':'不剪枝＋M2','M2_change54':'投影变化54＋M2','A06_full108':'不剪枝＋A06','A06_change54':'投影变化54＋A06',
       'LS_full108':'不剪枝＋LS','LS_change54':'投影变化54＋LS','M0_odd54_8':'投影固定54＋同8对','M0_change54_8':'投影变化54＋同8对',
       'R_all_history_stream':'不剪枝完整历史','text_only':'纯文字'}
for method,name in names.items():
    reading='none' if method=='text_only' else 'encoded_in_method'
    out.append('| '+name+' | '+' | '.join(value('SE478_'+group,method,reading) for group in ('all','ge120s','ge300s'))+' |')
out+=['','### 220548四种中层剪枝＋固定事件LS','','同478题、每帧对54视觉Token、每3帧对固定划分事件。上表旧LS按特征变化划分事件，作为历史参照。','',
      '| 方法 | 总体/478 | 间隔≥120秒/85 | 间隔≥300秒/25 |','|---|---:|---:|---:|']
for method,name in [('initial','训练前评分器'),('fixed12','固定剪枝'),('old_CE','旧CE'),('guarded','修补版')]:
    out.append('| '+name+' | '+' | '.join(value('SE478_pruned_LS_'+g,method,'LS') for g in ('all','gap_ge120s','gap_ge300s'))+' |')
out+=['','修补相对训练前24得29失，相对固定33得34失，相对旧CE27得27失。','',
      '## 表7：18题当前/旧问题配对诊断','','5来源，12当前＋6旧问题。每格“当前正确/12；旧题正确/6”；同刻双对在历史附表保留。','','| 视觉路径 | M0 | M2 | A06 | LS | 同8对M0控制 |','|---|---|---|---|---|']
for method,name in [('full','不剪枝'),('fixed12','固定中层'),('heuristic11','免训练内容'),('mask_ce','旧CE')]:
    cells=[]
    for policy in ('M0','M2','A06','LS','M0_same8'):
        a=value('D18_current',method,policy);b=value('D18_old',method,policy)
        cells.append('—' if a=='—' else a+'；'+b)
    out.append('| '+name+' | '+' | '.join(cells)+' |')
out+=['','最新修补版未测。小面板是诊断材料，不代表广泛能力。','',
      '## 表8：新概率收益头，独立于剪枝的冻结读取实验','',
      '1114训练题、20轮、底座不更新。下列均为不剪枝108＋LS864，近期432只含4对；不是修补剪枝结果。短片迁移0更新，ID与SHA训练隔离已核。开发与短片均已打开，不当新未见测试。','',
      '| 读取方式 | NExT短片/690 | 配套开发/542 |','|---|---:|---:|']
for reading,name in [('recent','近期4对'),('all','LS全部已保存节点'),('nearest2','近期＋最近2历史节点'),
                     ('head','内容可见收益头'),('matched_count_nearest','可见头同数量较近节点控制'),
                     ('same_head_binary','可见头限制二选一'),('lexical_rule','固定时间词规则')]:
    out.append(f"| {name} | {value('N345','utility_content_visible',reading)} | {value('N271','utility_visible',reading)} |")
for reading,name in [('head','盲历史收益头'),('matched_count_nearest','盲头同数量控制'),('same_head_binary','盲头限制二选一')]:
    out.append(f"| {name} | {value('N345','utility_blind_history',reading)} | {value('N271','utility_blind',reading)} |")
out += ['', '可见头短片比LS多1次正确、开发少5次；盲头分别少2/3次。尚无稳定胜过联合的证据。读取量与得失、训练拟合指标需分开解释，不能据此启动新一轮扩头。','',
      '## 历史附表导航与口径','','以下摘录原结果的表格，并保留来源及章节。只继承当时成绩，不继承旧文档的“下一步/等待”状态；不把旧数据重新命名成未见测试。重叠的复现数字只算同一份证据。','']
archives = [
('H1 早期384预算银行与LS_depth','runs/2b_ls_depth_prepare_20261021_01/LS_depth组合对照结果_20261022.md','171＋T98，长视频支持16与人工6分报。'),
('H2 投影选择与ToMe/Div24','runs/2b_div24_prepare_20261023_01/Div24五臂结果与阶段决定.md','48→24，有限完整档案保真，不是有界银行。'),
('H3 变化与空间混合','runs/2b_temporal24_prepare_20261023_01/C2三臂结果与阶段决定.md','同24、同覆盖；完整档案。'),
('H4 变化选择×四银行','runs/2b_compressed_banks_prepare_20261023_01/P3九组有界记忆压缩结果.md','384预算；不同于288×384下108→54。'),
('H5 同覆盖8/16对','runs/2b_matched_coverage_prepare_20261023_01/同覆盖五组诊断结果与阶段决定.md','区分空间压缩与时间覆盖。'),
('H6 QProbe读取选择','runs/2b_question_probe_prepare_20261024_01/QProbe24真实结果与阶段决定.md','同LS保存内容，问题到来后再选择Token。'),
('H7 早期训练、KL与307回归','runs/hard-mask-main-220062/二值选择训练结果复盘.md','主表单排列345/307；双排列数据见源文。这里的修补指二值mask修订，非220499守门修补。'),
('H8 300题解释生成后的选项成绩','runs/evidence-scale-analysis-20261102-01/扩展测评结果与使用说明.md','NExT150全短片，StreamingBench150只读近期8帧对；不等于75题组合。'),
('H9 API辅助解释判读','runs/evidence-scale-nextqa-judge-20261102-01/辅助判读结果.md','23题子集，依据充分/关键错误另算，不混入选项准确率。'),
('H10 路由/节点选择/保护候选','长短期记忆候选与测评表现清单_20261103.md','保留第1—7节成绩表；第8节旧待办不继承。'),
('H11 中层剪枝与后置选择','视觉中层剪枝质量与完整成本结果_20261026.md','同S-EMBER478题质量与独立成本见源协议。'),
('H12 免训练内容确认','runs/content_confirm_prepare_20261101_01/内容中层剪枝确认结果.md','按原确认面板单独解释，不并入后续训练结果。'),
]
manifest={}
table_count=9
for title,name,note in archives:
    p=R/name
    assert p.exists(),name
    raw=p.read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest()
    text=raw.decode('utf-8-sig')
    if name.startswith('长短期'):text=text.split('## 8.')[0]
    out += [f'### {title}', '', note, '', f'来源：[原始结果](<{p.as_posix()}>)。','']
    lines=text.splitlines();heading='';i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('#'):heading=line.lstrip('#').strip()
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].startswith('|'):
                block.append(lines[i]);i+=1
            if len(block)>=3:
                out += [f'原章节：{heading}', '',*block,''];table_count+=1
            continue
        i+=1
out += ['## 结果文件与范围检查','',
        f'当前主结果的{len(rows)}行机器索引见[results.json](<E:/VIT Model/runs/eval_inventory_20261104_01/results.json>)。历史附表原文件SHA另存archive_sources.json。本汇总不包括未运行候选的预期分数。', '',
        '最新补测设计见[统一整理及补测计划](<E:/VIT Model/剪枝与记忆评测统一整理及补测计划_20261104.md>)。未测组合统一留空，后续完成后由新结果快照补入，不以不兼容实验填格。']
dest.write_text('\n'.join(out)+'\n',encoding='utf-8')
(R/'runs/eval_inventory_20261104_01/archive_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(dest),tables=table_count,archive_sources=len(manifest)),ensure_ascii=False))
