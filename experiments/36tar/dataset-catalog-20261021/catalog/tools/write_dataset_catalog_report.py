"""Render existing audited dataset inventory, attach known cluster paths, no inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/dataset_catalog_20261021_01'
CAT = OUT / 'catalog'
BIN_NAMES = [('lt30', '不足30秒'), ('30to60', '30—不足60秒'), ('60to180', '1—不足3分钟'),
             ('180to300', '3—不足5分钟'), ('300to600', '5—不足10分钟'), ('ge600', '至少10分钟')]


def read(name):
    return json.loads((CAT / name).read_text(encoding='utf-8'))


def save(name, value):
    (CAT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def duration(x):
    return '—' if x is None else f'{x:.2f}秒'


def table(headers, rows):
    def safe(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(safe, headers)) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |'] +
        ['| ' + ' | '.join(map(safe, row)) + ' |' for row in rows])


def main():
    media, questions, releases, summary = [read(n) for n in ('media.json', 'questions.json', 'releases.json', 'summary.json')]
    existing = json.loads((OUT / 'cluster_inventory.json').read_text(encoding='utf-8'))['existing']
    for item in media:
        aliases = existing[item['sha256']]
        item['cluster_path'] = item['cluster_preferred'] if item['cluster_preferred'] in aliases or not aliases else aliases[0]
        item['cluster_aliases_before_sync'] = aliases
        item['cluster_present_before_sync'] = bool(aliases)
    save('media.json', media)
    for item in questions:
        label = item.get('longstream_review')
        if label and item['id'].startswith('ovo:'):
            item['review_status'] = 'longstream_' + label['verdict']
        item['memberships'] = ['补充78题_复用原比较视频' if x == '补充78题_同原比较24视频' else x for x in item['memberships']]
    save('questions.json', questions)
    releases[1]['name'] = '补充78题_复用原比较视频'
    extra = json.loads((ROOT / 'runs/2b_early9_diagnostic_20261020_01/requests.json').read_text(encoding='utf-8'))
    for q in questions:
        if q['id'] in {x['id'] for x in extra} and '2B首轮早期9题_171的子集' not in q['memberships']:
            q['memberships'].append('2B首轮早期9题_171的子集')
    save('questions.json', questions)
    lookup = {q['id']: q for q in questions}
    if not any(x['name'] == '2B首轮早期9题_171的子集' for x in releases):
        ids = [x['id'] for x in extra]
        releases.append(dict(name='2B首轮早期9题_171的子集', questions=len(ids), media_files=len({lookup[x]['media_id'] for x in ids}),
            ids=ids, media_ids=sorted({lookup[x]['media_id'] for x in ids}), origin='2b_early9_diagnostic_20261020_01',
            reviewed=True, additive_to_other_releases=False,
            task_counts=dict(Counter(lookup[x]['official_task'] for x in ids)),
            reviewed_types=dict(Counter(lookup[x]['reviewed_type'] or 'not_reclassified' for x in ids))))
    save('releases.json', releases)
    provenance = read('provenance.json')
    p = ROOT / 'runs/2b_early9_diagnostic_20261020_01/requests.json'
    provenance[p.relative_to(ROOT).as_posix()] = dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    save('provenance.json', provenance)
    official = [q for q in questions if q['origin'] == 'official']
    deltas = [q['query_time_s'] - q['duration_s'] for q in questions if q['query_time_s'] > q['duration_s']]
    summary['author_query_vs_container_header'] = dict(items_with_positive_rounding_delta=len(deltas),
        max_positive_delta_s=max(deltas, default=0), original_query_times_preserved=True)
    save('summary.json', summary)
    kinds = summary['video_kinds']
    type_names = {q['official_task']: q['task_name'] for q in questions}
    report = ['# 项目数据集总目录（截至2026-10-21）',
        '这是数据资产与固定题目的清单快照。持续工作进度只记《实验总纲.md》。本次仅整理既有数据，不运行模型，不恢复暂停中的小时级测试。',
        '## 1．先看总数：这些数字分别代表什么',
        '- 本地实际存在 **182个视频文件，合计2,595,713,895字节，约2.42 GiB**。173个MP4、7个WEBM、2个MKV，SHA互不重复。另有6张静态照片，约4.71 MiB，不能算视频。上次2.34 GiB只统计MP4；本次补齐其他格式。',
        '- 其中164个是基准来源文件：124个NExT-GQA视频、40个OVO视频。OVO中4个是HM3D仿真环境，36个为真实视频；不能把164个全部叫自然拍摄。另18个是早期学习素材/派生控制。不同视频ID也不保证独立拍摄场次。',
        '- 固定题目目录共 **519道官方原题、6道人工长流题、同一个洋葱问题的3个提问截点**，合计528个“题目—时刻”条目；合并洋葱重复问句后为526个语义题组。不是528道全部合格、全部独立的新题。',
        '- 最早的15次自然画面窗口检查另列为机制检查，不并入正式题库。新的EgoLife 1920片段/约27.68 GiB仅有元数据，没有下载；96题只是目标，实际准入0题。',
        '## 2．从最初到新增：各批数据怎样组成',
        '下表有候选全集，也有已测子集，**各行不能直接相加**。“已测”只表示运行过该协议，不表示建立了独立正式基准。',
        table(['批次或用途', '视频文件', '题目/时刻', '说明'], [
            ['最初候选集',100,235,'开发20/46、比较40/95、原留出40/94；其中有仿真与未准入题'],
            ['补充78道原题',20,78,'复用原比较视频，新增题目、不增加视频'],
            ['新增24源',24,72,'后来已下载；其中50题进入2B171，不能仍称全新未见留出'],
            ['以上三批合计',124,385,'与上次协作共享124视频/385候选清单一致'],
            ['T新来源候选',40,119,'媒体全部已取回并校验；21题未准入，保留原因'],
            ['T准入实际比较',39,98,'是119的子集；其余20歧义、1不支持'],
            ['长流新增官方原题',6,15,'复用已有6源；与旧13题组成28官方题，没有增加视频'],
            ['人工长流开发题',6,6,'与28原题合为34中途提问；单助手标注'],
            ['洋葱延迟问法',1,3,'同一个事件/问句，在278/342/472秒查询；不算3个独立事件'],
            ['新的小时级计划',0,0,'4会话/96题目标暂缓，1920片段未下载']]),
        '## 3．按时长看：目前真正有哪些视频',
        '时长从本地视频容器/视频流头重新读取，没有用问题查询时刻充当整段长度；不是本次重新做了逐帧PTS审计。分界左闭右开，例如60秒归入1—不足3分钟。累计时长是文件时长之和，不等于独立场景时长。',
        table(['时长', '全部视频', '其中自然视频', '基准文件164', '官方原题519'], [
            [label, summary['media']['duration_bins'].get(key,0), kinds['natural_video']['duration_bins'].get(key,0),
             summary['official_benchmark_videos']['duration_bins'].get(key,0),summary['official_duration_bins'].get(key,0)] for key,label in BIN_NAMES]),
        f"全部文件累计约{summary['media']['sum_seconds']/3600:.2f}小时；最长{summary['media']['max_seconds']/60:.2f}分钟。**当前没有已下载的小时级单会话。** 把许多短视频时长相加，不能替代一个持续很久、需要记早期事件的流式测评。",
        f"部分作者整秒查询时刻比容器头时长略大：{len(deltas)}个条目，最大约{max(deltas,default=0):.4f}秒。目录同时保留两个原值，不把查询时刻改写成视频时长；实际取帧仍不得超出文件或查询截点。",
        '## 4．按问题类型看：它们各考什么',
        '保留作者类型，不依问句关键词偷偷重标。CW/CH是作者因果类标签，TN/TP/TC是作者时间类标签；个别原标签与题意并不完全一致，例如某些“如何”实际是在问可见工具。审核后的F事实、T时序、C因果另列，不能与作者标签混为同一统计。题型也不等于实测难度。',
        table(['作者题型', '通俗含义', '原题数', '主要用途/限制'], [
            ['CW',type_names['CW'],summary['official_task_counts']['CW'],'原因/目的；主观动机未必能从像素确定'],
            ['CH',type_names['CH'],summary['official_task_counts']['CH'],'如何完成、借助什么；有的更接近事实题'],
            ['TN',type_names['TN'],summary['official_task_counts']['TN'],'某动作之后发生什么'],
            ['TP',type_names['TP'],summary['official_task_counts']['TP'],'某动作之前发生什么'],
            ['TC',type_names['TC'],summary['official_task_counts']['TC'],'同时/关联的时间关系，沿作者标签保留'],
            ['EPM',type_names['EPM'],summary['official_task_counts']['EPM'],'过去物体、位置、颜色或事件；不保证证据很早'],
            ['ASI',type_names['ASI'],summary['official_task_counts']['ASI'],'动作的先后关系'],
            ['HLD',type_names['HLD'],summary['official_task_counts']['HLD'],'无关/缺证据问题；与正例记忆题分报']]),
        '另有人工事件顺序4题、人工旧画面细节2题，以及洋葱位置问句的3个时间条目；这些不是作者原题。519原题中12题对应4个仿真视频，正式自然集应单列排除，不能全部视为自然视频能力题。',
        '### 问题类型 × 视频时长（519道官方原题，含未准入题）',
        table(['类型']+[label for _,label in BIN_NAMES]+['合计'], [[task]+[summary['official_task_by_duration'][task].get(key,0) for key,_ in BIN_NAMES]+[summary['official_task_counts'][task]] for task in ['CW','CH','TN','TP','TC','EPM','ASI','HLD']]),
        '## 5．实际测试过哪些子集',
        table(['协议/子集','视频','题目','审核事实F / 时序T / 因果C'], [[x['name'],x['media_files'],x['questions'],str(x['reviewed_types'])] for x in releases if x['reviewed']]),
        '0.8B与2B共用的一部分题目不重复计数，也不把0.8B结果当2B结果。2B171=开放74+已打开47+新增24源中50题；T98是后来新增批次。长流34又被用于可读性/A06/A04，它们是重跑同题，不是新的102/170道题。',
        '长流34中28官方题为16画面支持、11歧义、1未建立，另6人工开发题单列；34不是34道干净独立题。可读性只用人工6题，旧节点由人工指定是定位参照，不是自动检索。',
        '## 6．早期0.8B架构学习素材',
        table(['素材组','视频/照片','时长','检查内容'],[
            ['baseline_synthetic', '3视频','各4秒','颜色变化、最小输入与状态续接'],
            ['window_synthetic', '2视频','各12秒','同前缀不同未来，排查偷看未来'],
            ['stability_synthetic', '3视频','4秒、4秒、120秒','重置与有限持续输入'],
            ['natural_commons_v1', '2原视频+1派生','约8.34/23.00/6.53秒','猫、洗手、VFR；15个固定窗口检查'],
            ['general_stream_v1', '2原视频','约43.32/17.05秒','狗与鸽子，主体变化'],
            ['numeric_hand_hygiene_v1', '1原视频','约20.05秒','洗手画面与数值/接口诊断'],
            ['history_integration_v1及02', '2派生拼接','各6秒','狗→鸽→猫的受控会话，不是自然事件序列'],
            ['readout_transfer_v1', '2原视频','约13.36/24.60秒','新狗/猫素材的证据读取迁移'],
            ['coat_controls/readout_holdout', '6静态照片','不适用','同物种毛色及静态识别；重复成帧对也不算视频']]),
        '以上素材不构成新的自然长视频测试集。渲染缩略图、预览PNG、实验source_snapshot中的复制件不在资产总数里；源照片另记，防止把截图当新增视频。',
        '## 7．如何找到文件、题目与集群副本',
        '- [完整视频/照片清单](视频清单.md)：每项列时长、类型、本地相对路径、集群路径和SHA。',
        '- [完整题目清单](题目清单.md)：按类型列528条题目、对应视频、查询时刻、审核状态；选项与原标签保留在questions.json。',
        '- media.json、questions.json、releases.json可直接用于写下一轮评测配置；查询时刻与整段时长分列，尚未构建自动难度分级。',
        '- 本地路径均相对维护者项目根E:/VIT Model。集群路径均相对授权项目/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar，原目录不移动；统一目录放data/catalog-20261021。',
        '- 先前164个基准媒体已存在，按SHA复用；只补缺少的18早期视频和6照片，共89,595,580字节，避免再复制整套数据。Git只保存目录、题目、生成/核验工具及同步回执，不存媒体。',
        '- 标注来源固定版本与原清单SHA见provenance.json；原始官方标注池NExT-GQA3358行（3行证据异常）、OVO1640行（本任务EPM/ASI/HLD631行）只是候选来源，不能说这些视频都已下载。',
        '## 8．使用边界',
        '所有已看过模型输出的集合都应视为开发/已打开比较数据。历史split名不恢复未见性；不同视频ID不自动代表不同拍摄场次。没有证据表明这里所有题都需要长期记忆，early/late标签或证据时间距离只是代理。',
        '本次不重审像素、不改题目/答案、不按新模型分数挑选、不产生新的能力结论。既有单助手标注与未独立复核限制保留；暂停中的EgoLife素材和96题目标只放planned1920_NOT_DOWNLOADED.json，不能放进已就绪清单。',
        '## 9．无需GPU的核验入口',
        '在本目录运行`python verify_catalog.py`可检查目录SHA、题目/集合关系与统计一致性；本机加`--local-root "E:/VIT Model"`逐SHA核验现有188个视频/照片，集群加`--cluster-project /home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar`核验对应远端。Python3.11+仅用标准库，不联网、不下载、不启动GPU。',
        'tools中保存生成脚本供维护者追溯；它们依赖原工作区的历史输入与PyAV，不是下载器或评测入口。搭档直接使用统一JSON/Markdown目录及标准库核验器即可，不必重建旧工作区。']
    (CAT / '数据集总目录.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    by_media = Counter(q['media_id'] for q in questions)
    (CAT / '视频清单.md').write_text('# 全部既有媒体清单\n\n每行一个实际文件；时长来自容器头，不代表独立拍摄场次。路径映射以media.json为机器可读依据。\n\n'+table(
        ['类别','来源/本地相对路径','时长','题目条目','集群相对路径','SHA256'],
        [[a['material_kind'],a['local_path'],duration(a['duration_s']),by_media[a['id']],a['cluster_path'],a['sha256']] for a in media])+'\n',encoding='utf-8')
    detail=['# 全部固定题目与查询截点\n\n528条目包括候选/未准入/人工题；3条洋葱条目共用1个语义题组。不是528道都已合格的新独立题。完整选项、标签、集合成员身份与证据字段见questions.json。']
    for task in ['CW','CH','TN','TP','TC','EPM','ASI','HLD','manual_order','manual_detail','anchored_location']:
        subset=[q for q in questions if q['official_task']==task]
        detail.append('## '+task+'：'+type_names[task]+f'（{len(subset)}条）')
        detail.append(table(['ID','来源','整段秒数','查询秒数','问题','审核状态'],[
            [q['id'],q['source_path'],f"{q['duration_s']:.2f}",q['query_time_s'],q['question'],q['review_status']] for q in subset]))
    (CAT / '题目清单.md').write_text('\n\n'.join(detail)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='reports_written',core_files=len(list(CAT.iterdir())),existing_media=164,missing_media=24)))


if __name__ == '__main__':
    main()
