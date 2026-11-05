"""Blinded-to-model-outcomes content packet; no fake independent review."""
from pathlib import Path
from common import ROOT, PREP, INPUTS, read, sha


def main():
    manifest = read(INPUTS / 'manifest_LOCAL_ONLY.json')
    tasks = {t['id']: t for t in read(INPUTS / 'tasks_LOCAL_ONLY.json')}
    audit = read(PREP / 'cluster_media_CPU.json') if (PREP / 'cluster_media_CPU.json').exists() else None
    audited = {r['pair_id']: r for r in audit['sources']} if audit else {}
    lines = ['# 三组旧题—当前题：独立内容复核材料', '',
        '材料用途：开发诊断准入；不含模型成绩、候选答案或官方正确选项。', '',
        '请先分别作答，再记录证据是否充分。当前题只依据实际已提交的最近8帧；'
        '不采用尚未提交的查询时刻单帧。旧题图来自作者定位区间，只作证据线索；'
        '不能据此证明其他时段没有重复。无法确定时填待定或排除，不强行凑配对。', '',
        '历史替换是人工诊断：用另一来源0—7秒的四对特征替换历史视觉值，'
        '目标时间标签与布局故意保持不变，替换来源另外记录。它不是自然视频历史。', '',
        '请在 independent_review_LOCAL_ONLY.json 填写本人身份、独立角色、结论和证据区间；'
        '不要把同一助手再次查看写成独立复核。原始标签和五轮换成绩不在本材料中展示。', '',
        f"固定清单SHA256：`{sha(INPUTS / 'manifest_LOCAL_ONLY.json')}`", '']
    for pair in manifest['pairs']:
        pid = pair['pair_id']
        lines += [f'## {pid}', '', f"源视频：`{Path(pair['source_path']).name}`", '',
            f"提问时间：{pair['query_time_s']}秒；模型最后已提交：{pair['committed_through_s']}秒。", '',
            '### 当前题', '', tasks[pid + ':current']['question'], '']
        lines += [f'{chr(65 + i)}. {option}' for i, option in enumerate(tasks[pid + ':current']['options'])]
        lines += ['', f"![实际最近四对输入](<{(ROOT / pair['image_relative_path']).as_posix()}>)", '',
            '请填写：答案内容；充分证据帧；是否有歧义；是否只看最后一帧就足够。', '',
            '### 旧题', '', tasks[pid + ':old']['question'], '']
        lines += [f'{chr(65 + i)}. {option[3:] if option[:3] == chr(65 + i) + ". " else option}'
                  for i, option in enumerate(tasks[pid + ':old']['options'])]
        if pid in audited:
            lines += ['', f"作者定位区间：{audited[pid]['author_span_s']}秒（不是独立裁决）。", '',
                f"![旧证据实际采样画面](<{(ROOT / audited[pid]['old_card']['relative_path']).as_posix()}>)"]
        lines += ['', '请填写：独立答案；证据区间；截至提问时是否成立；近期是否也能推出答案。', '',
            '### 无关历史供体', '', f"供体源：`{Path(pair['donor_source_path']).name}`，使用0—7秒四对。", '']
        donor = next(p for p in manifest['pairs'] if p['source_path'] == pair['donor_source_path'])
        if donor['pair_id'] in audited:
            lines += [f"![供体实际采样画面](<{(ROOT / audited[donor['pair_id']]['donor_card']['relative_path']).as_posix()}>)", '']
        lines += ['请确认：供体没有提供目标当前题的真实场景证据；若题目措辞对供体也成立或存在明显答案线索，请记录。', '']
    path = PREP / '独立内容复核材料_LOCAL_ONLY.md'
    if path.exists():
        raise FileExistsError(path)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(dict(path=str(path), model_scores_in_packet=False, review_completed=False))


if __name__ == '__main__':
    main()
