"""Record the assistant's actual image review, not independent human admission."""
from common import ROOT, INPUTS, PREP, read, save, sha, require_development_review


OBSERVATIONS = {
    'P01:old': (2, [248.0, 250.0],
        'The red detergent container is lowered into the gray laundry hamper at 248-249s. '
        'The recent 260-267s screen frames do not establish its storage location.'),
    'P01:current': (0, [266.0, 267.0],
        'The extended finger points at the washing-area touchscreen. It may be a separate '
        'payment/control terminal; the image does not establish a built-in washer screen or physical contact.'),
    'P02:old': (3, [205.0, 209.0],
        'The held octopus plush is yellow/mustard. Green-white kitchen tools also appear in this old interval.'),
    'P02:current': (0, [336.0, 343.0],
        'The displayed kitchen tools are predominantly white and bright green. The same tools/colors '
        'also appear in the old evidence, so history may legitimately support this answer.'),
    'P03:old': (None, [69.0, 77.0],
        'SLEEP WEAR is readable on the first pale shirt at 69-70s. A second purple set is handled later, '
        'but its specific white sticker text is not confidently readable in these sampled pixels. '
        'Do not transfer the first sticker reading to the second set. Keep official index 3 unchanged.'),
    'P03:current': (0, [96.0, 97.0],
        'The foreground rolling display holds suitcases and travel bags. The question at 98s '
        'must use committed evidence only through 97s.'),
}


def main():
    manifest = read(INPUTS / 'manifest_LOCAL_ONLY.json')
    tasks = read(INPUTS / 'tasks_LOCAL_ONLY.json')
    audit = {r['pair_id']: r for r in read(PREP / 'cluster_media_CPU.json')['sources']}
    pairs = {p['pair_id']: p for p in manifest['pairs']}
    rows = []
    for task in tasks:
        answer, interval, note = OBSERVATIONS[task['id']]
        pair = pairs[task['pair_id']]
        relative = (pair['image_relative_path'] if task['role'] == 'current'
                    else audit[task['pair_id']]['old_card']['relative_path'])
        assert interval[1] <= pair['committed_through_s']
        rows.append(dict(task_id=task['id'], pair_id=task['pair_id'], role=task['role'],
            decision='include_development' if answer is not None else 'uncertain_reference_only',
            reviewed_correct_index=answer, unchanged_frozen_correct_index=task['correct_index'],
            evidence_sufficient=answer is not None, primary_content_scored=answer is not None,
            evidence_intervals_s=[interval], observation=note,
            evidence_images=[dict(relative_path=relative, sha256=sha(ROOT / relative))],
            reviewed_before_new_model_answers=True))
    donor_reviews = []
    for pair in manifest['pairs']:
        donor = next(p for p in manifest['pairs'] if p['source_path'] == pair['donor_source_path'])
        card = audit[donor['pair_id']]['donor_card']
        donor_reviews.append(dict(target_pair_id=pair['pair_id'], donor_source_pair_id=donor['pair_id'],
            source_path=pair['donor_source_path'], interval_s=[0.0, 7.0],
            supplies_actual_target_scene_evidence=False,
            shared_colors_or_generic_objects_not_excluded=True,
            evidence_images=[dict(relative_path=card['relative_path'], sha256=card['sha256'])],
            limitation='Different-source values are an artificial intervention; not proof of zero semantic overlap.'))
    review = dict(schema='assistant-development-review-v1', reviewed_date='2026-10-27',
        scope='assistant_reviewed_development_only', reviewer_role='assistant',
        reviewer_identity='same project assistant; author and reviewer not independent',
        independent_review=False, formal_benchmark_admitted=False,
        user_delegation='你能自己做吗，我不太想审核',
        manifest_sha256=sha(INPUTS / 'manifest_LOCAL_ONLY.json'), tasks=rows, donor_reviews=donor_reviews,
        old_evidence_necessity_not_established_by_this_review=True,
        new_model_answers_seen=False, GPU_authorized=False)
    require_development_review(manifest, review)
    save(PREP / 'assistant_review_LOCAL_ONLY.json', review)
    policy = dict(schema='development-primary-score-overlay-v1',
        manifest_sha256=review['manifest_sha256'],
        review_sha256=sha(PREP / 'assistant_review_LOCAL_ONLY.json'),
        unchanged_schedule_answers=90, primary_content_answers=65,
        original_label_reference_answers=75, uncertain_primary_exclusions=10,
        historical_value_swap_old_answers_unscored=15,
        primary_task_ids=[r['task_id'] for r in rows if r['primary_content_scored']],
        reference_only_task_ids=['P03:old'], primary_pair_ids=['P01', 'P02'],
        official_labels_changed=False, all_raw_answers_retained=True,
        independent_human_review_required_for_this_development_run=False,
        formal_independent_admission=False)
    save(PREP / 'review_scoring_policy_LOCAL_ONLY.json', policy)
    report = PREP / '助手内容审核结果_LOCAL_ONLY.md'
    text = '''# 三组开发配对：助手内容审核结果

用户已将本轮审核交由助手完成，无需再由用户填写复核表。本文件是内容审核快照，持续进度只维护《实验总纲.md》。

本次实际查看三张近期输入卡、三张作者旧证据区间卡和三张供体卡。审核者也是前期问题编写助手，已见官方标签，因此不是独立或盲审；用途限于开发诊断。没有查看新增模型回答，没有按新成绩筛题。

| 组别 | 当前题 | 旧题 | 开发用途 |
|---|---|---|---|
| P01 | 手指指向洗衣区域操作触摸屏，证据清楚 | 248—249秒可见洗衣液放入灰色洗衣篮 | 当前题与旧题均进入主要内容分析 |
| P02 | 厨具主要为白色和绿色，证据清楚 | 205—209秒拿起的章鱼玩具为黄色 | 两题进入主要分析；旧图已有绿白厨具，标注合理重复线索 |
| P03 | 96—97秒前景陈列架为行李箱和旅行包，证据清楚 | 第一套贴纸可读，但第二套目标贴纸的文字不足以可靠确认 | 当前题进入主要分析；旧题保留原标签与全部回答，单列参考 |

P01只能判断指向触摸屏，不能由这些图断言触屏一定内置于洗衣机、或手指确实接触屏幕。P02的颜色存在黄绿色边界，但与给定其他选项能清楚区分。P03不使用尚未提交的98秒画面。旧区间有证据，不等于全前缀无后续重复；本轮不宣称已证明长期记忆必需。

供体画面均来自另一真实来源0—7秒，不提供目标当前场景的实际证据；但常见颜色或物品可能重合。因此替换控制若引起答案变化，只支持特定历史特征值参与读取变化，不能单独证明唯一语义或内部注意力机制。

原定90次回答和官方标签不变：主要内容评分65次；原标签参照仍保留75次；其中P03旧题A/B的10次单列存疑参考，另15次旧题历史替换不按原真值计分。两题同时正确只对P01/P02报告，五轮换分母每臂10，不能称10个独立配对。睡衣旧题不是被删除的模型失败，而是在新结果产生之前标记的证据疑点。

这完成了开发材料审核，不代表真实2B测试已经运行或新结构已有效。后续执行必须同时绑定原清单和本次评分覆盖文件，重新核验修改过的准入/评分源码；原CPU回执不冒充最新源码验证。独立人工复核仍为0，但不再作为本轮开发诊断的前置条件。正式独立评测的要求不由本记录满足。
'''
    with report.open('x', encoding='utf-8') as handle:
        handle.write(text)
    print(dict(status='assistant_development_review_complete', primary_tasks=5,
        uncertain_reference_tasks=1, primary_content_answers=65, raw_answers_planned=90,
        human_review_wait=False, GPU_jobs=0))


if __name__ == '__main__':
    main()
