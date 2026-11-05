"""Rebuild the consolidated metadata package from preserved source snapshots."""
from pathlib import Path
import hashlib
import json

import package_dataset
import register_diagnostic_expansion
import expand_pair_drafts
import write_dataset_navigation
import validate_package


def main():
    root, dest = package_dataset.ROOT, package_dataset.DEST
    names = ['development', 'regression', 'development_streamingbench',
             'development_short', 'development_diagnosis', 'short_validation', 'streamingbench_candidate']
    old = {n: hashlib.sha256((dest / (n + '.json')).read_bytes()).hexdigest() for n in names}
    package_dataset.main()
    register_diagnostic_expansion.main()
    expand_pair_drafts.main()
    write_dataset_navigation.main()
    validate_package.main()
    assert old == {n:hashlib.sha256((dest / (n + '.json')).read_bytes()).hexdigest() for n in names}
    document = dest / '使用说明.md'
    text = document.read_text('utf-8').replace('首批3组6题草案', '11组22题配对草案')
    anchor_row = '| diagnosis_anchors | 30视频60道作者原题 | 指定早晚时刻的普通读取核验；不等同历史依赖题 |\n'
    if anchor_row not in text:
        text = text.replace('| diagnosis_candidate |', anchor_row + '| diagnosis_candidate |', 1)
    text = text.replace('两个candidate清单保留准入限制。',
        '两个candidate清单保留准入限制。新增diagnosis_anchors可执行普通原题读取，但其early/current只是作者原题所在时刻，不是近期充分/历史必要标签。')
    if 'complete_package.py' not in text:
        text += '\n从保留快照重建全部清单使用 `python experiments/eval-consolidation-20261105/complete_package.py`；单独运行旧package_dataset.py只会重建首批核心。重建不下载媒体、不调用模型。\n'
    document.write_text(text, encoding='utf-8')
    print('All seven pre-existing frozen manifest hashes preserved; expanded package ready.')


if __name__ == '__main__':
    main()
