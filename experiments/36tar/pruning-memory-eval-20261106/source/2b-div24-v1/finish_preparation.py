"""Freeze primary answer scoring and update the requested eight-arm combination plan."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent

def main():
    text=(ROOT/'experiments/2b-ls-depth-v1/analyze.py').read_text(encoding='utf-8')
    text=text.replace("(('LS_depth','LS'),('LS_depth','A06'),('LS_depth','M2'),\n                                   ('LS_depth','reservoir4'),('A06','M2'),('LS','M2'))",
        "tuple((a,b) for a in ARMS for b in ARMS if a != b)")
    text=text.replace('expected_answers=1818','expected_answers=1515').replace('expected=1818','expected=1515')
    text=text.replace("'bank-only costs not comparable to observation-main runs'", "'projection compression only; observation main unchanged; no end-to-end speed claim'")
    (HERE/'analyze.py').write_text(text,encoding='utf-8',newline='\n')
    plan=ROOT/'下一阶段计划_因果视觉Token压缩与流式成本验证_20261023.md'
    text=plan.read_text(encoding='utf-8')
    text=text.replace('记忆侧固定M0近期窗口和LS两个版本，形成四臂：',
        '按用户2026-10-23补充，记忆侧固定M0近期窗口、原记忆森林M2、A06、LS四个版本，形成八臂：')
    text=text.replace('| M0 | M0-48 | M0-24 |\n| 固定LS | LS-48 | LS-24 |',
        '| M0 | M0-48 | M0-24 |\n| 原记忆森林M2 | M2-48 | M2-24 |\n| A06 | A06-48 | A06-24 |\n| 固定LS | LS-48 | LS-24 |')
    text=text.replace('共1212答', '共2424答').replace('P3的四臂', 'P3的八臂')
    text=text.replace('目标144×4=576答','目标144×8=1152答')
    text=text.replace('P3四臂＋同覆盖诊断','P3八臂＋同覆盖诊断').replace('1280答上限','2492答上限')
    text=text.replace('P4新来源四臂','P4新来源八臂').replace('目标576答','目标1152答')
    text=text.replace('P3，四臂同384', 'P3，八臂同384')
    text=text.replace('再运行候选','再运行候选')
    text=text.replace('冻结后再运行候选','冻结后再运行候选')
    text=text.replace('| 固定LS |', '| 固定LS |')
    text=text.replace('首次只验证一个压缩器；不同时改LS边界', '首次只验证一个压缩器；M2/A06/LS各自冻结，不同时改LS边界')
    text=text.replace('同容量组合验证；', '同容量组合验证；')
    # Later stages retain a proposal ceiling; never silently extend the wall time.
    text += '\n\n2026-10-23执行补充：组合新增原M2和A06后，P3为303×8＋34×2=2492答、最多19936生成Token；P4目标144×8=1152答。每槽60分钟仍是待预检的预算上限，须用首轮实测估算是否可容纳；不足时先重新说明预算或冻结不按结果挑选的分批安排，不自动加时。\n'
    plan.write_text(text,encoding='utf-8',newline='\n')

if __name__=='__main__': main()
