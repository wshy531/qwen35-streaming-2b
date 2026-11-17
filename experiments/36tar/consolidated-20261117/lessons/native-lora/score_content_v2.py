"""Post-run content parsing correction: leading option is not an English article."""
import re,sys,runpy
from pathlib import Path
import analyze
legacy=analyze.choice
def choice(text,n):
    cleaned=text.strip().upper();allowed='ABCDE'[:n]
    # Multiple line-start options may be a list, not a committed answer.
    lines=re.findall(r'^\s*\*{0,2}(['+allowed+r'])[.)：:]\s+',cleaned,re.M)
    if len(set(lines))>1:return None
    match=re.match(r'^\*{0,2}(['+allowed+r'])(?:[.)：:](?:\s|$)|$)',cleaned)
    if match:return ord(match.group(1))-65
    if re.search(r'\b(?:ANSWER|OPTION|CHOICE)\b',cleaned):return legacy(text,n)
    return None
cases=[('B. A white sweater and black pants.',4,1),('D. A pedestrian bridge.',4,3),('Answer: (B).',4,1),('B or C',4,None),('B. cup\nC. bowl',4,None),('A cup is visible.',4,None),('E.',4,None),('B',4,1)]
if __name__=='__main__':
    for text,n,gold in cases:assert choice(text,n)==gold,(text,choice(text,n))
    analyze.choice=choice
    analyze.main()
