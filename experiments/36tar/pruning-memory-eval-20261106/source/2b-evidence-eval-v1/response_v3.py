"""Earliest explicit answer wins, with all later disagreements retained."""
import re
from response import parse as baseline_parse
def parse(text,count,protocol,shift=0):
    result=baseline_parse(text,count,protocol,shift)
    if protocol=='A':return dict(result,parser_version='v3_earliest_explicit')
    candidates=[]
    for m in re.finditer(r'(?im)^\s*(?:\*\*)?Answer\s*:\s*([A-E])\b',text):
        candidates.append((m.start(),m.group(1),m.end()))
    first=re.match(r'\A\s*(?:\*\*)?([A-E])\s*[.)]\s+[^\n]+',text)
    if first:candidates.append((first.start(),first.group(1),first.end()))
    candidates.sort();valid=[c for c in candidates if c[1] in 'ABCDE'[:count]]
    result.update(selected_index=None,original_selected_index=None,answer_parseable=False,
                  answer_conflict=len({x[1] for x in valid})>1,parser_version='v3_earliest_explicit')
    if valid:
        selected=ord(valid[0][1])-65
        result.update(selected_index=selected,original_selected_index=(selected+shift)%count,answer_parseable=True)
    result['raw_explanation_available']=bool(valid and text.splitlines()[1:])
    # No unlabeled sentence is silently promoted to verified Evidence/Support.
    return result
