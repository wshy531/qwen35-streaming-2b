"""Post-profile parser repair: explicit leading 'A. answer' is an answer too.

Old parser and raw outputs remain frozen. This change is applied uniformly to
all methods/protocols, not only wrong or favorable rows. No generation changes.
"""
import re
from response import parse as old_parse,fact_payload,support_payload
def parse(text,count,protocol,shift=0):
    result=old_parse(text,count,protocol,shift)
    result['parser_version']='v2_explicit_leading_choice'
    result['used_leading_choice_fallback']=False
    if protocol=='B' and not result['answer_parseable']:
        leading=re.match(r'\A\s*(?:\*\*)?([A-E])\s*[.)]\s+[^\n]+',text)
        if leading and leading.group(1) in 'ABCDE'[:count]:
            index=ord(leading.group(1))-65
            result.update(selected_index=index,original_selected_index=(index+shift)%count,
                          answer_parseable=True,used_leading_choice_fallback=True)
    return result
