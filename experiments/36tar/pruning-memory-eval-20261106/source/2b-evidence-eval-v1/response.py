"""Deterministic answer parsing and explicit judge data boundaries; no network."""
import re
EVIDENCE_INSTRUCTION='''Give your selected option first, using "Answer: <letter>".
Then give 1-3 short observations from the provided video input under "Evidence:".
Use the supplied time labels where possible. Distinguish what is visible from an inference.
Under "Support:", briefly state how those observations support your answer.
Do not invent unseen events or future frames. If the supplied evidence is insufficient,
say so explicitly. Keep the response concise; do not give a long reasoning trace.'''
def question(q,shift,protocol):
    choices=q['options'][shift:]+q['options'][:shift]
    suffix='Answer with one option letter.' if protocol=='A' else EVIDENCE_INSTRUCTION
    assert protocol in ('A','B')
    return q['question']+'\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(choices))+'\n'+suffix
def parse(text,count,protocol,shift=0):
    allowed='ABCDE'[:count];conflict=False
    if protocol=='B':
        matches=list(re.finditer(r'(?im)^\s*Answer\s*:\s*([A-E])\b',text))
        selected=matches[0].group(1) if matches else None
        conflict=len({m.group(1) for m in matches})>1
    else:
        matches=re.findall(r'\b([A-E])\b',text)
        selected=matches[0] if len(set(matches))==1 else None
    if selected is None or selected not in allowed:selected=None
    evidence=re.search(r'(?is)\bEvidence\s*:\s*(.*?)(?:\bSupport\s*:|\Z)',text)
    support=re.search(r'(?is)\bSupport\s*:\s*(.*)\Z',text)
    return dict(selected_index=None if selected is None else ord(selected)-65,
        original_selected_index=None if selected is None else (ord(selected)-65+shift)%count,
        evidence=evidence.group(1).strip() if evidence else '',support=support.group(1).strip() if support else '',
        answer_conflict=conflict,answer_parseable=selected is not None,
        explanation_present=bool(evidence and evidence.group(1).strip()),
        only_letter_format=text.strip() in allowed)
def fact_payload(material,response):
    # Only allow known fields. Never forward complete question/answer metadata.
    return dict(material_id=material['material_id'],frames=material['frames'],
                complete=material['complete'],observations=response['evidence'])
def support_payload(q,response,facts):
    index=response['original_selected_index']
    return dict(question=q['question'],proposed_answer=None if index is None else q['options'][index],
                observations=response['evidence'],support=response['support'],verified_facts=facts)
