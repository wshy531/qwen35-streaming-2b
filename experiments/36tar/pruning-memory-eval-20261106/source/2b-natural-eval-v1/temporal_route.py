"""R0: one fixed, answer-blind question-type gate over an unchanged M2 bank."""
import re
TEMPORAL=re.compile(r'\b(?:before|after|earlier|initially|first)\b|\bat (?:the )?start\b|\b(?:at|in) (?:the )?beginning\b|\bin (?:the )?middle\b',re.I)
def select(bank,question):
    if not isinstance(question,str) or not question.strip():raise ValueError('nonempty original question required')
    recent=bank[-4:];old=bank[:-4]
    historical=bool(TEMPORAL.search(question));use_old=historical and bool(old)
    return dict(all_memory=list(bank),long_only=list(old or bank),recent4=list(recent),temporal_route=list(old if use_old else bank)),dict(has_temporal_cue=historical,used_long_only=use_old,empty_long_fallback=historical and not old,rule='fixed lexical cue; not semantic anchor localization',bank_nodes=len(bank),long_nodes=len(old),recent_nodes=len(recent))
