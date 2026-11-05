"""One frozen evidence-sufficiency question, two balanced code positions."""
import math

VERSION = 'recent-evidence-sufficiency-v1-20261028'
STATES = ('LOCAL', 'HISTORY')
INSTRUCTION = '''Assess whether the supplied recent video frames contain sufficient visual evidence to answer the question below. Do not answer the question itself.
Choose LOCAL only when the facts needed to distinguish the answer options, including any required event, state, count or temporal relation, are supported within these frames.
Choose HISTORY when earlier observations are required or the local evidence is missing, ambiguous or unreadable. Absence from this window does not establish absence earlier in the video. Do not substitute a plausible option or general knowledge for visual evidence.
This decision requests more evidence; it does not assert that historical memory will contain it.
'''

def render(question, options, view):
    if not isinstance(question, str) or not question.strip() or view not in (0, 1):
        raise ValueError('nonempty question and a registered view required')
    if not isinstance(options, list) or len(options) != 5 or any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError('five nonempty option contents required')
    order = STATES[view:] + STATES[:view]
    text = INSTRUCTION + '\n' + '\n'.join(f'{chr(65+i)} = {s}' for i, s in enumerate(order))
    text += '\n\nQuestion: ' + question + '\nOptions:\n'
    text += '\n'.join(f'{chr(65+i)}. {o}' for i, o in enumerate(options))
    text += '\nReturn only the decision code A or B, not an answer option.'
    return dict(text=text, codes=['A', 'B'], semantic_order=list(order), view=view)

def aggregate(views):
    if len(views) != 2 or {v['view'] for v in views} != {0, 1}:
        raise ValueError('exactly two complementary views required')
    scores = {s: 0.0 for s in STATES}
    for v in views:
        expected = list(STATES[v['view']:] + STATES[:v['view']])
        if v['semantic_order'] != expected or len(v['candidate_logits']) != 2:
            raise ValueError('semantic code mapping changed')
        x = v['candidate_logits']
        if any(not math.isfinite(a) for a in x):
            raise ValueError('finite logits required')
        m = max(x); z = m + math.log(sum(math.exp(a-m) for a in x))
        for state, a in zip(expected, x):
            scores[state] += (a-z) / 2
    route = 'recent' if scores['LOCAL'] > scores['HISTORY'] else 'full'
    return dict(route=route, semantic_logprob=scores, tie_falls_back_to_full=True,
                calibrated_confidence=False, history_sufficiency_not_asserted=True)
