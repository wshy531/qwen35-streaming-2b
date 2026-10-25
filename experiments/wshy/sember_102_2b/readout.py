"""Explicit complete-question allowance; all frozen readout math is reused."""
from functools import partial
import re

SUFFIX_TOKENS = 512
PROMPT_VERSION = 'single-option-label-v1'


def question_prompt(row):
    """Preserve option content/order; remove one existing *structural* label.

    Accept fully labelled or fully unlabelled options. Mixed/reordered labels
    fail rather than silently moving answer content. Never mutate source rows.
    """
    options = row['options']
    if not 2 <= len(options) <= 5 or any(not isinstance(v, str) or not v.strip() for v in options):
        raise ValueError('two to five nonempty options required')
    labels = [re.match(r'^([A-E])\.\s+', v) for v in options]
    if any(labels):
        if not all(labels) or any(m.group(1) != chr(65+i) for i, m in enumerate(labels)):
            raise ValueError('mixed or reordered embedded option labels')
        options = [v[m.end():] for v, m in zip(options, labels)]
    if any(not v.strip() for v in options):
        raise ValueError('empty option content')
    return row['question'] + '\n' + '\n'.join(
        f'{chr(65+i)}. {v}' for i, v in enumerate(options)) + \
        '\nReturn only the single option letter. Do not explain.'


def complete_question_runtime(env):
    adapted = dict(env)
    adapted['Limits'] = partial(env['Limits'], suffix_tokens=SUFFIX_TOKENS)
    adapted['prompt'] = question_prompt
    return adapted
