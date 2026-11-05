"""Versioned collaboration prompt; no answer keys, model calls or prompt search."""
import re

PROMPT_VERSION = 'wsy-sember-short-v1-20261027'
UPSTREAM_SHA256 = 'e232aefb69afa6c2dc53f63ed80cf1271df0123e148588f51a683a1cee83c710'


def clean_options(row):
    options = row['options']
    if not 2 <= len(options) <= 5 or any(not isinstance(v, str) or not v.strip() for v in options):
        raise ValueError('two to five nonempty options required')
    labels = [re.match(r'^([A-E])\.\s+', value) for value in options]
    if any(labels):
        if not all(labels) or any(match.group(1) != chr(65 + i) for i, match in enumerate(labels)):
            raise ValueError('mixed/reordered structural labels')
        options = [value[match.end():] for value, match in zip(options, labels)]
    if any(not value.strip() for value in options):
        raise ValueError('empty option content')
    return list(options)


def rotate(row, shift):
    options = clean_options(row)
    if type(shift) is not int or shift not in range(len(options)):
        raise ValueError('valid integer option rotation required')
    mapping = [(i + shift) % len(options) for i in range(len(options))]
    return dict(question=row['question'], options=[options[i] for i in mapping]), mapping


def question_prompt(row):
    options = clean_options(row)
    return row['question'] + '\n' + '\n'.join(f'{chr(65+i)}. {value}' for i, value in enumerate(options)) + \
        '\nReturn only the single option letter. Do not explain.'
