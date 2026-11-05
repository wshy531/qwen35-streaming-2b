"""Pinned official prompt and MCQ parser, with format reported separately."""
import re


def prompt(q):
    options=[v if re.match(r'^[A-E]\.\s',v) else f'{chr(65+i)}. {v}' for i,v in enumerate(q['options'])]
    return ('After reviewing the video, answer the following multiple-choice question.\n\n'
        f'Question: {q["question"]}\n\n'+'\n'.join('  '+v for v in options)+'\n\n'
        f'IMPORTANT: Respond with ONLY a single letter ({"/".join(chr(65+i) for i in range(len(options)))}). '
        'Do NOT include any explanation, reasoning, or additional text. Just the letter.\n\nAnswer:')


def parse_choice(text,num_options):
    if not text:return None
    text=text.strip().upper();valid=list('ABCDE'[:num_options])
    if text in valid:return text
    match=re.search(r'\b([A-E])\b',text)
    return match.group(1) if match and match.group(1) in valid else None


def format_ok(text,num_options):return text.strip() in list('ABCDE'[:num_options])
