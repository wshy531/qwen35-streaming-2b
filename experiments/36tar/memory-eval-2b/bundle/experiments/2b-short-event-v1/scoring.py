"""Frozen dual measures: option content, strict formatting; never rewrite past scores."""
import re

STRICT = re.compile(r'\s*([A-Ea-e])\s*[.!]?\s*')
LEADING = re.compile(r'^\s*([A-Ea-e])(?:\b|[.!)\s])')

def measure(text, n_options, truth):
    if type(n_options) is not int or not 2 <= n_options <= 5:
        raise ValueError('2 to 5 options required')
    if not isinstance(text, str):
        text = ''
    strict = STRICT.fullmatch(text)
    leading = LEADING.match(text)
    def index(match):
        i = ord(match[1].upper()) - ord('A') if match else -1
        return i if 0 <= i < n_options else None
    a, b = index(strict), index(leading)
    return dict(strict_choice=a, content_choice=b, strict_correct=a == truth,
                content_correct=b == truth, format_compliant=a is not None,
                content_parseable=b is not None,
                interpretation='first explicit option only; explanation semantics not graded')
