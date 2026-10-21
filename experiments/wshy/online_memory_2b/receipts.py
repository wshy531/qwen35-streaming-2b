"""Bind successful inherited regression output to the current CPU proof."""
import re
import sys

from .protocol import read, sha

INHERITED_COUNTS = [3, 12, 186, 8]
WINDOWS_OWNER_TEST = 'test_live_source.LiveSourceTests.test_capture_worker_exits_when_parent_dies'
WINDOWS_OWNER_REASON = 'Windows owner-handle guard'


def successful_counts(text, expected):
    counts = [int(value) for value in re.findall(r'^Ran (\d+) tests? in .+$', text, re.M)]
    if (counts != expected or len(re.findall(r'^OK$', text, re.M)) != len(expected)
            or re.search(r'^(?:FAILED|OK \().*$', text, re.M)):
        raise ValueError('incomplete, skipped or failed unittest suites')
    return counts


def inherited_result(text, platform=None):
    """Keep the existing Windows-only skip explicit; reject every other skip."""
    platform = sys.platform if platform is None else platform
    ran = list(re.finditer(r'^Ran (\d+) tests? in .+$', text, re.M))
    summaries = list(re.finditer(r'^OK(?: \(skipped=(\d+)\))?$', text, re.M))
    skipped = re.findall(r'^test_.* \.\.\. skipped .*$', text, re.M)
    if ([int(m[1]) for m in ran] != INHERITED_COUNTS or len(summaries) != 4
            or len(re.findall(r'^OK.*$', text, re.M)) != 4
            or re.search(r'^(?:FAILED|ERROR:|FAIL:)', text, re.M)):
        raise ValueError('incomplete or failed inherited unittest suites')
    flags = [int(m[1] or 0) for m in summaries]
    allowed_line = (WINDOWS_OWNER_TEST.rsplit('.',1)[1]+' ('+WINDOWS_OWNER_TEST+
                    ") ... skipped '"+WINDOWS_OWNER_REASON+"'")
    exceptions = []
    if skipped or any(m[1] is not None for m in summaries):
        match = re.search('^'+re.escape(allowed_line)+'$', text, re.M)
        if (not platform.startswith('linux') or skipped != [allowed_line]
                or flags != [0,0,1,0] or match is None
                or not summaries[1].end() < match.start() < ran[2].start()):
            raise ValueError('unexpected inherited platform skip')
        exceptions = [dict(test=WINDOWS_OWNER_TEST, reason=WINDOWS_OWNER_REASON,
                           classification='not_applicable_on_linux')]
    return dict(suite_counts=list(INHERITED_COUNTS), tests_discovered=sum(INHERITED_COUNTS),
                tests_run=sum(INHERITED_COUNTS)-len(exceptions), tests_skipped=len(exceptions),
                platform_skips=exceptions)


def verify_inherited(out, source_lock_sha256, receipt_sha256, log_sha256):
    receipt_path, log_path = out / 'inherited.json', out / 'inherited.txt'
    if sha(receipt_path) != receipt_sha256 or sha(log_path) != log_sha256:
        raise ValueError('inherited CPU receipt/log identity changed')
    receipt = read(receipt_path)
    result = inherited_result(log_path.read_text(encoding='utf-8'))
    if (receipt['status'] != 'passed' or any(receipt.get(k) != v for k,v in result.items())
            or receipt['source_lock_sha256'] != source_lock_sha256
            or receipt['log_sha256'] != log_sha256):
        raise ValueError('matching successful 209-test receipt required')
    return receipt
