"""Prepare bounded review packets while the approved relay runs (CPU only)."""
from collections import deque
import hashlib
import time

from audit_validation import media_record
from fetch_validation import ROOT, read, save
from review_diagnosis_prefixes import V0Pairs, sheet

OUT = ROOT / 'runs/diagnosis_expansion_review_20261105_01'
DOWNLOAD = ROOT / 'runs/diagnosis_expansion_download_20261105_01/download.json'


def main():
    queue = read(ROOT / 'runs/eval_consolidation_20261105_01/new_diagnosis_source_queue.json')
    raw = read(ROOT / 'research/dataset-survey-20261029/questions_real.json')
    receipts = {r['name']: r for r in read(ROOT / 'runs/eval_consolidation_20261105_03/acquisition.json')['complete']
                if r.get('panel') == 'diagnosis'}
    # The pilot receipt records its panel separately in its acquisition plan.
    if not receipts:
        names = {q['source_id'] for q in queue}
        receipts = {r['name']: r for r in read(ROOT / 'runs/eval_consolidation_20261105_03/acquisition.json')['complete']
                    if r['name'] in names}
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'review_packets.json').exists()
    results, labels = [], {}
    started = time.monotonic()
    while time.monotonic() - started < 3500:
        progress = read(DOWNLOAD)
        receipts.update({r['name']: r for r in progress['complete']})
        done = {r['source_id'] for r in results}
        available = [q for q in queue if q['source_id'] in receipts and q['source_id'] not in done]
        for item in available:
            name = item['source_id']
            receipt = receipts[name]
            early = item['early_anchor']['query_time_s']
            late = item['current_anchor']['query_time_s']
            row = dict(source_id=name, media_path=receipt['media_path'], media_sha256=receipt['sha256'],
                       early_query_s=early, late_query_s=late,
                       review_status='pending_content_review', source_capture_isolation='unverified')
            try:
                info = media_record(receipt, query_times=(early, late))
                assert 0 < early < late <= info['duration_s']
                assert all(v is not None for v in info['near_cutoff_pts'].values())
                recent, overview = deque(maxlen=8), []
                targets = deque(late * i / 15 for i in range(16))
                trace = []

                def commit(first, second, padding=False):
                    assert not padding and first.timestamp < second.timestamp <= late + 1e-8
                    recent.append((first, second))
                    for packet in (first, second):
                        trace.append(dict(frame_id=packet.frame_id, pts=packet.timestamp,
                                          rgb_sha256=hashlib.sha256(packet.rgb.tobytes()).hexdigest()))
                        if targets and packet.timestamp >= targets[0]:
                            overview.append(packet)
                            while targets and packet.timestamp >= targets[0]:
                                targets.popleft()

                sampler = V0Pairs(ROOT / receipt['media_path'], commit, config='auto4')
                folder = OUT / name[:-4]
                folder.mkdir()
                try:
                    first_status = sampler.advance(early)
                    early_packets = [p for pair in recent for p in pair]
                    assert early_packets
                    row['early_window'] = sheet(folder / 'early.jpg', early_packets)
                    last_status = sampler.advance(late)
                    current = [p for pair in recent for p in pair]
                    assert len(current) == 16
                    row['recent_window'] = sheet(folder / 'recent8.jpg', current)
                    row['overview'] = sheet(folder / 'overview.jpg', overview)
                finally:
                    sampler.close()
                row.update(technical_pass=True, **info, early_sampler=first_status, late_sampler=last_status)
                save(folder / 'trace.json', trace)
                for key, slot in (('early_anchor', 'early'), ('current_anchor', 'current')):
                    q = dict(item[key])
                    original = raw[q['annotation_row']]['questions'][q['question_index']]
                    labels[q['id']] = original['answer']
                    row[slot + '_question'] = q
            except Exception as error:
                row.update(technical_pass=False, error_type=type(error).__name__, error=str(error)[:240])
            results.append(row)
            save(OUT / 'review_packets.json', results)
            save(OUT / 'official_anchor_labels_LOCAL_ONLY.json', labels)
            print(f'{len(results)}/30 packets, technical_pass={sum(r["technical_pass"] for r in results)}', flush=True)
        if progress['status'] != 'running' and not available:
            break
        if not available:
            time.sleep(8)
    save(OUT / 'preparation_receipt.json', dict(processed=len(results), expected=30,
         technical_pass=sum(r['technical_pass'] for r in results),
         status='complete' if len(results) == 30 else 'partial', seconds=time.monotonic()-started,
         content_admission_not_implied=True, GPU=0))


if __name__ == '__main__':
    main()
