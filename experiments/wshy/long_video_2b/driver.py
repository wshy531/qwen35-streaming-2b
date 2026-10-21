"""Share pair encoding across bounded sessions, with synchronous fresh readouts."""
from .protocol import ARMS
from .scheduler import replay_source


def run_sessions(frames, queries, samples, encode_pair, readout, prompt, on_event, on_answer):
    """Readout must create/release fresh language state and guard model RNG/rope.

    Real-model callers must wrap it with online_memory_2b.runner.isolated_readout.
    This adapter owns visual sessions only. It never loads model weights.
    """
    from ..online_memory_2b.session import MemorySession
    from ..online_memory_2b.runner import guarded_banks

    sessions = {arm: MemorySession(arm, readout) for arm in ARMS[:-1]}
    answers = generated = encodes = 0

    def append(index, first, second):
        nonlocal encodes
        block = encode_pair(index, first, second)
        encodes += 1
        for arm, session in sessions.items():
            record = session.append_pair(block, frame_ids=(first.frame_id, second.frame_id), padded_tail=False)
            on_event(dict(event='append', arm=arm, **record))
        del block
        combined = sum(s.status()['resident_bytes'] for s in sessions.values())
        if combined > len(sessions)*(8*394560+8192):
            raise RuntimeError('combined visual bank budget exceeded')

    def ask(query, prefix):
        nonlocal answers, generated
        # Match the frozen question/options prompt without exposing label fields.
        text = prompt(dict(question=query['question'], options=list(query['options'])))
        for arm in ARMS:
            session = sessions.get(arm)
            result = guarded_banks(sessions, lambda: session.ask(text) if session else readout([], text))
            ids = result['generated_ids']
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 8
                    or any(type(value) is not int or value < 0 for value in ids)
                    or not isinstance(result['text'], str)):
                raise RuntimeError('invalid answer or generation budget exceeded')
            answers += 1
            generated += len(ids)
            on_answer(dict(id=query['id'], source_path=query['source_path'], query_time_s=query['query_time_s'],
                           arm=arm, prefix=dict(prefix), status='ok', **result))
        on_event(dict(event='query', id=query['id'], **prefix, all_banks_unchanged=True))

    try:
        result = replay_source(frames, queries, samples, append, ask)
        if result['encoded_pairs'] != encodes or answers != len(queries)*len(ARMS):
            raise RuntimeError('encoding or answer denominator mismatch')
        for arm, session in sessions.items():
            session.reset()
            fresh = MemorySession(arm)
            try:
                if session.state_digest() != fresh.state_digest() or session.status()['resident_bytes']:
                    raise RuntimeError('reset retained previous video')
            finally:
                fresh.close()
        return dict(result, answers=answers, generated_tokens=generated, reset_checks=len(sessions),
                    model_loads_by_adapter=0, all_QA_banks_unchanged=True)
    finally:
        for session in sessions.values():
            session.close()
