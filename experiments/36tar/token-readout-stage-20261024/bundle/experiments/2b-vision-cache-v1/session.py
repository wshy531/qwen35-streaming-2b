"""Two isolated M0 banks, one causal input stream, explicit visual baseline."""
from common import ARMS
from solo_session import Session as SoloSession


class Session:
    def __init__(self, env, model, builder, cache):
        self.cache = cache
        self.sessions = {arm:SoloSession(env, model, builder, arm) for arm in ARMS}
        self.pairs = self.queries = 0
        self.vision_encodings = 0

    def _append(self, first, second=None):
        order = ARMS[self.pairs%2:] + ARMS[:self.pairs%2]
        for arm in order:
            session = self.sessions[arm]
            if arm == 'full48_8':
                before = self.cache.fingerprint()
                with self.cache.bypass():
                    state = session.snapshot_tail(first) if second is None else session.append_pair(first, second)
                assert self.cache.fingerprint() == before
            else:
                state = session.snapshot_tail(first) if second is None else session.append_pair(first, second)
                assert self.cache.last['pair_index'] == self.pairs
        self.pairs += 1
        self.vision_encodings += 2
        return dict(**self.status(), cache_last=self.cache.last, encoding_order=list(order))

    def append_pair(self, first, second):
        return self._append(first, second)

    def snapshot_tail(self, packet):
        return self._append(packet)

    def status(self):
        states = {a:s.status() for a,s in self.sessions.items()}
        assert len({s['pairs'] for s in states.values()}) == 1
        assert len({s['last_pts'] for s in states.values()}) == 1
        return dict(pairs=self.pairs, last_pts=states[ARMS[0]]['last_pts'],
                    vision_encodings=self.vision_encodings, arms=states,
                    reference_bytes=self.cache.reference_bytes(),
                    no_main_language_cache=True)

    def ask(self, question, *, read_from, on_token):
        if read_from != 'visual_cache':
            raise ValueError('explicit visual cache comparison required')
        before = self.cache.fingerprint()
        order = ARMS[self.queries%2:] + ARMS[:self.queries%2]
        results = {}
        for arm in order:
            result = self.sessions[arm].ask(question, read_from='standalone',
                on_token=lambda item:on_token(item))
            results[arm] = result['arms'][arm]
        assert self.cache.fingerprint() == before
        self.queries += 1
        return dict(arms=results, order=list(order), visual_reference_unchanged_by_QA=True,
                    all_QA_branches_released=True)

    def close(self):
        for session in self.sessions.values():
            session.close()
        self.cache.reset()
