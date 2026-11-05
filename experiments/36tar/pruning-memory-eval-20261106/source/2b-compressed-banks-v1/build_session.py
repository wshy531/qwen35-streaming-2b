"""Independent session with explicit timing and selector trace delivery."""
from pathlib import Path
HERE=Path(__file__).parent;ROOT=HERE.parents[1]
s=(ROOT/'experiments/2b-ls-depth-v1/session.py').read_text(encoding='utf-8')
s=s.replace('from readout import context, Limits','from bank_readout import context, Limits')
s=s.replace('Limits(2048, 384, 8, 128)','Limits(2048, 384, 16, 128)')
s=s.replace('six_banks','nine_banks').replace('six fixed banks','nine fixed banks').replace('ls-depth-qa','compressed-banks-qa')
s=s.replace('self.tail_snapshot_only = self.invalid = False','self.tail_snapshot_only = self.invalid = False\n        self.timing = dict(preprocess_seconds=0.,vision_seconds=0.,selection_banks_seconds=0.,pair_total_seconds=0.)')
s=s.replace('        inputs, _ = self.builder.pair', '        total_started=time.monotonic()\n        inputs, _ = self.builder.pair')
s=s.replace('        saved = self.model.model.rope_deltas\n        try:', '        saved = self.model.model.rope_deltas\n        if device.type==\'cuda\':torch.cuda.synchronize(device)\n        vision_started=time.monotonic()\n        try:',1)
s=s.replace('                self.bank.append(features, self.pairs, first.timestamp, second.timestamp)',
'''                if device.type=='cuda':torch.cuda.synchronize(device)
                vision_finished=time.monotonic()
                trace=self.bank.append(features, self.pairs, first.timestamp, second.timestamp)
                banks_finished=time.monotonic()''')
s=s.replace('        self.pairs += 1', '''        self.timing['preprocess_seconds']+=vision_started-total_started
        self.timing['vision_seconds']+=vision_finished-vision_started
        self.timing['selection_banks_seconds']+=banks_finished-vision_finished
        self.timing['pair_total_seconds']+=banks_finished-total_started
        self.pairs += 1''',1)
s=s.replace('        return self.status()','        return dict(**self.status(),selection_trace=trace)',1)
s=s.replace('before, digest = self.status(), self.bank.fingerprint()', 'before, digest = self.status(), self.bank.fingerprint()\n        selector_before=self.bank.selector_digest()')
s=s.replace('assert self.status() == before and self.bank.fingerprint() == digest','assert self.status() == before and self.bank.fingerprint() == digest\n        assert self.bank.selector_digest()==selector_before')
s=s.replace('banks=self.bank.status(), no_main_language_cache=True)', 'banks=self.bank.status(), no_main_language_cache=True)')
(HERE/'session.py').write_text(s,encoding='utf-8',newline='\n')
print('Nine-arm bank-only session prepared; no pretrained load or GPU')
