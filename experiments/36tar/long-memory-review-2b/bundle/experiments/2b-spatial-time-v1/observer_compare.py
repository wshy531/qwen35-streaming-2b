"""Five preregistered banks, one visual pass, unchanged observing hybrid core."""
from policy import ROOT,FixedForest,RecentBank,raw_block
import sys

sys.path.insert(0,str(ROOT/'experiments/2b-long-stream-v1'))
sys.path.insert(0,str(ROOT/'experiments/2b-memory-readability-v1'))
from observer import Observer
from online_bank import OnlineBank,nodes_digest
from diagnostic import answer
from first_2b_baseline import tensor_digest

ARMS=('M0_384','M2_384','M0_768','M2_768','S24_384')
BUDGETS={a:768 if a.endswith('768') else 384 for a in ARMS}


class CompareObserver:
    def __init__(self,env,model,builder,*,main_tokens=2048):
        self.env,self.model,self.builder=env,model,builder
        self.core=Observer(env,model,builder,policy='M2',main_tokens=main_tokens,question_tokens=128,answer_tokens=8)
        self.m0=OnlineBank('M0');self.recent768=RecentBank(768)
        self.forest768=FixedForest(budget=768);self.s24=FixedForest(budget=384,old_tokens=24)

    def live_nodes(self):
        return dict(M0_384=[p.payload for p in self.m0.bank.read()],M2_384=self.core.bank.bank.nodes(),
            M0_768=self.recent768.nodes(),M2_768=self.forest768.nodes(),S24_384=self.s24.nodes())

    def nodes(self):return {a:[n.copy() for n in ns] for a,ns in self.live_nodes().items()}

    def status(self):
        banks=dict(M0_384=self.m0.status(),M2_384=self.core.bank.status(),M0_768=self.recent768.status(),
            M2_768=self.forest768.status(),S24_384=self.s24.status())
        profiles={a:[dict(first=n.first,last=n.last,max_token_mass=float(n.mass.max()),
            max_mass_fraction=float(n.mass.max()/n.mass.sum())) for n in ns] for a,ns in self.live_nodes().items()}
        return dict(**self.core.status(),banks=banks,mass_profiles=profiles)

    def append_pair(self,first,second):
        original=self.model.model.get_video_features;captured=[]
        def capture(*args,**kwargs):
            assert not captured
            value=original(*args,**kwargs);captured.extend(value.pooler_output);return value
        self.model.model.get_video_features=capture
        try:state=self.core.append_pair(first,second)
        finally:self.model.model.get_video_features=original
        assert captured;features=self.env['torch'].cat(captured,0);index=self.core.pairs-1
        self.m0.append(features,index,first.timestamp,second.timestamp)
        block=raw_block(features,index,first.timestamp,second.timestamp)
        self.recent768.append(block)
        for bank in (self.forest768,self.s24):bank.append(block,starts_event=state['bank']['starts_event'])
        captured.clear();del features,block
        for arm,nodes in self.live_nodes().items():assert sum(len(n.features) for n in nodes)<=BUDGETS[arm]
        return self.status()

    def ask(self,question,*,read_from,on_token):
        assert read_from=='five_banks'
        before=self.status();core_digest=tensor_digest(self.core.main.cache)
        banks=self.nodes();hashes={a:nodes_digest(ns) for a,ns in banks.items()};answers={}
        for arm in ARMS:
            answers[arm]=answer(self.env,self.model,self.builder,banks[arm],question,
                lambda item,a=arm:on_token(dict(arm=a,**item)))
        assert self.status()==before and tensor_digest(self.core.main.cache)==core_digest
        assert all(nodes_digest(ns)==hashes[a] for a,ns in self.nodes().items())
        return dict(arms=answers,question_did_not_change_main_or_banks=True)

    def close(self):self.core.close()
