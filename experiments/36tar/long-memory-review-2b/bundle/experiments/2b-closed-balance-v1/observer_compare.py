"""Share one real visual encoding and one observing core across three banks."""
from policy import ROOT,DepthForest384
import sys

sys.path.insert(0,str(ROOT/'experiments/2b-long-stream-v1'))
sys.path.insert(0,str(ROOT/'experiments/2b-memory-readability-v1'))
from observer import Observer
from online_bank import OnlineBank,nodes_digest
from diagnostic import answer
from first_2b_baseline import tensor_digest
from features import raw_block

ARMS=('M0','M2','A06')


class CompareObserver:
    def __init__(self,env,model,builder,*,main_tokens=2048):
        self.env,self.model,self.builder=env,model,builder
        self.core=Observer(env,model,builder,policy='M2',main_tokens=main_tokens,question_tokens=128,answer_tokens=8)
        self.m0=OnlineBank('M0');self.a06=DepthForest384()

    def nodes(self):return dict(M0=self.m0.export(),M2=self.core.bank.export(),A06=self.a06.export())

    def status(self):
        return dict(**self.core.status(),banks=dict(M0=self.m0.status(),M2=self.core.bank.status(),A06=self.a06.status()))

    def append_pair(self,first,second):
        original=self.model.model.get_video_features;captured=[]
        def capture(*args,**kwargs):
            assert not captured,'more than one visual encoding for one arriving pair'
            value=original(*args,**kwargs);captured.extend(value.pooler_output);return value
        self.model.model.get_video_features=capture
        try:state=self.core.append_pair(first,second)
        finally:self.model.model.get_video_features=original
        assert captured
        features=self.env['torch'].cat(captured,0);index=self.core.pairs-1
        self.m0.append(features,index,first.timestamp,second.timestamp)
        self.a06.append(raw_block(features,index,first.timestamp,second.timestamp),starts_event=state['bank']['starts_event'])
        captured.clear();del features
        result=self.status()
        assert result['banks']['A06']['tokens']==result['banks']['M2']['visual_tokens']<=384
        return result

    def ask(self,question,*,read_from,on_token):
        assert read_from=='three_banks'
        before=self.status();core_digest=tensor_digest(self.core.main.cache)
        banks=self.nodes();hashes={a:nodes_digest(ns) for a,ns in banks.items()};answers={}
        for arm in ARMS:
            answers[arm]=answer(self.env,self.model,self.builder,banks[arm],question,
                lambda item,a=arm:on_token(dict(arm=a,**item)))
        assert self.status()==before and tensor_digest(self.core.main.cache)==core_digest
        assert all(nodes_digest(ns)==hashes[a] for a,ns in self.nodes().items())
        return dict(arms=answers,question_did_not_change_main_or_banks=True)

    def close(self):self.core.close()
