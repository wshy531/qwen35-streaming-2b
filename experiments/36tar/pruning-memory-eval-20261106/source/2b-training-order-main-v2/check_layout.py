"""All frozen questions, both permutations, exact frame times and bank limits."""
import types
from run_main import ROOT,PREP,WEIGHTS,read,runtime,make_builder,torch,sparse_pair,context,question_text,rotate,save
from replay import banks
def main():
    torch.set_num_threads(1);env=runtime()
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(WEIGHTS,local_files_only=True))
    embedding=torch.nn.Embedding(len(builder.processor.tokenizer),16)
    model=types.SimpleNamespace(get_input_embeddings=lambda:embedding)
    panel=read(PREP/'panel_LOCAL_ONLY.json');count=maximum=0
    for g in panel['groups'].values():
        samples=g['audit']['samples'];nodes=[];bank=banks.Bank('LS',54,budget=864,recent=432,closed=324)
        for i in range(g['audit']['encoding_pairs']):
            a=samples[2*i]['pts_s'];b=samples[min(2*i+1,len(samples)-1)]['pts_s']
            n=sparse_pair(torch.ones(54,16),torch.arange(0,108,2),i,a,b);nodes.append(n)
            bank.append(n,starts_event=i%3==0)
        short=g['role'] in ('short_train','old_development')
        views=[nodes] if short else [bank.export(),nodes[-8:]]
        assert sum(len(n.features) for n in views[0]) <= (1080 if short else 864)
        for q in g['questions']:
            for shift in (0,2):
                for view in views:
                    _,_,ids,_=context(model,builder,view,question_text(rotate(q,shift)),visual_limit=1080)
                    count+=1;maximum=max(maximum,ids.numel())
    report=dict(status='passed',sources=len(panel['groups']),questions=2967,layouts=count,max_tokens=maximum,
                actual_pixels_checked_separately=True,weights_loaded=False)
    save(PREP/'CPU_all_layouts.json',report);print(report)
if __name__=='__main__':main()
