"""Random tiny hybrid-language interface check; no pretrained weights or GPU."""
import json
import os
import sys
from guard import *
from qa import ask
sys.path.insert(0,str(ROOT/'experiments/2b-prune-checkpoints-v1'))
from worker import runtime, make_builder


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    env = runtime()
    proc = env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg = env['tiny_config'](proc)
    model = env['Qwen3_5ForConditionalGeneration'](
        env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder = make_builder(env, proc)
    width = model.config.text_config.hidden_size
    results = []
    with torch.no_grad(), env['fp32_reference']():
        for arm in ARMS:
            memory = RecentDetail(arm)
            for i in range(20):
                memory.append(torch.randn(108,width),i,2.*i,2.*i+1)
            before = memory.fingerprint()
            r = ask(env,model,builder,memory,'What is visible now? Answer briefly.',limit=1)
            assert before == memory.fingerprint() and r['branch_released']
            assert r['visual_tokens'] == memory.status()['read_visual_tokens']
            memory.append(torch.randn(108,width),20,40.,41.)
            memory.close()
            results.append(dict(arm=arm,visual_tokens=r['visual_tokens'],input_tokens=r['input_tokens'],
                                state_unchanged=True,continued_after_query=True,branch_released=True))
    result = dict(passed=True,micro_QA=results,pretrained_weight_loads=0,GPU=0,
                  capability_evidence=False,description='Random tiny model plus real tokenizer and hybrid QA interface')
    target = ROOT/'runs/recent_detail_prepare_20261103_01/CPU_QA_local.json'
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
