"""Real7-frame file input, random8-layer model; isolated streamed QA then resume."""
import os
import torch
from common import ROOT, OUT, ARMS, runtime, read, save
from session import Session
from io_source import BoundedAVSource


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1);torch.manual_seed(942)
    env = runtime()
    builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        ROOT/'models/Qwen3.5-2B', local_files_only=True), canvas=(192, 256), resize_mode='letterbox')
    cfg = env['tiny_config'](builder.processor)
    cfg['text_config'].update(num_hidden_layers=8,
        layer_types=(['linear_attention']*3+['full_attention'])*2,
        num_attention_heads=4, num_key_value_heads=1)
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **cfg, attn_implementation='sdpa')).float().eval()
    group = next(g for g in read(OUT/'inputs/groups.json') if g['cohort'] == 'T98')
    key = 'cluster_media' if (ROOT/'repo').exists() else 'local_media'
    source = BoundedAVSource(ROOT/group[key], dict(group, frames=group['frames'][:7]),
                             env['FramePacket'], config='auto4')
    packets = list(source.packets());assert len(packets) == 7
    with torch.inference_mode(), env['fp32_reference']():
        session = Session(env, model, builder)
        session.append_pair(*packets[:2]);session.append_pair(*packets[2:4])
        digest = session.bank.fingerprint();tokens = []
        prior = session.ask_arm('LS_full48', 'What is visible?', lambda item: None)
        result = session.ask('What is visible?', read_from='readout_banks', on_token=tokens.append)
        after = session.ask_arm('LS_full48', 'What is visible?', lambda item: None)
        assert prior['generated_ids'] == after['generated_ids'] == result['arms']['LS_full48']['generated_ids']
        assert session.bank.fingerprint() == digest
        for arm in ARMS:
            row = result['arms'][arm]
            assert row['generated_ids'] == [t['token_id'] for t in tokens if t['arm'] == arm]
            assert row['banks_unchanged'] and row['branch_released']
            assert row['QA_includes_full_probe_and_rebuild']
        assert result['arms']['LS_probe24']['probe']['completed_decoder_layers'] == 3
        assert len(model.model.language_model.layers[3].self_attn._forward_pre_hooks) == 0
        session.append_pair(*packets[4:6]);session.snapshot_tail(packets[6])
        assert session.vision_encodings == 4
        try:session.append_pair(*packets[:2])
        except RuntimeError:pass
        else:raise AssertionError('query-only EOF tail continued')
        session.close();assert not session.bank.banks and session.bank.descriptor is None
    save(OUT/'tiny_session_check.json', dict(status='passed', answers=5, probes=1,
        real_frames=7, vision_encodings=4, streamed_tokens=len(tokens),
        QA=[dict(arm=a, input_tokens=r['input_tokens'], visual_tokens=r['visual_tokens'],
                 generated_tokens=len(r['generated_ids'])) for a, r in result['arms'].items()],
        full_answer_before_after_probe_exact=True, banks_unchanged=True,
        post_QA_continuation=True, EOF_tail_closes=True, all_hooks_removed=True,
        pretrained_loads=0, GPU_jobs=0, no_capability_or_speed_claim=True))
    print(dict(status='passed', answers=5, probes=1, real_frames=7, vision_encodings=4), flush=True)


if __name__ == '__main__':main()
