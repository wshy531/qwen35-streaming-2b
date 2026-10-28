"""CPU-only official API compatibility check; failure is evidence, never patched."""
import argparse
import os
import time
import traceback
from audit_common import ROOT, OUT, MODEL, save, official_identity, tiny_config, template, paired_inputs, tensor_sha


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.video_utils import VideoMetadata
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    question = 'What is visible?'
    frames = [np.full((64,96,3), 30+i*35, dtype=np.uint8) for i in range(4)]
    torch.manual_seed(20261025)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(processor), attn_implementation='sdpa')).float().eval()
    identity = official_identity(model)
    cases = {}
    for count in (2,4):
        metadata = VideoMetadata(total_num_frames=count, fps=12., frames_indices=[i*24 for i in range(count)])
        inputs = processor(text=[template(processor, question)], videos=[np.stack(frames[:count])],
            video_metadata=[metadata], do_sample_frames=False, do_resize=False,
            return_tensors='pt', return_mm_token_type_ids=True)
        cases['official_video_' + str(count)] = inputs
        cases['matched_pairs_' + str(count)] = paired_inputs(processor, frames[:count], [i*2. for i in range(count)], question)
    results = []
    with torch.inference_mode():
        for name, inputs in cases.items():
            began = time.monotonic()
            row = dict(case=name, input_tokens=inputs.input_ids.shape[1], grid=inputs.video_grid_thw.tolist(),
                input_ids_sha256=tensor_sha(inputs.input_ids), pixels_sha256=tensor_sha(inputs.pixel_values_videos))
            model.model.rope_deltas = None
            try:
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=2,
                    use_cache=True, return_dict_in_generate=True, output_scores=True,
                    pad_token_id=processor.tokenizer.eos_token_id)
                row.update(status='passed', generated_ids=outputs.sequences[0,inputs.input_ids.shape[1]:].tolist(),
                    first_score_sha256=tensor_sha(outputs.scores[0]), finite=bool(torch.isfinite(outputs.scores[0]).all()))
            except Exception as exc:
                row.update(status='failed_unmodified', error_type=type(exc).__name__, error=str(exc),
                    traceback=traceback.format_exc()[-3500:])
            row['seconds'] = time.monotonic()-began
            results.append(row)
    identity_after = official_identity(model)
    assert identity_after == identity
    result = dict(status='compatibility_recorded', **identity, results=results,
        pretrained_loads=0, GPU_jobs=0, model_or_processor_patches=0, synthetic_frames=True,
        not_capability_evidence=True)
    save(output, result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
