"""CPU-only actual processor/token counts; no model instance or GPU access."""
from collections import defaultdict
import hashlib
import os
import sys
import time

from prepare import OUT, ROOT, read, save, sha


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicitly hidden GPU required')
    import psutil
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
    from score_memory_eval import prompt
    started = time.monotonic()
    save(OUT/'capacity_check_started.json',dict(source_sha256=sha(__file__),
        model_loads=0,pretrained_processor_only=True,wall_seconds=600,
        process_RSS_cap=int(1.5*2**30),GPU_jobs=0))
    processor = AutoProcessor.from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    originals = read(OUT/'candidate_questions119.json')
    traces = read(OUT/'sampling_all40_1fps.json')['videos']
    results = []
    peak = 0
    for trace in traces:
        qs = [q for q in originals if q['source_path'] == trace['source_path']]
        pixels = []
        for page in trace['pages']:
            assert sha(ROOT/page['path']) == page['sha256']
            with Image.open(ROOT/page['path']) as image:
                for item in page['frames']:
                    rgb = image.crop(item['crop_xyxy']).tobytes()
                    assert hashlib.sha256(rgb).hexdigest() == item['rgb_sha256']
                    pixels.append(np.frombuffer(rgb,dtype=np.uint8).reshape(192,256,3))
        assert len(pixels) == len(trace['frames'])
        odd = bool(len(pixels)%2)
        if odd:
            pixels.append(pixels[-1])
        video = np.stack(pixels)
        indices = [f['frame_id'] for f in trace['frames']]
        if odd:
            indices.append(indices[-1])
        metadata = dict(fps=trace['average_fps'],total_num_frames=trace['decoded_frames'],
                        frames_indices=indices)
        for q in qs:
            text = processor.apply_chat_template([{'role':'user','content':[
                {'type':'video'},{'type':'text','text':prompt(q)}]}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs = processor(text=[text],videos=[video],video_metadata=[metadata],
                do_resize=False,do_sample_frames=False,return_tensors='pt')
            length = int(inputs.input_ids.shape[1])
            visual = int((inputs.input_ids==processor.video_token_id).sum())
            assert visual == len(video)//2*48
            results.append(dict(id=q['id'],source_path=q['source_path'],
                sampled_real_frames=len(trace['frames']),tail_duplicate=odd,
                input_tokens=length,visual_tokens=visual,max_new_tokens=8,
                inherited3072_guard_pass=length+8<=3072,
                proposed8192_context_guard_pass=length+8<=8192,
                proposed6144_visual_guard_pass=visual<=6144,
                model_forward_performed=False))
            del inputs
            peak = max(peak,psutil.Process().memory_info().rss)
            if peak>1.5*2**30 or time.monotonic()-started>600:
                raise MemoryError('bounded CPU processor check')
        del video,pixels
        print(dict(processor_sources=len({x['source_path'] for x in results}),questions=len(results)),flush=True)
    assert len(results) == 119
    save(OUT/'actual_processor_capacity119.json',dict(
        status='CPU_processor_only_counts_NOT_model_runtime_preflight',questions=results,
        inherited3072_overflow=sum(not r['inherited3072_guard_pass'] for r in results),
        maximum_input_tokens=max(r['input_tokens'] for r in results),
        maximum_visual_tokens=max(r['visual_tokens'] for r in results),
        proposed8192_all_fit=all(r['proposed8192_context_guard_pass'] for r in results),
        proposed6144_visual_all_fit=all(r['proposed6144_visual_guard_pass'] for r in results),
        process_peak_RSS_bytes=peak,seconds=time.monotonic()-started,
        model_instances=0,pretrained_model_loads=0,GPU_jobs=0,
        sampling_sha256=sha(OUT/'sampling_all40_1fps.json'),source_sha256=sha(__file__),
        note='No inherited code changed. Larger common full-input adapter still needs actual context/position preflight and GPU approval.'))
    print(dict(questions=119,max_tokens=max(r['input_tokens'] for r in results),
               old_guard_overflows=sum(not r['inherited3072_guard_pass'] for r in results),peak_RSS=peak),flush=True)


if __name__ == '__main__':
    main()
