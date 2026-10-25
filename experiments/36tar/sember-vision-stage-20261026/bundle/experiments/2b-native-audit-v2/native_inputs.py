"""Causal normal-video or exact-pair inputs, using unmodified installed processors."""
import hashlib
import math
import sys
import time
from audit_common import ROOT, MODEL, paired_inputs, template, tensor_sha


def matched(path, question, expected):
    # Only the pure input decoder is reused, not any model or streaming execution module.
    sys.path.insert(0, str(ROOT/'experiments/2b-sember-v1'))
    from sember_source import CausalSource, PairDriver
    from transformers import AutoProcessor
    source = CausalSource(path)
    packets = []
    driver = PairDriver(source, lambda a,b,**kwargs:packets.extend((a,b)))
    try:
        state = driver.advance(question['question_time'])
    finally:
        driver.close()
    for key in ('real_frames','pairs','pending_real_frames','last_committed_pts','submitted_prefix_sha256','true_EOF_tail_padded'):
        assert state[key] == expected[key], (question['question_id'],key,state[key],expected[key])
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    sys.path.insert(0, str(ROOT/'experiments/2b-sember-v1'))
    from sember_protocol import prompt
    inputs = paired_inputs(processor,[p.rgb for p in packets],[p.timestamp for p in packets],prompt(question))
    return inputs, dict(protocol='matched_current_pairs',prefix=state,
        input_ids_sha256=tensor_sha(inputs.input_ids), pixels_sha256=tensor_sha(inputs.pixel_values_videos),
        grid=inputs.video_grid_thw.tolist(), input_tokens=inputs.input_ids.shape[1],
        visual_tokens=int((inputs.input_ids==processor.video_token_id).sum()), frames=len(packets))


def normal_video(path, question):
    import av
    import numpy as np
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata
    sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
    from sember_protocol import prompt
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count=4
        stream.codec_context.thread_type='AUTO'
        fps = float(stream.average_rate)
        total = int(stream.frames)
        assert fps > 0 and total > 0, 'explicit source frame count/fps required; no guessed defaults'
        # S-EMBER's frozen sources are CFR12. Actual emitted PTS are checked below.
        assert abs(fps-12.) < 1e-6
        prefix_frames = min(total,math.floor((question['question_time']+1e-9)*fps)+1)
        metadata = VideoMetadata(total_num_frames=prefix_frames,fps=fps)
        indices = processor.video_processor.sample_frames(metadata)
        assert len(indices) <= 768 and len(indices) >= 2 and max(indices) < prefix_frames
        selected = set(int(i) for i in indices)
        raw = {}
        origin = None
        records = []
        for i,frame in enumerate(container.decode(stream)):
            assert frame.time is not None
            if origin is None:origin=float(frame.time)
            pts=float(frame.time)-origin
            if i in selected:
                assert pts <= question['question_time']+1e-9
                assert abs(pts-i/fps) < 1e-6, 'non-CFR source requires an explicit PTS index; no guessing'
                rgb=frame.to_ndarray(format='rgb24')
                raw[i]=rgb
                records.append(dict(frame_id=i,pts_s=pts,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
            if i >= max(selected):break
        assert set(raw) == selected
        frames=np.stack([raw[int(i)] for i in indices]);del raw
        assert frames.nbytes <= 3*2**30
    metadata.frames_indices=[int(i) for i in indices]
    # The official sampler already selected the frames. Avoid sampling them a second time.
    inputs=processor(text=[template(processor,prompt(question))],videos=[frames],
        video_metadata=[metadata],do_sample_frames=False,return_tensors='pt',return_mm_token_type_ids=True)
    return inputs,dict(protocol='official_saved_default_video',fps=fps,prefix_source_frames=prefix_frames,
        selected_frame_indices=metadata.frames_indices,selected_frames=len(indices),frames_sha256_records=records,
        input_ids_sha256=tensor_sha(inputs.input_ids),pixels_sha256=tensor_sha(inputs.pixel_values_videos),
        grid=inputs.video_grid_thw.tolist(),input_tokens=inputs.input_ids.shape[1],
        visual_tokens=int((inputs.input_ids==processor.video_token_id).sum()),
        official_sampler_fps=processor.video_processor.fps,official_sampler_max_frames=processor.video_processor.max_frames,
        official_size=dict(shortest_edge=processor.video_processor.size.shortest_edge,
            longest_edge=processor.video_processor.size.longest_edge),no_future_frames=True)
