"""Yield at most one arrived pair; never buffer decoded video history."""
import hashlib
import time


def decode_pairs(env, path, frames, query_time):
    """Yield (packets, specs, decode seconds) before decoding the next pair."""
    pending, position, origin, previous = [], 0, None, None
    started = time.monotonic()
    with env['av'].open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                raise ValueError('frame without real PTS')
            real = float(frame.time)
            if origin is None:
                origin = real
            pts = real - origin
            if previous is not None and pts <= previous:
                raise ValueError('decoder PTS is not increasing')
            previous = pts
            spec = frames[position]
            if index != spec['frame_id']:
                if index > spec['frame_id']:
                    raise ValueError('requested frame was skipped')
                continue
            if pts > query_time + 1e-9 or abs(pts - spec['pts_s']) > 1e-7:
                raise ValueError('future frame or PTS identity drift')
            rgb = env['canvas_bytes'](frame)
            if hashlib.sha256(rgb).hexdigest() != spec['rgb_sha256']:
                raise ValueError('sampled RGB identity drift')
            packet = env['FramePacket'](index, pts, env['np'].frombuffer(
                rgb, dtype='uint8').reshape(192, 256, 3).copy(), index + 1)
            pending.append((packet, spec))
            position += 1
            final = position == len(frames)
            if len(pending) == 2 or final:
                padded = len(pending) == 1
                if padded:
                    pending.append(pending[0])
                elapsed = time.monotonic() - started
                yield pending, padded, elapsed
                pending = []
                # Time spent encoding/asking while suspended is not decode time.
                started = time.monotonic()
            if final:
                return  # do not request even one extra frame from the decoder
    raise ValueError('end of file before all registered frames arrived')
