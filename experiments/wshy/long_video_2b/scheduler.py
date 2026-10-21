"""Pause before pulling future samples; retain at most one unmatched RGB frame."""
from bisect import bisect_right
import hashlib
import json

from .protocol import finite_number, validate_queries


def validate_frames(frames):
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError('locked nonempty frame specifications required')
    result, last_id, last_time = [], -1, -1.
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {'frame_id', 'pts_s', 'rgb_sha256'}:
            raise ValueError('frame specification field allowlist violated')
        index, pts, digest = frame['frame_id'], frame['pts_s'], frame['rgb_sha256']
        if (type(index) is not int or index <= last_id or not finite_number(pts) or pts < 0 or pts <= last_time
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('chronological frame identity and SHA256 required')
        result.append(dict(frame))
        last_id, last_time = index, pts
    return tuple(result)


def frame_bytes(spec):
    return (json.dumps(spec, sort_keys=True, separators=(',', ':'))+'\n').encode()


def replay_source(frames, queries, samples, on_pair, on_query):
    """Callbacks must finish synchronously. No query padding, including at EOF.

    Metadata is preflight information; RGB samples are pulled only when their
    locked PTS is at or before the next query. The sampler must not prefetch.
    Feature encoding/storage belongs to on_pair, called once per complete pair.
    """
    specs = validate_frames(frames)
    groups = validate_queries(queries)
    if len(groups) != 1:
        raise ValueError('one source per continuous replay required')
    ordered = next(iter(groups.values()))
    times = [f['pts_s'] for f in specs]
    maximum = bisect_right(times, ordered[-1]['query_time_s'])
    if maximum < 2 or any(bisect_right(times, q['query_time_s']) < 2 for q in ordered):
        raise ValueError('at least one complete arrived pair per query required')
    iterator = iter(samples)
    received = committed = pairs = 0
    pending = None
    arrived_hash, committed_hash = hashlib.sha256(), hashlib.sha256()
    historical_hash = hashlib.sha256(b'[')
    try:
        for query in ordered:
            target = bisect_right(times, query['query_time_s'])
            while received < target:
                try:
                    packet = next(iterator)
                except StopIteration as error:
                    raise ValueError('EOF before locked query prefix') from error
                spec = specs[received]
                if (type(packet.frame_id) is not int or packet.frame_id != spec['frame_id'] or not finite_number(packet.timestamp)
                        or abs(packet.timestamp-spec['pts_s']) > 1e-7
                        or packet.timestamp > query['query_time_s']):
                    raise ValueError('sample identity drift or future frame')
                arrived_hash.update(frame_bytes(spec))
                received += 1
                if pending is None:
                    pending = packet
                else:
                    on_pair(pairs, pending, packet)
                    for offset, entry in enumerate(specs[received-2:received], start=committed):
                        committed_hash.update(frame_bytes(entry))
                        if offset:
                            historical_hash.update(b', ')
                        historical_hash.update(json.dumps(entry, sort_keys=True).encode())
                    pairs += 1
                    committed += 2
                    pending = None
                del packet
            # This mapping contains scalar metadata only, never packets/features.
            historical = historical_hash.copy()
            historical.update(b']')
            state = dict(arrived_frames=received, committed_frames=committed,
                         committed_pairs=pairs, pending_frame=dict(specs[received-1]) if pending is not None else None,
                         arrived_frames_sha256=arrived_hash.hexdigest(),
                         committed_frames_sha256=committed_hash.hexdigest(),
                         historical_committed_frames_sha256=historical.hexdigest(),
                         committed_through_s=specs[committed-1]['pts_s'],
                         observation_age_s=query['query_time_s']-specs[committed-1]['pts_s'], query_padding=False)
            on_query(query, state)
        return dict(queries=len(ordered), sampled_frames=received, encoded_pairs=pairs,
                    unmatched_final_frame=int(pending is not None), query_padding=False)
    finally:
        pending = None
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()


def decode_samples(env, path, frames):
    """Sequential PyAV decoder, RGB/PTS verification, no history or lookahead."""
    specs = validate_frames(frames)
    position, origin, previous = 0, None, None
    with env['av'].open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None or not finite_number(float(frame.time)):
                raise ValueError('finite real PTS required')
            real = float(frame.time)
            if origin is None:
                origin = real
            pts = real-origin
            if previous is not None and pts <= previous:
                raise ValueError('decoder PTS must increase')
            previous = pts
            spec = specs[position]
            if index < spec['frame_id']:
                continue
            if index > spec['frame_id'] or abs(pts-spec['pts_s']) > 1e-7:
                raise ValueError('locked raw frame or PTS identity drift')
            rgb = env['canvas_bytes'](frame)
            if hashlib.sha256(rgb).hexdigest() != spec['rgb_sha256']:
                raise ValueError('locked RGB identity drift')
            packet = env['FramePacket'](index, pts, env['np'].frombuffer(
                rgb, dtype='uint8').reshape(192, 256, 3).copy(), index+1)
            # Release the decoder frame and byte buffer before suspending.
            del frame, rgb
            position += 1
            yield packet
            del packet
            if position == len(specs):
                return
    raise ValueError('EOF before all locked frame specifications')
