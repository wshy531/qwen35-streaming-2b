"""Compare causal input state across an explicitly declared canvas migration.

The old reference's RGB byte budget describes 288x384 uint8 pixels. The 7B
decoder emits 280x392 uint8 pixels. All temporal/input counters stay exact.
"""
OLD_CANVAS=(288,384,3)
NEW_CANVAS=(280,392,3)
FIELD='maximum_pending_RGB_bytes'

def compare_state(actual,reference):
    assert set(actual)==set(reference),('state schema mismatch',set(actual)^set(reference))
    old_bytes=OLD_CANVAS[0]*OLD_CANVAS[1]*OLD_CANVAS[2]
    new_bytes=NEW_CANVAS[0]*NEW_CANVAS[1]*NEW_CANVAS[2]
    assert reference[FIELD] in (old_bytes,new_bytes),('unknown reference canvas',reference[FIELD])
    assert actual[FIELD]==new_bytes,('runtime canvas mismatch',actual[FIELD])
    diffs={k:dict(expected=reference[k],actual=actual[k]) for k in actual if actual[k]!=reference[k]}
    forbidden={k:v for k,v in diffs.items() if k!=FIELD}
    assert not forbidden,('causal state mismatch',forbidden)
    return dict(passed=True,reference_canvas_bytes=reference[FIELD],actual_canvas_bytes=actual[FIELD],
        exact_temporal_fields=[k for k in actual if k!=FIELD],differences=diffs)
