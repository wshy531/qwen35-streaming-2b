"""Approved input-quality and option-position controls; no frozen edits."""
import copy
import re
from ..sember_102_2b.readout import question_prompt

ARMS = ("B", "R", "F", "T")
CONTEXT_TOKENS = 24000

def rotate(query, shift):
    """New display position j contains original option (j + shift) mod 5."""
    question_prompt(query)  # Reuse validated structural-label rules.
    if len(query['options']) != 5 or type(shift) is not int or not 0 <= shift < 5:
        raise ValueError('this protocol requires five choices and shifts 0..4')
    content = [re.sub(r'^[A-E]\.\s+', '', s, count=1) for s in query['options']]
    order = [(j + shift) % 5 for j in range(5)]
    result = copy.deepcopy(query)
    result['options'] = [f'{chr(65+j)}. {content[i]}' for j, i in enumerate(order)]
    return result, order

def matched_dense(anchors, candidates):
    """Target-2-FPS samples within fixed old support; never lose an old frame.

    The temporal patch size is two. If the union has odd cardinality, omit
    the latest newly added frame, never an anchor or either old endpoint.
    This is boundary-matched densification, not an exactly uniform 2-FPS grid.
    """
    if len(anchors) < 2 or len(anchors) % 2:
        raise ValueError('complete baseline pairs required')
    if any(a['pts_s'] >= b['pts_s'] for a, b in zip(anchors, anchors[1:])):
        raise ValueError('anchors must be chronological')
    ids = {a['frame_id'] for a in anchors}
    if len(ids) != len(anchors):
        raise ValueError('duplicate anchor')
    lo, hi = anchors[0]['pts_s'], anchors[-1]['pts_s']
    merged = {a['frame_id']: dict(a) for a in anchors}
    for row in candidates:
        fid = row['frame_id']
        if fid in merged and merged[fid] != row:
            raise ValueError('conflicting frame identity')
        if not lo <= row['pts_s'] <= hi:
            continue
        merged[fid] = dict(row)
    rows = sorted(merged.values(), key=lambda r: r['pts_s'])
    if any(a['pts_s'] >= b['pts_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('duplicate or nonmonotonic PTS')
    omitted = None
    if len(rows) % 2:
        index = next(i for i in range(len(rows)-1, -1, -1) if rows[i]['frame_id'] not in ids)
        omitted = rows.pop(index)
    assert rows[0] == anchors[0] and rows[-1] == anchors[-1]
    assert ids <= {r['frame_id'] for r in rows} and len(rows) % 2 == 0
    return rows, omitted

def canvas(frame, height, width):
    """Scale the historical two-stage pipeline directly from decoded originals."""
    from PIL import Image
    if (height, width) not in ((192, 256), (288, 384)):
        raise ValueError('unregistered canvas')
    image = frame.to_image()
    image.thumbnail((width, width), Image.Resampling.BICUBIC)
    scale = min(width/image.width, height/image.height)
    size = (max(1, min(width, round(image.width*scale))),
            max(1, min(height, round(image.height*scale))))
    image = image.resize(size, Image.Resampling.BICUBIC)
    result = Image.new('RGB', (width, height), (0, 0, 0))
    result.paste(image, ((width-size[0])//2, (height-size[1])//2))
    return result
