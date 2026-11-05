"""Exact original six-case preprocessing; retained for pixel-identical input, not a new transform.

Copied canvas from 2b-perception-interference-v1/audit_media.py; check_pixels.py
checks function AST identity. This differs from the later pruning dataset's one-step resize.
"""
def canvas(frame):
    from PIL import Image
    image = frame.to_image()
    image.thumbnail((384, 384), Image.Resampling.BICUBIC)
    scale = min(384 / image.width, 288 / image.height)
    size = (max(1, min(384, round(image.width * scale))), max(1, min(288, round(image.height * scale))))
    image = image.resize(size, Image.Resampling.BICUBIC)
    result = Image.new('RGB', (384, 288))
    result.paste(image, ((384 - size[0]) // 2, (288 - size[1]) // 2))
    return result
