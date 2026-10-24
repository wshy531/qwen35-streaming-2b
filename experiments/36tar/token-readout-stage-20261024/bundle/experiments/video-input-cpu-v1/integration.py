"""Default bounded AUTO4 input for fixed-canonical file evaluation.

This factory preserves the manifest protocol; it is not a live-camera sampler.
Use decoder_config='slice1' explicitly for a lower-resource file input.
"""
import common as _input_common  # Initialize canonical source dependency paths first.
from source import BoundedAVSource

DEFAULT_DECODER_CONFIG = 'auto4'


def long_source(env,path,group,*,decoder_config=DEFAULT_DECODER_CONFIG):
    if group['cohort']!='stream34':raise ValueError('canonical long replay source required')
    return BoundedAVSource(path,group,env['FramePacket'],config=decoder_config)


def short_packets(env,path,group,*,decoder_config=DEFAULT_DECODER_CONFIG):
    if group['cohort']=='stream34':raise ValueError('short prefix packets required')
    source=BoundedAVSource(path,group,env['FramePacket'],config=decoder_config)
    try:
        yield from source.packets()
    finally:
        source.close()
