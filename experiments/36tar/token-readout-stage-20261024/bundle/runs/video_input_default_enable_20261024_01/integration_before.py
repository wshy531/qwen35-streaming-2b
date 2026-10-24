"""Explicit opt-in adapters for fixed-canonical evaluation inputs.

This factory does not replace defaults or assume that arbitrary live videos
have a known frame manifest. It shares the same frame-packet/replay contract.
"""
from source import BoundedAVSource


def long_source(env,path,group,*,decoder_config):
    if group['cohort']!='stream34':raise ValueError('canonical long replay source required')
    return BoundedAVSource(path,group,env['FramePacket'],config=decoder_config)


def short_packets(env,path,group,*,decoder_config):
    if group['cohort']=='stream34':raise ValueError('short prefix packets required')
    source=BoundedAVSource(path,group,env['FramePacket'],config=decoder_config)
    try:
        yield from source.packets()
    finally:
        source.close()
