"""V0 auto4 packets; prefix sources stop at their explicit last frame."""
from io_source import BoundedAVSource
class ExactPackets(BoundedAVSource):
    def __init__(self,env,path,frames,digest):
        super().__init__(path,dict(cohort='prefix',frames=frames,sha256=digest),env['FramePacket'],config='auto4')
