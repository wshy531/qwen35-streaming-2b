"""Opt-in long-video preparation and CPU preflight; no automatic cluster action."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.wshy.online_memory_2b.isolation import configure
configure()
from experiments.wshy.long_video_2b.runner import main

if __name__ == '__main__':
    main()
