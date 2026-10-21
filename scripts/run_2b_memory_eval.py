"""Portable entry point for bounded online memory preparation and evaluation."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from experiments.wshy.online_memory_2b.isolation import configure

configure()

from experiments.wshy.online_memory_2b.runner import main

if __name__=='__main__':
    main()
