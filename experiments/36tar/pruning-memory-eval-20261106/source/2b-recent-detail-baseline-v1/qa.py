"""Reuse the established independent hybrid QA session without editing it."""
import sys
from types import SimpleNamespace
from guard import ROOT
sys.path.insert(0, str(ROOT/'experiments/2b-midprune-memory-v1'))
from adapter import ask as existing_ask


class ReadView:
    def __init__(self, memory):
        self.memory = memory
        self.banks = {'guard': SimpleNamespace(budget=memory.status()['total_visual_budget'])}

    def fingerprint(self):
        return self.memory.fingerprint()

    def export(self, policy):
        if policy != 'guard':
            raise ValueError('registered read path required')
        return self.memory.export()


def ask(env, model, builder, memory, prompt, limit=128):
    return existing_ask(env, model, builder, ReadView(memory), 'guard', prompt, limit)
