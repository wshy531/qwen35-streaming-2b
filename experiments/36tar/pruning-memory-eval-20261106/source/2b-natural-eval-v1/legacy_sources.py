"""Owned 2B-width diagnostic copy; old diagnostic algorithms remain unchanged."""
from types import SimpleNamespace
import features,forest
def load():return SimpleNamespace(features=features,forest=forest)
