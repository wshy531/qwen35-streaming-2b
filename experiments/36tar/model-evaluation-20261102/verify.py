"""Recompute public stage tables from blinded item-level records, without model assets."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("full", "fixed12", "heuristic11", "mask_ce", "text_only")

def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))

def main():
    for entry in load("source_manifest.json"):
        data = (ROOT / entry["snapshot"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        compile(data, entry["snapshot"], "exec")
    summary = load("results/content.json")
    with (ROOT / "results/outcomes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 300 and len({row["item"] for row in rows}) == 300
    for ds, methods in summary["content"].items():
        selected = [row for row in rows if row["dataset"] == ds]
        assert len(selected) == 150
        for arm in ARMS:
            assert sum(int(row[arm]) for row in selected) == methods[arm]["correct"]
        for pair in (pair for pair in summary["paired"] if pair["dataset"] == ds):
            diff = Counter(int(row[pair["a"]]) - int(row[pair["b"]]) for row in selected)
            assert (diff[1], diff[-1], diff[0]) == (pair["a_wins"], pair["b_wins"], pair["same"])
    aux = load("results/judge_records.json")
    assert len(aux) == 115
    matrix = load("results/judge_tables.json")
    for arm in ARMS:
        sub = [x for x in aux if x["arm"] == arm]
        assert len(sub) == len({x["item"] for x in sub}) == 23
        ref = matrix[arm]
        assert Counter(x["support"] for x in sub) == ref["support"]
        assert sum(x["correct"] for x in sub) == ref["official_correct"]
        assert sum(x["core_error"] is True for x in sub) == ref["core_errors"]
    print("Verified 300 five-method questions, 20 paired comparisons, 115 auxiliary judgments and source snapshots.")

if __name__ == "__main__":
    main()
