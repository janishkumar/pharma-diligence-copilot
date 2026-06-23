"""Train/test leakage tests — Section 19.4 (Phase 2)"""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
TRAIN_PATH = ROOT / "data" / "finetune" / "train_v1.jsonl"
TEST_PATH = ROOT / "data" / "finetune" / "test_v1.jsonl"


@pytest.mark.skipif(not TRAIN_PATH.exists() or not TEST_PATH.exists(), reason="Phase 2 data not yet generated")
def test_no_exact_overlap():
    train_qs = set()
    for line in TRAIN_PATH.read_text().splitlines():
        if line.strip():
            train_qs.add(json.loads(line)["question"])
    for line in TEST_PATH.read_text().splitlines():
        if line.strip():
            q = json.loads(line)["question"]
            assert q not in train_qs, f"Test question appears in training set: {q[:80]}"
