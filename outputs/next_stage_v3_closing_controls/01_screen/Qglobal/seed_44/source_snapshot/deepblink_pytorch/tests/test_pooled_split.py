"""The pooled experiment must not reuse origin labels or old assignments."""

import importlib.util
from collections import Counter
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("run_clean220_random", SCRIPTS/"run_clean220_random.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def inventory():
    return [dict(sample_id=f"sample_{i}", source_group=f"source_{i}",
                 cohort="old" if i < 113 else "new", subset="train") for i in range(220)]


def test_split_is_pooled_fixed_and_disjoint():
    rows = experiment.pooled_split(inventory(), 20261024)
    assert Counter(r["subset"] for r in rows) == {"train":176, "valid":22, "test":22}
    assert len({r["sample_id"] for r in rows}) == 220
    changed = inventory()[::-1]
    for row in changed:
        row["cohort"] = "arbitrary"
        row["subset"] = "test"
    assert rows == experiment.pooled_split(changed, 20261024)
    assert all("cohort" not in row for row in rows)
    assert rows != experiment.pooled_split(inventory(), 20261025)


def test_repeated_source_requires_grouped_assignment():
    rows = inventory()
    rows[1]["source_group"] = rows[0]["source_group"]
    with pytest.raises(ValueError, match="source groups"):
        experiment.pooled_split(rows, 20261024)
