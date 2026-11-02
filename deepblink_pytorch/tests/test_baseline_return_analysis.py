"""Returned-result failures that must not pass acceptance silently."""
import io
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_baseline_return import grid_ap, validate_budget, verify_archive
from research_v4.common import dump, write_csv, sha256


def test_ap31_area_handles_ties_and_unattained_recall():
    points = [dict(recall=0, precision=0), dict(recall=.25, precision=.5),
              dict(recall=.25, precision=.8), dict(recall=.5, precision=.6)]
    assert grid_ap(points) == pytest.approx(.25 * .8 + .25 * .6)
    assert grid_ap(points[::-1]) == pytest.approx(.35)
    with pytest.raises(ValueError):
        grid_ap([dict(recall=float("nan"), precision=.5)])


def budget_fixture(folder):
    history = [dict(epoch=e, optimizer_updates=22 * e, train_loss=1., val_loss=1., seconds=1.) for e in range(1, 101)]
    sweep = [dict(epoch=e, threshold_index=i, f1=.5 + e / 1000, precision=.6,
                  tp=e, fp=10, fn=372-e) for e in range(5, 101, 5) for i in range(31)]
    snapshots = [dict(epoch=e, native_val_loss=1/e) for e in range(5, 101, 5)]
    selection = {role: sweep[-1] for role in ("primary", "native_secondary")}
    write_csv(folder / "training_metrics.csv", history)
    write_csv(folder / "all_epoch_threshold_metrics.csv", sweep)
    dump(folder / "snapshots.json", snapshots)
    dump(folder / "selection.json", selection)
    return history, sweep, selection


def test_duplicate_threshold_pair_rejected_even_with_620_rows(tmp_path):
    _, sweep, _ = budget_fixture(tmp_path)
    assert validate_budget(tmp_path)["updates"] == 2200
    sweep[0] = sweep[1]
    write_csv(tmp_path / "all_epoch_threshold_metrics.csv", sweep)
    with pytest.raises(ValueError, match="budget differs"):
        validate_budget(tmp_path)


def test_incomplete_updates_and_wrong_selected_checkpoint_rejected(tmp_path):
    history, _, selection = budget_fixture(tmp_path)
    history[-1]["optimizer_updates"] = 2178
    write_csv(tmp_path / "training_metrics.csv", history)
    with pytest.raises(ValueError, match="update sequence"):
        validate_budget(tmp_path)
    budget_fixture(tmp_path)
    selection["primary"] = dict(selection["primary"], epoch=95)
    dump(tmp_path / "selection.json", selection)
    with pytest.raises(ValueError, match="tie-break"):
        validate_budget(tmp_path)


def test_archive_bytes_and_extracted_bytes_are_both_required(tmp_path):
    archive = tmp_path / "fixture.tar.gz"
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    content = b"synthetic integrity fixture"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("result.txt")
        member.size = len(content)
        stream.addfile(member, io.BytesIO(content))
    (extracted / "result.txt").write_bytes(content)
    # Windows CRLF is deliberately accepted while parsing checksum sidecars.
    archive.with_name(archive.name + ".sha256").write_bytes((sha256(archive) + "  fixture.tar.gz\r\n").encode("ascii"))
    assert len(verify_archive(archive, extracted)) == 1
    (extracted / "result.txt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Extracted bytes differ"):
        verify_archive(archive, extracted)
