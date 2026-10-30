"""Small append-only experiment event log with atomic status and file hashes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


class ExperimentLog:
    """Own a new output directory. Existing results are never overwritten."""
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / "config.json", config)
        write_json(self.output / "environment.json", dict(
            python=sys.version, platform=platform.platform(), executable=sys.executable))

    def event(self, event, **fields):
        row = dict(time_utc=datetime.now(timezone.utc).isoformat(), event=event, **fields)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        write_json(self.output / "status.json", row)

    def __enter__(self):
        self.event("started")
        return self

    def __exit__(self, kind, error, tb):
        if error is not None:
            (self.output / "failure.txt").write_text(
                "".join(traceback.format_exception(kind, error, tb)), encoding="utf-8")
            self.event("failed", error=str(error))
        else:
            self.event("completed")
        return False
