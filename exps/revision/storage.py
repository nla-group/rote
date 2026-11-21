"""Incremental and resumable per-stage records for revision experiments."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("rb+") as stream:
        while True:
            line_start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                if not line.endswith(b"\n"):
                    stream.write(b"\n")
                continue
            try:
                records.append(json.loads(line.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not line.endswith(b"\n") and stream.read(1) == b"":
                    stream.truncate(line_start)
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return records


class StageStore:
    def __init__(self, path, config, total, seeds):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        config_path = self.path / "config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as stream:
                if json.load(stream) != config:
                    raise ValueError(f"Refusing to resume {path} with a changed configuration")
        else:
            atomic_json(config_path, config)
        seed_path = self.path / "seeds.csv"
        if seed_path.exists():
            with seed_path.open(encoding="utf-8", newline="") as stream:
                existing = list(csv.DictReader(stream))
            if [{key: "" if value is None else str(value)
                 for key, value in row.items()} for row in seeds] != existing:
                raise ValueError(f"Refusing to resume {path} with a changed seed manifest")
        else:
            atomic_csv(seed_path, seeds)
        self.records = read_jsonl(self.path / "records.jsonl")
        self.completed = {row["case_id"] for row in self.records}
        if len(self.records) != len(self.completed) or len(self.records) > total:
            raise ValueError("Duplicate or excess case IDs in saved results")
        results_path = self.path / "results.csv"
        if self.records:
            # A crash between JSONL append and CSV replacement must not leave
            # a complete manifest pointing to an incomplete public CSV.
            atomic_csv(results_path, self.records)
        elif results_path.exists():
            raise ValueError("Results CSV exists without authoritative JSONL records")
        self.total = total
        self._manifest("running")
        append_jsonl(self.path / "events.jsonl", {"time": _now(), "event": "stage_started",
                                                   "completed": len(self.completed)})

    def _manifest(self, status, error=None):
        data = {"status": status, "completed": len(self.completed), "total": self.total,
                "updated_at": _now()}
        if error is not None:
            data["error"] = f"{type(error).__name__}: {error}"
        atomic_json(self.path / "manifest.json", data)

    def append(self, record):
        case_id = record["case_id"]
        if case_id in self.completed:
            raise ValueError(f"Duplicate case ID {case_id}")
        append_jsonl(self.path / "records.jsonl", record)
        self.records.append(record)
        self.completed.add(case_id)
        atomic_csv(self.path / "results.csv", self.records)
        self._manifest("running")
        append_jsonl(self.path / "events.jsonl", {"time": _now(), "event": "case_completed",
                                                   "case_id": case_id})

    def finish(self):
        if len(self.completed) != self.total:
            raise RuntimeError(f"Stage incomplete: {len(self.completed)}/{self.total}")
        self._manifest("complete")
        append_jsonl(self.path / "events.jsonl", {"time": _now(), "event": "stage_completed"})

    def fail(self, error):
        self._manifest("failed", error)
        append_jsonl(self.path / "events.jsonl", {"time": _now(), "event": "stage_failed",
                                                   "error": str(error)})
