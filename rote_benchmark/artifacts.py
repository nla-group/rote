"""Durable, inspectable artifacts for custom ROTE benchmark runs."""

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value):
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(value), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv(path: Path, records: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            pd.DataFrame(records).to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_run_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        base = Path(os.environ.get("ROTE_RUNS_DIR", Path.cwd() / "rote_runs")).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        for _ in range(10):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = base / f"run-{stamp}-{uuid.uuid4().hex[:8]}"
            try:
                path.mkdir()
                return path.resolve()
            except FileExistsError:
                continue
        raise FileExistsError("Could not allocate a unique ROTE run directory")
    path = Path(output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


@dataclass(frozen=True)
class SavedRun:
    """A saved run loaded from disk, with its configuration and result table."""

    path: Path
    config: dict
    results: pd.DataFrame
    manifest: dict

    @property
    def results_csv(self) -> Path:
        return self.path / "results.csv"

    @property
    def records_jsonl(self) -> Path:
        return self.path / "records.jsonl"

    @property
    def events_jsonl(self) -> Path:
        return self.path / "events.jsonl"


class _RunWriter:
    def __init__(self, config: dict, total: int, output_dir: str | Path | None):
        if total < 0:
            raise ValueError("total must be nonnegative")
        safe_config = _json_safe(config)
        self.path = _new_run_dir(output_dir)
        self.records: list[dict] = []
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "completed": 0,
            "total": total,
            "files": ["config.json", "results.csv", "records.jsonl", "events.jsonl"],
        }
        _write_json(self.path / "config.json", safe_config)
        _write_json(self.path / "manifest.json", self.manifest)
        _append_jsonl(self.path / "events.jsonl", {"time": _utc_now(), "event": "run_started"})

    def append(self, record: dict) -> None:
        safe_record = _json_safe(record)
        if not isinstance(safe_record, dict):
            raise TypeError("record must be a mapping")
        _append_jsonl(self.path / "records.jsonl", safe_record)
        self.records.append(safe_record)
        _write_csv(self.path / "results.csv", self.records)
        self.manifest["completed"] = len(self.records)
        self.manifest["updated_at"] = _utc_now()
        _write_json(self.path / "manifest.json", self.manifest)
        _append_jsonl(self.path / "events.jsonl", {
            "time": _utc_now(), "event": "evaluation_completed",
            "model": safe_record.get("model"), "run": safe_record.get("run"),
            "seed_id": safe_record.get("seed_id"),
        })

    def finish(self) -> None:
        self.manifest["status"] = "complete"
        self.manifest["updated_at"] = _utc_now()
        _write_json(self.path / "manifest.json", self.manifest)
        _append_jsonl(self.path / "events.jsonl", {"time": _utc_now(), "event": "run_completed"})

    def fail(self, error: BaseException) -> None:
        self.manifest["status"] = "failed"
        self.manifest["updated_at"] = _utc_now()
        self.manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(self.path / "manifest.json", self.manifest)
        _append_jsonl(self.path / "events.jsonl", {
            "time": _utc_now(), "event": "run_failed", "error": self.manifest["error"],
        })


def load_run(path: str | Path) -> SavedRun:
    """Load a run directory; JSONL records remain authoritative after a crash."""
    path = Path(path).expanduser().resolve()
    with (path / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with (path / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    records_path = path / "records.jsonl"
    if records_path.exists():
        records = []
        with records_path.open(encoding="utf-8") as stream:
            lines = stream.readlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith("\n"):
                    break
                raise
        results = pd.DataFrame(records)
    elif (path / "results.csv").exists():
        results = pd.read_csv(path / "results.csv")
    else:
        results = pd.DataFrame()
    return SavedRun(path=path, config=config, results=results, manifest=manifest)


def save_benchmark(
    results: pd.DataFrame,
    config: dict,
    *,
    output_dir: str | Path | None = None,
) -> SavedRun:
    """Persist an existing result table to a new run directory.

    Use this for results produced with the lower-level evaluation API or for
    imported experiment CSVs. ``config`` is user-defined JSON-serializable
    metadata. An explicit ``output_dir`` must not already exist.
    """
    frame = pd.DataFrame(results)
    if frame.empty:
        raise ValueError("results must contain at least one row")
    if not isinstance(config, dict):
        if is_dataclass(config):
            config = asdict(config)
        else:
            raise TypeError("config must be a dict or dataclass")
    writer = _RunWriter({**config, "schema_version": SCHEMA_VERSION}, len(frame), output_dir)
    try:
        for record in frame.to_dict(orient="records"):
            writer.append(record)
        writer.finish()
    except BaseException as error:
        writer.fail(error)
        raise
    return load_run(writer.path)
