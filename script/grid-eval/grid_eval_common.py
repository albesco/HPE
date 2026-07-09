#!/usr/bin/env python3
"""Shared utilities for local grid-evaluation scripts."""

from __future__ import annotations

import csv
import itertools
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


GridParams = dict[str, Any]
StatusPayload = dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "failed", "error": f"Invalid JSON in {path}: {exc}"}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if text.startswith("0") and text not in {"0"} and not text.startswith("0."):
            raise ValueError
        return int(text)
    except ValueError:
        pass
    try:
        parsed = float(text)
    except ValueError:
        return text
    return parsed if math.isfinite(parsed) else text


def parse_grid_param(value: str) -> tuple[str, list[Any]]:
    if "=" not in value:
        raise ValueError(f"Invalid --grid-param {value!r}; expected name=value1,value2")
    name, raw_values = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid --grid-param {value!r}; empty parameter name")
    values = [parse_scalar(item) for item in raw_values.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Invalid --grid-param {value!r}; empty value list")
    return name, values


def grid_from_cli_params(values: Iterable[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for value in values:
        name, parsed_values = parse_grid_param(value)
        if name in grid:
            raise ValueError(f"Duplicate grid parameter: {name}")
        grid[name] = parsed_values
    return grid


def load_json_or_yaml(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML grid files") from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded or {}
    raise ValueError(f"Unsupported grid file extension: {path}")


def extract_grid_mapping(payload: dict[str, Any]) -> dict[str, list[Any]]:
    grid = payload.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("Grid file must contain a mapping at key 'grid'")
    normalized: dict[str, list[Any]] = {}
    for name, values in grid.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Grid parameter names must be non-empty strings")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid parameter {name!r} must be a non-empty list")
        normalized[name] = values
    return normalized


def grid_from_file(path: Path) -> dict[str, list[Any]]:
    return extract_grid_mapping(load_json_or_yaml(path))


def cartesian_grid(grid: dict[str, list[Any]]) -> list[GridParams]:
    if not grid:
        raise ValueError("Grid is empty")
    names = list(grid.keys())
    configs: list[GridParams] = []
    for values in itertools.product(*(grid[name] for name in names)):
        configs.append(dict(zip(names, values)))
    return configs


def formatted_value(name: str, value: Any) -> str:
    if isinstance(value, float):
        if name in {"lr", "lr0"}:
            return f"{value:.5f}"
        return f"{value:g}"
    return str(value)


def config_name(index: int, params: GridParams) -> str:
    parts = [f"cfg_{index:02d}"]
    for name, value in params.items():
        safe_value = formatted_value(name, value).replace("/", "-").replace(" ", "")
        parts.append(f"{name}_{safe_value}")
    return "_".join(parts)


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def read_status(run_dir: Path) -> StatusPayload:
    return read_json(status_path(run_dir))


def update_status(run_dir: Path, payload: StatusPayload) -> None:
    current = read_status(run_dir)
    current.update(payload)
    current["updated_at"] = utc_now()
    write_json(status_path(run_dir), current)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def run_logged_subprocess(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        log_file.write(f"\n# Started {utc_now()}\n".encode("utf-8"))
        log_file.write((command_to_text(command) + "\n").encode("utf-8"))
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        return_code = process.wait()
        log_file.write(f"\n# Finished {utc_now()} exit_code={return_code}\n".encode("utf-8"))
    return return_code


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def metric_value(row: dict[str, Any], key: str) -> float:
    value = parse_float(row.get(key))
    return value if value is not None else float("-inf")


def choose_best(rows: list[dict[str, Any]], ranking: Callable[[dict[str, Any]], tuple[Any, ...]]) -> dict[str, Any] | None:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return None
    return max(completed, key=ranking)
