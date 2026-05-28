from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_MASK_RANGES = [
    (15.50, 19.03),
    (19.57, 23.07),
    (23.67, 27.21),
    (27.75, 31.28),
]

DISTANCE_COLUMN = "Distance (m)"
DEFAULT_ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis")


def default_output_path(input_path: Path, suffix: str, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def ensure_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def parse_range_list(text: str | None, default: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    if text is None or not text.strip():
        if default is None:
            return []
        return list(default)

    ranges = []
    chunks = [chunk.strip() for chunk in re.split(r"[;|\n]+", text) if chunk.strip()]
    for chunk in chunks:
        if "," in chunk:
            parts = [part.strip() for part in chunk.split(",") if part.strip()]
        else:
            parts = [part.strip() for part in re.split(r"\s*-\s*|\s+", chunk) if part.strip()]

        if len(parts) != 2:
            raise ValueError(f"Bad range: {chunk!r}. Use start-end;start-end")

        start, end = float(parts[0]), float(parts[1])
        if start > end:
            start, end = end, start
        ranges.append((start, end))

    return ranges


def parse_float_pair(text: str | None) -> tuple[float, float] | None:
    if text is None or not text.strip():
        return None

    parts = [part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()]
    if len(parts) != 2:
        raise ValueError("Use two numbers, for example: -500 500")

    low, high = float(parts[0]), float(parts[1])
    if low > high:
        low, high = high, low
    return low, high


def parse_columns(text: str | None) -> list[str] | None:
    if text is None or not text.strip():
        return None
    return [col.strip() for col in text.split(",") if col.strip()]


def read_csv_flexible(path: Path, usecols: Iterable[str] | None = None) -> tuple[pd.DataFrame, str]:
    errors = []
    for encoding in DEFAULT_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, usecols=usecols), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except ValueError:
            raise

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Could not decode {path} with {', '.join(DEFAULT_ENCODINGS)}. {errors}",
    )


def prompt_path(prompt: str, default: Path | None = None, must_exist: bool = True) -> Path:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{prompt}{suffix}: ").strip()
        path = ensure_path(value) if value else default

        if path is None:
            print("Please enter a path.")
            continue

        if must_exist and not path.exists():
            print(f"Path does not exist: {path}")
            continue

        return path


def prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            return default
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def prompt_int(prompt: str, default: int | None = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            return default
        try:
            result = int(value)
            if result <= 0:
                raise ValueError
            return result
        except ValueError:
            print("Please enter a positive integer.")


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.")


def prompt_ranges(prompt: str, default: list[tuple[float, float]]) -> list[tuple[float, float]]:
    print("Range format: start-end;start-end")
    print("Default ranges:", "; ".join(f"{start}-{end}" for start, end in default))
    while True:
        value = input(f"{prompt} [default]: ").strip()
        try:
            return parse_range_list(value, default)
        except ValueError as exc:
            print(exc)


def guess_value_column(columns: Iterable[str], preferred: str | None = None) -> str:
    columns = list(columns)
    if preferred:
        if preferred not in columns:
            raise ValueError(f"Column not found: {preferred}")
        return preferred

    for candidate in ("strain", "Time 1", "ch1"):
        if candidate in columns:
            return candidate

    numeric_candidates = [col for col in columns if col != DISTANCE_COLUMN]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        "Could not guess strain column. Use --value-column. "
        f"Available columns: {', '.join(columns)}"
    )


def numeric_signal_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = []
    for col in df.columns:
        if col in {DISTANCE_COLUMN, "s"} or str(col).startswith("Unnamed"):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric_cols.append(col)
    return numeric_cols
