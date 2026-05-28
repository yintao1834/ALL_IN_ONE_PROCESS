from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import (
    DISTANCE_COLUMN,
    guess_value_column,
    prompt_path,
    read_csv_flexible,
)


def _read_strain_series(path: Path, value_column: str | None = None) -> pd.Series:
    df, encoding = read_csv_flexible(path)
    column = guess_value_column(df.columns, value_column)
    if DISTANCE_COLUMN not in df.columns:
        raise ValueError(f"Column not found in {path}: {DISTANCE_COLUMN}")

    df = df[[DISTANCE_COLUMN, column]].copy()
    df[DISTANCE_COLUMN] = pd.to_numeric(df[DISTANCE_COLUMN], errors="coerce")
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[DISTANCE_COLUMN, column])

    series = df.set_index(DISTANCE_COLUMN)[column]
    series.name = path.stem
    print(f"Read {path} with encoding {encoding}, rows: {len(series)}")
    return series


def accumulate_files(
    input_paths: list[Path],
    output_path: Path | None = None,
    value_column: str | None = None,
) -> Path:
    if not input_paths:
        raise ValueError("At least one input file is required.")

    series_list = []
    for idx, path in enumerate(input_paths, start=1):
        series = _read_strain_series(path, value_column)
        series.name = f"{idx}_{path.stem}"
        series_list.append(series)
    aligned = pd.concat(series_list, axis=1).sort_index()

    result_df = pd.DataFrame(index=aligned.index)
    cumulative = pd.Series(0.0, index=aligned.index)
    for column in aligned.columns:
        cumulative = cumulative.add(aligned[column].fillna(0), fill_value=0)
        result_df[f"cumulative_{column}"] = cumulative

    result_df["strain"] = cumulative
    result_df.index.name = DISTANCE_COLUMN
    result_df = result_df.reset_index()

    if output_path is None:
        output_path = input_paths[0].parent / "accumulate_result.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    return output_path


def _prompt_input_paths() -> list[Path]:
    print("Enter files in the order they should be accumulated.")
    print("Use semicolons to enter multiple files on one line, or press Enter after each file.")
    paths: list[Path] = []
    while True:
        value = input("Input CSV path [blank to finish]: ").strip()
        if not value:
            if paths:
                return paths
            print("Please enter at least one path.")
            continue

        for item in value.split(";"):
            path = Path(item.strip()).expanduser().resolve()
            if not path.exists():
                print(f"Path does not exist: {path}")
                continue
            paths.append(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accumulate one or more strain CSV files.")
    parser.add_argument("inputs", nargs="*", type=Path, help="Input CSV paths in accumulation order.")
    parser.add_argument("--output", type=Path, help="Output CSV path.")
    parser.add_argument("--value-column", help="Signal column name. Defaults to strain or Time 1.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_paths = args.inputs or _prompt_input_paths()
    output_path = args.output
    if output_path is None:
        default_output = input_paths[0].parent / "accumulate_result.csv"
        output_path = prompt_path("Output CSV", default_output, must_exist=False)

    accumulate_files(input_paths, output_path, args.value_column)


if __name__ == "__main__":
    main()
