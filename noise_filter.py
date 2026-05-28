from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import (
    DEFAULT_MASK_RANGES,
    DISTANCE_COLUMN,
    default_output_path,
    guess_value_column,
    parse_range_list,
    prompt_float,
    prompt_path,
    prompt_ranges,
    read_csv_flexible,
)


def _range_mask(distance: pd.Series, ranges: list[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=distance.index)
    for start, end in ranges:
        mask |= distance.between(start, end)
    return mask


def replace_threshold_spikes(series: pd.Series, threshold: float, active_mask: pd.Series) -> tuple[pd.Series, int]:
    cleaned = series.copy()
    spike_mask = active_mask & (cleaned.abs() > threshold)

    neighbor_values = pd.concat(
        [cleaned.shift(1), cleaned.shift(-1)],
        axis=1,
    ).median(axis=1, skipna=True)

    cleaned.loc[spike_mask] = neighbor_values.loc[spike_mask].fillna(0)
    return cleaned, int(spike_mask.sum())


def filter_noise_dataframe(
    df: pd.DataFrame,
    threshold: float,
    mask_ranges: list[tuple[float, float]] | None = None,
    value_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    filter_ranges = mask_ranges or DEFAULT_MASK_RANGES
    value_column = guess_value_column(df.columns, value_column)
    if DISTANCE_COLUMN not in df.columns:
        raise ValueError(f"Column not found: {DISTANCE_COLUMN}")

    result_df = df[[DISTANCE_COLUMN, value_column]].copy()
    result_df[DISTANCE_COLUMN] = pd.to_numeric(result_df[DISTANCE_COLUMN], errors="coerce")
    result_df[value_column] = pd.to_numeric(result_df[value_column], errors="coerce")
    result_df = result_df.dropna(subset=[DISTANCE_COLUMN, value_column]).reset_index(drop=True)

    filter_mask = _range_mask(result_df[DISTANCE_COLUMN], filter_ranges)
    result_df[value_column], spike_count = replace_threshold_spikes(
        result_df[value_column],
        threshold,
        filter_mask,
    )

    result_df = result_df.rename(columns={value_column: "strain"})
    stats = {
        "selected_rows": int(filter_mask.sum()),
        "unchanged_rows": int((~filter_mask).sum()),
        "spikes_replaced": spike_count,
    }
    return result_df, stats


def filter_noise(
    input_path: Path,
    threshold: float,
    mask_ranges: list[tuple[float, float]] | None = None,
    output_path: Path | None = None,
    value_column: str | None = None,
) -> Path:
    df, encoding = read_csv_flexible(input_path)
    result_df, stats = filter_noise_dataframe(df, threshold, mask_ranges, value_column)

    if output_path is None:
        output_path = default_output_path(input_path, "_processed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"Read {input_path} with encoding {encoding}")
    print(f"Filter range selected {stats['selected_rows']} rows; outside rows unchanged: {stats['unchanged_rows']}.")
    print(f"Threshold replacement count: {stats['spikes_replaced']}")
    print(f"Saved: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter one strain CSV file by threshold inside selected distance ranges.")
    parser.add_argument("input", nargs="?", type=Path, help="Input CSV path.")
    parser.add_argument("--threshold", type=float, help="Absolute threshold for spike replacement.")
    parser.add_argument("--ranges", help="Filter ranges, for example 15.5-19.03;19.57-23.07")
    parser.add_argument("--output", type=Path, help="Output CSV path.")
    parser.add_argument("--value-column", help="Signal column name. Defaults to strain or Time 1.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input or prompt_path("Input CSV")
    threshold = args.threshold
    if threshold is None:
        threshold = prompt_float("Threshold")

    if args.ranges is None:
        ranges = prompt_ranges("Distance filter ranges", DEFAULT_MASK_RANGES)
    else:
        ranges = parse_range_list(args.ranges, DEFAULT_MASK_RANGES)

    filter_noise(
        input_path=input_path,
        threshold=threshold,
        mask_ranges=ranges,
        output_path=args.output,
        value_column=args.value_column,
    )


if __name__ == "__main__":
    main()
