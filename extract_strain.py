from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import (
    DEFAULT_MASK_RANGES,
    DISTANCE_COLUMN,
    guess_value_column,
    parse_range_list,
    prompt_float,
    prompt_int,
    prompt_path,
    prompt_ranges,
    read_csv_flexible,
)


def _trim_series_center(series: pd.Series, target_length: int) -> tuple[pd.Series, tuple[int, int]]:
    series = series.reset_index(drop=True)
    current_length = len(series)

    if current_length < target_length:
        raise ValueError("Target length cannot be longer than a channel length.")

    extra = current_length - target_length
    if extra == 0:
        return series, (0, 0)

    trim_front = extra // 2
    trim_back = extra - trim_front
    end = current_length - trim_back if trim_back else current_length
    return series.iloc[trim_front:end].reset_index(drop=True), (trim_front, trim_back)



def extract_strain_dataframe(
    df: pd.DataFrame,
    ranges: list[tuple[float, float]] | None = None,
    pipe_length: float = 2.92,
    target_length: int | None = None,
    value_column: str | None = None,
    reverse_channels: tuple[int, ...] = (2, 4),
) -> tuple[pd.DataFrame, dict[str, object]]:
    ranges = ranges or DEFAULT_MASK_RANGES
    value_column = guess_value_column(df.columns, value_column)

    if DISTANCE_COLUMN not in df.columns:
        raise ValueError(f"Column not found: {DISTANCE_COLUMN}")

    result_df = df[[DISTANCE_COLUMN, value_column]].copy()
    result_df[DISTANCE_COLUMN] = pd.to_numeric(result_df[DISTANCE_COLUMN], errors="coerce")
    result_df[value_column] = pd.to_numeric(result_df[value_column], errors="coerce")
    result_df = result_df.dropna(subset=[DISTANCE_COLUMN, value_column])

    channels: list[pd.Series] = []
    channel_lengths = []
    for channel_number, (start, end) in enumerate(ranges, start=1):
        mask = (result_df[DISTANCE_COLUMN] > start) & (result_df[DISTANCE_COLUMN] < end)
        channel = result_df.loc[mask, value_column].reset_index(drop=True)

        if channel_number in reverse_channels:
            channel = channel.iloc[::-1].reset_index(drop=True)

        channels.append(channel)
        channel_lengths.append(len(channel))

    if not channels:
        raise ValueError("At least one range is required.")

    min_length = min(channel_lengths)
    max_length = max(channel_lengths)
    if target_length is None:
        target_length = min_length

    if target_length <= 0:
        raise ValueError("Target length must be positive.")
    if target_length > min_length:
        raise ValueError("Target length cannot be larger than the shortest selected channel.")

    output_data = {}
    trim_counts = []
    for channel_number, channel in enumerate(channels, start=1):
        adjusted, trim_count = _trim_series_center(channel, target_length)
        trim_counts.append(trim_count)
        output_data[f"ch{channel_number}"] = adjusted.to_numpy()

    s = np.linspace(0, pipe_length, target_length)
    out_df = pd.DataFrame(output_data, index=s)
    out_df.index.name = "s"

    stats = {
        "channel_lengths": channel_lengths,
        "output_length": target_length,
        "trim_counts": trim_counts,
        "trimmed": min_length != max_length or target_length != min_length,
    }
    return out_df, stats


def extract_strain(
    input_path: Path,
    ranges: list[tuple[float, float]] | None = None,
    pipe_length: float = 2.92,
    target_length: int | None = None,
    output_path: Path | None = None,
    value_column: str | None = None,
    reverse_channels: tuple[int, ...] = (2, 4),
) -> Path:
    df, encoding = read_csv_flexible(input_path)
    out_df, stats = extract_strain_dataframe(
        df=df,
        ranges=ranges,
        pipe_length=pipe_length,
        target_length=target_length,
        value_column=value_column,
        reverse_channels=reverse_channels,
    )

    print(f"Read {input_path} with encoding {encoding}")
    display_ranges = ranges or DEFAULT_MASK_RANGES
    for channel_number, (start, end) in enumerate(display_ranges, start=1):
        length = stats["channel_lengths"][channel_number - 1]
        print(f"ch{channel_number} range {start}-{end}, length: {length}")

    if stats["trimmed"]:
        print(f"Center-trimming all channels to {stats['output_length']} points.")
    else:
        print(f"All channels have {stats['output_length']} points.")

    if output_path is None:
        output_dir = input_path.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"output_{input_path.stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path)

    print(f"Saved: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract channel strain by distance ranges.")
    parser.add_argument("input", nargs="?", type=Path, help="Input CSV path.")
    parser.add_argument("--ranges", help="Extraction ranges, for example 15.5-19.03;19.57-23.07")
    parser.add_argument("--pipe-length", type=float, help="Pipe length for output s-axis.")
    parser.add_argument("--target-length", type=int, help="Optional crop length. Defaults to shortest selected range.")
    parser.add_argument("--output", type=Path, help="Output CSV path.")
    parser.add_argument("--value-column", help="Signal column name. Defaults to strain or Time 1.")
    parser.add_argument("--reverse-channels", default="2,4", help="Comma-separated channel numbers to reverse.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input or prompt_path("Input CSV")

    if args.ranges is None:
        ranges = prompt_ranges("Extraction ranges", DEFAULT_MASK_RANGES)
    else:
        ranges = parse_range_list(args.ranges, DEFAULT_MASK_RANGES)

    pipe_length = args.pipe_length
    if pipe_length is None:
        pipe_length = prompt_float("Pipe length", 2.92)

    reverse_channels = tuple(
        int(item.strip()) for item in args.reverse_channels.split(",") if item.strip()
    )

    extract_strain(
        input_path=input_path,
        ranges=ranges,
        pipe_length=pipe_length,
        target_length=args.target_length,
        output_path=args.output,
        value_column=args.value_column,
        reverse_channels=reverse_channels,
    )


if __name__ == "__main__":
    main()
