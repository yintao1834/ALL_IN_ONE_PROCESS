from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline_utils import (
    DISTANCE_COLUMN,
    numeric_signal_columns,
    parse_columns,
    parse_float_pair,
    prompt_path,
    read_csv_flexible,
)


def _load_visualization_data(input_path: Path) -> tuple[pd.DataFrame, str]:
    df, encoding = read_csv_flexible(input_path)

    first_col = df.columns[0]
    if first_col.startswith("Unnamed"):
        df = df.rename(columns={first_col: "s"})

    return df, encoding


def _pick_x_column(df: pd.DataFrame) -> str | None:
    for candidate in (DISTANCE_COLUMN, "s"):
        if candidate in df.columns:
            return candidate
    return None


def visualize_file(
    input_path: Path,
    columns: list[str] | None = None,
    strain_range: tuple[float, float] | None = None,
    distance_range: tuple[float, float] | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    df, encoding = _load_visualization_data(input_path)

    x_column = _pick_x_column(df)
    if x_column is None:
        x = pd.Series(range(len(df)), name="index")
    else:
        df[x_column] = pd.to_numeric(df[x_column], errors="coerce")
        x = df[x_column]

    if columns is None:
        columns = numeric_signal_columns(df)

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Column not found: {', '.join(missing)}")

    if not columns:
        raise ValueError("No numeric strain columns found.")

    plot_df = df.copy()
    if distance_range is not None:
        low, high = distance_range
        plot_df = plot_df.loc[x.between(low, high)]
        x = plot_df[x_column] if x_column is not None else pd.Series(range(len(plot_df)))

    fig, ax = plt.subplots(figsize=(14, 7))
    for column in columns:
        y = pd.to_numeric(plot_df[column], errors="coerce")
        ax.plot(x, y, label=column)

    ax.set_title(input_path.name)
    ax.set_xlabel(x_column or "index")
    ax.set_ylabel("Strain")
    ax.legend()
    ax.grid(True, alpha=0.25)

    if strain_range is not None:
        ax.set_ylim(*strain_range)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved figure: {output_path}")

    print(f"Read {input_path} with encoding {encoding}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize one strain CSV file.")
    parser.add_argument("input", nargs="?", type=Path, help="Input CSV path.")
    parser.add_argument("--columns", help="Comma-separated columns to plot, for example ch1,ch2.")
    parser.add_argument("--strain-range", help="Y-axis range, for example -500 500.")
    parser.add_argument("--distance-range", help="X-axis range, for example 15 32.")
    parser.add_argument("--output", type=Path, help="Optional image output path.")
    parser.add_argument("--no-show", action="store_true", help="Save/prepare figure without opening the window.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input or prompt_path("Input CSV")

    strain_range = parse_float_pair(args.strain_range)
    if args.strain_range is None:
        typed = input("Strain display range, for example -500 500 [auto]: ").strip()
        strain_range = parse_float_pair(typed)

    distance_range = parse_float_pair(args.distance_range)
    columns = parse_columns(args.columns)

    visualize_file(
        input_path=input_path,
        columns=columns,
        strain_range=strain_range,
        distance_range=distance_range,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
