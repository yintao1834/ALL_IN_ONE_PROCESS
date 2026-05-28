from __future__ import annotations

from pathlib import Path

from data_accumulate import accumulate_files
from extract_strain import extract_strain
from noise_filter import filter_noise
from pipeline_utils import (
    DEFAULT_MASK_RANGES,
    parse_float_pair,
    prompt_float,
    prompt_path,
    prompt_ranges,
    prompt_yes_no,
)
from visualize_strain import visualize_file


def _parse_paths(text: str) -> list[Path]:
    return [Path(item.strip()).expanduser().resolve() for item in text.split(";") if item.strip()]


def main() -> None:
    print("Step 1/4: visualize")
    raw_path = prompt_path("Raw strain CSV")
    strain_range_text = input("Strain display range, for example -500 500 [auto]: ").strip()
    visualize_file(
        input_path=raw_path,
        strain_range=parse_float_pair(strain_range_text),
        show=True,
    )

    print("\nStep 2/4: noise filter")
    threshold = prompt_float("Threshold")
    mask_ranges = prompt_ranges("Distance mask ranges", DEFAULT_MASK_RANGES)
    processed_path = filter_noise(
        input_path=raw_path,
        threshold=threshold,
        mask_ranges=mask_ranges,
    )

    current_path = processed_path

    print("\nStep 3/4: accumulate (optional)")
    if prompt_yes_no("Accumulate files", default=False):
        print("Enter files in accumulation order, separated by semicolons.")
        print(f"Press Enter to use only current processed file: {processed_path}")
        text = input("Files: ").strip()
        input_paths = _parse_paths(text) if text else [processed_path]
        output_default = processed_path.parent / "accumulate_result.csv"
        output_path = prompt_path("Accumulated output CSV", output_default, must_exist=False)
        current_path = accumulate_files(input_paths, output_path)
    else:
        print("Skipped accumulation.")

    print("\nStep 4/4: extract strain")
    extract_ranges = prompt_ranges("Extraction ranges", DEFAULT_MASK_RANGES)
    pipe_length = prompt_float("Pipe length", 2.92)

    extract_strain(
        input_path=current_path,
        ranges=extract_ranges,
        pipe_length=pipe_length,
        target_length=None,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
