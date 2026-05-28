from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
CASE_DIR = BASE_DIR / "accumulate"
OUTPUT_DIR = BASE_DIR / "cumulative_output"
OUTPUT_DIR.mkdir(exist_ok=True)

data_paths = [
    CASE_DIR / f"{50 * num}mm-{50 * (num + 1)}mm_strain_extracted.csv"
    for num in range(0, 3)
]

cumulative = None

for i, path in enumerate(data_paths, start=1):

    df = pd.read_csv(path, encoding="cp932", index_col=0)

    current_strain = df[["ch1", "ch2", "ch3", "ch4"]].copy()

    if cumulative is None:
        cumulative = current_strain.copy()
    else:
        cumulative = cumulative + current_strain

    cumulative.to_csv(
        OUTPUT_DIR / f"cumulative_strain_{i}.csv",
        encoding="utf-8-sig"
    )