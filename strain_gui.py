from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from extract_strain import extract_strain_dataframe
from noise_filter import filter_noise_dataframe
from pipeline_utils import (
    DEFAULT_MASK_RANGES,
    DISTANCE_COLUMN,
    guess_value_column,
    numeric_signal_columns,
    read_csv_flexible,
)


class StrainProcessorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Strain Processor")
        self.root.geometry("1280x820")

        self.input_path: Path | None = None
        self.raw_df: pd.DataFrame | None = None
        self.filtered_df: pd.DataFrame | None = None
        self.extracted_df: pd.DataFrame | None = None

        self.file_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str((Path.cwd() / "output").resolve()))
        self.value_column_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="300")
        self.pipe_length_var = tk.StringVar(value="2.92")
        self.reverse_channels_var = tk.StringVar(value="2,4")
        self.extract_source_var = tk.StringVar(value="filtered")
        self.x_min_var = tk.StringVar()
        self.x_max_var = tk.StringVar()
        self.y_min_var = tk.StringVar()
        self.y_max_var = tk.StringVar()
        self.filter_range_vars = self._make_range_vars(DEFAULT_MASK_RANGES)
        self.extract_range_vars = self._make_range_vars(DEFAULT_MASK_RANGES)

        self._build_layout()
        self._set_status("Open a CSV file to begin.")

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsw")
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_file_panel(left)
        self._build_visual_panel(left)
        self._build_filter_panel(left)
        self._build_extract_panel(left)
        self._build_log_panel(left)
        self._build_plot_panel(right)

    def _build_file_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="File", padding=10)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        ttk.Entry(frame, textvariable=self.file_var, width=44).grid(row=0, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self.browse_input).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(frame, text="Load", command=self.load_input).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(frame, text="Output folder").grid(row=1, column=0, sticky="w", pady=(10, 2))
        ttk.Entry(frame, textvariable=self.output_dir_var, width=44).grid(row=2, column=0, sticky="ew")
        ttk.Button(frame, text="Browse", command=self.browse_output_dir).grid(row=2, column=1, padx=(8, 0))

    def _build_visual_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Visualize", padding=10)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        ttk.Label(frame, text="Signal column").grid(row=0, column=0, sticky="w")
        self.value_column_box = ttk.Combobox(
            frame,
            textvariable=self.value_column_var,
            state="readonly",
            width=22,
        )
        self.value_column_box.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0))

        ttk.Label(frame, text="Distance min/max").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.x_min_var, width=9).grid(row=1, column=1, sticky="ew", padx=(8, 4), pady=(8, 0))
        ttk.Entry(frame, textvariable=self.x_max_var, width=9).grid(row=1, column=2, sticky="ew", padx=(4, 8), pady=(8, 0))

        ttk.Label(frame, text="Strain min/max").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.y_min_var, width=9).grid(row=2, column=1, sticky="ew", padx=(8, 4), pady=(8, 0))
        ttk.Entry(frame, textvariable=self.y_max_var, width=9).grid(row=2, column=2, sticky="ew", padx=(4, 8), pady=(8, 0))

        plot_buttons = ttk.Frame(frame)
        plot_buttons.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        plot_buttons.columnconfigure(0, weight=1)
        plot_buttons.columnconfigure(1, weight=1)
        ttk.Button(plot_buttons, text="Plot current", command=self.plot_current).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(plot_buttons, text="Plot raw", command=self.plot_raw).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _build_filter_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Noise Filter", padding=10)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Threshold").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.threshold_var, width=12).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(frame, text="Filter ranges").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        self._build_range_table(
            frame,
            row=1,
            range_vars=self.filter_range_vars,
            labels=("Range 1", "Range 2", "Range 3", "Range 4"),
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)
        ttk.Button(buttons, text="Apply filter", command=self.apply_filter).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Reset filter", command=self.reset_filter).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="Save filtered CSV", command=self.save_filtered).grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _build_extract_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Extract Strain", padding=10)
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Source").grid(row=0, column=0, sticky="w")
        source_box = ttk.Combobox(
            frame,
            textvariable=self.extract_source_var,
            values=("filtered", "raw"),
            state="readonly",
            width=12,
        )
        source_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(frame, text="Extract ranges").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        self._build_range_table(
            frame,
            row=1,
            range_vars=self.extract_range_vars,
            labels=("ch1", "ch2", "ch3", "ch4"),
        )

        ttk.Label(frame, text="Pipe length").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.pipe_length_var, width=12).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frame, text="Reverse channels").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.reverse_channels_var, width=12).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Extract", command=self.extract).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Save extracted CSV", command=self.save_extracted).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Status", padding=10)
        frame.grid(row=4, column=0, sticky="nsew")
        parent.rowconfigure(4, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.log = ScrolledText(frame, height=9, width=48, state="disabled", wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("No data loaded")
        self.ax.grid(True, alpha=0.25)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Open strain CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if path:
            self.file_var.set(path)
            self.load_input()

    def browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_dir_var.set(path)

    def load_input(self) -> None:
        try:
            path = Path(self.file_var.get()).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)

            df, encoding = read_csv_flexible(path)
            self.input_path = path
            self.raw_df = df
            self.filtered_df = None
            self.extracted_df = None

            signal_columns = numeric_signal_columns(df)
            self.value_column_box["values"] = signal_columns
            if signal_columns:
                try:
                    self.value_column_var.set(guess_value_column(df.columns))
                except ValueError:
                    self.value_column_var.set(signal_columns[0])

            self._set_status(f"Loaded {path.name} with {encoding}. Rows: {len(df)}. Columns: {', '.join(df.columns)}")
            self.plot_raw()
        except Exception as exc:
            self._show_error("Load failed", exc)

    def plot_raw(self) -> None:
        try:
            df = self._require_raw()
            column = self._selected_value_column()
            self._plot_dataframe(
                df=df,
                title=f"Raw: {self.input_path.name if self.input_path else ''}",
                y_columns=[column],
                x_column=DISTANCE_COLUMN if DISTANCE_COLUMN in df.columns else None,
                x_range=self._optional_pair(self.x_min_var, self.x_max_var),
                y_range=self._optional_pair(self.y_min_var, self.y_max_var),
            )
            self._set_status(f"Plotted raw column: {column}")
        except Exception as exc:
            self._show_error("Plot failed", exc)

    def plot_current(self) -> None:
        try:
            if self.filtered_df is not None:
                self._plot_dataframe(
                    df=self.filtered_df,
                    title=f"Current filtered: {self.input_path.name if self.input_path else ''}",
                    y_columns=["strain"],
                    x_column=DISTANCE_COLUMN,
                    x_range=self._optional_pair(self.x_min_var, self.x_max_var),
                    y_range=self._optional_pair(self.y_min_var, self.y_max_var),
                )
                self._set_status("Plotted current filtered data.")
                return

            self.plot_raw()
        except Exception as exc:
            self._show_error("Plot failed", exc)

    def apply_filter(self) -> None:
        try:
            df, source_label, value_column = self._filter_source()
            threshold = float(self.threshold_var.get())
            ranges = self._ranges_from_vars(self.filter_range_vars, require_all=False)

            self.filtered_df, stats = filter_noise_dataframe(
                df=df,
                threshold=threshold,
                mask_ranges=ranges,
                value_column=value_column,
            )
            self._plot_dataframe(
                df=self.filtered_df,
                title=f"Filtered from {source_label}: threshold={threshold:g}",
                y_columns=["strain"],
                x_column=DISTANCE_COLUMN,
                x_range=self._optional_pair(self.x_min_var, self.x_max_var),
                y_range=self._optional_pair(self.y_min_var, self.y_max_var),
            )
            self._set_status(
                f"Filter applied to {source_label} data. "
                f"Selected rows: {stats['selected_rows']}; outside rows unchanged: {stats['unchanged_rows']}; "
                f"spikes replaced: {stats['spikes_replaced']}."
            )
        except Exception as exc:
            self._show_error("Filter failed", exc)

    def reset_filter(self) -> None:
        try:
            self._require_raw()
            self.filtered_df = None
            self.extracted_df = None
            self.plot_raw()
            self._set_status("Filtered data reset. Next filter will start from raw data.")
        except Exception as exc:
            self._show_error("Reset failed", exc)

    def save_filtered(self) -> None:
        try:
            if self.filtered_df is None:
                raise ValueError("Run Apply filter first.")
            default_name = self._default_stem("processed") + ".csv"
            path = self._ask_save_path(default_name)
            if not path:
                return
            self.filtered_df.to_csv(path, index=False)
            self._set_status(f"Saved filtered CSV: {path}")
        except Exception as exc:
            self._show_error("Save failed", exc)

    def extract(self) -> None:
        try:
            source_df, source_label = self._extract_source()
            ranges = self._ranges_from_vars(self.extract_range_vars, require_all=True)
            pipe_length = float(self.pipe_length_var.get())
            reverse_channels = self._parse_reverse_channels()

            self.extracted_df, stats = extract_strain_dataframe(
                df=source_df,
                ranges=ranges,
                pipe_length=pipe_length,
                target_length=None,
                value_column="strain" if "strain" in source_df.columns else self._selected_value_column(),
                reverse_channels=reverse_channels,
            )
            self._plot_dataframe(
                df=self.extracted_df,
                title=f"Extracted from {source_label}",
                y_columns=list(self.extracted_df.columns),
                x_column=None,
                x_range=None,
                y_range=self._optional_pair(self.y_min_var, self.y_max_var),
            )
            self._set_status(
                "Extracted strain. "
                f"Channel lengths before adjustment: {stats['channel_lengths']}; "
                f"output points: {stats['output_length']}; trim counts: {stats['trim_counts']}."
            )
        except Exception as exc:
            self._show_error("Extract failed", exc)

    def save_extracted(self) -> None:
        try:
            if self.extracted_df is None:
                raise ValueError("Run Extract first.")
            default_name = self._default_stem("extracted") + ".csv"
            path = self._ask_save_path(default_name)
            if not path:
                return
            self.extracted_df.to_csv(path)
            self._set_status(f"Saved extracted CSV: {path}")
        except Exception as exc:
            self._show_error("Save failed", exc)

    def _plot_dataframe(
        self,
        df: pd.DataFrame,
        title: str,
        y_columns: list[str],
        x_column: str | None,
        x_range: tuple[float, float] | None,
        y_range: tuple[float, float] | None,
    ) -> None:
        plot_df = df.copy()

        if x_column and x_column in plot_df.columns:
            x = pd.to_numeric(plot_df[x_column], errors="coerce")
            if x_range is not None:
                plot_df = plot_df.loc[x.between(*x_range)]
                x = pd.to_numeric(plot_df[x_column], errors="coerce")
            x_label = x_column
        else:
            x = pd.Series(plot_df.index, index=plot_df.index)
            x_label = plot_df.index.name or "index"

        self.ax.clear()
        for column in y_columns:
            if column not in plot_df.columns:
                continue
            y = pd.to_numeric(plot_df[column], errors="coerce")
            self.ax.plot(x, y, label=column, linewidth=1.2)

        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel("Strain")
        self.ax.grid(True, alpha=0.25)
        if y_range is not None:
            self.ax.set_ylim(*y_range)
        if y_columns:
            self.ax.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _require_raw(self) -> pd.DataFrame:
        if self.raw_df is None:
            raise ValueError("Load a CSV file first.")
        return self.raw_df

    def _selected_value_column(self) -> str:
        df = self._require_raw()
        column = self.value_column_var.get().strip()
        if not column:
            column = guess_value_column(df.columns)
            self.value_column_var.set(column)
        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")
        return column

    def _extract_source(self) -> tuple[pd.DataFrame, str]:
        if self.extract_source_var.get() == "filtered":
            if self.filtered_df is None:
                raise ValueError("Filtered source selected. Run Apply filter first, or switch Source to raw.")
            return self.filtered_df, "filtered"
        return self._require_raw(), "raw"

    def _filter_source(self) -> tuple[pd.DataFrame, str, str]:
        if self.filtered_df is not None:
            return self.filtered_df, "current filtered", "strain"
        return self._require_raw(), "raw", self._selected_value_column()

    def _make_range_vars(
        self,
        ranges: list[tuple[float, float]],
    ) -> list[tuple[tk.StringVar, tk.StringVar]]:
        return [
            (tk.StringVar(value=f"{start:.2f}"), tk.StringVar(value=f"{end:.2f}"))
            for start, end in ranges
        ]

    def _build_range_table(
        self,
        parent: ttk.Frame,
        row: int,
        range_vars: list[tuple[tk.StringVar, tk.StringVar]],
        labels: tuple[str, ...],
    ) -> None:
        table = ttk.Frame(parent)
        table.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        table.columnconfigure(1, weight=1, uniform="range_entry")
        table.columnconfigure(2, weight=1, uniform="range_entry")

        ttk.Label(table, text="Start").grid(row=0, column=1, sticky="ew", padx=(6, 4))
        ttk.Label(table, text="End").grid(row=0, column=2, sticky="ew", padx=(4, 0))

        for idx, ((start_var, end_var), label) in enumerate(zip(range_vars, labels), start=1):
            ttk.Label(table, text=label).grid(row=idx, column=0, sticky="w", pady=(4, 0))
            ttk.Entry(table, textvariable=start_var, width=9).grid(
                row=idx,
                column=1,
                sticky="ew",
                padx=(6, 4),
                pady=(4, 0),
            )
            ttk.Entry(table, textvariable=end_var, width=9).grid(
                row=idx,
                column=2,
                sticky="ew",
                padx=(4, 0),
                pady=(4, 0),
            )

    def _ranges_from_vars(
        self,
        range_vars: list[tuple[tk.StringVar, tk.StringVar]],
        require_all: bool,
    ) -> list[tuple[float, float]]:
        ranges = []
        for idx, (start_var, end_var) in enumerate(range_vars, start=1):
            start_text = start_var.get().strip()
            end_text = end_var.get().strip()
            if not start_text and not end_text and not require_all:
                continue
            if not start_text or not end_text:
                raise ValueError(f"Range {idx} needs both start and end values.")

            start, end = float(start_text), float(end_text)
            if start > end:
                start, end = end, start
            ranges.append((start, end))

        if not ranges:
            raise ValueError("Enter at least one range.")
        return ranges

    def _optional_pair(
        self,
        min_var: tk.StringVar,
        max_var: tk.StringVar,
    ) -> tuple[float, float] | None:
        low_text = min_var.get().strip()
        high_text = max_var.get().strip()
        if not low_text and not high_text:
            return None
        if not low_text or not high_text:
            raise ValueError("Enter both min and max values, or leave both blank.")
        low, high = float(low_text), float(high_text)
        return (low, high) if low <= high else (high, low)

    def _parse_reverse_channels(self) -> tuple[int, ...]:
        text = self.reverse_channels_var.get().strip()
        if not text:
            return ()
        channels = []
        for item in text.split(","):
            item = item.strip()
            if item:
                channels.append(int(item))
        return tuple(channels)

    def _ask_save_path(self, default_name: str) -> Path | None:
        output_dir = Path(self.output_dir_var.get()).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            initialdir=output_dir,
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        return Path(path).resolve() if path else None

    def _default_stem(self, suffix: str) -> str:
        if self.input_path is None:
            return suffix
        return f"{self.input_path.stem}_{suffix}"

    def _set_status(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _show_error(self, title: str, exc: Exception) -> None:
        messagebox.showerror(title, str(exc))
        self._set_status(f"{title}: {exc}")


def main() -> None:
    root = tk.Tk()
    StrainProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
