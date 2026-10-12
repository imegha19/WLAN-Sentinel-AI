from pathlib import Path
import argparse
import json

import pandas as pd


# ============================================================
# WLAN SENTINEL AI — REAL DATASET INSPECTION
# ============================================================
# This module inspects CSV files in chunks.
# It does NOT train a model and does NOT claim to detect attacks.
# AWID3 schema, label names, and encoding must be verified from
# the actual dataset once access is approved.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNK_SIZE = 10_000
PREVIEW_ROWS = 10

# Common label-column names used by different network datasets.
# These are candidates to review, not guaranteed AWID3 column names.
LABEL_COLUMN_CANDIDATES = [
    "label",
    "labels",
    "class",
    "attack",
    "category",
    "target",
    "class_label",
    "attack_label",
    "traffic_label",
]


def find_label_candidates(columns):
    """
    Return columns whose names resemble common target/label names.

    This is a suggestion only. Always verify the correct target column
    from the actual dataset documentation.
    """
    candidates = []

    for column in columns:
        normalized = str(column).strip().lower().replace(" ", "_").replace("-", "_")

        if normalized in LABEL_COLUMN_CANDIDATES:
            candidates.append(str(column))
        elif any(
            word in normalized
            for word in ("label", "class", "attack", "target", "category")
        ):
            candidates.append(str(column))

    return candidates


def inspect_csv(file_path, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Inspect a CSV file chunk by chunk.

    Reports:
      - total rows and columns
      - column names and first-chunk data types
      - missing-value counts and percentages
      - a small preview
      - likely label-column names to review

    The full dataset is not collected into one DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Dataset path is not a file: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("This version supports CSV files only.")

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    total_rows = 0
    chunk_count = 0
    missing_counts = None
    column_names = None
    first_chunk_dtypes = None
    preview = None

    try:
        reader = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            low_memory=True,
        )

        for chunk in reader:
            chunk_count += 1
            total_rows += len(chunk)

            if column_names is None:
                column_names = list(chunk.columns)
                first_chunk_dtypes = {
                    str(column): str(dtype)
                    for column, dtype in chunk.dtypes.items()
                }
                missing_counts = pd.Series(
                    0,
                    index=chunk.columns,
                    dtype="int64",
                )

            missing_counts = missing_counts.add(
                chunk.isna().sum(),
                fill_value=0,
            )

            # Keep only the first PREVIEW_ROWS rows.
            if preview is None:
                preview = chunk.head(PREVIEW_ROWS).copy()
            elif len(preview) < PREVIEW_ROWS:
                remaining = PREVIEW_ROWS - len(preview)
                preview = pd.concat(
                    [preview, chunk.head(remaining)],
                    ignore_index=True,
                )

            del chunk

    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV file is empty.") from error

    if total_rows == 0 or column_names is None:
        raise ValueError(
            "The CSV contains a header but no data rows, or has no readable columns."
        )

    schema = pd.DataFrame(
        {
            "column": [str(column) for column in column_names],
            "first_chunk_dtype": [
                first_chunk_dtypes.get(str(column), "unknown")
                for column in column_names
            ],
            "missing_values": [
                int(missing_counts.get(column, 0))
                for column in column_names
            ],
        }
    )

    schema["missing_percent"] = (
        schema["missing_values"] / total_rows * 100
    ).round(2)

    label_candidates = find_label_candidates(column_names)

    return {
        "file_name": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "total_rows": total_rows,
        "total_columns": len(column_names),
        "chunks_read": chunk_count,
        "columns": [str(column) for column in column_names],
        "schema": schema,
        "preview": preview,
        "label_column_candidates": label_candidates,
        "inspection_notes": [
            "Inspection only; no model was trained.",
            "First-chunk data types may not capture mixed types later in the file.",
            "Label candidates are suggestions and must be verified against dataset documentation.",
            "Real-world WLAN intrusion detection has not been established by this inspection.",
        ],
    }


def print_report(report):
    """Print the inspection report in the terminal."""

    print("\n" + "=" * 60)
    print("WLAN SENTINEL AI — REAL DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"File: {report['file_name']}")
    print(f"File size: {report['file_size_bytes']:,} bytes")
    print(f"Total records: {report['total_rows']:,}")
    print(f"Total columns: {report['total_columns']:,}")
    print(f"Chunks read: {report['chunks_read']:,}")

    print("\n--- Column Names ---")
    for column in report["columns"]:
        print(f"- {column}")

    print("\n--- Schema and Missing Values ---")
    print(report["schema"].to_string(index=False))

    print("\n--- Likely Label Columns to Review ---")
    if report["label_column_candidates"]:
        for column in report["label_column_candidates"]:
            print(f"- {column}")
        print(
            "These are candidates only. Confirm the correct target column "
            "from the dataset documentation."
        )
    else:
        print(
            "No obvious label column was detected by name. "
            "Review the dataset documentation manually."
        )

    print("\n--- Preview (first rows) ---")
    print(report["preview"].to_string(index=False))

    print("\n--- Important Notes ---")
    for note in report["inspection_notes"]:
        print(f"- {note}")


def save_report_json(report, output_path):
    """
    Save a compact JSON report. The preview and schema are converted
    to JSON-compatible structures.
    """
    output_path = Path(output_path)

    serializable_report = {
        "file_name": report["file_name"],
        "file_size_bytes": report["file_size_bytes"],
        "total_rows": report["total_rows"],
        "total_columns": report["total_columns"],
        "chunks_read": report["chunks_read"],
        "columns": report["columns"],
        "schema": report["schema"].to_dict(orient="records"),
        "label_column_candidates": report["label_column_candidates"],
        "inspection_notes": report["inspection_notes"],
    }

    output_path.write_text(
        json.dumps(serializable_report, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a CSV dataset in chunks for WLAN Sentinel AI."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=str(PROJECT_DIR / "sample_traffic.csv"),
        help="Path to a CSV file (defaults to sample_traffic.csv in the project folder).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per processing chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save a JSON report beside the inspected CSV.",
    )

    args = parser.parse_args()

    try:
        report = inspect_csv(args.file, chunk_size=args.chunk_size)
        print_report(report)

        if args.save_json:
            input_path = Path(args.file).resolve()
            output_path = input_path.with_name(
                f"{input_path.stem}_inspection_report.json"
            )
            save_report_json(report, output_path)
            print(f"\nJSON report saved to: {output_path}")

    except (FileNotFoundError, ValueError, pd.errors.ParserError, OSError) as error:
        print(f"\nInspection failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
