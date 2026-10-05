"""Parser for HomeCredit_columns_description.csv metadata file."""

import csv
import json
import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
METADATA_FILE = ROOT / "dataset" / "HomeCredit_columns_description.csv"
OUTPUT_FILE = ROOT / "reports" / "profiling" / "column_dictionary.json"


def parse_column_dictionary(
    csv_path: pathlib.Path = METADATA_FILE,
    output_path: pathlib.Path = OUTPUT_FILE,
) -> dict[str, Any]:
    """Parse HomeCredit_columns_description.csv into structured JSON dictionary."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    dictionary: dict[str, dict[str, dict[str, str | None]]] = {}
    rows_parsed = 0

    with open(csv_path, mode="r", encoding="latin1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table = row.get("Table", "").strip()
            col = row.get("Row", "").strip()
            desc = row.get("Description", "").strip()
            special = row.get("Special", "").strip() or None

            if not table or not col:
                continue

            if table not in dictionary:
                dictionary[table] = {}

            dictionary[table][col] = {
                "description": desc,
                "special": special,
            }
            rows_parsed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "tables_mapped": list(dictionary.keys()),
        "total_columns_mapped": rows_parsed,
        "dictionary": dictionary,
    }
