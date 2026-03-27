from __future__ import annotations

import csv
from pathlib import Path


def create_output_csv(output_list: list[dict], output_csv: str | Path) -> None:
    structure_dict_to_csv(output_list, output_csv)


def structure_dict_to_csv(structure_dict: list[dict], output_csv: str | Path) -> None:
    if not structure_dict:
        raise ValueError("structure_dict must contain at least one row")
    keys = list(structure_dict[0].keys())
    with Path(output_csv).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, keys)
        writer.writeheader()
        writer.writerows(structure_dict)
