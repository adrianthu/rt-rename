from __future__ import annotations

from pathlib import Path
import openpyxl as xl

from .constants import GUIDELINE_PATH

SHEET_NAME = "TG263 v20170815"
GUIDELINE_COLUMN_BY_TYPE = {"standard": "F", "reverse": "G"}
GUIDELINE_KIND_TO_TYPE = {"TG263": "standard", "TG263_reverse": "reverse"}


def _unique_regions(values: list[str | None]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique


def load_guideline(
    nomenclature_xlsx: str | Path,
    type: str = "standard",
    description: bool = False,
    regions: list[str] | None = None,
) -> list[dict[str, str]]:
    if type not in GUIDELINE_COLUMN_BY_TYPE:
        raise ValueError("type must be one of: standard, reverse")

    workbook = xl.load_workbook(nomenclature_xlsx, data_only=True)
    worksheet = workbook[SHEET_NAME]
    region_values = [cell.value for cell in worksheet["D"][1:]]
    selected_regions = regions or _unique_regions(region_values)
    selected_column = GUIDELINE_COLUMN_BY_TYPE[type]

    structures: list[dict[str, str]] = []
    for cell in worksheet["D"]:
        if cell.value not in selected_regions:
            continue
        structure_name = worksheet[f"{selected_column}{cell.row}"].value
        if not structure_name:
            continue
        entry = {"name": str(structure_name)}
        if description:
            entry["description"] = str(worksheet[f"H{cell.row}"].value or "")
        structures.append(entry)

    workbook.close()
    return structures


def read_guideline(
    regions: list[str] | None,
    guideline: str,
    description: bool = True,
    workbook_path: Path = GUIDELINE_PATH,
) -> list[dict[str, str]]:
    try:
        guideline_type = GUIDELINE_KIND_TO_TYPE[guideline]
    except KeyError as exc:
        raise ValueError(f"Unsupported guideline: {guideline}") from exc

    return load_guideline(
        nomenclature_xlsx=workbook_path,
        type=guideline_type,
        description=description,
        regions=regions,
    )
