from __future__ import annotations

from pathlib import Path
import base64
import io

import pandas as pd

from .constants import EXCLUDED_NRRD_SUFFIXES, TARGET_VOLUME_MARKERS
from .dicom_utils import dataset_from_upload_contents, read_dicom_rtstruct_names


def sort_key(filename: str) -> str:
    return filename.lower()


def _stringify(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _should_filter_target_volumes(tv_filter: bool | str) -> bool:
    if isinstance(tv_filter, bool):
        return tv_filter
    return str(tv_filter).strip().lower() == "true"


def _is_target_volume(name: str) -> bool:
    upper_name = name.upper()
    return any(marker in upper_name for marker in TARGET_VOLUME_MARKERS)


def make_structure_row(
    local_name: str,
    tg263_name: str = "",
    confidence: str = "",
    verify: str = "",
    accept: bool = False,
    comment: str = "",
    raw_output: str = "",
    timestamp: str = "",
) -> dict[str, object]:
    return {
        "local name": local_name,
        "TG263 name": tg263_name,
        "confidence": confidence,
        "verify": verify,
        "accept": accept,
        "comment": comment,
        "raw output": raw_output,
        "timestamp": timestamp,
    }


def load_structures_dir(dir_path: str | Path, filter: str | None = None) -> list[str]:
    structures = [path.name for path in Path(dir_path).iterdir() if path.is_file()]
    if filter == "synthRAD2025":
        structures = [
            name
            for name in structures
            if name.endswith(".nrrd") and not name.endswith(EXCLUDED_NRRD_SUFFIXES)
        ]
    return [Path(structure).stem for structure in structures]


def file_to_upload_contents(file_path: str | Path) -> str:
    encoded = base64.b64encode(Path(file_path).read_bytes()).decode("utf-8")
    return f"data:application/octet-stream;base64,{encoded}"


def parse_filenames(
    filenames: list[str],
    tv_filter: bool | str = True,
) -> list[dict[str, object]]:
    structures: list[str] = []
    for filename in filenames:
        if not filename.endswith(".nrrd"):
            continue
        if filename.endswith(EXCLUDED_NRRD_SUFFIXES):
            continue
        local_name = Path(filename).stem
        if _should_filter_target_volumes(tv_filter) and _is_target_volume(local_name):
            continue
        structures.append(local_name)

    return [make_structure_row(name) for name in sorted(structures, key=sort_key)]


def _read_csv_frame(contents: str) -> pd.DataFrame:
    decoded = base64.b64decode(contents.split(",", 1)[1])
    return pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))


def parse_csv(contents: str, filename: str = "") -> list[dict[str, object]]:
    data_frame = _read_csv_frame(contents)
    if data_frame.empty:
        return []

    normalized_columns = {column.strip().lower(): column for column in data_frame.columns}
    local_name_column = normalized_columns.get("local name", data_frame.columns[0])

    rows: list[dict[str, object]] = []
    for _, record in data_frame.iterrows():
        local_name = _stringify(record[local_name_column]).replace(".nrrd", "")
        rows.append(
            make_structure_row(
                local_name=local_name,
                tg263_name=_stringify(record.get(normalized_columns.get("tg263 name", ""), "")),
                confidence=_stringify(record.get(normalized_columns.get("confidence", ""), "")),
                verify=_stringify(record.get(normalized_columns.get("verify", ""), "")),
                accept=_coerce_bool(record.get(normalized_columns.get("accept", ""), False)),
                comment=_stringify(record.get(normalized_columns.get("comment", ""), "")),
                raw_output=_stringify(
                    record.get(
                        normalized_columns.get("raw output")
                        or normalized_columns.get("raw_output", ""),
                        "",
                    )
                ),
                timestamp=_stringify(record.get(normalized_columns.get("timestamp", ""), "")),
            )
        )
    return rows


def parse_dicom(
    contents: str,
    filename: str,
    tv_filter: bool | str = False,
) -> list[dict[str, object]]:
    dataset = dataset_from_upload_contents(contents)
    roi_names = read_dicom_rtstruct_names(dataset)
    if _should_filter_target_volumes(tv_filter):
        roi_names = [name for name in roi_names if not _is_target_volume(name)]
    return [make_structure_row(name) for name in sorted(roi_names, key=sort_key)]
