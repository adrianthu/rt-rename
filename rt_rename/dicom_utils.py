from __future__ import annotations

from pathlib import Path
import base64
import io

import pydicom
from pydicom.dataset import Dataset


RTSTRUCT_MODALITY = "RTSTRUCT"
CT_MODALITY = "CT"


def dataset_from_upload_contents(contents: str) -> Dataset:
    decoded = base64.b64decode(contents.split(",", 1)[1])
    return pydicom.dcmread(io.BytesIO(decoded))


def is_rtstruct_dataset(dataset: Dataset) -> bool:
    return str(getattr(dataset, "Modality", "")).upper() == RTSTRUCT_MODALITY


def is_ct_image_dataset(dataset: Dataset) -> bool:
    return (
        str(getattr(dataset, "Modality", "")).upper() == CT_MODALITY
        and hasattr(dataset, "PixelData")
    )


def read_dicom_rtstruct_names(dicom_source: str | Path | Dataset) -> list[str]:
    try:
        if isinstance(dicom_source, (str, Path)):
            rtstruct = pydicom.dcmread(str(dicom_source))
        else:
            rtstruct = dicom_source

        sequence = getattr(rtstruct, "StructureSetROISequence", None)
        if not sequence:
            return []
        return [roi.ROIName for roi in sequence if getattr(roi, "ROIName", None)]
    except Exception:
        return []


def write_dicom_rtstruct_names(
    dicom_file_path: str | Path,
    new_names_map: dict[str, str],
    output_file_path: str | Path | None = None,
) -> None:
    rtstruct = pydicom.dcmread(str(dicom_file_path))
    updated_dataset = update_dicom(rtstruct, [
        {
            "local name": key,
            "TG263 name": value,
            "accept": True,
        }
        for key, value in new_names_map.items()
    ])
    save_path = output_file_path or dicom_file_path
    updated_dataset.save_as(str(save_path))


def update_dicom(dicom_file: Dataset, structure_dict: list[dict]) -> Dataset:
    new_names_map = {
        row.get("local name", ""): row.get("TG263 name", "")
        for row in structure_dict
        if row.get("accept") and row.get("TG263 name")
    }
    if not new_names_map:
        return dicom_file

    for roi in getattr(dicom_file, "StructureSetROISequence", []):
        roi_name = getattr(roi, "ROIName", None)
        if roi_name in new_names_map:
            roi.ROIName = new_names_map[roi_name]
    return dicom_file
