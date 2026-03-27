from __future__ import annotations

import argparse
from pathlib import Path

import pydicom

from rt_rename.dicom_utils import dataset_from_upload_contents, update_dicom
from rt_rename.exports import structure_dict_to_csv
from rt_rename.parsers import file_to_upload_contents, parse_csv, parse_dicom
from rt_rename.rename_service import rename_structures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RT-Rename against a CSV or DICOM RTStruct file."
    )
    parser.add_argument("input_path", help="Input CSV or DICOM RTStruct file")
    parser.add_argument("output_csv", help="Path to write the renamed CSV output")
    parser.add_argument("--model", required=True, help="Model display name from config/models.json")
    parser.add_argument("--prompt", required=True, help="Prompt file name from the config directory")
    parser.add_argument(
        "--guideline",
        default="TG263",
        choices=["TG263", "TG263_reverse"],
        help="Nomenclature guideline to use",
    )
    parser.add_argument(
        "--region",
        action="append",
        dest="regions",
        help="Region to include. Repeat the flag to include multiple regions.",
    )
    parser.add_argument(
        "--tv-filter",
        action="store_true",
        help="Filter out target volume structures such as PTV/GTV/CTV/ITV.",
    )
    parser.add_argument(
        "--uncertain",
        action="store_true",
        help="Run repeated inference and store an entropy-based uncertainty score.",
    )
    parser.add_argument(
        "--output-dicom",
        help="Optional output path for an updated DICOM RTStruct when the input is DICOM.",
    )
    return parser.parse_args()


def load_rows(input_path: Path, tv_filter: bool):
    contents = file_to_upload_contents(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return parse_csv(contents, input_path.name), contents
    if suffix == ".dcm":
        return parse_dicom(contents, input_path.name, tv_filter), contents
    raise ValueError("input_path must be a .csv or .dcm file")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    rows, contents = load_rows(input_path, args.tv_filter)
    renamed_rows = rename_structures(
        model=args.model,
        prompt=args.prompt,
        guideline=args.guideline,
        regions=args.regions,
        structure_dict=rows,
        uncertain=args.uncertain,
        progress_callback=print,
    )
    structure_dict_to_csv(renamed_rows, args.output_csv)

    if input_path.suffix.lower() == ".dcm" and args.output_dicom:
        dataset = dataset_from_upload_contents(contents)
        updated_dataset = update_dicom(dataset, renamed_rows)
        pydicom.dcmwrite(args.output_dicom, updated_dataset)


if __name__ == "__main__":
    main()
