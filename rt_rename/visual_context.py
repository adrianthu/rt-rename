from __future__ import annotations

from dataclasses import dataclass
import base64
from io import BytesIO

from matplotlib import image as mpimg
from matplotlib.path import Path as MplPath
import numpy as np
from pydicom.dataset import Dataset

from .dicom_utils import dataset_from_upload_contents, is_ct_image_dataset


@dataclass(frozen=True)
class StructureSliceImages:
    axial: str
    sagittal: str
    coronal: str


@dataclass(frozen=True)
class CtSeries:
    volume: np.ndarray
    row_direction: np.ndarray
    column_direction: np.ndarray
    normal_direction: np.ndarray
    origin: np.ndarray
    row_spacing: float
    column_spacing: float
    slice_positions: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.volume.shape

    @property
    def slice_spacing(self) -> float:
        if len(self.slice_positions) > 1:
            return float(np.median(np.diff(self.slice_positions)))
        return 1.0


def _rescaled_pixels(dataset: Dataset) -> np.ndarray:
    slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    return dataset.pixel_array.astype(np.float32) * slope + intercept


def _orientation_vector(dataset: Dataset, start: int, end: int) -> np.ndarray:
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    if orientation is None or len(orientation) != 6:
        raise ValueError("CT slice is missing ImageOrientationPatient.")
    vector = np.asarray(orientation[start:end], dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Encountered zero-length CT orientation vector.")
    return vector / norm


def _slice_position(dataset: Dataset, normal: np.ndarray) -> float:
    position = getattr(dataset, "ImagePositionPatient", None)
    if position is None or len(position) != 3:
        raise ValueError("CT slice is missing ImagePositionPatient.")
    return float(np.dot(np.asarray(position, dtype=np.float64), normal))


def _ct_series_from_datasets(datasets: list[Dataset]) -> CtSeries:
    if not datasets:
        raise ValueError("No CT slices were uploaded.")

    row_direction = _orientation_vector(datasets[0], 0, 3)
    column_direction = _orientation_vector(datasets[0], 3, 6)
    normal_direction = np.cross(row_direction, column_direction)
    normal_norm = np.linalg.norm(normal_direction)
    if normal_norm == 0:
        raise ValueError("Could not derive CT slice normal from orientation.")
    normal_direction = normal_direction / normal_norm

    sorted_datasets = sorted(
        datasets,
        key=lambda dataset: _slice_position(dataset, normal_direction),
    )
    origin = np.asarray(sorted_datasets[0].ImagePositionPatient, dtype=np.float64)
    pixel_spacing = getattr(sorted_datasets[0], "PixelSpacing", None)
    if pixel_spacing is None or len(pixel_spacing) != 2:
        raise ValueError("CT slice is missing PixelSpacing.")

    volume = np.stack([_rescaled_pixels(dataset) for dataset in sorted_datasets], axis=0)
    slice_positions = np.asarray(
        [_slice_position(dataset, normal_direction) for dataset in sorted_datasets],
        dtype=np.float64,
    )
    return CtSeries(
        volume=volume,
        row_direction=row_direction,
        column_direction=column_direction,
        normal_direction=normal_direction,
        origin=origin,
        row_spacing=float(pixel_spacing[0]),
        column_spacing=float(pixel_spacing[1]),
        slice_positions=slice_positions,
    )


def _collect_referenced_sop_instance_uids(rtstruct: Dataset) -> set[str]:
    referenced_uids: set[str] = set()
    for roi_contour in getattr(rtstruct, "ROIContourSequence", []) or []:
        for contour in getattr(roi_contour, "ContourSequence", []) or []:
            for contour_image in getattr(contour, "ContourImageSequence", []) or []:
                uid = getattr(contour_image, "ReferencedSOPInstanceUID", "")
                if uid:
                    referenced_uids.add(str(uid))
    return referenced_uids


def _load_ct_series_from_uploads(rtstruct: Dataset, ct_upload_contents: list[str]) -> CtSeries:
    ct_datasets = [
        dataset_from_upload_contents(contents)
        for contents in ct_upload_contents
        if contents
    ]
    ct_datasets = [dataset for dataset in ct_datasets if is_ct_image_dataset(dataset)]
    if not ct_datasets:
        raise ValueError("No CT images were found in the uploaded CT files.")

    referenced_uids = _collect_referenced_sop_instance_uids(rtstruct)
    if referenced_uids:
        referenced_datasets = [
            dataset
            for dataset in ct_datasets
            if str(getattr(dataset, "SOPInstanceUID", "")) in referenced_uids
        ]
        if referenced_datasets:
            ct_datasets = referenced_datasets

    return _ct_series_from_datasets(ct_datasets)


def _polygon_mask(
    image_shape: tuple[int, int],
    row_coordinates: np.ndarray,
    column_coordinates: np.ndarray,
) -> np.ndarray:
    max_row, max_column = image_shape
    min_row = max(0, int(np.floor(np.min(row_coordinates))))
    max_row_bound = min(max_row, int(np.ceil(np.max(row_coordinates))) + 1)
    min_column = max(0, int(np.floor(np.min(column_coordinates))))
    max_column_bound = min(max_column, int(np.ceil(np.max(column_coordinates))) + 1)
    if min_row >= max_row_bound or min_column >= max_column_bound:
        return np.zeros(image_shape, dtype=bool)

    local_rows = np.arange(min_row, max_row_bound, dtype=np.float64) + 0.5
    local_columns = np.arange(min_column, max_column_bound, dtype=np.float64) + 0.5
    grid_columns, grid_rows = np.meshgrid(local_columns, local_rows)
    path = MplPath(np.column_stack([column_coordinates, row_coordinates]))
    local_mask = path.contains_points(
        np.column_stack([grid_columns.ravel(), grid_rows.ravel()])
    ).reshape(grid_rows.shape)

    mask = np.zeros(image_shape, dtype=bool)
    mask[min_row:max_row_bound, min_column:max_column_bound] = local_mask
    return mask


def _normalize_ct_slice(image: np.ndarray) -> np.ndarray:
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    lower = float(np.percentile(finite_values, 1))
    upper = float(np.percentile(finite_values, 99))
    if upper <= lower:
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def _repeat_superior_inferior_axis(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if scale <= 1.1:
        return image, mask
    repeat_factor = max(1, int(round(scale)))
    return (
        np.repeat(image, repeat_factor, axis=0),
        np.repeat(mask, repeat_factor, axis=0),
    )


def _to_data_url(image: np.ndarray, mask: np.ndarray) -> str:
    normalized = _normalize_ct_slice(image)
    rgb = np.stack([normalized, normalized, normalized], axis=-1)
    rgb[mask] = rgb[mask] * 0.45 + np.array([0.55, 0.0, 0.0], dtype=np.float32)
    buffer = BytesIO()
    mpimg.imsave(buffer, rgb, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


class StructureImageContext:
    def __init__(self, rtstruct: Dataset, ct_series: CtSeries):
        self._rtstruct = rtstruct
        self._ct_series = ct_series
        self._mask_cache: dict[str, np.ndarray | None] = {}
        self._image_cache: dict[str, StructureSliceImages | None] = {}
        self._roi_name_to_number = {
            str(getattr(roi, "ROIName", "")): int(getattr(roi, "ROINumber"))
            for roi in getattr(rtstruct, "StructureSetROISequence", []) or []
            if getattr(roi, "ROIName", None) and getattr(roi, "ROINumber", None) is not None
        }

    @classmethod
    def from_uploads(
        cls,
        rtstruct_contents: str,
        ct_upload_contents: list[str],
    ) -> "StructureImageContext":
        rtstruct = dataset_from_upload_contents(rtstruct_contents)
        ct_series = _load_ct_series_from_uploads(rtstruct, ct_upload_contents)
        return cls(rtstruct=rtstruct, ct_series=ct_series)

    def get_slice_images(self, structure_name: str) -> StructureSliceImages | None:
        if structure_name not in self._image_cache:
            mask = self._get_structure_mask(structure_name)
            if mask is None or not np.any(mask):
                self._image_cache[structure_name] = None
            else:
                self._image_cache[structure_name] = self._render_structure_images(mask)
        return self._image_cache[structure_name]

    def _get_structure_mask(self, structure_name: str) -> np.ndarray | None:
        if structure_name in self._mask_cache:
            return self._mask_cache[structure_name]

        roi_number = self._roi_name_to_number.get(structure_name)
        if roi_number is None:
            self._mask_cache[structure_name] = None
            return None

        mask = np.zeros(self._ct_series.shape, dtype=bool)
        found_contour = False
        for roi_contour in getattr(self._rtstruct, "ROIContourSequence", []) or []:
            if int(getattr(roi_contour, "ReferencedROINumber", -1)) != roi_number:
                continue
            for contour in getattr(roi_contour, "ContourSequence", []) or []:
                contour_data = np.asarray(
                    getattr(contour, "ContourData", []),
                    dtype=np.float64,
                )
                if contour_data.size < 9:
                    continue
                points = contour_data.reshape(-1, 3)
                slice_projection = np.dot(points, self._ct_series.normal_direction)
                slice_index = int(
                    np.argmin(
                        np.abs(
                            self._ct_series.slice_positions
                            - float(np.mean(slice_projection))
                        )
                    )
                )
                delta = points - self._ct_series.origin
                row_coordinates = (
                    np.dot(delta, self._ct_series.row_direction) / self._ct_series.row_spacing
                )
                column_coordinates = (
                    np.dot(delta, self._ct_series.column_direction)
                    / self._ct_series.column_spacing
                )
                mask[slice_index] |= _polygon_mask(
                    mask.shape[1:],
                    row_coordinates=row_coordinates,
                    column_coordinates=column_coordinates,
                )
                found_contour = True

        self._mask_cache[structure_name] = mask if found_contour else None
        return self._mask_cache[structure_name]

    def _render_structure_images(self, mask: np.ndarray) -> StructureSliceImages:
        occupied = np.argwhere(mask)
        z_min, y_min, x_min = occupied.min(axis=0)
        z_max, y_max, x_max = occupied.max(axis=0)
        z_center = int(round((z_min + z_max) / 2))
        y_center = int(round((y_min + y_max) / 2))
        x_center = int(round((x_min + x_max) / 2))

        z_margin = max(2, int((z_max - z_min + 1) * 0.5))
        y_margin = max(8, int((y_max - y_min + 1) * 0.35))
        x_margin = max(8, int((x_max - x_min + 1) * 0.35))

        axial_rows = slice(
            max(0, y_min - y_margin),
            min(mask.shape[1], y_max + y_margin + 1),
        )
        axial_columns = slice(
            max(0, x_min - x_margin),
            min(mask.shape[2], x_max + x_margin + 1),
        )
        axial_image = self._ct_series.volume[z_center, axial_rows, axial_columns]
        axial_mask = mask[z_center, axial_rows, axial_columns]

        superior_inferior = slice(
            max(0, z_min - z_margin),
            min(mask.shape[0], z_max + z_margin + 1),
        )

        coronal_image = self._ct_series.volume[superior_inferior, y_center, axial_columns]
        coronal_mask = mask[superior_inferior, y_center, axial_columns]
        sagittal_image = self._ct_series.volume[superior_inferior, axial_rows, x_center]
        sagittal_mask = mask[superior_inferior, axial_rows, x_center]

        scale = self._ct_series.slice_spacing / min(
            self._ct_series.row_spacing,
            self._ct_series.column_spacing,
        )
        coronal_image, coronal_mask = _repeat_superior_inferior_axis(
            coronal_image,
            coronal_mask,
            scale,
        )
        sagittal_image, sagittal_mask = _repeat_superior_inferior_axis(
            sagittal_image,
            sagittal_mask,
            scale,
        )

        return StructureSliceImages(
            axial=_to_data_url(axial_image, axial_mask),
            sagittal=_to_data_url(sagittal_image, sagittal_mask),
            coronal=_to_data_url(coronal_image, coronal_mask),
        )
